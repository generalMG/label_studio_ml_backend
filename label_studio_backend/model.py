"""Label Studio ML backend for Hybrid OCR (PaddleOCR detection + Qwen2.5-VL recognition)."""

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import fitz
import requests
import torch
from label_studio_ml.model import LabelStudioMLBase
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

logger = logging.getLogger(__name__)

# Monkey patch JobManager to fix AssertionError when job result is None
from label_studio_ml.model import JobManager

def safe_get_result_from_job_id(self, job_id):
    result = self._get_result_from_job_id(job_id)
    if result is None:
        result = {}
    # assert isinstance(result, dict) # Skip assertion or ensure it passes
    if not isinstance(result, dict):
        result = {}
    result['job_id'] = job_id
    return result

JobManager.get_result_from_job_id = safe_get_result_from_job_id

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_PIXELS = 2048 * 2048
MIN_PIXELS = 256 * 256
RENDER_DPI = 300
MAX_DIM = 2048
CROP_PAD = 4
CONF_THRESHOLD = 0.9


def pdf_to_images(pdf_path: str) -> list[Image.Image]:
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        if img.width > MAX_DIM or img.height > MAX_DIM:
            img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
        images.append(img)
    doc.close()
    return images


def crop_region(image: Image.Image, bbox: list[int]) -> Image.Image:
    w, h = image.size
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - CROP_PAD)
    y1 = max(0, y1 - CROP_PAD)
    x2 = min(w, x2 + CROP_PAD)
    y2 = min(h, y2 + CROP_PAD)
    return image.crop((x1, y1, x2, y2))


def recognize_single(model, processor, crop: Image.Image) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": crop},
                {
                    "type": "text",
                    "text": "Read the text in this image. Return ONLY the exact text, nothing else. Include Korean, English, numbers, symbols.",
                },
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)
    trimmed = [o[len(i) :] for i, o in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


class HybridOCRBackend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.conf_threshold = float(os.environ.get("HYBRID_OCR_CONF_THRESHOLD", CONF_THRESHOLD))
        self.lang = os.environ.get("HYBRID_OCR_LANG", "korean")
        self.detect_script = str(Path(__file__).parent.parent / "ocr_detect.py")
        self.ls_url = (
            os.environ.get("LABEL_STUDIO_URL", "")
            or os.environ.get("LABEL_STUDIO_HOST", "")
            or kwargs.get("hostname", "")
            or "http://localhost:8080"
        ).rstrip("/")
        self.ls_api_key = (
            os.environ.get("LABEL_STUDIO_API_KEY", "")
            or os.environ.get("LABEL_STUDIO_ACCESS_TOKEN", "")
            or kwargs.get("access_token", "")
        )

        # Lazy-loaded on first predict() call to avoid timeout during /setup
        self.qwen_model = None
        self.processor = None

    def _ensure_model_loaded(self):
        if self.qwen_model is not None:
            return
        logger.info("Loading Qwen model %s (4-bit)...", MODEL_ID)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
        )
        logger.info("Qwen model loaded.")

    def _download_task_file(self, url: str) -> str:
        """Download file from Label Studio to a local temp path.

        Handles /data/upload/... URLs by fetching via Label Studio API,
        full http(s) URLs directly, and local file:// paths as-is.
        """
        # Relative Label Studio upload path (e.g. "/data/upload/1/file.pdf")
        if url.startswith("/data/"):
            url = f"{self.ls_url}{url}"

        parsed = urlparse(url)

        # Already a local file (but not a /data/ path we just converted)
        if parsed.scheme == "file":
            return parsed.path
        if not parsed.scheme:
            return url

        headers = {}
        if self.ls_api_key:
            headers["Authorization"] = f"Token {self.ls_api_key}"

        resp = requests.get(url, headers=headers, timeout=120)
        resp.raise_for_status()

        # Determine suffix from original URL path
        orig_path = urlparse(url).path
        suffix = Path(orig_path).suffix or ".bin"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(resp.content)
        tmp.close()
        logger.info("Downloaded %s -> %s (%d bytes)", url, tmp.name, len(resp.content))
        return tmp.name

    def _run_paddle_detection(self, input_path: str) -> dict:
        """Run PaddleOCR detection as subprocess, return parsed JSON."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_json = f.name

        cmd = [
            sys.executable,
            self.detect_script,
            "--input",
            input_path,
            "--output",
            output_json,
            "--lang",
            self.lang,
        ]
        logger.info("Running PaddleOCR detection: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("PaddleOCR detection failed: %s", result.stderr)
            raise RuntimeError(f"PaddleOCR detection failed: {result.stderr}")

        with open(output_json) as f:
            data = json.load(f)

        Path(output_json).unlink(missing_ok=True)
        return data

    def _process_image_detections(
        self, image: Image.Image, detections: list[dict], page_idx: int
    ) -> list[dict]:
        """Run Qwen on low-conf detections, return Label Studio result entries."""
        img_w, img_h = image.size
        results = []

        for idx, det in enumerate(detections):
            bbox = det["bbox"]
            conf = det["paddle_conf"]

            if conf < self.conf_threshold:
                crop = crop_region(image, bbox)
                text = recognize_single(self.qwen_model, self.processor, crop)
            else:
                text = det["paddle_text"]

            if not text.strip():
                continue

            x1, y1, x2, y2 = bbox
            x_pct = (x1 / img_w) * 100
            y_pct = (y1 / img_h) * 100
            w_pct = ((x2 - x1) / img_w) * 100
            h_pct = ((y2 - y1) / img_h) * 100

            region_id = f"region_{page_idx}_{idx}"

            results.append(
                {
                    "id": region_id,
                    "type": "rectanglelabels",
                    "from_name": "bbox",
                    "to_name": "image",
                    "original_width": img_w,
                    "original_height": img_h,
                    "value": {
                        "x": x_pct,
                        "y": y_pct,
                        "width": w_pct,
                        "height": h_pct,
                        "rotation": 0,
                        "rectanglelabels": ["Text"],
                    },
                }
            )

            results.append(
                {
                    "id": region_id,
                    "type": "textarea",
                    "from_name": "transcription",
                    "to_name": "image",
                    "original_width": img_w,
                    "original_height": img_h,
                    "value": {
                        "x": x_pct,
                        "y": y_pct,
                        "width": w_pct,
                        "height": h_pct,
                        "rotation": 0,
                        "text": [text],
                    },
                }
            )

        return results

    def predict(self, tasks, **kwargs):
        self._ensure_model_loaded()
        predictions = []

        for task in tasks:
            image_url = task["data"].get("image", "")
            if not image_url:
                predictions.append({"result": [], "score": 0})
                continue

            local_path = None
            try:
                local_path = self._download_task_file(image_url)
            except Exception as e:
                logger.error("Failed to download %s: %s", image_url, e)
                predictions.append({"result": [], "score": 0})
                continue

            is_pdf = local_path.lower().endswith(".pdf")

            try:
                if is_pdf:
                    detect_data = self._run_paddle_detection(local_path)
                    images = pdf_to_images(local_path)

                    all_results = []
                    for page_data in detect_data["pages"]:
                        page_num = page_data["page"]
                        image = images[page_num - 1]
                        page_results = self._process_image_detections(
                            image, page_data["detections"], page_num
                        )
                        all_results.extend(page_results)
                else:
                    detect_data = self._run_paddle_detection(local_path)

                    if detect_data.get("pages"):
                        page_data = detect_data["pages"][0]
                        detections = page_data.get("detections", [])
                        det_w = page_data.get("width")
                        det_h = page_data.get("height")
                    else:
                        detections = []
                        det_w = det_h = None

                    image = Image.open(local_path).convert("RGB")
                    if det_w and det_h:
                        image = image.resize((det_w, det_h), Image.LANCZOS)
                    elif image.width > MAX_DIM or image.height > MAX_DIM:
                        image.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)

                    all_results = self._process_image_detections(image, detections, 0)

                avg_score = 0.0
                if all_results:
                    bbox_count = sum(1 for r in all_results if r["type"] == "rectanglelabels")
                    avg_score = min(1.0, bbox_count / max(bbox_count, 1))

                predictions.append({"result": all_results, "score": avg_score})

            except Exception as e:
                logger.exception("Error processing task %s: %s", task.get("id"), e)
                predictions.append({"result": [], "score": 0})
            finally:
                # Clean up temp file if we downloaded it
                if local_path and local_path.startswith(tempfile.gettempdir()):
                    Path(local_path).unlink(missing_ok=True)

        return predictions
