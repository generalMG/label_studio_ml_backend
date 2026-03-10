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
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
try:
    from .security_utils import (
        build_allowed_hosts,
        parse_bool,
        same_host,
        validate_remote_http_url,
    )
except ImportError:
    from security_utils import (
        build_allowed_hosts,
        parse_bool,
        same_host,
        validate_remote_http_url,
    )

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

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MAX_PIXELS = 2048 * 2048
MIN_PIXELS = 256 * 256
RENDER_DPI = 300
MAX_DIM = 2048
CROP_PAD = 4
CONF_THRESHOLD = 0.9
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
MAX_PDF_PAGES = 200
MAX_RENDER_PIXELS = 16 * 1024 * 1024
DOWNLOAD_CONNECT_TIMEOUT_SEC = 10.0
DOWNLOAD_READ_TIMEOUT_SEC = 120.0


def _safe_render_matrix(page: fitz.Page, max_render_pixels: int) -> fitz.Matrix:
    base_zoom = RENDER_DPI / 72
    target_w = max(page.rect.width * base_zoom, 1.0)
    target_h = max(page.rect.height * base_zoom, 1.0)
    target_pixels = target_w * target_h
    if target_pixels > max_render_pixels:
        scale = (max_render_pixels / target_pixels) ** 0.5
        zoom = max(base_zoom * scale, 0.1)
    else:
        zoom = base_zoom
    return fitz.Matrix(zoom, zoom)


def pdf_to_images(
    pdf_path: str,
    max_pages: int = MAX_PDF_PAGES,
    max_render_pixels: int = MAX_RENDER_PIXELS,
) -> list[Image.Image]:
    doc = fitz.open(pdf_path)
    images = []
    try:
        if doc.page_count > max_pages:
            raise ValueError(
                f"PDF has {doc.page_count} pages; limit is {max_pages}. "
                "Increase MAX_PDF_PAGES to allow this file."
            )
        for page in doc:
            mat = _safe_render_matrix(page, max_render_pixels)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            if img.width > MAX_DIM or img.height > MAX_DIM:
                img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
            images.append(img)
    finally:
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
        self.allowed_download_hosts = build_allowed_hosts(
            self.ls_url,
            os.environ.get("ALLOWED_DOWNLOAD_HOSTS", ""),
        )
        self.allow_local_task_files = parse_bool(
            os.environ.get("ALLOW_LOCAL_TASK_FILES"),
            default=False,
        )
        self.max_download_bytes = max(
            1,
            int(os.environ.get("MAX_DOWNLOAD_BYTES", MAX_DOWNLOAD_BYTES)),
        )
        self.max_pdf_pages = max(1, int(os.environ.get("MAX_PDF_PAGES", MAX_PDF_PAGES)))
        self.max_render_pixels = max(
            1,
            int(os.environ.get("MAX_RENDER_PIXELS", MAX_RENDER_PIXELS)),
        )
        self.download_connect_timeout = max(
            0.1,
            float(os.environ.get("DOWNLOAD_CONNECT_TIMEOUT_SEC", DOWNLOAD_CONNECT_TIMEOUT_SEC)),
        )
        self.download_read_timeout = max(
            0.1,
            float(os.environ.get("DOWNLOAD_READ_TIMEOUT_SEC", DOWNLOAD_READ_TIMEOUT_SEC)),
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
        self.qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
        )
        logger.info("Qwen model loaded.")

    def _download_remote_file(self, url: str) -> str:
        parsed = validate_remote_http_url(url, self.allowed_download_hosts)

        headers = {}
        if self.ls_api_key and same_host(url, self.ls_url):
            headers["Authorization"] = f"Token {self.ls_api_key}"

        suffix = Path(parsed.path).suffix or ".bin"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_path = tmp.name
        tmp.close()

        total_bytes = 0
        timeout = (self.download_connect_timeout, self.download_read_timeout)

        try:
            with requests.get(
                url,
                headers=headers,
                timeout=timeout,
                stream=True,
                allow_redirects=False,
            ) as resp:
                if 300 <= resp.status_code < 400:
                    raise ValueError("Redirect responses are not allowed for downloads")
                resp.raise_for_status()

                with open(tmp_path, "wb") as out:
                    for chunk in resp.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        total_bytes += len(chunk)
                        if total_bytes > self.max_download_bytes:
                            raise ValueError(
                                f"Download exceeds MAX_DOWNLOAD_BYTES={self.max_download_bytes}"
                            )
                        out.write(chunk)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

        logger.info("Downloaded %s -> %s (%d bytes)", url, tmp_path, total_bytes)
        return tmp_path

    def _download_task_file(self, url: str) -> tuple[str, bool]:
        """Resolve a task input to local path.

        Returns: (path, should_cleanup).
        """
        if url.startswith("/data/"):
            url = f"{self.ls_url}{url}"

        parsed = urlparse(url)

        if parsed.scheme in {"file", ""}:
            if not self.allow_local_task_files:
                raise ValueError(
                    "Local task paths are disabled. "
                    "Set ALLOW_LOCAL_TASK_FILES=true only in trusted environments."
                )
            local_path = parsed.path if parsed.scheme == "file" else url
            if not Path(local_path).exists():
                raise FileNotFoundError(local_path)
            return local_path, False

        return self._download_remote_file(url), True

    def _validate_pdf_limits(self, pdf_path: str) -> None:
        doc = fitz.open(pdf_path)
        try:
            if doc.page_count > self.max_pdf_pages:
                raise ValueError(
                    f"PDF has {doc.page_count} pages; limit is {self.max_pdf_pages}. "
                    "Increase MAX_PDF_PAGES to allow this file."
                )
        finally:
            doc.close()

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
            "--max-pages",
            str(self.max_pdf_pages),
            "--max-render-pixels",
            str(self.max_render_pixels),
        ]
        logger.info("Running PaddleOCR detection: %s", " ".join(cmd))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("PaddleOCR detection failed: %s", result.stderr)
                raise RuntimeError(f"PaddleOCR detection failed: {result.stderr}")

            with open(output_json) as f:
                data = json.load(f)
            return data
        finally:
            Path(output_json).unlink(missing_ok=True)

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
            cleanup_local_path = False
            try:
                local_path, cleanup_local_path = self._download_task_file(image_url)
            except Exception as e:
                logger.error("Failed to download %s: %s", image_url, e)
                predictions.append({"result": [], "score": 0})
                continue

            is_pdf = local_path.lower().endswith(".pdf")

            try:
                if is_pdf:
                    self._validate_pdf_limits(local_path)
                    detect_data = self._run_paddle_detection(local_path)
                    images = pdf_to_images(
                        local_path,
                        max_pages=self.max_pdf_pages,
                        max_render_pixels=self.max_render_pixels,
                    )

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
                if cleanup_local_path and local_path:
                    Path(local_path).unlink(missing_ok=True)

        return predictions
