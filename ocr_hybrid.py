import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import fitz
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_PIXELS = 2048 * 2048
MIN_PIXELS = 256 * 256
RENDER_DPI = 300
MAX_DIM = 2048
CROP_PAD = 4
CONF_THRESHOLD = 0.9

KR_FONT = "/home/mg_server/.local/share/fonts/NotoSansCJKkr-Regular.otf"


def pdf_to_images(pdf_path: str) -> list[Image.Image]:
    doc = fitz.open(pdf_path)
    images = []
    for i, page in enumerate(doc):
        mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        if img.width > MAX_DIM or img.height > MAX_DIM:
            img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
        print(f"  Page {i + 1}: {img.width}x{img.height}")
        images.append(img)
    doc.close()
    return images


def crop_region(image: Image.Image, det: dict) -> Image.Image:
    w, h = image.size
    x1, y1, x2, y2 = det["bbox"]
    x1 = max(0, x1 - CROP_PAD)
    y1 = max(0, y1 - CROP_PAD)
    x2 = min(w, x2 + CROP_PAD)
    y2 = min(h, y2 + CROP_PAD)
    return image.crop((x1, y1, x2, y2))


def load_qwen():
    print(f"Loading {MODEL_ID} (4-bit)...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
    )
    print("Model loaded.")
    return model, processor


def recognize_single(model, processor, crop: Image.Image) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": crop},
                {"type": "text", "text": "Read the text in this image. Return ONLY the exact text, nothing else. Include Korean, English, numbers, symbols."},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, generated_ids)]
    output = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    return output


def draw_detections(image: Image.Image, detections: list[dict], page_num: int, stats: dict) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(KR_FONT, 10)
        font_title = ImageFont.truetype(KR_FONT, 16)
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_title = font

    for det in detections:
        poly = det["polygon"]
        text = det.get("qwen_text") or det.get("paddle_text", "")
        x1, y1 = det["bbox"][0], det["bbox"][1]

        # Green = Qwen-recognized, yellow = PaddleOCR kept (high confidence)
        color = (0, 255, 0) if det.get("qwen_text") else (255, 255, 0)

        if poly:
            pts = [(p[0], p[1]) for p in poly] + [(poly[0][0], poly[0][1])]
            draw.line(pts, fill=color, width=2)
        else:
            draw.rectangle(det["bbox"], outline=color, width=2)

        display = f"{text[:35]}"
        tb = draw.textbbox((x1, y1 - 13), display, font=font)
        draw.rectangle([tb[0] - 1, tb[1] - 1, tb[2] + 1, tb[3] + 1], fill=(0, 0, 0))
        draw.text((x1, y1 - 13), display, fill=color, font=font)

    title = f"Page {page_num} | {len(detections)} det | {stats['qwen']} Qwen / {stats['paddle']} Paddle"
    draw.text((10, 10), title, fill=(255, 0, 0), font=font_title)
    return img


def main():
    parser = argparse.ArgumentParser(description="Hybrid PaddleOCR + Qwen2.5-VL text extraction")
    parser.add_argument("--input", required=True, help="Input PDF path")
    parser.add_argument("--output-dir", default="./output_visuals", help="Output directory")
    parser.add_argument("--lang", default="korean", help="OCR language")
    parser.add_argument("--conf-threshold", type=float, default=CONF_THRESHOLD,
                        help="PaddleOCR confidence threshold; below this sends to Qwen (default: 0.9)")
    args = parser.parse_args()

    pdf_path = Path(args.input)
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Phase 1: PaddleOCR detection (separate process) ===
    print("=== Phase 1: PaddleOCR Detection (subprocess) ===")
    detect_json = output_dir / f"{pdf_path.stem}_detections_intermediate.json"
    detect_script = Path(__file__).parent / "ocr_detect.py"

    cmd = [
        sys.executable, str(detect_script),
        "--input", str(pdf_path),
        "--output", str(detect_json),
        "--lang", args.lang,
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Phase 1 failed with exit code {result.returncode}")
        return

    with open(detect_json) as f:
        detect_data = json.load(f)

    total_det = sum(len(p["detections"]) for p in detect_data["pages"])
    low_conf = sum(
        1 for p in detect_data["pages"]
        for d in p["detections"]
        if d["paddle_conf"] < args.conf_threshold
    )
    print(f"Phase 1 complete: {total_det} regions, {low_conf} below conf {args.conf_threshold} → sending to Qwen.\n")

    # === Phase 2: Qwen recognition (individual crops, low-confidence only) ===
    print("=== Phase 2: Qwen Recognition ===")
    print(f"Converting {pdf_path.name} to images...")
    images = pdf_to_images(str(pdf_path))

    model, processor = load_qwen()

    all_results = []
    total_qwen = 0
    total_paddle = 0

    for page_data in detect_data["pages"]:
        page_num = page_data["page"]
        detections = page_data["detections"]
        img = images[page_num - 1]

        low = [d for d in detections if d["paddle_conf"] < args.conf_threshold]
        high = len(detections) - len(low)
        print(f"\nPage {page_num}/{len(images)}: {len(detections)} det, {len(low)} → Qwen, {high} → keep Paddle")

        t0 = time.time()
        qwen_count = 0
        for det in detections:
            if det["paddle_conf"] >= args.conf_threshold:
                continue
            crop = crop_region(img, det)
            qwen_text = recognize_single(model, processor, crop)
            det["qwen_text"] = qwen_text
            qwen_count += 1
            if qwen_count % 10 == 0:
                print(f"  {qwen_count}/{len(low)} crops recognized...")

        elapsed = time.time() - t0
        print(f"  Qwen: {qwen_count} crops in {elapsed:.1f}s")

        for det in detections:
            p = det["paddle_text"]
            q = det.get("qwen_text", "")
            conf = det["paddle_conf"]
            src = "Q" if q else "P"
            text = q if q else p
            marker = " *" if q and p != q else ""
            print(f"  [{det['bbox']}] ({conf:.2f}) [{src}] {text}{marker}")

        stats = {"qwen": qwen_count, "paddle": len(detections) - qwen_count}
        total_qwen += stats["qwen"]
        total_paddle += stats["paddle"]

        annotated = draw_detections(img, detections, page_num, stats)
        out_path = output_dir / f"hybrid_page_{page_num:02d}.png"
        annotated.save(out_path)

        all_results.append({
            "page": page_num,
            "width": img.width,
            "height": img.height,
            "detections": detections,
        })

    json_path = output_dir / f"{pdf_path.stem}_hybrid.json"
    with open(json_path, "w") as f:
        json.dump({"source": str(pdf_path), "pages": len(images), "results": all_results}, f, indent=2, ensure_ascii=False)

    detect_json.unlink(missing_ok=True)

    total = sum(len(r["detections"]) for r in all_results)
    print(f"\nDone. {total} regions across {len(images)} pages.")
    print(f"  Qwen recognized: {total_qwen} | Paddle kept: {total_paddle}")
    print(f"JSON: {json_path}")
    print(f"Visuals: {output_dir}/")


if __name__ == "__main__":
    main()
