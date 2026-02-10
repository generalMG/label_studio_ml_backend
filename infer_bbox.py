import argparse
import json
import re
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

# Grounding prompt - asks model to return bbox coordinates
PROMPT = (
    "Detect and localize the following elements in this engineering drawing. "
    "For each element found, provide its bounding box coordinates as [x1, y1, x2, y2] "
    "where coordinates are pixel positions relative to image dimensions.\n\n"
    "Categories to detect:\n"
    "1. TEXT_BLOCK - any block of text, labels, or annotations\n"
    "2. DIMENSION - dimensional callouts, measurements, tolerances\n"
    "3. TABLE - tables including title blocks, BOM, revision history\n"
    "4. DRAWING_VIEW - main drawing views or detail views\n\n"
    "Return results as a JSON array with format:\n"
    '[{"label": "category: description", "bbox": [x1, y1, x2, y2]}]\n'
    "Use absolute pixel coordinates based on the image size."
)

# Colors for each category
COLORS = {
    "TEXT_BLOCK": (0, 255, 0),       # green
    "DIMENSION": (255, 165, 0),      # orange
    "TABLE": (0, 120, 255),          # blue
    "DRAWING_VIEW": (255, 0, 255),   # magenta
    "UNKNOWN": (255, 0, 0),          # red
}


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


def load_model():
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


def run_inference(model, processor, image: Image.Image) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    t0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=4096)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, generated_ids)]
    output = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    print(f"  Inference: {time.time() - t0:.1f}s")
    return output


def parse_detections(raw_output: str, img_w: int, img_h: int) -> list[dict]:
    """Parse model output to extract bounding box detections."""
    detections = []

    # Try parsing as JSON array first
    json_match = re.search(r'\[.*\]', raw_output, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "bbox" in item:
                        bbox = item["bbox"]
                        if len(bbox) == 4:
                            # Clamp coordinates to image bounds
                            x1 = max(0, min(int(bbox[0]), img_w))
                            y1 = max(0, min(int(bbox[1]), img_h))
                            x2 = max(0, min(int(bbox[2]), img_w))
                            y2 = max(0, min(int(bbox[3]), img_h))
                            if x2 > x1 and y2 > y1:
                                detections.append({
                                    "label": item.get("label", "UNKNOWN"),
                                    "bbox": [x1, y1, x2, y2],
                                })
                return detections
        except json.JSONDecodeError:
            pass

    # Fallback: try to find bbox patterns like [x1, y1, x2, y2] with labels
    pattern = r'"?label"?\s*:\s*"([^"]+)".*?"?bbox"?\s*:\s*\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]'
    for m in re.finditer(pattern, raw_output, re.DOTALL):
        label = m.group(1)
        x1, y1, x2, y2 = float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))
        x1 = max(0, min(int(x1), img_w))
        y1 = max(0, min(int(y1), img_h))
        x2 = max(0, min(int(x2), img_w))
        y2 = max(0, min(int(y2), img_h))
        if x2 > x1 and y2 > y1:
            detections.append({"label": label, "bbox": [x1, y1, x2, y2]})

    return detections


def get_category(label: str) -> str:
    label_upper = label.upper()
    for cat in COLORS:
        if cat in label_upper:
            return cat
    return "UNKNOWN"


def draw_detections(image: Image.Image, detections: list[dict], page_num: int) -> Image.Image:
    """Draw bounding boxes and labels on the image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_small = font

    for det in detections:
        label = det["label"]
        x1, y1, x2, y2 = det["bbox"]
        category = get_category(label)
        color = COLORS[category]

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw label background
        text_bbox = draw.textbbox((x1, y1 - 16), label, font=font_small)
        draw.rectangle([text_bbox[0] - 1, text_bbox[1] - 1, text_bbox[2] + 1, text_bbox[3] + 1], fill=color)
        draw.text((x1, y1 - 16), label, fill=(255, 255, 255), font=font_small)

    # Page number watermark
    draw.text((10, 10), f"Page {page_num} | {len(detections)} detections", fill=(255, 0, 0), font=font)

    return img


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL bbox detection on CAD drawings")
    parser.add_argument("--input", required=True, help="Input PDF path")
    parser.add_argument("--output-dir", default="./output_visuals", help="Output directory")
    args = parser.parse_args()

    pdf_path = Path(args.input)
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {pdf_path.name} to images...")
    images = pdf_to_images(str(pdf_path))
    print(f"{len(images)} pages.\n")

    model, processor = load_model()

    all_results = []
    for i, img in enumerate(images):
        page_num = i + 1
        print(f"\nPage {page_num}/{len(images)}...")

        raw_output = run_inference(model, processor, img)
        detections = parse_detections(raw_output, img.width, img.height)

        print(f"  Parsed {len(detections)} detections")
        for det in detections:
            print(f"    {det['label']}: {det['bbox']}")

        # Draw and save
        annotated = draw_detections(img, detections, page_num)
        out_path = output_dir / f"page_{page_num:02d}.png"
        annotated.save(out_path)
        print(f"  Saved: {out_path}")

        all_results.append({
            "page": page_num,
            "raw_output": raw_output,
            "detections": detections,
        })

    # Save JSON summary
    json_path = output_dir / f"{pdf_path.stem}_detections.json"
    with open(json_path, "w") as f:
        json.dump({"source": str(pdf_path), "pages": len(images), "results": all_results}, f, indent=2)
    print(f"\nJSON saved to {json_path}")
    print(f"Visuals saved to {output_dir}/")


if __name__ == "__main__":
    main()
