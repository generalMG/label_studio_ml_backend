import argparse
import json
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR

RENDER_DPI = 300
MAX_DIM = 2048


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


def run_ocr(ocr_engine, image: Image.Image) -> list[dict]:
    img_array = np.array(image)
    results = list(ocr_engine.predict(img_array))

    detections = []
    if not results:
        return detections

    r = results[0]
    texts = r.get("rec_texts", [])
    scores = r.get("rec_scores", [])
    polys = r.get("dt_polys", [])
    boxes = r.get("rec_boxes", [])

    for idx in range(len(texts)):
        text = texts[idx]
        score = float(scores[idx])
        poly = polys[idx].tolist() if idx < len(polys) else []
        box = boxes[idx].tolist() if idx < len(boxes) else []

        # bbox from rec_boxes: [x1, y1, x2, y2]
        if len(box) == 4:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        elif poly:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        else:
            continue

        detections.append({
            "text": text,
            "confidence": round(score, 4),
            "bbox": [x1, y1, x2, y2],
            "polygon": [[int(p[0]), int(p[1])] for p in poly] if poly else [],
        })

    return detections


def draw_detections(image: Image.Image, detections: list[dict], page_num: int) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Use Korean-capable font
    kr_font = "/home/mg_server/.local/share/fonts/NotoSansCJKkr-Regular.otf"
    fallback = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        font = ImageFont.truetype(kr_font, 10)
        font_title = ImageFont.truetype(kr_font, 16)
    except (IOError, OSError):
        font = ImageFont.truetype(fallback, 10)
        font_title = ImageFont.truetype(fallback, 16)

    for det in detections:
        poly = det["polygon"]
        text = det["text"]
        conf = det["confidence"]
        x1, y1 = det["bbox"][0], det["bbox"][1]

        # Draw polygon outline
        if poly:
            pts = [(p[0], p[1]) for p in poly] + [(poly[0][0], poly[0][1])]
            draw.line(pts, fill=(0, 255, 0), width=2)
        else:
            draw.rectangle(det["bbox"], outline=(0, 255, 0), width=2)

        # Label
        display = f"{text[:30]} ({conf:.2f})"
        tb = draw.textbbox((x1, y1 - 13), display, font=font)
        draw.rectangle([tb[0] - 1, tb[1] - 1, tb[2] + 1, tb[3] + 1], fill=(0, 0, 0))
        draw.text((x1, y1 - 13), display, fill=(0, 255, 0), font=font)

    draw.text((10, 10), f"Page {page_num} | {len(detections)} text regions", fill=(255, 0, 0), font=font_title)
    return img


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR text extraction from CAD PDFs")
    parser.add_argument("--input", required=True, help="Input PDF path")
    parser.add_argument("--output-dir", default="./output_visuals", help="Output directory")
    parser.add_argument("--lang", default="korean", help="OCR language (default: korean)")
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

    print("Initializing PaddleOCR...")
    ocr = PaddleOCR(
        use_textline_orientation=True,
        lang=args.lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )
    print("Ready.\n")

    all_results = []
    for i, img in enumerate(images):
        page_num = i + 1
        print(f"Page {page_num}/{len(images)}...", end=" ")

        detections = run_ocr(ocr, img)
        print(f"{len(detections)} text regions")

        for det in detections:
            print(f"  [{det['bbox']}] ({det['confidence']:.2f}) {det['text']}")

        annotated = draw_detections(img, detections, page_num)
        out_path = output_dir / f"ocr_page_{page_num:02d}.png"
        annotated.save(out_path)

        all_results.append({
            "page": page_num,
            "width": img.width,
            "height": img.height,
            "detections": detections,
        })

    json_path = output_dir / f"{pdf_path.stem}_ocr.json"
    with open(json_path, "w") as f:
        json.dump({"source": str(pdf_path), "pages": len(images), "results": all_results}, f, indent=2, ensure_ascii=False)

    total = sum(len(r["detections"]) for r in all_results)
    print(f"\nDone. {total} total text regions across {len(images)} pages.")
    print(f"JSON: {json_path}")
    print(f"Visuals: {output_dir}/")


if __name__ == "__main__":
    main()
