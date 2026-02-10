"""Phase 1: PaddleOCR text detection. Run as separate process to avoid GPU conflicts with PyTorch."""
import argparse
import json
import sys
from pathlib import Path

import fitz
import numpy as np
from PIL import Image
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
        print(f"  Page {i + 1}: {img.width}x{img.height}", file=sys.stderr)
        images.append(img)
    doc.close()
    return images


def detect_text(ocr_engine, image: Image.Image) -> list[dict]:
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
        poly = polys[idx].tolist() if idx < len(polys) else []
        box = boxes[idx].tolist() if idx < len(boxes) else []

        if len(box) == 4:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        elif poly:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        else:
            continue

        detections.append({
            "paddle_text": texts[idx],
            "paddle_conf": round(float(scores[idx]), 4),
            "bbox": [x1, y1, x2, y2],
            "polygon": [[int(p[0]), int(p[1])] for p in poly] if poly else [],
        })
    return detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, help="Output JSON path for detections")
    parser.add_argument("--lang", default="korean")
    args = parser.parse_args()

    pdf_path = Path(args.input)
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found", file=sys.stderr)
        sys.exit(1)

    print("Converting PDF to images...", file=sys.stderr)
    images = pdf_to_images(str(pdf_path))
    print(f"{len(images)} pages.\n", file=sys.stderr)

    print("Initializing PaddleOCR...", file=sys.stderr)
    ocr = PaddleOCR(
        use_textline_orientation=True,
        lang=args.lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )

    all_pages = []
    for i, img in enumerate(images):
        detections = detect_text(ocr, img)
        print(f"  Page {i + 1}: {len(detections)} regions", file=sys.stderr)
        all_pages.append({
            "page": i + 1,
            "width": img.width,
            "height": img.height,
            "detections": detections,
        })

    total = sum(len(p["detections"]) for p in all_pages)
    print(f"Total: {total} regions detected.", file=sys.stderr)

    output = {
        "source": str(pdf_path),
        "pages_count": len(images),
        "pages": all_pages,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved detections to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
