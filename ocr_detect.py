"""Phase 1: PaddleOCR text detection. Run as separate process to avoid GPU conflicts with PyTorch."""
import argparse
import json
import os
import sys
from pathlib import Path

import fitz
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

RENDER_DPI = 300
MAX_DIM = 2048
MAX_PDF_PAGES = 200
MAX_RENDER_PIXELS = 16 * 1024 * 1024


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
        for i, page in enumerate(doc):
            mat = _safe_render_matrix(page, max_render_pixels)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            if img.width > MAX_DIM or img.height > MAX_DIM:
                img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
            print(f"  Page {i + 1}: {img.width}x{img.height}", file=sys.stderr)
            images.append(img)
    finally:
        doc.close()
    return images


def detect_text(ocr_engine, image: Image.Image, detect_only: bool = False) -> list[dict]:
    img_array = np.array(image)
    results = list(ocr_engine.predict(img_array))
    detections = []
    if not results:
        return detections

    r = results[0]
    polys = r.get("dt_polys", [])

    if detect_only:
        for idx in range(len(polys)):
            poly = polys[idx].tolist() if idx < len(polys) else []
            if not poly:
                continue
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            detections.append({
                "paddle_text": "",
                "paddle_conf": 0.0,
                "bbox": [x1, y1, x2, y2],
                "polygon": [[int(p[0]), int(p[1])] for p in poly],
            })
        return detections

    texts = r.get("rec_texts", [])
    scores = r.get("rec_scores", [])
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
    parser.add_argument(
        "--max-pages",
        type=int,
        default=int(os.environ.get("MAX_PDF_PAGES", MAX_PDF_PAGES)),
        help="Maximum PDF page count allowed",
    )
    parser.add_argument(
        "--max-render-pixels",
        type=int,
        default=int(os.environ.get("MAX_RENDER_PIXELS", MAX_RENDER_PIXELS)),
        help="Maximum rendered pixels per page before downscaling",
    )
    parser.add_argument("--detect-only", action="store_true",
                        help="Skip PaddleOCR recognition, return bboxes only (conf=0)")
    args = parser.parse_args()

    pdf_path = Path(args.input)
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found", file=sys.stderr)
        sys.exit(1)
    if args.max_pages < 1:
        print("Error: --max-pages must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.max_render_pixels < 1:
        print("Error: --max-render-pixels must be >= 1", file=sys.stderr)
        sys.exit(1)

    print("Converting PDF to images...", file=sys.stderr)
    try:
        images = pdf_to_images(
            str(pdf_path),
            max_pages=args.max_pages,
            max_render_pixels=args.max_render_pixels,
        )
    except Exception as e:
        print(f"Error: failed to render PDF: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"{len(images)} pages.\n", file=sys.stderr)

    print("Initializing PaddleOCR...", file=sys.stderr)
    if args.detect_only:
        print("  (detect-only mode: skipping recognition)", file=sys.stderr)
    ocr = PaddleOCR(
        use_textline_orientation=True,
        lang=args.lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )

    all_pages = []
    for i, img in enumerate(images):
        detections = detect_text(ocr, img, detect_only=args.detect_only)
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
