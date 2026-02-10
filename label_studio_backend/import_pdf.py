"""Import PDFs into Label Studio as per-page images.

Converts each PDF page to a PNG, uploads it to Label Studio via API,
and creates one task per page.

Usage:
    python import_pdf.py --pdf /path/to/file.pdf
    python import_pdf.py --pdf /path/to/file.pdf --project 1 --ls-url http://localhost:8080 --api-key <token>
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import fitz
import requests
from PIL import Image

RENDER_DPI = 300
MAX_DIM = 2048


def pdf_to_page_images(pdf_path: str) -> list[tuple[Image.Image, int]]:
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        if img.width > MAX_DIM or img.height > MAX_DIM:
            img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
        pages.append((img, i + 1))
    doc.close()
    return pages


def upload_and_create_tasks(pages, pdf_name, project_id, ls_url, api_key):
    headers = {"Authorization": f"Token {api_key}"}
    imported = 0

    for img, page_num in pages:
        # Save page image to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.save(tmp, format="PNG")
            tmp_path = tmp.name

        filename = f"{pdf_name}_page_{page_num:03d}.png"

        try:
            # Upload file to Label Studio
            with open(tmp_path, "rb") as f:
                resp = requests.post(
                    f"{ls_url}/api/projects/{project_id}/import",
                    headers=headers,
                    files={"file": (filename, f, "image/png")},
                    timeout=120,
                )
            resp.raise_for_status()
            imported += 1
            print(f"  Page {page_num}: uploaded as {filename}")
        except requests.HTTPError as e:
            print(f"  Page {page_num}: FAILED - {e} - {resp.text}", file=sys.stderr)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    return imported


def main():
    parser = argparse.ArgumentParser(description="Import PDF pages into Label Studio")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--project", type=int, default=1, help="Label Studio project ID (default: 1)")
    parser.add_argument("--ls-url", default=None, help="Label Studio URL (default: env LABEL_STUDIO_URL or http://localhost:8080)")
    parser.add_argument("--api-key", default=None, help="Label Studio API key (default: env LABEL_STUDIO_API_KEY)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found", file=sys.stderr)
        sys.exit(1)

    ls_url = (args.ls_url or os.environ.get("LABEL_STUDIO_URL") or "http://localhost:8080").rstrip("/")
    api_key = args.api_key or os.environ.get("LABEL_STUDIO_API_KEY", "")
    if not api_key:
        print("Error: API key required. Set LABEL_STUDIO_API_KEY or use --api-key", file=sys.stderr)
        sys.exit(1)

    print(f"Converting {pdf_path.name} to page images...")
    pages = pdf_to_page_images(str(pdf_path))
    print(f"  {len(pages)} pages extracted.")

    print(f"Uploading to Label Studio project {args.project} at {ls_url}...")
    count = upload_and_create_tasks(pages, pdf_path.stem, args.project, ls_url, api_key)
    print(f"Done. {count}/{len(pages)} pages imported.")


if __name__ == "__main__":
    main()
