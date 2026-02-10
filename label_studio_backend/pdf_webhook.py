"""Webhook handler that auto-converts PDF uploads into per-page image tasks.

When Label Studio fires a TASKS_CREATED webhook, this handler:
1. Checks each new task for a PDF file URL
2. Downloads the PDF, renders each page to PNG at 300 DPI
3. Uploads the page images back to Label Studio as new tasks
4. Deletes the original PDF task

Configure in Label Studio: Project Settings > Webhooks > Add Webhook
  URL:    http://localhost:9091/pdf-convert
  Events: Task Created
"""

import io
import logging
import os
import tempfile
import threading
from pathlib import Path

import fitz
import requests
from flask import Blueprint, request, jsonify
from PIL import Image

logger = logging.getLogger(__name__)

pdf_webhook = Blueprint("pdf_webhook", __name__)

RENDER_DPI = 300
MAX_DIM = 2048
PDF_EXTENSIONS = (".pdf",)


def _get_ls_config():
    ls_url = (
        os.environ.get("LABEL_STUDIO_URL", "")
        or os.environ.get("LABEL_STUDIO_HOST", "")
        or "http://localhost:8080"
    ).rstrip("/")
    api_key = (
        os.environ.get("LABEL_STUDIO_API_KEY", "")
        or os.environ.get("LABEL_STUDIO_ACCESS_TOKEN", "")
    )
    return ls_url, api_key


def _is_pdf_url(url: str) -> bool:
    return any(url.lower().endswith(ext) for ext in PDF_EXTENSIONS)


def _download_file(url: str, ls_url: str, api_key: str) -> bytes:
    if url.startswith("/data/"):
        url = f"{ls_url}{url}"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Token {api_key}"
    resp = requests.get(url, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.content


def _pdf_bytes_to_pages(pdf_bytes: bytes) -> list[Image.Image]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        if img.width > MAX_DIM or img.height > MAX_DIM:
            img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
        pages.append(img)
    doc.close()
    return pages


def _upload_page_image(img: Image.Image, filename: str, project_id: int, ls_url: str, api_key: str):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    resp = requests.post(
        f"{ls_url}/api/projects/{project_id}/import",
        headers={"Authorization": f"Token {api_key}"},
        files={"file": (filename, buf, "image/png")},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def _delete_task(task_id: int, ls_url: str, api_key: str):
    resp = requests.delete(
        f"{ls_url}/api/tasks/{task_id}",
        headers={"Authorization": f"Token {api_key}"},
        timeout=30,
    )
    resp.raise_for_status()


def _process_pdf_task(task: dict, project_id: int, ls_url: str, api_key: str):
    """Convert a PDF task into per-page image tasks, then delete the original."""
    task_id = task.get("id")
    image_url = task.get("data", {}).get("image", "")

    if not image_url or not _is_pdf_url(image_url):
        return

    logger.info("PDF detected in task %s: %s — converting to page images", task_id, image_url)

    try:
        pdf_bytes = _download_file(image_url, ls_url, api_key)
        pages = _pdf_bytes_to_pages(pdf_bytes)
        logger.info("Task %s: %d pages extracted from PDF", task_id, len(pages))

        # Derive a clean name from the URL
        pdf_stem = Path(image_url).stem
        # Strip the UUID prefix Label Studio adds (e.g. "61c43086-YERTAYEV_ARMAN")
        if len(pdf_stem) > 9 and pdf_stem[8] == "-":
            pdf_stem = pdf_stem[9:]

        for i, img in enumerate(pages):
            filename = f"{pdf_stem}_page_{i + 1:03d}.png"
            _upload_page_image(img, filename, project_id, ls_url, api_key)
            logger.info("  Page %d/%d uploaded: %s", i + 1, len(pages), filename)

        # Delete the original PDF task
        _delete_task(task_id, ls_url, api_key)
        logger.info("Task %s (PDF) deleted, replaced with %d page images", task_id, len(pages))

    except Exception:
        logger.exception("Failed to process PDF task %s", task_id)


@pdf_webhook.route("/pdf-convert", methods=["POST"])
def handle_pdf_webhook():
    """Receive Label Studio webhook and convert PDF tasks to images."""
    payload = request.json
    if not payload:
        return jsonify({"status": "ignored", "reason": "empty payload"}), 200

    action = payload.get("action")

    # Only handle task creation events
    if action not in ("TASKS_CREATED", "TASK_CREATED"):
        return jsonify({"status": "ignored", "reason": f"action={action}"}), 200

    ls_url, api_key = _get_ls_config()
    project_id = None

    # Extract project ID from payload
    project = payload.get("project")
    if isinstance(project, dict):
        project_id = project.get("id")
    elif isinstance(project, (int, str)):
        project_id = int(project)

    if not project_id:
        # Try to get from tasks
        tasks = payload.get("tasks", [])
        if not tasks:
            task = payload.get("task")
            if task:
                tasks = [task]
        if tasks and tasks[0].get("project"):
            project_id = tasks[0]["project"]

    if not project_id:
        return jsonify({"status": "error", "reason": "no project_id"}), 400

    # Collect tasks (Label Studio sends either "tasks" list or single "task")
    tasks = payload.get("tasks", [])
    if not tasks:
        task = payload.get("task")
        if task:
            tasks = [task]

    # Filter to only PDF tasks
    pdf_tasks = [t for t in tasks if _is_pdf_url(t.get("data", {}).get("image", ""))]

    if not pdf_tasks:
        return jsonify({"status": "ok", "pdf_tasks": 0}), 200

    # Process in background thread so webhook returns quickly
    def _run():
        for t in pdf_tasks:
            _process_pdf_task(t, project_id, ls_url, api_key)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return jsonify({"status": "processing", "pdf_tasks": len(pdf_tasks)}), 200
