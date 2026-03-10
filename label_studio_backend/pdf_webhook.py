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
import threading
from pathlib import Path
from urllib.parse import urlparse

import fitz
import requests
from flask import Blueprint, request, jsonify
from PIL import Image
try:
    from .security_utils import (
        build_allowed_hosts,
        parse_authorization_token,
        parse_bool,
        same_host,
        validate_remote_http_url,
        verify_shared_secret,
    )
except ImportError:
    from security_utils import (
        build_allowed_hosts,
        parse_authorization_token,
        parse_bool,
        same_host,
        validate_remote_http_url,
        verify_shared_secret,
    )

logger = logging.getLogger(__name__)

pdf_webhook = Blueprint("pdf_webhook", __name__)

RENDER_DPI = 300
MAX_DIM = 2048
PDF_EXTENSIONS = (".pdf",)
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
MAX_PDF_PAGES = 200
MAX_RENDER_PIXELS = 16 * 1024 * 1024
DOWNLOAD_CONNECT_TIMEOUT_SEC = 10.0
DOWNLOAD_READ_TIMEOUT_SEC = 120.0


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
    allowed_hosts = build_allowed_hosts(ls_url, os.environ.get("ALLOWED_DOWNLOAD_HOSTS", ""))
    return ls_url, api_key, allowed_hosts


def _get_int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(1, value)


def _get_float_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(0.1, value)


def _is_pdf_url(url: str) -> bool:
    path = urlparse(url).path if "://" in url else url
    return any(path.lower().endswith(ext) for ext in PDF_EXTENSIONS)


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


def _check_webhook_auth():
    require_auth = parse_bool(os.environ.get("REQUIRE_WEBHOOK_AUTH"), default=True)
    if not require_auth:
        return True, "", 200

    secret = os.environ.get("PDF_WEBHOOK_SECRET", "").strip()
    if not secret:
        logger.error(
            "REQUIRE_WEBHOOK_AUTH=true but PDF_WEBHOOK_SECRET is not configured."
        )
        return False, "webhook auth misconfigured", 503

    header_name = os.environ.get("PDF_WEBHOOK_SECRET_HEADER", "X-Webhook-Secret").strip()
    provided = request.headers.get(header_name, "")
    if not provided:
        provided = parse_authorization_token(request.headers.get("Authorization", ""))

    if not verify_shared_secret(provided, secret):
        return False, "unauthorized", 401
    return True, "", 200


def _download_file(
    url: str,
    ls_url: str,
    api_key: str,
    allowed_hosts,
    max_download_bytes: int,
    connect_timeout: float,
    read_timeout: float,
) -> bytes:
    if url.startswith("/data/"):
        url = f"{ls_url}{url}"
    validate_remote_http_url(url, allowed_hosts)

    headers = {}
    if api_key and same_host(url, ls_url):
        headers["Authorization"] = f"Token {api_key}"

    timeout = (connect_timeout, read_timeout)
    chunks = bytearray()

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
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            chunks.extend(chunk)
            if len(chunks) > max_download_bytes:
                raise ValueError(
                    f"Download exceeds MAX_DOWNLOAD_BYTES={max_download_bytes}"
                )
    return bytes(chunks)


def _pdf_bytes_to_pages(
    pdf_bytes: bytes,
    max_pages: int,
    max_render_pixels: int,
) -> list[Image.Image]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
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
            pages.append(img)
    finally:
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


def _process_pdf_task(task: dict, project_id: int, ls_url: str, api_key: str, allowed_hosts):
    """Convert a PDF task into per-page image tasks, then delete the original."""
    task_id = task.get("id")
    image_url = task.get("data", {}).get("image", "")

    if not image_url or not _is_pdf_url(image_url):
        return

    logger.info("PDF detected in task %s: %s — converting to page images", task_id, image_url)

    try:
        max_download_bytes = _get_int_env("MAX_DOWNLOAD_BYTES", MAX_DOWNLOAD_BYTES)
        max_pdf_pages = _get_int_env("MAX_PDF_PAGES", MAX_PDF_PAGES)
        max_render_pixels = _get_int_env("MAX_RENDER_PIXELS", MAX_RENDER_PIXELS)
        connect_timeout = _get_float_env(
            "DOWNLOAD_CONNECT_TIMEOUT_SEC",
            DOWNLOAD_CONNECT_TIMEOUT_SEC,
        )
        read_timeout = _get_float_env(
            "DOWNLOAD_READ_TIMEOUT_SEC",
            DOWNLOAD_READ_TIMEOUT_SEC,
        )

        pdf_bytes = _download_file(
            image_url,
            ls_url,
            api_key,
            allowed_hosts,
            max_download_bytes,
            connect_timeout,
            read_timeout,
        )
        pages = _pdf_bytes_to_pages(pdf_bytes, max_pdf_pages, max_render_pixels)
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
    ok, reason, status_code = _check_webhook_auth()
    if not ok:
        return jsonify({"status": "error", "reason": reason}), status_code

    payload = request.json
    if not payload:
        return jsonify({"status": "ignored", "reason": "empty payload"}), 200

    action = payload.get("action")

    # Only handle task creation events
    if action not in ("TASKS_CREATED", "TASK_CREATED"):
        return jsonify({"status": "ignored", "reason": f"action={action}"}), 200

    ls_url, api_key, allowed_hosts = _get_ls_config()
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
            _process_pdf_task(t, project_id, ls_url, api_key, allowed_hosts)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return jsonify({"status": "processing", "pdf_tasks": len(pdf_tasks)}), 200
