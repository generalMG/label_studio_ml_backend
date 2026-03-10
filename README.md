# Labeling - Qwen Backend

Hybrid OCR backend for technical drawings and document images, using:
- `Qwen/Qwen3-VL-8B-Instruct` (4-bit quantized)
- PaddleOCR (detection + optional recognition)
- Label Studio ML backend integration

## Current Capabilities

- Hybrid pipeline: PaddleOCR first pass, Qwen fallback on low-confidence regions.
- `--detect-only` mode in detection stage to force all regions through Qwen.
- Qwen-only utility scripts for structured extraction, OCR, and bbox grounding.
- Label Studio ML backend with lazy model loading.
- PDF auto-conversion webhook (`/pdf-convert`) that splits PDF tasks into page images.
- Security controls for webhook auth, host allowlisting, file-size/page limits, and local-file restrictions.

## Pipeline Overview

```text
Input PDF/Image
   |
   +--> PaddleOCR detection (ocr_detect.py, subprocess)
   |      - Regions + confidence
   |
   +--> Hybrid merge (ocr_hybrid.py / label_studio_backend/model.py)
          - High confidence: keep PaddleOCR text
          - Low confidence: crop + Qwen recognition
          - Output: JSON + optional annotated images
```

## Project Structure

```text
qwen_backend/
├── ocr_hybrid.py
├── ocr_detect.py
├── ocr_extract.py
├── infer.py
├── infer_bbox.py
├── infer_ocr.py
├── compare_models.py
├── label_studio_backend/
│   ├── _wsgi.py
│   ├── model.py
│   ├── pdf_webhook.py
│   ├── import_pdf.py
│   └── security_utils.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── tests/
    └── test_security_utils.py
```

## Requirements

### Hardware

- NVIDIA GPU recommended (8GB+ VRAM for 4-bit Qwen3-VL-8B).
- CUDA-capable environment.

### Software

- Python 3.10+
- Linux environment recommended for GPU runtime
- Optional: Docker + Docker Compose

Install dependencies:

```bash
pip install -r requirements.txt
```

## Standalone Script Usage

### 1) Hybrid OCR (recommended for batch extraction)

```bash
python ocr_hybrid.py \
  --input /path/to/file.pdf \
  --output-dir ./output_visuals \
  --lang korean \
  --conf-threshold 0.99
```

Optional flags:
- `--detect-only`: skip PaddleOCR recognition and send all detected regions to Qwen.
- `--max-pages`: max PDF pages allowed (default: `200`).
- `--max-render-pixels`: max rendered pixels/page before downscale (default: `16777216`).

### 2) Detection-only stage

```bash
python ocr_detect.py \
  --input /path/to/file.pdf \
  --output ./detections.json \
  --lang korean
```

### 3) Qwen-only utilities

```bash
python infer.py --input /path/to/file.pdf --output-dir ./output
python infer_bbox.py --input /path/to/file.pdf --output-dir ./output_visuals
python infer_ocr.py --input /path/to/file.pdf --output-dir ./output_visuals
```

### 4) Compare Qwen2.5 vs Qwen3

```bash
python compare_models.py --input /path/to/file.pdf --page 1 --output-dir ./compare_output
```

## Label Studio Integration (Local Setup)

This is the direct setup when Label Studio and backend run on the same machine.
Get your Label Studio API token from **Account & Settings > Access Token**.

### 1) Prepare environment variables

```bash
export LABEL_STUDIO_URL=http://localhost:8080
export LABEL_STUDIO_API_KEY=<your_label_studio_token>

# Webhook security (required by default)
export REQUIRE_WEBHOOK_AUTH=true
export PDF_WEBHOOK_SECRET=<set_a_long_random_secret>
export PDF_WEBHOOK_SECRET_HEADER=X-Webhook-Secret

# Optional compatibility/security controls
export ALLOWED_DOWNLOAD_HOSTS=
export ALLOW_LOCAL_TASK_FILES=false
export MAX_DOWNLOAD_BYTES=104857600
export MAX_PDF_PAGES=200
export MAX_RENDER_PIXELS=16777216
export DOWNLOAD_CONNECT_TIMEOUT_SEC=10
export DOWNLOAD_READ_TIMEOUT_SEC=120

# OCR behavior
export HYBRID_OCR_CONF_THRESHOLD=0.9
export HYBRID_OCR_LANG=korean
```

### 2) Start Label Studio ML backend

From repo root:

```bash
python label_studio_backend/_wsgi.py --port 9091 --log-level INFO
```

### 3) Configure Label Studio project labeling interface

Use this labeling config (names must match backend output):

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="bbox" toName="image">
    <Label value="Text" background="green"/>
  </RectangleLabels>
  <TextArea name="transcription" toName="image"
            editable="true" perRegion="true" required="false"
            maxSubmissions="1" rows="3"/>
</View>
```

### 4) Connect ML backend in Label Studio

1. Open project: **Settings > Machine Learning**.
2. Add backend URL: `http://localhost:9091`.
3. Enable interactive preannotations if desired.

If Label Studio runs in Docker but backend runs on host, use `http://host.docker.internal:9091` instead.

### 5) Configure PDF conversion webhook

1. Open project: **Settings > Webhooks > Add Webhook**.
2. URL: `http://localhost:9091/pdf-convert`
3. Events: `Task Created` (or `TASK_CREATED` equivalent)
4. Add header:
   - Header name: `X-Webhook-Secret` (or value of `PDF_WEBHOOK_SECRET_HEADER`)
   - Header value: exactly `PDF_WEBHOOK_SECRET`

### 6) Verify end-to-end

- Upload an image task: should get bbox + transcription preannotations.
- Upload a PDF task: webhook should create per-page image tasks and delete original PDF task.

## Label Studio Integration (Docker Compose)

### 1) Set environment values

Create a `.env` file in repo root:

```env
POSTGRES_PASSWORD=replace_me
PDF_WEBHOOK_SECRET=replace_with_long_random_secret
REQUIRE_WEBHOOK_AUTH=true
LABEL_STUDIO_HOST=http://localhost:8080
LABEL_STUDIO_API_KEY=<your_label_studio_token>
```

`docker-compose.yml` now auto-forwards `LABEL_STUDIO_API_KEY` to the backend, so no compose file edits are required.

### 2) Start stack

```bash
docker compose up --build -d
```

Services:
- Label Studio: `http://localhost:8080`
- Backend: `http://localhost:9090`

### 3) Connect ML backend from Label Studio UI

Because both run in the same Compose network, use:
- ML backend URL: `http://qwen-backend:9090`
- Webhook URL: `http://qwen-backend:9090/pdf-convert`

### 4) Add webhook header

Use the same header/secret pair as local setup:
- `X-Webhook-Secret: <PDF_WEBHOOK_SECRET>`

## Batch Import Utility

You can convert a PDF into page-image tasks directly via API:

```bash
python label_studio_backend/import_pdf.py \
  --pdf /path/to/file.pdf \
  --project 1 \
  --ls-url http://localhost:8080 \
  --api-key <your_label_studio_token>
```

## Configuration Reference

### Core backend env vars

| Variable | Purpose | Default |
|---|---|---|
| `LABEL_STUDIO_URL` | Label Studio base URL | `http://localhost:8080` |
| `LABEL_STUDIO_API_KEY` | API token used for authenticated LS requests | empty |
| `LABEL_STUDIO_ACCESS_TOKEN` | Alternate token var | empty |
| `HYBRID_OCR_CONF_THRESHOLD` | Qwen fallback threshold in ML backend | `0.9` |
| `HYBRID_OCR_LANG` | PaddleOCR language | `korean` |

### Security env vars

| Variable | Purpose | Default |
|---|---|---|
| `REQUIRE_WEBHOOK_AUTH` | Require secret for `/pdf-convert` webhook | `true` |
| `PDF_WEBHOOK_SECRET` | Shared secret for webhook auth | empty in code, set in deployment |
| `PDF_WEBHOOK_SECRET_HEADER` | Header name for shared secret | `X-Webhook-Secret` |
| `ALLOWED_DOWNLOAD_HOSTS` | Extra allowed hosts (comma-separated) | empty |
| `ALLOW_LOCAL_TASK_FILES` | Allow `file://` and raw local paths | `false` |

### File/processing limits

| Variable | Purpose | Default |
|---|---|---|
| `MAX_DOWNLOAD_BYTES` | Max downloaded file size | `104857600` |
| `MAX_PDF_PAGES` | Max pages per PDF | `200` |
| `MAX_RENDER_PIXELS` | Max render pixels per page | `16777216` |
| `DOWNLOAD_CONNECT_TIMEOUT_SEC` | HTTP connect timeout | `10` |
| `DOWNLOAD_READ_TIMEOUT_SEC` | HTTP read timeout | `120` |

## Output Artifacts

- `ocr_hybrid.py` outputs:
  - `{input_stem}_hybrid.json`
  - `hybrid_page_XX.png`
- `ocr_detect.py` outputs:
  - detection JSON (`pages[].detections[]` with bbox/polygon/confidence)
- Label Studio backend outputs:
  - prediction list with paired `rectanglelabels` + `textarea` entries

## Troubleshooting

- `401 unauthorized` on `/pdf-convert`:
  - Webhook header is missing or secret mismatch.
- `503 webhook auth misconfigured`:
  - `REQUIRE_WEBHOOK_AUTH=true` but `PDF_WEBHOOK_SECRET` not set.
- `Host not allowed for download`:
  - Add host to `ALLOWED_DOWNLOAD_HOSTS` or use Label Studio-hosted files.
- `Download exceeds MAX_DOWNLOAD_BYTES` or PDF page limit errors:
  - Increase limits if trusted and resources permit.
- No predictions in Label Studio:
  - Verify labeling config names: `bbox`, `transcription`, `image`.
  - Check backend logs for Paddle/Qwen model errors.
