# CAD AI - Qwen Backend

AI-powered text extraction pipeline for CAD engineering drawings using **Qwen2.5-VL-7B-Instruct** (vision-language model) and **PaddleOCR**, with full **Label Studio** integration for collaborative annotation.

## Architecture

```
                          INPUT (PDF / Image)
                                 │
                    ┌────────────┴────────────┐
                    │   PDF → Images (300 DPI) │
                    │   Max dimension: 2048px  │
                    └────────────┬────────────┘
                                 │
          ┌──────────────────────┴──────────────────────┐
          │          HYBRID OCR PIPELINE                 │
          │                                              │
          │  Phase 1: PaddleOCR Detection (subprocess)   │
          │  ├─ Text regions + confidence scores         │
          │  └─ Bounding boxes + polygons                │
          │                                              │
          │  Phase 2: Qwen2.5-VL Recognition (selective) │
          │  ├─ Only low-confidence regions (< 0.9)      │
          │  ├─ Crops each region with 4px padding       │
          │  └─ VLM reads exact text from crop           │
          │                                              │
          └──────────────────────┬──────────────────────┘
                                 │
               ┌─────────────────┼─────────────────┐
               │                 │                  │
          Annotated PNGs    JSON Results    Label Studio
          (color-coded)     (structured)    Predictions
```

**Key design decisions:**
- PaddleOCR runs as a **subprocess** to avoid GPU memory conflicts with PyTorch/Qwen
- Qwen model uses **4-bit NF4 quantization** (~8GB VRAM)
- Qwen is invoked **selectively** only on low-confidence regions, not on every detection
- Model loading is **lazy** (first prediction call) to avoid timeout during Label Studio `/setup`

## Project Structure

```
qwen_backend/
├── ocr_hybrid.py              # Production hybrid pipeline (PaddleOCR + Qwen)
├── ocr_detect.py              # Phase 1: PaddleOCR detection (runs as subprocess)
├── ocr_extract.py             # Standalone PaddleOCR with visualization
├── infer.py                   # Qwen-only structured text extraction
├── infer_bbox.py              # Qwen-only bounding box detection
├── infer_ocr.py               # Qwen-only OCR with bounding boxes
├── label_studio_backend/
│   ├── _wsgi.py               # WSGI app entry point (Flask server)
│   ├── model.py               # HybridOCRBackend (Label Studio ML backend)
│   ├── pdf_webhook.py         # Auto-converts PDF uploads to page images
│   └── import_pdf.py          # Batch PDF import utility
├── output/                    # JSON inference results
└── output_visuals/            # Annotated images
```

### File Roles

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| `ocr_hybrid.py` | Production pipeline. Phase 1: PaddleOCR detects all text regions. Phase 2: Qwen re-reads low-confidence crops. | PDF | Annotated PNGs + JSON |
| `ocr_detect.py` | PaddleOCR-only detection. Designed to run as subprocess. | PDF/Image | JSON (detections) |
| `ocr_extract.py` | Standalone PaddleOCR with polygon visualization. | PDF | Annotated PNGs + JSON |
| `infer.py` | Qwen-only. Extracts text, dimensions, tables as structured text. | PDF | JSON (analysis text) |
| `infer_bbox.py` | Qwen-only. Detects element categories (TEXT_BLOCK, DIMENSION, TABLE, DRAWING_VIEW) with bounding boxes. | PDF | Annotated PNGs + JSON |
| `infer_ocr.py` | Qwen-only. Full OCR — reads every text element with coordinates. | PDF | Annotated PNGs + JSON |

## Requirements

### Hardware
- NVIDIA GPU with ~8GB+ VRAM (for 4-bit quantized Qwen2.5-VL-7B)
- CUDA-compatible environment

### Python Dependencies

```
torch
transformers
qwen-vl-utils
bitsandbytes
accelerate
PyMuPDF            # (fitz)
Pillow
paddleocr
paddlepaddle-gpu   # or paddlepaddle for CPU
numpy
requests
label-studio-ml
flask
```

## Usage

### Standalone Hybrid Pipeline

```bash
python ocr_hybrid.py --input drawing.pdf --output-dir ./results --lang korean --conf-threshold 0.9
```

Arguments:
- `--input` — Path to input PDF (required)
- `--output-dir` — Output directory for annotated PNGs and JSON (default: `./output_visuals`)
- `--lang` — PaddleOCR language (default: `korean`)
- `--conf-threshold` — Confidence threshold; regions below this are sent to Qwen (default: `0.9`)

### Other Inference Scripts

```bash
# Qwen-only structured extraction
python infer.py --input drawing.pdf --output-dir ./output

# Qwen-only bounding box detection (categorized)
python infer_bbox.py --input drawing.pdf --output-dir ./output_visuals

# Qwen-only OCR with coordinates
python infer_ocr.py --input drawing.pdf --output-dir ./output_visuals

# PaddleOCR-only with visualization
python ocr_extract.py --input drawing.pdf --output-dir ./output_visuals --lang korean
```

### Label Studio ML Backend

#### Start the server

```bash
cd label_studio_backend
python _wsgi.py --port 9091 --log-level INFO
```

Server options:
- `-p`, `--port` — Server port (default: `9091`)
- `--host` — Server host (default: `0.0.0.0`)
- `--debug` — Enable Flask debug mode
- `--log-level` — `DEBUG`, `INFO`, `WARNING`, `ERROR`
- `--check` — Validate model instance before launch
- `--model-dir` — Model storage directory

#### Connect to Label Studio

1. In Label Studio, go to **Project Settings > Machine Learning**
2. Add ML Backend URL: `http://localhost:9091`
3. Predictions will appear as bounding boxes with transcribed text

#### Set up PDF auto-conversion webhook

1. Go to **Project Settings > Webhooks > Add Webhook**
2. URL: `http://localhost:9091/pdf-convert`
3. Events: **Task Created**

When a PDF is uploaded, the webhook automatically:
- Converts each page to a PNG image
- Creates one Label Studio task per page
- Deletes the original PDF task

#### Batch import PDFs

```bash
python label_studio_backend/import_pdf.py \
  --pdf /path/to/drawing.pdf \
  --project 1 \
  --ls-url http://localhost:8080 \
  --api-key <token>
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LABEL_STUDIO_URL` | Label Studio server URL | `http://localhost:8080` |
| `LABEL_STUDIO_API_KEY` | Label Studio API token | — |
| `HYBRID_OCR_CONF_THRESHOLD` | Confidence threshold for Qwen fallback | `0.9` |
| `HYBRID_OCR_LANG` | PaddleOCR language code | `korean` |
| `MODEL_DIR` | Model storage directory | `/tmp/label-studio-ml-backend` |
| `REDIS_HOST` | Redis host for job queue | `localhost` |
| `REDIS_PORT` | Redis port | `6379` |
| `RQ_QUEUE_NAME` | Redis queue name | `default` |

### Constants (Shared Across Scripts)

| Constant | Value | Description |
|----------|-------|-------------|
| `MODEL_ID` | `Qwen/Qwen2.5-VL-7B-Instruct` | Vision-language model |
| `RENDER_DPI` | `300` | PDF rendering resolution |
| `MAX_DIM` | `2048` | Maximum image dimension (px) |
| `MAX_PIXELS` | `2048 * 2048` | Qwen processor max input pixels |
| `MIN_PIXELS` | `256 * 256` | Qwen processor min input pixels |
| `CROP_PAD` | `4` | Padding (px) around cropped regions |
| `CONF_THRESHOLD` | `0.9` | Default PaddleOCR confidence threshold |

## Output Format

### Hybrid Pipeline JSON

```json
{
  "source": "drawing.pdf",
  "pages": 10,
  "results": [
    {
      "page": 1,
      "width": 2048,
      "height": 1448,
      "detections": [
        {
          "paddle_text": "SECTION A-A",
          "paddle_conf": 0.72,
          "qwen_text": "SECTION A-A",
          "bbox": [120, 340, 290, 365],
          "polygon": [[120,340], [290,340], [290,365], [120,365]]
        }
      ]
    }
  ]
}
```

### Label Studio Prediction Format

Each detection produces two annotation entries — a rectangle label and a text transcription:

```json
{
  "result": [
    {
      "id": "region_1_0",
      "type": "rectanglelabels",
      "from_name": "bbox",
      "to_name": "image",
      "original_width": 2048,
      "original_height": 1448,
      "value": {
        "x": 5.86,
        "y": 23.48,
        "width": 8.30,
        "height": 1.73,
        "rectanglelabels": ["Text"]
      }
    },
    {
      "id": "region_1_0",
      "type": "textarea",
      "from_name": "transcription",
      "to_name": "image",
      "value": {
        "text": ["SECTION A-A"]
      }
    }
  ],
  "score": 1.0
}
```

Coordinates are normalized to percentages (0-100) of image dimensions.

### Visualization Color Coding

**Hybrid pipeline** (`ocr_hybrid.py`):
- Green — Qwen-recognized text (low-confidence PaddleOCR region re-read by Qwen)
- Yellow — PaddleOCR-kept text (high-confidence, used as-is)

**Bbox detection** (`infer_bbox.py`):
- Green — `TEXT_BLOCK`
- Orange — `DIMENSION`
- Blue — `TABLE`
- Magenta — `DRAWING_VIEW`
- Red — `UNKNOWN`

## Label Studio Labeling Config

Recommended labeling configuration for the annotation project:

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
