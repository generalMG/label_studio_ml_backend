import argparse
import json
import time
from pathlib import Path

import fitz
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_PIXELS = 2048 * 2048
MIN_PIXELS = 256 * 256
RENDER_DPI = 300
MAX_DIM = 2048

PROMPT = (
    "Extract the following from this engineering drawing:\n"
    "1. All visible text (labels, notes, annotations)\n"
    "2. Dimensional information (measurements, tolerances)\n"
    "3. Any tables (title block, BOM, revision history)\n\n"
    "Return the results in structured format."
)


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
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, generated_ids)]
    output = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    print(f"  Inference: {time.time() - t0:.1f}s")
    return output


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL CAD drawing extraction")
    parser.add_argument("--input", required=True, help="Input PDF path")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
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

    results = []
    for i, img in enumerate(images):
        page_num = i + 1
        print(f"\nPage {page_num}/{len(images)}...")
        analysis = run_inference(model, processor, img)
        results.append({"page": page_num, "width": img.width, "height": img.height, "analysis": analysis})
        print(analysis)
        print("---")

    output_file = output_dir / f"{pdf_path.stem}_results.json"
    with open(output_file, "w") as f:
        json.dump({"source": str(pdf_path), "pages": len(images), "results": results}, f, indent=2)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
