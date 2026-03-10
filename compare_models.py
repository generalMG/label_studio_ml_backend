"""Compare Qwen2.5-VL-7B vs Qwen3-VL-8B on the same test image.

Loads each model sequentially (4-bit NF4), runs OCR inference on a single
page, saves JSON outputs, and prints a side-by-side summary.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import fitz
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig

MAX_PIXELS = 2048 * 2048
MIN_PIXELS = 256 * 256
RENDER_DPI = 300
MAX_DIM = 2048

PROMPT = (
    "Read all text visible in this engineering drawing image. "
    "Return ONLY the extracted text, preserving layout where possible. "
    "Include Korean, English, numbers, symbols, dimensions, and annotations."
)

MODELS = [
    {
        "name": "Qwen2.5-VL-7B",
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "class": "Qwen2_5_VLForConditionalGeneration",
    },
    {
        "name": "Qwen3-VL-8B",
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "class": "Qwen3VLForConditionalGeneration",
    },
]


def load_test_image(path: str, page: int = 1) -> Image.Image:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        doc = fitz.open(str(p))
        pg = doc[page - 1]
        mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
        pix = pg.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
    else:
        img = Image.open(str(p)).convert("RGB")

    if img.width > MAX_DIM or img.height > MAX_DIM:
        img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
    return img


def load_model(model_info: dict):
    from transformers import Qwen2_5_VLForConditionalGeneration

    model_id = model_info["model_id"]
    cls_name = model_info["class"]

    print(f"\nLoading {model_info['name']} ({model_id}, 4-bit NF4)...")
    t0 = time.time()

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    if cls_name == "Qwen3VLForConditionalGeneration":
        from transformers import Qwen3VLForConditionalGeneration
        model_cls = Qwen3VLForConditionalGeneration
    else:
        model_cls = Qwen2_5_VLForConditionalGeneration

    model = model_cls.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
    )
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")
    return model, processor, load_time


def run_inference(model, processor, image: Image.Image) -> tuple[str, float]:
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
    output = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    inference_time = time.time() - t0
    return output, inference_time


def unload_model(model, processor):
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()
    print("  Model unloaded, VRAM freed.")


def main():
    parser = argparse.ArgumentParser(description="Compare Qwen2.5-VL-7B vs Qwen3-VL-8B")
    parser.add_argument("--input", default=None,
                        help="Test image (PNG/JPG) or PDF path. "
                             "Defaults to output_visuals/hybrid_page_01.png")
    parser.add_argument("--page", type=int, default=1, help="PDF page number (1-indexed)")
    parser.add_argument("--output-dir", default="./compare_output", help="Output directory")
    args = parser.parse_args()

    # Default test image
    if args.input is None:
        candidates = [
            Path(__file__).parent / "output_visuals" / "hybrid_page_01.png",
            Path(__file__).parent.parent / "label-studio-auto" / "input_samples" / "FileBundle_1-10.pdf",
        ]
        for c in candidates:
            if c.exists():
                args.input = str(c)
                break
        if args.input is None:
            print("Error: no test image found. Pass --input <path>.")
            return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Test input: {args.input}")
    image = load_test_image(args.input, args.page)
    print(f"Image size: {image.width}x{image.height}")

    results = {}

    for model_info in MODELS:
        name = model_info["name"]
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        model, processor, load_time = load_model(model_info)
        output_text, inference_time = run_inference(model, processor, image)

        results[name] = {
            "model_id": model_info["model_id"],
            "load_time_s": round(load_time, 1),
            "inference_time_s": round(inference_time, 1),
            "output_length": len(output_text),
            "output_lines": output_text.count("\n") + 1,
            "output_text": output_text,
        }

        # Save individual output
        out_file = output_dir / f"{name.replace('.', '_').replace('-', '_')}_output.txt"
        out_file.write_text(output_text, encoding="utf-8")
        print(f"  Inference: {inference_time:.1f}s | Output: {len(output_text)} chars, {output_text.count(chr(10))+1} lines")
        print(f"  Saved: {out_file}")

        unload_model(model, processor)

    # Save combined JSON
    json_path = output_dir / "comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "input": args.input,
            "image_size": [image.width, image.height],
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Qwen2.5-VL-7B':>18} {'Qwen3-VL-8B':>18}")
    print(f"{'-'*61}")

    r25 = results["Qwen2.5-VL-7B"]
    r3 = results["Qwen3-VL-8B"]

    print(f"{'Load time (s)':<25} {r25['load_time_s']:>18.1f} {r3['load_time_s']:>18.1f}")
    print(f"{'Inference time (s)':<25} {r25['inference_time_s']:>18.1f} {r3['inference_time_s']:>18.1f}")
    print(f"{'Output chars':<25} {r25['output_length']:>18} {r3['output_length']:>18}")
    print(f"{'Output lines':<25} {r25['output_lines']:>18} {r3['output_lines']:>18}")

    # Check text overlap
    words_25 = set(r25["output_text"].split())
    words_3 = set(r3["output_text"].split())
    if words_25 and words_3:
        overlap = len(words_25 & words_3) / max(len(words_25 | words_3), 1) * 100
        print(f"{'Word overlap %':<25} {overlap:>17.1f}%")

    print(f"\nFull comparison saved to: {json_path}")


if __name__ == "__main__":
    main()
