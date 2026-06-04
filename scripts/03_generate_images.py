#!/usr/bin/env python3
"""
Phase 3: Generate images locally from visual prompts produced in Phase 2.

Usage:
    python scripts/03_generate_images.py [--limit N] [--device mps|cuda|cpu] [--model sd-turbo|sdxl-turbo]

Output folder is named after the model: outputs/images-{model}/
Checkpointing: skips rows where the output PNG already exists.
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import append_error, load_error_ytids, load_filtered_df

try:
    import torch
    from diffusers import AutoPipelineForText2Image
    from PIL import Image
except ImportError:
    print("Missing dependencies. Install with: pip install diffusers torch Pillow")
    sys.exit(1)

# Whitelist: slug → HuggingFace repo ID. Output dir is derived from the slug key,
# not the raw CLI arg, so path traversal via --model is not possible.
_MODELS = {
    "sd-turbo": "stabilityai/sd-turbo",
    "sdxl-turbo": "stabilityai/sdxl-turbo",
}

PROMPTS_DIR = Path("outputs/prompts")
CSV_PATH = Path("datasets/musiccaps-public.csv")


def load_pipeline(model_id: str, device: str):
    use_fp16 = device == "cuda"
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
        variant="fp16" if use_fp16 else None,
    )
    pipe = pipe.to(device)
    if device == "mps":
        pipe.enable_attention_slicing()
    return pipe


def generate_image(pipe, visual_prompt: str, device: str) -> Image.Image:
    generator = torch.Generator(device=device)
    return pipe(
        prompt=visual_prompt,
        num_inference_steps=4,
        guidance_scale=0.0,
        generator=generator,
        width=512,
        height=512,
    ).images[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows (for smoke testing)")
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--model", default="sd-turbo", choices=list(_MODELS))
    args = parser.parse_args()

    model_id = _MODELS[args.model]
    images_dir = Path(f"outputs/images-{args.model}")
    images_dir.mkdir(parents=True, exist_ok=True)

    df = load_filtered_df(CSV_PATH)
    if args.limit:
        df = df.head(args.limit)

    prior_errors = load_error_ytids()

    print(f"Loading {args.model} on {args.device}...")
    pipe = load_pipeline(model_id, args.device)
    print("Model loaded.")

    completed = skipped = failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Generating images ({args.model})"):
        ytid = row["ytid"]
        out_path = images_dir / f"{ytid}.png"
        prompt_path = PROMPTS_DIR / f"{ytid}.json"

        if out_path.exists():
            skipped += 1
            continue

        if ytid in prior_errors or not prompt_path.exists():
            skipped += 1
            continue

        try:
            visual_prompt = json.loads(prompt_path.read_text())["visual_prompt"]
            image = generate_image(pipe, visual_prompt, args.device)
            image.save(out_path)
            completed += 1
        except Exception as e:
            append_error(ytid, "image", str(e))
            failed += 1

    print(f"\nDone. completed={completed}  skipped={skipped}  failed={failed}")


if __name__ == "__main__":
    main()
