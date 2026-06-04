#!/usr/bin/env python3
"""
Phase 3 (API variant): Generate images via Stability AI REST API from visual prompts.

Usage:
    python scripts/04_generate_images_api.py [--limit N] [--model core|ultra|sd3-large]

Requires STABILITY_API_KEY in .env. Get a key at https://platform.stability.ai/account/keys

Checkpointing: skips rows where outputs/images_api/{ytid}.png already exists.
Costs are tracked in outputs/costs.jsonl alongside Phase 2 costs.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import append_error, append_flat_cost, load_error_ytids, load_filtered_df

try:
    import requests
except ImportError:
    print("requests package not found. Install with: pip install requests")
    sys.exit(1)

IMAGES_DIR = Path("outputs/images_api")
PROMPTS_DIR = Path("outputs/prompts")
CSV_PATH = Path("datasets/musiccaps-public.csv")

_API_BASE = "https://api.stability.ai/v2beta/stable-image/generate"

# USD per image — check https://platform.stability.ai/pricing for current rates
_MODEL_COSTS = {
    "core": 0.003,
    "ultra": 0.008,
    "sd3-large": 0.065,
}

_YTID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _safe_ytid(ytid: str) -> str:
    if not _YTID_RE.match(ytid):
        raise ValueError(f"Invalid ytid: {ytid!r}")
    return ytid


def generate_image_api(
    prompt: str, model: str, api_key: str, max_retries: int = 5
) -> bytes:
    endpoint = f"{_API_BASE}/{model}"
    data = {"prompt": prompt, "output_format": "png"}
    if model == "sd3-large":
        endpoint = f"{_API_BASE}/sd3"
        data["model"] = "sd3-large"

    delay = 2
    for attempt in range(max_retries):
        response = requests.post(
            endpoint,
            headers={"authorization": f"Bearer {api_key}", "accept": "image/*"},
            files={"none": ""},
            data=data,
            timeout=60,
        )
        if response.status_code == 200:
            return response.content
        if response.status_code == 429 and attempt < max_retries - 1:
            time.sleep(min(delay, 60))
            delay *= 2
            continue
        raise RuntimeError(
            f"Stability API error {response.status_code}: {response.text[:200]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--model",
        default="core",
        choices=list(_MODEL_COSTS),
        help="Stability AI model to use (default: core)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("STABILITY_API_KEY", "").strip()
    if not api_key:
        print("Error: STABILITY_API_KEY not set in .env")
        sys.exit(1)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_filtered_df(CSV_PATH)
    if args.limit:
        df = df.head(args.limit)

    prior_errors = load_error_ytids()

    completed = skipped = failed = 0
    session_cost = 0.0
    cost_per_image = _MODEL_COSTS[args.model]

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Generating images ({args.model})"):
        ytid = row["ytid"]
        out_path = IMAGES_DIR / f"{ytid}.png"
        prompt_path = PROMPTS_DIR / f"{ytid}.json"

        if out_path.exists():
            skipped += 1
            continue

        if ytid in prior_errors or not prompt_path.exists():
            skipped += 1
            continue

        try:
            safe_ytid = _safe_ytid(ytid)
            visual_prompt = json.loads(prompt_path.read_text())["visual_prompt"]
            image_bytes = generate_image_api(visual_prompt, args.model, api_key)

            out_path = IMAGES_DIR / f"{safe_ytid}.png"
            out_path.write_bytes(image_bytes)

            append_flat_cost(safe_ytid, f"stability-{args.model}", cost_per_image)
            session_cost += cost_per_image
            completed += 1

        except Exception as e:
            append_error(ytid, "image_api", str(e))
            failed += 1

    print(f"\nDone. completed={completed}  skipped={skipped}  failed={failed}")
    print(f"Estimated session cost: ${session_cost:.4f} @ ${cost_per_image}/image ({args.model})")


if __name__ == "__main__":
    main()
