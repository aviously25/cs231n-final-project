#!/usr/bin/env python3
"""
Phase 2: Generate aesthetic visual descriptions for each MusicCaps clip via GPT-4o-mini.

Usage:
    OPENAI_API_KEY=sk-... python scripts/02_generate_prompts.py [--limit N]

Checkpointing: skips rows where outputs/prompts/{ytid}.json already exists.
Cost is tracked per-call in outputs/costs.jsonl.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import append_cost, append_error, load_error_ytids, load_filtered_df, total_cost_so_far

try:
    from openai import OpenAI
except ImportError:
    print("openai package not found. Install with: pip install openai")
    sys.exit(1)

PROMPTS_DIR = Path("outputs/prompts")
CSV_PATH = Path("datasets/musiccaps-public.csv")
MODEL = "gpt-5.5"

_SYSTEM_PROMPT = (
    "You are a creative visual artist. Given a music description, respond with a single "
    "image generation prompt that captures the aesthetic vibe and mood of the track. "
    "Focus on visual atmosphere, color palette, lighting, and scene — not instruments or music theory. "
    "Respond with the prompt text only, no preamble."
)


def build_user_message(caption: str, aspect_list: list[str]) -> str:
    tags_str = str(aspect_list)
    return (
        f"{caption}\n\n"
        f"{tags_str}\n\n"
        "given these descriptions, if you were to generate an image that encapsulates "
        "the vibe of this track, what prompt would you use?"
    )


def call_with_backoff(client: OpenAI, messages: list[dict], max_retries: int = 5) -> object:
    delay = 2
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=200,
                temperature=0.8,
            )
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()
            if is_rate_limit and attempt < max_retries - 1:
                time.sleep(min(delay, 60))
                delay *= 2
                continue
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows (for smoke testing)")
    args = parser.parse_args()

    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    client = OpenAI()  # reads OPENAI_API_KEY from env

    df = load_filtered_df(CSV_PATH)
    if args.limit:
        df = df.head(args.limit)

    # skip rows that failed audio download
    audio_errors = load_error_ytids(phase="audio")

    completed = skipped = failed = 0
    session_cost = 0.0

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Generating prompts")):
        ytid = row["ytid"]
        out_path = PROMPTS_DIR / f"{ytid}.json"

        if out_path.exists():
            skipped += 1
            continue

        if ytid in audio_errors:
            skipped += 1
            continue

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(row["caption"], row["aspect_list_parsed"])},
        ]

        try:
            response = call_with_backoff(client, messages)
            visual_prompt = response.choices[0].message.content.strip()
            usage = response.usage

            cost = append_cost(ytid, MODEL, usage.prompt_tokens, usage.completion_tokens)
            session_cost += cost

            payload = {
                "ytid": ytid,
                "caption": row["caption"],
                "aspect_list": row["aspect_list_parsed"],
                "visual_prompt": visual_prompt,
            }
            out_path.write_text(json.dumps(payload, indent=2))
            completed += 1

        except Exception as e:
            append_error(ytid, "prompt", str(e))
            failed += 1

        if (i + 1) % 100 == 0:
            running_total = total_cost_so_far()
            tqdm.write(f"  [{i+1} rows] session cost: ${session_cost:.4f}  total ever: ${running_total:.4f}")

    print(f"\nDone. completed={completed}  skipped={skipped}  failed={failed}")
    print(f"Session cost: ${session_cost:.4f}  |  All-time total: ${total_cost_so_far():.4f}")


if __name__ == "__main__":
    main()
