#!/usr/bin/env python3
"""
Phase 2: Generate aesthetic visual descriptions for each MusicCaps clip via OpenAI.

Usage:
    python scripts/02_generate_prompts.py [--limit N] [--concurrency 50]

Checkpointing: skips rows where outputs/prompts/{ytid}.json already exists.
Cost is tracked per-call in outputs/costs.jsonl.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from tqdm.asyncio import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import append_cost, append_error, load_error_ytids, load_filtered_df, total_cost_so_far

try:
    from openai import AsyncOpenAI
except ImportError:
    print("openai package not found. Install with: pip install openai")
    sys.exit(1)

PROMPTS_DIR = Path("outputs/prompts")
CSV_PATH = Path("datasets/musiccaps-public.csv")
MODEL = "gpt-5.4-mini"

_SYSTEM_PROMPT = (
    "You are a creative visual artist. Given a music description, respond with a single "
    "image generation prompt that captures the aesthetic vibe and mood of the track. "
    "Focus on visual atmosphere, color palette, lighting, and scene — not instruments or music theory. "
    "Respond with the prompt text only, no preamble."
)


def build_user_message(caption: str, aspect_list: list[str]) -> str:
    return (
        f"{caption}\n\n"
        f"{aspect_list}\n\n"
        "given these descriptions, if you were to generate an image that encapsulates "
        "the vibe of this track, what prompt would you use?"
    )


async def process_row(client: AsyncOpenAI, sem: asyncio.Semaphore, row, counters: dict) -> None:
    ytid = row["ytid"]
    out_path = PROMPTS_DIR / f"{ytid}.json"

    if out_path.exists():
        counters["skipped"] += 1
        return

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(row["caption"], row["aspect_list_parsed"])},
    ]

    delay = 2
    async with sem:
        for attempt in range(5):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    max_completion_tokens=200,
                    temperature=0.8,
                )
                break
            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()
                if is_rate_limit and attempt < 4:
                    await asyncio.sleep(min(delay, 60))
                    delay *= 2
                    continue
                append_error(ytid, "prompt", err_str)
                counters["failed"] += 1
                return

    try:
        visual_prompt = response.choices[0].message.content.strip()
        usage = response.usage
        cost = append_cost(ytid, MODEL, usage.prompt_tokens, usage.completion_tokens)
        counters["session_cost"] += cost

        payload = {
            "ytid": ytid,
            "caption": row["caption"],
            "aspect_list": row["aspect_list_parsed"],
            "visual_prompt": visual_prompt,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        counters["completed"] += 1
    except Exception as e:
        append_error(ytid, "prompt", str(e))
        counters["failed"] += 1


async def run(limit: int | None, concurrency: int) -> None:
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_filtered_df(CSV_PATH)
    if limit:
        df = df.head(limit)

    audio_errors = load_error_ytids(phase="audio")
    rows = [row for _, row in df.iterrows() if row["ytid"] not in audio_errors]

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)
    counters = {"completed": 0, "skipped": 0, "failed": 0, "session_cost": 0.0}

    tasks = [process_row(client, sem, row, counters) for row in rows]
    await tqdm.gather(*tasks, desc=f"Generating prompts (concurrency={concurrency})")

    print(f"\nDone. completed={counters['completed']}  skipped={counters['skipped']}  failed={counters['failed']}")
    print(f"Session cost: ${counters['session_cost']:.4f}  |  All-time total: ${total_cost_so_far():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=50, help="Max simultaneous API requests")
    args = parser.parse_args()
    asyncio.run(run(args.limit, args.concurrency))


if __name__ == "__main__":
    main()
