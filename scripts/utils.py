import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# load .env from project root (one level up from scripts/)
load_dotenv(Path(__file__).parent.parent / ".env")

COSTS_PATH = Path("outputs/costs.jsonl")
ERRORS_PATH = Path("outputs/errors.jsonl")

# GPT-4o-mini pricing (USD per token)
_INPUT_COST_PER_TOKEN = 0.15 / 1_000_000
_OUTPUT_COST_PER_TOKEN = 0.60 / 1_000_000


def load_filtered_df(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["aspect_list_parsed"] = df["aspect_list"].apply(_parse_aspect_list)
    mask = df["aspect_list_parsed"].apply(lambda tags: "low quality" not in tags)
    return df[mask].reset_index(drop=True)


def _parse_aspect_list(val: str) -> list[str]:
    # aspect_list column is stored as a Python list literal e.g. ['tag1', 'tag2']
    # extract all single-quoted strings using regex — avoids eval()
    if not isinstance(val, str):
        return []
    return re.findall(r"'([^']*)'", val)


def load_error_ytids(phase: str | None = None) -> set[str]:
    if not ERRORS_PATH.exists():
        return set()
    ytids = set()
    with open(ERRORS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if phase is None or entry.get("phase") == phase:
                    ytids.add(entry["ytid"])
            except json.JSONDecodeError:
                continue
    return ytids


def append_error(ytid: str, phase: str, reason: str) -> None:
    entry = {"ytid": ytid, "phase": phase, "reason": reason, "ts": _now()}
    _append_jsonl(ERRORS_PATH, entry)


def append_flat_cost(ytid: str, model: str, cost_usd: float) -> None:
    entry = {"ytid": ytid, "model": model, "cost_usd": round(cost_usd, 8), "ts": _now()}
    _append_jsonl(COSTS_PATH, entry)


def append_cost(ytid: str, model: str, input_tokens: int, output_tokens: int) -> float:
    cost = input_tokens * _INPUT_COST_PER_TOKEN + output_tokens * _OUTPUT_COST_PER_TOKEN
    entry = {
        "ytid": ytid,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 8),
        "ts": _now(),
    }
    _append_jsonl(COSTS_PATH, entry)
    return cost


def total_cost_so_far() -> float:
    if not COSTS_PATH.exists():
        return 0.0
    total = 0.0
    with open(COSTS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                total += json.loads(line).get("cost_usd", 0.0)
            except json.JSONDecodeError:
                continue
    return total


def _append_jsonl(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
