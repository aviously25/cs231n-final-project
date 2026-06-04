#!/usr/bin/env python3
"""
Phase 1: Download and trim 10-second audio clips from YouTube for each MusicCaps row.

Usage:
    python scripts/01_download_audio.py [--limit N]

Checkpointing: skips rows where outputs/audio/{ytid}.wav already exists or
where the ytid has a prior audio-phase error in outputs/errors.jsonl.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm

# allow running from project root
sys.path.insert(0, str(Path(__file__).parent))
from utils import append_error, load_error_ytids, load_filtered_df

AUDIO_DIR = Path("outputs/audio")
CSV_PATH = Path("datasets/musiccaps-public.csv")


def download_and_trim(ytid: str, start_s: float, end_s: float, out_path: Path) -> None:
    url = f"https://www.youtube.com/watch?v={ytid}"
    duration = end_s - start_s

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_audio = Path(tmpdir) / f"{ytid}.%(ext)s"

        yt_cmd = [
            "yt-dlp",
            "--quiet",
            "--no-warnings",
            "-x",                          # audio only
            "--audio-format", "wav",
            "--audio-quality", "0",
            "-o", str(tmp_audio),
            url,
        ]
        result = subprocess.run(yt_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()}")

        # find the downloaded file (extension may vary before conversion)
        candidates = list(Path(tmpdir).glob(f"{ytid}.*"))
        if not candidates:
            raise RuntimeError("yt-dlp produced no output file")
        tmp_file = candidates[0]

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_s),
            "-t", str(duration),
            "-i", str(tmp_file),
            "-ar", "48000",
            "-ac", "1",
            str(out_path),
        ]
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows (for smoke testing)")
    args = parser.parse_args()

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    df = load_filtered_df(CSV_PATH)
    if args.limit:
        df = df.head(args.limit)

    prior_errors = load_error_ytids(phase="audio")

    completed = skipped = failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading audio"):
        ytid = row["ytid"]
        out_path = AUDIO_DIR / f"{ytid}.wav"

        if out_path.exists():
            skipped += 1
            continue

        if ytid in prior_errors:
            skipped += 1
            continue

        try:
            download_and_trim(ytid, float(row["start_s"]), float(row["end_s"]), out_path)
            completed += 1
        except Exception as e:
            append_error(ytid, "audio", str(e))
            failed += 1

    print(f"\nDone. completed={completed}  skipped={skipped}  failed={failed}")


if __name__ == "__main__":
    main()
