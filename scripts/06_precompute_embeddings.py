#!/usr/bin/env python3
"""
Phase 6: Precompute CLAP audio embeddings and CLIP text embeddings for all MusicCaps clips.

Phase A — CLAP audio: saves outputs/embeddings/audio/{ytid}.npy  shape [512] float32
Phase B — CLIP text:  saves outputs/embeddings/text/{ytid}.npy   shape [77, 1024] float16

Both phases are checkpointed: files are skipped if they already exist.
CLAP is deleted from memory before CLIP loads to stay within typical laptop RAM.

Usage:
    python scripts/06_precompute_embeddings.py
    python scripts/06_precompute_embeddings.py --limit 20   # smoke test
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import append_error, load_error_ytids, load_filtered_df

CSV_PATH = Path("datasets/musiccaps-public.csv")
AUDIO_DIR = Path("outputs/audio")
PROMPTS_DIR = Path("outputs/prompts")
EMB_AUDIO_DIR = Path("outputs/embeddings/audio")
EMB_TEXT_DIR = Path("outputs/embeddings/text")

CLAP_MODEL_ID = "laion/clap-htsat-fused"
SD_TURBO_ID = "stabilityai/sd-turbo"
AUDIO_BATCH = 16


def phase_a_audio(df, skip_ytids: set[str]) -> None:
    from transformers import ClapModel, ClapProcessor

    print("Phase A — CLAP audio embeddings")
    EMB_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    model = ClapModel.from_pretrained(CLAP_MODEL_ID)
    processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
    model.eval()

    import librosa

    rows = [
        row for _, row in df.iterrows()
        if row["ytid"] not in skip_ytids
        and not (EMB_AUDIO_DIR / f"{row['ytid']}.npy").exists()
        and (AUDIO_DIR / f"{row['ytid']}.wav").exists()
    ]

    completed = skipped = failed = 0

    for i in range(0, len(rows), AUDIO_BATCH):
        batch_rows = rows[i : i + AUDIO_BATCH]
        wav_paths = [AUDIO_DIR / f"{row['ytid']}.wav" for row in batch_rows]

        waveforms = []
        valid_rows = []
        for row, wav_path in zip(batch_rows, wav_paths):
            try:
                wav, _ = librosa.load(str(wav_path), sr=48000, mono=True)
                waveforms.append(wav)
                valid_rows.append(row)
            except Exception as e:
                append_error(row["ytid"], "embed_audio", str(e))
                failed += 1

        if not waveforms:
            continue

        try:
            inputs = processor(audio=waveforms, sampling_rate=48000, return_tensors="pt", padding=True)
            with torch.no_grad():
                # pooler_output is [B, 512] — CLAP projection is applied inside get_audio_features
                out = model.get_audio_features(**inputs)
                embs = out.pooler_output if hasattr(out, "pooler_output") else out
            embs = embs / embs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            embs_np = embs.float().numpy()

            for row, emb in zip(valid_rows, embs_np):
                np.save(EMB_AUDIO_DIR / f"{row['ytid']}.npy", emb)
                completed += 1
        except Exception as e:
            for row in valid_rows:
                append_error(row["ytid"], "embed_audio", str(e))
                failed += 1

        if (i // AUDIO_BATCH + 1) % 20 == 0:
            tqdm.write(f"  [{completed} saved, {failed} failed]")

    already_done = len(df) - len(rows)
    print(f"Phase A done. saved={completed}  already_existed={already_done}  failed={failed}")

    del model, processor
    gc.collect()


def phase_b_text(df, skip_ytids: set[str]) -> None:
    from transformers import CLIPTextModel, CLIPTokenizer

    print("Phase B — CLIP text embeddings")
    EMB_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained(SD_TURBO_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(SD_TURBO_ID, subfolder="text_encoder")
    text_encoder.eval()

    rows = [
        row for _, row in df.iterrows()
        if row["ytid"] not in skip_ytids
        and not (EMB_TEXT_DIR / f"{row['ytid']}.npy").exists()
        and (PROMPTS_DIR / f"{row['ytid']}.json").exists()
    ]

    completed = skipped = failed = 0

    for row in tqdm(rows, desc="CLIP text", unit="clip"):
        ytid = row["ytid"]
        try:
            payload = json.loads((PROMPTS_DIR / f"{ytid}.json").read_text())
            prompt = payload["visual_prompt"]

            tokens = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                out = text_encoder(**tokens)
            emb = out.last_hidden_state.squeeze(0)  # [77, 1024]
            np.save(EMB_TEXT_DIR / f"{ytid}.npy", emb.half().numpy())
            completed += 1
        except Exception as e:
            append_error(ytid, "embed_text", str(e))
            failed += 1

    already_done = len(df) - len(rows)
    print(f"Phase B done. saved={completed}  already_existed={already_done}  failed={failed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows")
    parser.add_argument("--phase", choices=["a", "b", "both"], default="both")
    args = parser.parse_args()

    df = load_filtered_df(CSV_PATH)
    if args.limit:
        df = df.head(args.limit)

    prior_errors = load_error_ytids()

    if args.phase in ("a", "both"):
        phase_a_audio(df, prior_errors)
    if args.phase in ("b", "both"):
        phase_b_text(df, prior_errors)


if __name__ == "__main__":
    main()
