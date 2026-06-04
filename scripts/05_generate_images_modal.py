#!/usr/bin/env python3
"""
Phase 3 (Modal variant): Generate images via SDXL-Turbo on a remote A10G GPU.

Images are written to a Modal Volume during generation — safe if your laptop
disconnects mid-run. Run again after reconnecting to download what finished.

Usage:
    modal run scripts/05_generate_images_modal.py           # full dataset
    modal run scripts/05_generate_images_modal.py --limit 10  # smoke test

Requires: pip install modal  &&  modal setup (authenticate once)

Output: outputs/images-sdxl-turbo/{ytid}.png  (downloaded from Modal Volume)
"""

import io
import json
import re
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parent))

# ── Modal image & volume ───────────────────────────────────────────────────────

def _download_model():
    from diffusers import AutoPipelineForText2Image
    import torch

    AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
    )


_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "diffusers",
        "transformers",
        "accelerate",
        "torch",
        "Pillow",
    )
    .run_function(_download_model)  # weights baked into image layer
)

_vol = modal.Volume.from_name("cs231n-images-sdxl-turbo", create_if_missing=True)
_VOLUME_DIR = "/outputs"

app = modal.App("cs231n-image-gen")

# ── Remote class ───────────────────────────────────────────────────────────────

_YTID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@app.cls(gpu="A10G", image=_image, volumes={_VOLUME_DIR: _vol})
class ImageGenerator:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import AutoPipelineForText2Image

        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")

    @modal.method()
    def generate(self, ytid: str, prompt: str) -> str:
        import torch

        if not _YTID_RE.match(ytid):
            raise ValueError(f"Invalid ytid: {ytid!r}")

        out_path = Path(_VOLUME_DIR) / f"{ytid}.png"
        if out_path.exists():
            return ytid  # already done (e.g. resumed run)

        generator = torch.Generator(device="cuda")
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=4,
            guidance_scale=0.0,
            generator=generator,
            width=512,
            height=512,
        ).images[0]

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        out_path.write_bytes(buf.getvalue())
        _vol.commit()
        return ytid


# ── Local entrypoint ───────────────────────────────────────────────────────────

IMAGES_DIR = Path("outputs/images-sdxl-turbo")
PROMPTS_DIR = Path("outputs/prompts")
CSV_PATH = Path("datasets/musiccaps-public.csv")


@app.local_entrypoint()
def main(limit: int = 0):
    from tqdm import tqdm
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import append_error, load_error_ytids, load_filtered_df

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_filtered_df(CSV_PATH)
    if limit:
        df = df.head(limit)

    prior_errors = load_error_ytids()

    # Check volume for already-generated files (handles resumed runs)
    done_on_volume = {
        entry.path.lstrip("/")
        for entry in _vol.listdir("/", recursive=False)
        if entry.path.endswith(".png")
    }

    todo = []
    for _, row in df.iterrows():
        ytid = row["ytid"]
        if not _YTID_RE.match(ytid):
            continue
        if (IMAGES_DIR / f"{ytid}.png").exists():
            continue
        if ytid in prior_errors:
            continue
        if not (PROMPTS_DIR / f"{ytid}.json").exists():
            continue
        if f"{ytid}.png" in done_on_volume:
            continue  # generated but not yet downloaded — will grab below
        prompt = json.loads((PROMPTS_DIR / f"{ytid}.json").read_text())["visual_prompt"]
        todo.append((ytid, prompt))

    if todo:
        print(f"Generating {len(todo)} images on Modal (A10G)...")
        generator = ImageGenerator()
        ytids = [t[0] for t in todo]
        prompts = [t[1] for t in todo]

        completed = failed = 0
        t0 = time.monotonic()
        results = generator.generate.starmap(zip(ytids, prompts), return_exceptions=True)
        with tqdm(zip(ytids, results), total=len(todo), desc="Generating (A10G)", unit="img") as bar:
            for ytid, result in bar:
                if isinstance(result, Exception):
                    append_error(ytid, "image_modal", str(result))
                    failed += 1
                else:
                    completed += 1
                elapsed = time.monotonic() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                bar.set_postfix(ok=completed, err=failed, img_s=f"{rate:.1f}")

        elapsed = time.monotonic() - t0
        print(f"\nGeneration done. completed={completed}  failed={failed}  "
              f"total={elapsed:.0f}s  avg={elapsed/max(completed,1):.1f}s/img")
    else:
        print("Nothing new to generate.")

    # Download everything from the volume not yet on local disk
    to_download = [
        entry for entry in _vol.listdir("/", recursive=False)
        if entry.path.endswith(".png")
        and not (IMAGES_DIR / Path(entry.path).name).exists()
    ]

    if to_download:
        print(f"Downloading {len(to_download)} images from volume...")
        for entry in tqdm(to_download, desc="Downloading", unit="img"):
            name = Path(entry.path).name
            data = b"".join(_vol.read_file(entry.path))
            (IMAGES_DIR / name).write_bytes(data)
        print("Download complete.")
    else:
        print("All volume images already on disk.")
