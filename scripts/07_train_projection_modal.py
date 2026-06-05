#!/usr/bin/env python3
"""
Phase 7: Train AudioProjection MLP on Modal (A10G GPU).

The projection network maps CLAP audio embeddings [512] → SD-Turbo conditioning [77, 1024],
enabling inference without an LLM: audio → CLAP → projection → SD-Turbo → image.

Usage:
    modal run scripts/07_train_projection_modal.py
    modal run scripts/07_train_projection_modal.py --epochs 200 --batch_size 128

Outputs:
    outputs/checkpoints/projection_best.pt  (downloaded from Modal Volume after training)
"""

import sys
from pathlib import Path

import modal

# ── Modal image ────────────────────────────────────────────────────────────────

def _install_deps():
    pass  # all deps installed via pip_install below


_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy",
        "tqdm",
    )
)

_vol = modal.Volume.from_name("cs231n-projection-embeddings", create_if_missing=True)
_ckpt_vol = modal.Volume.from_name("cs231n-projection-checkpoints", create_if_missing=True)
_EMBED_DIR = "/embeddings"
_CKPT_DIR = "/checkpoints"

app = modal.App("cs231n-projection-train")

# ── Model definition (shared local + remote) ───────────────────────────────────


class AudioProjection:
    """Defined as plain class; the actual nn.Module is constructed inside Modal."""
    pass


# ── Remote training class ──────────────────────────────────────────────────────


@app.cls(
    gpu="A10G",
    image=_image,
    volumes={_EMBED_DIR: _vol, _CKPT_DIR: _ckpt_vol},
    timeout=3600,
)
class ProjectionTrainer:
    @modal.enter()
    def load(self):
        import numpy as np
        import torch
        import torch.nn as nn
        from pathlib import Path

        class _AudioProjection(nn.Module):
            def __init__(self, in_dim=512, hidden_dim=1024, seq_len=77, out_dim=1024, dropout=0.1):
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.LayerNorm(in_dim),
                    nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(hidden_dim, out_dim),
                )
                self.pos_offsets = nn.Parameter(torch.zeros(seq_len, out_dim))

            def forward(self, x):
                h = self.mlp(x)
                h = h.unsqueeze(1).expand(-1, 77, -1)
                return h + self.pos_offsets

        self.device = "cuda"
        self.model_cls = _AudioProjection

        audio_dir = Path(_EMBED_DIR) / "audio"
        text_dir = Path(_EMBED_DIR) / "text"

        ytids = sorted(
            p.stem for p in audio_dir.glob("*.npy")
            if (text_dir / f"{p.stem}.npy").exists()
        )
        print(f"Found {len(ytids)} paired embeddings")

        audio_embs, text_embs = [], []
        for ytid in ytids:
            audio_embs.append(np.load(audio_dir / f"{ytid}.npy").astype(np.float32))
            text_embs.append(np.load(text_dir / f"{ytid}.npy").astype(np.float32))

        self.audio = torch.tensor(np.stack(audio_embs))
        self.text = torch.tensor(np.stack(text_embs))
        self.ytids = ytids
        print(f"audio shape: {self.audio.shape}  text shape: {self.text.shape}")

    @modal.method()
    def train(self, epochs: int = 200, batch_size: int = 128, lr: float = 3e-4, val_frac: float = 0.1):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        n = len(self.audio)
        n_val = max(1, int(n * val_frac))
        n_train = n - n_val

        indices = torch.randperm(n)
        train_idx, val_idx = indices[:n_train], indices[n_train:]

        train_ds = TensorDataset(self.audio[train_idx], self.text[train_idx])
        val_ds = TensorDataset(self.audio[val_idx], self.text[val_idx])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = self.model_cls().to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()

        best_val = float("inf")
        history = []

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for xa, xt in train_loader:
                xa, xt = xa.to(self.device), xt.to(self.device)
                pred = model(xa)
                loss = criterion(pred, xt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xa)
            train_loss /= n_train

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xa, xt in val_loader:
                    xa, xt = xa.to(self.device), xt.to(self.device)
                    val_loss += criterion(model(xa), xt).item() * len(xa)
            val_loss /= n_val
            scheduler.step()

            history.append({"epoch": epoch, "train": train_loss, "val": val_loss})

            if epoch % 10 == 0 or epoch == 1:
                print(f"  epoch {epoch:4d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}")

            if val_loss < best_val:
                best_val = val_loss
                ckpt = {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "model_state_dict": model.state_dict(),
                    "config": {"in_dim": 512, "hidden_dim": 1024, "seq_len": 77, "out_dim": 1024},
                }
                torch.save(ckpt, f"{_CKPT_DIR}/projection_best.pt")
                _ckpt_vol.commit()

        print(f"Training complete. best_val_loss={best_val:.6f}")
        return history


# ── Local entrypoint ───────────────────────────────────────────────────────────

EMB_AUDIO_DIR = Path("outputs/embeddings/audio")
EMB_TEXT_DIR = Path("outputs/embeddings/text")
CKPT_DIR = Path("outputs/checkpoints")


@app.local_entrypoint()
def main(epochs: int = 200, batch_size: int = 128, lr: float = 3e-4):
    from tqdm import tqdm

    # Count available pairs
    audio_files = {p.stem for p in EMB_AUDIO_DIR.glob("*.npy")} if EMB_AUDIO_DIR.exists() else set()
    text_files = {p.stem for p in EMB_TEXT_DIR.glob("*.npy")} if EMB_TEXT_DIR.exists() else set()
    pairs = sorted(audio_files & text_files)
    print(f"Found {len(pairs)} paired embeddings locally — uploading to Modal Volume")

    if not pairs:
        print("No paired embeddings found. Run scripts/06_precompute_embeddings.py first.")
        return

    # Upload audio embeddings
    print("Uploading audio embeddings...")
    with _vol.batch_upload(force=True) as batch:
        for ytid in tqdm(pairs, desc="audio", unit="file"):
            batch.put_file(str(EMB_AUDIO_DIR / f"{ytid}.npy"), f"/audio/{ytid}.npy")

    # Upload text embeddings
    print("Uploading text embeddings...")
    with _vol.batch_upload(force=True) as batch:
        for ytid in tqdm(pairs, desc="text", unit="file"):
            batch.put_file(str(EMB_TEXT_DIR / f"{ytid}.npy"), f"/text/{ytid}.npy")

    print(f"Uploaded {len(pairs)} pairs. Starting training (epochs={epochs}, batch_size={batch_size})...")

    trainer = ProjectionTrainer()
    history = trainer.train.remote(epochs=epochs, batch_size=batch_size, lr=lr)

    print(f"Training finished. Downloading checkpoint...")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    data = b"".join(_ckpt_vol.read_file("/projection_best.pt"))
    (CKPT_DIR / "projection_best.pt").write_bytes(data)
    print(f"Checkpoint saved to {CKPT_DIR}/projection_best.pt")

    # Print final few epochs of loss curve
    if history:
        print("\nFinal training history (last 5 epochs):")
        for entry in history[-5:]:
            print(f"  epoch {entry['epoch']:4d}  train={entry['train']:.6f}  val={entry['val']:.6f}")
