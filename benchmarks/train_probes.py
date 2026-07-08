"""Consumer-view downstream comparison through identical probes.

Simulates the AI-ready-dataset consumer: token arms train probes **only on the
exported token files** (the release artifact) — no PF/constituent access:

    0. null_features       FeatureClassifier on preprocessed constituents
                           (full-information reference ceiling)
    1. contrastive_tokens  VectorClassifier on exported continuous latents
    B. vqvae_tokens        VectorClassifier on z_q reconstructed from the
                           exported (indices, codebooks) release

All arms share the probe architecture, data budget, split, and trainer budget.

Usage:
    python train_probes.py [--epochs 15] [--n-jets 250000] [--arms null contrastive vqvae]
"""

import argparse
import json

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from common import (
    LATENT_DIM,
    PROBE_KW,
    PROBES_DIR,
    SEED,
    STRUCT_H5,
    TOKENIZER_DIR,
    TOKENS_DIR,
    datamodule_cfg,
    make_datamodule,
)

from heptokens.export_latents import export as export_latents
from heptokens.export_tokens import export as export_tokens
from heptokens.models.token_classifier import FeatureClassifier, VectorClassifier


def run_exports() -> dict:
    """Export both token releases over the full dataset (idempotent)."""
    paths = {}
    for name, export_fn in [
        ("contrastive", export_latents),
        ("library", export_latents),
        ("vqvae", export_tokens),
    ]:
        out_dir = TOKENS_DIR / name
        out_path = out_dir / f"{STRUCT_H5.stem}.npz"
        paths[name] = out_path
        if out_path.exists():
            print(f"[export] {out_path} exists, skipping")
            continue
        ckpt = TOKENIZER_DIR / name / "checkpoints" / "best.ckpt"
        if not ckpt.exists():
            ckpt = TOKENIZER_DIR / name / "checkpoints" / "last.ckpt"
        if not ckpt.exists():
            print(f"[export] no checkpoint for {name}, skipping")
            paths.pop(name)
            continue
        cfg = OmegaConf.create(
            {
                "ckpt_path": str(ckpt),
                "output_dir": str(out_dir),
                "datamodule": datamodule_cfg(0.0, 0.0, 1.0, batch_size=2048),
                "trainer": {
                    "_target_": "lightning.Trainer",
                    "accelerator": "auto",
                    "devices": 1,
                    "enable_progress_bar": True,
                    "logger": False,
                },
            }
        )
        print(f"[export] {name} <- {ckpt}")
        export_fn(cfg)
    return paths


class LatentTokenDataset(Dataset):
    """Consumer view of the continuous-token release (latents npz)."""

    def __init__(self, npz_path, start, stop):
        data = np.load(npz_path)
        self.z = data["latents"][start:stop]  # [n, 150, D] float32
        self.labels = data["labels"][start:stop].astype(np.int64)
        self.mask = np.any(self.z != 0.0, axis=-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "z_q": torch.from_numpy(self.z[i]),
            "mask": torch.from_numpy(self.mask[i]),
            "labels": torch.tensor(self.labels[i]),
        }


class VqTokenDataset(Dataset):
    """Consumer view of the discrete-token release (indices + codebooks npz).

    z_q[j] = sum_q codebooks[q, indices[j, q]]  — the documented reconstruction.
    """

    def __init__(self, npz_path, start, stop):
        data = np.load(npz_path)
        self.indices = data["indices"][start:stop].astype(np.int64)  # [n, 150, Q]
        self.labels = data["labels"][start:stop].astype(np.int64)
        self.codebooks = data["codebooks"].astype(np.float32)  # [Q, K, D]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        idx = self.indices[i]  # [150, Q]
        mask = idx[:, 0] >= 0
        safe = np.where(idx < 0, 0, idx)
        z = np.zeros((idx.shape[0], self.codebooks.shape[-1]), dtype=np.float32)
        for q in range(self.codebooks.shape[0]):
            z += self.codebooks[q][safe[:, q]]
        z[~mask] = 0.0
        return {
            "z_q": torch.from_numpy(z),
            "mask": torch.from_numpy(mask),
            "labels": torch.tensor(self.labels[i]),
        }


def probe_loaders(dataset_cls, npz_path, n_jets, batch_size, workers):
    n_train, n_val = int(0.7 * n_jets), int(0.15 * n_jets)
    mk = lambda a, b, sh: DataLoader(  # noqa: E731
        dataset_cls(npz_path, a, b),
        batch_size=batch_size,
        shuffle=sh,
        num_workers=workers,
        drop_last=sh,
    )
    return (
        mk(0, n_train, True),
        mk(n_train, n_train + n_val, False),
    )


def train_probe(arm, model, train_loader, val_loader, epochs):
    run_dir = PROBES_DIR / arm
    run_dir.mkdir(parents=True, exist_ok=True)
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        callbacks=[
            ModelCheckpoint(
                dirpath=run_dir / "checkpoints",
                filename="best",
                monitor="valid/total_loss",
                mode="min",
                save_top_k=1,
                enable_version_counter=False,
                auto_insert_metric_name=False,
            ),
            EarlyStopping(monitor="valid/total_loss", patience=4, mode="min"),
        ],
        logger=CSVLogger(save_dir=run_dir, name="csv"),
        log_every_n_steps=20,
        default_root_dir=run_dir,
    )
    trainer.fit(model, train_loader, val_loader)

    preds = trainer.predict(model, val_loader, ckpt_path="best")
    logits = torch.cat([p["output"] for p in preds]).float()
    labels = torch.cat([p["label"] for p in preds]).squeeze(-1)
    probs = torch.softmax(logits, dim=1).numpy()
    labels = labels.numpy()

    metrics = {
        "acc": float(accuracy_score(labels, probs.argmax(1))),
        "auc_macro_ovr": float(roc_auc_score(labels, probs, multi_class="ovr", average="macro")),
        "n_val": int(len(labels)),
        "epochs_run": trainer.current_epoch,
    }
    np.savez_compressed(run_dir / "preds.npz", probs=probs, labels=labels)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[{arm}] {metrics}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-jets", type=int, default=250_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--arms", nargs="+", default=["null", "contrastive", "vqvae"],
        choices=["null", "contrastive", "library", "vqvae"],
    )
    args = parser.parse_args()

    L.seed_everything(SEED, workers=True)
    torch.set_float32_matmul_precision("medium")

    token_paths = run_exports()
    summary = {}

    if "null" in args.arms:
        dm = make_datamodule(
            batch_size=args.batch_size, num_workers=args.workers, num_samples=args.n_jets
        )
        dm.setup()
        probe = FeatureClassifier(data_sample=dm.get_data_sample(), **PROBE_KW)
        summary["null_features"] = train_probe(
            "null_features", probe, dm.train_dataloader(), dm.val_dataloader(), args.epochs
        )

    if "contrastive" in args.arms:
        tr, va = probe_loaders(
            LatentTokenDataset, token_paths["contrastive"], args.n_jets,
            args.batch_size, args.workers,
        )
        probe = VectorClassifier(codebook_dim=LATENT_DIM, **PROBE_KW)
        summary["contrastive_tokens"] = train_probe("contrastive_tokens", probe, tr, va, args.epochs)

    if "library" in args.arms and "library" in token_paths:
        tr, va = probe_loaders(
            LatentTokenDataset, token_paths["library"], args.n_jets,
            args.batch_size, args.workers,
        )
        probe = VectorClassifier(codebook_dim=LATENT_DIM, **PROBE_KW)
        summary["library_tokens"] = train_probe("library_tokens", probe, tr, va, args.epochs)

    if "vqvae" in args.arms:
        tr, va = probe_loaders(
            VqTokenDataset, token_paths["vqvae"], args.n_jets, args.batch_size, args.workers
        )
        probe = VectorClassifier(codebook_dim=LATENT_DIM, **PROBE_KW)
        summary["vqvae_tokens"] = train_probe("vqvae_tokens", probe, tr, va, args.epochs)

    (PROBES_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
