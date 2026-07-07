"""Shared paths and builders for the heptokens validation benchmark.

Dataset: /home/abhijith/train.h5 — 620k jets, 5 balanced classes, up to 150
constituents with (pt [GeV], deta, dphi), zero-padded, float16.

Layout (all heavy outputs under results/ which is gitignored):
    results/benchmarks/data/        structured h5 + fitted scalers
    results/benchmarks/tokenizers/  checkpoints + CSV logs per tokenizer
    results/benchmarks/tokens/      exported token datasets (the "release")
    results/benchmarks/probes/      probe logs, metrics, predictions
    benchmarks/figures/             committed validation plots
"""

from functools import partial
from pathlib import Path

import joblib

REPO = Path(__file__).resolve().parent.parent
RAW_H5 = Path("/home/abhijith/train.h5")

OUT = REPO / "results" / "benchmarks"
DATA_DIR = OUT / "data"
TOKENIZER_DIR = OUT / "tokenizers"
TOKENS_DIR = OUT / "tokens"
PROBES_DIR = OUT / "probes"
FIGURES_DIR = REPO / "benchmarks" / "figures"

STRUCT_H5 = DATA_DIR / "jets5c.h5"
CST_SCALER = DATA_DIR / "cst_scaler.joblib"
JET_SCALER = DATA_DIR / "jet_scaler.joblib"

N_CLASSES = 5
N_FEATURES = 3  # (pt, deta, dphi)
NUM_ELEMENTS = 150
LATENT_DIM = 16
SEED = 42

DATASET_KW = dict(
    file_path=str(STRUCT_H5),
    obj_group="jets",
    set_group="csts",
    obj_features=["pt", "nconst"],
    set_features=["pt", "deta", "dphi"],
    label_key="label",
    mask_key="valid",
)


def make_transforms() -> dict:
    """Preprocessing transform dict for direct (non-Hydra) module construction."""
    from heptokens.data.collation import preprocess_batch

    return {
        "preprocess": partial(
            preprocess_batch,
            cst_fn=joblib.load(CST_SCALER),
            jet_fn=joblib.load(JET_SCALER),
        )
    }


def preprocess_cfg() -> dict:
    """Same preprocessing as a Hydra-instantiable config (for export entrypoints)."""
    return {
        "preprocess": {
            "_target_": "heptokens.data.collation.preprocess_batch",
            "_partial_": True,
            "cst_fn": {"_target_": "joblib.load", "filename": str(CST_SCALER)},
            "jet_fn": {"_target_": "joblib.load", "filename": str(JET_SCALER)},
        }
    }


def datamodule_cfg(train_frac, val_frac, test_frac, batch_size, num_workers=8) -> dict:
    """Hydra-instantiable datamodule config (for export entrypoints)."""
    return {
        "_target_": "heptokens.data.structured_array.StructuredArrayModule",
        "dataset": {
            "_target_": "heptokens.data.structured_array.StructuredArrayHDF5Dataset",
            "_partial_": True,
            **DATASET_KW,
        },
        "data_path": str(STRUCT_H5),
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "transforms": preprocess_cfg(),
        "seed": SEED,
    }


def make_datamodule(batch_size=1024, num_workers=8, num_samples=None, transforms="preprocess"):
    """Structured datamodule for direct construction (training scripts)."""
    from heptokens.data.structured_array import (
        StructuredArrayHDF5Dataset,
        StructuredArrayModule,
    )

    kw = dict(DATASET_KW)
    if num_samples is not None:
        kw["num_samples"] = num_samples
    return StructuredArrayModule(
        dataset=partial(StructuredArrayHDF5Dataset, **kw),
        train_frac=0.7,
        val_frac=0.15,
        test_frac=0.15,
        seed=SEED,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        transforms=make_transforms() if transforms == "preprocess" else transforms,
    )


def transformer_partial(d_model=128, n_heads=8, num_layers=4, dim_feedforward=512):
    from heptokens.models.transformer import Transformer

    return partial(
        Transformer,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=0.0,
        activation="gelu",
    )


PROBE_KW = dict(
    n_classes=N_CLASSES,
    d_model=64,
    n_heads=4,
    num_layers=2,
    dim_feedforward=128,
    dropout=0.0,
    pooling="mean",
    learning_rate=1e-3,
)
