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

# Aspen Open Jets (real CMS 2016 RunG data; no truth labels).
AOJ_RAW_H5 = DATA_DIR / "aoj" / "RunG_batch0.h5"
AOJ_STRUCT_H5 = DATA_DIR / "aoj_structured.h5"
AOJ_CST_SCALER = DATA_DIR / "aoj_cst_scaler.joblib"
AOJ_JET_SCALER = DATA_DIR / "aoj_jet_scaler.joblib"

# Particle-library criterion (user spec): same PID, same charge, dpT <= 10%,
# |deta| and |dphi| <= 0.1. Feature columns for AOJ: [pt, eta, phi, pid, charge].
LIBRARY_CRITERION = dict(
    pt_index=0, eta_index=1, phi_index=2,
    dpt_frac=0.1, deta_max=0.1, dphi_max=0.1,
)
AOJ_DATASET_KW = dict(
    obj_group="jets",
    set_group="csts",
    obj_features=["pt", "nconst"],
    set_features=["pt", "eta", "phi", "pid", "charge"],
    label_key="label",
    mask_key="valid",
)

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


def make_transforms(data: str = "jets5c") -> dict:
    """Preprocessing transform dict for direct (non-Hydra) module construction.

    ``keep_raw`` runs first so physics-conditioned losses see physical units;
    other consumers simply ignore the extra ``csts_raw`` key.
    """
    from heptokens.data.collation import keep_raw_batch, preprocess_batch

    cst, jet = (CST_SCALER, JET_SCALER) if data == "jets5c" else (AOJ_CST_SCALER, AOJ_JET_SCALER)
    return {
        "keep_raw": keep_raw_batch,
        "preprocess": partial(
            preprocess_batch,
            cst_fn=joblib.load(cst),
            jet_fn=joblib.load(jet),
        ),
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


def make_datamodule(
    batch_size=1024, num_workers=8, num_samples=None, transforms="preprocess", data="jets5c"
):
    """Structured datamodule for direct construction (training scripts)."""
    from heptokens.data.structured_array import (
        StructuredArrayHDF5Dataset,
        StructuredArrayModule,
    )

    kw = dict(DATASET_KW if data == "jets5c" else AOJ_DATASET_KW)
    if data != "jets5c":
        kw["file_path"] = str(AOJ_STRUCT_H5)
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
        transforms=make_transforms(data) if transforms == "preprocess" else transforms,
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
