"""Tests for the continuous contrastive tokenizer stack."""

from pathlib import Path

import numpy as np
import pytest
import torch as T
from omegaconf import OmegaConf

from heptokens.data.augmentations import ConstituentAugmentation, TwoViewTransform
from heptokens.export_latents import export as export_latents
from heptokens.models.losses import info_nce_loss, supcon_physics_loss, vicreg_loss
from heptokens.models.utils import (
    compute_alignment_uniformity,
    compute_embedding_collapse,
    compute_knn_pid_purity,
)
from heptokens.train import train

# ---------------------------------------------------------------------- losses


def test_vicreg_loss_finite_and_componentized():
    za, zb = T.randn(64, 8), T.randn(64, 8)
    loss, comps = vicreg_loss(za, zb)
    assert T.isfinite(loss)
    assert set(comps) == {"sim_loss", "var_loss", "cov_loss"}


def test_vicreg_detects_collapse():
    """A constant embedding must be penalised by the variance term."""
    z = T.zeros(64, 8)
    _, comps = vicreg_loss(z, z)
    assert comps["sim_loss"] == 0.0
    assert comps["var_loss"] == pytest.approx(1.0, abs=0.02)  # relu(gamma - sqrt(eps))


def test_info_nce_prefers_matched_views():
    z = T.randn(128, 8)
    matched, _ = info_nce_loss(z, z + 0.01 * T.randn_like(z))
    random, _ = info_nce_loss(z, T.randn(128, 8))
    assert matched < random


def test_supcon_physics_positive_pairs():
    """Same-PID close-kinematics constituents form positives; isolated ones don't."""
    # feats columns: [pt, eta, phi, pid]
    feats = T.tensor(
        [
            [50.0, 0.10, 0.10, 211.0],
            [51.0, 0.11, 0.11, 211.0],  # positive of row 0
            [50.0, 2.00, -2.0, 211.0],  # same pid, far away
            [50.0, 0.10, 0.10, 22.0],  # same place, different pid
        ]
    )
    za = T.randn(4, 8)
    loss, comps = supcon_physics_loss(za, feats, pid_index=3, dpt_frac=0.25, dr_max=0.2)
    assert T.isfinite(loss)
    assert comps["positive_frac"] > 0

    # No neighbourhood at all -> zero loss, zero positives.
    lonely = T.tensor(
        [
            [50.0, 0.0, 0.0, 211.0],
            [500.0, 2.0, 2.0, 22.0],
        ]
    )
    loss0, comps0 = supcon_physics_loss(T.randn(2, 8), lonely, pid_index=3)
    assert loss0 == 0.0
    assert comps0["positive_frac"] == 0.0


# ---------------------------------------------------------------- augmentations


def test_augmentation_preserves_mask_categoricals_and_shape():
    csts = T.randn(4, 10, 5).abs()  # col 0 = pt (positive)
    csts[..., 4] = T.randint(0, 3, (4, 10)).float()  # col 4 = categorical
    mask = T.rand(4, 10) > 0.3
    csts[~mask] = 0.0

    aug = ConstituentAugmentation(
        pt_indices=(0,),
        eta_indices=(1,),
        phi_indices=(2,),
        categorical_indices=(4,),
        pt_smear_sigma=0.1,
        angle_smear_sigma=0.05,
        noise_sigma=0.01,
    )
    out = aug(csts, mask)

    assert out.shape == csts.shape
    assert T.equal(out[~mask], csts[~mask])  # padding untouched
    assert T.equal(out[..., 4], csts[..., 4])  # categorical untouched
    assert not T.equal(out[mask], csts[mask])  # something was augmented
    assert (out[..., 0][mask] > 0).all()  # pT stays positive
    assert not T.equal(aug(csts, mask), aug(csts, mask))  # stochastic


def test_two_view_transform_adds_view_keys():
    batch = {"csts": T.randn(2, 5, 3), "mask": T.ones(2, 5, dtype=T.bool)}
    aug = ConstituentAugmentation(pt_indices=(0,), pt_smear_sigma=0.1)
    out = TwoViewTransform(aug)(batch)
    assert "csts_view0" in out and "csts_view1" in out
    assert not T.equal(out["csts_view0"], out["csts_view1"])
    assert T.equal(out["csts"], batch["csts"])  # original untouched


# --------------------------------------------------------------------- metrics


def test_embedding_collapse_metrics_separate_healthy_from_collapsed():
    healthy = compute_embedding_collapse(T.randn(256, 8))
    collapsed = compute_embedding_collapse(T.zeros(256, 8))

    assert healthy["effective_rank"] > 4
    assert healthy["active_dims"] == pytest.approx(1.0)
    assert collapsed["active_dims"] == 0.0
    assert collapsed["std_mean"] == 0.0


def test_alignment_uniformity():
    za = T.randn(256, 8)
    out = compute_alignment_uniformity(za, za + 0.01 * T.randn_like(za))
    assert T.isfinite(out["uniformity"])
    assert out["alignment"] < 0.1  # near-identical views align


def test_knn_pid_purity_on_separated_clusters():
    z = T.cat([T.randn(64, 4) - 10, T.randn(64, 4) + 10])
    pid = T.cat([T.zeros(64), T.ones(64)])
    assert compute_knn_pid_purity(z, pid, k=5) > 0.95


# ----------------------------------------------------- end-to-end smoke + export


@pytest.fixture
def contrastive_cfg(hdf5_file, tmp_path):
    output_dir = tmp_path / "results"
    full_path = output_dir / "test_project" / "contrastive_run"
    full_path.mkdir(parents=True)

    datamodule = {
        "_target_": "heptokens.data.structured_array.StructuredArrayModule",
        "dataset": {
            "_target_": "heptokens.data.structured_array.StructuredArrayHDF5Dataset",
            "_partial_": True,
            "file_path": str(hdf5_file),
            "obj_group": "jets",
            "set_group": "tracks",
            "obj_features": ["pt", "eta"],
            "set_features": ["pt", "eta", "phi"],
            "label_key": "label",
            "mask_key": "valid",
        },
        "data_path": str(hdf5_file),
        "train_frac": 0.7,
        "val_frac": 0.15,
        "test_frac": 0.15,
        "batch_size": 32,
        "num_workers": 0,
    }

    return OmegaConf.create(
        {
            "seed": 42,
            "precision": "medium",
            "compile": None,
            "tags": None,
            "full_resume": False,
            "ckpt_flag": "last.ckpt",
            "ckpt_path": None,
            "weight_ckpt_path": None,
            "test_only": False,
            "full_path": str(full_path),
            "datamodule": datamodule,
            "model": {
                "_target_": "heptokens.models.contrastive.LitContrastiveTokenizer",
                "encoder": {
                    "_target_": "heptokens.models.coders.Encoder",
                    "_partial_": True,
                    "model": {
                        "_target_": "heptokens.models.coders.CoderModel",
                        "_partial_": True,
                        "hidden_dims": [16, 16],
                    },
                },
                "loss_fn": {
                    "_target_": "heptokens.models.losses.vicreg_loss",
                    "_partial_": True,
                },
                "augmentation": {
                    "_target_": "heptokens.data.augmentations.ConstituentAugmentation",
                    "pt_indices": [0],
                    "eta_indices": [1],
                    "phi_indices": [2],
                    "pt_smear_sigma": 0.1,
                    "angle_smear_sigma": 0.05,
                },
                "latent_dim": 4,
                "proj_dim": 8,
                "proj_hidden": 8,
                "learning_rate": 1e-3,
            },
            "trainer": {
                "_target_": "lightning.Trainer",
                "max_epochs": 1,
                "limit_train_batches": 2,
                "limit_val_batches": 1,
                "accelerator": "cpu",
                "devices": 1,
                "enable_progress_bar": False,
                "default_root_dir": str(full_path),
                "reload_dataloaders_every_n_epochs": 0,
            },
            "callbacks": {
                "checkpoint": {
                    "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "dirpath": str(full_path / "checkpoints"),
                    "filename": "last",
                    "enable_version_counter": False,
                    "auto_insert_metric_name": False,
                },
                "monitor": {
                    "_target_": "heptokens.callbacks.contrastive.ContrastiveMonitor",
                    "pid_index": None,
                },
            },
            "logger": {
                "_target_": "lightning.pytorch.loggers.CSVLogger",
                "save_dir": str(full_path),
                "name": "test",
            },
        }
    )


def test_train_contrastive_then_export_latents(contrastive_cfg, hdf5_file, tmp_path):
    train(contrastive_cfg)

    ckpt_path = Path(contrastive_cfg.full_path) / "checkpoints" / "last.ckpt"
    assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"

    export_dir = tmp_path / "latents_output"
    export_cfg = OmegaConf.create(
        {
            "ckpt_path": str(ckpt_path),
            "output_dir": str(export_dir),
            "full_path": str(export_dir),
            "datamodule": OmegaConf.to_container(contrastive_cfg.datamodule),
            "trainer": {
                "_target_": "lightning.Trainer",
                "accelerator": "cpu",
                "devices": 1,
                "enable_progress_bar": False,
            },
        }
    )
    export_cfg.datamodule.train_frac = 0.0
    export_cfg.datamodule.val_frac = 0.0
    export_cfg.datamodule.test_frac = 1.0

    export_latents(export_cfg)

    stem = hdf5_file.stem
    npz_path = export_dir / f"{stem}.npz"
    assert npz_path.exists()

    data = np.load(npz_path)
    assert "latents" in data
    assert "codebooks" not in data  # continuous tokens: no lookup table
    assert data["latents"].dtype == np.float32
    assert data["latents"].ndim == 3
    assert data["latents"].shape[2] == 4  # latent_dim
    assert np.isfinite(data["latents"]).all()
    assert "labels" in data  # fixture dataset has labels

    meta_path = export_dir / f"{stem}_meta.json"
    assert meta_path.exists()
