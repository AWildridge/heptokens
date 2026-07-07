"""Tests for the particle-library tokenizer stack.

Criterion under test (user specification): particles with same PID, same
charge, relative dpT <= dpt_frac and |deta|, |dphi| <= 0.1 are positives;
everything else is repelled.
"""

import h5py
import numpy as np
import pytest
import torch as T
from omegaconf import OmegaConf

from heptokens.data.collation import keep_raw_batch
from heptokens.data.transforms import (
    PassthroughColumnsTransformer,
    create_preprocessing_transformer,
)
from heptokens.models.losses import physics_kernel_loss, supcon_physics_loss
from heptokens.models.particle_encoder import ParticleEncoder
from heptokens.models.utils import compute_retrieval_precision
from heptokens.train import train

# Columns: [pt, eta, phi, pid, charge]
PID, CHG = 3, 4
KERNEL_KW = dict(pt_index=0, eta_index=1, phi_index=2, pid_index=PID, charge_index=CHG)


def make_feats(rows):
    return T.tensor(rows, dtype=T.float32)


# ------------------------------------------------------------- data plumbing


def test_passthrough_scaler_preserves_categoricals():
    rng = np.random.default_rng(0)
    X = np.concatenate(
        [
            rng.lognormal(2, 1, (1000, 1)).astype(np.float32),
            rng.normal(0, 1, (1000, 2)).astype(np.float32),
            rng.choice([22.0, 211.0, -211.0], (1000, 1)).astype(np.float32),
            rng.choice([-1.0, 0.0, 1.0], (1000, 1)).astype(np.float32),
        ],
        axis=1,
    )
    scaler = PassthroughColumnsTransformer(
        create_preprocessing_transformer(mode="quantile", n_quantiles=100),
        continuous_indices=[0, 1, 2],
    )
    scaler.fit(X)
    out = scaler.transform(X)

    assert np.array_equal(out[:, 3:], X[:, 3:])  # categoricals untouched
    assert not np.allclose(out[:, 0], X[:, 0])  # continuous scaled
    assert abs(out[:, 0].mean()) < 0.1  # roughly normal
    back = scaler.inverse_transform(out)
    assert np.allclose(back[:, :3], X[:, :3], atol=1e-3)
    assert scaler.n_features_in_ == 5


def test_keep_raw_batch_snapshots_before_preprocessing():
    csts = T.randn(2, 4, 3)
    batch = {"csts": csts, "mask": T.ones(2, 4, dtype=T.bool)}
    out = keep_raw_batch(batch)
    original = csts.clone()
    batch["csts"] *= 100.0  # simulate in-place preprocessing
    assert T.equal(out["csts_raw"], original)


# ----------------------------------------------------------- ParticleEncoder


def test_particle_encoder_shapes_mask_and_pid_mapping():
    enc = ParticleEncoder(
        input_dim=5,
        output_dim=8,
        continuous_indices=(0, 1, 2),
        pid_index=PID,
        charge_index=CHG,
        pdg_ids=(22, 211, -211),
        hidden_dims=(16,),
    )
    x = T.randn(2, 6, 5)
    x[..., PID] = T.tensor([22.0, 211.0, -211.0, 999.0, 22.0, 211.0])  # 999 = unknown
    x[..., CHG] = T.tensor([0.0, 1.0, -1.0, 0.0, 0.0, 1.0])
    mask = T.ones(2, 6, dtype=T.bool)
    mask[0, -1] = False

    z = enc({"csts": x, "mask": mask})
    assert z.shape == (2, 6, 8)
    assert T.all(z[0, -1] == 0)  # masked zeroed

    ids = enc._map_pid(x[..., PID])
    assert ids[0, 0] == 0 and ids[0, 1] == 1 and ids[0, 2] == 2
    assert ids[0, 3] == 3  # unknown -> "other" bucket

    # Same particle anywhere -> same token (context-free property).
    z_single = enc({"csts": x[:1, :1]})
    z_moved = enc({"csts": x[:1, :1].repeat(1, 3, 1)})
    assert T.allclose(z_single[0, 0], z_moved[0, 2], atol=1e-6)

    with pytest.raises(ValueError):
        ParticleEncoder(input_dim=3, pid_index=5)


# ------------------------------------------------------------- kernel loss


def test_kernel_prefers_physically_close_pairs():
    """Embedding arrangements matching the kernel geometry score lower loss."""
    feats = make_feats(
        [
            [50.0, 0.50, 1.00, 211.0, 1.0],  # anchor
            [52.0, 0.52, 1.02, 211.0, 1.0],  # close: same pid/charge, dpt 4%, d(eta,phi) 0.02
            [50.0, 1.50, -2.00, 211.0, 1.0],  # far in detector
        ]
    )
    good = T.tensor([[1.0, 0.0], [1.0, 0.05], [-1.0, 0.0]])  # 0~1 together, 2 away
    bad = T.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.05]])  # 0~2 together, 1 away
    loss_good, comps = physics_kernel_loss(good, feats, **KERNEL_KW)
    loss_bad, _ = physics_kernel_loss(bad, feats, **KERNEL_KW)
    assert loss_good < loss_bad
    assert comps["positive_frac"] > 0


def test_kernel_pid_and_charge_gates():
    z = T.randn(3, 4)
    # Identical kinematics, different PID -> no kernel mass at all.
    diff_pid = make_feats(
        [[50.0, 0.5, 1.0, 211.0, 1.0], [50.0, 0.5, 1.0, 22.0, 0.0], [50.0, 0.5, 1.0, 13.0, -1.0]]
    )
    loss, comps = physics_kernel_loss(z, diff_pid, **KERNEL_KW)
    assert loss == 0.0 and comps["positive_frac"] == 0.0

    # Same PID but opposite charge (e.g. pi+ vs pi-) -> charge gate blocks.
    same_pid_diff_q = make_feats(
        [[50.0, 0.5, 1.0, 211.0, 1.0], [50.0, 0.5, 1.0, 211.0, -1.0], [9.0, 2.0, -2.0, 22.0, 0.0]]
    )
    loss, comps = physics_kernel_loss(z, same_pid_diff_q, **KERNEL_KW)
    assert loss == 0.0 and comps["positive_frac"] == 0.0

    # Without the charge gate the pi+/pi- pair would count.
    kw = dict(KERNEL_KW, charge_index=None)
    _, comps = physics_kernel_loss(z, same_pid_diff_q, **kw)
    assert comps["positive_frac"] > 0


def test_kernel_region_modes_and_grad():
    rng = T.Generator().manual_seed(0)
    feats = T.rand(64, 5, generator=rng)
    feats[:, 0] = feats[:, 0] * 20 + 1
    feats[:, PID] = 211.0
    feats[:, CHG] = 1.0
    z = T.randn(64, 8, requires_grad=True)

    loss_c, _ = physics_kernel_loss(z, feats, region="coords", **KERNEL_KW)
    loss_r, _ = physics_kernel_loss(z, feats, region="cone", dr_max=0.1, **KERNEL_KW)
    assert T.isfinite(loss_c) and T.isfinite(loss_r)
    loss_c.backward()
    assert T.isfinite(z.grad).all()

    with pytest.raises(ValueError):
        physics_kernel_loss(z, feats, region="banana", **KERNEL_KW)


def test_kernel_var_cov_regularizer():
    feats = T.rand(32, 5)
    feats[:, 0] += 1
    feats[:, PID] = 211.0
    feats[:, CHG] = 1.0
    z = T.randn(32, 8)
    _, comps = physics_kernel_loss(z, feats, var_weight=1.0, cov_weight=1.0, **KERNEL_KW)
    assert "var_loss" in comps and "cov_loss" in comps


def test_supcon_charge_gate_and_coordinate_windows():
    # dphi = 0.5 violates the 0.1 per-coordinate window but passes a 0.6 cone.
    feats = make_feats(
        [[50.0, 0.50, 1.00, 211.0, 1.0], [50.0, 0.50, 1.50, 211.0, 1.0]]
    )
    z = T.randn(2, 4)
    _, comps = supcon_physics_loss(
        z, feats, pid_index=PID, charge_index=CHG, deta_max=0.1, dphi_max=0.1
    )
    assert comps["positive_frac"] == 0.0
    _, comps = supcon_physics_loss(z, feats, pid_index=PID, charge_index=CHG, dr_max=0.6)
    assert comps["positive_frac"] > 0

    # Opposite charge blocks even at identical kinematics.
    feats[1] = T.tensor([50.0, 0.50, 1.00, 211.0, -1.0])
    _, comps = supcon_physics_loss(
        z, feats, pid_index=PID, charge_index=CHG, deta_max=0.1, dphi_max=0.1
    )
    assert comps["positive_frac"] == 0.0


# --------------------------------------------------------- retrieval metric


def test_retrieval_precision_separates_organized_from_random():
    rng = np.random.default_rng(1)
    n = 200
    group = rng.integers(0, 2, n)  # two particle "species+region" groups
    feats = np.zeros((n, 5), dtype=np.float32)
    feats[:, 0] = 50 + rng.normal(0, 1, n)  # tight pt
    feats[:, 1] = np.where(group == 0, 0.5, -1.5) + rng.normal(0, 0.01, n)
    feats[:, 2] = np.where(group == 0, 1.0, -2.0) + rng.normal(0, 0.01, n)
    feats[:, 3] = np.where(group == 0, 211.0, 22.0)
    feats[:, 4] = np.where(group == 0, 1.0, 0.0)

    organized = T.tensor(np.stack([group * 10.0, group * -3.0], axis=1), dtype=T.float32)
    organized = organized + 0.01 * T.randn(n, 2)
    res = compute_retrieval_precision(organized, T.tensor(feats), **KERNEL_KW)
    assert res["precision_at_k"] > 0.9

    random_z = T.randn(n, 2)
    res_rand = compute_retrieval_precision(random_z, T.tensor(feats), **KERNEL_KW)
    assert res_rand["precision_at_k"] < res["precision_at_k"]
    assert 0 < res_rand["criterion_density"] < 1


# ----------------------------------------------------------- end-to-end smoke


@pytest.fixture
def pid_hdf5_file(tmp_path):
    """Synthetic two-group HDF5 with PID and charge columns."""
    n_jets, n_csts = 80, 15
    rng = np.random.default_rng(7)
    path = tmp_path / "pid_jets.h5"
    pid_choices = np.array([211.0, -211.0, 22.0])
    charge_of = {211.0: 1.0, -211.0: -1.0, 22.0: 0.0}
    with h5py.File(path, "w") as f:
        jets = f.create_group("jets")
        jets.create_dataset("pt", data=rng.uniform(30, 300, n_jets).astype(np.float32))
        jets.create_dataset("nconst", data=np.full(n_jets, n_csts, np.float32))
        jets.create_dataset("label", data=rng.choice([0, 1], n_jets).astype(np.int64))
        trk = f.create_group("csts")
        pid = rng.choice(pid_choices, (n_jets, n_csts))
        trk.create_dataset("pt", data=rng.uniform(1, 50, (n_jets, n_csts)).astype(np.float32))
        trk.create_dataset("eta", data=rng.uniform(-2, 2, (n_jets, n_csts)).astype(np.float32))
        trk.create_dataset("phi", data=rng.uniform(-3.1, 3.1, (n_jets, n_csts)).astype(np.float32))
        trk.create_dataset("pid", data=pid.astype(np.float32))
        trk.create_dataset(
            "charge", data=np.vectorize(charge_of.get)(pid).astype(np.float32)
        )
        valid = np.ones((n_jets, n_csts), dtype=bool)
        valid[:, -3:] = rng.random((n_jets, 3)) > 0.5
        trk.create_dataset("valid", data=valid)
    return path


def test_train_particle_library_smoke(pid_hdf5_file, tmp_path):
    full_path = tmp_path / "library_run"
    full_path.mkdir(parents=True)
    cfg = OmegaConf.create(
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
            "datamodule": {
                "_target_": "heptokens.data.structured_array.StructuredArrayModule",
                "dataset": {
                    "_target_": "heptokens.data.structured_array.StructuredArrayHDF5Dataset",
                    "_partial_": True,
                    "file_path": str(pid_hdf5_file),
                    "obj_group": "jets",
                    "set_group": "csts",
                    "obj_features": ["pt", "nconst"],
                    "set_features": ["pt", "eta", "phi", "pid", "charge"],
                    "label_key": "label",
                    "mask_key": "valid",
                },
                "data_path": str(pid_hdf5_file),
                "train_frac": 0.7,
                "val_frac": 0.15,
                "test_frac": 0.15,
                "batch_size": 16,
                "num_workers": 0,
                "transforms": {
                    "keep_raw": {
                        "_target_": "heptokens.data.collation.keep_raw_batch",
                        "_partial_": True,
                    },
                },
            },
            "model": {
                "_target_": "heptokens.models.contrastive.LitContrastiveTokenizer",
                "encoder": {
                    "_target_": "heptokens.models.particle_encoder.ParticleEncoder",
                    "_partial_": True,
                    "continuous_indices": [0, 1, 2],
                    "pid_index": 3,
                    "charge_index": 4,
                    "pdg_ids": [211, -211, 22],
                    "pid_emb_dim": 4,
                    "charge_emb_dim": 2,
                    "hidden_dims": [16, 16],
                },
                "loss_fn": {
                    "_target_": "heptokens.models.losses.physics_kernel_loss",
                    "_partial_": True,
                    "pt_index": 0,
                    "eta_index": 1,
                    "phi_index": 2,
                    "pid_index": 3,
                    "charge_index": 4,
                    # generous windows so the tiny fixture has positives
                    "dpt_frac": 0.5,
                    "deta_max": 0.5,
                    "dphi_max": 0.5,
                },
                "loss_inputs": "features",
                "use_projector": False,
                "latent_dim": 4,
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
            },
            "logger": {
                "_target_": "lightning.pytorch.loggers.CSVLogger",
                "save_dir": str(full_path),
                "name": "test",
            },
        }
    )
    train(cfg)
    assert (full_path / "checkpoints" / "last.ckpt").exists()
