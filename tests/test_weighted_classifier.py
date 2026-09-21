"""Per-sample weight support: dataset emission and weighted cross-entropy."""

import h5py
import numpy as np
import pytest
import torch

from heptokens.data.structured_array import StructuredArrayHDF5Dataset
from heptokens.models.token_classifier import VectorClassifier

KW = dict(n_classes=2, d_model=16, n_heads=2, num_layers=1, dim_feedforward=32,
          pooling="mean", codebook_dim=4)


def _batch(n=8, weights=None):
    torch.manual_seed(0)
    batch = {
        "z_q": torch.randn(n, 10, 4),
        "mask": torch.ones(n, 10, dtype=torch.bool),
        "labels": torch.randint(0, 2, (n,)),
    }
    if weights is not None:
        batch["weights"] = weights
    return batch


def test_unit_weights_match_unweighted():
    torch.manual_seed(1)
    model = VectorClassifier(**KW)
    plain = _batch()
    weighted = dict(plain, weights=torch.ones(len(plain["labels"])))
    loss_plain = model._shared_step(plain, "train")
    loss_weighted = model._shared_step(weighted, "train")
    assert torch.allclose(loss_plain, loss_weighted, atol=1e-6)


def test_weighted_loss_matches_manual():
    torch.manual_seed(2)
    model = VectorClassifier(**KW)
    w = torch.tensor([2.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0])
    batch = _batch(weights=w)
    loss = model._shared_step(batch, "train")
    with torch.no_grad():
        out = model.forward(batch)
        per = torch.nn.functional.cross_entropy(
            out, batch["labels"], label_smoothing=0.1, reduction="none"
        )
        expected = (per * w).sum() / w.sum()
    assert torch.allclose(loss, expected, atol=1e-6)


def test_predict_step_passes_weights():
    model = VectorClassifier(**KW)
    out = model.predict_step(_batch(weights=torch.arange(8.0)))
    assert torch.equal(out["weight"], torch.arange(8.0))
    assert "weight" not in model.predict_step(_batch())


def test_dataset_emits_weights(tmp_path):
    path = tmp_path / "toy.h5"
    n, nc = 20, 5
    with h5py.File(path, "w") as f:
        jets = f.create_group("jets")
        jets.create_dataset("pt", data=np.linspace(300, 400, n).astype(np.float32))
        jets.create_dataset("label", data=(np.arange(n) % 2).astype(np.int64))
        jets.create_dataset("weight", data=np.arange(1, n + 1, dtype=np.float32))
        csts = f.create_group("csts")
        csts.create_dataset("pt", data=np.random.rand(n, nc).astype(np.float32))
        csts.create_dataset("valid", data=np.ones((n, nc), dtype=bool))

    kw = dict(file_path=str(path), obj_group="jets", set_group="csts",
              obj_features=["pt"], set_features=["pt"], label_key="label",
              mask_key="valid")
    ds = StructuredArrayHDF5Dataset(**kw, weight_key="weight")
    sample = ds[3]
    assert sample["weights"] == pytest.approx(4.0)
    assert "weights" not in StructuredArrayHDF5Dataset(**kw)[3]
