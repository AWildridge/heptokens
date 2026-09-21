"""Head-to-head comparison harness: continuous contrastive tokenizer vs the
existing heptokens baselines, evaluated with the library's own probe models.

Arms (consumer-view: probes only see what the interface provides):
    0. Null reference — FeatureClassifier on raw constituent features
       (upper reference: full information, no tokenizer).
    1. Continuous contrastive tokens — frozen LitContrastiveTokenizer injected
       via ContinuousTokenizer transform, probed with VectorClassifier.
    B. Discrete VQ baseline — frozen LitVqVae injected via VqvaeTokenizer
       transform, probed with the same VectorClassifier.

At this smoke scale (synthetic random data) only the *mechanics* are asserted:
every arm trains, validates, and reports finite accuracy/AUC through the same
probe protocol. Physics conclusions require the AOJ-scale run with the same
components.
"""

from functools import partial

import lightning as L
import pytest
import torch as T

from heptokens.data.augmentations import ConstituentAugmentation
from heptokens.data.collation import ContinuousTokenizer, VqvaeTokenizer
from heptokens.data.structured_array import StructuredArrayHDF5Dataset, StructuredArrayModule
from heptokens.models.coders import CoderModel, Decoder, Encoder
from heptokens.models.contrastive import LitContrastiveTokenizer
from heptokens.models.token_classifier import FeatureClassifier, VectorClassifier
from heptokens.models.vq_vae import LitVqVae

LATENT_DIM = 4
N_CLASSES = 3


def make_datamodule(hdf5_file, transforms=None):
    dataset = partial(
        StructuredArrayHDF5Dataset,
        file_path=str(hdf5_file),
        obj_group="jets",
        set_group="tracks",
        obj_features=["pt", "eta"],
        set_features=["pt", "eta", "phi"],
        label_key="label",
        mask_key="valid",
    )
    return StructuredArrayModule(
        dataset=dataset,
        train_frac=0.7,
        val_frac=0.3,
        test_frac=0.0,
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        transforms=transforms,
    )


def make_trainer(tmp_path, name):
    return L.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
        default_root_dir=str(tmp_path / name),
    )


@pytest.fixture
def tokenizer_ckpts(hdf5_file, tmp_path):
    """Train both tokenizers (tiny budgets) and return their checkpoint paths."""
    L.seed_everything(42)
    dm = make_datamodule(hdf5_file)
    dm.setup()
    data_sample = dm.get_data_sample()

    # --- Arm B: discrete VQ baseline (existing heptokens model)
    vqvae = LitVqVae(
        encoder=partial(Encoder, model=partial(CoderModel, hidden_dims=[16])),
        decoder=partial(Decoder, model=partial(CoderModel, hidden_dims=[16])),
        codebook_size=8,
        codebook_dim=LATENT_DIM,
        num_quantizers=2,
        learning_rate=1e-3,
        data_sample=data_sample,
    )
    trainer = make_trainer(tmp_path, "vqvae")
    trainer.fit(vqvae, datamodule=dm)
    vq_ckpt = tmp_path / "vqvae.ckpt"
    trainer.save_checkpoint(vq_ckpt)

    # --- Arm 1: continuous contrastive tokenizer (new)
    contrastive = LitContrastiveTokenizer(
        encoder=partial(Encoder, model=partial(CoderModel, hidden_dims=[16])),
        augmentation=ConstituentAugmentation(
            pt_indices=(0,),
            eta_indices=(1,),
            phi_indices=(2,),
            pt_smear_sigma=0.1,
            angle_smear_sigma=0.05,
        ),
        latent_dim=LATENT_DIM,
        proj_dim=8,
        proj_hidden=8,
        learning_rate=1e-3,
        data_sample=data_sample,
    )
    trainer = make_trainer(tmp_path, "contrastive")
    trainer.fit(contrastive, datamodule=dm)
    ct_ckpt = tmp_path / "contrastive.ckpt"
    trainer.save_checkpoint(ct_ckpt)

    return {"vqvae": vq_ckpt, "contrastive": ct_ckpt}


def _run_probe(probe, dm, tmp_path, name):
    trainer = make_trainer(tmp_path, name)
    trainer.fit(probe, datamodule=dm)
    (metrics,) = trainer.validate(probe, datamodule=dm)
    return metrics


def test_compare_tokenizers_through_shared_probes(hdf5_file, tmp_path, tokenizer_ckpts):
    """All arms must train and report finite metrics through the same probe."""
    L.seed_everything(42)
    probe_kwargs = dict(
        n_classes=N_CLASSES,
        d_model=16,
        n_heads=2,
        num_layers=1,
        dim_feedforward=32,
        pooling="mean",
        learning_rate=1e-3,
    )
    results = {}

    # Arm 0 (null reference): raw features, existing FeatureClassifier.
    dm = make_datamodule(hdf5_file)
    dm.setup()
    probe = FeatureClassifier(data_sample=dm.get_data_sample(), **probe_kwargs)
    results["null_features"] = _run_probe(probe, dm, tmp_path, "probe_null")

    # Arm 1 (new): continuous tokens via ContinuousTokenizer transform.
    dm = make_datamodule(
        hdf5_file, transforms={"tokenize": ContinuousTokenizer(str(tokenizer_ckpts["contrastive"]))}
    )
    probe = VectorClassifier(codebook_dim=LATENT_DIM, **probe_kwargs)
    results["contrastive_tokens"] = _run_probe(probe, dm, tmp_path, "probe_contrastive")

    # Arm B (baseline): VQ z_q via the existing VqvaeTokenizer transform.
    dm = make_datamodule(
        hdf5_file, transforms={"tokenize": VqvaeTokenizer(str(tokenizer_ckpts["vqvae"]))}
    )
    probe = VectorClassifier(codebook_dim=LATENT_DIM, **probe_kwargs)
    results["vqvae_tokens"] = _run_probe(probe, dm, tmp_path, "probe_vqvae")

    for arm, metrics in results.items():
        assert "valid/acc" in metrics, f"{arm}: probe did not report accuracy"
        for key in ("valid/acc", "valid/auc", "valid/total_loss"):
            assert T.isfinite(T.tensor(metrics[key])), f"{arm}: {key} not finite"

    header = f"{'arm':<20} {'valid/acc':>10} {'valid/auc':>10}"
    rows = [
        f"{arm:<20} {m['valid/acc']:>10.3f} {m['valid/auc']:>10.3f}" for arm, m in results.items()
    ]
    print("\n".join(["", "Tokenizer comparison (smoke scale):", header, *rows]))
