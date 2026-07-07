"""Train the continuous contrastive tokenizer and the VQ-VAE baseline.

Matched budgets: identical transformer encoders (d_model 128, 4 layers) and the
same per-constituent capacity — contrastive latent_dim 16 vs VQ codebook_dim 16
(codebook 512, 3 quantizers). Both trained on the same 620k-jet dataset with the
same preprocessing, split, and trainer budget.

Usage:
    python train_tokenizers.py --model contrastive [--epochs 20]
    python train_tokenizers.py --model vqvae       [--epochs 20]
"""

import argparse
from functools import partial

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from common import LATENT_DIM, SEED, TOKENIZER_DIR, make_datamodule, transformer_partial

from heptokens.callbacks.contrastive import ContrastiveMonitor
from heptokens.callbacks.recon import BaseReconstructionMonitor
from heptokens.data.augmentations import ConstituentAugmentation
from heptokens.models.coders import Decoder, Encoder
from heptokens.models.contrastive import LitContrastiveTokenizer
from heptokens.models.utils import warmup_cosine_scheduler
from heptokens.models.vq_vae import LitVqVae


def build_contrastive(data_sample):
    # Views are made on the (preprocessed) features: additive smearing on all
    # three scaled columns emulates detector resolution in the scaled space.
    augmentation = ConstituentAugmentation(
        pt_indices=(),
        eta_indices=(0, 1, 2),
        phi_indices=(),
        pt_smear_sigma=0.0,
        angle_smear_sigma=0.05,
        wrap_phi=False,
    )
    return LitContrastiveTokenizer(
        encoder=partial(Encoder, model=transformer_partial()),
        augmentation=augmentation,
        latent_dim=LATENT_DIM,
        proj_dim=32,
        proj_hidden=64,
        learning_rate=1e-3,
        scheduler=partial(warmup_cosine_scheduler, warmup_epochs=3),
        data_sample=data_sample,
    )


def build_vqvae(data_sample):
    return LitVqVae(
        encoder=partial(Encoder, model=transformer_partial()),
        decoder=partial(Decoder, model=transformer_partial()),
        codebook_size=512,
        codebook_dim=LATENT_DIM,
        num_quantizers=3,
        learning_rate=1e-3,
        scheduler=partial(warmup_cosine_scheduler, warmup_epochs=3),
        data_sample=data_sample,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["contrastive", "vqvae"], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    L.seed_everything(SEED, workers=True)
    torch.set_float32_matmul_precision("medium")

    run_dir = TOKENIZER_DIR / args.model
    run_dir.mkdir(parents=True, exist_ok=True)

    dm = make_datamodule(batch_size=args.batch_size, num_workers=args.workers)
    dm.setup()
    data_sample = dm.get_data_sample()
    print(f"data sample: {tuple(data_sample.shape)}")

    if args.model == "contrastive":
        model = build_contrastive(data_sample)
        monitor_cb = ContrastiveMonitor(max_batches=4)
    else:
        model = build_vqvae(data_sample)
        monitor_cb = BaseReconstructionMonitor(max_batches=4)

    callbacks = [
        ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            filename="last",
            enable_version_counter=False,
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            filename="best",
            monitor="val/total_loss",
            mode="min",
            save_top_k=1,
            enable_version_counter=False,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        monitor_cb,
    ]

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=run_dir, name="csv", flush_logs_every_n_steps=50),
        log_every_n_steps=20,
        enable_progress_bar=False,
        default_root_dir=run_dir,
    )
    trainer.fit(model, datamodule=dm)
    print(f"Done. Checkpoints in {run_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
