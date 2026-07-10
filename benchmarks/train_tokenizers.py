"""Train the tokenizer arms: particle library, contrastive-VICReg, VQ-VAE.

The `library` arm is the particle-library tokenizer (user specification):
a context-free per-particle MLP trained with the graded physics-kernel
objective — particles with same PID, same charge, dpT <= 10% and
|deta|, |dphi| <= 0.1 cluster; the rest are repelled. On jets5c (no PID/charge
columns, jet-relative coordinates) the kernel runs kinematics-only — the
mechanics check; the physics run is `--data aoj`.

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

from common import (
    LATENT_DIM,
    LIBRARY_CRITERION,
    SEED,
    TOKENIZER_DIR,
    make_datamodule,
    transformer_partial,
)

from heptokens.callbacks.contrastive import ContrastiveMonitor
from heptokens.callbacks.recon import BaseReconstructionMonitor
from heptokens.data.augmentations import ConstituentAugmentation
from heptokens.models.coders import Decoder, Encoder
from heptokens.models.contrastive import LitContrastiveTokenizer
from heptokens.models.contrastive_vqvae import LitContrastiveVqVae
from heptokens.models.losses import physics_kernel_loss
from heptokens.models.particle_encoder import ParticleDecoder, ParticleEncoder
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


def build_library(
    data_sample,
    data,
    region="coords",
    hard_windows=False,
    recon_weight=0.0,
    dpt_frac=None,
    latent_dim=LATENT_DIM,
    hidden_dims=(128, 128, 64),
):
    """Particle-library tokenizer (context-free MLP + graded physics kernel).

    With ``recon_weight > 0`` this becomes the **hybrid** objective: the
    ParticleDecoder reconstruction term preserves particle information while
    the kernel shapes the metric. ``hidden_dims`` sets the encoder MLP width
    (the capacity ablation).
    """
    has_pid = data == "aoj"  # jets5c carries only (pt, deta, dphi)
    criterion = dict(LIBRARY_CRITERION)
    if dpt_frac is not None:
        criterion["dpt_frac"] = dpt_frac
    encoder = partial(
        ParticleEncoder,
        continuous_indices=(0, 1, 2),
        pid_index=3 if has_pid else None,
        charge_index=4 if has_pid else None,
        pid_emb_dim=8,
        charge_emb_dim=2,
        hidden_dims=hidden_dims,
    )
    if hard_windows:
        from heptokens.models.losses import supcon_physics_loss

        loss_fn = partial(
            supcon_physics_loss,
            pid_index=3 if has_pid else None,
            charge_index=4 if has_pid else None,
            pt_index=0, eta_index=1, phi_index=2,
            dpt_frac=criterion["dpt_frac"], deta_max=0.1, dphi_max=0.1,
            max_points=4096,
        )
    else:
        loss_fn = partial(
            physics_kernel_loss,
            pid_index=3 if has_pid else None,
            charge_index=4 if has_pid else None,
            region=region,
            dr_max=0.1,
            max_points=4096,
            **criterion,
        )
    decoder = None
    if recon_weight > 0:
        decoder = partial(
            ParticleDecoder,
            continuous_indices=(0, 1, 2),
            pid_index=3 if has_pid else None,
            charge_index=4 if has_pid else None,
            hidden_dims=hidden_dims,
        )
    return LitContrastiveTokenizer(
        encoder=encoder,
        loss_fn=loss_fn,
        loss_inputs="features",
        use_projector=False,
        decoder=decoder,
        recon_weight=recon_weight,
        latent_dim=latent_dim,
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


def build_vqvae_particle(
    data_sample, codebook_size=512, num_quantizers=3, hidden_dims=(128, 128, 64)
):
    """Apples-to-apples VQ baseline: the library's exact ParticleEncoder, a
    residual-VQ bottleneck (default 3 x 512 = 27 bits/particle), and a
    full-particle reconstruction target (L1 kinematics + CE PID + CE charge) so
    identity information must pass through the bottleneck. Only the objective
    differs from the library arm. ``codebook_size``/``num_quantizers`` set the
    rate (e.g. 8192 x 1 = 13 bits for the production-analog anchor)."""
    return LitVqVae(
        encoder=partial(
            ParticleEncoder,
            continuous_indices=(0, 1, 2),
            pid_index=3,
            charge_index=4,
            pid_emb_dim=8,
            charge_emb_dim=2,
            hidden_dims=hidden_dims,
        ),
        decoder=partial(
            ParticleDecoder,
            continuous_indices=(0, 1, 2),
            pid_index=3,
            charge_index=4,
            hidden_dims=hidden_dims,
        ),
        codebook_size=codebook_size,
        codebook_dim=LATENT_DIM,
        num_quantizers=num_quantizers,
        learning_rate=1e-3,
        scheduler=partial(warmup_cosine_scheduler, warmup_epochs=3),
        data_sample=data_sample,
    )


def build_contrastive_vqvae(
    data_sample,
    data,
    region="coords",
    dpt_frac=None,
    kernel_weight=1.0,
    kernel_on="z_q",
    codebook_size=512,
    num_quantizers=3,
    hidden_dims=(128, 128, 64),
):
    """Contrastive-VQ: the same ParticleEncoder + residual-VQ + full-particle
    reconstruction as ``build_vqvae_particle``, plus the graded physics kernel
    applied to the quantized token ``z_q`` — so the codebook is *shaped* by the
    library criterion. Matched to the plain VQ at the same rate; only the kernel
    term differs, isolating the value of criterion-shaping the codebook."""
    has_pid = data == "aoj"
    criterion = dict(LIBRARY_CRITERION)
    if dpt_frac is not None:
        criterion["dpt_frac"] = dpt_frac
    loss_fn = partial(
        physics_kernel_loss,
        pid_index=3 if has_pid else None,
        charge_index=4 if has_pid else None,
        region=region,
        dr_max=0.1,
        max_points=4096,
        **criterion,
    )
    return LitContrastiveVqVae(
        encoder=partial(
            ParticleEncoder,
            continuous_indices=(0, 1, 2),
            pid_index=3 if has_pid else None,
            charge_index=4 if has_pid else None,
            pid_emb_dim=8,
            charge_emb_dim=2,
            hidden_dims=hidden_dims,
        ),
        decoder=partial(
            ParticleDecoder,
            continuous_indices=(0, 1, 2),
            pid_index=3 if has_pid else None,
            charge_index=4 if has_pid else None,
            hidden_dims=hidden_dims,
        ),
        loss_fn=loss_fn,
        kernel_weight=kernel_weight,
        kernel_on=kernel_on,
        codebook_size=codebook_size,
        codebook_dim=LATENT_DIM,
        num_quantizers=num_quantizers,
        learning_rate=1e-3,
        scheduler=partial(warmup_cosine_scheduler, warmup_epochs=3),
        data_sample=data_sample,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["library", "hybrid", "contrastive", "vqvae",
                 "vqvae_particle", "contrastive_vqvae"],
        required=True,
    )
    parser.add_argument("--data", choices=["jets5c", "aoj"], default="jets5c")
    parser.add_argument("--region", choices=["coords", "cone"], default="coords")
    parser.add_argument("--hard-windows", action="store_true")
    parser.add_argument("--recon-weight", type=float, default=1.0,
                        help="hybrid only: weight of the reconstruction term")
    parser.add_argument("--kernel-weight", type=float, default=1.0,
                        help="contrastive_vqvae only: weight of the kernel term")
    parser.add_argument("--kernel-on", choices=["z_q", "z_e"], default="z_q",
                        help="contrastive_vqvae only: organize quantized z_q or continuous z_e")
    parser.add_argument("--dpt-frac", type=float, default=None,
                        help="override the kernel dpT width (default 0.1)")
    parser.add_argument("--latent-dim", type=int, default=None,
                        help="override token dimension (default 16)")
    parser.add_argument("--codebook-size", type=int, default=512,
                        help="VQ codes per quantizer (default 512)")
    parser.add_argument("--num-quantizers", type=int, default=3,
                        help="VQ residual-quantizer stages (default 3)")
    parser.add_argument("--hidden-dims", default=None,
                        help="encoder MLP widths, comma-separated (default 128,128,64)")
    parser.add_argument("--run-name", default=None, help="override output dir name")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    run_name = args.run_name or (
        args.model if args.data == "jets5c" else f"{args.model}_{args.data}"
    )
    if args.seed != SEED:
        run_name = f"{run_name}_s{args.seed}"
    run_dir = TOKENIZER_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dm = make_datamodule(batch_size=args.batch_size, num_workers=args.workers, data=args.data)
    dm.setup()
    data_sample = dm.get_data_sample()
    print(f"data sample: {tuple(data_sample.shape)}")

    hidden_dims = (
        tuple(int(x) for x in args.hidden_dims.split(","))
        if args.hidden_dims
        else (128, 128, 64)
    )

    if args.model in ("library", "hybrid"):
        model = build_library(
            data_sample,
            args.data,
            region=args.region,
            hard_windows=args.hard_windows,
            recon_weight=args.recon_weight if args.model == "hybrid" else 0.0,
            dpt_frac=args.dpt_frac,
            latent_dim=args.latent_dim or LATENT_DIM,
            hidden_dims=hidden_dims,
        )
        monitor_cb = ContrastiveMonitor(
            max_batches=4, pid_index=3 if args.data == "aoj" else None
        )
    elif args.model == "contrastive_vqvae":
        model = build_contrastive_vqvae(
            data_sample,
            args.data,
            region=args.region,
            dpt_frac=args.dpt_frac,
            kernel_weight=args.kernel_weight,
            kernel_on=args.kernel_on,
            codebook_size=args.codebook_size,
            num_quantizers=args.num_quantizers,
            hidden_dims=hidden_dims,
        )
        monitor_cb = ContrastiveMonitor(
            max_batches=4, pid_index=3 if args.data == "aoj" else None
        )
    elif args.model == "contrastive":
        model = build_contrastive(data_sample)
        monitor_cb = ContrastiveMonitor(max_batches=4)
    elif args.model == "vqvae_particle":
        model = build_vqvae_particle(
            data_sample,
            codebook_size=args.codebook_size,
            num_quantizers=args.num_quantizers,
            hidden_dims=hidden_dims,
        )
        monitor_cb = BaseReconstructionMonitor(max_batches=4)
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
