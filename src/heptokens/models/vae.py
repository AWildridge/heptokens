"""Continuous VAE per-constituent tokenizer — the continuous half of the
continuous-vs-discrete study.

``LitVae`` is the continuous counterpart to ``LitVqVae``: an encoder maps each
constituent to a Gaussian posterior ``q(z|x) = N(mu, sigma)`` (the encoder emits
``2*latent_dim`` = ``[mu, log-std]``), a sample ``z`` is decoded back to the full
particle, and a KL term regularises the posterior toward a unit prior:

    total = reconstruction_weight * recon(z, batch)      # information / cross-check
          + beta * KL(q(z|x) || N(0, I))                 # rate (bits = KL / ln2)
          (+ kernel_weight * kernel(mu, feats))          # optional library geometry

The **released token is the posterior mean ``mu``** (deterministic), so
``encode(batch)[0]`` follows the same convention as ``LitVqVae``/
``LitContrastiveTokenizer`` and the eval harness reads it unchanged.

Two properties make this the natural home for the physics-contrastive term
(cf. the VQ arm, where the straight-through estimator blunts it): ``mu`` is a
smooth, differentiable embedding, so a CLR-style kernel on it has a clean
gradient; and the **KL is a rate** — ``KL_nats / ln2`` bits/particle — which
places the continuous token on the same bits-per-particle axis as the VQ index
cost, the honest answer to "is a continuous token actually compressive?".
"""

import logging
import math
from typing import Dict, Optional, Tuple

import torch
from lightning import LightningModule

from heptokens.models.utils import ScheduledOptimiserMixin, compute_embedding_collapse

log = logging.getLogger(__name__)

LN2 = math.log(2.0)


class LitVae(ScheduledOptimiserMixin, LightningModule):
    """Continuous VAE tokenizer producing a per-constituent posterior mean token.

    Args:
        encoder: Partial finalised with ``(input_dim=..., output_dim=2*latent_dim)``
            — the encoder emits ``[mu, log-std]`` stacked on the last axis.
        decoder: Partial finalised with ``(input_dim=latent_dim, output_dim=...)``;
            exposes ``compute_loss(z, batch)`` (e.g. ``ParticleDecoder``).
        latent_dim: Token dimension (the released ``mu``).
        beta: KL weight (β-VAE). Linearly annealed 0→beta over
            ``kl_anneal_epochs`` epochs so the decoder learns to use the latent
            before the prior pulls it back.
        kl_anneal_epochs: Warmup length for β (0 disables annealing).
        free_bits: Per-dimension KL floor (nats); the loss ignores KL below this
            per dim, preventing posterior collapse of individual dimensions.
        loss_fn: Optional ``(z, feats) -> (loss, comps)`` physics-contrastive term
            applied to ``mu`` (e.g. ``physics_kernel_loss``/``supcon_physics_loss``).
            ``None`` → a plain VAE (and the state after ``load_from_checkpoint``,
            which only calls ``encode``).
        kernel_weight: Weight of the contrastive term.
        raw_key: Batch key holding physical-unit features (stashed by
            ``keep_raw_batch``); falls back to the model input when absent.
    """

    def __init__(
        self,
        encoder,
        decoder,
        latent_dim: int = 4,
        beta: float = 1.0,
        kl_anneal_epochs: int = 5,
        free_bits: float = 0.0,
        reconstruction_weight: float = 1.0,
        learning_rate: float = 1e-3,
        loss_fn=None,
        kernel_weight: float = 1.0,
        optimizer=None,
        scheduler=None,
        raw_key: str = "csts_raw",
        input_key: str = "csts",
        mask_key: str = "mask",
        data_sample: torch.Tensor = None,
        input_dim: int | None = None,
        **kwargs,
    ):
        super().__init__()
        # data_sample (tensor) and loss_fn (partial) are kept out of hparams:
        # the former bloats checkpoints, the latter is training-only (inference
        # calls only ``encode``). input_dim is persisted so load_from_checkpoint
        # rebuilds the coders without a data sample.
        self.save_hyperparameters(ignore=["data_sample", "loss_fn"])

        self.learning_rate = learning_rate
        self.latent_dim = latent_dim
        self.beta = beta
        self.kl_anneal_epochs = kl_anneal_epochs
        self.free_bits = free_bits
        self.reconstruction_weight = reconstruction_weight
        self.loss_fn = loss_fn
        self.kernel_weight = kernel_weight
        self.raw_key = raw_key
        self.input_key = input_key
        self.mask_key = mask_key

        if data_sample is not None:
            input_dim = data_sample.shape[-1]
        elif input_dim is None:
            input_dim = 3
        self.hparams["input_dim"] = input_dim

        # Encoder emits [mu, log-std]; decoder maps a latent sample -> features.
        self.encoder = encoder(input_dim=input_dim, output_dim=2 * latent_dim)
        self.decoder = decoder(input_dim=latent_dim, output_dim=input_dim)

    # ------------------------------------------------------------------ api
    def encode_stats(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the posterior ``(mu, log_std)``, each ``[batch, n_csts, latent_dim]``."""
        h = self.encoder(batch)
        mu, log_std = torch.chunk(h, 2, dim=-1)
        return mu, log_std

    def encode(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Released token = deterministic posterior mean ``mu``.

        Returns ``(mu, None, None)`` — the same tuple convention as
        ``LitVqVae.encode`` so monitors and tokenizer transforms can call
        ``encode(batch)[0]`` on either model.
        """
        mu, _ = self.encode_stats(batch)
        return mu, None, None

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encode(batch)[0]

    # ------------------------------------------------------------ internals
    def _kl_per_dim(
        self, mu: torch.Tensor, log_std: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, float]:
        """Per-dimension KL averaged over valid constituents, plus the total
        KL as a rate in bits/particle (``KL_nats / ln2``, no free-bits floor)."""
        if mask is not None:
            mu_v, ls_v = mu[mask], log_std[mask]
        else:
            mu_v = mu.reshape(-1, mu.shape[-1])
            ls_v = log_std.reshape(-1, log_std.shape[-1])
        # KL(N(mu, sigma^2) || N(0, 1)) per dimension.
        kl = 0.5 * (mu_v.pow(2) + (2 * ls_v).exp() - 1.0 - 2 * ls_v)  # [M, latent]
        kl_per_dim = kl.mean(dim=0)
        kl_bits = float((kl.sum(dim=-1).mean() / LN2).detach()) if mu_v.shape[0] else 0.0
        return kl_per_dim, kl_bits

    def _beta(self) -> float:
        if self.kl_anneal_epochs and self.kl_anneal_epochs > 0:
            return self.beta * min(1.0, (self.current_epoch + 1) / self.kl_anneal_epochs)
        return self.beta

    def _kernel_term(
        self, mu: torch.Tensor, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], dict]:
        if self.loss_fn is None:
            return None, {}
        mask = batch.get(self.mask_key)
        feats = batch.get(self.raw_key, batch[self.input_key])
        if mask is not None:
            mu_flat, f_flat = mu[mask], feats[mask]
        else:
            mu_flat = mu.reshape(-1, mu.shape[-1])
            f_flat = feats.reshape(-1, feats.shape[-1])
        if mu_flat.shape[0] < 3:
            log.warning("Batch with <3 valid constituents; skipping kernel term.")
            return None, {}
        return self.loss_fn(mu_flat, f_flat)

    def _shared_step(
        self, batch: Dict[str, torch.Tensor], prefix: str, batch_idx: int
    ) -> torch.Tensor:
        mask = batch.get(self.mask_key)
        mu, log_std = self.encode_stats(batch)

        # Reparameterised sample for reconstruction (mean at eval for stability);
        # zero the masked positions (never reach the decoder, but keep it clean).
        if self.training:
            z = mu + torch.randn_like(mu) * log_std.exp()
        else:
            z = mu
        if mask is not None:
            z = z * mask.unsqueeze(-1)
        recon_loss = self.decoder.compute_loss(z, batch)

        kl_per_dim, kl_bits = self._kl_per_dim(mu, log_std, mask)
        kl_term = torch.clamp(kl_per_dim, min=self.free_bits).sum()
        beta = self._beta()

        kernel_loss, components = self._kernel_term(mu, batch)

        total = self.reconstruction_weight * recon_loss + beta * kl_term
        if kernel_loss is not None:
            total = total + self.kernel_weight * kernel_loss

        self.log(f"{prefix}/total_loss", total, prog_bar=True)
        self.log(f"{prefix}/recon_loss", recon_loss, prog_bar=True)
        self.log(f"{prefix}/kl", kl_term, prog_bar=True)
        self.log(f"{prefix}/kl_rate_bits", kl_bits, prog_bar=True)
        self.log(f"{prefix}/beta", beta)
        for name, value in components.items():
            self.log(f"{prefix}/{name}", value, prog_bar=(name in ("kernel_loss", "supcon_loss")))

        # Collapse diagnostics on the released token (posterior mean).
        if prefix == "val" and batch_idx == 0:
            with torch.no_grad():
                flat = mu[mask] if mask is not None else mu.reshape(-1, mu.shape[-1])
                if flat.shape[0] > 2:
                    for name, value in compute_embedding_collapse(flat).items():
                        self.log(f"val/{name}", value)
        return total

    # ------------------------------------------------------------ lightning
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train", batch_idx)

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        total = self._shared_step(batch, "val", batch_idx)
        return {"val_loss": total}

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Return continuous tokens (posterior mean) + passthrough labels."""
        result = {"latents": self.encode(batch)[0]}
        if "labels" in batch:
            result["labels"] = batch["labels"]
        if "eventNumber" in batch:
            result["eventNumber"] = batch["eventNumber"]
        if "weights" in batch:
            result["weights"] = batch["weights"]
        return result
