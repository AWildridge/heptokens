"""Continuous contrastive per-constituent tokenizer.

``LitContrastiveTokenizer`` learns a continuous embedding for each constituent
with a two-view self-supervised objective (VICReg by default). There is no
decoder and no codebook: the encoder output *is* the token, and collapse is
prevented by the loss itself (variance/covariance terms or negatives) instead
of by quantization.

Mirrors ``LitVqVae``'s constructor contract (``encoder`` partial,
``data_sample``, absorbed ``n_classes``) so ``heptokens.train`` and the
bundled Hydra plumbing work unchanged, and keeps the ``encode(batch)[0]``
convention used by monitors and downstream tokenizer transforms.
"""

import logging
from functools import partial
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from lightning import LightningModule

from heptokens.models.coders import Encoder
from heptokens.models.losses import vicreg_loss
from heptokens.models.utils import ScheduledOptimiserMixin, compute_embedding_collapse

log = logging.getLogger(__name__)


class LitContrastiveTokenizer(ScheduledOptimiserMixin, LightningModule):
    """Two-view contrastive tokenizer producing continuous per-constituent tokens.

    Args:
        encoder: Partial of an encoder class (e.g. ``heptokens.models.coders.Encoder``
            wrapping a ``Transformer``); finalised with
            ``(input_dim=..., output_dim=latent_dim)``.
        loss_fn: Callable ``(za, zb) -> (loss, components)`` on flattened valid
            embeddings ``[M, proj_dim]``. Defaults to VICReg.
        latent_dim: Dimension of the released token (encoder output).
        proj_dim: Dimension of the projector output used only by the loss.
        proj_hidden: Hidden width of the projector MLP.
        augmentation: Optional ``ConstituentAugmentation``; when set and the
            batch carries no precomputed views, two views are generated on the
            fly (on-device). Alternatively supply views via the datamodule
            ``transforms`` hook (``TwoViewTransform``).
        view_keys: Batch keys holding precomputed views.
        data_sample: Sample tensor used to infer the input feature dimension.
        **kwargs: Absorbs ``n_classes`` and other trainer-injected extras.
    """

    def __init__(
        self,
        encoder: Encoder = None,
        loss_fn=None,
        latent_dim: int = 16,
        proj_dim: int = 32,
        proj_hidden: int = 64,
        learning_rate: float = 1e-3,
        optimizer=None,
        scheduler=None,
        augmentation=None,
        view_keys: Tuple[str, str] = ("csts_view0", "csts_view1"),
        input_key: str = "csts",
        mask_key: str = "mask",
        loss_inputs: str = "two_view",
        raw_key: str = "csts_raw",
        use_projector: bool = True,
        decoder=None,
        recon_weight: float = 0.0,
        contrastive_weight: float = 1.0,
        fsq_levels: Tuple[int, ...] = None,
        fsq_kernel_on: str = "z_e",
        data_sample: torch.Tensor = None,
        input_dim: int | None = None,
        **kwargs,
    ):
        super().__init__()
        # data_sample is a tensor: keep it out of hparams (checkpoint bloat and
        # pathologically slow YAML hparams serialization by loggers). The
        # scalar input_dim is persisted instead so load_from_checkpoint
        # rebuilds the encoder without a data sample.
        self.save_hyperparameters(ignore=["data_sample"])

        self.learning_rate = learning_rate
        self.latent_dim = latent_dim
        self.view_keys = tuple(view_keys)
        self.input_key = input_key
        self.mask_key = mask_key
        self.augmentation = augmentation
        if loss_inputs not in ("two_view", "features"):
            raise ValueError(f"Unknown loss_inputs mode: {loss_inputs!r}")
        self.loss_inputs = loss_inputs
        self.raw_key = raw_key
        self.use_projector = use_projector

        if data_sample is not None:
            input_dim = data_sample.shape[-1]
        elif input_dim is None:
            input_dim = 3
        self.hparams["input_dim"] = input_dim

        if encoder is None:
            encoder = Encoder
        self.encoder = encoder(input_dim=input_dim, output_dim=latent_dim)

        # Optional FSQ bottleneck (codebook-free discrete token): the released
        # token becomes the rounded grid point z_q; the physics term is applied
        # to the smooth bounded latent z_e (default) or to z_q via the STE.
        self.fsq = None
        self.fsq_kernel_on = fsq_kernel_on
        self.contrastive_weight = contrastive_weight
        if fsq_levels is not None:
            from heptokens.models.fsq import FSQ

            if len(fsq_levels) != latent_dim:
                raise ValueError(
                    f"len(fsq_levels)={len(fsq_levels)} must equal latent_dim={latent_dim}"
                )
            if loss_inputs != "features":
                raise ValueError("The FSQ bottleneck supports only loss_inputs='features'.")
            if fsq_kernel_on not in ("z_e", "z_q"):
                raise ValueError(f"Unknown fsq_kernel_on: {fsq_kernel_on!r}")
            self.fsq = FSQ(fsq_levels)

        # Optional hybrid objective: a decoder adds a reconstruction term so
        # the token preserves particle information alongside the metric the
        # contrastive loss shapes (kernel organizes, reconstruction retains).
        self.decoder = None
        self.recon_weight = recon_weight
        if decoder is not None and recon_weight > 0:
            self.decoder = decoder(input_dim=latent_dim, output_dim=input_dim)

        self.projector = nn.Sequential(
            nn.Linear(latent_dim, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, proj_dim),
        )

        self.loss_fn = loss_fn if callable(loss_fn) else partial(vicreg_loss)

    # ------------------------------------------------------------------ api
    def encode(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Encode a batch to continuous tokens.

        Returns ``(z, None, None)`` with ``z`` of shape
        ``[batch, n_csts, latent_dim]`` — same tuple convention as
        ``LitVqVae.encode`` so monitors and tokenizer transforms can call
        ``encode(batch)[0]`` on either model. With an FSQ bottleneck the
        released token is the quantized grid point.
        """
        z = self.encoder(batch)
        if self.fsq is not None:
            z = self.fsq(z)[0]
        return z, None, None

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encode(batch)[0]

    # ------------------------------------------------------------ internals
    def _encode_view(self, view: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        view_batch = {self.encoder.input_key: view}
        if mask is not None:
            view_batch[self.encoder.mask_key] = mask
        return self.encoder(view_batch)

    def _get_views(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        mask = batch.get(self.mask_key)
        k0, k1 = self.view_keys
        if k0 in batch and k1 in batch:
            return batch[k0], batch[k1], mask
        if self.augmentation is not None:
            csts = batch[self.input_key]
            return self.augmentation(csts, mask), self.augmentation(csts, mask), mask
        raise KeyError(
            f"Batch has no views ({k0!r}/{k1!r}) and no augmentation is configured. "
            "Either add a TwoViewTransform to the datamodule transforms or pass "
            "`augmentation=` to the model."
        )

    def _features_step(
        self, batch: Dict[str, torch.Tensor], prefix: str
    ) -> Optional[torch.Tensor]:
        """Single-view, feature-conditioned objective (particle-library mode).

        No augmented views: the loss receives the embeddings plus the matching
        *physical-unit* features (``raw_key``, stashed by ``keep_raw_batch``
        before preprocessing; falls back to the model input) and builds its
        pair structure from physics — e.g. ``physics_kernel_loss``.
        """
        z = self.encoder(batch)
        mask = batch.get(self.mask_key)
        feats = batch.get(self.raw_key, batch[self.input_key])

        z_token, z_kernel = z, z
        if self.fsq is not None:
            z_q, z_e = self.fsq(z)
            z_token = z_q
            z_kernel = z_e if self.fsq_kernel_on == "z_e" else z_q

        if mask is not None:
            z_flat, f_flat = z_kernel[mask], feats[mask]
        else:
            z_flat = z_kernel.reshape(-1, z_kernel.shape[-1])
            f_flat = feats.reshape(-1, feats.shape[-1])

        if z_flat.shape[0] < 3:
            log.warning("Batch with <3 valid constituents; skipping step.")
            return None

        if self.use_projector:
            z_flat = self.projector(z_flat)

        if self.contrastive_weight != 0:
            loss, components = self.loss_fn(z_flat, f_flat)
            loss = self.contrastive_weight * loss
        else:
            loss, components = z_flat.new_zeros(()), {}
        if self.decoder is not None:
            recon = self.decoder.compute_loss(z_token, batch)
            components = {**components, "recon_loss": recon}
            loss = loss + self.recon_weight * recon
        self.log(f"{prefix}/total_loss", loss, prog_bar=True)
        for name, value in components.items():
            self.log(f"{prefix}/{name}", value)
        return loss

    def _shared_step(self, batch: Dict[str, torch.Tensor], prefix: str) -> Optional[torch.Tensor]:
        if self.loss_inputs == "features":
            return self._features_step(batch, prefix)

        view0, view1, mask = self._get_views(batch)
        z0 = self._encode_view(view0, mask)
        z1 = self._encode_view(view1, mask)

        if mask is not None:
            za, zb = z0[mask], z1[mask]
        else:
            za = z0.reshape(-1, z0.shape[-1])
            zb = z1.reshape(-1, z1.shape[-1])

        if za.shape[0] < 2:
            log.warning("Batch with <2 valid constituents; skipping step.")
            return None

        loss, components = self.loss_fn(self.projector(za), self.projector(zb))

        self.log(f"{prefix}/total_loss", loss, prog_bar=True)
        for name, value in components.items():
            self.log(f"{prefix}/{name}", value)

        return loss

    # ------------------------------------------------------------ lightning
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, "val")

        # Collapse diagnostics on the *token* (pre-projector) embedding.
        if batch_idx == 0:
            with torch.no_grad():
                z = self.encode(batch)[0]
                mask = batch.get(self.mask_key)
                flat = z[mask] if mask is not None else z.reshape(-1, z.shape[-1])
                if flat.shape[0] > 2:
                    for name, value in compute_embedding_collapse(flat).items():
                        self.log(f"val/{name}", value)

        return loss

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Return continuous tokens (and passthrough labels/eventNumber if present)."""
        result = {"latents": self.encode(batch)[0]}
        if "labels" in batch:
            result["labels"] = batch["labels"]
        if "eventNumber" in batch:
            result["eventNumber"] = batch["eventNumber"]
        if "weights" in batch:
            result["weights"] = batch["weights"]
        return result
