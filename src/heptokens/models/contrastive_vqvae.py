"""Contrastive-VQ tokenizer: a codebook shaped by the particle-library criterion.

``LitContrastiveVqVae`` is a VQ-VAE whose released token is a stack of codebook
indices (genuine compression, ``num_quantizers * log2(codebook_size)`` bits),
but whose codebook is *organized* by the graded physics-kernel objective so that
nearby codes mean physically similar particles. It is ``LitVqVae`` plus one term
— the physics kernel applied to the quantized embedding ``z_q``:

    total = kernel_weight * kernel(z_q, feats)          # library geometry
          + reconstruction_weight * recon(z_q, batch)   # information / cross-check
          + commitment                                  # VQ codebook commitment

Only the kernel term distinguishes it from a plain ``LitVqVae`` at the same rate,
so a matched-config comparison isolates the value of criterion-shaping the
codebook. The kernel operates on ``z_q`` (the released code geometry); the
straight-through estimator carries its gradient to the encoder, and the codebook
EMA then follows the now criterion-organized encoder clusters.
"""

import logging
from typing import Dict, Optional, Tuple

import torch

from heptokens.models.utils import compute_codebook_utilization
from heptokens.models.vq_vae import LitVqVae

log = logging.getLogger(__name__)


class LitContrastiveVqVae(LitVqVae):
    """VQ-VAE whose codebook is shaped by the physics-kernel contrastive loss.

    Args (beyond ``LitVqVae``):
        loss_fn: Callable ``(z, feats) -> (loss, components)`` on flattened valid
            quantized embeddings + physical-unit features (e.g.
            ``physics_kernel_loss``). ``None`` degrades to a plain VQ-VAE — and
            is the state after ``load_from_checkpoint``, which is fine because
            inference only calls ``encode``.
        kernel_weight: Weight of the kernel term.
        raw_key: Batch key holding physical-unit features (stashed by
            ``keep_raw_batch`` before preprocessing); falls back to the model
            input when absent.
    """

    def __init__(
        self,
        *args,
        loss_fn=None,
        kernel_weight: float = 1.0,
        kernel_on: str = "z_q",
        raw_key: str = "csts_raw",
        input_key: str = "csts",
        mask_key: str = "mask",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Training-only attributes; deliberately not persisted in hparams. The
        # parent already saves the encoder + codebook config, which is all
        # ``encode`` needs at load time (the kernel is never run at inference).
        self.loss_fn = loss_fn
        self.kernel_weight = kernel_weight
        # Where the kernel organizes: the quantized code ``z_q`` (shapes the
        # released geometry directly, but the gradient flows through the
        # straight-through estimator) or the continuous pre-quantization
        # ``z_e`` (a clean gradient — the kernel organizes the manifold and the
        # VQ then tiles it).
        if kernel_on not in ("z_q", "z_e"):
            raise ValueError(f"kernel_on must be 'z_q' or 'z_e', got {kernel_on!r}")
        self.kernel_on = kernel_on
        self.raw_key = raw_key
        self.input_key = input_key
        self.mask_key = mask_key

    def _kernel_term(
        self, z_q: torch.Tensor, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], dict]:
        if self.loss_fn is None:
            return None, {}
        mask = batch.get(self.mask_key)
        feats = batch.get(self.raw_key, batch[self.input_key])
        if mask is not None:
            zq_flat, f_flat = z_q[mask], feats[mask]
        else:
            zq_flat = z_q.reshape(-1, z_q.shape[-1])
            f_flat = feats.reshape(-1, feats.shape[-1])
        if zq_flat.shape[0] < 3:
            log.warning("Batch with <3 valid constituents; skipping kernel term.")
            return None, {}
        return self.loss_fn(zq_flat, f_flat)

    def _shared_step(
        self, batch: Dict[str, torch.Tensor], prefix: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_q, indices, commit_loss = self.encode(batch)
        recon_loss = self.decoder.compute_loss(z_q, batch)
        kernel_input = z_q if self.kernel_on == "z_q" else self.encoder(batch)
        kernel_loss, components = self._kernel_term(kernel_input, batch)

        total = self.reconstruction_weight * recon_loss + commit_loss
        if kernel_loss is not None:
            total = total + self.kernel_weight * kernel_loss

        self.log(f"{prefix}/total_loss", total, prog_bar=True)
        self.log(f"{prefix}/recon_loss", recon_loss, prog_bar=True)
        self.log(f"{prefix}/commit_loss", commit_loss, prog_bar=True)
        # kernel_loss is carried inside ``components`` (avoid a double-log).
        for name, value in components.items():
            self.log(f"{prefix}/{name}", value, prog_bar=(name == "kernel_loss"))

        utils = compute_codebook_utilization(
            indices, batch.get(self.mask_key), self.codebook_size
        )
        for q, u in enumerate(utils):
            self.log(f"{prefix}/codebook_util_q{q}", u)
        self.log(f"{prefix}/codebook_util_avg", sum(utils) / len(utils))
        return total, indices

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")[0]

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        total, indices = self._shared_step(batch, "val")
        return {"val_loss": total, "indices": indices}
