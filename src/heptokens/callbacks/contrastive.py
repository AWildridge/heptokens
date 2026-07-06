"""Validation-time monitoring for continuous (contrastive) tokenizers."""

import logging

import torch as T
from lightning.pytorch.callbacks import Callback

from heptokens.models.utils import (
    compute_alignment_uniformity,
    compute_embedding_collapse,
    compute_knn_pid_purity,
)

log = logging.getLogger(__name__)


class ContrastiveMonitor(Callback):
    """Log representation-quality metrics for continuous token embeddings.

    The continuous analog of monitoring codebook utilization for the VQ-VAE:
    collapse diagnostics (effective rank, per-dim variance, redundancy),
    hypersphere uniformity, and — when a PID column is configured — kNN PID
    purity as a physics-faithfulness probe.

    Args:
        input_key: Batch key of the constituent tensor.
        mask_key: Batch key of the validity mask (``None`` for maskless data).
        pid_index: Feature column holding the (categorical) PID, for kNN
            purity. ``None`` disables the purity metric.
        knn_k: Number of neighbours for the purity metric.
        log_every_n_epochs: Compute every N validation epochs.
        max_batches: Number of validation batches to process per epoch.
    """

    def __init__(
        self,
        input_key: str = "csts",
        mask_key: str | None = "mask",
        pid_index: int | None = None,
        knn_k: int = 8,
        log_every_n_epochs: int = 1,
        max_batches: int = 1,
    ):
        super().__init__()
        self.input_key = input_key
        self.mask_key = mask_key
        self.pid_index = pid_index
        self.knn_k = knn_k
        self.log_every_n_epochs = log_every_n_epochs
        self.max_batches = max_batches

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.sanity_checking:
            return
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
        if batch_idx >= self.max_batches:
            return

        with T.no_grad():
            z = pl_module.encode(batch)[0]

        mask = batch.get(self.mask_key) if self.mask_key else None
        flat = z[mask] if mask is not None else z.reshape(-1, z.shape[-1])
        if flat.shape[0] < 3:
            return

        metrics = compute_embedding_collapse(flat)
        metrics.update(compute_alignment_uniformity(flat))

        if self.pid_index is not None:
            csts = batch[self.input_key]
            pid = csts[..., self.pid_index]
            pid_flat = pid[mask] if mask is not None else pid.reshape(-1)
            metrics["knn_pid_purity"] = compute_knn_pid_purity(flat, pid_flat, k=self.knn_k)

        for name, value in metrics.items():
            pl_module.log(f"val/{name}", value, prog_bar=False)
