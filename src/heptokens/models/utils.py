import logging
import math
from typing import Sequence

import torch as T
from lightning import LightningModule
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


class ScheduledOptimiserMixin:
    """Mixin providing ``configure_optimizers`` via Hydra-injected partials.

    Models using this mixin may accept ``optimizer`` and ``scheduler`` as
    ``functools.partial`` objects (typically via Hydra ``_partial_: true``).
    If no optimizer partial is provided, falls back to Adam with
    ``self.learning_rate``.  If no scheduler partial is provided, no
    scheduler is used.
    """

    def configure_optimizers(self):
        hp = getattr(self, "hparams", {})

        # --- optimizer ---
        opt_partial = hp.get("optimizer", None)
        if callable(opt_partial):
            opt = opt_partial(self.parameters())
        else:
            lr = self.__dict__.get("learning_rate") or hp.get("learning_rate", 1e-3)
            opt = T.optim.Adam(self.parameters(), lr=lr)

        # --- scheduler (optional) ---
        sched_partial = hp.get("scheduler", None)
        if callable(sched_partial):
            sched = sched_partial(optimizer=opt, model=self)
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
            }

        return opt


def get_max_steps(model: LightningModule) -> int:
    """Get the maximum number of steps from the model trainer."""
    try:
        logger.info("Attempting to get the max steps from the model trainer")
        max_steps = model.trainer.max_steps
        if max_steps < 1:
            steps_per_epoch = len(model.trainer.datamodule.train_dataloader())
            max_epochs = model.trainer.max_epochs
            max_steps = steps_per_epoch * max_epochs
        logger.info(f"Success:  max_steps = {max_steps}")
    except Exception as e:
        logger.info(f"Failed to get max steps from the model trainer: {e}")
        max_steps = 0
    return max_steps


def warmup_cosine_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
    model: LightningModule | None = None,
    max_epochs: int = -1,
) -> T.optim.lr_scheduler.SequentialLR:
    """Linear warmup followed by cosine annealing, configured in epochs.

    If ``max_epochs`` is -1 (default), it is read from ``model.trainer``.
    """
    if max_epochs < 1 and model is not None:
        max_epochs = model.trainer.max_epochs
    if max_epochs < 1:
        raise ValueError("max_epochs must be positive (set it or pass a model with a trainer).")

    warmup = min(warmup_epochs, max_epochs)
    warmup_sched = T.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-2,
        total_iters=warmup,
    )
    cosine_sched = T.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(max_epochs - warmup, 1),
        eta_min=min_lr,
    )
    return T.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup],
    )


def linear_warmup_cosine_decay(
    optimizer: Optimizer,
    warmup_steps: int = 1000,
    total_steps: int = 10000,
    final_factor: float = 5e-2,
    init_factor: float = 1e-5,
    model: LightningModule | None = None,
) -> LambdaLR:
    """Return a scheduler with a linear warmup and a cosine decay."""
    # Attempt to get the max steps from the model trainer
    if total_steps == -1 and model is not None:
        total_steps = get_max_steps(model)

    warmup_steps = max(1, warmup_steps)  # Avoid division by zero
    assert 0 < final_factor < 1, "Final factor must be less than 1"
    assert 0 < init_factor < 1, "Initial factor must be less than 1"
    assert 0 < warmup_steps < total_steps, "Total steps must be greater than warmup"

    def fn(x: int) -> float:
        if x <= warmup_steps:
            return init_factor + x * (1 - init_factor) / warmup_steps
        if x >= total_steps:
            return final_factor
        t = (x - warmup_steps) / (total_steps - warmup_steps) * math.pi
        return (1 + math.cos(t)) * (1 - final_factor) / 2 + final_factor

    return LambdaLR(optimizer, fn)


def compute_codebook_utilization(
    indices: T.Tensor,
    mask: T.Tensor | None,
    codebook_size: int,
) -> Sequence[float]:
    """Compute per-quantizer codebook utilization.

    Args:
        indices: [batch_size, n_csts, num_quantizers] with -1 for masked.
        mask: [batch_size, n_csts] boolean mask, or None for maskless modalities.
        codebook_size: Number of codes in each codebook.

    Returns:
        List of utilization fractions, one per quantizer.
    """
    if mask is not None:
        valid_indices = indices[mask]  # [n_valid, num_quantizers]
    else:
        valid_indices = indices.reshape(-1, indices.shape[-1])  # all positions valid
    utilizations = []
    for q in range(valid_indices.shape[-1]):
        unique_codes = valid_indices[:, q].unique().numel()
        utilizations.append(unique_codes / codebook_size)
    return utilizations


def compute_embedding_collapse(z: T.Tensor, eps: float = 1e-8) -> dict:
    """Collapse diagnostics for continuous embeddings — the analog of
    ``compute_codebook_utilization`` for non-quantized tokens.

    Args:
        z: Flattened valid embeddings [n_points, dim].

    Returns:
        dict with:
        - ``std_mean`` / ``std_min``: per-dimension std statistics (0 ⇒ collapse)
        - ``active_dims``: fraction of dimensions with std > 0.01
        - ``effective_rank``: exp of the entropy of the normalised singular
          values (1 ⇒ all variance in one direction)
        - ``corr_offdiag``: mean |off-diagonal| of the correlation matrix
          (1 ⇒ fully redundant dimensions)
    """
    if z.shape[0] > 4096:
        idx = T.randperm(z.shape[0], device=z.device)[:4096]
        z = z[idx]
    z = z.float()
    std = z.std(dim=0)

    centered = z - z.mean(dim=0)
    svals = T.linalg.svdvals(centered)
    p = svals.pow(2) / (svals.pow(2).sum() + eps)
    effective_rank = T.exp(-(p * (p + eps).log()).sum())

    denom = std.clamp(min=eps)
    corr = (centered.T @ centered) / max(z.shape[0] - 1, 1)
    corr = corr / (denom.unsqueeze(0) * denom.unsqueeze(1))
    n_dim = corr.shape[0]
    off = (corr.abs().sum() - corr.diagonal().abs().sum()) / max(n_dim * (n_dim - 1), 1)

    return {
        "std_mean": std.mean(),
        "std_min": std.min(),
        "active_dims": (std > 0.01).float().mean(),
        "effective_rank": effective_rank,
        "corr_offdiag": off,
    }


def compute_alignment_uniformity(
    za: T.Tensor,
    zb: T.Tensor | None = None,
    t: float = 2.0,
    max_points: int = 2048,
) -> dict:
    """Alignment & uniformity metrics (Wang & Isola, arXiv:2005.10242).

    Args:
        za: Embeddings [n_points, dim] (normalised internally).
        zb: Optional matched second-view embeddings; enables the alignment
            metric (mean squared distance between positive pairs).
        t: Uniformity temperature.

    Returns:
        dict with ``uniformity`` (lower = more uniform on the hypersphere)
        and, when ``zb`` is given, ``alignment`` (lower = better).
    """
    if za.shape[0] > max_points:
        idx = T.randperm(za.shape[0], device=za.device)[:max_points]
        za = za[idx]
        zb = zb[idx] if zb is not None else None

    za_n = T.nn.functional.normalize(za.float(), dim=-1)
    sq_dists = T.cdist(za_n, za_n).pow(2)
    n = za_n.shape[0]
    off_diag = ~T.eye(n, dtype=T.bool, device=za_n.device)
    uniformity = T.log(T.exp(-t * sq_dists[off_diag]).mean())

    out = {"uniformity": uniformity}
    if zb is not None:
        zb_n = T.nn.functional.normalize(zb.float(), dim=-1)
        out["alignment"] = (za_n - zb_n).pow(2).sum(dim=-1).mean()
    return out


def compute_knn_pid_purity(
    z: T.Tensor,
    pid: T.Tensor,
    k: int = 8,
    max_points: int = 2048,
) -> T.Tensor:
    """Fraction of k-nearest neighbours sharing the anchor's PID.

    A physics-faithfulness probe: high purity means the embedding metric
    groups same-species particles together.

    Args:
        z: Flattened valid embeddings [n_points, dim].
        pid: Matching categorical PID values [n_points].
        k: Number of neighbours (excluding the anchor itself).
    """
    if z.shape[0] > max_points:
        idx = T.randperm(z.shape[0], device=z.device)[:max_points]
        z, pid = z[idx], pid[idx]

    n = z.shape[0]
    k = min(k, n - 1)
    if k < 1:
        return T.tensor(float("nan"), device=z.device)

    dists = T.cdist(z.float(), z.float())
    dists.fill_diagonal_(float("inf"))
    knn_idx = dists.topk(k, largest=False).indices  # [n, k]
    same = pid[knn_idx] == pid.unsqueeze(1)
    return same.float().mean()


def compute_retrieval_precision(
    z: T.Tensor,
    feats: T.Tensor,
    pt_index: int = 0,
    eta_index: int = 1,
    phi_index: int = 2,
    pid_index: int | None = None,
    charge_index: int | None = None,
    dpt_frac: float = 0.1,
    deta_max: float | None = 0.1,
    dphi_max: float | None = 0.1,
    dr_max: float | None = None,
    k: int = 8,
    max_points: int = 2048,
    eps: float = 1e-6,
) -> dict:
    """Library quality: do nearest neighbours satisfy the particle criterion?

    For each anchor, the fraction of its k nearest embedding neighbours that
    satisfy the *hard* particle-library criterion (same PID, same charge,
    relative ΔpT ≤ ``dpt_frac``, same detector region). Reported alongside the
    criterion's density among all pairs — the random-retrieval baseline.

    Args:
        z: Flattened valid embeddings [n_points, dim].
        feats: Matching physical-unit features [n_points, n_feats].
        deta_max/dphi_max: Per-coordinate region windows; set ``dr_max``
            instead for a cone criterion.

    Returns:
        dict with ``precision_at_k`` and ``criterion_density``.
    """
    if z.shape[0] > max_points:
        idx = T.randperm(z.shape[0], device=z.device)[:max_points]
        z, feats = z[idx], feats[idx]

    n = z.shape[0]
    k = min(k, n - 1)
    if k < 1:
        nan = T.tensor(float("nan"), device=z.device)
        return {"precision_at_k": nan, "criterion_density": nan}

    pt = feats[:, pt_index]
    dpt = (pt.unsqueeze(0) - pt.unsqueeze(1)).abs()
    pt_scale = T.maximum(pt.unsqueeze(0), pt.unsqueeze(1)).clamp(min=eps)
    crit = dpt / pt_scale <= dpt_frac

    deta = feats[:, eta_index].unsqueeze(0) - feats[:, eta_index].unsqueeze(1)
    dphi = feats[:, phi_index].unsqueeze(0) - feats[:, phi_index].unsqueeze(1)
    dphi = T.remainder(dphi + T.pi, 2 * T.pi) - T.pi
    if dr_max is not None:
        crit = crit & ((deta.pow(2) + dphi.pow(2)).sqrt() <= dr_max)
    else:
        crit = crit & (deta.abs() <= deta_max) & (dphi.abs() <= dphi_max)

    if pid_index is not None:
        pid = feats[:, pid_index]
        crit = crit & (pid.unsqueeze(0) == pid.unsqueeze(1))
    if charge_index is not None:
        charge = feats[:, charge_index].round()
        crit = crit & (charge.unsqueeze(0) == charge.unsqueeze(1))
    crit.fill_diagonal_(False)

    dists = T.cdist(z.float(), z.float())
    dists.fill_diagonal_(float("inf"))
    knn_idx = dists.topk(k, largest=False).indices  # [n, k]
    hits = T.gather(crit, 1, knn_idx).float().mean()

    density = crit.float().sum() / max(n * (n - 1), 1)
    return {"precision_at_k": hits, "criterion_density": density}


class JetBackbone(nn.Module):
    """Generalised backbone for the jet models.

    Simply wraps the constituent embedding and encoder together in a single module.
    Easy for saving and loading using the pickle module.
    """

    def __init__(
        self,
        cst_emb: nn.Module,
        jet_emb: nn.Module,
        encoder,  # type: Transformer  # TODO: write this class!
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.cst_emb = cst_emb
        self.jet_emb = jet_emb
        self.encoder = encoder
        self.causal = causal
        self.dim = encoder.dim
        self.outp_dim = encoder.outp_dim

    def forward(
        self,
        csts: T.Tensor,
        mask: T.Tensor,
        jets: T.Tensor,
    ) -> T.Tensor:
        """Pass through the complete backbone."""
        csts = self.cst_emb(csts)
        jets = self.jet_emb(jets)
        x = self.encoder(csts, mask=mask, ctxt=jets, causal=self.causal)
        new_mask = self.encoder.get_combined_mask(mask)  # Registers
        return x, new_mask
