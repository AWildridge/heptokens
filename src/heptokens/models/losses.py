"""Self-supervised losses for continuous per-constituent tokenization.

All losses operate on flattened valid-constituent embeddings ``[M, D]`` (the
caller applies the boolean mask before flattening) and return
``(total_loss, components)`` where ``components`` is a dict of scalar tensors
for logging.

Default choice is VICReg: it needs no negative pairs, which matters at
constituent level where in-batch negatives are physically unsound (two soft
pions from different jets are near-identical particles and must not be
repelled). InfoNCE is provided as a fallback; the physics-kernel SupCon makes
physically similar constituents positives *by construction*.
"""

import torch as T
import torch.nn.functional as F


def vicreg_loss(
    za: T.Tensor,
    zb: T.Tensor,
    sim_weight: float = 25.0,
    var_weight: float = 25.0,
    cov_weight: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> tuple[T.Tensor, dict]:
    """VICReg (arXiv:2105.04906) on matched embeddings ``[M, D]``.

    - invariance: MSE between the two views of the same constituent
    - variance: hinge keeping each dimension's std above ``gamma``
      (the anti-collapse term)
    - covariance: penalises off-diagonal covariance (redundancy reduction)
    """
    sim = F.mse_loss(za, zb)

    def _variance(z: T.Tensor) -> T.Tensor:
        std = (z.var(dim=0) + eps).sqrt()
        return F.relu(gamma - std).mean()

    var = 0.5 * (_variance(za) + _variance(zb))

    def _covariance(z: T.Tensor) -> T.Tensor:
        z = z - z.mean(dim=0)
        m = max(z.shape[0] - 1, 1)
        cov = (z.T @ z) / m
        off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        return off_diag / z.shape[1]

    cov = 0.5 * (_covariance(za) + _covariance(zb))

    total = sim_weight * sim + var_weight * var + cov_weight * cov
    return total, {"sim_loss": sim, "var_loss": var, "cov_loss": cov}


def info_nce_loss(
    za: T.Tensor,
    zb: T.Tensor,
    temperature: float = 0.1,
    max_pairs: int = 2048,
) -> tuple[T.Tensor, dict]:
    """Symmetric InfoNCE / NT-Xent on matched embeddings ``[M, D]``.

    Rows are subsampled to ``max_pairs`` to bound the O(M^2) logits matrix
    (constituent counts per batch are large).

    WARNING: in-batch negatives at constituent level repel physically identical
    particles from different jets. Prefer ``vicreg_loss`` or
    ``supcon_physics_loss`` for production; this is kept as a baseline.
    """
    m = za.shape[0]
    if m > max_pairs:
        idx = T.randperm(m, device=za.device)[:max_pairs]
        za, zb = za[idx], zb[idx]

    za = F.normalize(za, dim=-1)
    zb = F.normalize(zb, dim=-1)
    logits = za @ zb.T / temperature
    labels = T.arange(za.shape[0], device=za.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    return loss, {"nce_loss": loss}


def supcon_physics_loss(
    za: T.Tensor,
    feats: T.Tensor,
    zb: T.Tensor | None = None,
    pid_index: int | None = None,
    pt_index: int = 0,
    eta_index: int = 1,
    phi_index: int = 2,
    dpt_frac: float = 0.25,
    dr_max: float = 0.2,
    temperature: float = 0.1,
    max_points: int = 1024,
    eps: float = 1e-6,
) -> tuple[T.Tensor, dict]:
    """Supervised-contrastive loss with physics-defined positives.

    Positives for anchor ``i`` are constituents ``j`` with the same PID (when
    ``pid_index`` is given) whose kinematics lie in the neighbourhood
    ``|pt_i - pt_j| / max(pt_i, pt_j) <= dpt_frac`` and ``dR(i, j) <= dr_max``.
    This encodes the hypothesis that same-PID, similar-kinematics particles
    should embed to the same region — and makes physically identical
    constituents positives *by construction* (no false negatives).

    Args:
        za: Embeddings ``[M, D]`` of the valid constituents.
        feats: Matching *physical* features ``[M, F]`` used to define
            positives (pre-preprocessing values).
        zb: Optional second-view embeddings; when given, the positive set of
            anchor ``i`` always includes its own second view (SupCon
            "multiview" form) and candidates come from both views.
        pid_index: Column of ``feats`` holding the (categorical) PID.
        pt_index / eta_index / phi_index: Kinematic columns of ``feats``.
        dpt_frac: Relative-pT neighbourhood size.
        dr_max: Delta-R neighbourhood size.
        temperature: Softmax temperature.
        max_points: Subsample cap (pairwise matrices are O(M^2)).
    """
    m = za.shape[0]
    if m > max_points:
        idx = T.randperm(m, device=za.device)[:max_points]
        za, feats = za[idx], feats[idx]
        if zb is not None:
            zb = zb[idx]

    if zb is not None:
        z = T.cat([za, zb], dim=0)
        feats = T.cat([feats, feats], dim=0)
    else:
        z = za

    n = z.shape[0]
    if n < 2:
        zero = z.sum() * 0.0
        return zero, {"supcon_loss": zero, "positive_frac": zero}

    pt = feats[:, pt_index]
    eta = feats[:, eta_index]
    phi = feats[:, phi_index]

    dpt = (pt.unsqueeze(0) - pt.unsqueeze(1)).abs()
    pt_scale = T.maximum(pt.unsqueeze(0), pt.unsqueeze(1)).clamp(min=eps)
    same_pt = dpt / pt_scale <= dpt_frac

    deta = eta.unsqueeze(0) - eta.unsqueeze(1)
    dphi = T.remainder(phi.unsqueeze(0) - phi.unsqueeze(1) + T.pi, 2 * T.pi) - T.pi
    same_dir = (deta.pow(2) + dphi.pow(2)).sqrt() <= dr_max

    positives = same_pt & same_dir
    if pid_index is not None:
        pid = feats[:, pid_index]
        positives = positives & (pid.unsqueeze(0) == pid.unsqueeze(1))

    if zb is not None:
        # A constituent's own second view is always a positive.
        half = n // 2
        pair = T.arange(half, device=z.device)
        positives[pair, pair + half] = True
        positives[pair + half, pair] = True

    self_mask = T.eye(n, dtype=T.bool, device=z.device)
    positives = positives & ~self_mask

    z = F.normalize(z, dim=-1)
    logits = z @ z.T / temperature
    logits = logits.masked_fill(self_mask, float("-inf"))
    log_prob = logits - T.logsumexp(logits, dim=1, keepdim=True)

    n_pos = positives.sum(dim=1)
    has_pos = n_pos > 0
    if not has_pos.any():
        zero = z.sum() * 0.0
        return zero, {"supcon_loss": zero, "positive_frac": zero}

    # where() (not multiply): log_prob holds -inf at self-positions and -inf * 0 = nan.
    pos_log_prob = T.where(positives, log_prob, T.zeros_like(log_prob))
    mean_log_prob_pos = pos_log_prob.sum(dim=1)[has_pos] / n_pos[has_pos]
    loss = -mean_log_prob_pos.mean()
    positive_frac = has_pos.float().mean()
    return loss, {"supcon_loss": loss, "positive_frac": positive_frac}
