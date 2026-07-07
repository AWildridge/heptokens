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


def _wrap_dphi(dphi: T.Tensor) -> T.Tensor:
    """Wrap azimuthal differences to [-pi, pi)."""
    return T.remainder(dphi + T.pi, 2 * T.pi) - T.pi


def physics_kernel_loss(
    z: T.Tensor,
    feats: T.Tensor,
    pt_index: int = 0,
    eta_index: int = 1,
    phi_index: int = 2,
    pid_index: int | None = None,
    charge_index: int | None = None,
    dpt_frac: float = 0.1,
    deta_max: float = 0.1,
    dphi_max: float = 0.1,
    dr_max: float = 0.1,
    region: str = "coords",
    temperature: float = 0.1,
    max_points: int = 1024,
    var_weight: float = 0.0,
    cov_weight: float = 0.0,
    eps: float = 1e-6,
) -> tuple[T.Tensor, dict]:
    """Graded physics-kernel contrastive loss — the particle-library objective.

    The target geometry is stated in physics variables: two constituents are
    similar to the degree that they have the **same PID**, the **same charge**,
    similar pT (relative), and head to the same detector region. Hard gates on
    the categorical properties, a smooth Gaussian kernel on the kinematics —
    no cliff at an arbitrary window edge, no augmentations needed, and
    positives are found across *all* jets in the batch (the particle library).

        s_ij = [pid_i == pid_j] * [q_i == q_j]
               * exp(-1/2 (dpt%_ij / dpt_frac)^2) * exp(-1/2 region_ij^2)

    with ``region^2 = (deta/deta_max)^2 + (dphi/dphi_max)^2`` for
    ``region="coords"`` or ``(dR/dr_max)^2`` for ``region="cone"``.

    The loss is soft-target contrastive cross-entropy: each kernel row (self
    excluded, normalised) is the target distribution for the softmax over
    embedding similarities — non-positives are repelled through the softmax
    denominator ("the rest should be away"). Anchors with no kernel mass in
    the sampled set are dropped; their complement is logged as
    ``positive_frac``.

    Args:
        z: Embeddings ``[M, D]`` of valid constituents (normalised internally).
        feats: Matching *physical-unit* features ``[M, F]`` — carry them past
            preprocessing with the ``keep_raw_batch`` transform.
        pid_index / charge_index: Columns of ``feats`` holding PDG id and
            charge; ``None`` disables that gate (e.g. datasets without PID).
        region: ``"coords"`` (per-coordinate windows) or ``"cone"`` (dR).
        var_weight / cov_weight: Optional VICReg variance/covariance terms on
            ``z`` as extra anti-collapse insurance (defaults off — the softmax
            denominator already prevents collapse).
    """
    m = z.shape[0]
    if m > max_points:
        idx = T.randperm(m, device=z.device)[:max_points]
        z, feats = z[idx], feats[idx]

    n = z.shape[0]
    if n < 3:
        zero = z.sum() * 0.0
        return zero, {"kernel_loss": zero, "positive_frac": zero}

    pt = feats[:, pt_index]
    dpt = (pt.unsqueeze(0) - pt.unsqueeze(1)).abs()
    pt_scale = T.maximum(pt.unsqueeze(0), pt.unsqueeze(1)).clamp(min=eps)
    log_k = -0.5 * (dpt / pt_scale / dpt_frac).pow(2)

    deta = feats[:, eta_index].unsqueeze(0) - feats[:, eta_index].unsqueeze(1)
    dphi = _wrap_dphi(feats[:, phi_index].unsqueeze(0) - feats[:, phi_index].unsqueeze(1))
    if region == "coords":
        log_k = log_k - 0.5 * ((deta / deta_max).pow(2) + (dphi / dphi_max).pow(2))
    elif region == "cone":
        log_k = log_k - 0.5 * (deta.pow(2) + dphi.pow(2)) / dr_max**2
    else:
        raise ValueError(f"Unknown region mode: {region!r}")

    kernel = T.exp(log_k)
    if pid_index is not None:
        pid = feats[:, pid_index]
        kernel = kernel * (pid.unsqueeze(0) == pid.unsqueeze(1))
    if charge_index is not None:
        charge = feats[:, charge_index].round()
        kernel = kernel * (charge.unsqueeze(0) == charge.unsqueeze(1))

    self_mask = T.eye(n, dtype=T.bool, device=z.device)
    kernel = kernel.masked_fill(self_mask, 0.0)

    mass = kernel.sum(dim=1)
    valid = mass > 1e-6
    if not valid.any():
        zero = z.sum() * 0.0
        return zero, {"kernel_loss": zero, "positive_frac": zero}

    z_n = F.normalize(z, dim=-1)
    logits = (z_n @ z_n.T / temperature).masked_fill(self_mask, float("-inf"))
    log_prob = F.log_softmax(logits, dim=1)

    q = kernel[valid] / mass[valid].unsqueeze(1)
    # where() (not multiply): log_prob holds -inf at self-positions and 0 * -inf = nan.
    cross_entropy = -T.where(q > 0, q * log_prob[valid], T.zeros_like(q)).sum(dim=1)
    loss = cross_entropy.mean()

    components = {
        "kernel_loss": loss,
        "positive_frac": valid.float().mean(),
        "kernel_mass_mean": mass[valid].mean(),
    }

    if var_weight > 0 or cov_weight > 0:
        reg, reg_comps = vicreg_loss(
            z, z, sim_weight=0.0, var_weight=var_weight, cov_weight=cov_weight
        )
        loss = loss + reg
        components["var_loss"] = reg_comps["var_loss"]
        components["cov_loss"] = reg_comps["cov_loss"]

    return loss, components


def supcon_physics_loss(
    za: T.Tensor,
    feats: T.Tensor,
    zb: T.Tensor | None = None,
    pid_index: int | None = None,
    charge_index: int | None = None,
    pt_index: int = 0,
    eta_index: int = 1,
    phi_index: int = 2,
    dpt_frac: float = 0.25,
    dr_max: float = 0.2,
    deta_max: float | None = None,
    dphi_max: float | None = None,
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
    dphi = _wrap_dphi(phi.unsqueeze(0) - phi.unsqueeze(1))
    if deta_max is not None and dphi_max is not None:
        # Per-coordinate detector-region windows.
        same_dir = (deta.abs() <= deta_max) & (dphi.abs() <= dphi_max)
    else:
        same_dir = (deta.pow(2) + dphi.pow(2)).sqrt() <= dr_max

    positives = same_pt & same_dir
    if pid_index is not None:
        pid = feats[:, pid_index]
        positives = positives & (pid.unsqueeze(0) == pid.unsqueeze(1))
    if charge_index is not None:
        charge = feats[:, charge_index].round()
        positives = positives & (charge.unsqueeze(0) == charge.unsqueeze(1))

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
