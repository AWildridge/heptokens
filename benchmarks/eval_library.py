"""Library-quality evaluation on AOJ (real CMS data, label-free).

Compares embedding providers on the particle-library criterion
(same PID, same charge, dpT <= 10%, |deta| & |dphi| <= 0.1):

    library            trained graded-kernel tokenizer (coords region)
    library_cone       ablation: dR-cone kernel
    library_hard       ablation: hard-window SupCon
    null_handcrafted   PID one-hot + charge + scaled kinematics — UNTRAINED.
                       The "is learning paying rent" reference.
    raw_kinematics     scaled (pt, eta, phi) only
    v1_contrastive     jets5c-trained transformer-VICReg (context leakage ref)
    v1_vqvae           jets5c-trained VQ z_q

Metrics: retrieval precision@k under the hard criterion, kNN PID purity,
within-PID pT recovery R^2 (PID-shortcut check), effective rank.
Figures: retrieval bars, PID-island PCA, distance-vs-dpT cliff check
(graded vs hard windows).

Run after the AOJ training chain: python eval_library.py
"""

import copy
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from common import FIGURES_DIR, LIBRARY_CRITERION, SEED, TOKENIZER_DIR, make_datamodule, make_transforms
from make_plots import BASE, C_CLASS, INK, INK2, MUTED, savefig  # shared style (rcParams on import)

from heptokens.models.contrastive import LitContrastiveTokenizer
from heptokens.models.contrastive_vqvae import LitContrastiveVqVae
from heptokens.models.utils import (
    compute_embedding_collapse,
    compute_knn_pid_purity,
    compute_retrieval_precision,
)
from heptokens.models.vae import LitVae
from heptokens.models.vq_vae import LitVqVae

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CRIT = dict(LIBRARY_CRITERION, pid_index=3, charge_index=4)
PDG_GROUPS = {211: "pi+-", 22: "gamma", 130: "K_L", 11: "e+-", 13: "mu+-"}


@torch.no_grad()
def collect(n_batches=6, batch_size=1024):
    dm = make_datamodule(batch_size=batch_size, num_workers=4, transforms=None, data="aoj")
    dm.setup()
    preprocess = make_transforms("aoj")["preprocess"]

    raws, scaleds, masks = [], [], []
    for i, raw in enumerate(dm.val_dataloader()):
        if i >= n_batches:
            break
        raw_copy = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in raw.items()}
        scaled = preprocess(copy.deepcopy(raw))
        raws.append(raw_copy["csts"])
        scaleds.append(scaled["csts"])
        masks.append(raw_copy["mask"])
    return (
        torch.cat(raws),
        torch.cat(scaleds),
        torch.cat(masks).bool(),
    )


def load_tokenizer(name, cls=LitContrastiveTokenizer):
    ckpt = TOKENIZER_DIR / name / "checkpoints" / "best.ckpt"
    if not ckpt.exists():
        ckpt = TOKENIZER_DIR / name / "checkpoints" / "last.ckpt"
    if not ckpt.exists():
        return None
    return cls.load_from_checkpoint(ckpt, map_location=DEVICE, weights_only=False).eval()


def seed_runs(base: str):
    """Run dirs for a base arm and its seed variants: base, base_s43, ..."""
    runs = []
    for d in sorted(TOKENIZER_DIR.glob(f"{base}*")):
        suffix = d.name[len(base):]
        if suffix == "" or suffix.startswith("_s"):
            runs.append((base.replace("_aoj", "") + suffix, d.name))
    return runs


def residual_kmeans_quantize(z, n_stages=3, k=512, fit_n=200_000, seed=SEED):
    """Post-hoc residual k-means: the rate-matched library row.

    3 stages x 512 codes = 27 bits/particle — identical rate to the trained
    residual-VQ baseline, applied to *frozen* library tokens (tests the
    "post-hoc compression instead of trained quantization" claim).
    """
    from sklearn.cluster import MiniBatchKMeans

    rng = np.random.default_rng(seed)
    z_np = z.numpy().astype(np.float32)
    fit_idx = rng.choice(len(z_np), min(fit_n, len(z_np)), replace=False)

    residual = z_np.copy()
    quantized = np.zeros_like(z_np)
    for stage in range(n_stages):
        km = MiniBatchKMeans(n_clusters=k, random_state=seed + stage, n_init=3)
        km.fit(residual[fit_idx])
        assign = km.predict(residual)
        quantized += km.cluster_centers_[assign]
        residual -= km.cluster_centers_[assign]
    return torch.from_numpy(quantized)


@torch.no_grad()
def embedding_providers(raw, scaled, mask):
    """Return (providers, extras): flat embeddings on valid constituents plus
    side diagnostics (e.g. VQ codebook utilization)."""
    from heptokens.models.utils import compute_codebook_utilization

    out, extras = {}, {}

    def encode_with(model, csts):
        batch = {"csts": csts.to(DEVICE), "mask": mask.to(DEVICE)}
        return model.encode(batch)[0][mask].cpu()

    def vq_encode(model, csts, chunk=256):
        """Chunked VQ encode -> (z_q on valid constituents, codebook util list).

        Chunked over jets so the ``[N_constituents, codebook_size]`` distance
        matrix stays small — an 8192 codebook over the full pool is a ~10 GB
        spike that crashes the 20 GB MIG slice otherwise. Chunking is exact
        (VQ assignment is per-constituent independent)."""
        zqs, idxs = [], []
        for s in range(0, csts.shape[0], chunk):
            msl = mask[s:s + chunk]
            batch = {"csts": csts[s:s + chunk].to(DEVICE), "mask": msl.to(DEVICE)}
            z_q, idx, _ = model.encode(batch)
            zqs.append(z_q[msl.to(z_q.device)].cpu())
            idxs.append(idx.cpu())
        util = compute_codebook_utilization(
            torch.cat(idxs, dim=0), mask.cpu(), codebook_size=model.codebook_size
        )
        return torch.cat(zqs), util

    def vae_encode(model, csts, chunk=256):
        """Chunked VAE encode -> (posterior mean mu on valid constituents,
        measured KL-rate in bits/particle = KL_nats / ln2). Jet-chunked, which
        is exact for the attention backbone (attention is intra-jet)."""
        mus, kls = [], []
        for s in range(0, csts.shape[0], chunk):
            msl = mask[s:s + chunk]
            batch = {"csts": csts[s:s + chunk].to(DEVICE), "mask": msl.to(DEVICE)}
            mu, log_std = model.encode_stats(batch)
            m = msl.to(mu.device)
            mus.append(mu[m].cpu())
            kl = 0.5 * (mu[m].pow(2) + (2 * log_std[m]).exp() - 1 - 2 * log_std[m]).sum(-1)
            kls.append(kl.cpu())
        kl_bits = float(torch.cat(kls).mean() / np.log(2))
        return torch.cat(mus), kl_bits

    # Library arm + seed variants, and its ablations.
    for provider, run in seed_runs("library_aoj"):
        model = load_tokenizer(run)
        if model is not None:
            out[provider] = encode_with(model, scaled)
    for name in ["library_aoj_cone", "library_aoj_hard"]:
        model = load_tokenizer(name)
        if model is not None:
            out[name.replace("_aoj", "")] = encode_with(model, scaled)

    # Iteration-3 hybrid arms (kernel + reconstruction), any run dir hybrid*.
    for d in sorted(TOKENIZER_DIR.glob("hybrid*")):
        model = load_tokenizer(d.name)
        if model is not None:
            out[d.name.replace("_aoj", "")] = encode_with(model, scaled)

    # Apples-to-apples VQ baseline: same encoder/inputs, RVQ bottleneck,
    # full-particle reconstruction objective. + seed variants.
    for provider, run in seed_runs("vqvae_particle_aoj"):
        model = load_tokenizer(run, cls=LitVqVae)
        if model is not None:
            name = provider.replace("vqvae_particle", "vq_particle")
            out[name], util = vq_encode(model, scaled)
            extras[name] = {f"codebook_util_q{q}": float(v) for q, v in enumerate(util)}

    # Contrastive-VQ: same RVQ rate as vqvae_particle, but the codebook is
    # shaped by the physics kernel (z_q or z_e variants). Matched config ->
    # isolates the kernel.
    for d in sorted(TOKENIZER_DIR.glob("contrastive_vqvae_aoj*")):
        model = load_tokenizer(d.name, cls=LitContrastiveVqVae)
        if model is not None:
            name = d.name.replace("contrastive_vqvae", "contrastive_vq").replace("_aoj", "")
            out[name], util = vq_encode(model, scaled)
            extras[name] = {f"codebook_util_q{q}": float(v) for q, v in enumerate(util)}

    # Production-analog anchor: plain VQ at a single 8192 codebook (13 bits).
    anchor = load_tokenizer("vqvae_particle_k8192", cls=LitVqVae)
    if anchor is not None:
        out["vq_k8192"], util = vq_encode(anchor, scaled)
        extras["vq_k8192"] = {f"codebook_util_q{q}": float(v) for q, v in enumerate(util)}

    # Low-dim continuous library + encoder-capacity ablation (kernel-only),
    # named library_aoj_d{4,6,8} and library_aoj_h{32,64,256}.
    for d in sorted(TOKENIZER_DIR.glob("library_aoj_d*")) + sorted(
        TOKENIZER_DIR.glob("library_aoj_h*")
    ):
        model = load_tokenizer(d.name)
        if model is not None:
            out[d.name.replace("_aoj", "")] = encode_with(model, scaled)

    # --- Campaign 5: continuous(VAE) vs discrete(VQ), +-physics, context-free
    #     vs context-aware. Names: c5_vae* / c5_vaeK* (VAE, recon / +physics),
    #     c5_vqp* / c5_cvq* (plain VQ / contrastive-VQ, RVQ 3x1024 = 30 bits);
    #     an "_attn" suffix marks the context-aware attention encoder.
    for d in sorted(TOKENIZER_DIR.glob("c5_vae*")):
        model = load_tokenizer(d.name, cls=LitVae)
        if model is not None:
            name = d.name.replace("_aoj", "")
            out[name], kl_bits = vae_encode(model, scaled)
            extras[name] = {"kl_rate_bits": round(kl_bits, 3),
                            "bits_fp16": int(out[name].shape[-1]) * 16}
    for pat, cls in [("c5_vqp*", LitVqVae), ("c5_cvq*", LitContrastiveVqVae)]:
        for d in sorted(TOKENIZER_DIR.glob(pat)):
            model = load_tokenizer(d.name, cls=cls)
            if model is not None:
                name = d.name.replace("_aoj", "")
                out[name], util = vq_encode(model, scaled)
                extras[name] = {f"codebook_util_q{q}": float(v) for q, v in enumerate(util)}

    # Rate-matched library row: post-hoc RVQ at 27 bits/particle.
    if "library" in out:
        out["library_rvq27"] = residual_kmeans_quantize(out["library"])

    # Untrained hand-crafted null: PID one-hot + charge + scaled kinematics.
    pdg_vals = torch.tensor([22.0, 211.0, -211.0, 130.0, 11.0, -11.0, 13.0, -13.0])
    pid = raw[..., 3][mask]
    onehot = (pid.unsqueeze(-1) == pdg_vals).float()
    charge = raw[..., 4][mask].unsqueeze(-1)
    out["null_handcrafted"] = torch.cat([scaled[..., :3][mask], onehot, charge], dim=-1)

    out["raw_kinematics"] = scaled[..., :3][mask]

    v1 = load_tokenizer("contrastive")
    if v1 is not None:
        out["v1_contrastive"] = encode_with(v1, scaled[..., :3])
    vq = load_tokenizer("vqvae", cls=LitVqVae)
    if vq is not None:
        batch = {"csts": scaled[..., :3].to(DEVICE), "mask": mask.to(DEVICE)}
        out["v1_vqvae"] = vq.encode(batch)[0][mask].cpu()

    return out, extras


@torch.no_grad()
def criterion_recall_at_k(z, raw_flat, k=8, pool=30_000, chunk=2_000, dpt_frac=0.1):
    """Availability-aware retrieval: of an anchor's *available* criterion
    partners in the pool, what fraction rank inside its top-k neighbours?

    Precision@k at realistic pool sizes is bounded by partner availability
    (criterion density ~3e-5), which flattens all methods to the same lift.
    Recall against the availability ceiling — hits@k / min(k, n_partners),
    averaged over anchors that have at least one partner — is 1.0 for a
    perfect embedding regardless of density.
    """
    rng = np.random.default_rng(SEED)
    n = min(pool, len(z))
    sel = torch.from_numpy(rng.choice(len(z), n, replace=False))
    z = torch.nn.functional.normalize(z[sel].to(DEVICE).float(), dim=-1)
    f = raw_flat[sel].to(DEVICE)

    pid, charge = f[:, 3], f[:, 4].round()
    pt, eta, phi = f[:, 0], f[:, 1], f[:, 2]

    recalls = []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        dpt = (pt[start:end, None] - pt[None, :]).abs()
        scale = torch.maximum(pt[start:end, None], pt[None, :]).clamp(min=1e-6)
        crit = (dpt / scale <= dpt_frac)
        crit &= (eta[start:end, None] - eta[None, :]).abs() <= 0.1
        dphi = (phi[start:end, None] - phi[None, :] + np.pi) % (2 * np.pi) - np.pi
        crit &= dphi.abs() <= 0.1
        crit &= pid[start:end, None] == pid[None, :]
        crit &= charge[start:end, None] == charge[None, :]
        rows = torch.arange(start, end, device=DEVICE)
        crit[rows - start, rows] = False  # self

        n_partners = crit.sum(dim=1)
        has = n_partners > 0
        if not has.any():
            continue

        sims = z[start:end] @ z.T
        sims[rows - start, rows] = float("-inf")
        topk = sims.topk(k, dim=1).indices  # [c, k]
        hits = torch.gather(crit, 1, topk).sum(dim=1).float()
        recalls.append((hits[has] / n_partners[has].clamp(max=k).float()).cpu())

    return float(torch.cat(recalls).mean()) if recalls else float("nan")


def linear_probe_accuracy(z, labels, n_train=15_000, n_test=5_000, seed=SEED):
    """Accuracy of a multinomial linear probe — information-sufficiency check
    (neutral: neither method trains on this objective)."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(z))[: n_train + n_test]
    tr, te = idx[:n_train], idx[n_train:]
    clf = LogisticRegression(max_iter=500)
    clf.fit(z[tr].numpy(), labels[tr])
    return float(clf.score(z[te].numpy(), labels[te]))


def aggregate_seeds(results):
    """Group provider rows like 'library', 'library_s43' into mean ± std."""
    import re

    groups = {}
    for name in results:
        base = re.sub(r"_s\d+$", "", name)
        groups.setdefault(base, []).append(name)

    agg = {}
    for base, names in groups.items():
        if len(names) < 2:
            continue
        metrics = {}
        for key in results[names[0]]:
            vals = [results[n][key] for n in names if isinstance(results[n].get(key), float)]
            if vals:
                metrics[key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n_seeds": len(vals),
                }
        agg[base] = metrics
    return agg


def within_pid_pt_r2(z, raw_flat):
    """Mean R^2 of linear pT recovery from tokens within each PID group."""
    pid = np.abs(raw_flat[:, 3].numpy())
    logpt = np.log(raw_flat[:, 0].numpy().clip(1e-3))
    zn = z.numpy()
    r2s, weights = [], []
    for code in PDG_GROUPS:
        m = pid == code
        if m.sum() < 200:
            continue
        X = np.concatenate([zn[m], np.ones((m.sum(), 1))], axis=1)
        y = logpt[m]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        r2s.append(1 - resid.var() / max(y.var(), 1e-9))
        weights.append(m.sum())
    return float(np.average(r2s, weights=weights)) if r2s else float("nan")


@torch.no_grad()
def pair_distance_curves(providers, raw_flat, pool=20_000, chunk=2_000, max_pairs=400_000):
    """Cosine distance vs dpT%% for same-PID/charge/**region** pairs (cliff check).

    Pairs are mined with the pT gate *off* (same PID, same charge, same
    region) so the curve spans the dpT axis across the 10%% window edge —
    that edge is exactly where hard windows should show a cliff.
    """
    rng = np.random.default_rng(SEED)
    n = min(pool, len(raw_flat))
    sel = torch.from_numpy(rng.choice(len(raw_flat), n, replace=False))
    f = raw_flat[sel].to(DEVICE)
    pid, charge = f[:, 3], f[:, 4].round()
    pt, eta, phi = f[:, 0], f[:, 1], f[:, 2]

    pairs_a, pairs_b = [], []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        crit = (eta[start:end, None] - eta[None, :]).abs() <= 0.1
        dphi = (phi[start:end, None] - phi[None, :] + np.pi) % (2 * np.pi) - np.pi
        crit &= dphi.abs() <= 0.1
        crit &= pid[start:end, None] == pid[None, :]
        crit &= charge[start:end, None] == charge[None, :]
        rows = torch.arange(start, end, device=DEVICE)
        crit[rows - start, rows] = False
        a, b = crit.nonzero(as_tuple=True)
        pairs_a.append((a + start).cpu())
        pairs_b.append(b.cpu())
        if sum(len(p) for p in pairs_a) > max_pairs:
            break
    a = torch.cat(pairs_a)[:max_pairs]
    b = torch.cat(pairs_b)[:max_pairs]

    pt_c = pt.cpu()
    dpt = (pt_c[a] - pt_c[b]).abs() / torch.maximum(pt_c[a], pt_c[b]).clamp(min=1e-6)
    dpt = dpt.numpy()

    curves = {}
    bins = np.linspace(0, 0.5, 26)
    for name in ["library", "library_hard"]:
        if name not in providers:
            continue
        z = torch.nn.functional.normalize(providers[name][sel], dim=-1)
        dist = (1 - (z[a] * z[b]).sum(1)).numpy()
        med = [
            np.median(dist[(dpt >= lo) & (dpt < hi)])
            if ((dpt >= lo) & (dpt < hi)).sum() > 50
            else np.nan
            for lo, hi in zip(bins[:-1], bins[1:])
        ]
        curves[name] = (0.5 * (bins[1:] + bins[:-1]), np.array(med))
    return curves


def provider_bits(name: str, dim: int) -> int:
    """Release rate in bits/particle: VQ = num_quantizers * log2(codebook_size);
    a continuous fp16 token = dim * 16."""
    if name == "vq_k8192":
        return 13  # 1 x log2(8192)
    if name.startswith(("vq_particle", "contrastive_vq")) or name in (
        "library_rvq27", "v1_vqvae"
    ):
        return 27  # 3 x log2(512)
    if name.startswith(("c5_vqp", "c5_cvq")):
        return 30  # 3 x log2(1024)
    return dim * 16  # continuous fp16


def fig_rate(results):
    """Compression curve: criterion recall vs release rate (bits/particle).

    The one figure the iteration is about — does the library criterion survive
    compression, and where does it break?  x is log-scale bits; the ~50-bit raw
    particle is marked so anti-compressive tokens (right of it) are visible.
    """
    def family(name):
        if name.startswith("c5_vaeK"):
            return "VAE + physics", "#2f8fd0"
        if name.startswith("c5_vae"):
            return "VAE (recon)", "#7fc7ec"
        if name.startswith("c5_cvq") or name.startswith("contrastive_vq"):
            return "contrastive-VQ", "#8250df"
        if name.startswith("c5_vqp"):
            return "plain VQ (c5)", "#eda100"
        if name.startswith("hybrid"):
            return "hybrid (continuous)", "#e34948"
        if name.startswith("vq") or name in ("library_rvq27", "v1_vqvae"):
            return "plain VQ", "#eda100"
        if name.startswith("library"):
            return "library (continuous)", "#1baf7a"
        return "reference", MUTED

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.axvline(50, color=BASE, lw=1, ls=":")
    ax.annotate("raw particle ~50 bits", (50, 0.03), xycoords=("data", "axes fraction"),
                fontsize=7.5, color=MUTED, rotation=90, va="bottom", ha="right")
    seen = set()
    for name, r in results.items():
        bits, y = r.get("bits"), r.get("criterion_recall_at_8")
        if bits is None or y is None:
            continue
        label, color = family(name)
        # Triangle marker = context-aware attention encoder (VAE bits = KL-rate).
        marker = "^" if name.endswith("_attn") else "o"
        ax.scatter(bits, y, s=55, color=color, marker=marker, zorder=3,
                   label=None if label in seen else label)
        seen.add(label)
        ax.annotate(name, (bits, y), xytext=(5, 3), textcoords="offset points",
                    fontsize=6.6, color=INK2)
    ax.scatter([], [], marker="^", color=MUTED, label="context-aware (attn)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("release rate (bits / particle, log scale — VAE = KL-rate)")
    ax.set_ylabel("criterion recall@8")
    ax.set_title("Compression curve: does the criterion survive fewer bits?")
    ax.legend(fontsize=8, loc="lower right")
    savefig(fig, "library_rate.png")


def fig_retrieval(results):
    order = [n for n in [
        "library", "library_rvq27", "vq_particle", "library_cone", "library_hard",
        "null_handcrafted", "raw_kinematics", "v1_contrastive", "v1_vqvae",
    ] if n in results] + sorted(n for n in results if n.startswith("hybrid"))
    vals = [results[n]["criterion_recall_at_8"] for n in order]

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    y = np.arange(len(order))[::-1]
    ax.barh(y, vals, height=0.55, color="#2a78d6")
    for yi, v in zip(y, vals):
        ax.annotate(f"{v:.3f}", (v, yi), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, order, fontsize=8.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("criterion recall@8 (available same-PID/charge/pT/region partners found)")
    ax.set_title("Does the embedding rank criterion partners on top?")
    ax.grid(axis="y", visible=False)
    savefig(fig, "library_retrieval.png")


def fig_pareto(results):
    """The two poles at once: criterion geometry vs information retention."""
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.axvspan(0.999, 1.001, color="#cde2fb", alpha=0.4, lw=0)
    ax.axhspan(0.9, 1.02, color="#cde2fb", alpha=0.4, lw=0)
    ax.annotate("iteration-3 target", (0.999, 0.92), fontsize=8, color=MUTED,
                ha="right")
    for name, r in results.items():
        x, y = r["criterion_recall_at_8"], r["within_pid_pt_r2"]
        is_hybrid = name.startswith("hybrid")
        color = "#e34948" if is_hybrid else (
            "#1baf7a" if name.startswith("library") else
            "#eda100" if "vq" in name else "#2a78d6")
        ax.scatter(x, y, s=55, color=color, zorder=3)
        ax.annotate(name, (x, y), xytext=(5, 4), textcoords="offset points",
                    fontsize=7.5, color=INK2)
    ax.set_xlabel("criterion recall@8 (organization)")
    ax.set_ylabel("within-PID pT R² (information)")
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Organization vs information — all interfaces")
    savefig(fig, "library_pareto.png")


def fig_pid_islands(providers, raw_flat):
    if "library" not in providers:
        return
    from sklearn.manifold import TSNE

    # Balanced subsample per species: pi/gamma dominate ~90% of candidates and
    # would drown the leptons; linear PCA also under-shows the 16-D structure
    # (purity 0.91 lives in higher components) — t-SNE renders the islands.
    rng = np.random.default_rng(SEED)
    pid_all = np.abs(raw_flat[:, 3].numpy())
    sel = np.concatenate([
        rng.permutation(np.flatnonzero(pid_all == code))[:2500] for code in PDG_GROUPS
    ])
    z = providers["library"][sel].numpy()
    pid = pid_all[sel]
    p = TSNE(n_components=2, random_state=SEED, init="pca", perplexity=30).fit_transform(z)

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for color, (code, label) in zip(C_CLASS, PDG_GROUPS.items()):
        m = pid == code
        ax.scatter(p[m, 0], p[m, 1], s=2.5, c=color, alpha=0.4, lw=0, label=label)
    leg = ax.legend(markerscale=5, fontsize=8.5, loc="best")
    for h in leg.legend_handles:
        h.set_alpha(1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title("Library tokens (AOJ, real CMS) — t-SNE colored by particle species",
                 fontsize=10)
    savefig(fig, "library_pid_islands.png")


def fig_cliff(curves):
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    styles = {"library": ("graded kernel", "#2a78d6"), "library_hard": ("hard windows", "#1baf7a")}
    for name, (x, y) in curves.items():
        label, color = styles.get(name, (name, INK))
        ax.plot(x, y, color=color, label=label)
    ax.axvline(0.1, color=BASE, lw=1, ls=":")
    ax.annotate("dpT window", (0.1, 0.02), xycoords=("data", "axes fraction"),
                fontsize=7.5, color=MUTED)
    ax.set_xlabel(r"pair $\Delta p_T$ (relative) — same PID/charge/region pairs")
    ax.set_ylabel("embedding cosine distance (median)")
    ax.set_title("Window-edge geometry: graded kernel vs hard windows")
    ax.legend()
    savefig(fig, "library_cliff_check.png")


def main() -> None:
    print("[collect] AOJ validation constituents ...")
    raw, scaled, mask = collect()
    raw_flat = raw[mask]
    pid_flat = raw_flat[:, 3]

    providers, extras = embedding_providers(raw, scaled, mask)
    print(f"[providers] {list(providers)}  ({len(raw_flat):,} constituents)")

    rng = np.random.default_rng(SEED)
    sub = torch.from_numpy(rng.choice(len(raw_flat), min(4000, len(raw_flat)), replace=False))
    pid_classes = raw_flat[:, 3].long().numpy()
    charge_classes = (raw_flat[:, 4].round().long() + 1).numpy()

    results = {}
    for name, z in providers.items():
        res = compute_retrieval_precision(z[sub], raw_flat[sub], k=8, **CRIT)
        purity = compute_knn_pid_purity(z[sub], pid_flat[sub], k=8)
        rank = compute_embedding_collapse(z[sub])["effective_rank"]
        results[name] = {
            "criterion_recall_at_8": criterion_recall_at_k(z, raw_flat),
            "recall_at_8_dpt5": criterion_recall_at_k(z, raw_flat, dpt_frac=0.05),
            "knn_pid_purity": float(purity),
            "pid_probe_acc": linear_probe_accuracy(z, pid_classes),
            "charge_probe_acc": linear_probe_accuracy(z, charge_classes),
            "within_pid_pt_r2": within_pid_pt_r2(z, raw_flat),
            "effective_rank": float(rank),
            "criterion_density": float(res["criterion_density"]),
            "dim": int(z.shape[-1]),
            "bits": extras.get(name, {}).get(
                "kl_rate_bits", provider_bits(name, int(z.shape[-1]))
            ),
            **extras.get(name, {}),
        }
        print(f"[{name:18s}] " + json.dumps(results[name]))

    agg = aggregate_seeds(results)
    if agg:
        print("[aggregate] " + json.dumps(agg, indent=2))

    curves = pair_distance_curves(providers, raw_flat)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (FIGURES_DIR / "library_metrics.json").write_text(
        json.dumps({"providers": results, "aggregate_over_seeds": agg}, indent=2)
    )
    fig_retrieval(results)
    fig_pareto(results)
    fig_rate(results)
    fig_pid_islands(providers, raw_flat)
    fig_cliff(curves)
    print(f"[done] metrics + figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
