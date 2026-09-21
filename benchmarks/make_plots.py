"""Validation plots and baseline checks for the tokenizer benchmark.

Produces (benchmarks/figures/):
    loss_contrastive.png       training-loss components + val loss
    loss_vqvae.png             VQ losses (recon / commit) + val loss
    collapse_diagnostics.png   effective rank, active dims, std, redundancy vs epoch
    codebook_vqvae.png         per-quantizer utilization vs epoch + code-usage spectrum
    vq_reconstruction.png      input vs reconstruction per feature (scaled space)
    embedding_maps.png         PCA of constituent embeddings, by jet class and by pT
    physics_faithfulness.png   embedding distance vs dR and vs d(pT) pairs
    robustness.png             embedding stability vs augmentation strength
    alignment_uniformity.png   Wang-Isola plane: raw features vs both tokenizers
    probe_roc.png              per-arm one-vs-rest ROC curves (5 classes)
    probe_comparison.png       accuracy / macro-AUC / kNN-purity comparison bars
    summary.json               consolidated numbers for the README table

Run after train_tokenizers.py (both models) and train_probes.py.
"""

import copy
import json

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve

from common import (
    FIGURES_DIR,
    LATENT_DIM,
    PROBES_DIR,
    SEED,
    TOKENIZER_DIR,
    make_datamodule,
    make_transforms,
)

from heptokens.data.augmentations import ConstituentAugmentation
from heptokens.models.contrastive import LitContrastiveTokenizer
from heptokens.models.utils import (
    compute_alignment_uniformity,
    compute_embedding_collapse,
    compute_knn_pid_purity,
)
from heptokens.models.vq_vae import LitVqVae

# ---------------------------------------------------------------------- style
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"

# Entity colors, fixed across every figure (categorical slots 1-3).
C_NULL = "#2a78d6"  # raw-feature reference
C_CON = "#1baf7a"  # contrastive tokens
C_VQ = "#eda100"  # VQ tokens
# Class slots 1-5 for the 5 jet classes.
C_CLASS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7"]
# Sequential blue ramp (magnitude).
BLUES = LinearSegmentedColormap.from_list(
    "seqblue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

mpl.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASE,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": INK2,
        "lines.linewidth": 2.0,
        "font.family": "sans-serif",
        "figure.dpi": 150,
    }
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUMMARY: dict = {}


def savefig(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {name}")


def load_metrics(model_name) -> pd.DataFrame:
    csv = sorted((TOKENIZER_DIR / model_name / "csv").rglob("metrics.csv"))[-1]
    return pd.read_csv(csv)


def epochify(df, col):
    """Per-epoch mean of a (possibly per-step) logged column."""
    sub = df[["epoch", col]].dropna()
    return sub.groupby("epoch")[col].mean()


# ------------------------------------------------------------------ loss figs
def fig_losses():
    df = load_metrics("contrastive")
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for col, label, color, ls in [
        ("train/sim_loss", "invariance (sim)", C_CON, "-"),
        ("train/var_loss", "variance hinge", "#4a3aa7", "-"),
        ("train/cov_loss", "covariance", "#e34948", "-"),
        ("val/total_loss", "val total", INK2, "--"),
    ]:
        if col in df:
            s = epochify(df, col)
            ax.plot(s.index, s.values, ls, color=color, label=label)
            ax.annotate(
                label, (s.index[-1], s.values[-1]), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color=INK2, va="center",
            )
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Contrastive tokenizer — VICReg components")
    ax.legend(loc="upper right")
    savefig(fig, "loss_contrastive.png")
    SUMMARY["contrastive_final_val_loss"] = float(epochify(df, "val/total_loss").iloc[-1])

    df = load_metrics("vqvae")
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for col, label, color, ls in [
        ("train/recon_loss", "reconstruction", C_VQ, "-"),
        ("train/commit_loss", "commitment", "#4a3aa7", "-"),
        ("val/total_loss", "val total", INK2, "--"),
    ]:
        if col in df:
            s = epochify(df, col)
            ax.plot(s.index, s.values, ls, color=color, label=label)
            ax.annotate(
                label, (s.index[-1], s.values[-1]), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color=INK2, va="center",
            )
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("VQ-VAE baseline — loss components")
    ax.legend(loc="upper right")
    savefig(fig, "loss_vqvae.png")
    SUMMARY["vqvae_final_val_loss"] = float(epochify(df, "val/total_loss").iloc[-1])
    if "val/scaled_mae" in df:
        SUMMARY["vqvae_final_scaled_mae"] = float(epochify(df, "val/scaled_mae").iloc[-1])


def fig_collapse():
    df = load_metrics("contrastive")
    panels = [
        ("val/effective_rank", "effective rank", LATENT_DIM, "latent dim"),
        ("val/active_dims", "active dimensions (frac)", 1.0, "all active"),
        ("val/std_mean", "mean per-dim std", None, None),
        ("val/corr_offdiag", "mean |off-diag corr|", None, None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2))
    for ax, (col, title, ref, ref_label) in zip(axes.ravel(), panels):
        if col not in df:
            ax.set_visible(False)
            continue
        s = epochify(df, col)
        ax.plot(s.index, s.values, color=C_CON)
        if ref is not None:
            ax.axhline(ref, color=BASE, lw=1, ls=":")
            ax.annotate(ref_label, (0.02, ref), xycoords=("axes fraction", "data"),
                        fontsize=7.5, color=MUTED, va="bottom")
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("epoch")
        SUMMARY[f"contrastive_{col.split('/')[-1]}"] = float(s.iloc[-1])
    fig.suptitle("Continuous-token collapse diagnostics (validation)", x=0.01, ha="left",
                 fontsize=11, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, "collapse_diagnostics.png")


def fig_codebook(ev):
    df = load_metrics("vqvae")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    ax = axes[0]
    for q, color in enumerate(["#2a78d6", "#1baf7a", "#eda100"]):
        col = f"val/codebook_util_q{q}"
        if col in df:
            s = epochify(df, col)
            ax.plot(s.index, s.values, color=color, label=f"quantizer {q}")
            SUMMARY[f"vqvae_codebook_util_q{q}"] = float(s.iloc[-1])
    ax.set_xlabel("epoch")
    ax.set_ylabel("fraction of codes used")
    ax.set_ylim(0, 1.05)
    ax.set_title("Codebook utilization")
    ax.legend()

    ax = axes[1]
    idx = ev["vq_indices"]  # [n_valid, Q]
    for q, color in enumerate(["#2a78d6", "#1baf7a", "#eda100"]):
        counts = np.bincount(idx[:, q], minlength=512).astype(float)
        counts = np.sort(counts)[::-1] / counts.sum()
        ax.plot(np.arange(1, 513), counts, color=color, label=f"quantizer {q}")
    ax.set_yscale("log")
    ax.set_xlabel("code rank")
    ax.set_ylabel("usage fraction")
    ax.set_title("Code-usage spectrum (val)")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "codebook_vqvae.png")


# ------------------------------------------------------------------ eval pass
@torch.no_grad()
def collect_eval(n_batches=8, batch_size=512):
    """One shared evaluation pass: raw + scaled batches, embeddings, recon."""
    con = LitContrastiveTokenizer.load_from_checkpoint(
        TOKENIZER_DIR / "contrastive" / "checkpoints" / "best.ckpt",
        map_location=DEVICE, weights_only=False,
    ).eval()
    vq = LitVqVae.load_from_checkpoint(
        TOKENIZER_DIR / "vqvae" / "checkpoints" / "best.ckpt",
        map_location=DEVICE, weights_only=False,
    ).eval()

    dm = make_datamodule(batch_size=batch_size, num_workers=4, transforms=None)  # RAW
    dm.setup()
    preprocess = make_transforms()["preprocess"]

    out = {k: [] for k in
           ["csts_raw", "csts_scaled", "mask", "labels", "z_con", "z_vq", "recon", "vq_idx"]}
    for i, raw in enumerate(dm.val_dataloader()):
        if i >= n_batches:
            break
        scaled = preprocess(copy.deepcopy(raw))
        batch = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                 for k, v in scaled.items()}
        z_con = con.encode(batch)[0]
        z_vq, idx, _ = vq.encode(batch)
        recon = vq.decode(z_vq, batch)
        out["csts_raw"].append(raw["csts"])
        out["csts_scaled"].append(scaled["csts"].cpu())
        out["mask"].append(raw["mask"])
        out["labels"].append(raw["labels"])
        out["z_con"].append(z_con.cpu())
        out["z_vq"].append(z_vq.cpu())
        out["recon"].append(recon.cpu())
        out["vq_idx"].append(idx.cpu())
    ev = {k: torch.cat(v) for k, v in out.items()}

    mask = ev["mask"].bool()
    ev["flat_z_con"] = ev["z_con"][mask].numpy()
    ev["flat_z_vq"] = ev["z_vq"][mask].numpy()
    ev["vq_indices"] = ev["vq_idx"][mask].numpy()
    ev["flat_scaled"] = ev["csts_scaled"][mask].numpy()
    ev["flat_raw"] = ev["csts_raw"][mask].numpy()
    ev["flat_jetclass"] = ev["labels"].unsqueeze(1).expand(-1, mask.shape[1])[mask].numpy()
    ev["models"] = (con, vq)
    ev["preprocess"] = preprocess
    return ev


def fig_vq_reconstruction(ev):
    names = ["pT", "Δη", "Δφ"]
    mask = ev["mask"].bool()
    x = ev["csts_scaled"][mask].numpy()
    r = ev["recon"][mask].numpy()
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    for i, (ax, name) in enumerate(zip(axes, names)):
        lo, hi = np.percentile(x[:, i], [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)
        ax.hist(x[:, i], bins=bins, histtype="step", color=INK, lw=1.6, label="input")
        ax.hist(r[:, i], bins=bins, histtype="stepfilled", color=C_VQ, alpha=0.45,
                label="VQ reconstruction")
        mae = np.abs(x[:, i] - r[:, i]).mean()
        ax.annotate(f"MAE {mae:.3f}", (0.03, 0.92), xycoords="axes fraction",
                    fontsize=8.5, color=INK2)
        ax.set_title(f"{name} (scaled)", fontsize=9.5)
        ax.set_yticks([])
        if i == 0:
            ax.legend(loc="lower left", fontsize=8)
    SUMMARY["vqvae_scaled_mae_eval"] = float(np.abs(x - r).mean())
    fig.suptitle("VQ-VAE reconstruction fidelity (the discrete bottleneck cost)",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    savefig(fig, "vq_reconstruction.png")


def fig_embeddings(ev):
    rng = np.random.default_rng(SEED)
    n = min(20_000, len(ev["flat_z_con"]))
    sel = rng.choice(len(ev["flat_z_con"]), n, replace=False)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.6))
    for row, (name, z) in enumerate([("Contrastive", ev["flat_z_con"]),
                                     ("VQ-VAE", ev["flat_z_vq"])]):
        p = PCA(n_components=2, random_state=SEED).fit_transform(z[sel])
        ax = axes[row, 0]
        for c in range(5):
            m = ev["flat_jetclass"][sel] == c
            ax.scatter(p[m, 0], p[m, 1], s=2.2, c=C_CLASS[c], alpha=0.35, lw=0,
                       label=f"class {c}" if row == 0 else None)
        ax.set_title(f"{name} tokens — by jet class", fontsize=9.5)
        ax = axes[row, 1]
        pt = ev["flat_scaled"][sel][:, 0]
        sc = ax.scatter(p[:, 0], p[:, 1], s=2.2, c=pt, cmap=BLUES, alpha=0.5, lw=0)
        ax.set_title(f"{name} tokens — by constituent pT (scaled)", fontsize=9.5)
        fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02).set_label("pT (scaled)", color=MUTED)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    leg = axes[0, 0].legend(loc="best", markerscale=4, fontsize=8)
    for h in leg.legend_handles:
        h.set_alpha(1)
    fig.suptitle("Per-constituent embedding maps (PCA of validation constituents)",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    savefig(fig, "embedding_maps.png")


def _pair_sample(ev, n_jets=800, pairs_per_jet=40):
    """Sample same-jet constituent pairs: physical dR, d(pT), embedding distances."""
    rng = np.random.default_rng(SEED)
    mask = ev["mask"].bool().numpy()
    res = {"dr": [], "dpt": [], "d_con": [], "d_vq": []}
    zc, zv = ev["z_con"].numpy(), ev["z_vq"].numpy()
    raw = ev["csts_raw"].numpy()
    jet_ids = rng.choice(len(mask), min(n_jets, len(mask)), replace=False)
    for j in jet_ids:
        vi = np.flatnonzero(mask[j])
        if len(vi) < 4:
            continue
        a = rng.choice(vi, pairs_per_jet)
        b = rng.choice(vi, pairs_per_jet)
        keep = a != b
        a, b = a[keep], b[keep]
        deta = raw[j, a, 1] - raw[j, b, 1]
        dphi = raw[j, a, 2] - raw[j, b, 2]
        res["dr"].append(np.hypot(deta, dphi))
        res["dpt"].append(np.abs(np.log(raw[j, a, 0] / np.clip(raw[j, b, 0], 1e-6, None))))

        def cosd(z):
            za, zb = z[j, a], z[j, b]
            num = (za * zb).sum(1)
            den = np.linalg.norm(za, axis=1) * np.linalg.norm(zb, axis=1) + 1e-9
            return 1.0 - num / den

        res["d_con"].append(cosd(zc))
        res["d_vq"].append(cosd(zv))
    return {k: np.concatenate(v) for k, v in res.items()}


def fig_physics_faithfulness(ev):
    pairs = _pair_sample(ev)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
    specs = [("dr", r"pair $\Delta R$", np.linspace(0, 1.0, 14)),
             ("dpt", r"pair $|\Delta \log p_T|$", np.linspace(0, 4, 14))]
    for ax, (key, xlabel, bins) in zip(axes, specs):
        x = pairs[key]
        centers = 0.5 * (bins[1:] + bins[:-1])
        for dkey, label, color in [("d_con", "contrastive", C_CON), ("d_vq", "VQ-VAE", C_VQ)]:
            med, lo, hi = [], [], []
            for k in range(len(bins) - 1):
                m = (x >= bins[k]) & (x < bins[k + 1])
                if m.sum() < 30:
                    med.append(np.nan), lo.append(np.nan), hi.append(np.nan)
                    continue
                q = np.percentile(pairs[dkey][m], [25, 50, 75])
                lo.append(q[0]), med.append(q[1]), hi.append(q[2])
            ax.plot(centers, med, color=color, label=label)
            ax.fill_between(centers, lo, hi, color=color, alpha=0.18, lw=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("embedding cosine distance")
        ax.legend()
    # correlation summary
    for dkey, name in [("d_con", "contrastive"), ("d_vq", "vqvae")]:
        SUMMARY[f"corr_dist_dr_{name}"] = float(np.corrcoef(pairs["dr"], pairs[dkey])[0, 1])
        SUMMARY[f"corr_dist_dpt_{name}"] = float(np.corrcoef(pairs["dpt"], pairs[dkey])[0, 1])
    fig.suptitle("Does embedding distance track physical distance? (same-jet pairs)",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    savefig(fig, "physics_faithfulness.png")


@torch.no_grad()
def fig_robustness(ev):
    con, vq = ev["models"]
    sigmas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
    batch_scaled = {
        "csts": ev["csts_scaled"][:1024].to(DEVICE),
        "mask": ev["mask"][:1024].to(DEVICE),
    }
    mask = batch_scaled["mask"].bool()
    base = {"contrastive": con.encode(batch_scaled)[0][mask],
            "vqvae": vq.encode(batch_scaled)[0][mask]}
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for name, model, color in [("contrastive", con, C_CON), ("vqvae", vq, C_VQ)]:
        sims = []
        for s in sigmas:
            aug = ConstituentAugmentation(pt_indices=(), eta_indices=(0, 1, 2),
                                          angle_smear_sigma=s, wrap_phi=False)
            b2 = dict(batch_scaled)
            b2["csts"] = aug(batch_scaled["csts"], batch_scaled["mask"])
            z2 = model.encode(b2)[0][mask]
            sims.append(float(torch.nn.functional.cosine_similarity(
                base[name], z2, dim=-1).mean()))
        ax.plot(sigmas, sims, "-o", color=color, ms=5, label=name)
        SUMMARY[f"robustness_cos@0.05_{name}"] = sims[2]
    ax.axvline(0.05, color=BASE, lw=1, ls=":")
    ax.annotate("train σ", (0.05, 0.02), xycoords=("data", "axes fraction"),
                fontsize=7.5, color=MUTED, ha="left")
    ax.set_xscale("log")
    ax.set_xlabel("feature smearing σ (scaled space)")
    ax.set_ylabel("cosine similarity to unsmeared")
    ax.set_title("Token stability under detector-like smearing")
    ax.legend()
    savefig(fig, "robustness.png")


@torch.no_grad()
def fig_alignment_uniformity(ev):
    con, vq = ev["models"]
    batch = {"csts": ev["csts_scaled"][:2048].to(DEVICE), "mask": ev["mask"][:2048].to(DEVICE)}
    mask = batch["mask"].bool()
    aug = ConstituentAugmentation(pt_indices=(), eta_indices=(0, 1, 2),
                                  angle_smear_sigma=0.05, wrap_phi=False)
    b2 = dict(batch)
    b2["csts"] = aug(batch["csts"], batch["mask"])

    pts = {}
    for name, embed, color in [
        ("raw features", lambda b: b["csts"], C_NULL),
        ("contrastive", lambda b: con.encode(b)[0], C_CON),
        ("VQ-VAE", lambda b: vq.encode(b)[0], C_VQ),
    ]:
        za, zb = embed(batch)[mask], embed(b2)[mask]
        m = compute_alignment_uniformity(za, zb)
        pts[name] = (float(m["uniformity"]), float(m["alignment"]), color)
        SUMMARY[f"uniformity_{name.replace(' ', '_')}"] = float(m["uniformity"])
        SUMMARY[f"alignment_{name.replace(' ', '_')}"] = float(m["alignment"])

    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    for name, (u, a, color) in pts.items():
        ax.scatter(u, a, s=90, color=color, zorder=3)
        ax.annotate(name, (u, a), xytext=(8, 5), textcoords="offset points",
                    fontsize=9, color=INK2)
    ax.set_xlabel("uniformity (lower = spreads the hypersphere better)")
    ax.set_ylabel("alignment (lower = views closer)")
    ax.set_title("Alignment–uniformity plane (σ=0.05 views)")
    ax.annotate("better", xy=(0.04, 0.06), xycoords="axes fraction", fontsize=9,
                color=MUTED, style="italic")
    savefig(fig, "alignment_uniformity.png")


def fig_knn_purity(ev):
    z_r = torch.from_numpy(ev["flat_scaled"])
    z_c = torch.from_numpy(ev["flat_z_con"])
    z_v = torch.from_numpy(ev["flat_z_vq"])
    cls = torch.from_numpy(ev["flat_jetclass"])
    out = {}
    for name, z in [("raw features", z_r), ("contrastive", z_c), ("VQ-VAE", z_v)]:
        out[name] = float(compute_knn_pid_purity(z, cls, k=8, max_points=4000))
        SUMMARY[f"knn_class_purity_{name.replace(' ', '_')}"] = out[name]
    return out


def fig_probes(knn):
    arms = [("null_features", "raw features (ref)", C_NULL),
            ("contrastive_tokens", "contrastive tokens", C_CON),
            ("vqvae_tokens", "VQ tokens", C_VQ)]
    metrics = {}
    for arm, _, _ in arms:
        p = PROBES_DIR / arm / "metrics.json"
        if p.exists():
            metrics[arm] = json.loads(p.read_text())
    if not metrics:
        print("[probes] no metrics found, skipping probe figures")
        return

    # ROC curves, one panel per arm
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (arm, label, _) in zip(axes, [a for a in arms if a[0] in metrics]):
        d = np.load(PROBES_DIR / arm / "preds.npz")
        probs, labels = d["probs"], d["labels"]
        for c in range(probs.shape[1]):
            fpr, tpr, _ = roc_curve(labels == c, probs[:, c])
            ax.plot(fpr, tpr, color=C_CLASS[c], lw=1.6, label=f"class {c}")
        ax.plot([0, 1], [0, 1], color=BASE, lw=1, ls=":")
        ax.set_title(f"{label}\nmacro AUC {metrics[arm]['auc_macro_ovr']:.3f}", fontsize=9)
        ax.set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[-1].legend(loc="lower right", fontsize=7.5)
    fig.suptitle("Consumer-view probes — one-vs-rest ROC (validation)",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    savefig(fig, "probe_roc.png")

    # Comparison bars: acc | macro AUC | kNN purity
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 2.8))
    panels = [
        ("acc", "probe accuracy", lambda a: metrics[a]["acc"] if a in metrics else np.nan),
        ("auc", "probe macro AUC", lambda a: metrics[a]["auc_macro_ovr"] if a in metrics else np.nan),
        ("knn", "constituent kNN class purity",
         lambda a: knn.get({"null_features": "raw features",
                            "contrastive_tokens": "contrastive",
                            "vqvae_tokens": "VQ-VAE"}[a], np.nan)),
    ]
    for ax, (key, title, getv) in zip(axes, panels):
        names = [label for _, label, _ in arms]
        vals = [getv(arm) for arm, _, _ in arms]
        colors = [c for _, _, c in arms]
        y = np.arange(len(names))[::-1]
        ax.barh(y, vals, height=0.55, color=colors)
        for yi, v in zip(y, vals):
            if np.isfinite(v):
                ax.annotate(f"{v:.3f}", (v, yi), xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=8.5, color=INK2)
        ax.set_yticks(y, names, fontsize=8.5)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlim(0, 1.02 * np.nanmax(vals) if np.isfinite(np.nanmax(vals)) else 1)
        if key == "knn":
            ax.axvline(0.2, color=BASE, lw=1, ls=":")
            ax.annotate("class prior", (0.2, -0.45), xycoords=("data", "data"),
                        fontsize=7.5, color=MUTED, ha="center", annotation_clip=False)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Baseline checks — same probe, three interfaces", x=0.01, ha="left",
                 fontsize=11, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    savefig(fig, "probe_comparison.png")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_losses()
    fig_collapse()
    print("[eval] running shared evaluation pass ...")
    ev = collect_eval()
    fig_codebook(ev)
    fig_vq_reconstruction(ev)
    fig_embeddings(ev)
    fig_physics_faithfulness(ev)
    fig_robustness(ev)
    fig_alignment_uniformity(ev)
    knn = fig_knn_purity(ev)
    fig_probes(knn)
    (FIGURES_DIR / "summary.json").write_text(json.dumps(SUMMARY, indent=2, sort_keys=True))
    print(json.dumps(SUMMARY, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
