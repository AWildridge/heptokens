"""Campaign 7 — consumer-view QCD-vs-top probes on AOJ MC (record 18610).

Every arm trains the SAME binary transformer classifier; only the input
interface differs. Token arms consume frozen exports of tokenizers trained on
REAL AOJ RunG data (release-realistic transfer to MC); reference arms consume
the raw/untrained interfaces. Cross-section weights are used end-to-end:
pT-matched + class-normalized in the training loss, physical and matched
schemes in the test metrics (user-decided protocol: "match + report both").

Split contract: exports iterate the h5 in rng(SEED).permutation(n) order (the
datamodule's test_frac=1 path), and every in-memory arm is built in the same
permuted order — so contiguous slices 70/15/15 give identical train/val/test
membership for all arms, with a held-out TEST slice for the reported metrics.

Usage:
    python train_probes_mc.py [--arms ...] [--seeds 42 43 44] [--epochs 20]
    python train_probes_mc.py --figures-only
"""

import argparse
import json

import h5py
import joblib
import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

from common import (
    AOJ_CST_SCALER,
    AOJMC_STRUCT_H5,
    PROBES_DIR,
    SEED,
    TOKENIZER_DIR,
    TOKENS_DIR,
    aojmc_datamodule_cfg,
)
from eval_library import grid_centers

from heptokens.export_latents import export as export_latents
from heptokens.export_tokens import export as export_tokens
from heptokens.models.particle_encoder import DEFAULT_PDG_IDS
from heptokens.models.token_classifier import VectorClassifier

MC_PROBES_DIR = PROBES_DIR / "mc"
PT_BIN_EDGES = [300.0, 470.0, 600.0, 800.0, 1200.0]

# Shared probe: the repo's standard transformer scale (matches
# transformer_partial()); the tiny jets5c PROBE_KW would risk capacity-limiting
# the raw-feature ceiling. pooling MUST stay "mean" (ClsTokenPooler is broken).
MC_PROBE_KW = dict(
    n_classes=2, d_model=128, n_heads=8, num_layers=4, dim_feedforward=512,
    dropout=0.0, pooling="mean", learning_rate=1e-3,
)

# arm -> (export kind, tokenizer run dir, model_class) ; None = untrained/raw.
EXPORT_ARMS = {
    "library_d6": ("latents", "library_aoj_d6", "contrastive"),
    "hybrid_w30": ("latents", "hybrid_w30", "contrastive"),
    "vaeK_b003_d6": ("latents", "c6_vaeK_aoj_b003_d6", "vae"),
    "vae_b01_d6": ("latents", "c6_vae_aoj_b01_d6", "vae"),
    "vqvae_particle": ("tokens", "vqvae_particle_aoj", None),  # heptokens baseline
    "c5_vqp": ("tokens", "c5_vqp_aoj", None),
}
# raw_features = the full-information ceiling: scaled kinematics + PID one-hot
# + charge. (Feeding raw PDG codes through a linear layer swamps the O(1)
# kinematics with an O(200) column — measured: AUC 0.58 vs one-hot encoding.)
# raw_kinematics = the PID-blind reference (scaled pt/eta/phi only).
ALL_ARMS = ["raw_features", "raw_kinematics", "physics_grid_double", *EXPORT_ARMS]


# --------------------------------------------------------------------- exports
def run_exports(arms) -> dict:
    paths = {}
    for name in arms:
        if name not in EXPORT_ARMS:
            continue
        kind, run, model_class = EXPORT_ARMS[name]
        out_dir = TOKENS_DIR / "mc" / name
        out_path = out_dir / f"{AOJMC_STRUCT_H5.stem}.npz"
        paths[name] = out_path
        if out_path.exists():
            print(f"[export] {out_path} exists, skipping")
            continue
        ckpt = TOKENIZER_DIR / run / "checkpoints" / "best.ckpt"
        if not ckpt.exists():
            ckpt = TOKENIZER_DIR / run / "checkpoints" / "last.ckpt"
        if not ckpt.exists():
            raise FileNotFoundError(f"no checkpoint for arm {name} ({run})")
        cfg = {
            "ckpt_path": str(ckpt),
            "output_dir": str(out_dir),
            "datamodule": aojmc_datamodule_cfg(0.0, 0.0, 1.0, batch_size=2048),
            "trainer": {
                "_target_": "lightning.Trainer",
                "accelerator": "auto",
                "devices": 1,
                "enable_progress_bar": True,
                "logger": False,
            },
        }
        if kind == "latents":
            cfg["model_class"] = model_class
            cfg["latent_dtype"] = "fp16"
        print(f"[export] {name} <- {ckpt}")
        (export_latents if kind == "latents" else export_tokens)(OmegaConf.create(cfg))
    return paths


# ------------------------------------------------------------------ meta / h5
def load_meta():
    """Per-jet metadata in the EXPORT (permuted) order, with hard assertions."""
    with h5py.File(AOJMC_STRUCT_H5, "r") as f:
        label = f["jets/label"][:]
        weight = f["jets/weight"][:].astype(np.float64)
        pt = f["jets/pt"][:]
    perm = np.random.default_rng(SEED).permutation(len(label))
    return dict(label=label[perm], weight=weight[perm], pt=pt[perm], perm=perm)


def assert_export_aligned(npz_path, meta):
    d = np.load(npz_path)
    assert np.array_equal(d["labels"].astype(np.int64), meta["label"]), (
        f"{npz_path}: export label order != h5[perm] — split contract broken")
    if "weights" in d:
        assert np.allclose(d["weights"], meta["weight"], rtol=1e-5), (
            f"{npz_path}: export weights != h5[perm]")


# ------------------------------------------------------------- training weights
def build_weights(meta, n, pt_match=True, clip_q=0.999):
    """(train_w, eval_w_physical, eval_w_matched) in permuted order, length n.

    train_w = physical x [pT-match on top] x per-class normalization, derived
    from the TRAIN slice only. Matched eval weights reuse the train-derived
    multiplier (standard practice).
    """
    label, weight, pt = meta["label"][:n], meta["weight"][:n].copy(), meta["pt"][:n]
    if clip_q is not None:
        for c in (0, 1):
            m = label == c
            weight[m] = np.minimum(weight[m], np.quantile(weight[m], clip_q))

    n_train = int(0.7 * n)
    matched = weight.copy()
    if pt_match:
        # Reweight top to the xsec-weighted QCD pT spectrum (train slice, log bins).
        tr = slice(0, n_train)
        bins = np.geomspace(pt[tr].min(), pt[tr].max() + 1e-3, 25)
        hq, _ = np.histogram(pt[tr][label[tr] == 0], bins=bins,
                             weights=weight[tr][label[tr] == 0], density=True)
        ht, _ = np.histogram(pt[tr][label[tr] == 1], bins=bins,
                             weights=weight[tr][label[tr] == 1], density=True)
        ratio = np.where(ht > 0, hq / np.maximum(ht, 1e-12), 1.0)
        ib = np.clip(np.digitize(pt, bins) - 1, 0, len(ratio) - 1)
        matched[label == 1] *= ratio[ib][label == 1]

    train_w = matched.copy()
    for c in (0, 1):  # class balance, factors from the train slice
        m = label[:n_train] == c
        train_w[label == c] *= m.sum() / max(train_w[:n_train][m].sum(), 1e-12)
    for name, w in [("physical", weight), ("matched", matched), ("train", train_w)]:
        for c, cname in [(0, "QCD"), (1, "top")]:
            wc = w[:n_train][label[:n_train] == c]
            print(f"[weights] {name:8s} {cname}: sum={wc.sum():.4g} "
                  f"ESS={wc.sum()**2 / (wc**2).sum():,.0f}")
    return train_w.astype(np.float32), weight.astype(np.float32), matched.astype(np.float32)


# ------------------------------------------------------------------- interfaces
class ArrayDataset(Dataset):
    """Uniform consumer view: precomputed [n,150,D] vectors (+mask/labels/weights)."""

    def __init__(self, z, mask, labels, weights, start, stop, input_key="z_q"):
        self.z, self.mask = z[start:stop], mask[start:stop]
        self.labels = labels[start:stop].astype(np.int64)
        self.weights = weights[start:stop]
        self.input_key = input_key

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            self.input_key: torch.from_numpy(self.z[i].astype(np.float32)),
            "mask": torch.from_numpy(self.mask[i]),
            "labels": torch.tensor(self.labels[i]),
            "weights": torch.tensor(self.weights[i]),
        }


def scale_kinematics(csts, valid, scaler):
    """Apply the frozen AOJ cst scaler to [n,150,5] raw features (valid rows)."""
    out = np.zeros_like(csts, dtype=np.float32)
    out[valid] = scaler.transform(csts[valid]).astype(np.float32)
    return out


def build_untrained_inputs(arm, meta, n):
    """Raw / null / grid interfaces, in permuted order, as fp16 arrays."""
    with h5py.File(AOJMC_STRUCT_H5, "r") as f:
        perm = meta["perm"][:n]
        csts = np.stack(
            [f[f"csts/{k}"][:][perm] for k in ("pt", "eta", "phi", "pid", "charge")],
            axis=-1,
        )
        valid = f["csts/valid"][:][perm].astype(bool)

    scaler = joblib.load(AOJ_CST_SCALER)
    pdg = np.array(DEFAULT_PDG_IDS, dtype=np.float32)
    if arm == "raw_kinematics":
        z = scale_kinematics(csts, valid, scaler)[..., :3]
    elif arm == "raw_features":
        scaled = scale_kinematics(csts, valid, scaler)
        onehot = (csts[..., 3:4] == pdg).astype(np.float32) * valid[..., None]
        z = np.concatenate([scaled[..., :3], onehot, csts[..., 4:5] * valid[..., None]], axis=-1)
    elif arm == "physics_grid_double":
        parts = []
        for off in (0.0, 0.5):
            g = grid_centers(torch.from_numpy(csts), offset=off).numpy()
            parts.append(scale_kinematics(g, valid, scaler)[..., :3])
        onehot = (csts[..., 3:4] == pdg).astype(np.float32) * valid[..., None]
        z = np.concatenate([*parts, onehot, csts[..., 4:5] * valid[..., None]], axis=-1)
    else:
        raise ValueError(arm)
    z[~valid] = 0.0
    return z.astype(np.float16), valid


def build_export_inputs(arm, npz_path, n):
    d = np.load(npz_path)
    if "latents" in d:
        z = d["latents"][:n]
        mask = np.any(z != 0.0, axis=-1)
        return z.astype(np.float16), mask
    idx = d["indices"][:n].astype(np.int64)
    books = d["codebooks"].astype(np.float32)  # [Q, K, D]
    mask = idx[..., 0] >= 0
    safe = np.where(idx < 0, 0, idx)
    z = np.zeros((*idx.shape[:2], books.shape[-1]), dtype=np.float32)
    for q in range(books.shape[0]):
        z += books[q][safe[..., q]]
    z[~mask] = 0.0
    return z.astype(np.float16), mask


# ------------------------------------------------------------------ metrics
def weighted_metrics(labels, scores, w, prefix):
    fpr, tpr, _ = roc_curve(labels, scores, sample_weight=w)
    order = np.argsort(tpr)
    rej = {}
    for eff in (0.3, 0.5):
        f = float(np.interp(eff, tpr[order], fpr[order]))
        rej[f"{prefix}_rejection_at_{int(eff*100)}"] = float(1.0 / max(f, 1e-9))
    pred = (scores > 0.5).astype(int)
    accs = [
        (w[(labels == c)] * (pred[labels == c] == c)).sum() / max(w[labels == c].sum(), 1e-12)
        for c in (0, 1)
    ]
    return {
        f"{prefix}_auc": float(roc_auc_score(labels, scores, sample_weight=w)),
        f"{prefix}_balanced_acc": float(0.5 * (accs[0] + accs[1])),
        **rej,
    }


def evaluate(labels, scores, w_phys, w_match, pt):
    m = weighted_metrics(labels, scores, w_phys, "physical")
    m.update(weighted_metrics(labels, scores, w_match, "matched"))
    for lo, hi in zip(PT_BIN_EDGES[:-1], PT_BIN_EDGES[1:]):
        sel = (pt >= lo) & (pt < hi)
        if sel.sum() > 500 and len(np.unique(labels[sel])) == 2:
            m[f"auc_pt_{int(lo)}_{int(hi)}"] = float(
                roc_auc_score(labels[sel], scores[sel], sample_weight=w_phys[sel])
            )
    return m


# ------------------------------------------------------------------- training
def train_arm(arm, seed, z, mask, meta, weights, n, args):
    run_dir = MC_PROBES_DIR / f"{arm}_s{seed}"
    if (run_dir / "metrics.json").exists():
        print(f"[skip] {arm}_s{seed}")
        return json.loads((run_dir / "metrics.json").read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    L.seed_everything(seed, workers=True)

    train_w, w_phys, w_match = weights
    labels = meta["label"][:n]
    n_tr, n_va = int(0.7 * n), int(0.15 * n)
    # No multiprocessing anywhere in this harness: forked dataloader workers
    # deadlocked twice on this machine (once at predict, once in fit's sanity
    # check — py-spy: main thread blocked on the worker queue with idle
    # children). The dataset is fully in-RAM, so workers only bought fork
    # overhead anyway; worker-less epochs measured within ~10% of the same.
    mk = lambda a, b, w, sh: DataLoader(  # noqa: E731
        ArrayDataset(z, mask, labels, w, a, b),
        batch_size=args.batch_size, shuffle=sh, num_workers=0, drop_last=sh,
    )
    tr = mk(0, n_tr, train_w, True)
    va = mk(n_tr, n_tr + n_va, train_w, False)
    te = mk(n_tr + n_va, n, w_phys, False)

    probe = VectorClassifier(codebook_dim=z.shape[2], **MC_PROBE_KW)

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        callbacks=[
            ModelCheckpoint(
                dirpath=run_dir / "checkpoints", filename="best",
                monitor="valid/total_loss", mode="min", save_top_k=1,
                enable_version_counter=False, auto_insert_metric_name=False,
            ),
            EarlyStopping(monitor="valid/total_loss", patience=4, mode="min"),
        ],
        logger=CSVLogger(save_dir=run_dir, name="csv"),
        log_every_n_steps=50,
        enable_progress_bar=False,
        default_root_dir=run_dir,
    )
    trainer.fit(probe, tr, va)

    preds = trainer.predict(probe, te, ckpt_path="best")
    logits = torch.cat([p["output"] for p in preds]).float()
    scores = torch.softmax(logits, dim=1)[:, 1].numpy()
    te_labels = labels[n_tr + n_va : n]
    te_pt = meta["pt"][n_tr + n_va : n]
    metrics = evaluate(te_labels, scores, w_phys[n_tr + n_va : n],
                       w_match[n_tr + n_va : n], te_pt)
    metrics.update(n_test=int(len(te_labels)), epochs_run=trainer.current_epoch, seed=seed)

    np.savez_compressed(run_dir / "preds.npz", scores=scores, labels=te_labels,
                        w_phys=w_phys[n_tr + n_va : n], w_match=w_match[n_tr + n_va : n],
                        pt=te_pt)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[{arm}_s{seed}] " + json.dumps({k: round(v, 4) for k, v in metrics.items()
                                            if isinstance(v, float)}))
    return metrics


# -------------------------------------------------------------------- figures
def make_figures():
    import matplotlib
    matplotlib.use("Agg")
    from make_plots import BASE, INK2, MUTED, savefig  # noqa: F401  (rcParams)
    import matplotlib.pyplot as plt

    per_arm = {}
    for arm in ALL_ARMS:
        runs = sorted(MC_PROBES_DIR.glob(f"{arm}_s*/metrics.json"))
        if runs:
            per_arm[arm] = [json.loads(p.read_text()) for p in runs]
    if not per_arm:
        print("[figures] nothing to plot")
        return
    summary = {
        arm: {
            k: {"mean": float(np.mean([m[k] for m in ms])),
                "std": float(np.std([m[k] for m in ms])),
                "n_seeds": len(ms)}
            for k in ms[0] if isinstance(ms[0][k], float)
        }
        for arm, ms in per_arm.items()
    }
    (MC_PROBES_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({a: {"matched_auc": s["matched_auc"], "physical_auc": s["physical_auc"]}
                      for a, s in summary.items()}, indent=2))

    # AUC bars (matched + physical, seed error bars).
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    arms = [a for a in ALL_ARMS if a in summary]
    y = np.arange(len(arms))[::-1]
    for ax, key, title in [(axes[0], "matched_auc", "pT-matched"),
                           (axes[1], "physical_auc", "physical spectrum")]:
        vals = [summary[a][key]["mean"] for a in arms]
        errs = [summary[a][key]["std"] for a in arms]
        ax.barh(y, vals, xerr=errs, height=0.55, color="#2a78d6")
        ax.set_yticks(y, arms, fontsize=8.5)
        ax.set_xlim(0.5, 1.0)
        ax.set_xlabel(f"weighted ROC AUC ({title})")
        for yi, v in zip(y, vals):
            ax.annotate(f"{v:.4f}", (v, yi), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8)
    fig.suptitle("Campaign 7 — QCD vs top on AOJ MC, identical probes (test split)")
    savefig(fig, "c7_auc_summary.png")

    # ROC overlay (seed-42 runs, matched weights) + rejection panel.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for arm in arms:
        p = MC_PROBES_DIR / f"{arm}_s42" / "preds.npz"
        if not p.exists():
            continue
        d = np.load(p)
        fpr, tpr, _ = roc_curve(d["labels"], d["scores"], sample_weight=d["w_match"])
        sel = tpr > 0.05
        axes[0].plot(tpr[sel], 1.0 / np.maximum(fpr[sel], 1e-9), lw=1.6, label=arm)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("signal efficiency (top)")
    axes[0].set_ylabel("background rejection 1/FPR")
    axes[0].legend(fontsize=7.5, frameon=False)
    bins = list(zip(PT_BIN_EDGES[:-1], PT_BIN_EDGES[1:]))
    for arm in arms:
        vals = [summary[arm].get(f"auc_pt_{int(lo)}_{int(hi)}", {}).get("mean") for lo, hi in bins]
        xs = [0.5 * (lo + hi) for (lo, hi), v in zip(bins, vals) if v is not None]
        ys = [v for v in vals if v is not None]
        axes[1].plot(xs, ys, marker="o", ms=4, lw=1.4, label=arm)
    axes[1].set_xlabel("jet pT bin centre [GeV]")
    axes[1].set_ylabel("weighted AUC (physical)")
    axes[1].legend(fontsize=7.5, frameon=False)
    fig.suptitle("Campaign 7 — rejection curves and pT dependence")
    savefig(fig, "c7_roc.png")


# ----------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=ALL_ARMS, choices=ALL_ARMS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--n-jets", type=int, default=None, help="cap total jets")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-pt-match", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()

    if args.figures_only:
        make_figures()
        return

    torch.set_float32_matmul_precision("medium")
    MC_PROBES_DIR.mkdir(parents=True, exist_ok=True)

    token_paths = run_exports(args.arms)
    meta = load_meta()
    n = len(meta["label"]) if args.n_jets is None else min(args.n_jets, len(meta["label"]))
    for p in token_paths.values():
        assert_export_aligned(p, meta)
    weights = build_weights(meta, n, pt_match=not args.no_pt_match)

    for arm in args.arms:
        pending = [s for s in args.seeds
                   if not (MC_PROBES_DIR / f"{arm}_s{s}" / "metrics.json").exists()]
        if not pending:
            print(f"[skip] {arm} (all seeds done)")
            continue
        if arm in token_paths:
            z, mask = build_export_inputs(arm, token_paths[arm], n)
        else:
            z, mask = build_untrained_inputs(arm, meta, n)
        print(f"[arm] {arm}: z {z.shape} ({z.dtype})")
        for seed in args.seeds:
            train_arm(arm, seed, z, mask, meta, weights, n, args)
        del z, mask

    make_figures()


if __name__ == "__main__":
    main()
