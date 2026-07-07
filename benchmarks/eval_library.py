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

import numpy as np
import torch

from common import FIGURES_DIR, LIBRARY_CRITERION, SEED, TOKENIZER_DIR, make_datamodule, make_transforms

from heptokens.models.contrastive import LitContrastiveTokenizer
from heptokens.models.utils import (
    compute_embedding_collapse,
    compute_knn_pid_purity,
    compute_retrieval_precision,
)
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


@torch.no_grad()
def embedding_providers(raw, scaled, mask):
    """Yield (name, flat_embedding [M, D]) on valid constituents."""
    out = {}

    def encode_with(model, csts):
        batch = {"csts": csts.to(DEVICE), "mask": mask.to(DEVICE)}
        return model.encode(batch)[0][mask].cpu()

    for name in ["library_aoj", "library_aoj_cone", "library_aoj_hard"]:
        model = load_tokenizer(name)
        if model is not None:
            out[name.replace("_aoj", "")] = encode_with(model, scaled)

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

    return out


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


def pair_distance_curves(providers, raw_flat, n_pairs=200_000):
    """Cosine distance vs dpT%% for same-PID/charge/region pairs (cliff check)."""
    rng = np.random.default_rng(SEED)
    n = len(raw_flat)
    a = rng.integers(0, n, n_pairs)
    b = rng.integers(0, n, n_pairs)
    raw_np = raw_flat.numpy()
    same = (
        (raw_np[a, 3] == raw_np[b, 3])
        & (raw_np[a, 4] == raw_np[b, 4])
        & (np.abs(raw_np[a, 1] - raw_np[b, 1]) <= 0.1)
        & (np.abs(np.remainder(raw_np[a, 2] - raw_np[b, 2] + np.pi, 2 * np.pi) - np.pi) <= 0.1)
        & (a != b)
    )
    a, b = a[same], b[same]
    dpt = np.abs(raw_np[a, 0] - raw_np[b, 0]) / np.maximum(raw_np[a, 0], raw_np[b, 0]).clip(1e-6)

    curves = {}
    for name in ["library", "library_hard"]:
        if name not in providers:
            continue
        z = torch.nn.functional.normalize(providers[name], dim=-1).numpy()
        dist = 1 - (z[a] * z[b]).sum(1)
        bins = np.linspace(0, 0.5, 21)
        med = [
            np.median(dist[(dpt >= lo) & (dpt < hi)])
            if ((dpt >= lo) & (dpt < hi)).sum() > 50
            else np.nan
            for lo, hi in zip(bins[:-1], bins[1:])
        ]
        curves[name] = (0.5 * (bins[1:] + bins[:-1]), np.array(med))
    return curves


def main() -> None:
    print("[collect] AOJ validation constituents ...")
    raw, scaled, mask = collect()
    raw_flat = raw[mask]
    pid_flat = raw_flat[:, 3]

    providers = embedding_providers(raw, scaled, mask)
    print(f"[providers] {list(providers)}  ({len(raw_flat):,} constituents)")

    rng = np.random.default_rng(SEED)
    sub = torch.from_numpy(rng.choice(len(raw_flat), min(4000, len(raw_flat)), replace=False))

    results = {}
    for name, z in providers.items():
        res = compute_retrieval_precision(z[sub], raw_flat[sub], k=8, **CRIT)
        purity = compute_knn_pid_purity(z[sub], pid_flat[sub], k=8)
        rank = compute_embedding_collapse(z[sub])["effective_rank"]
        results[name] = {
            "retrieval_precision_at_8": float(res["precision_at_k"]),
            "criterion_density": float(res["criterion_density"]),
            "knn_pid_purity": float(purity),
            "effective_rank": float(rank),
            "within_pid_pt_r2": within_pid_pt_r2(z, raw_flat),
            "dim": int(z.shape[-1]),
        }
        print(f"[{name:18s}] " + json.dumps(results[name]))

    curves = pair_distance_curves(providers, raw_flat)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (FIGURES_DIR / "library_metrics.json").write_text(json.dumps(results, indent=2))
    np.savez(
        FIGURES_DIR / "library_curves.npz",
        **{f"{k}_x": v[0] for k, v in curves.items()},
        **{f"{k}_y": v[1] for k, v in curves.items()},
    )
    print(f"[done] metrics + curves written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
