"""Download + convert Aspen Open Jets MC (record 18610) for Campaign 7.

QCD jets come in four pT bins, each file carrying a per-jet ``weight`` dataset
(cross-section weights — the record's pt_weights.py contract for combining
bins). Top jets are ZPrime->ttbar at several mass points. This script:

    1. downloads each raw file (sequential, resumable-ish, deleted after use —
       only ~23 GB of disk are free),
    2. applies a common jet-pT window (default [300, 1200] GeV) so neither
       class can be identified by phase space alone,
    3. subsamples to the requested budget, scaling kept weights by 1/keep_frac
       so per-bin Σweight (the physics) is preserved,
    4. assembles results/benchmarks/data/aojmc_structured.h5 in the heptokens
       two-group schema with jets/{...,label,weight,origin}.

NO scaler is fitted here: the MC is consumed with the frozen real-AOJ scalers
(aoj_cst_scaler/aoj_jet_scaler) — the release contract. The script prints the
diagnostics that decide whether that contract holds on MC (PID coverage vs the
encoder vocabulary, scaler saturation fraction, per-bin Σweight).

Usage:
    python prepare_aoj_mc.py [--qcd-per-bin 62500] [--top-total 250000]
                             [--pt-min 300] [--pt-max 1200] [--smoke]
"""

import argparse
import json
import shutil
import subprocess
import sys

import h5py
import joblib
import numpy as np

from common import AOJ_CST_SCALER, AOJMC_RAW_DIR, AOJMC_STRUCT_H5, SEED

BASE_URL = "https://www.fdr.uni-hamburg.de/record/18610/files"
CHARGE_COL, PDG_COL, E_COL = 8, 9, 3
ENCODER_PDG_IDS = (22, 211, -211, 130, 11, -11, 13, -13, 1, 2)  # particle_encoder.DEFAULT_PDG_IDS

# QCD: one batch per pT bin is far more than the budget needs.
QCD_FILES = [
    ("QCD_Pt300_batch0.h5", "qcd_pt300"),
    ("QCD_Pt470_batch0.h5", "qcd_pt470"),
    ("QCD_Pt600_batch0.h5", "qcd_pt600"),
    ("QCD_Pt800_batch0.h5", "qcd_pt800"),
]
# Top: iterate mass points x batches until the in-window quota is met (missing
# batches on the record are skipped gracefully).
TOP_MASSES = [(1000, 100), (1200, 120), (2500, 250), (4000, 400)]
TOP_MAX_BATCHES = 10


def free_gb(path) -> float:
    return shutil.disk_usage(path).free / 1e9


def download(fname: str, dest) -> bool:
    """curl one record file; returns False on failure (e.g. batch doesn't exist)."""
    url = f"{BASE_URL}/{fname}?download=1"
    print(f"[download] {fname} ...", flush=True)
    r = subprocess.run(
        ["curl", "-sL", "--fail", "--retry", "3", "--max-time", "3000", "-o", str(dest), url],
        check=False,
    )
    if r.returncode != 0:
        dest.unlink(missing_ok=True)
        print(f"[download] {fname} FAILED (rc={r.returncode}) — skipping")
        return False
    print(f"[download] {fname} done ({dest.stat().st_size / 1e9:.2f} GB)")
    return True


def derive_constituents(pf: np.ndarray):
    """PFCands -> (pt, eta, phi, pid, charge, valid) — identical to prepare_aoj.py."""
    valid = pf[..., E_COL] > 0
    px, py, pz = pf[..., 0], pf[..., 1], pf[..., 2]
    pt = np.hypot(px, py)
    safe_pt = np.where(valid, pt, 1.0)
    eta = np.arcsinh(pz / safe_pt) * valid
    phi = np.arctan2(py, px) * valid
    pt = pt * valid
    return pt, eta, phi, pf[..., PDG_COL], pf[..., CHARGE_COL], valid


def process_file(raw_path, quota, pt_min, pt_max, rng):
    """Window-cut, subsample (weight-preserving), derive constituents.

    quota=None keeps every in-window jet (used for top files; the pooled top
    sample is subsampled once at assembly time instead).
    """
    with h5py.File(raw_path, "r") as f:
        jk = f["jet_kinematics"][:]
        weight = f["weight"][:].astype(np.float64) if "weight" in f else np.ones(len(jk))
        in_window = (jk[:, 0] >= pt_min) & (jk[:, 0] <= pt_max)
        idx = np.flatnonzero(in_window)
        if quota is not None and len(idx) > quota:
            keep_frac = quota / len(idx)
            idx = np.sort(rng.choice(idx, quota, replace=False))
        else:
            keep_frac = 1.0
        pf = f["PFCands"][:][idx] if len(idx) else np.zeros((0, 150, 11), np.float32)

    jk, weight = jk[idx], weight[idx] / keep_frac
    pt, eta, phi, pid, charge, valid = derive_constituents(pf)
    return dict(
        jet_pt=jk[:, 0].astype(np.float32),
        jet_eta=jk[:, 1].astype(np.float32),
        jet_phi=jk[:, 2].astype(np.float32),
        jet_m=jk[:, 3].astype(np.float32),
        nconst=valid.sum(axis=1).astype(np.float32),
        weight=weight.astype(np.float32),
        cst_pt=pt.astype(np.float32),
        cst_eta=eta.astype(np.float32),
        cst_phi=phi.astype(np.float32),
        cst_pid=pid.astype(np.float32),
        cst_charge=charge.astype(np.float32),
        valid=valid,
    )


def fetch_chunk(fname, origin, quota, args, rng):
    """Download -> process -> cache chunk npz -> delete raw. Returns chunk path or None."""
    tag = "all" if quota is None else str(quota)
    chunk = AOJMC_RAW_DIR / f"{fname.replace('.h5', '')}_q{tag}_w{args.pt_min:g}-{args.pt_max:g}.npz"
    if chunk.exists():
        print(f"[chunk] {chunk.name} exists, skipping download")
        return chunk
    if free_gb(AOJMC_RAW_DIR) < 4.0:
        sys.exit(f"ABORT: <4 GB free before downloading {fname}")
    raw = AOJMC_RAW_DIR / fname
    if not raw.exists() and not download(fname, raw):
        return None
    try:
        data = process_file(raw, quota, args.pt_min, args.pt_max, rng)
    finally:
        raw.unlink(missing_ok=True)  # raw shards are re-downloadable; disk is not
    if len(data["weight"]) == 0:
        print(f"[chunk] {fname}: 0 jets in window — dropped")
        return None
    np.savez_compressed(chunk, origin=origin, **data)
    print(f"[chunk] {chunk.name}: {len(data['weight']):,} jets")
    return chunk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qcd-per-bin", type=int, default=62_500)
    parser.add_argument("--top-total", type=int, default=250_000)
    parser.add_argument("--pt-min", type=float, default=300.0)
    parser.add_argument("--pt-max", type=float, default=1200.0)
    parser.add_argument("--smoke", action="store_true", help="tiny quotas for a pipeline check")
    args = parser.parse_args()
    if args.smoke:
        args.qcd_per_bin, args.top_total = 2_000, 8_000

    AOJMC_RAW_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    chunks = []  # (path, label)
    for fname, origin in QCD_FILES:
        c = fetch_chunk(fname, origin, args.qcd_per_bin, args, rng)
        if c is not None:
            chunks.append((c, 0))

    # Top: walk mass points x batches until the pooled in-window yield covers
    # the quota (batch files that don't exist on the record are skipped).
    top_yield = 0
    for batch in range(TOP_MAX_BATCHES):
        if top_yield >= args.top_total:
            break
        for mass, width in TOP_MASSES:
            if top_yield >= args.top_total:
                break
            fname = f"top_tops_ZPrime_M{mass}_W{width}_batch{batch}.h5"
            c = fetch_chunk(fname, f"top_M{mass}_b{batch}", None, args, rng)
            if c is not None:
                top_yield += len(np.load(c)["weight"])
                chunks.append((c, 1))
    print(f"[top] pooled in-window yield: {top_yield:,} (quota {args.top_total:,})")

    # ---- assemble ---------------------------------------------------------
    fields = ["jet_pt", "jet_eta", "jet_phi", "jet_m", "nconst", "weight",
              "cst_pt", "cst_eta", "cst_phi", "cst_pid", "cst_charge", "valid"]
    parts, labels, origins, origin_names = [], [], [], []
    for path, label in chunks:
        d = np.load(path)
        n = len(d["weight"])
        origin_names.append(str(d["origin"]))
        parts.append({k: d[k] for k in fields})
        labels.append(np.full(n, label, dtype=np.int64))
        origins.append(np.full(n, len(origin_names) - 1, dtype=np.int8))
    data = {k: np.concatenate([p[k] for p in parts]) for k in fields}
    label = np.concatenate(labels)
    origin = np.concatenate(origins)

    # Pooled top subsample to quota (weight-preserving).
    top_idx = np.flatnonzero(label == 1)
    if len(top_idx) > args.top_total:
        keep_frac = args.top_total / len(top_idx)
        keep = np.sort(rng.choice(top_idx, args.top_total, replace=False))
        sel = np.concatenate([np.flatnonzero(label == 0), keep])
        sel.sort()
        for k in fields:
            data[k] = data[k][sel]
        scale = np.where(label[sel] == 1, 1.0 / keep_frac, 1.0).astype(np.float32)
        data["weight"] = data["weight"] * scale
        label, origin = label[sel], origin[sel]

    n = len(label)
    if not (np.any(label == 0) and np.any(label == 1)):
        sys.exit("ABORT: need both classes present")

    print(f"Writing {AOJMC_STRUCT_H5} ({n:,} jets: "
          f"{int((label == 0).sum()):,} QCD / {int((label == 1).sum()):,} top) ...")
    with h5py.File(AOJMC_STRUCT_H5, "w") as f:
        jets = f.create_group("jets")
        jets.create_dataset("pt", data=data["jet_pt"])
        jets.create_dataset("eta", data=data["jet_eta"])
        jets.create_dataset("phi", data=data["jet_phi"])
        jets.create_dataset("m", data=data["jet_m"])
        jets.create_dataset("nconst", data=data["nconst"])
        jets.create_dataset("label", data=label)
        jets.create_dataset("weight", data=data["weight"])
        jets.create_dataset("origin", data=origin)
        jets.attrs["origin_names"] = json.dumps(origin_names)
        jets.attrs["pt_window"] = [args.pt_min, args.pt_max]
        csts = f.create_group("csts")
        csts.create_dataset("pt", data=data["cst_pt"])
        csts.create_dataset("eta", data=data["cst_eta"])
        csts.create_dataset("phi", data=data["cst_phi"])
        csts.create_dataset("pid", data=data["cst_pid"])
        csts.create_dataset("charge", data=data["cst_charge"])
        csts.create_dataset("valid", data=data["valid"])

    # ---- diagnostics (the frozen-scaler contract on MC) --------------------
    print("\n=== diagnostics ===")
    for i, name in enumerate(origin_names):
        m = origin == i
        if m.any():
            print(f"  {name:16s} n={int(m.sum()):7,}  sum(w)={data['weight'][m].sum():.4g}  "
                  f"<pT>={data['jet_pt'][m].mean():.0f} GeV")
    vpid = data["cst_pid"][data["valid"]]
    codes, counts = np.unique(vpid.astype(np.int64), return_counts=True)
    unknown = sum(c for code, c in zip(codes, counts) if code not in ENCODER_PDG_IDS)
    print(f"  PID codes: {dict(zip(codes.tolist(), counts.tolist()))}")
    print(f"  PID outside encoder vocabulary: {unknown / len(vpid):.2%} -> 'other' bucket")
    scaler = joblib.load(AOJ_CST_SCALER)
    vpt = data["cst_pt"][data["valid"]]
    sat = float((vpt > 500.0).mean())  # real-AOJ csts are essentially all below this
    print(f"  constituent pT > 500 GeV (scaler saturation risk): {sat:.3%}")
    sub = rng.choice(len(vpid), min(100_000, len(vpid)), replace=False)
    feats = np.stack([vpt, data["cst_eta"][data["valid"]], data["cst_phi"][data["valid"]],
                      vpid, data["cst_charge"][data["valid"]]], axis=-1)[sub]
    scaled = scaler.transform(feats)
    print(f"  frozen-scaler output mean/std (pt,eta,phi): "
          f"{scaled[:, :3].mean(0).round(3)} {scaled[:, :3].std(0).round(3)}")
    for lbl, name in [(0, "QCD"), (1, "top")]:
        m = label == lbl
        w = data["weight"][m]
        ess = float(w.sum() ** 2 / (w**2).sum())
        print(f"  {name}: n={int(m.sum()):,} ESS={ess:,.0f} "
              f"pT [{data['jet_pt'][m].min():.0f}, {data['jet_pt'][m].max():.0f}]")
    print("Done.")


if __name__ == "__main__":
    main()
