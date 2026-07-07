"""Convert an Aspen Open Jets shard to the heptokens two-group schema.

Input: results/benchmarks/data/aoj/RunG_batch0.h5 (record 16505) with
    PFCands        [N, 150, 11]  (px, py, pz, E, d0, d0Err, dz, dzErr,
                                  charge, pdgId, puppiWeight)
    jet_kinematics [N, 4]        (pt, eta, phi, m)

Output: results/benchmarks/data/aoj_structured.h5
    csts/{pt, eta, phi, pid, charge, valid}   — ABSOLUTE detector eta/phi
    jets/{pt, eta, phi, m, nconst, label}     — label is a dummy (real data)
plus a PassthroughColumnsTransformer scaler (log-quantile on pt/eta/phi,
PID and charge pass through untouched).

This is real CMS collision data: no truth labels — evaluation on AOJ is
label-free (library metrics), per the benchmark protocol.
"""

import h5py
import joblib
import numpy as np

from common import AOJ_CST_SCALER, AOJ_JET_SCALER, AOJ_RAW_H5, AOJ_STRUCT_H5, SEED

from heptokens.data.transforms import (
    PassthroughColumnsTransformer,
    create_preprocessing_transformer,
)

CHARGE_COL, PDG_COL, E_COL = 8, 9, 3


def main() -> None:
    print(f"Reading {AOJ_RAW_H5} ...")
    with h5py.File(AOJ_RAW_H5, "r") as f:
        pf = f["PFCands"][:]  # [N, 150, 11] float32
        jk = f["jet_kinematics"][:]  # [N, 4] (pt, eta, phi, m)

    valid = pf[..., E_COL] > 0
    px, py, pz = pf[..., 0], pf[..., 1], pf[..., 2]
    pt = np.hypot(px, py)
    # Guard padded rows (pt=0) before dividing.
    safe_pt = np.where(valid, pt, 1.0)
    eta = np.arcsinh(pz / safe_pt) * valid
    phi = np.arctan2(py, px) * valid
    pt = pt * valid

    nconst = valid.sum(axis=1).astype(np.float32)
    print(
        f"{len(jk):,} jets | mean nconst {nconst.mean():.1f} | "
        f"jet pT [{jk[:, 0].min():.0f}, {jk[:, 0].max():.0f}] GeV"
    )

    print(f"Writing {AOJ_STRUCT_H5} ...")
    with h5py.File(AOJ_STRUCT_H5, "w") as f:
        jets = f.create_group("jets")
        jets.create_dataset("pt", data=jk[:, 0].astype(np.float32))
        jets.create_dataset("eta", data=jk[:, 1].astype(np.float32))
        jets.create_dataset("phi", data=jk[:, 2].astype(np.float32))
        jets.create_dataset("m", data=jk[:, 3].astype(np.float32))
        jets.create_dataset("nconst", data=nconst)
        jets.create_dataset("label", data=np.zeros(len(jk), dtype=np.int64))  # dummy
        csts = f.create_group("csts")
        csts.create_dataset("pt", data=pt.astype(np.float32))
        csts.create_dataset("eta", data=eta.astype(np.float32))
        csts.create_dataset("phi", data=phi.astype(np.float32))
        csts.create_dataset("pid", data=pf[..., PDG_COL].astype(np.float32))
        csts.create_dataset("charge", data=pf[..., CHARGE_COL].astype(np.float32))
        csts.create_dataset("valid", data=valid)

    # ---- scalers: fit on the train split of the datamodule's permutation
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(jk))
    train_idx = np.sort(perm[: int(0.7 * len(jk))])

    feats = np.stack([pt, eta, phi, pf[..., PDG_COL], pf[..., CHARGE_COL]], axis=-1)
    train_csts = feats[train_idx][valid[train_idx]]
    if len(train_csts) > 2_000_000:
        sub = np.random.default_rng(SEED).choice(len(train_csts), 2_000_000, replace=False)
        train_csts = train_csts[sub]

    print(f"Fitting passthrough cst scaler on {len(train_csts):,} constituents ...")
    cst_scaler = PassthroughColumnsTransformer(
        create_preprocessing_transformer(
            mode="log_quantile", log_feature_indices=[0], n_quantiles=500, n_features=3
        ),
        continuous_indices=[0, 1, 2],
    )
    cst_scaler.fit(train_csts)
    joblib.dump(cst_scaler, AOJ_CST_SCALER)

    # Must match AOJ_DATASET_KW obj_features order: [pt, nconst].
    jet_scaler = create_preprocessing_transformer(mode="quantile", n_quantiles=500)
    jet_scaler.fit(np.stack([jk[:, 0], nconst], axis=1)[train_idx])
    joblib.dump(jet_scaler, AOJ_JET_SCALER)

    scaled = cst_scaler.transform(train_csts[:100_000])
    print("scaled continuous mean/std:", scaled[:, :3].mean(0).round(3), scaled[:, :3].std(0).round(3))
    print("pid/charge untouched:", np.array_equal(scaled[:, 3:], train_csts[:100_000, 3:]))
    print("Done.")


if __name__ == "__main__":
    main()
