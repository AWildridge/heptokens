"""Convert the flat 620k-jet file to the heptokens two-group schema and fit scalers.

Input : /home/abhijith/train.h5   feature [N, 150, 3] float16, label [N] int64
Output: results/benchmarks/data/jets5c.h5
            jets/{pt, nconst, label}
            csts/{pt, deta, dphi, valid}
        + cst_scaler.joblib (log-quantile, log on pt) and jet_scaler.joblib (quantile)

Scalers are fitted on the 70% train split of the same seed-42 permutation the
datamodule uses, so no validation/test information leaks into the scaling.
"""

import h5py
import joblib
import numpy as np

from common import CST_SCALER, DATA_DIR, JET_SCALER, RAW_H5, SEED, STRUCT_H5

from heptokens.data.transforms import create_preprocessing_transformer


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading {RAW_H5} ...")
    with h5py.File(RAW_H5, "r") as f:
        feats = f["feature"][:].astype(np.float32)  # [N, 150, 3]
        labels = f["label"][:].astype(np.int64)

    valid = np.any(feats != 0.0, axis=-1)  # [N, 150]
    jet_pt = (feats[..., 0] * valid).sum(axis=1)
    nconst = valid.sum(axis=1).astype(np.float32)
    print(f"{len(labels):,} jets | mean nconst {nconst.mean():.1f} | classes {np.bincount(labels)}")

    print(f"Writing {STRUCT_H5} ...")
    with h5py.File(STRUCT_H5, "w") as f:
        jets = f.create_group("jets")
        jets.create_dataset("pt", data=jet_pt.astype(np.float32))
        jets.create_dataset("nconst", data=nconst)
        jets.create_dataset("label", data=labels)
        csts = f.create_group("csts")
        csts.create_dataset("pt", data=feats[..., 0])
        csts.create_dataset("deta", data=feats[..., 1])
        csts.create_dataset("dphi", data=feats[..., 2])
        csts.create_dataset("valid", data=valid)

    # ---- fit scalers on the train split only (same permutation as the datamodule)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(labels))
    train_idx = np.sort(perm[: int(0.7 * len(labels))])

    train_valid = valid[train_idx]
    train_csts = feats[train_idx][train_valid]  # [n_valid_csts, 3]
    if len(train_csts) > 2_000_000:
        sub = np.random.default_rng(SEED).choice(len(train_csts), 2_000_000, replace=False)
        train_csts = train_csts[sub]

    print(f"Fitting cst scaler (log_quantile) on {len(train_csts):,} constituents ...")
    cst_scaler = create_preprocessing_transformer(
        mode="log_quantile", log_feature_indices=[0], n_quantiles=500, n_features=3
    )
    cst_scaler.fit(train_csts)
    joblib.dump(cst_scaler, CST_SCALER)

    print("Fitting jet scaler (quantile) ...")
    jet_scaler = create_preprocessing_transformer(mode="quantile", n_quantiles=500)
    jet_scaler.fit(np.stack([jet_pt, nconst], axis=1)[train_idx])
    joblib.dump(jet_scaler, JET_SCALER)

    scaled = cst_scaler.transform(train_csts[:100_000])
    print("scaled cst mean/std per col:", scaled.mean(0).round(3), scaled.std(0).round(3))
    print("Done.")


if __name__ == "__main__":
    main()
