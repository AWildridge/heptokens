"""Column-pruned Arrow scans and small ragged Awkward batches for analysis."""

from pathlib import Path

import awkward as ak
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import vector

from .source import read_json, sha256
from .tables import SCHEMAS

vector.register_awkward()


def partitions(root):
    root = Path(root)
    ready = read_json(root / "READY.json")
    if sha256(root / "validation.json") != ready["validation_sha256"]:
        raise ValueError("Validation report differs from READY")
    for task in ready["source_partitions"]:
        partition = root / f"source-{task:05d}"
        if sha256(partition / "COMMIT.json") != ready["partition_commit_sha256"][partition.name]:
            raise ValueError("Partition commit differs from READY")
        yield partition


def scan_table(root, name, columns=None, filter=None, batch_size=65536):
    """Return an Arrow scanner; only READY-listed source partitions are read."""
    if name not in SCHEMAS:
        raise ValueError(f"Unknown table: {name}")
    paths = [str(p / (name + ".parquet")) for p in partitions(root)]
    return ds.dataset(paths, format="parquet").scanner(
        columns=columns, filter=filter, batch_size=batch_size
    )


def detector_batches(root, detector):
    """Yield model inputs and join keys separately, with no truth or weights.

    Every constituent is retained for selected jets. There is no padding,
    truncation, or assumed correspondence between detector constituent ranks.
    """
    if detector not in ["CMS", "ATLAS"]:
        raise ValueError("Detector must be CMS or ATLAS")
    for partition in partitions(root):
        jets = pq.ParquetFile(partition / f"inputs/{detector}/jets.parquet")
        csts = pq.ParquetFile(partition / f"inputs/{detector}/constituents.parquet")
        if jets.num_row_groups != csts.num_row_groups:
            raise ValueError("Reco jet/constituent row groups differ")
        for i in range(jets.num_row_groups):
            j = ak.from_arrow(
                jets.read_row_group(
                    i, columns=["px", "py", "pz", "energy", "jet_id", "event_id", "multiplicity"]
                )
            )
            c = ak.from_arrow(
                csts.read_row_group(
                    i, columns=["px", "py", "pz", "energy", "reco_charge", "object_type"]
                )
            )
            if int(ak.sum(j.multiplicity)) != len(c):
                raise ValueError("Reco ragged offsets are inconsistent")
            jet_p4 = ak.zip(dict(px=j.px, py=j.py, pz=j.pz, E=j.energy), with_name="Momentum4D")
            cst_p4 = ak.zip(dict(px=c.px, py=c.py, pz=c.pz, E=c.energy), with_name="Momentum4D")
            yield dict(
                inputs=dict(
                    jets=jet_p4,
                    constituents=ak.unflatten(cst_p4, j.multiplicity),
                    charge=ak.unflatten(c.reco_charge, j.multiplicity),
                    object_type=ak.unflatten(c.object_type, j.multiplicity),
                ),
                keys=dict(event_id=j.event_id, jet_id=j.jet_id),
            )
