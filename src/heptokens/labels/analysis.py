"""Column-pruned uproot scans and small ragged Awkward batches for analysis."""

from pathlib import Path

import awkward as ak
import numpy as np
import vector

from . import SCHEMA_VERSION
from .source import read_json, sha256
from .tables import ROOT_FILENAME, SCHEMAS, iter_table, open_root, read_arrays

vector.register_awkward()


def partitions(root):
    root = Path(root)
    ready = read_json(root / "READY.json")
    if ready["schema"] != SCHEMA_VERSION:
        raise ValueError("Unsupported label output schema; use a ROOT v2 dataset")
    if sha256(root / "validation.json") != ready["validation_sha256"]:
        raise ValueError("Validation report differs from READY")
    for task in ready["source_partitions"]:
        partition = root / f"source-{task:05d}"
        if sha256(partition / "COMMIT.json") != ready["partition_commit_sha256"][partition.name]:
            raise ValueError("Partition commit differs from READY")
        yield partition


def scan_table(root, name, columns=None, filter=None, batch_size=65536):
    """Yield column-pruned Awkward batches from READY-listed ROOT partitions.

    An optional filter is a callable returning a row mask for each batch.
    Missing values are decoded using the stored __present branches.
    """
    if name not in SCHEMAS:
        raise ValueError(f"Unknown table: {name}")
    if filter is not None and not callable(filter):
        raise ValueError("filter must be a callable returning an Awkward row mask")
    for partition in partitions(root):
        for batch in iter_table(partition, name, columns, batch_size):
            yield batch if filter is None else batch[filter(batch)]


def detector_batches(root, detector, batch_size=4096):
    """Yield model inputs and join keys separately, with no truth or weights.

    Constituent ranges follow jet multiplicities and are checked against their
    stored jet/event IDs. Physical ROOT basket boundaries do not define joins.
    """
    if detector not in ["CMS", "ATLAS"]:
        raise ValueError("Detector must be CMS or ATLAS")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    jet_name, cst_name = f"inputs/{detector}/jets", f"inputs/{detector}/constituents"
    for partition in partitions(root):
        with open_root(partition / ROOT_FILENAME) as file:
            jets, csts = file[jet_name], file[cst_name]
            offset = 0
            for start in range(0, jets.num_entries, batch_size):
                j = read_arrays(
                    jets,
                    jet_name,
                    ["px", "py", "pz", "energy", "jet_id", "event_id", "multiplicity"],
                    start,
                    min(start + batch_size, jets.num_entries),
                )
                counts = ak.to_numpy(j.multiplicity)
                stop = offset + int(np.sum(counts))
                c = read_arrays(
                    csts,
                    cst_name,
                    [
                        "px",
                        "py",
                        "pz",
                        "energy",
                        "reco_charge",
                        "object_type",
                        "jet_id",
                        "event_id",
                    ],
                    offset,
                    stop,
                )
                if len(c) != stop - offset or any(
                    not np.array_equal(ak.to_numpy(c[key]), np.repeat(ak.to_numpy(j[key]), counts))
                    for key in ["jet_id", "event_id"]
                ):
                    raise ValueError(
                        "Reco ragged offsets or constituent identities are inconsistent"
                    )
                offset = stop
                jet_p4 = ak.zip(dict(px=j.px, py=j.py, pz=j.pz, E=j.energy), with_name="Momentum4D")
                cst_p4 = ak.zip(dict(px=c.px, py=c.py, pz=c.pz, E=c.energy), with_name="Momentum4D")
                yield dict(
                    inputs=dict(
                        jets=jet_p4,
                        constituents=ak.unflatten(cst_p4, counts),
                        charge=ak.unflatten(c.reco_charge, counts),
                        object_type=ak.unflatten(c.object_type, counts),
                    ),
                    keys=dict(event_id=j.event_id, jet_id=j.jet_id),
                )
            if offset != csts.num_entries:
                raise ValueError("Dangling reconstructed constituents")
