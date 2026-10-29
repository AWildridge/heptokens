"""Typed ROOT TTrees, explicit missingness, and streaming uproot access."""

import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from . import SCHEMA_VERSION

ROOT_FILENAME = "labels.root"

F = "float64"
S = "string"
INT = "int64"
B = "bool"
F32 = "float32"
CARTESIAN = {k: F32 for k in ["px", "py", "pz", "energy"]}
P4 = {k: F for k in ["px", "py", "pz", "energy", "pt", "eta", "phi", "rapidity", "mass"]}
P4.update(mass2_raw=F, mass_valid=B)
KIN = dict(
    ht=F,
    n_jets=INT,
    n_b_jets=INT,
    n_c_jets=INT,
    heavy_flavor_valid=B,
    dijet_mass=F,
    dijet_delta_y=F,
    dijet_delta_phi=F,
    dijet_valid=B,
)
NODE = dict(
    event_id=S,
    jet_id=S,
    node_id=S,
    parent_id=S,
    constituent_indices="var * int64",
    is_leaf=B,
    child_hard_id=S,
    child_soft_id=S,
    delta_r=F,
    z=F,
    kt=F,
    shower_category=S,
    shower_category_valid=B,
    **P4,
)
ANGLE = dict(
    event_id=S,
    jet_id=S,
    angle_id=S,
    first_node_id=S,
    second_node_id=S,
    daughter_branch=S,
    psi=F,
    cos2psi=F,
    sin2psi=F,
    geometric_valid=B,
    resolution_valid=B,
    valid=B,
)
SCHEMAS = {
    "bookkeeping/events": dict(
        event_id=S,
        generator_event_id=S,
        truth_fingerprint=S,
        sample_id=S,
        source_id=INT,
        source_uri=S,
        run="uint64",
        lumi="uint64",
        event="uint64",
        export_event=INT,
        chunk_id=S,
        delphes_seed=INT,
        split=S,
        original_weight=F,
        generator_weights="var * float64",
        lhe_weights_json=S,
        cross_section_pb=F,
        sample_sum_weights=F,
        physical_weight_pb=F,
        normalization_valid=B,
        generator_ht=F,
        generator_ht_valid=B,
        generator_pthat=F,
        generator_pthat_valid=B,
        selection_low_GeV=F,
        selection_high_GeV=F,
        selection_variable=S,
        synchronization_valid=B,
    ),
    "truth/events": dict(
        event_id=S,
        incoming_pdg_ids="var * int64",
        incoming_flavors_valid=B,
        hard_process_id=INT,
        hard_process_valid=B,
        **KIN,
    ),
    "truth/particles": dict(
        event_id=S,
        particle_index=INT,
        pdg_id="int32",
        charge=F,
        charge_valid=B,
        status=INT,
        is_pileup=B,
        x_mm=F,
        y_mm=F,
        z_mm=F,
        generated_mass=F,
        vertex_ct_mm=F,
        vertex_time_valid=B,
        mother_indices="var * int64",
        ancestry_valid=B,
        **CARTESIAN,
    ),
    "truth/jets": dict(
        event_id=S,
        jet_id=S,
        constituent_indices="var * int64",
        multiplicity=INT,
        charged_multiplicity=INT,
        charged_multiplicity_valid=B,
        heavy_hadron_flavor=S,
        heavy_hadron_pdg_ids="var * int64",
        heavy_hadron_flavor_valid=B,
        parton_flavor=S,
        parton_pdg_ids="var * int64",
        parton_flavor_valid=B,
        **P4,
    ),
    "truth/nodes": NODE,
    "truth/angles": ANGLE,
}
for detector in ["CMS", "ATLAS"]:
    SCHEMAS[f"inputs/{detector}/jets"] = dict(
        event_id=S,
        jet_id=S,
        detector=S,
        original_index=INT,
        original_split=INT,
        constituent_indices="var * int64",
        multiplicity=INT,
        **P4,
    )
    SCHEMAS[f"inputs/{detector}/events"] = dict(
        event_id=S,
        detector=S,
        **{k: v for k, v in KIN.items() if k not in ["n_b_jets", "n_c_jets", "heavy_flavor_valid"]},
    )
    SCHEMAS[f"inputs/{detector}/constituents"] = dict(
        event_id=S,
        jet_id=S,
        constituent_index=INT,
        stored_index=INT,
        detector=S,
        object_type="int8",
        reco_charge="int8",
        **CARTESIAN,
    )
    SCHEMAS[f"inputs/{detector}/nodes"] = {
        k: v for k, v in NODE.items() if k not in ["shower_category", "shower_category_valid"]
    }
    SCHEMAS[f"inputs/{detector}/angles"] = ANGLE
    SCHEMAS[f"targets/{detector}/jet_matches"] = dict(
        event_id=S,
        reco_jet_id=S,
        truth_jet_id=S,
        matched=B,
        unmatched=B,
        ambiguous=B,
        candidate_count=INT,
        delta_r=F,
        pt_response=F,
        mass_response=F,
    )
    SCHEMAS[f"targets/{detector}/truth_jet_coverage"] = dict(
        event_id=S, truth_jet_id=S, matched=B, reco_jet_id=S
    )
    # A LIST of truth particle IDs is intentional: a tower can aggregate many
    # stable particles. Null means no retained reference, rather than zero matches.
    SCHEMAS[f"targets/{detector}/object_truth"] = dict(
        event_id=S,
        jet_id=S,
        constituent_index=INT,
        truth_particle_indices="var * int64",
        association_valid=B,
        species_valid=B,
        ancestry_valid=B,
        unavailable_reason=S,
    )


def schema_for(name, metadata):
    fields = {}
    for key, dtype in SCHEMAS[name].items():
        unit = (
            "GeV"
            if key
            in [
                "px",
                "py",
                "pz",
                "energy",
                "pt",
                "mass",
                "kt",
                "ht",
                "dijet_mass",
                "generated_mass",
                "generator_ht",
                "generator_pthat",
                "selection_low_GeV",
                "selection_high_GeV",
            ]
            else "radian"
            if key in ["phi", "psi", "dijet_delta_phi"]
            else "mm"
            if key.endswith("_mm")
            else "pb"
            if key.endswith("_pb")
            else "GeV^2"
            if key == "mass2_raw"
            else "1"
        )
        fields[key] = dict(type=dtype, unit=unit, presence=key + "__present")
    return dict(fields=fields, provenance=metadata)


def branch_arrays(name, rows):
    """Encode null scalars/lists using a separate boolean presence branch.

    Empty lists and empty strings are valid values. A false __present branch
    marks a missing value, independently of every physics validity flag.
    """
    arrays = {}
    for key, dtype in SCHEMAS[name].items():
        values = [row.get(key) for row in rows]
        arrays[key + "__present"] = np.asarray([v is not None for v in values], dtype=np.bool_)
        if dtype.startswith("var * "):
            items = [v if v is not None else [] for v in values]
            offsets = np.concatenate(([0], np.cumsum([len(v) for v in items], dtype=np.int64)))
            content = np.asarray([x for v in items for x in v], dtype=dtype[6:])
            arrays[key] = ak.Array(
                ak.contents.ListOffsetArray(
                    ak.index.Index64(offsets), ak.contents.NumpyArray(content)
                )
            )
        elif dtype == "string":
            arrays[key] = ak.Array([v if v is not None else "" for v in values])
        else:
            arrays[key] = np.asarray([v if v is not None else 0 for v in values], dtype=dtype)
    return arrays


def read_arrays(tree, name, columns=None, entry_start=None, entry_stop=None):
    """Read selected logical columns and restore missing values as Awkward options."""
    columns = list(SCHEMAS[name]) if columns is None else list(columns)
    if not columns or any(c not in SCHEMAS[name] for c in columns):
        raise ValueError(f"Unknown or empty columns for {name}: {columns}")
    physical = [v for c in columns for v in (c, c + "__present")]
    raw = tree.arrays(physical, entry_start=entry_start, entry_stop=entry_stop, library="ak")
    return ak.zip({c: ak.mask(raw[c], raw[c + "__present"]) for c in columns}, depth_limit=1)


def open_root(path):
    # Canonical local ROOT handler: avoid the stalled FSSpecSource observed at Purdue.
    return uproot.open(
        path, handler=uproot.source.file.MemmapSource, object_cache=None, array_cache=None
    )


def iter_table(partition, name, columns=None, batch_size=4096):
    if name not in SCHEMAS or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("Unknown table or invalid batch size")
    with open_root(Path(partition) / ROOT_FILENAME) as file:
        tree = file[name]
        for start in range(0, tree.num_entries, batch_size):
            yield read_arrays(tree, name, columns, start, min(start + batch_size, tree.num_entries))


def read_rows(partition, name):
    return [row for batch in iter_table(partition, name) for row in ak.to_list(batch)]


class TableWriters:
    """One ROOT file per source, with a typed TTree per logical table."""

    def __init__(self, root, metadata):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.counts = dict.fromkeys(SCHEMAS, 0)
        self.closed = False
        self.file = uproot.recreate(self.root / ROOT_FILENAME, compression=uproot.ZSTD(4))
        try:
            self.file["label_schema"] = json.dumps(
                dict(
                    schema=SCHEMA_VERSION,
                    tables={name: schema_for(name, metadata) for name in SCHEMAS},
                    missingness="Mask placeholder values wherever FIELD__present is false.",
                ),
                sort_keys=True,
            )
            self.writers = {}
            for name, schema in SCHEMAS.items():
                branches = dict(schema)
                branches.update({key + "__present": "bool" for key in schema})
                self.writers[name] = self.file.mktree(name, branches)
        except BaseException:
            self.close()
            raise

    def write(self, tables):
        if self.closed:
            raise ValueError("ROOT writer is closed")
        for name, rows in tables.items():
            if rows:
                self.writers[name].extend(branch_arrays(name, rows))
                self.counts[name] += len(rows)

    def close(self):
        if not self.closed:
            self.file.close()
            self.closed = True
