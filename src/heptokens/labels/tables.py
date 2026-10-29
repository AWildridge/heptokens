"""Explicit Arrow schemas keep empty and entirely unavailable targets typed."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

F = pa.float64()
S = pa.string()
INT = pa.int64()
B = pa.bool_()
F32 = pa.float32()
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
    constituent_indices=pa.list_(INT),
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
        run=pa.uint64(),
        lumi=pa.uint64(),
        event=pa.uint64(),
        export_event=INT,
        chunk_id=S,
        delphes_seed=INT,
        split=S,
        original_weight=F,
        generator_weights=pa.list_(F),
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
        incoming_pdg_ids=pa.list_(INT),
        incoming_flavors_valid=B,
        hard_process_id=INT,
        hard_process_valid=B,
        **KIN,
    ),
    "truth/particles": dict(
        event_id=S,
        particle_index=INT,
        pdg_id=pa.int32(),
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
        mother_indices=pa.list_(INT),
        ancestry_valid=B,
        **CARTESIAN,
    ),
    "truth/jets": dict(
        event_id=S,
        jet_id=S,
        constituent_indices=pa.list_(INT),
        multiplicity=INT,
        charged_multiplicity=INT,
        charged_multiplicity_valid=B,
        heavy_hadron_flavor=S,
        heavy_hadron_pdg_ids=pa.list_(INT),
        heavy_hadron_flavor_valid=B,
        parton_flavor=S,
        parton_pdg_ids=pa.list_(INT),
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
        constituent_indices=pa.list_(INT),
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
        object_type=pa.int8(),
        reco_charge=pa.int8(),
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
        truth_particle_indices=pa.list_(INT),
        association_valid=B,
        species_valid=B,
        ancestry_valid=B,
        unavailable_reason=S,
    )


def schema_for(name, metadata):
    fields = []
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
        fields.append(pa.field(key, dtype, metadata={b"unit": unit.encode()}))
    return pa.schema(fields, metadata={k.encode(): str(v).encode() for k, v in metadata.items()})


class TableWriters:
    def __init__(self, root, metadata):
        self.root = Path(root)
        self.writers = {}
        self.schemas = {name: schema_for(name, metadata) for name in SCHEMAS}
        self.counts = dict.fromkeys(SCHEMAS, 0)

    def write(self, tables):
        for name, rows in tables.items():
            if not rows:
                continue
            table = pa.Table.from_pylist(rows, schema=self.schemas[name])
            if name not in self.writers:
                path = self.root / (name + ".parquet")
                path.parent.mkdir(parents=True, exist_ok=True)
                self.writers[name] = pq.ParquetWriter(path, self.schemas[name], compression="zstd")
            self.writers[name].write_table(table)
            self.counts[name] += len(rows)

    def close(self):
        for name, schema in self.schemas.items():
            if name in self.writers:
                self.writers[name].close()
            else:
                path = self.root / (name + ".parquet")
                path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pylist([], schema=schema), path, compression="zstd")
