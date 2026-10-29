"""Physics identities, truth-content certification, missing labels and publication."""

import gzip
import json
import math
from dataclasses import replace

import awkward as ak
import h5py
import numpy as np
import pytest
import uproot

from heptokens.labels.analysis import detector_batches, scan_table
from heptokens.labels.cli import account_weights, detector_catalogs
from heptokens.labels.physics import (
    Config,
    ca_tree,
    cartesian_fields,
    charge,
    common_jets,
    event_identity,
    event_kinematics,
    match_jets,
    p4_fields,
    plane_angles,
    split_for,
)
from heptokens.labels.produce import produce_source
from heptokens.labels.source import atomic_json, iter_events, sha256, truth_fingerprint
from heptokens.labels.tables import SCHEMAS, TableWriters, iter_table, open_root, read_rows
from heptokens.labels.validation import check_tree, validate_run


def particle(index, pid, p4):
    return dict(barcode=index, pdg_id=pid, p4=p4, generated_mass=0.0, position=[0.0, 0.0, 0.0, 0.0])


def jet(index, pt, eta, phi, event="e"):
    return dict(
        event_id=event,
        jet_id=index,
        **p4_fields(
            [pt * math.cos(phi), pt * math.sin(phi), pt * math.sinh(eta), pt * math.cosh(eta)]
        ),
    )


def test_identity_groups_replicas_and_detector_views():
    a = event_identity("63193", 1, 4, 5)
    assert a == event_identity("63193", 1, 4, 5)
    assert a != event_identity("63217", 1, 4, 5)
    assert a != event_identity("63193", 1, 4, 6)
    assert split_for(a) in ["train", "validation", "test"]


def test_truth_fingerprint_uses_content_not_order():
    a = [particle(1, 211, [10.0, 2.0, 1.0, 11.0]), particle(2, 22, [4.0, 1.0, 1.0, 5.0])]
    assert truth_fingerprint(a) == truth_fingerprint(a[::-1])
    b = [dict(a[0], pdg_id=321), a[1]]
    assert truth_fingerprint(a) != truth_fingerprint(b)


def test_charge_unknown_is_missing():
    assert charge(211) == 1 and charge(-211) == -1 and charge(22) == 0
    assert charge(999999) is None


def test_truth_jet_neutrino_muon_and_acceptance_selection():
    ps = [
        particle(1, 211, [40.0, 0, 0, 40.0]),
        particle(2, 12, [500.0, 0, 0, 500.0]),
        particle(3, 13, [10.0, 0, 0, 10.0]),
        particle(4, 22, [1.0, 0, 1000, 1001.0]),
    ]
    cfg = Config(jet_min_pt=20)
    a = common_jets(ps, cfg)
    assert len(a) == 1 and a[0]["constituent_indices"] == [1, 3]
    assert a[0]["p4"][0] == 50
    assert common_jets(ps, replace(cfg, include_muons=False))[0]["constituent_indices"] == [1]


def test_full_binary_tree_conserves_membership_and_p4():
    p4 = [[20.0, 0, 1, 21], [4.0, 1, 2, 5], [3.0, -0.5, 1, 4], [0.5, 0.2, -0.1, 1]]
    n = ca_tree(p4, [1, 2, 3, 4], "e", "j")
    assert len(n) == 7
    check_tree(n, [dict(jet_id="j", constituent_indices=[1, 2, 3, 4])])
    assert all(x["shower_category"] == "unknown" and not x["shower_category_valid"] for x in n)


def test_analytic_signed_angle_and_both_daughters():
    def node(name, p4, hard=None, soft=None):
        return dict(
            event_id="e",
            jet_id="j",
            node_id=name,
            is_leaf=hard is None,
            child_hard_id=hard,
            child_soft_id=soft,
            z=0.25,
            delta_r=0.2,
            kt=2.0,
            **p4_fields(p4),
        )

    # C is along +z. n1 along +x, n2 along (+x,-y), hence psi=-pi/4.
    n = [
        node("a", [0, 1, 3, 8], "b", "c"),
        node("b", [0, 1, 1, 3]),
        node("c", [0, 0, 2, 5], "d", "e"),
        node("d", [1, 1, 1, 2.5]),
        node("e", [-1, -1, 1, 2.5]),
    ]
    a = plane_angles(n, Config())[0]
    assert a["daughter_branch"] == "soft"
    assert a["psi"] == pytest.approx(-math.pi / 4)
    assert a["sin2psi"] == pytest.approx(-1)
    assert a["cos2psi"] == pytest.approx(0, abs=1e-12)
    assert a["valid"]
    n[1].update(is_leaf=False, child_hard_id="f", child_soft_id="g")
    n.extend([node("f", [0, 0.5, 0.5, 1.5]), node("g", [0, 0.5, 0.5, 1.5])])
    assert {
        a["daughter_branch"] for a in plane_angles(n, Config()) if a["first_node_id"] == "a"
    } == {"hard", "soft"}


def test_degenerate_planes_are_masked():
    n = ca_tree([[10, 0, 0, 10], [5, 0, 0, 5], [1, 0, 0, 1]], [1, 2, 3], "e", "j")
    angles = plane_angles(n, Config())
    assert len(angles) == 1
    assert not angles[0]["geometric_valid"] and not angles[0]["valid"]
    assert angles[0]["psi"] is None


def test_global_matching_maximizes_cardinality_before_distance():
    # Greedy picks r0->t0, stranding r1. The global solution matches both.
    reco = [jet("r0", 100, 0, 0.04), jet("r1", 100, 0, -0.05)]
    truth = [jet("t0", 100, 0, 0), jet("t1", 100, 0, 0.15)]
    m = match_jets(reco, truth, Config(match_dr=0.13))
    assert [a["truth_jet_id"] for a in m] == ["t1", "t0"]
    assert all(a["ambiguous"] for a in m)
    assert not any(
        a["matched"] for a in match_jets(reco, truth, Config(match_dr=0.13, reject_ambiguous=True))
    )


def test_matching_wraps_phi_and_respects_response():
    r = [jet("r", 100, 0, math.pi - 0.01)]
    t = [jet("t", 100, 0, -math.pi + 0.01)]
    assert match_jets(r, t, Config())[0]["delta_r"] == pytest.approx(0.02)
    assert not match_jets([jet("r", 10000, 0, 0)], [jet("t", 100, 0, 0)], Config())[0]["matched"]
    assert match_jets(r, [], Config())[0]["unmatched"]
    assert match_jets([], t, Config()) == []


def test_continuous_event_kinematics():
    r = event_kinematics([jet("a", 100, 0, 0), jet("b", 80, 0, math.pi)])
    assert r["ht"] == 180 and r["n_jets"] == 2
    assert r["dijet_mass"] == pytest.approx(math.sqrt(32000))
    assert r["n_b_jets"] is None and not r["heavy_flavor_valid"]


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / "raw"
    chunk = root / "sources/00000/chunk-000000"
    chunk.mkdir(parents=True)
    source = dict(
        task=0,
        record="63193",
        dataset="/QCD_Pt_300to470_test",
        uri="root://example/original",
        checksum="adler32:0",
    )
    record = dict(
        record="63193",
        dataset=source["dataset"],
        events=2,
        cross_section=dict(total_value="10", total_value_uncertainty="1", filter_efficiency="1"),
    )
    manifest = dict(files=[source], records=[record])
    (root / "provenance").mkdir()
    atomic_json(root / "provenance/cms-qcd-source-manifest.json", manifest)
    refs = [
        dict(
            run=1,
            lumi=10,
            event=i + 1,
            export_event=i,
            weight=w,
            generator_weights=[w],
            lhe_weights=None,
        )
        for i, w in enumerate([2.0, -1.0])
    ]
    ps = [
        particle(1, 211, [40.0, 0, 1.0, 41.0]),
        particle(2, 22, [8.0, 1, 1.0, 9.0]),
        particle(3, 13, [3.0, -0.2, 0.5, 4.0]),
    ]
    raw = "HepMC::Version 2.06.09\nHepMC::IO_GenEvent-START_EVENT_LISTING\n"
    for e in refs:
        raw += (
            f"E {e['export_event']} 0 0 0 0 0 0 1 0 0 0 1 {e['weight']}\n"
            "U GEV MM\nV -1 0 0 0 0 0 0 3 0\n"
        )
        for p in ps:
            raw += (
                f"P {p['barcode']} {p['pdg_id']} " + " ".join(map(str, p["p4"])) + " 0 1 0 0 0 0\n"
            )
    raw += "HepMC::IO_GenEvent-END_EVENT_LISTING\n"
    import hashlib

    original_hash = hashlib.sha256(raw.encode()).hexdigest()
    with gzip.open(chunk / "truth.hepmc.gz", "wt") as f:
        f.write(raw)
    with gzip.open(chunk / "truth-reference.json.gz", "wt") as f:
        json.dump(
            dict(source_key=source["uri"], events=refs, sha256=original_hash, scope="stable only"),
            f,
        )
    for detector in ["CMS", "ATLAS"]:
        (chunk / f"delphes_card_{detector}.tcl").write_text(
            (
                "module FastJetFinder FastJetFinder {\n set ParameterR 0.5\n set JetAlgorithm"
                " 6\n set JetPTMin 20\n set InputArray Test/particles\n}\n"
            )
        )
        order = [0, 1] if detector == "CMS" else [1, 0]
        with h5py.File(chunk / f"{detector}.h5", "w") as f:
            f.attrs.update(schema="delphes-ragged-p4-v1", detector=detector)
            for k in ["run", "lumi", "event", "export_event", "weight"]:
                f.create_dataset("events/" + k, data=[refs[i][k] for i in order])
            f.create_dataset("events/split", data=[0, 0])
            for k, values in dict(
                event_offsets=[0, 1, 2],
                jet_p4=[[51.0, 0.8, 2.5, 54.0]] * 2,
                constituent_offsets=[0, 3, 6],
                constituent_p4=[p["p4"] for p in ps] * 2,
                charge=[1, 0, -1] * 2,
                constituent_type=[1, 2, 1] * 2,
            ).items():
                f.create_dataset("Jet/" + k, data=values)
    validation = dict(
        status="passed",
        events=2,
        input_sha256=original_hash,
        detectors={
            d: dict(
                status="passed",
                truth_and_weights="passed",
                card_sha256=sha256(chunk / f"delphes_card_{d}.tcl"),
                stable_particle_counts=[3, 3],
            )
            for d in ["CMS", "ATLAS"]
        },
    )
    atomic_json(chunk / "validation.json", validation)
    names = [
        "CMS.h5",
        "ATLAS.h5",
        "truth.hepmc.gz",
        "truth-reference.json.gz",
        "validation.json",
        "delphes_card_CMS.tcl",
        "delphes_card_ATLAS.tcl",
    ]
    receipt = dict(
        status="complete",
        events=2,
        input_sha256=original_hash,
        source=source,
        seed=42,
        sha256={n: sha256(chunk / n) for n in names},
    )
    atomic_json(chunk / "receipt.json", receipt)
    atomic_json(
        root / "sources/00000/COMPLETE.json",
        dict(
            status="complete",
            source=source,
            exported_events=2,
            chunks=[dict(chunk="chunk-000000", events=2)],
            manifest_sha256=sha256(root / "provenance/cms-qcd-source-manifest.json"),
        ),
    )
    return root


def test_detector_mapping_is_by_identity_even_with_permuted_rows(campaign):
    events = list(iter_events(campaign / "sources/00000/chunk-000000", Config()))
    assert len(events) == 2 and events[0]["reference"]["weight"] == 2.0
    assert events[1]["reference"]["weight"] == -1.0
    assert events[0]["event_id"] != events[1]["event_id"]


def test_changed_truth_payload_is_rejected(campaign):
    with (campaign / "sources/00000/chunk-000000/truth.hepmc.gz").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        list(iter_events(campaign / "sources/00000/chunk-000000", Config()))


def test_slim_end_to_end_and_signed_normalization(campaign, tmp_path):
    normalization = account_weights(campaign, tmp_path / "weight-accounting", [0])
    assert normalization["valid"] and normalization["samples"]["63193"]["sum_weights"] == 1.0
    out = tmp_path / "labels"
    result = produce_source(campaign, out, 0, Config(), normalization=normalization)
    assert result["events"] == 2
    assert result["tables"]["truth/nodes"]["rows"] == 0
    jets = read_rows(out / "source-00000", "truth/jets")
    assert all(j["heavy_hadron_flavor"] == "unknown" and not j["parton_flavor_valid"] for j in jets)
    book = read_rows(out / "source-00000", "bookkeeping/events")
    assert [r["physical_weight_pb"] for r in book] == [20.0, -10.0]
    assert all(r["generator_ht"] is None and not r["generator_ht_valid"] for r in book)
    report = validate_run(out)
    assert (
        report["status"] == "passed"
        and report["unique_events"] == 2
        and report["campaign_complete"]
    )
    assert report["trees_valid"] is None and report["deferred_targets"]
    assert produce_source(campaign, out, 0, Config(), normalization=normalization)["events"] == 2
    with pytest.raises(ValueError, match="differs"):
        produce_source(campaign, out, 0, Config(radius=0.6), normalization=normalization)


def test_optional_trees_and_missing_flavor(campaign, tmp_path):
    out = tmp_path / "trees"
    result = produce_source(campaign, out, 0, Config(include_trees=True))
    assert result["tables"]["truth/nodes"]["rows"] == 10
    report = validate_run(out)
    assert report["trees_valid"]
    assert (
        report["summaries"]["63193/CMS/truth/raw_generator_weight"]["fractions"][
            "parton_flavor_missing/jets"
        ]
        == 1.0
    )


def test_partial_accounting_does_not_claim_unit_weights(campaign, tmp_path):
    p = campaign / "provenance/cms-qcd-source-manifest.json"
    a = json.loads(p.read_text())
    a["records"][0]["events"] = 3
    atomic_json(p, a)
    c = campaign / "sources/00000/COMPLETE.json"
    r = json.loads(c.read_text())
    r["manifest_sha256"] = sha256(p)
    atomic_json(c, r)
    result = account_weights(campaign, tmp_path / "partial-account", [0])
    assert not result["valid"] and result["samples"]["63193"]["sum_weights"] == 1.0


def test_slim_awkward_reader_separates_inputs_and_bookkeeping(campaign, tmp_path):
    out = tmp_path / "analysis"
    produce_source(campaign, out, 0, Config())
    validate_run(out)
    batches = list(detector_batches(out, "CMS", batch_size=1))
    assert sum(len(b["inputs"]["jets"]) for b in batches) == 2
    import awkward as ak

    assert [ak.to_list(ak.num(b["inputs"]["constituents"])) for b in batches] == [[3], [3]]
    assert set(batches[0]) == {"inputs", "keys"}
    scanned = list(scan_table(out, "bookkeeping/events", columns=["original_weight"]))
    assert scanned[0].fields == ["original_weight"]
    assert ak.to_list(scanned[0].original_weight) == [2.0, -1.0]
    detector_catalogs(out)
    assert (out / "CMS/source-00000.root").is_file()
    assert (out / "ATLAS/source-00000.root").is_file()
    with uproot.open(
        out / "CMS/source-00000.root", handler=uproot.source.file.MemmapSource
    ) as file:
        assert file["inputs/CMS/jets"].classname == "TTree"
        assert len(file["inputs/CMS/jets"].arrays(["jet_id", "px"], library="ak")) == 2
        assert (
            sum(
                len(a)
                for a in file["inputs/CMS/jets"].iterate(
                    ["jet_id", "constituent_indices"], step_size=1, library="ak"
                )
            )
            == 2
        )
    assert detector_catalogs(out)["sources"] == 1


def test_report_streams_one_event_and_preserves_weighted_sums(campaign, tmp_path, monkeypatch):
    import heptokens.labels.validation as validation

    out = tmp_path / "streaming"
    produce_source(campaign, out, 0, Config())
    original = validation.partition_batches
    monkeypatch.setattr(validation, "partition_batches", lambda p: original(p, events_per_batch=1))
    report = validate_run(out)
    counts = report["summaries"]["63193/CMS/truth/raw_generator_weight"]["counts"]
    assert counts["jets"] == 2 and counts["parton_flavor_unknown_sumw"] == 1.0
    assert counts["parton_flavor_unknown_sumw2"] == 5.0


def test_spacelike_mass_is_flagged_not_made_physical():
    p = p4_fields([100.0, 0.0, 0.0, 1.0])
    assert not p["mass_valid"] and p["mass"] is None and p["mass2_raw"] < 0


def test_jet_identity_is_independent_of_tree_or_matching_options():
    cfg = Config()
    assert (
        cfg.jet_definition_digest
        == replace(cfg, include_trees=True, match_dr=0.2).jet_definition_digest
    )
    assert cfg.jet_definition_digest != replace(cfg, radius=0.6).jet_definition_digest


def test_config_requires_explicit_boolean_for_expensive_stage():
    with pytest.raises(ValueError, match="boolean"):
        Config(include_trees="false")


def test_cartesian_slim_path_preserves_components_exactly():
    p = [100.25, -3.5, 4.25, 101.0]
    full = p4_fields(p)
    assert cartesian_fields(p) == {k: full[k] for k in ["px", "py", "pz", "energy"]}


def test_plot_and_reader_changes_do_not_invalidate_production(monkeypatch):
    import heptokens.labels.produce as producer

    before = producer.code_provenance()
    original = producer.sha256
    monkeypatch.setattr(
        producer,
        "sha256",
        lambda p: "plot-only-change" if str(p).endswith("validation.py") else original(p),
    )
    after = producer.code_provenance()
    assert before["software_id"] == after["software_id"]
    assert before["module_sha256"] != after["module_sha256"]


def test_root_missing_empty_jagged_and_numeric_precision(tmp_path):
    writer = TableWriters(tmp_path, {"test": "ROOT roundtrip"})
    records = [
        dict(
            event_id="a",
            event=2**64 - 1,
            generator_weights=None,
            original_weight=1.123456789012345,
            lhe_weights_json=None,
        ),
        dict(
            event_id="b", event=0, generator_weights=[], original_weight=-2.0, lhe_weights_json=""
        ),
        dict(event_id="c", event=42, generator_weights=[1.0, -0.125], original_weight=0.0),
    ]
    writer.write({"bookkeeping/events": records[:2]})
    writer.write(
        {"bookkeeping/events": records[2:], "truth/particles": [dict(event_id="a", px=1.123456789)]}
    )
    writer.close()
    writer.close()
    with open_root(tmp_path / "labels.root") as file:
        assert all(file[name].classname == "TTree" for name in SCHEMAS)
        raw = file["bookkeeping/events"].arrays(library="ak")
        assert ak.to_list(raw.generator_weights__present) == [False, True, True]
        assert ak.to_list(raw.generator_weights) == [[], [], [1.0, -0.125]]
        assert ak.to_numpy(raw.event).dtype == np.dtype("uint64")
        assert ak.to_numpy(raw.original_weight).dtype == np.dtype("float64")
        assert file["truth/nodes"].num_entries == 0
        assert "constituent_indices" in file["truth/nodes"].keys()
        particles = file["truth/particles"].arrays(["px"], library="np")
        assert particles["px"].dtype == np.dtype("float32")
    decoded = read_rows(tmp_path, "bookkeeping/events")
    assert [r["generator_weights"] for r in decoded] == [None, [], [1.0, -0.125]]
    assert [r["lhe_weights_json"] for r in decoded] == [None, "", None]
    assert decoded[0]["original_weight"] == records[0]["original_weight"]
    assert decoded[0]["event"] == 2**64 - 1
    chunks = list(iter_table(tmp_path, "bookkeeping/events", ["event_id"], batch_size=1))
    assert [ak.to_list(b.event_id) for b in chunks] == [["a"], ["b"], ["c"]]
    assert all(b.fields == ["event_id"] for b in chunks)


def test_root_corruption_rejected_by_validation_and_resume(campaign, tmp_path):
    out = tmp_path / "corruption"
    produce_source(campaign, out, 0, Config())
    file = out / "source-00000/labels.root"
    with file.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="Changed sidecar"):
        validate_run(out)
    with pytest.raises(ValueError, match="checksum mismatch"):
        produce_source(campaign, out, 0, Config())
    assert not (out / "READY.json").exists()


def test_root_zero_selected_jets_retains_event_coverage(campaign, tmp_path):
    out = tmp_path / "empty-jets"
    result = produce_source(campaign, out, 0, Config(jet_min_pt=1e9))
    assert result["events"] == 2
    assert result["tables"]["truth/jets"]["rows"] == 0
    assert result["tables"]["inputs/CMS/constituents"]["rows"] == 0
    assert validate_run(out)["unique_events"] == 2
    assert list(detector_batches(out, "CMS")) == []
    with open_root(out / "source-00000/labels.root") as file:
        assert file["inputs/ATLAS/events"].num_entries == 2
        assert file["truth/jets"].num_entries == 0
    assert not list(out.rglob("*.parquet"))


def test_root_reader_and_producer_reject_legacy_output_contract(campaign, tmp_path):
    out = tmp_path / "legacy-schema"
    produce_source(campaign, out, 0, Config())
    validate_run(out)
    metadata = json.loads((out / "metadata.json").read_text())
    metadata["contract"]["schema"] = "qcd-labels-v1"
    atomic_json(out / "metadata.json", metadata)
    ready = json.loads((out / "READY.json").read_text())
    ready["schema"] = "qcd-labels-v1"
    atomic_json(out / "READY.json", ready)
    with pytest.raises(ValueError, match="Unsupported label output schema"):
        validate_run(out)
    with pytest.raises(ValueError, match="Unsupported label output schema"):
        list(scan_table(out, "bookkeeping/events"))
    with pytest.raises(ValueError, match="differs"):
        produce_source(campaign, out, 0, Config())
