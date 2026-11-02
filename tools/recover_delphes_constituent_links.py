"""Extract native constituent contributor links after exact compact-data parity.

This reads an existing native ROOT; it never runs Delphes or edits raw data.
Use the pinned ROOT/Delphes environment plus the label dependency overlay.
Output is an additive, uproot-readable TTree sidecar, not replacement detector data.
"""

import argparse
import collections
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import awkward as ak
import h5py
import numpy as np
import ROOT
import uproot
from compact_delphes import convert

from heptokens.labels.source import hepmc_events, read_json, sha256


def p4_rows(particles):
    return np.asarray(
        [[p.Px, p.Py, p.Pz, p.E, p.X, p.Y, p.Z] for p in particles], dtype=np.float32
    ).reshape(-1, 7)


def compare_compact(expected, actual):
    """Every dataset, every row, exact stored dtype, no comparison tolerance."""
    with h5py.File(expected, "r") as a, h5py.File(actual, "r") as b:
        keys = []
        a.visititems(lambda n, o: keys.append(n) if isinstance(o, h5py.Dataset) else None)
        other = []
        b.visititems(lambda n, o: other.append(n) if isinstance(o, h5py.Dataset) else None)
        if keys != other:
            raise ValueError("Different compact dataset inventory")
        for key in keys:
            if a[key].dtype != b[key].dtype or a[key].shape != b[key].shape:
                raise ValueError("Different compact shape/dtype: " + key)
            np.testing.assert_array_equal(a[key][:], b[key][:], err_msg=key)
            if a[key][:].tobytes() != b[key][:].tobytes():
                raise ValueError("Compact stored bit pattern differs: " + key)
    return keys


def jagged(values):
    offsets = np.concatenate(([0], np.cumsum([len(x) for x in values], dtype=np.int64)))
    content = np.asarray([i for x in values for i in x], dtype=np.int64)
    return ak.Array(
        ak.contents.ListOffsetArray(ak.index.Index64(offsets), ak.contents.NumpyArray(content))
    )


def check_native_receipt(validation, receipt, detector, native_sha256):
    if validation.get("status") not in ["passed", "simulated"]:
        raise ValueError("Native simulation did not complete")
    if (
        validation["input_sha256"] != receipt["input_sha256"]
        or validation["seed"] != receipt["seed"]
        or validation["events"] != receipt["events"]
    ):
        raise ValueError("Native input, seed or event count differs")
    record = validation["detectors"][detector]
    if record["card_sha256"] != receipt["sha256"]["delphes_card_" + detector + ".tcl"]:
        raise ValueError("Native response card differs")
    if record["output_sha256"] != native_sha256:
        raise ValueError("Native ROOT hash differs from simulation receipt")


def recover(native, compact, chunk, detector, delphes, output):
    start = time.monotonic()
    native, compact, chunk, delphes, output = map(Path, [native, compact, chunk, delphes, output])
    if output.resolve().is_relative_to(chunk.resolve()) or output.resolve().is_relative_to(
        native.parent.resolve()
    ):
        raise ValueError("Sidecars must be outside original detector and chunk directories")
    receipt = read_json(chunk / "receipt.json")
    reference = read_json(chunk / "truth-reference.json.gz")
    native_sha256 = sha256(native)
    native_receipt = native.parent / "validation.json"
    check_native_receipt(read_json(native_receipt), receipt, detector, native_sha256)
    for path, name in [
        (compact, detector + ".h5"),
        (chunk / "truth.hepmc.gz", "truth.hepmc.gz"),
        (chunk / "truth-reference.json.gz", "truth-reference.json.gz"),
    ]:
        if sha256(path) != receipt["sha256"][name]:
            raise ValueError("Fresh archived payload hash differs: " + name)
    digest = hashlib.sha256()
    with gzip.open(chunk / "truth.hepmc.gz", "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    if digest.hexdigest() != receipt["input_sha256"]:
        raise ValueError("Decoded HepMC digest differs from detector input")
    output.mkdir(parents=True, exist_ok=False)
    reference["source_key"] = receipt["source"]["uri"]
    recompacted = output / "recompacted.h5"
    convert(native, recompacted, delphes, reference)
    checked = compare_compact(compact, recompacted)
    ROOT.gInterpreter.AddIncludePath(str(delphes.resolve()))
    ROOT.gInterpreter.AddIncludePath(str((delphes / "external").resolve()))
    ROOT.gSystem.Load(str((delphes / "libDelphes.so").resolve()))
    f = ROOT.TFile.Open(str(native.resolve()))
    tree = f.Get("Delphes")
    if not tree or tree.GetEntries() != receipt["events"]:
        raise ValueError("Native event count differs")
    rows, event_rows = [], []
    counters = collections.Counter()
    h5row = {"Jet": 0, "FatJet": 0}
    stable = iter(hepmc_events(chunk / "truth.hepmc.gz"))
    task = receipt["source"]["task"]
    chunk_index = int(chunk.name.split("-")[-1])
    try:
        for entry, ref in enumerate(reference["events"]):
            tree.GetEntry(entry)
            event = next(stable)
            if (
                int(tree.Event.At(0).Number) != ref["export_event"]
                or event["export_event"] != ref["export_event"]
            ):
                raise ValueError("Native event number differs")
            particles = list(tree.Particle)
            if len(particles) != len(event["particles"]):
                raise ValueError("Native/stable particle count differs")
            if any(p.Status != 1 or p.IsPU != 0 or p.T != 0 for p in particles):
                raise ValueError("Native truth is not the frozen stable hard-scatter replay")
            np.testing.assert_array_equal(
                [p.PID for p in particles], [p["pdg_id"] for p in event["particles"]]
            )
            wanted = np.asarray(
                [[*p["p4"], *p["position"][:3]] for p in event["particles"]], dtype=np.float32
            ).reshape(-1, 7)
            np.testing.assert_array_equal(p4_rows(particles), wanted)
            by_uid = {int(p.GetUniqueID()): i for i, p in enumerate(particles)}
            if len(by_uid) != len(particles):
                raise ValueError("Duplicate native truth UID")
            barcodes = [p["barcode"] for p in event["particles"]]
            event_rows.append(
                dict(
                    event_index=entry,
                    run=ref["run"],
                    lumi=ref["lumi"],
                    event=ref["event"],
                    export_event=ref["export_event"],
                    weight=ref["weight"],
                    truth_count=len(particles),
                )
            )
            counters["stable_particles_checked"] += len(particles)
            for collection in ["Jet", "FatJet"]:
                if not tree.GetBranch(collection):
                    continue
                for ji, jet in enumerate(getattr(tree, collection)):
                    for ci, obj in enumerate(jet.Constituents):
                        kind = str(obj.ClassName())
                        if kind == "GenParticle":
                            contributors = [obj]
                        elif hasattr(obj, "Particle"):
                            contributors = [obj.Particle.GetObject()]
                        elif hasattr(obj, "Particles"):
                            contributors = list(obj.Particles)
                        else:
                            raise ValueError("No native contributor relation for " + kind)
                        indices = []
                        for particle in contributors:
                            if not particle or particle.ClassName() != "GenParticle":
                                raise ValueError("Unresolved/non-generator native reference")
                            uid = int(particle.GetUniqueID())
                            if uid not in by_uid:
                                raise ValueError("Reference is outside this event truth branch")
                            index = by_uid[uid]
                            if int(particle.PID) != int(particles[index].PID):
                                raise ValueError("Resolved contributor identity differs")
                            indices.append(index)
                        if len(indices) != len(set(indices)):
                            raise ValueError("Repeated native contributor within one constituent")
                        counters["constituents"] += 1
                        counters[kind + "_constituents"] += 1
                        counters[kind + "_empty_contributors"] += not indices
                        counters[kind + "_multiple_contributors"] += len(indices) > 1
                        counters["contributor_edges"] += len(indices)
                        rows.append(
                            dict(
                                event_index=entry,
                                collection=collection,
                                jet_index=ji,
                                constituent_index=ci,
                                compact_constituent_row=h5row[collection],
                                object_class=kind,
                                truth_indices=indices,
                                truth_barcodes=[barcodes[i] for i in indices],
                            )
                        )
                        h5row[collection] += 1
                counters[collection + "_jets"] += len(getattr(tree, collection))
            counters["events"] += 1
        if next(stable, None) is not None:
            raise ValueError("Unconsumed stable events")
    finally:
        f.Close()
    with h5py.File(compact, "r") as h5:
        if any(h5row[c] != len(h5[c]["constituent_p4"]) for c in h5row):
            raise ValueError("Sidecar does not cover every compact constituent row")
    schema = dict(
        event_index="int64",
        collection="string",
        jet_index="int64",
        constituent_index="int64",
        compact_constituent_row="int64",
        object_class="string",
        truth_indices="var * int64",
        truth_barcodes="var * int64",
    )
    event_schema = dict(
        event_index="int64",
        run="uint64",
        lumi="uint64",
        event="uint64",
        export_event="uint64",
        weight="float64",
        truth_count="int64",
    )
    metadata = dict(
        schema="delphes-native-constituent-links-v1",
        source_task=task,
        chunk_index=chunk_index,
        detector=detector,
        native_root_sha256=native_sha256,
        compact_sha256=sha256(compact),
        native_validation_sha256=sha256(native_receipt),
        receipt_sha256=sha256(chunk / "receipt.json"),
        input_hepmc_sha256=digest.hexdigest(),
        libDelphes_sha256=sha256(delphes / "libDelphes.so"),
        tool_sha256=sha256(__file__),
        relation=(
            "native ROOT contributor references, not deltaR matches "
            "or per-contributor energy fractions"
        ),
        truth_index=(
            "zero-based index in native Particle array; truth_barcodes "
            "index the immutable archived HepMC"
        ),
        compact_parity=(
            "all dataset bit patterns exact, including dtype/shape/memberships/float32 P4; "
            "HDF5 file bytes need not agree"
        ),
        normalization="original raw weights; no event selections",
        unchanged_originals=True,
        runtime=dict(
            python=sys.version,
            executable=sys.executable,
            ROOT=ROOT.gROOT.GetVersion(),
            numpy=np.__version__,
            uproot=uproot.__version__,
            awkward=ak.__version__,
            h5py=h5py.__version__,
        ),
    )
    target = output / "constituent-links.root"
    with uproot.recreate(target, compression=uproot.ZSTD(4)) as root:
        root["metadata"] = json.dumps(metadata, sort_keys=True)
        t = root.mktree("links", schema)
        for begin in range(0, len(rows), 4096):
            batch = rows[begin : begin + 4096]
            t.extend(
                {
                    k: jagged([r[k] for r in batch])
                    if dtype.startswith("var *")
                    else ak.Array([r[k] for r in batch])
                    if dtype == "string"
                    else np.asarray([r[k] for r in batch], dtype=dtype)
                    for k, dtype in schema.items()
                }
            )
        t = root.mktree("events", event_schema)
        t.extend(
            {
                k: np.asarray([r[k] for r in event_rows], dtype=dtype)
                for k, dtype in event_schema.items()
            }
        )
    with uproot.open(target, handler=uproot.source.file.MemmapSource) as root:
        if root["links"].classname != "TTree" or root["links"].num_entries != len(rows):
            raise ValueError("Sidecar ROOT readback failed")
        if root["events"].num_entries != len(event_rows):
            raise ValueError("Sidecar event readback failed")
    result = dict(
        status="passed",
        source_task=task,
        chunk_index=chunk_index,
        detector=detector,
        counters=dict(counters),
        checked_compact_datasets=checked,
        metadata=metadata,
        sidecar=dict(file=str(target), bytes=target.stat().st_size, sha256=sha256(target)),
        native_root=str(native),
        native_root_bytes=native.stat().st_size,
        wall_seconds=time.monotonic() - start,
    )
    (output / "recovery.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["native", "compact", "chunk", "delphes", "output"]:
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--detector", choices=["CMS", "ATLAS"], required=True)
    args = p.parse_args()
    print(
        json.dumps(
            recover(args.native, args.compact, args.chunk, args.detector, args.delphes, args.output)
        )
    )
