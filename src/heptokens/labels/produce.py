"""Streaming, restartable per-source production of immutable label partitions."""

import importlib.metadata
import json
import os
import resource
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from . import SCHEMA_VERSION
from .physics import (
    ca_tree,
    cartesian_fields,
    charge,
    common_jets,
    event_kinematics,
    match_jets,
    p4_fields,
    plane_angles,
    stable_id,
)
from .source import atomic_json, iter_events, read_json, sample_records, sha256, source_root
from .tables import SCHEMAS, TableWriters

UNAVAILABLE = {
    "heavy_hadron_flavor": (
        "No decayed heavy hadrons or b-to-c ancestry retained. Need pruned/full generator "
        "particles, mother links, and terminal weakly decaying b/c hadrons for ghost "
        "association. Do not interpret stable daughters as heavy flavor."
    ),
    "parton_flavor": (
        "No outgoing hard partons or shower genealogy retained. Need generator status "
        "flags, mother/daughter links, and a versioned association prescription. Unknown is"
        " not light flavor."
    ),
    "shower_splitting_category": (
        "C/A geometry is not a generator shower. Need retained generator shower history and"
        " an explicit node association to assign q->qg, g->gg, or g->qqbar."
    ),
    "reconstructed_object_truth": (
        "Compaction discarded Track.Particle and Tower.Particles TRefs and Particle "
        "branches. Need original Delphes ROOT or reference-preserving re-compaction; retain"
        " list-valued tower associations."
    ),
    "generator_ht_and_pthat": (
        "Exporter did not retain GenEventInfoProduct binningValues, qScale, or LHE partons."
        " Synthetic HepMC E-header zeros are not measurements. Sample boundaries are pthat,"
        " not HT."
    ),
    "incoming_and_hard_process": (
        "Synthetic stable-only HepMC headers do not retain incoming particles or "
        "signal_process_id. Need original GenEventInfoProduct/LHE/GEN records."
    ),
    "original_pileup_and_time": (
        "Original pileup was excluded; production time was set to zero by exporter. Need "
        "the original event collections to obtain either."
    ),
    "spin_state": (
        "No helicities, density matrices, or paired spin-on/off generator runs exist in "
        "retained inputs. Only geometric angular observables are produced."
    ),
}

PRODUCTION_MODULES = {"__init__.py", "physics.py", "source.py", "tables.py", "produce.py"}
PRODUCTION_PACKAGES = {
    "awkward",
    "numpy",
    "vector",
    "fastjet",
    "particle",
    "pyarrow",
    "h5py",
    "scipy",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def code_provenance():
    repo = Path(__file__).resolve().parents[3]

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    try:
        revision, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision, dirty = None, None
    files = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    versions = {}
    for name in [
        "awkward",
        "numpy",
        "vector",
        "fastjet",
        "particle",
        "pyarrow",
        "h5py",
        "scipy",
        "hist",
        "mplhep",
        "matplotlib",
    ]:
        versions[name] = importlib.metadata.version(name)
    env = repo / "environments/cms-gpu"
    return dict(
        git_revision=revision,
        git_dirty=dirty,
        module_sha256=files,
        software_id=stable_id(
            *[f"{k}:{v}" for k, v in sorted(files.items()) if k in PRODUCTION_MODULES]
        ),
        package_versions=versions,
        python=sys.version,
        executable=sys.executable,
        environment_manifest_sha256=sha256(env / "pixi.toml")
        if (env / "pixi.toml").exists()
        else None,
        environment_lock_sha256=sha256(env / "pixi.lock") if (env / "pixi.lock").exists() else None,
        command_argv=sys.argv,
        hostname=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
    )


def event_tables(ev, sample, cfg, normalization=None):
    tables = {name: [] for name in SCHEMAS}
    eid, ref = ev["event_id"], ev["reference"]
    raw_weight = float(ref["weight"])
    norm_valid = normalization is not None and normalization.get("valid", False)
    sumw = normalization["sum_weights"] if norm_valid else None
    norm_valid = bool(norm_valid and sumw and sample["cross_section_pb"] is not None)
    tables["bookkeeping/events"].append(
        dict(
            event_id=eid,
            generator_event_id=ev["generator_event_id"],
            truth_fingerprint=ev["truth_fingerprint"],
            sample_id=ev["sample_id"],
            source_id=ev["source_id"],
            source_uri=ev["source_uri"],
            run=ref["run"],
            lumi=ref["lumi"],
            event=ref["event"],
            export_event=ref["export_event"],
            chunk_id=ev["chunk_id"],
            delphes_seed=ev["evidence"]["seed"],
            split=ev["split"],
            original_weight=raw_weight,
            generator_weights=ref.get("generator_weights"),
            lhe_weights_json=None
            if ref.get("lhe_weights") is None
            else json.dumps(ref["lhe_weights"]),
            cross_section_pb=sample["cross_section_pb"],
            sample_sum_weights=sumw,
            physical_weight_pb=raw_weight * sample["cross_section_pb"] / sumw
            if norm_valid
            else None,
            normalization_valid=norm_valid,
            generator_ht=None,
            generator_ht_valid=False,
            generator_pthat=None,
            generator_pthat_valid=False,
            selection_low_GeV=sample["selection_low_GeV"],
            selection_high_GeV=sample["selection_high_GeV"],
            selection_variable=sample["selection_variable"],
            synchronization_valid=True,
        )
    )
    particles = {p["barcode"]: p for p in ev["truth"]}
    truth_jets = []
    selected_particle_ids = set()
    for j in common_jets(ev["truth"], cfg):
        ids = j["constituent_indices"]
        charges = [charge(particles[i]["pdg_id"]) for i in ids]
        jid = stable_id(eid, "truth-antikt", cfg.jet_definition_digest, *ids)
        row = dict(
            event_id=eid,
            jet_id=jid,
            constituent_indices=ids,
            multiplicity=len(ids),
            charged_multiplicity=sum(q != 0 for q in charges)
            if all(q is not None for q in charges)
            else None,
            charged_multiplicity_valid=all(q is not None for q in charges),
            heavy_hadron_flavor="unknown",
            heavy_hadron_pdg_ids=None,
            heavy_hadron_flavor_valid=False,
            parton_flavor="unknown",
            parton_pdg_ids=None,
            parton_flavor_valid=False,
            **p4_fields(j["p4"]),
        )
        truth_jets.append(row)
        selected_particle_ids.update(ids)
        if cfg.include_trees:
            nodes = ca_tree([particles[i]["p4"] for i in ids], ids, eid, jid)
            tables["truth/nodes"].extend(nodes)
            tables["truth/angles"].extend(plane_angles(nodes, cfg))
    for p in particles.values():
        if not cfg.save_all_truth_particles and p["barcode"] not in selected_particle_ids:
            continue
        q = charge(p["pdg_id"])
        tables["truth/particles"].append(
            dict(
                event_id=eid,
                particle_index=p["barcode"],
                pdg_id=p["pdg_id"],
                charge=q,
                charge_valid=q is not None,
                status=1,
                is_pileup=False,
                x_mm=p["position"][0],
                y_mm=p["position"][1],
                z_mm=p["position"][2],
                generated_mass=p["generated_mass"],
                vertex_ct_mm=p["position"][3],
                vertex_time_valid=False,
                mother_indices=None,
                ancestry_valid=False,
                **cartesian_fields(p["p4"]),
            )
        )
    tables["truth/jets"] = truth_jets
    tables["truth/events"].append(
        dict(
            event_id=eid,
            incoming_pdg_ids=None,
            incoming_flavors_valid=False,
            hard_process_id=None,
            hard_process_valid=False,
            **event_kinematics(truth_jets),
        )
    )
    for detector, view in ev["reco"].items():
        reco_jets = []
        for j in view["jets"]:
            fields = p4_fields(j["p4"])
            # Preserve all original objects; only selected jets participate in matching.
            jid = stable_id(eid, detector, "Jet", j["original_index"])
            ids = list(range(len(j["constituent_p4"])))
            row = dict(
                event_id=eid,
                jet_id=jid,
                detector=detector,
                original_index=j["original_index"],
                original_split=view["original_split"],
                constituent_indices=ids,
                multiplicity=len(ids),
                **fields,
            )
            selected = (
                fields["eta"] is not None
                and fields["pt"] >= cfg.jet_min_pt
                and abs(fields["eta"]) < cfg.jet_abs_eta
            )
            if not selected and not cfg.save_unselected_reco:
                continue
            tables[f"inputs/{detector}/jets"].append(row)
            if selected:
                reco_jets.append(row)
            for i, (p4, q, kind) in enumerate(zip(j["constituent_p4"], j["charge"], j["kind"])):
                tables[f"inputs/{detector}/constituents"].append(
                    dict(
                        event_id=eid,
                        jet_id=jid,
                        constituent_index=i,
                        stored_index=j["constituent_start"] + i,
                        detector=detector,
                        object_type=int(kind),
                        reco_charge=int(q),
                        **cartesian_fields(p4),
                    )
                )
                tables[f"targets/{detector}/object_truth"].append(
                    dict(
                        event_id=eid,
                        jet_id=jid,
                        constituent_index=i,
                        truth_particle_indices=None,
                        association_valid=False,
                        species_valid=False,
                        ancestry_valid=False,
                        unavailable_reason="truth TRefs discarded during compaction",
                    )
                )
            # Both detector trees use their own resolved constituents. They are
            # never assigned truth tree node IDs or rank-paired across detectors.
            if selected and cfg.include_trees:
                nodes = ca_tree(j["constituent_p4"], ids, eid, jid)
                tables[f"inputs/{detector}/nodes"].extend(nodes)
                tables[f"inputs/{detector}/angles"].extend(plane_angles(nodes, cfg))
        matches = match_jets(reco_jets, truth_jets, cfg)
        tables[f"targets/{detector}/jet_matches"] = matches
        coverage = {m["truth_jet_id"]: m["reco_jet_id"] for m in matches if m["matched"]}
        tables[f"targets/{detector}/truth_jet_coverage"] = [
            dict(
                event_id=eid,
                truth_jet_id=j["jet_id"],
                matched=j["jet_id"] in coverage,
                reco_jet_id=coverage.get(j["jet_id"]),
            )
            for j in truth_jets
        ]
        tables[f"inputs/{detector}/events"].append(
            dict(event_id=eid, detector=detector, **event_kinematics(reco_jets))
        )
    return tables


def produce_source(
    campaign, output, task, cfg, max_chunks=None, events_per_chunk=None, normalization=None
):
    start = time.monotonic()
    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    cpu_start = usage_start.ru_utime + usage_start.ru_stime
    campaign, output = Path(campaign), Path(output)
    if (
        events_per_chunk is not None
        and events_per_chunk <= 0
        or max_chunks is not None
        and max_chunks <= 0
    ):
        raise ValueError("Limits must be positive")
    if campaign.resolve() == output.resolve() or campaign.resolve() in output.resolve().parents:
        raise ValueError("Sidecars must be outside the immutable raw campaign")
    source = source_root(campaign, task)
    manifest_path = campaign / "provenance/cms-qcd-source-manifest.json"
    manifest_hash = sha256(manifest_path)
    manifest = read_json(manifest_path)
    entry = manifest["files"][task]
    if entry["task"] != task:
        raise ValueError("Manifest source-task indexing differs")
    sample = sample_records(manifest)[str(entry["record"])]
    complete = read_json(source / "COMPLETE.json")
    if complete.get("manifest_sha256", manifest_hash) != manifest_hash:
        raise ValueError("Source completion refers to another manifest")
    chunks = [source / x["chunk"] for x in complete["chunks"]]
    if len(set(chunks)) != len(chunks):
        raise ValueError("Duplicate chunk in source completion")
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    provenance = code_provenance()
    contract = dict(
        schema=SCHEMA_VERSION,
        config=asdict(cfg),
        config_sha256=cfg.digest,
        source_manifest_sha256=manifest_hash,
        software_id=provenance["software_id"],
        max_chunks=max_chunks,
        events_per_chunk=events_per_chunk,
        normalization=normalization,
        package_versions={
            k: v for k, v in provenance["package_versions"].items() if k in PRODUCTION_PACKAGES
        },
    )
    output.mkdir(parents=True, exist_ok=True)
    # Metadata is initialized by the CLI before workers start. Atomic exclusive
    # creation here also makes standalone worker calls safe.
    import fcntl

    with (output / ".metadata.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (output / "metadata.json").exists():
            if read_json(output / "metadata.json")["contract"] != json.loads(json.dumps(contract)):
                raise ValueError(
                    "Output configuration/software differs; choose a new versioned output"
                )
        else:
            atomic_json(
                output / "metadata.json",
                dict(
                    created_utc=utc_now(),
                    contract=contract,
                    provenance=provenance,
                    samples=sample_records(manifest),
                    source_files=len(manifest["files"]),
                    unavailable=UNAVAILABLE,
                    question=(
                        "Which retained truth and detector-comparison targets can be "
                        "produced without inventing generator information?"
                    ),
                    success_criteria=(
                        "Truth-content synchronization evidence is hash-bound; IDs, splits "
                        "and one-to-one joins validate; unavailable labels have explicit "
                        "masks; raw inputs remain unchanged."
                    ),
                    definition_document="docs/qcd-label-production.md",
                    units="GeV, mm, radians, pb",
                    scope="validation prefix"
                    if max_chunks or events_per_chunk
                    else "all committed source events",
                    particle_precision=(
                        "Cartesian constituent/truth-particle P4 float32; full precision "
                        "compressed originals retained"
                    ),
                    composite_ids=(
                        "truth particle=(event_id,particle_index); reco "
                        "constituent=(jet_id,constituent_index)"
                    ),
                    deferred_targets=[]
                    if cfg.include_trees
                    else ["C/A binary trees", "splitting geometry", "splitting-plane angles"],
                ),
            )
    target = output / f"source-{task:05d}"
    # One lock per partition prevents two batch retries from racing on publication.
    with (output / f".source-{task:05d}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (target / "COMMIT.json").exists():
            saved = read_json(target / "COMMIT.json")
            for name, info in saved["tables"].items():
                if sha256(target / name) != info["sha256"]:
                    raise ValueError("Existing sidecar partition checksum mismatch")
            return saved
        if target.exists():
            raise ValueError("Uncommitted partition exists; preserve it and choose a new output")
        temporary = output / f".source-{task:05d}.partial-{uuid.uuid4().hex}"
        temporary.mkdir()
        writers = TableWriters(
            temporary,
            dict(
                schema=SCHEMA_VERSION,
                config_sha256=cfg.digest,
                software_id=provenance["software_id"],
            ),
        )
        count, evidence = 0, []
        buffered = {name: [] for name in SCHEMAS}
        try:
            for chunk in chunks:
                first = True
                for ev in iter_events(chunk, cfg, events_per_chunk, entry):
                    if first:
                        evidence.append(dict(chunk=chunk.name, **ev["evidence"]))
                        first = False
                    norm = (
                        None if normalization is None else normalization["samples"][ev["sample_id"]]
                    )
                    tables = event_tables(ev, sample, cfg, norm)
                    for name, rows in tables.items():
                        buffered[name].extend(rows)
                    count += 1
                    if count % 16 == 0:
                        writers.write(buffered)
                        buffered = {name: [] for name in SCHEMAS}
                writers.write(buffered)
                buffered = {name: [] for name in SCHEMAS}
            writers.close()
            table_info = {}
            for name in SCHEMAS:
                path = temporary / (name + ".parquet")
                table_info[name + ".parquet"] = dict(
                    rows=writers.counts[name], bytes=path.stat().st_size, sha256=sha256(path)
                )
            result = dict(
                status="complete",
                schema=SCHEMA_VERSION,
                source_id=task,
                sample_id=sample["sample_id"],
                events=count,
                chunks=len(chunks),
                complete_source=max_chunks is None and events_per_chunk is None,
                source_complete_sha256=sha256(source / "COMPLETE.json"),
                source_manifest_sha256=manifest_hash,
                config_sha256=cfg.digest,
                software_id=provenance["software_id"],
                started_elapsed_seconds=time.monotonic() - start,
                cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime
                + resource.getrusage(resource.RUSAGE_SELF).ru_stime
                - cpu_start,
                max_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                created_utc=utc_now(),
                tables=table_info,
                synchronization=evidence,
            )
            atomic_json(temporary / "COMMIT.json", result)
            temporary.rename(target)
            return result
        except BaseException as error:
            try:
                writers.close()
            except Exception:
                pass
            atomic_json(
                temporary / "FAILED.json",
                dict(error=repr(error), events_written=count, created_utc=utc_now()),
            )
            raise
