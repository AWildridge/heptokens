"""Hash-bound MiniAOD ancestry joins and terminal-heavy-hadron ghost labels.

The complete native generator archive remains authoritative. These tables expose
the retained pruned graph and links from the exact original stable replay; no
generator, detector, missing time or unretained genealogy is synthesized.
"""

import math
from collections import defaultdict, deque
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

METHOD = {
    "version": "miniaod-retained-heavy-ghost-v1",
    "scope": "retained packed/pruned MiniAOD graph, not the unpruned generator history",
    "stable_join": (
        "event ID and weight; per-vertex packed order, exact PDG/mass/xyz, "
        "ROOT signed-mass polar convention, P4 rtol2e-12/atol1e-10GeV"
    ),
    "ghost_scale": 1e-20,
    "ghost_scale_audit": [1e-18, 1e-20, 1e-22],
    "heavy_selector": (
        "CMSSW_10_6_25 CandMCTagUtils heavy-hadron predicate; no daughter of the same heavy flavour"
    ),
    "jet_label": (
        "b precedence; c only for associated charm without a retained b ancestor; "
        "none means no associated retained b or prompt-c hadron; charm from b is separate"
    ),
    "mother_target": "truth/generator_particles keyed by event_id and zero-based particle_index",
    "pthat": (
        "GenEventInfoProduct binningValues[0], set to Pythia8 info.pTHat() "
        "in original GEN CMSSW_10_6_19; qScale is separate"
    ),
}


def has_flavour(pid, flavour):
    pid = abs(int(pid))
    return (pid // 100) % 10 == flavour or (pid // 1000) % 10 == flavour


def relations(record, prefix, role, n):
    offsets = record[prefix + role + "Offset"]
    indices = record[prefix + role + "Index"]
    states = record[prefix + role + "State"]
    if (
        len(offsets) != n + 1
        or offsets[0] != 0
        or offsets[-1] != len(indices)
        or len(states) != len(indices)
    ):
        raise ValueError("Generator graph offsets/lengths differ")
    result = []
    target = len(record["Pruned_pdgId"])
    for i in range(n):
        if offsets[i] > offsets[i + 1]:
            raise ValueError("Generator graph offsets not monotonic")
        links = []
        for j in range(offsets[i], offsets[i + 1]):
            if states[j] == 0:
                if indices[j] != -1:
                    raise ValueError("Null generator edge has a nonnull index")
            elif states[j] == 2 and 0 <= indices[j] < target:
                links.append(indices[j])
            else:
                raise ValueError("Unavailable or out-of-range generator reference")
        result.append(links)
    return result


def graph_info(record):
    pid = record["Pruned_pdgId"]
    n = len(pid)
    mothers = relations(record, "Pruned_", "mother", n)
    daughters = relations(record, "Pruned_", "daughter", n)
    colours = [0] * n
    from_b = [False] * n

    def visit(i):
        if colours[i] == 1:
            raise ValueError("Cycle in retained generator ancestry")
        if colours[i] == 2:
            return from_b[i]
        colours[i] = 1
        inherited = []
        for parent in mothers[i]:
            inherited.append(visit(parent) or has_flavour(pid[parent], 5))
        from_b[i] = any(inherited)
        colours[i] = 2
        return from_b[i]

    for i in range(n):
        visit(i)
    bs = [
        i
        for i in range(n)
        if has_flavour(pid[i], 5) and not any(has_flavour(pid[d], 5) for d in daughters[i])
    ]
    cs = [
        i
        for i in range(n)
        if has_flavour(pid[i], 4) and not any(has_flavour(pid[d], 4) for d in daughters[i])
    ]
    return dict(mothers=mothers, daughters=daughters, from_b=from_b, b=bs, c=cs)


def packed_p4(pt, eta, phi, mass):
    # ROOT PtEtaPhiM4D::M2 is negative for a negative stored mass. Squaring
    # blindly changes the energy of quantized, nominally massless particles.
    momentum = pt * math.cosh(eta)
    return [
        pt * math.cos(phi),
        pt * math.sin(phi),
        pt * math.sinh(eta),
        math.sqrt(max(0.0, momentum * momentum + mass * abs(mass))),
    ]


def packed_join(event, record):
    n = len(record["Packed_pdgId"])
    if len(event["truth"]) != n or any(s != 1 for s in record["Packed_status"]):
        raise ValueError("Packed/stable particle count or status differs")
    groups = defaultdict(deque)
    for i in range(n):
        groups[tuple(record["Packed_" + v + "_mm"][i] for v in ["x", "y", "z"])].append(i)
    mapping = {}
    maximum = 0.0
    for particle in event["truth"]:
        if particle["position"][3] != 0:
            raise ValueError("Unexpected replay particle time")
        vertex = tuple(particle["position"][:3])
        if not groups[vertex]:
            raise ValueError("Replay vertex missing from original packed record")
        index = groups[vertex].popleft()
        if (
            particle["pdg_id"] != record["Packed_pdgId"][index]
            or particle["generated_mass"] != record["Packed_mass"][index]
        ):
            raise ValueError("Packed/replay PDG, mass or per-vertex order differs")
        pt, eta, phi, mass = [record["Packed_" + k][index] for k in ["pt", "eta", "phi", "mass"]]
        p4 = packed_p4(pt, eta, phi, mass)
        np.testing.assert_allclose(p4, particle["p4"], rtol=2e-12, atol=1e-10)
        maximum = max(maximum, float(np.max(np.abs(np.asarray(p4) - particle["p4"]))))
        mapping[particle["barcode"]] = index
    if any(groups.values()):
        raise ValueError("Unjoined packed particles")
    return mapping, maximum


class GeneratorReader:
    def __init__(self, path, receipt, source):
        from .source import sha256

        proof = receipt["remote"]
        if (
            receipt["source"] != source
            or receipt["status"] != "complete"
            or proof["status"] != "two-stable-full-readbacks-passed"
        ):
            raise ValueError("Uncommitted or mismatched generator source")
        for role in ["first", "second"]:
            if (
                proof[role]["sha256"] != receipt["sha256"]
                or proof[role]["bytes"] != receipt["bytes"]
            ):
                raise ValueError("Generator readback proof differs")
        if Path(path).stat().st_size != receipt["bytes"] or sha256(path) != receipt["sha256"]:
            raise ValueError("Fresh generator payload hash differs")
        self.file = uproot.open(
            path, handler=uproot.source.file.MemmapSource, object_cache=None, array_cache=None
        )
        self.tree = self.file["gen/events"]
        self.receipt = receipt
        keys = self.tree.arrays(["run", "lumi", "event"], library="np")
        self.lookup = {
            tuple(map(int, k)): i
            for i, k in enumerate(zip(keys["run"], keys["lumi"], keys["event"]))
        }
        if len(self.lookup) != self.tree.num_entries:
            raise ValueError("Duplicate generator event identity")
        self.cache = None
        self.start = -1
        self.stats = dict(events=0, packed_particles=0, pruned_particles=0, max_p4_roundoff_GeV=0.0)

    def close(self):
        self.file.close()

    def join(self, event):
        ref = event["reference"]
        key = (ref["run"], ref["lumi"], ref["event"])
        if key not in self.lookup:
            raise ValueError("Replay event absent from generator backfill")
        entry = self.lookup[key]
        start = (entry // 64) * 64
        if start != self.start:
            self.cache = self.tree.arrays(entry_start=start, entry_stop=start + 64, library="ak")
            self.start = start
        record = ak.to_list(self.cache[entry - start])
        if float(record["weight"]) != float(ref["weight"]):
            raise ValueError("Generator/replay nominal weight differs")
        if (
            ref.get("generator_weights") is not None
            and record["weights"] != ref["generator_weights"]
        ):
            raise ValueError("Generator/replay variation weights differ")
        mapping, roundoff = packed_join(event, record)
        graph = graph_info(record)
        graph["packed_mothers"] = relations(
            record, "Packed_", "mother", len(record["Packed_pdgId"])
        )
        graph["packed_last"] = relations(
            record, "Packed_", "lastPruned", len(record["Packed_pdgId"])
        )
        self.stats["events"] += 1
        self.stats["packed_particles"] += len(mapping)
        self.stats["pruned_particles"] += len(record["Pruned_pdgId"])
        self.stats["max_p4_roundoff_GeV"] = max(self.stats["max_p4_roundoff_GeV"], roundoff)
        event["generator"] = dict(
            record=record,
            entry=entry,
            graph=graph,
            packed_mapping=mapping,
            source_uri=self.receipt["remote"]["second"]["url"],
            sha256=self.receipt["sha256"],
        )
        return event


def heavy_associations(event, truth_jets, cfg, scale=1e-20):
    import fastjet as fj

    from .physics import p4_array, pseudos

    gen = event["generator"]
    record = gen["record"]
    graph = gen["graph"]
    particles = event["truth"]
    p4 = np.asarray([p["p4"] for p in particles], dtype=float).reshape(-1, 4)
    vectors = p4_array(p4)
    pid = np.asarray([abs(p["pdg_id"]) for p in particles])
    mask = (
        (ak.to_numpy(vectors.pt) > 0)
        & (np.abs(ak.to_numpy(vectors.eta)) < cfg.particle_abs_eta)
        & ~np.isin(pid, cfg.invisible_pids)
    )
    if not cfg.include_muons:
        mask &= pid != 13
    indices = np.asarray([p["barcode"] for p in particles])
    inputs = pseudos(p4[mask], indices[mask])
    selected = sorted(set(graph["b"] + graph["c"]))
    for index in selected:
        values = np.asarray([record["Pruned_" + k][index] for k in ["px", "py", "pz", "energy"]])
        if not np.isfinite(values).all() or math.hypot(values[0], values[1]) == 0:
            raise ValueError("Terminal heavy hadron has no usable ghost direction")
        ghost = fj.PseudoJet(*(values * scale))
        ghost.set_user_index(-index - 1)
        inputs.append(ghost)
    expected = {tuple(j["constituent_indices"]): j for j in truth_jets}
    found = {}
    max_delta = 0.0
    if not inputs:
        if expected:
            raise ValueError("Missing physical clustering inputs")
        return found, max_delta
    sequence = fj.ClusterSequence(inputs, fj.JetDefinition(fj.antikt_algorithm, cfg.radius))
    for jet in sequence.inclusive_jets(cfg.jet_min_pt):
        if abs(jet.eta()) >= cfg.jet_abs_eta:
            continue
        members = tuple(sorted(p.user_index() for p in jet.constituents() if p.user_index() > 0))
        if members not in expected:
            raise ValueError("Ghosts changed physical jet membership")
        before = expected[members]
        delta = np.asarray([jet.px(), jet.py(), jet.pz(), jet.E()]) - np.asarray(
            [before[k] for k in ["px", "py", "pz", "energy"]]
        )
        np.testing.assert_allclose(delta, 0, rtol=0, atol=1e-9)
        max_delta = max(max_delta, float(np.max(np.abs(delta))))
        ghosts = sorted(-p.user_index() - 1 for p in jet.constituents() if p.user_index() < 0)
        bs = [i for i in ghosts if i in graph["b"]]
        cs = [i for i in ghosts if i in graph["c"]]
        from_b = [i for i in cs if graph["from_b"][i]]
        prompt = [i for i in cs if not graph["from_b"][i]]
        found[members] = dict(
            heavy_hadron_flavor="b" if bs else "c" if prompt else "none",
            heavy_hadron_pdg_ids=[record["Pruned_pdgId"][i] for i in ghosts],
            heavy_hadron_flavor_valid=True,
            b_hadron_indices=bs,
            c_hadron_indices=cs,
            c_from_b_indices=from_b,
            prompt_c_hadron_indices=prompt,
        )
    if set(found) != set(expected):
        raise ValueError("Ghost clustering changed selected jet inventory")
    return found, max_delta


def enrich_tables(event, tables, cfg, audit_scales=False):
    from .physics import p4_fields

    gen = event["generator"]
    r = gen["record"]
    g = gen["graph"]
    eid = event["event_id"]
    associations, delta = heavy_associations(event, tables["truth/jets"], cfg)
    if audit_scales:
        for scale in [1e-18, 1e-22]:
            alternate, _ = heavy_associations(event, tables["truth/jets"], cfg, scale)
            if alternate != associations:
                raise ValueError("Heavy labels depend on ghost rescaling")
    for row in tables["truth/jets"]:
        row.update(associations[tuple(row["constituent_indices"])])
    for row in tables["truth/particles"]:
        index = gen["packed_mapping"][row["particle_index"]]
        row.update(
            gen_packed_index=index,
            gen_status_flags=r["Packed_statusFlags"][index],
            mother_indices=g["packed_mothers"][index],
            ancestry_valid=True,
            last_pruned_indices=g["packed_last"][index],
        )
    tables["truth/generator_records"] = [
        dict(
            event_id=eid,
            source_uri=gen["source_uri"],
            source_sha256=gen["sha256"],
            source_entry=gen["entry"],
            tier="MINIAODSIM",
            packed_count=len(r["Packed_pdgId"]),
            pruned_count=len(r["Pruned_pdgId"]),
            join_valid=True,
            qscale=r["qScale"],
            binning_values=r["binningValues"],
            process_id=r["processId"],
            pdf_valid=r["hasPDF"],
            pdf_id1=r["pdfId1"] if r["hasPDF"] else None,
            pdf_id2=r["pdfId2"] if r["hasPDF"] else None,
            alpha_qcd=r["alphaQCD"],
            alpha_qed=r["alphaQED"],
            ghost_p4_max_delta_GeV=delta,
        )
    ]
    tables["truth/generator_particles"] = []
    for i, pid in enumerate(r["Pruned_pdgId"]):
        tables["truth/generator_particles"].append(
            dict(
                event_id=eid,
                particle_index=i,
                pdg_id=pid,
                status=r["Pruned_status"][i],
                three_charge=r["Pruned_threeCharge"][i],
                status_flags=r["Pruned_statusFlags"][i],
                collision_id=r["Pruned_collisionId"][i],
                x_mm=r["Pruned_x_mm"][i],
                y_mm=r["Pruned_y_mm"][i],
                z_mm=r["Pruned_z_mm"][i],
                generated_mass=r["Pruned_mass"][i],
                mother_indices=g["mothers"][i],
                daughter_indices=g["daughters"][i],
                terminal_b=i in g["b"],
                terminal_c=i in g["c"],
                has_b_ancestor=g["from_b"][i],
                **p4_fields([r["Pruned_" + k][i] for k in ["px", "py", "pz", "energy"]]),
            )
        )
    book = tables["bookkeeping/events"][0]
    book.update(
        generator_pthat=r["binningValues"][0] if r["binningValues"] else None,
        generator_pthat_valid=bool(r["binningValues"]),
    )
    truth = tables["truth/events"][0]
    truth.update(
        incoming_pdg_ids=[r["pdfId1"], r["pdfId2"]] if r["hasPDF"] else None,
        incoming_flavors_valid=r["hasPDF"],
        hard_process_id=r["processId"],
        hard_process_valid=True,
        n_b_jets=sum(j["heavy_hadron_flavor"] == "b" for j in tables["truth/jets"]),
        n_c_jets=sum(j["heavy_hadron_flavor"] == "c" for j in tables["truth/jets"]),
        heavy_flavor_valid=True,
    )
    by_id = {j["jet_id"]: j for j in tables["truth/jets"]}
    for detector in ["CMS", "ATLAS"]:
        for row in tables[f"targets/{detector}/jet_matches"]:
            jet = by_id.get(row["truth_jet_id"]) if row["matched"] else None
            row.update(
                heavy_hadron_flavor=jet["heavy_hadron_flavor"] if jet else "unknown",
                heavy_hadron_flavor_valid=jet is not None,
            )
    return tables
