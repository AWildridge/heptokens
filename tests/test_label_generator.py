"""Generator joins, CMS terminal-hadron ancestry and ghost invariance."""

import math

import numpy as np
import pytest

from heptokens.labels.generator import graph_info, heavy_associations, packed_join, packed_p4
from heptokens.labels.physics import Config, common_jets, p4_fields


def record_graph(pids, mothers):
    daughters = [[] for _ in pids]
    for child, parents in enumerate(mothers):
        for parent in parents:
            daughters[parent].append(child)
    r = dict(Pruned_pdgId=pids)
    for name, lists in [("mother", mothers), ("daughter", daughters)]:
        flat = [x for values in lists for x in values]
        r["Pruned_" + name + "Index"] = flat
        r["Pruned_" + name + "State"] = [2] * len(flat)
        r["Pruned_" + name + "Offset"] = np.cumsum([0] + [len(x) for x in lists]).tolist()
    return r


def test_root_signed_mass_roundtrip():
    vector = packed_p4(10, 0, 0, -1)
    assert vector[3] == math.sqrt(99)
    assert vector[3] != math.sqrt(101)


def test_terminal_hadron_selector_and_b_ancestry():
    r = record_graph([10511, 511, 421, 211, 411], [[], [0], [1], [2], []])
    graph = graph_info(r)
    assert graph["b"] == [1] and graph["c"] == [2, 4]
    assert graph["from_b"][2] and not graph["from_b"][4]


def test_cyclic_pruned_graph_rejected():
    with pytest.raises(ValueError, match="Cycle"):
        graph_info(record_graph([511, 421], [[1], [0]]))


def test_unavailable_graph_link_rejected():
    r = record_graph([511, 421], [[], [0]])
    r["Pruned_motherState"] = [1]
    with pytest.raises(ValueError, match="Unavailable"):
        graph_info(r)


def test_join_preserves_per_vertex_order_when_vertex_groups_reordered():
    r = dict(
        Packed_pdgId=[211, 22, 13],
        Packed_status=[1] * 3,
        Packed_mass=[0] * 3,
        Packed_x_mm=[0, 1, 0],
        Packed_y_mm=[0] * 3,
        Packed_z_mm=[0] * 3,
        Packed_pt=[10, 20, 30],
        Packed_eta=[0] * 3,
        Packed_phi=[0] * 3,
    )
    truth = []
    for barcode, index in enumerate([1, 0, 2], 1):
        truth.append(
            dict(
                barcode=barcode,
                pdg_id=r["Packed_pdgId"][index],
                generated_mass=0,
                position=[r["Packed_x_mm"][index], 0, 0, 0],
                p4=packed_p4(r["Packed_pt"][index], 0, 0, 0),
            )
        )
    mapping, delta = packed_join(dict(truth=truth), r)
    assert mapping == {1: 1, 2: 0, 3: 2} and delta == 0
    truth[1]["pdg_id"] = 22
    with pytest.raises(ValueError, match="PDG"):
        packed_join(dict(truth=truth), r)


def test_heavy_ghosts_do_not_change_jet_membership_or_scale():
    cfg = Config()
    particles = [
        dict(barcode=1, pdg_id=211, p4=[50, 0, 0, 50]),
        dict(barcode=2, pdg_id=-211, p4=[-60, 0, 0, 60]),
    ]
    r = record_graph([511, 421, 411], [[], [0], []])
    r.update(
        Pruned_px=[10, 5, -6], Pruned_py=[0, 0, 0], Pruned_pz=[0, 0, 0], Pruned_energy=[12, 6, 7]
    )
    event = dict(truth=particles, generator=dict(record=r, graph=graph_info(r)))
    jets = [
        dict(constituent_indices=j["constituent_indices"], **p4_fields(j["p4"]))
        for j in common_jets(particles, cfg)
    ]
    values = [heavy_associations(event, jets, cfg, scale)[0] for scale in [1e-18, 1e-20, 1e-22]]
    assert values[0] == values[1] == values[2]
    assert values[0][(1,)]["heavy_hadron_flavor"] == "b"
    assert values[0][(1,)]["c_from_b_indices"] == [1]
    assert values[0][(2,)]["heavy_hadron_flavor"] == "c"
    assert values[0][(2,)]["prompt_c_hadron_indices"] == [2]
