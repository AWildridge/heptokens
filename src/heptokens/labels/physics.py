"""Common particle jets, full C/A trees, and explicitly geometric plane angles."""

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import math

import awkward as ak
import fastjet as fj
import numpy as np
from particle import Particle, ParticleNotFound, InvalidParticle
from scipy.optimize import linear_sum_assignment
import vector

vector.register_awkward()


@dataclass(frozen=True)
class Config:
    include_trees: bool = False
    save_all_truth_particles: bool = False
    save_unselected_reco: bool = False
    radius: float = 0.5
    particle_abs_eta: float = 5.0
    jet_min_pt: float = 30.0
    jet_abs_eta: float = 2.5
    include_muons: bool = True
    invisible_pids: tuple = (12, 14, 16, 18)
    match_dr: float = 0.25
    match_response_min: float = 0.25
    match_response_max: float = 4.0
    reject_ambiguous: bool = False
    plane_min_z: float = 0.02
    plane_min_dr: float = 0.001
    plane_min_kt: float = 0.5
    plane_sine_epsilon: float = 1e-10
    split_namespace: str = "cms-qcd-stable-v1"

    def __post_init__(self):
        if not (0 < self.radius < math.pi and self.particle_abs_eta > 0
                and self.jet_min_pt > 0 and self.jet_abs_eta > 0
                and self.match_dr > 0 and 0 < self.match_response_min
                < self.match_response_max and 0 <= self.plane_min_z < 0.5
                and self.plane_min_dr >= 0 and self.plane_min_kt >= 0
                and self.plane_sine_epsilon > 0):
            raise ValueError("Invalid physics cuts")

    @property
    def digest(self):
        import json
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def stable_id(*parts):
    """Full SHA256, with length-delimited fields (no delimiter collisions)."""
    h = hashlib.sha256()
    for part in parts:
        raw = str(part).encode()
        h.update(len(raw).to_bytes(8, "big")); h.update(raw)
    return h.hexdigest()


def event_identity(sample_id, run, lumi, event, namespace="cms-qcd-stable-v1"):
    # Excluding source file and detector keeps copies/replicas of an event together.
    return stable_id(namespace, sample_id, int(run), int(lumi), int(event))


def split_for(event_id):
    bucket = int(event_id[:16], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


@lru_cache(maxsize=4096)
def charge(pdg_id):
    try:
        value = Particle.from_pdgid(int(pdg_id)).charge
        return None if value is None else float(value)
    except (ValueError, LookupError, ParticleNotFound, InvalidParticle):
        return None


def p4_fields(p4):
    p = vector.obj(px=float(p4[0]), py=float(p4[1]), pz=float(p4[2]), E=float(p4[3]))
    return dict(px=p.px, py=p.py, pz=p.pz, energy=p.E, pt=p.pt,
                eta=p.eta if p.pt > 0 else None, phi=p.phi,
                rapidity=p.rapidity if p.E > abs(p.pz) else None,
                mass=math.sqrt(max(0.0, p.mass2)))


def p4_array(rows):
    arr = np.asarray(rows, dtype=np.float64).reshape(-1, 4)
    return ak.zip(dict(px=arr[:, 0], py=arr[:, 1], pz=arr[:, 2], E=arr[:, 3]),
                  with_name="Momentum4D")


def pseudos(p4, indices):
    result = []
    for row, index in zip(p4, indices):
        p = fj.PseudoJet(*[float(x) for x in row]); p.set_user_index(int(index))
        result.append(p)
    return result


def members(jet):
    return sorted(c.user_index() for c in jet.constituents())


def jet_key(jet):
    return (-jet.pt(), tuple(members(jet)))


def common_jets(particles, cfg):
    p4 = np.asarray([p['p4'] for p in particles], dtype=float).reshape(-1, 4)
    if not len(p4):
        return []
    a = p4_array(p4)
    pid = np.asarray([p['pdg_id'] for p in particles])
    mask = (ak.to_numpy(a.pt) > 0) & (np.abs(ak.to_numpy(a.eta)) < cfg.particle_abs_eta)
    mask &= ~np.isin(np.abs(pid), cfg.invisible_pids)
    if not cfg.include_muons:
        mask &= np.abs(pid) != 13
    indices = np.asarray([p['barcode'] for p in particles])[mask]
    if not len(indices):
        return []
    seq = fj.ClusterSequence(pseudos(p4[mask], indices), fj.JetDefinition(fj.antikt_algorithm, cfg.radius))
    selected = sorted((j for j in seq.inclusive_jets(cfg.jet_min_pt)
                       if abs(j.eta()) < cfg.jet_abs_eta), key=jet_key)
    return [dict(p4=[j.px(), j.py(), j.pz(), j.E()], constituent_indices=members(j))
            for j in selected]


def ca_tree(p4, indices, event_id, jet_id):
    """Exclusive C/A R=max_allowable_R joins every constituent into one tree.

    The large radius prevents beam declarations from fragmenting a truth jet.
    C/A distances use rapidity and wrapped phi; E-scheme recombination.
    """
    if not len(indices):
        return []
    seq = fj.ClusterSequence(pseudos(p4, indices),
                             fj.JetDefinition(fj.cambridge_algorithm, fj.JetDefinition.max_allowable_R))
    roots = seq.exclusive_jets(1)
    if len(roots) != 1:
        raise ValueError("C/A failed to produce a full tree")
    stack = [(roots[0], None)]
    rows = []
    while stack:
        node, parent_id = stack.pop()
        membership = members(node)
        node_id = stable_id(jet_id, "ca", *membership)
        b, c = fj.PseudoJet(), fj.PseudoJet()
        internal = node.has_parents(b, c)
        row = dict(event_id=event_id, jet_id=jet_id, node_id=node_id, parent_id=parent_id,
                   constituent_indices=membership, is_leaf=not internal,
                   child_hard_id=None, child_soft_id=None, delta_r=None, z=None, kt=None,
                   shower_category="unknown", shower_category_valid=False,
                   **p4_fields([node.px(), node.py(), node.pz(), node.E()]))
        if internal:
            b, c = sorted([b, c], key=jet_key)
            dr = b.delta_R(c)
            row.update(child_hard_id=stable_id(jet_id, "ca", *members(b)),
                       child_soft_id=stable_id(jet_id, "ca", *members(c)), delta_r=dr,
                       z=min(b.pt(), c.pt()) / (b.pt()+c.pt()), kt=min(b.pt(), c.pt())*dr)
            stack.extend([(c, node_id), (b, node_id)])
        rows.append(row)
    if len(rows) != 2 * len(indices) - 1:
        raise ValueError("Incomplete binary C/A tree")
    return rows


def plane_angles(nodes, cfg):
    by_id = {n['node_id']: n for n in nodes}
    rows = []
    for a in nodes:
        if a['is_leaf']:
            continue
        # Include secondary branching on BOTH daughters, including the soft branch.
        for branch, other in [('child_hard_id', 'child_soft_id'), ('child_soft_id', 'child_hard_id')]:
            c, b = by_id[a[branch]], by_id[a[other]]
            if c['is_leaf']:
                continue
            d, e = by_id[c['child_hard_id']], by_id[c['child_soft_id']]
            def xyz(n):
                return np.array([n['px'], n['py'], n['pz']])
            pb, pc, pd, pe = map(xyz, [b, c, d, e])
            n1, n2 = np.cross(pb, pc), np.cross(pd, pe)
            s1, s2 = np.linalg.norm(n1), np.linalg.norm(n2)
            scale1, scale2 = np.linalg.norm(pb)*np.linalg.norm(pc), np.linalg.norm(pd)*np.linalg.norm(pe)
            geometric = (scale1 > 0 and scale2 > 0 and np.linalg.norm(pc) > 0
                         and s1 > cfg.plane_sine_epsilon*scale1
                         and s2 > cfg.plane_sine_epsilon*scale2)
            resolved = all(n['z'] >= cfg.plane_min_z and n['delta_r'] >= cfg.plane_min_dr
                           and n['kt'] >= cfg.plane_min_kt for n in [a, c])
            psi = None
            if geometric:
                n1 /= s1; n2 /= s2
                psi = math.atan2(float(np.dot(pc/np.linalg.norm(pc), np.cross(n1, n2))),
                                 float(np.clip(np.dot(n1, n2), -1, 1)))
            rows.append(dict(event_id=a['event_id'], jet_id=a['jet_id'],
                             angle_id=stable_id(a['node_id'], c['node_id'], 'plane'),
                             first_node_id=a['node_id'], second_node_id=c['node_id'],
                             daughter_branch='hard' if branch == 'child_hard_id' else 'soft',
                             psi=psi, cos2psi=None if psi is None else math.cos(2*psi),
                             sin2psi=None if psi is None else math.sin(2*psi),
                             geometric_valid=bool(geometric), resolution_valid=bool(resolved),
                             valid=bool(geometric and resolved)))
    return rows


def match_jets(reco, truth, cfg):
    """Maximum-cardinality/minimum-deltaR global one-to-one matching.

    Candidate cuts use eta/phi distance and reconstructed/truth pT response.
    Ambiguity is retained if either endpoint has multiple valid candidates.
    """
    nr, nt = len(reco), len(truth)
    dr = np.zeros((nr, nt)); response = np.zeros_like(dr)
    for i, r in enumerate(reco):
        for j, t in enumerate(truth):
            dphi = math.remainder(r['phi']-t['phi'], 2*math.pi)
            dr[i, j] = math.hypot(r['eta']-t['eta'], dphi)
            response[i, j] = r['pt']/t['pt']
    candidates = ((dr < cfg.match_dr) & (response >= cfg.match_response_min)
                  & (response <= cfg.match_response_max))
    ambiguous = candidates & ((candidates.sum(1)[:, None] > 1) | (candidates.sum(0)[None, :] > 1))
    allowed = candidates & ~ambiguous if cfg.reject_ambiguous else candidates
    # Per-row dummy assignments: the penalty exceeds the sum of all possible
    # distance savings, so maximizing cardinality takes precedence over distance.
    penalty = (nr + nt + 1)*(cfg.match_dr+1)
    cost = np.full((nr, nt+nr), penalty)
    cost[:, :nt] = np.where(allowed, dr, 3*penalty)
    chosen = {}
    if nr:
        rr, cc = linear_sum_assignment(cost)
        chosen = {int(i): int(j) for i, j in zip(rr, cc) if j < nt and allowed[i, j]}
    result = []
    for i, r in enumerate(reco):
        j = chosen.get(i)
        result.append(dict(event_id=r['event_id'], reco_jet_id=r['jet_id'],
                           truth_jet_id=None if j is None else truth[j]['jet_id'],
                           matched=j is not None, unmatched=j is None,
                           ambiguous=bool(ambiguous[i].any()), candidate_count=int(candidates[i].sum()),
                           delta_r=None if j is None else float(dr[i, j]),
                           pt_response=None if j is None else float(response[i, j]),
                           mass_response=None if j is None or truth[j]['mass'] <= 0 else r['mass']/truth[j]['mass']))
    return result


def event_kinematics(jets):
    result = dict(ht=sum(j['pt'] for j in jets), n_jets=len(jets), n_b_jets=None,
                  n_c_jets=None, heavy_flavor_valid=False, dijet_mass=None,
                  dijet_delta_y=None, dijet_delta_phi=None, dijet_valid=False)
    if len(jets) >= 2:
        a, b = sorted(jets, key=lambda j: (-j['pt'], j['jet_id']))[:2]
        p = np.array([a[k]+b[k] for k in ['px', 'py', 'pz', 'energy']])
        dy = None if a['rapidity'] is None or b['rapidity'] is None else a['rapidity']-b['rapidity']
        result.update(dijet_mass=p4_fields(p)['mass'], dijet_delta_y=dy,
                      dijet_delta_phi=math.remainder(a['phi']-b['phi'], 2*math.pi),
                      dijet_valid=dy is not None)
    return result
