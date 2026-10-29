"""Physics identities, truth-content certification, missing labels and publication."""

from dataclasses import replace
import gzip
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
import pytest

from heptokens.labels.cli import account_weights
from heptokens.labels.physics import (Config, ca_tree, charge, common_jets, event_identity,
    event_kinematics, match_jets, p4_fields, plane_angles, split_for)
from heptokens.labels.produce import produce_source
from heptokens.labels.source import atomic_json, iter_events, sha256, truth_fingerprint
from heptokens.labels.validation import check_tree, validate_run


def particle(index, pid, p4):
    return dict(barcode=index, pdg_id=pid, p4=p4, generated_mass=0., position=[0.,0.,0.,0.])


def jet(index, pt, eta, phi, event='e'):
    return dict(event_id=event, jet_id=index, **p4_fields(
        [pt*math.cos(phi),pt*math.sin(phi),pt*math.sinh(eta),pt*math.cosh(eta)]))


def test_identity_groups_replicas_and_detector_views():
    a = event_identity('63193',1,4,5)
    assert a == event_identity('63193',1,4,5)
    assert a != event_identity('63217',1,4,5)
    assert a != event_identity('63193',1,4,6)
    assert split_for(a) in ['train','validation','test']


def test_truth_fingerprint_uses_content_not_order():
    a = [particle(1,211,[10.,2.,1.,11.]),particle(2,22,[4.,1.,1.,5.])]
    assert truth_fingerprint(a) == truth_fingerprint(a[::-1])
    b = [dict(a[0],pdg_id=321),a[1]]
    assert truth_fingerprint(a) != truth_fingerprint(b)


def test_charge_unknown_is_missing():
    assert charge(211) == 1 and charge(-211) == -1 and charge(22) == 0
    assert charge(999999) is None


def test_truth_jet_neutrino_muon_and_acceptance_selection():
    ps = [particle(1,211,[40.,0,0,40.]),particle(2,12,[500.,0,0,500.]),
          particle(3,13,[10.,0,0,10.]),particle(4,22,[1.,0,1000,1001.])]
    cfg = Config(jet_min_pt=20)
    a = common_jets(ps,cfg)
    assert len(a) == 1 and a[0]['constituent_indices'] == [1,3]
    assert a[0]['p4'][0] == 50
    assert common_jets(ps,replace(cfg,include_muons=False))[0]['constituent_indices'] == [1]


def test_full_binary_tree_conserves_membership_and_p4():
    p4 = [[20.,0,1,21],[4.,1,2,5],[3.,-.5,1,4],[.5,.2,-.1,1]]
    n = ca_tree(p4,[1,2,3,4],'e','j')
    assert len(n) == 7
    check_tree(n,[dict(jet_id='j',constituent_indices=[1,2,3,4])])
    assert all(x['shower_category'] == 'unknown' and not x['shower_category_valid'] for x in n)


def test_analytic_signed_angle_and_both_daughters():
    def node(name,p4,hard=None,soft=None):
        return dict(event_id='e',jet_id='j',node_id=name,is_leaf=hard is None,
            child_hard_id=hard,child_soft_id=soft,z=.25,delta_r=.2,kt=2.,**p4_fields(p4))
    # C is along +z. n1 along +x, n2 along (+x,-y), hence psi=-pi/4.
    n = [node('a',[0,1,3,8],'b','c'),node('b',[0,1,1,3]),
         node('c',[0,0,2,5],'d','e'),node('d',[1,1,1,2.5]),node('e',[-1,-1,1,2.5])]
    a = plane_angles(n,Config())[0]
    assert a['daughter_branch'] == 'soft'
    assert a['psi'] == pytest.approx(-math.pi/4)
    assert a['sin2psi'] == pytest.approx(-1)
    assert a['cos2psi'] == pytest.approx(0,abs=1e-12)
    assert a['valid']
    n[1].update(is_leaf=False,child_hard_id='f',child_soft_id='g')
    n.extend([node('f',[0,.5,.5,1.5]),node('g',[0,.5,.5,1.5])])
    assert {a['daughter_branch'] for a in plane_angles(n,Config()) if a['first_node_id']=='a'} == {'hard','soft'}


def test_degenerate_planes_are_masked():
    n = ca_tree([[10,0,0,10],[5,0,0,5],[1,0,0,1]],[1,2,3],'e','j')
    angles = plane_angles(n,Config())
    assert len(angles) == 1
    assert not angles[0]['geometric_valid'] and not angles[0]['valid']
    assert angles[0]['psi'] is None


def test_global_matching_maximizes_cardinality_before_distance():
    # Greedy picks r0->t0, stranding r1. The global solution matches both.
    reco = [jet('r0',100,0,.04),jet('r1',100,0,-.05)]
    truth = [jet('t0',100,0,0),jet('t1',100,0,.15)]
    m = match_jets(reco,truth,Config(match_dr=.13))
    assert [a['truth_jet_id'] for a in m] == ['t1','t0']
    assert all(a['ambiguous'] for a in m)
    assert not any(a['matched'] for a in match_jets(reco,truth,Config(match_dr=.13,reject_ambiguous=True)))


def test_matching_wraps_phi_and_respects_response():
    r = [jet('r',100,0,math.pi-.01)]
    t = [jet('t',100,0,-math.pi+.01)]
    assert match_jets(r,t,Config())[0]['delta_r'] == pytest.approx(.02)
    assert not match_jets([jet('r',10000,0,0)],[jet('t',100,0,0)],Config())[0]['matched']
    assert match_jets(r,[],Config())[0]['unmatched']
    assert match_jets([],t,Config()) == []


def test_continuous_event_kinematics():
    r = event_kinematics([jet('a',100,0,0),jet('b',80,0,math.pi)])
    assert r['ht'] == 180 and r['n_jets'] == 2
    assert r['dijet_mass'] == pytest.approx(math.sqrt(32000))
    assert r['n_b_jets'] is None and not r['heavy_flavor_valid']


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path/'raw';chunk = root/'sources/00000/chunk-000000';chunk.mkdir(parents=True)
    source = dict(task=0,record='63193',dataset='/QCD_Pt_300to470_test',uri='root://example/original',checksum='adler32:0')
    record = dict(record='63193',dataset=source['dataset'],events=2,cross_section=dict(total_value='10',total_value_uncertainty='1',filter_efficiency='1'))
    manifest = dict(files=[source],records=[record])
    (root/'provenance').mkdir();atomic_json(root/'provenance/cms-qcd-source-manifest.json',manifest)
    refs = [dict(run=1,lumi=10,event=i+1,export_event=i,weight=w,generator_weights=[w],lhe_weights=None) for i,w in enumerate([2.,-1.])]
    ps = [particle(1,211,[40.,0,1.,41.]),particle(2,22,[8.,1,1.,9.]),particle(3,13,[3.,-.2,.5,4.])]
    raw = 'HepMC::Version 2.06.09\nHepMC::IO_GenEvent-START_EVENT_LISTING\n'
    for e in refs:
        raw += f"E {e['export_event']} 0 0 0 0 0 0 1 0 0 0 1 {e['weight']}\nU GEV MM\nV -1 0 0 0 0 0 0 3 0\n"
        for p in ps:
            raw += f"P {p['barcode']} {p['pdg_id']} "+' '.join(map(str,p['p4']))+' 0 1 0 0 0 0\n'
    raw += 'HepMC::IO_GenEvent-END_EVENT_LISTING\n'
    import hashlib
    original_hash = hashlib.sha256(raw.encode()).hexdigest()
    with gzip.open(chunk/'truth.hepmc.gz','wt') as f:f.write(raw)
    with gzip.open(chunk/'truth-reference.json.gz','wt') as f:
        json.dump(dict(source_key=source['uri'],events=refs,sha256=original_hash,scope='stable only'),f)
    for detector in ['CMS','ATLAS']:
        (chunk/f'delphes_card_{detector}.tcl').write_text('set ParameterR 0.5\n')
        order = [0,1] if detector == 'CMS' else [1,0]
        with h5py.File(chunk/f'{detector}.h5','w') as f:
            f.attrs.update(schema='delphes-ragged-p4-v1',detector=detector)
            for k in ['run','lumi','event','export_event','weight']:
                f.create_dataset('events/'+k,data=[refs[i][k] for i in order])
            f.create_dataset('events/split',data=[0,0])
            for k,values in dict(event_offsets=[0,1,2],jet_p4=[[51.,.8,2.5,54.]]*2,
                constituent_offsets=[0,3,6],constituent_p4=[p['p4'] for p in ps]*2,
                charge=[1,0,-1]*2,constituent_type=[1,2,1]*2).items():
                f.create_dataset('Jet/'+k,data=values)
    validation = dict(status='passed',events=2,input_sha256=original_hash,detectors={d:dict(
        status='passed',truth_and_weights='passed',card_sha256=sha256(chunk/f'delphes_card_{d}.tcl'),stable_particle_counts=[3,3]) for d in ['CMS','ATLAS']})
    atomic_json(chunk/'validation.json',validation)
    names = ['CMS.h5','ATLAS.h5','truth.hepmc.gz','truth-reference.json.gz','validation.json','delphes_card_CMS.tcl','delphes_card_ATLAS.tcl']
    receipt = dict(status='complete',events=2,input_sha256=original_hash,source=source,seed=42,sha256={n:sha256(chunk/n) for n in names})
    atomic_json(chunk/'receipt.json',receipt)
    atomic_json(root/'sources/00000/COMPLETE.json',dict(status='complete',source=source,exported_events=2,
        chunks=[dict(chunk='chunk-000000',events=2)],manifest_sha256=sha256(root/'provenance/cms-qcd-source-manifest.json')))
    return root


def test_detector_mapping_is_by_identity_even_with_permuted_rows(campaign):
    events = list(iter_events(campaign/'sources/00000/chunk-000000',Config()))
    assert len(events) == 2 and events[0]['reference']['weight'] == 2.
    assert events[1]['reference']['weight'] == -1.
    assert events[0]['event_id'] != events[1]['event_id']


def test_changed_truth_payload_is_rejected(campaign):
    with (campaign/'sources/00000/chunk-000000/truth.hepmc.gz').open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        list(iter_events(campaign/'sources/00000/chunk-000000',Config()))


def test_slim_end_to_end_and_signed_normalization(campaign,tmp_path):
    normalization = account_weights(campaign,tmp_path/'weight-accounting',[0])
    assert normalization['valid'] and normalization['samples']['63193']['sum_weights'] == 1.
    out = tmp_path/'labels'
    result = produce_source(campaign,out,0,Config(),normalization=normalization)
    assert result['events'] == 2
    assert result['tables']['truth/nodes.parquet']['rows'] == 0
    jets = pq.read_table(out/'source-00000/truth/jets.parquet').to_pylist()
    assert all(j['heavy_hadron_flavor']=='unknown' and not j['parton_flavor_valid'] for j in jets)
    book = pq.read_table(out/'source-00000/bookkeeping/events.parquet').to_pylist()
    assert [r['physical_weight_pb'] for r in book] == [20.,-10.]
    assert all(r['generator_ht'] is None and not r['generator_ht_valid'] for r in book)
    report = validate_run(out)
    assert report['status']=='passed' and report['unique_events']==2 and report['campaign_complete']
    assert report['trees_valid'] is None and report['deferred_targets']
    assert produce_source(campaign,out,0,Config(),normalization=normalization)['events'] == 2
    with pytest.raises(ValueError,match='differs'):
        produce_source(campaign,out,0,Config(radius=.6),normalization=normalization)


def test_optional_trees_and_missing_flavor(campaign,tmp_path):
    out = tmp_path/'trees'
    result = produce_source(campaign,out,0,Config(include_trees=True))
    assert result['tables']['truth/nodes.parquet']['rows'] == 10
    report = validate_run(out)
    assert report['trees_valid']
    assert report['summaries']['63193/CMS/truth/raw_generator_weight']['fractions']['parton_flavor_missing/jets'] == 1.


def test_partial_accounting_does_not_claim_unit_weights(campaign,tmp_path):
    p = campaign/'provenance/cms-qcd-source-manifest.json'
    a=json.loads(p.read_text());a['records'][0]['events']=3;atomic_json(p,a)
    result=account_weights(campaign,tmp_path/'partial-account',[0])
    assert not result['valid'] and result['samples']['63193']['sum_weights']==1.
