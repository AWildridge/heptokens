"""Read immutable campaign chunks and certify the scope of synchronization."""

import gzip
import hashlib
import json
from pathlib import Path
import re

import h5py
import numpy as np

from .physics import event_identity, split_for, stable_id


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    path = Path(path)
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as f:
        return json.load(f)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sample_records(manifest):
    result = {}
    for record in manifest['records']:
        match = re.search(r'QCD_Pt_(\d+)to(\d+)_', record['dataset'])
        if match is None:
            raise ValueError('Unrecognized sample selection; do not infer HT bins')
        xs = record.get('cross_section', {})
        result[str(record['record'])] = dict(
            sample_id=str(record['record']), dataset=record['dataset'],
            selection_variable='generator_pthat', selection_low_GeV=float(match[1]),
            selection_high_GeV=float(match[2]), expected_events=record['events'],
            cross_section_pb=float(xs['total_value']) if 'total_value' in xs else None,
            cross_section_uncertainty_pb=float(xs['total_value_uncertainty']) if 'total_value_uncertainty' in xs else None,
            filter_efficiency=float(xs['filter_efficiency']) if 'filter_efficiency' in xs else None,
            cross_section_source=f"https://opendata.cern.ch/record/{record['record']}",
            sample_sum_weights=None, sample_sum_weights_valid=False,
            normalization='raw generator weights; sigma_pb/sumw requires a complete measured sample sumw')
    return result


def source_root(campaign, task):
    name = f'{int(task):05d}'
    local = Path(campaign)/'sources'/name
    imported = Path(campaign)/'purdue-import-20261026/sources'/name
    complete = [p for p in [local, imported] if (p/'COMPLETE.json').is_file()]
    if len(complete) != 1:
        raise ValueError(f'Expected exactly one committed source {task}, found {complete}')
    return complete[0]


def hepmc_events(path):
    """Stream the stable-only HepMC2 export; synthetic headers are NOT hard truth."""
    current = None
    position = None
    with gzip.open(path, 'rt') as f:
        for line in f:
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == 'E':
                if current is not None:
                    yield current
                current = dict(export_event=int(tokens[1]), particles=[], weight=float(tokens[-1]))
            elif tokens[0] == 'U' and tokens[1:] != ['GEV', 'MM']:
                raise ValueError('Expected GeV/mm HepMC')
            elif tokens[0] == 'V':
                position = [float(x) for x in tokens[3:7]]
            elif tokens[0] == 'P':
                if current is None or position is None or int(tokens[8]) != 1:
                    raise ValueError('Expected stable-only exported particles')
                current['particles'].append(dict(barcode=int(tokens[1]), pdg_id=int(tokens[2]),
                                                 p4=[float(x) for x in tokens[3:7]],
                                                 generated_mass=float(tokens[7]), position=position[:]))
    if current is not None:
        yield current


def truth_fingerprint(particles):
    # Lexicographic canonicalization makes the digest independent of particle order.
    rows = np.asarray([[p['pdg_id'], *p['p4'], p['generated_mass'], *p['position']]
                       for p in particles], dtype='<f8').reshape(-1, 10)
    if not np.isfinite(rows).all():
        raise ValueError('Nonfinite stable truth')
    order = np.lexsort(tuple(rows[:, i] for i in range(rows.shape[1]-1, -1, -1)))
    return hashlib.sha256(rows[order].tobytes()).hexdigest()


def certify_chunk(chunk, expected_source=None):
    chunk = Path(chunk)
    receipt = read_json(chunk/'receipt.json')
    reference = read_json(chunk/'truth-reference.json.gz')
    validation = read_json(chunk/'validation.json')
    if receipt['status'] != 'complete' or validation['status'] != 'passed':
        raise ValueError('Incomplete/unvalidated input chunk')
    names = ['CMS.h5', 'ATLAS.h5', 'truth.hepmc.gz', 'truth-reference.json.gz',
             'validation.json', 'delphes_card_CMS.tcl', 'delphes_card_ATLAS.tcl']
    hashes = {}
    for name in names:
        actual = sha256(chunk/name)
        if actual != receipt['sha256'][name]:
            raise ValueError(f'Payload checksum mismatch: {chunk/name}')
        hashes[name] = actual
    if not (receipt['input_sha256'] == reference['sha256'] == validation['input_sha256']):
        raise ValueError('Truth input hashes disagree')
    uri = receipt['source']['uri']
    if uri != reference['source_key']:
        raise ValueError('Source identity differs from truth reference')
    if expected_source and any(receipt['source'][k] != expected_source[k]
                               for k in ['task', 'record', 'uri', 'checksum']):
        raise ValueError('Receipt source differs from frozen manifest')
    refs = reference['events']
    if len(refs) != receipt['events'] or len(refs) != validation['events']:
        raise ValueError('Reference event counts disagree')
    keys = [(r['run'], r['lumi'], r['event']) for r in refs]
    if len(set(keys)) != len(keys) or len({r['export_event'] for r in refs}) != len(refs):
        raise ValueError('Duplicate event identity in a chunk')
    maps = {}
    for detector in ['CMS', 'ATLAS']:
        v = validation['detectors'][detector]
        if v['status'] != 'passed' or v.get('truth_and_weights') != 'passed':
            raise ValueError('Archived detector truth-content validation is missing')
        if v['card_sha256'] != hashes[f'delphes_card_{detector}.tcl']:
            raise ValueError('Card differs from detector truth validation')
        with h5py.File(chunk/f'{detector}.h5') as f:
            if f.attrs.get('schema') != 'delphes-ragged-p4-v1' or f.attrs['detector'] != detector:
                raise ValueError('Unexpected reconstructed schema/detector')
            reco_keys = list(zip(*(f['events/'+k][:].tolist() for k in ['run', 'lumi', 'event'])))
            if len(reco_keys) != len(keys) or set(reco_keys) != set(keys):
                raise ValueError('Detector event identities differ')
            maps[detector] = {k: i for i, k in enumerate(reco_keys)}
            for r, key in zip(refs, keys):
                i = maps[detector][key]
                if (int(f['events/export_event'][i]) != r['export_event']
                        or float(f['events/weight'][i]) != r['weight']):
                    raise ValueError('Detector event mapping/weight mismatch')
    evidence = dict(
        scope='identical retained stable hadronization outcome; original pileup and genealogy absent',
        method='fresh payload SHA256 + archived ROOT truth-PDG/P4/vertex/weight validation + fresh identity join',
        independent_fresh_detector_truth_comparison=False,
        limitation='ROOT truth branches were discarded after archived validation; compact HDF5 has no truth references',
        input_hashes=hashes, original_hepmc_sha256=reference['sha256'],
        seed=receipt['seed'], receipt_sha256=sha256(chunk/'receipt.json'))
    return receipt, reference, validation, maps, evidence


def iter_events(chunk, cfg, limit=None, expected_source=None):
    receipt, reference, validation, maps, evidence = certify_chunk(chunk, expected_source)
    refs = {r['export_event']: (i, r) for i, r in enumerate(reference['events'])}
    seen, identities = set(), {}
    handles = {d: h5py.File(Path(chunk)/f'{d}.h5') for d in ['CMS', 'ATLAS']}
    try:
        for truth in hepmc_events(Path(chunk)/'truth.hepmc.gz'):
            export = truth['export_event']
            if export in seen or export not in refs:
                raise ValueError('HepMC identity duplicate or missing from reference')
            seen.add(export)
            i, ref = refs[export]
            if truth['weight'] != ref['weight']:
                raise ValueError('HepMC nominal weight mismatch')
            if len({p['barcode'] for p in truth['particles']}) != len(truth['particles']):
                raise ValueError('Duplicate particle barcode')
            for d in handles:
                counts = validation['detectors'][d]['stable_particle_counts']
                if counts[i] != len(truth['particles']):
                    raise ValueError('Stable truth count differs from archived detector validation')
            sample = str(receipt['source']['record'])
            group = event_identity(sample, ref['run'], ref['lumi'], ref['event'], cfg.split_namespace)
            fingerprint = truth_fingerprint(truth['particles'])
            # Same hard-event/replica group always shares a split; share targets only
            # when the stable state itself is identical.
            event_id = stable_id(group, fingerprint)
            if group in identities and identities[group] != fingerprint:
                raise ValueError('Conflicting stable states for one event in a chunk')
            identities[group] = fingerprint
            reco = {}
            key = (ref['run'], ref['lumi'], ref['event'])
            for d, f in handles.items():
                row = maps[d][key]; g = f['Jet']
                start, stop = map(int, g['event_offsets'][row:row+2])
                jets = []
                for j in range(start, stop):
                    a, b = map(int, g['constituent_offsets'][j:j+2])
                    jets.append(dict(original_index=j-start, p4=g['jet_p4'][j].astype(float),
                                     constituent_p4=g['constituent_p4'][a:b].astype(float),
                                     charge=g['charge'][a:b], kind=g['constituent_type'][a:b],
                                     constituent_start=a))
                reco[d] = dict(jets=jets, original_split=int(f['events/split'][row]))
            yield dict(event_id=event_id, generator_event_id=group, split=split_for(group),
                       truth_fingerprint=fingerprint, sample_id=sample, reference=ref,
                       truth=truth['particles'], reco=reco, source_uri=receipt['source']['uri'],
                       source_id=int(receipt['source']['task']), evidence=evidence)
            if limit is not None and len(seen) >= limit:
                break
        if limit is None and len(seen) != len(refs):
            raise ValueError('HepMC event count mismatch')
    finally:
        for f in handles.values():
            f.close()
