"""Run one bounded pilot chunk with the native Delphes PileUpMerger.

Supports the stock pileup card or native overlay added to the default CMS
card without changing its response/reconstruction. Production cards are immutable.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import awkward as ak
import numpy as np
import uproot
from replay_runtime import run_cpu_process
from replay_scratch_budget import durable_json


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def make_card(delphes, library, mean, sigma_z, seed, distribution=0, mean_z=0., base_card='pileup', counts=None):
    pileup_text = (delphes / 'cards/delphes_card_CMS_PileUp.tcl').read_text()
    text = pileup_text
    if base_card == 'default':
        text = (delphes / 'cards/delphes_card_CMS.tcl').read_text()
        merger = re.search(r'(?ms)^module PileUpMerger PileUpMerger \{.*?^\}', pileup_text).group(0)
        text = text.replace('set ExecutionPath {', 'set ExecutionPath {\n  PileUpMerger', 1)
        text = text.replace('set InputArray Delphes/stableParticles',
                            'set InputArray PileUpMerger/stableParticles', 1)
        text += '\n' + merger + '\n'
    text, n = re.subn(r'(?m)^\s*set PileUpFile \S+\s*$', '\n  set PileUpFile {' + str(library.resolve()) + '}', text)
    assert n == 1
    text, n = re.subn(r'(?m)^\s*set MeanPileUp \S+\s*$', '\n  set MeanPileUp %.12g\n  set PileUpDistribution %d' % (mean, distribution), text)
    assert n == 1
    text, n = re.subn(r'(?m)^\s*set VertexDistributionFormula .+$',
                     '\n  set VertexDistributionFormula {exp(-(t^2/160e-12^2/2))*exp(-((z-(%.12g))^2/%.12g^2/2))}' % (mean_z, sigma_z), text)
    assert n == 1
    # Retain native reconstructed branches and vertices. Hard truth is already
    # saved, checksummed and shared with the existing matched 100k validation.
    text = re.sub(r'(?m)^\s*add Branch Delphes/allParticles Particle GenParticle\s*$', '', text)
    text = re.sub(r'(?m)^\s*add Branch GenJetFinder/jets GenJet Jet\s*$', '', text)
    marker = 'module TreeWriter TreeWriter {'
    if base_card == 'pileup':
        text = text.replace(marker, marker + '''
  add Branch HCal/eflowTracks EFlowTrack Track
  add Branch ECal/eflowPhotons EFlowPhoton Tower
  add Branch HCal/eflowNeutralHadrons EFlowNeutralHadron Tower
  add Branch TrackPileUpSubtractor/eflowTracks CHSTrack Track
''')
    else:
        # Native default card already writes raw PF branches. No CHS, rho, jet
        # subtraction or detector-threshold change is introduced.
        text = text.replace(marker, marker + '\n  add Branch PileUpMerger/vertices Vertex Vertex\n')
        for branch in ['TrackMerger/tracks Track Track', 'Calorimeter/towers Tower Tower']:
            text = re.sub(r'(?m)^\s*add Branch ' + re.escape(branch) + r'\s*$', '', text)
    if counts is not None:
        marker = 'module PileUpMerger PileUpMerger {'
        if text.count(marker) != 1:
            raise ValueError('Expected exactly one native merger')
        text = text.replace(marker, marker + '\n  set OrderedPileUp true\n  set PileUpCountsFile {' + str(counts.resolve()) + '}')
    return text + '\nset RandomSeed %d\n' % seed


def extract(root_path, reference, base_card='pileup'):
    # These are local native outputs. Explicit mmap avoids a stalled asynchronous
    # FSSpecSource queue observed in the AF sandbox (2026-10-27); no data or
    # observable definition changes. Remote sources are not accepted here.
    with uproot.open(root_path, handler=uproot.source.file.MemmapSource) as source:
        tree = source['Delphes']
        if tree.num_entries != len(reference):
            raise ValueError('Native event count differs')
        names = ['Event.Number', 'Event.Weight', 'Vertex.Z', 'Vertex.T',
                 'EFlowTrack.PT', 'EFlowTrack.Eta',
                 'EFlowPhoton.ET', 'EFlowPhoton.Eta', 'EFlowNeutralHadron.ET', 'EFlowNeutralHadron.Eta',
                 'Jet.PT', 'Jet.Eta', 'Electron.PT', 'Electron.Eta', 'Muon.PT', 'Muon.Eta', 'MissingET.MET']
        if base_card == 'pileup':
            names += ['CHSTrack.PT', 'CHSTrack.Eta']
        a = tree.arrays(names, library='ak')
        numbers = ak.to_numpy(ak.firsts(a['Event.Number']))
        weights = ak.to_numpy(ak.firsts(a['Event.Weight']))
        np.testing.assert_array_equal(numbers, [r['export_event'] for r in reference])
        np.testing.assert_allclose(weights, [r['weight'] for r in reference], rtol=1e-6, atol=1e-9)
        if not np.isfinite(weights).all():
            raise ValueError('Nonfinite weights')
        def count(branch, field, pt):
            values = a[branch + '.' + field]
            eta = a[branch + '.Eta']
            if not np.isfinite(ak.to_numpy(ak.flatten(values))).all():
                raise ValueError('Nonfinite native object')
            return ak.to_numpy(ak.sum((values > pt) & (abs(eta) < 2.4), axis=1)).astype(int)
        charged = count('EFlowTrack', 'PT', .5)
        chs = count('CHSTrack', 'PT', .5) if base_card == 'pileup' else charged
        photons = count('EFlowPhoton', 'ET', .5)
        neutral = count('EFlowNeutralHadron', 'ET', .5) + photons
        counts = {'charged_pf': charged, 'chs_charged_pf': chs,
                  'neutral_pf': neutral, 'pf': charged + neutral,
                  'chs_pf': chs + neutral, 'photons': photons,
                  'jets': count('Jet', 'PT', 30.), 'electrons': count('Electron', 'PT', 10.),
                  'muons': count('Muon', 'PT', 10.),
                  'n_pu': ak.to_numpy(ak.num(a['Vertex.Z'], axis=1)) - 1,
                  'met': ak.to_numpy(ak.firsts(a['MissingET.MET']))}
        if (chs > charged).any() or (counts['n_pu'] < 0).any():
            raise ValueError('Invalid native subtraction or vertex count')
        rows = []
        for i, ref in enumerate(reference):
            row = {'key': [ref['run'], ref['lumi'], ref['event']], 'weight': float(weights[i])}
            row.update({key: float(value[i]) if key == 'met' else int(value[i]) for key, value in counts.items()})
            if base_card == 'default':
                row['chs_charged_pf'] = row['chs_pf'] = None
            rows.append(row)
        return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['input', 'delphes', 'library', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--mean', type=float, required=True)
    p.add_argument('--sigma-z-m', type=float, required=True)
    p.add_argument('--mean-z-m', type=float, default=0.)
    p.add_argument('--distribution', type=int, choices=[0, 2], default=0)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--base-card', choices=['pileup', 'default'], default='pileup')
    p.add_argument('--counts', type=Path, help='Ordered runtime only: one recorded in-time N per hard event')
    a = p.parse_args()
    a.delphes = a.delphes.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    record = a.output / 'summary.json'
    reference_path = Path(str(a.input) + '.json')
    reference = json.loads(reference_path.read_text())
    truth_sha = digest(a.input)
    if truth_sha != reference['sha256']:
        raise ValueError('Frozen hard truth checksum differs')
    expected_counts = None
    if a.counts is not None:
        expected_counts = [int(v) for v in a.counts.read_text().split()]
        if len(expected_counts) != len(reference['events']) or any(n < 0 for n in expected_counts):
            raise ValueError('Ordered count input differs from hard-event count')
    card = make_card(a.delphes, a.library, a.mean, a.sigma_z_m, a.seed, a.distribution, a.mean_z_m, a.base_card, a.counts)
    card_path = a.output / 'card.tcl'
    root_path = a.output / 'delphes.root'
    identity = {'truth_sha256': truth_sha, 'library_sha256': digest(a.library),
                'card_sha256': hashlib.sha256(card.encode()).hexdigest(), 'seed': a.seed}
    if a.counts is not None:
        identity.update(counts_sha256=digest(a.counts), detector_binary_sha256=digest(a.delphes / 'DelphesHepMC2'),
                        detector_library_sha256=digest(a.delphes / 'libDelphes.so'))
    if record.exists():
        old = json.loads(record.read_text())
        if old['identity'] != identity or digest(root_path) != old['root_sha256']:
            raise ValueError('Existing pilot identity differs')
        print('Reused validated', record)
        return
    if root_path.exists():
        raise RuntimeError('Unvalidated native ROOT retained; inspect before retry')
    card_path.write_text(card)
    started = time.time()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [str(a.delphes / 'DelphesHepMC2'), str(card_path.resolve()), str(root_path.resolve()), str(a.input.resolve())]
    with (a.output / 'native-command.log').open('w') as stream:
        run_cpu_process(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
    rows = extract(root_path, reference['events'], a.base_card)
    if a.mean == 0 and any(r['n_pu'] for r in rows):
        raise ValueError('Zero-pileup control has overlaid vertices')
    if a.distribution == 2 and a.counts is None and any(r['n_pu'] != int(a.mean) for r in rows):
        raise ValueError('Native fixed multiplicity differs from requested CMS count')
    if expected_counts is not None and [r['n_pu'] for r in rows] != expected_counts:
        raise ValueError('Native ordered counts differ from every recorded CMS N')
    durable_json(record, {'status': 'validated', 'identity': identity, 'events': len(rows),
                          'base_card': a.base_card, 'chs_available': a.base_card == 'pileup',
                          'mean_pileup_parameter': a.mean, 'sigma_z_m': a.sigma_z_m,
                          'vertex_time_model': 'Gaussian sigma160ps, truncated at800ps; assumed',
                          'root_sha256': digest(root_path), 'root_bytes': root_path.stat().st_size,
                          'seconds': time.time() - started, 'rows': rows})
    print('Validated', len(rows), 'events;', root_path.stat().st_size, 'native ROOT bytes', flush=True)


if __name__ == '__main__':
    main()
