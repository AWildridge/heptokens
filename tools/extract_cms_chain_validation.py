"""CMSSW Python 2: matched-event AOD/MiniAOD PF and hard-GEN audit.

No fitting or retuning. Counts use pT>0.5 GeV and |eta|<2.4. GEN retains
status-1 particles, excluding neutrinos for the visible count. This diagnoses
packing/truth retention; it does not rerun Geant4 or measure detector closure.
"""
from __future__ import print_function
import argparse
import json
import os
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
from DataFormats.FWLite import Events, Handle

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--input', required=True)
p.add_argument('--tier', choices=['aod', 'miniaod'], required=True)
p.add_argument('--reference', required=True)
p.add_argument('--limit', type=int, default=100)
p.add_argument('--output', required=True)
a = p.parse_args()
if os.path.exists(a.output):
    raise RuntimeError('Refusing to overwrite a finalized audit')
references = json.load(open(a.reference))['events'][:a.limit]
keys = [(int(x['run']), int(x['lumi']), int(x['event'])) for x in references]
wanted = set(keys)
if len(wanted) != len(keys):
    raise ValueError('Duplicate reference event identities')
mini = a.tier == 'miniaod'
pf = Handle('std::vector<pat::PackedCandidate>' if mini else 'std::vector<reco::PFCandidate>')
gen = Handle('std::vector<pat::PackedGenParticle>' if mini else 'std::vector<reco::GenParticle>')
weight = Handle('GenEventInfoProduct')
reference_weights = {k: float(v['weight']) for k, v in zip(keys, references)}
rows = {}
for event in Events([a.input]):
    aux = event.eventAuxiliary()
    key = (int(aux.run()), int(aux.luminosityBlock()), int(aux.event()))
    if key not in wanted:
        continue
    if key in rows:
        raise ValueError('Duplicate input event identity')
    event.getByLabel('packedPFCandidates' if mini else 'particleFlow', pf)
    event.getByLabel('packedGenParticles' if mini else 'genParticles', gen)
    event.getByLabel('generator', weight)
    if not pf.isValid() or not gen.isValid() or not weight.isValid():
        raise RuntimeError('Missing mandatory PF or GEN collection')
    w = float(weight.product().weight())
    if abs(w-reference_weights[key]) > 1e-6*max(1., abs(w)):
        raise ValueError('Matched generator weight differs from the original reference')
    row = dict(key=list(key), weight=w)
    for label, collection in [('pf', pf.product()), ('gen', gen.product())]:
        selected = [v for v in collection if float(v.pt()) > .5 and abs(float(v.eta())) < 2.4
                    and (label != 'gen' or (int(v.status()) == 1 and abs(int(v.pdgId())) not in (12,14,16)))]
        row[label+'_count'] = len(selected)
        row[label+'_charged'] = sum(int(v.charge()) != 0 for v in selected)
        row[label+'_neutral'] = len(selected)-row[label+'_charged']
        row[label+'_sumpt'] = sum(float(v.pt()) for v in selected)
    rows[key] = row
    if len(rows) == len(wanted):
        break
missing = [list(k) for k in keys if k not in rows]
result = dict(tier=a.tier, input=a.input, matched=len(rows), requested=len(keys),
              missing=missing, events=[rows[k] for k in keys if k in rows],
              cuts=dict(pt_min_gev=.5, abs_eta_max=2.4),
              approximation='MiniAOD GEN is packed hard-event truth; neither GEN collection includes reconstructed pileup')
with open(a.output+'.partial', 'w') as f:
    json.dump(result, f, indent=2)
    f.flush()
    os.fsync(f.fileno())
os.rename(a.output+'.partial', a.output)
print(json.dumps(dict(matched=len(rows), requested=len(keys), missing=len(missing))))
if missing:
    raise RuntimeError('Some requested matched events are absent; see saved audit')
