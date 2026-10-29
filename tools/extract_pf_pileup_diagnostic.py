"""Bounded MiniAOD PF/vertex/threshold audit; CMSSW Python 2 compatible.

Reads existing events only. Bunch-crossing metadata are event-level and never
used as PF truth labels. No reconstruction, retuning, or pileup removal.
"""
from __future__ import print_function
import argparse
import gzip
import hashlib
import json
import os
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
from DataFormats.FWLite import Events, Handle

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--input', required=True)
p.add_argument('--reference', required=True)
p.add_argument('--output', required=True)
p.add_argument('--limit', type=int, default=1000)
a = p.parse_args()
if os.path.exists(a.output):
    raise RuntimeError('Refusing to overwrite finalized evidence')
references = json.load(open(a.reference))['events'][:a.limit]
thresholds = [.5, .7, 1., 2., 5.]
categories = ['charged', 'pv_charged', 'nonpv_charged', 'photon',
              'neutral_hadron', 'other_neutral', 'electron', 'muon',
              'high_purity_charged']
handles = dict(pf=Handle('std::vector<pat::PackedCandidate>'),
               pu=Handle('std::vector<PileupSummaryInfo>'),
               weight=Handle('GenEventInfoProduct'))
labels = dict(pf='packedPFCandidates', pu='slimmedAddPileupInfo', weight='generator')
rows = []
for i, event in enumerate(Events([a.input])):
    if i >= len(references):
        break
    ref = references[i]
    aux = event.eventAuxiliary()
    key = [int(aux.run()), int(aux.luminosityBlock()), int(aux.event())]
    if key != [ref[x] for x in ['run', 'lumi', 'event']]:
        raise ValueError('Event identity mismatch')
    for name in handles:
        event.getByLabel(labels[name], handles[name])
        if not handles[name].isValid():
            raise RuntimeError('Missing ' + labels[name])
    weight = float(handles['weight'].product().weight())
    if abs(weight-ref['weight']) > 1e-6*max(1., abs(weight)):
        raise ValueError('Generator weight mismatch')
    counts = {c: [0]*len(thresholds) for c in categories}
    association = [0]*4
    timing = dict(selected=0, available=0, nonzero_valid=0)
    for obj in handles['pf'].product():
        pt, eta = float(obj.pt()), float(obj.eta())
        if pt <= .5 or abs(eta) >= 2.4:
            continue
        charge, pid = int(obj.charge()), abs(int(obj.pdgId()))
        timing['selected'] += 1
        if float(obj.timeError()) > 0:
            timing['available'] += 1
            if float(obj.dtimeAssociatedPV()) != 0:
                timing['nonzero_valid'] += 1
        if charge:
            pv = int(obj.fromPV())
            association[pv] += 1
            names = ['charged', 'pv_charged' if pv > 1 else 'nonpv_charged']
            if bool(obj.trackHighPurity()):
                names.append('high_purity_charged')
            if pid == 11:
                names.append('electron')
            if pid == 13:
                names.append('muon')
        else:
            names = ['photon' if pid == 22 else ('neutral_hadron' if pid == 130 else 'other_neutral')]
        for c in names:
            for j, cut in enumerate(thresholds):
                if pt > cut:
                    counts[c][j] += 1
    bunches = [dict(bx=int(v.getBunchCrossing()), n=int(v.getPU_NumInteractions()),
                    mean=float(v.getTrueNumInteractions()),
                    ids=len(v.getPU_EventID()), z=len(v.getPU_zpositions()))
               for v in handles['pu'].product()]
    if len([b for b in bunches if b['bx'] == 0]) != 1:
        raise ValueError('Invalid in-time metadata')
    rows.append(dict(key=key, weight=weight, counts=counts,
                     fromPV=association, timing=timing, pileup=bunches))
    if (i+1) % 100 == 0:
        print('Audited matched MiniAOD events', i+1)
        import sys
        sys.stdout.flush()
if len(rows) != len(references):
    raise ValueError('Incomplete extraction')
result = dict(events=rows, input=a.input, reference=a.reference,
              cmssw=os.environ.get('CMSSW_VERSION'), thresholds_gev=thresholds,
              abs_eta_max=2.4, categories=categories,
              limitation='Reconstructed vertex association and event-level BX counts are not candidate-level truth or an in-time-only FullSim rerun.')
with gzip.open(a.output+'.partial', 'wb') as stream:
    stream.write(json.dumps(result, separators=(',', ':')).encode('utf-8'))
os.rename(a.output+'.partial', a.output)
digest = hashlib.sha256(open(a.output, 'rb').read()).hexdigest()
print(json.dumps(dict(events=len(rows), sha256=digest, output=a.output)))
