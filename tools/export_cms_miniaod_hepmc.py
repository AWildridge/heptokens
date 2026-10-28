"""Export stable MiniAOD truth for paired Delphes replay using CMSSW FWLite.

Python 2.7 compatible; run in CMSSW_10_6_30, not the CMS Pixi Python.
Packed particles are rapidity-limited/quantized, not the full original HepMC.
Vertex coordinates cm -> mm. Production time is not available; ct=0.
"""
from __future__ import print_function
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
from DataFormats.FWLite import Events, Handle

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--input', required=True)
p.add_argument('--output', required=True)
p.add_argument('--events', type=int, default=100)
args = p.parse_args()
if args.events <= 0: p.error('--events must be positive')
if os.path.exists(args.output): raise RuntimeError('Output exists')
events = Events([args.input])
packed = Handle('std::vector<pat::PackedGenParticle>')
generator = Handle('GenEventInfoProduct')
pruned = Handle('std::vector<reco::GenParticle>')
record = {'input': args.input, 'requested_events': args.events,
          'scope': 'original CMS QCD MiniAOD stable hard-scatter truth replay; quantized and rapidity-limited; no original pileup replay',
          'units': 'GEV MM', 'vertex_time': 'not retained by PackedGenParticle; set ct=0',
          'event_selection': 'first requested events, no physics cuts; exploratory validation',
          'events': []}
with open(args.output, 'w') as output:
    output.write('HepMC::Version 2.06.09\nHepMC::IO_GenEvent-START_EVENT_LISTING\n')
    for index, event in enumerate(events):
        if index >= args.events: break
        event.getByLabel('packedGenParticles', packed)
        event.getByLabel('generator', generator)
        event.getByLabel('prunedGenParticles', pruned)
        if not packed.isValid() or not generator.isValid():
            raise RuntimeError('Missing generator collections')
        groups = defaultdict(list)
        reference = []
        for particle in packed.product():
            if particle.status() != 1: raise RuntimeError('Unexpected nonstable packed particle')
            values = [int(particle.pdgId()), particle.px(), particle.py(), particle.pz(), particle.energy(), particle.mass(), 10*particle.vx(), 10*particle.vy(), 10*particle.vz(), 0.0]
            if not all(not (math.isnan(v) or math.isinf(v)) for v in values):
                raise RuntimeError('Nonfinite generator values')
            reference.append(values)
            groups[tuple(values[6:])].append(values)
        if not reference: raise RuntimeError('No stable particles')
        weight = generator.product().weight()
        output.write('E %d 0 0 0 0 0 0 %d 0 0 0 1 %.17g\nU GEV MM\n' % (index,len(groups),weight))
        barcode = 0
        ordered = []
        for vertex, (position, particles) in enumerate(groups.items(), start=1):
            output.write('V %d 0 %s 0 %d 0\n' % (-vertex,' '.join('%.17g'%v for v in position),len(particles)))
            for values in particles:
                barcode += 1
                ordered.append(values)
                output.write('P %d %d %s 1 0 0 0 0\n' % (barcode,values[0],' '.join('%.17g'%v for v in values[1:6])))
        auxiliary = event.eventAuxiliary()
        record['events'].append({'export_event':index,'run':int(auxiliary.run()),'lumi':int(auxiliary.luminosityBlock()),'event':int(auxiliary.event()),'weight':weight,'particles':ordered,'pruned_particles':len(pruned.product()) if pruned.isValid() else None})
        if (index+1)%10==0: print('Exported %d events' % (index+1))
    output.write('HepMC::IO_GenEvent-END_EVENT_LISTING\n')
if len(record['events']) != args.events: raise RuntimeError('Fewer events than requested')
record['sha256'] = hashlib.sha256(open(args.output,'rb').read()).hexdigest()
with open(args.output+'.json','w') as stream: json.dump(record, stream, indent=2)
print('Export complete:',args.output)
