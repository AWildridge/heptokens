"""Python 2 FWLite: export every original event in bounded HepMC chunks."""
from __future__ import print_function
import argparse, hashlib, json, math, os
from collections import defaultdict
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
from DataFormats.FWLite import Events, Handle

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--input', required=True); p.add_argument('--output', required=True)
p.add_argument('--chunk-events', type=int, default=1000)
p.add_argument('--limit', type=int, default=None, help='Validation only; production leaves unset')
p.add_argument('--max-output-bytes',type=int,default=None,help='Optional hard limit for uncompressed export files')
a = p.parse_args()
if a.chunk_events <= 0 or (a.limit is not None and a.limit <= 0): p.error('Invalid count')
if a.max_output_bytes is not None and a.max_output_bytes<=0:p.error('Invalid export storage limit')
if os.path.exists(a.output): raise RuntimeError('Use fresh export directory')
os.makedirs(a.output)
events = Events([a.input]); expected = int(events.size())
if a.limit is not None: expected = min(expected, a.limit)
packed = Handle('std::vector<pat::PackedGenParticle>')
gen = Handle('GenEventInfoProduct'); lhe = Handle('LHEEventProduct'); chunks = []; refs = []; output = None
output_bytes=0
class BudgetFile(object):
    def __init__(self,stream):self.stream=stream
    def write(self,data):
        global output_bytes
        if output_bytes+len(data)>a.max_output_bytes:raise RuntimeError('Export storage budget reached; original source retained')
        result=self.stream.write(data);output_bytes+=len(data);return result
    def close(self):self.stream.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.stream.close()
def open_output(path):
    stream=open(path,'w')
    return stream if a.max_output_bytes is None else BudgetFile(stream)
def finish():
    output.write('HepMC::IO_GenEvent-END_EVENT_LISTING\n'); output.close()
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): digest.update(block)
    record = {'input': a.input, 'sha256': digest.hexdigest(), 'events': refs,
              'scope': 'original packed stable hard-scatter truth; quantized/rapidity-limited; no pileup or ancestry; vertex cm to mm, ct=0',
              'units': 'GEV MM', 'selection': 'all source events, no physics cuts' if a.limit is None else 'validation prefix'}
    with open_output(path+'.json') as stream: json.dump(record, stream)
    chunks.append({'file': os.path.basename(path), 'events': len(refs), 'sha256': record['sha256']})
    print('Completed chunk', len(chunks), 'events', len(refs))
count = 0
for i, event in enumerate(events):
    if i >= expected: break
    if i % a.chunk_events == 0:
        refs = []; path = os.path.join(a.output, 'chunk-%06d.hepmc' % (i//a.chunk_events))
        output = open_output(path); output.write('HepMC::Version 2.06.09\nHepMC::IO_GenEvent-START_EVENT_LISTING\n')
    event.getByLabel('packedGenParticles', packed); event.getByLabel('generator', gen)
    if not packed.isValid() or not gen.isValid(): raise RuntimeError('Missing truth')
    groups = defaultdict(list)
    for particle in packed.product():
        if particle.status() != 1: raise RuntimeError('Nonstable packed particle')
        v = [int(particle.pdgId()), particle.px(), particle.py(), particle.pz(), particle.energy(), particle.mass(), 10*particle.vx(), 10*particle.vy(), 10*particle.vz(), 0.0]
        if not all(not (math.isnan(x) or math.isinf(x)) for x in v): raise RuntimeError('Nonfinite truth')
        groups[tuple(v[6:])].append(v)
    if not groups: raise RuntimeError('Empty truth')
    weight = float(gen.product().weight()); weights = [float(x) for x in gen.product().weights()]
    if not all(not (math.isnan(x) or math.isinf(x)) for x in [weight]+weights): raise RuntimeError('Nonfinite weights')
    output.write('E %d 0 0 0 0 0 0 %d 0 0 0 1 %.17g\nU GEV MM\n' % (i, len(groups), weight))
    ordered = []; barcode = 0
    for vertex, (position, particles) in enumerate(groups.items(), 1):
        output.write('V %d 0 %s 0 %d 0\n' % (-vertex, ' '.join('%.17g'%x for x in position), len(particles)))
        for values in particles:
            barcode += 1; ordered.append(values)
            output.write('P %d %d %s 1 0 0 0 0\n' % (barcode, values[0], ' '.join('%.17g'%x for x in values[1:6])))
    aux = event.eventAuxiliary()
    event.getByLabel('externalLHEProducer', lhe)
    lhe_weights = [{'id': str(w.id), 'value': float(w.wgt)} for w in lhe.product().weights()] if lhe.isValid() else None
    if lhe_weights is not None and not all(not (math.isnan(w['value']) or math.isinf(w['value'])) for w in lhe_weights): raise RuntimeError('Nonfinite LHE weight')
    refs.append({'export_event': i, 'run': int(aux.run()), 'lumi': int(aux.luminosityBlock()), 'event': int(aux.event()), 'weight': weight, 'generator_weights': weights, 'lhe_weights': lhe_weights, 'particles': ordered})
    count += 1
    if len(refs) == a.chunk_events or count == expected: finish()
if count != expected or not count: raise RuntimeError('Event count mismatch')
with open(os.path.join(a.output, 'index.json'), 'w') as stream:
    json.dump({'source_events': int(events.size()), 'exported_events': count, 'validation_limit': a.limit, 'chunks': chunks}, stream, indent=2)
print('Exported all', count, 'events')
