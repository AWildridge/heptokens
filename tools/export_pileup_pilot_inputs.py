"""CMSSW Python 2: extract GEN stable particles or matched MiniAOD PU metadata.

GEN vertices use cm->mm and unavailable ct=0. No reconstruction or overlay.
Outputs are streamed gzip JSONL and finalized atomically with a summary.
"""
from __future__ import print_function
import argparse, collections, gzip, hashlib, json, math, os, struct
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
from DataFormats.FWLite import Events, Handle

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--mode', choices=['gen', 'metadata'], required=True)
p.add_argument('--input', required=True)
p.add_argument('--output', required=True)
p.add_argument('--limit', type=int)
p.add_argument('--reference')
p.add_argument('--native-pileup', action='store_true', help='GEN only: stream canonical modal-origin native records instead of JSON')
p.add_argument('--audit-gen-weights', action='store_true')
p.add_argument('--gzip-level', type=int, default=9)
a = p.parse_args()
if os.path.exists(a.output):
    raise RuntimeError('Existing finalized output must be checked, not overwritten')
if a.native_pileup and a.mode != 'gen':
    raise ValueError('Native records require GEN mode')
if not 0 <= a.gzip_level <= 9:
    raise ValueError('Invalid gzip level')
reference = json.load(open(a.reference))['events'] if a.reference else None
events = Events([a.input])
handle = Handle('std::vector<reco::GenParticle>' if a.mode == 'gen' else 'std::vector<PileupSummaryInfo>')
label = 'genParticles' if a.mode == 'gen' else 'slimmedAddPileupInfo'
count = particles = 0
seen = set()
minimum_z = maximum_z = 0.
weight_handle = Handle('GenEventInfoProduct') if a.audit_gen_weights else None
weights = collections.Counter()
ids, offsets, position = [], [], 0
stream_opener = open if a.native_pileup else lambda path, mode: gzip.open(path, mode, compresslevel=a.gzip_level)
with stream_opener(a.output + '.partial', 'wb') as stream:
    for i, event in enumerate(events):
        if a.limit is not None and i >= a.limit:
            break
        aux = event.eventAuxiliary()
        key = [int(aux.run()), int(aux.luminosityBlock()), int(aux.event())]
        if tuple(key) in seen:
            raise ValueError('Duplicate event identity')
        seen.add(tuple(key))
        ids.append(key)
        if weight_handle is not None:
            event.getByLabel('generator', weight_handle)
            if not weight_handle.isValid():
                raise ValueError('Missing GEN weight')
            weight = float(weight_handle.product().weight())
            if weight != 1.:
                raise ValueError('Non-unit GEN weight')
            weights[weight] += 1
        if reference and key != [reference[i][x] for x in ['run', 'lumi', 'event']]:
            raise ValueError('Hard event identity differs from frozen validation')
        event.getByLabel(label, handle)
        if not handle.isValid():
            raise RuntimeError('Missing ' + label)
        row = {'key': key}
        if a.mode == 'gen':
            values = []
            row['first_gen_vertices_mm'] = [[int(v.pdgId()), int(v.status()),
                                             10.*v.vx(), 10.*v.vy(), 10.*v.vz()]
                                            for v in list(handle.product())[:10]]
            for v in handle.product():
                if v.status() != 1:
                    continue
                item = [int(v.pdgId()), float(v.px()), float(v.py()), float(v.pz()),
                        float(v.energy()), float(v.mass()), 10.*v.vx(), 10.*v.vy(), 10.*v.vz(), 0.]
                if any(math.isnan(x) or math.isinf(x) for x in item):
                    raise ValueError('Nonfinite GEN particle')
                values.append(item)
            if not values:
                raise ValueError('Empty GEN event')
            row['particles'] = values
            particles += len(values)
            minimum_z = min(minimum_z, min(v[8] for v in values))
            maximum_z = max(maximum_z, max(v[8] for v in values))
        else:
            intime = [v for v in handle.product() if v.getBunchCrossing() == 0]
            if len(intime) != 1:
                raise ValueError('Expected exactly one in-time PU record')
            v = intime[0]
            row.update(n=int(v.getPU_NumInteractions()), mean=float(v.getTrueNumInteractions()),
                       z_cm=[float(z) for z in v.getPU_zpositions()],
                       ids=[[int(x.run()), int(x.luminosityBlock()), int(x.event())] for x in v.getPU_EventID()])
            if len(row['z_cm']) != row['n'] or len(row['ids']) != row['n']:
                raise ValueError('Incomplete in-time interaction metadata')
        if a.native_pileup:
            # Match the Python3 canonical converter's first-seen tie breaking.
            origins = collections.OrderedDict()
            for v in values:
                xyz = tuple(v[6:9])
                origins[xyz] = origins.get(xyz, 0) + 1
            origin = max(origins, key=origins.get)
            offsets.append(position)
            stream.write(struct.pack('>i', len(values)))
            for v in values:
                args = [int(v[0])] + [v[6+j] - origin[j] for j in range(3)] + [v[9]] + v[1:5]
                stream.write(struct.pack('>i8f', *args))
            position += 4 + 36 * len(values)
        else:
            stream.write((json.dumps(row, separators=(',', ':')) + '\n').encode('utf-8'))
        count += 1
        if count % 1000 == 0:
            print('Extracted', count, a.mode, 'events')
            with open(a.output + '.progress', 'w') as progress:
                json.dump({'events': count}, progress)
    if a.native_pileup:
        for value in offsets + [len(offsets)]:
            stream.write(struct.pack('>q', value))
if not count or (a.limit is not None and count != min(a.limit, int(events.size()))):
    raise ValueError('Incomplete input extraction')
if reference and count != len(reference):
    raise ValueError('Reference count differs')
os.rename(a.output + '.partial', a.output)
digest = hashlib.sha256()
with open(a.output, 'rb') as stream:
    for block in iter(lambda: stream.read(1048576), b''):
        digest.update(block)
with open(a.output + '.summary.json', 'w') as stream:
    json.dump({'mode': a.mode, 'input': a.input, 'events': count, 'particles': particles,
               'sha256': digest.hexdigest(), 'units': 'GeV/mm; unavailable ct=0',
               'particle_fields': ['pid','px','py','pz','energy','mass','x','y','z','ct'],
               'z_range_mm': [minimum_z, maximum_z], 'input_events': int(events.size()),
               'native_pileup': a.native_pileup, 'event_ids': ids if a.native_pileup else None,
               'gen_weight_counts': dict(weights), 'all_gen_weights_one': a.audit_gen_weights and sum(weights.values()) == count,
               'origin_convention': 'Subtract modal stable-particle xyz, first-seen tie; retain relative decay vertices, ct=0' if a.native_pileup else None}, stream, indent=2)
print('Completed', count, a.mode, 'events')
