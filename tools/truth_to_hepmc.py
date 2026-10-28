"""Replay stable non-pileup particles from an official Delphes event example.

This preserves detector inputs, not the original shower genealogy. Includes
neutrinos. Units: GeV, mm; TreeWriter T (seconds) is converted to ct (mm).
No flavor labels or original pileup/ancestry are reconstructed.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
import uproot

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--input', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
fields = ['PID', 'Status', 'IsPU', 'Px', 'Py', 'Pz', 'E', 'Mass', 'X', 'Y', 'Z', 'T']
with uproot.open(args.input) as source:
    tree = source['Delphes']
    arrays = tree.arrays(['Particle.' + key for key in fields], library='ak')
    weights = tree['Event.Weight'].array(library='ak')
args.output.parent.mkdir(parents=True, exist_ok=True)
records = []
with open(args.output, 'w') as stream:
    stream.write('HepMC::Version 2.06.09\nHepMC::IO_GenEvent-START_EVENT_LISTING\n')
    for index, event in enumerate(arrays):
        particles = {key: np.asarray(event['Particle.' + key]) for key in fields}
        mask = (particles['Status'] == 1) & (particles['IsPU'] == 0)
        particles = {key: values[mask] for key, values in particles.items()}
        if not len(particles['PID']): raise ValueError('Empty stable event')
        for values in particles.values():
            if not np.isfinite(values).all(): raise ValueError('Nonfinite truth')
        groups = defaultdict(list)
        for i in range(len(particles['PID'])):
            position = tuple(float(particles[key][i]) for key in ['X', 'Y', 'Z']) + (float(particles['T'][i]) * 2.99792458e11,)
            groups[position].append(i)
        weight = float(weights[index][0])
        stream.write(f'E {index} 0 0 0 0 0 0 {len(groups)} 0 0 0 1 {weight:.17g}\nU GEV MM\n')
        barcode = 0
        for vertex, (position, indices) in enumerate(groups.items(), start=1):
            stream.write(f'V {-vertex} 0 ' + ' '.join(f'{v:.17g}' for v in position) + f' 0 {len(indices)} 0\n')
            for i in indices:
                barcode += 1
                momentum = ' '.join(f'{float(particles[key][i]):.17g}' for key in ['Px', 'Py', 'Pz', 'E', 'Mass'])
                stream.write(f'P {barcode} {int(particles["PID"][i])} {momentum} 1 0 0 0 0\n')
        records.append({'event': index, 'stable_nonpu_particles': len(particles['PID']), 'weight': weight, 'vertices': len(groups)})
    stream.write('HepMC::IO_GenEvent-END_EVENT_LISTING\n')
record = {'input': str(args.input), 'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
          'output': str(args.output), 'output_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
          'scope': 'ten-event official JetClass-II example, stable non-PU detector-input replay; no ancestry, no pileup, no dataset-scale result', 'units': 'GeV/mm', 'events': records}
args.output.with_suffix('.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
