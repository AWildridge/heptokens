"""Check detector input replay and save summary; no physics inference or plots."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import uproot

p = argparse.ArgumentParser()
p.add_argument('--run', type=Path, required=True)
args = p.parse_args()
record = json.loads((args.run / 'stable-hard-events.json').read_text())
source = uproot.open(record['input'])['Delphes']
keys = ['PID', 'Status', 'IsPU', 'Px', 'Py', 'Pz', 'E', 'X', 'Y', 'Z', 'T']
truth = source.arrays(['Particle.' + key for key in keys], library='ak')
summary = {'scope': record['scope'], 'seed': 42, 'detectors': {},
           'question': 'Can retained example stable hard-scatter detector inputs be replayed with stock cards?',
           'success_criterion': 'All 10 events read; stable particle IDs, four-momenta and vertices preserved; finite reconstructed jets',
           'limitations': ['Original shower ancestry omitted', 'Pileup excluded and not regenerated', 'No original JetClass-II labels assigned', 'Default jet radii differ from original sample', 'Ten-event software validation, not validated physics']}
for detector in ['ATLAS', 'CMS']:
    with uproot.open(args.run / f'{detector}.root') as output:
        tree = output['Delphes']
        assert tree.num_entries == len(truth) == 10
        replay = tree.arrays(['Particle.' + key for key in keys], library='ak')
        counts = []
        for original, new in zip(truth, replay):
            select = (np.asarray(original['Particle.Status']) == 1) & (np.asarray(original['Particle.IsPU']) == 0)
            counts.append(int(select.sum()))
            # Export order is grouped by production vertex; compare canonical rows.
            def rows(event, selection=None):
                columns = [np.asarray(event['Particle.' + key]) for key in ['PID', 'Px', 'Py', 'Pz', 'E', 'X', 'Y', 'Z', 'T']]
                array = np.column_stack(columns)
                if selection is not None: array = array[selection]
                return array[np.lexsort(tuple(array[:, i] for i in range(array.shape[1]-1, -1, -1)))]
            np.testing.assert_allclose(rows(original, select), rows(new), rtol=1e-6, atol=1e-7)
        pt = tree['Jet.PT'].array(library='ak')
        flat = np.concatenate([np.asarray(event) for event in pt])
        assert len(flat) > 0 and np.isfinite(flat).all() and (flat > 0).all()
        weights = tree['Event.Weight'].array(library='ak')
        np.testing.assert_allclose([float(event[0]) for event in weights], [event['weight'] for event in record['events']], rtol=1e-6)
        card = args.run / f'delphes_card_{detector}.tcl'
        summary['detectors'][detector] = {'events': tree.num_entries, 'stable_particles': counts, 'jets': len(flat), 'pt_range_GeV': [float(flat.min()), float(flat.max())], 'card_sha256': hashlib.sha256(card.read_bytes()).hexdigest(), 'input_truth_check': 'passed', 'event_weights_check': 'passed'}
print(json.dumps(summary, indent=2))
