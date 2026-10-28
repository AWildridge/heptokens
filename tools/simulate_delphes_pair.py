"""Run stock CMS and ATLAS cards on the SAME saved HepMC2 detector inputs.

Requires a reference JSON at INPUT.hepmc.json with events/particles/weight.
No generation, jet re-selection, inference, plotting, or cross-section scaling.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

import numpy as np
import uproot


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--delphes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Run on a Slurm compute node')
    if not args.input.is_file():
        parser.error('Missing HepMC input')
    args.input = args.input.resolve()
    args.delphes = args.delphes.resolve()
    if args.output.exists():
        parser.error('Use a fresh output directory; prior results are preserved')
    reference_path = Path(str(args.input)+'.json')
    reference = json.loads(reference_path.read_text())
    if sha256(args.input) != reference['sha256']:
        raise ValueError('HepMC input checksum does not match reference')
    args.output.mkdir(parents=True)
    summary = {'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
               'question': 'Can identical retained/generated stable inputs be simulated with both default detector cards?',
               'success_criterion': 'Equal event count, preserved stable truth and weights, finite reconstructed jets',
               'input': str(args.input), 'input_sha256': reference['sha256'],
               'scope': reference['scope'], 'events': len(reference['events']),
               'seed': args.seed, 'slurm_job_id': os.environ['SLURM_JOB_ID'],
               'execution': 'CMS and ATLAS simulations run concurrently within the Slurm step',
               'normalization': 'raw generator weights, no luminosity/cross-section normalization',
               'checkpoint': None, 'detectors': {}, 'rtol': 1e-5, 'atol': 1e-7}
    record = args.output / 'validation.json'
    shutil.copy2(reference_path, args.output / 'input-reference.json')
    env = os.environ.copy()
    env['ROOT_INCLUDE_PATH'] = f'{args.delphes}:{args.delphes}/external:' + env.get('ROOT_INCLUDE_PATH', '')
    try:
        commands = {}
        for detector in ['CMS', 'ATLAS']:
            card = args.delphes / f'cards/delphes_card_{detector}.tcl'
            shutil.copy2(card, args.output / card.name)
            wrapper = args.output / f'{detector}-seed{args.seed}.tcl'
            wrapper.write_text(f'source {{{card.name}}}\nset RandomSeed {args.seed}\n')
            output = args.output / f'{detector}.root'
            command = [str(args.delphes / 'DelphesHepMC2'), str(wrapper.resolve()), str(output.resolve()), str(args.input)]
            entry = {'command': command, 'card_sha256': sha256(card), 'status': 'running'}
            summary['detectors'][detector] = entry
            record.write_text(json.dumps(summary, indent=2))
            commands[detector] = command

        def run_detector(detector):
            with open(args.output / f'{detector}.log', 'w') as log:
                subprocess.run(commands[detector], env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)

        # Two single-process CPU simulations fit within the requested 4 CPUs.
        # Inputs are immutable and outputs/logs are separate for each detector.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {detector: pool.submit(run_detector, detector)
                       for detector in ['CMS', 'ATLAS']}
            for detector, future in futures.items():
                future.result()
                summary['detectors'][detector]['status'] = 'simulated'
                record.write_text(json.dumps(summary, indent=2))

        for detector in ['CMS', 'ATLAS']:
            output = args.output / f'{detector}.root'
            entry = summary['detectors'][detector]
            with uproot.open(output) as source:
                tree = source['Delphes']
                if tree.num_entries != len(reference['events']):
                    raise ValueError('Event count mismatch')
                keys = ['PID','Px','Py','Pz','E','Mass','X','Y','Z','T']
                particles = tree.arrays(['Particle.'+key for key in keys], library='ak')
                weights = tree['Event.Weight'].array(library='ak')
                counts = []
                for expected, event, weight in zip(reference['events'], particles, weights):
                    actual = np.column_stack([np.asarray(event['Particle.'+key]) for key in keys]).astype(np.float64)
                    # Delphes TreeWriter time is seconds; reference is ct in mm.
                    actual[:, -1] *= 2.99792458e11
                    wanted = np.asarray(expected['particles'])
                    if actual.shape != wanted.shape:
                        raise ValueError('Stable particle count mismatch')
                    np.testing.assert_array_equal(actual[:,0], wanted[:,0])
                    np.testing.assert_allclose(actual[:,1:], wanted[:,1:], rtol=1e-5, atol=1e-7)
                    if not np.isfinite(float(weight[0])) or not np.isfinite(expected['weight']):
                        raise ValueError('Nonfinite generator weight')
                    np.testing.assert_allclose(float(weight[0]), expected['weight'], rtol=1e-6, atol=1e-9)
                    counts.append(len(actual))
                jets = tree.arrays(['Jet.PT','Jet.Eta','Jet.Phi','Jet.Mass'], library='ak')
                flat = np.concatenate([np.asarray(event) for event in jets['Jet.PT']])
                for key in jets.fields:
                    for event in jets[key]:
                        if not np.isfinite(np.asarray(event)).all():
                            raise ValueError('Nonfinite reconstructed jet')
                if not len(flat) or not (flat > 0).all():
                    raise ValueError('No valid reconstructed jets')
                entry.update(status='passed', output=str(output), output_sha256=sha256(output),
                             events=tree.num_entries, stable_particle_counts=counts, jets=len(flat),
                             jet_pt_range_GeV=[float(flat.min()),float(flat.max())],
                             truth_and_weights='passed')
            record.write_text(json.dumps(summary, indent=2))
            print(detector, 'passed', entry['events'], 'events', entry['jets'], 'jets', flush=True)
        summary['status'] = 'passed'
    except Exception as error:
        summary.update(status='failed', error=repr(error))
        raise
    finally:
        record.write_text(json.dumps(summary, indent=2))
    print(record, flush=True)


if __name__ == '__main__':
    main()
