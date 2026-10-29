"""Execute bounded matched AOD/MiniAOD diagnostics using explicit site commands."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from replay_scratch_budget import durable_json


def run(config):
    out = Path(config['output'])
    def execute(name, args):
        started = time.time()
        with (out/(name+'-command-output.txt')).open('w') as stream:
            p = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT,
                               timeout=config.get('command_timeout_seconds', 900))
        durable_json(out/(name+'-command.json'), dict(command=args, returncode=p.returncode,
                     elapsed_seconds=time.time()-started))
        if p.returncode:
            raise RuntimeError(name+' failed; original inputs and diagnostics retained')
    def audit(name, source, tier, command):
        target = out/(name+'.json')
        if not target.exists():
            execute(name, command+['python', config['audit_tool'], '--input', source,
                '--tier', tier, '--reference', config['reference'], '--limit', str(config['events']),
                '--output', str(target)])
        data = json.loads(target.read_text())
        if data['matched'] != config['events'] or data['missing']:
            raise ValueError('Audit does not cover every requested event')
        return {tuple(v['key']): v for v in data['events']}
    original = audit('original-miniaod', config['miniaod'], 'miniaod', config['analysis_command'])
    aod = audit('original-aod', config['aod'], 'aod', config['analysis_command'])
    fields = ['pf_count', 'pf_charged', 'pf_neutral', 'pf_sumpt',
              'gen_count', 'gen_charged', 'gen_neutral', 'gen_sumpt']
    def compare(left, right):
        if set(left) != set(right):
            raise ValueError('Unmatched event IDs')
        return {f: dict(left_mean=sum(v[f] for v in left.values())/len(left),
                        right_mean=sum(v[f] for v in right.values())/len(right),
                        changed_events=sum(left[k][f] != right[k][f] for k in left),
                        mean_delta=sum(right[k][f]-left[k][f] for k in left)/len(left))
                for f in fields}
    durable_json(out/'aod-vs-miniaod.json', dict(events=len(original),
        left='original AOD', right='original MiniAOD', values=compare(aod, original),
        caveat='Small matched QCD source-84 diagnostic; no campaign-wide normalization or fidelity claim'))
    execute('pat-replay', config['pat_command']+['cmsRun', config['pat_config']])
    regenerated = audit('regenerated-miniaod', config['regenerated'], 'miniaod', config['analysis_command'])
    durable_json(out/'pat-replay-comparison.json', dict(events=len(original),
        left='original MiniAOD', right='regenerated MiniAOD', values=compare(original, regenerated),
        caveat='UL16 recipe replay; tracked module equivalence must be checked before claiming exact production closure'))
    durable_json(out/'aod-pilot-status.json', dict(status='completed', events=len(original), epoch=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    try:
        run(cfg)
    except Exception as error:
        durable_json(Path(cfg['output'])/'aod-pilot-status.json',
                     dict(status='failed', error=repr(error), epoch=time.time(), files_deleted=False))
        raise
