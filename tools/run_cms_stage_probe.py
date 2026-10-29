"""Run a small, explicitly configured CMS stage chain with bounded CPU runtimes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from replay_scratch_budget import durable_json


def run(config):
    out = Path(config['output'])
    for stage in config['stages']:
        name = stage['name']
        target = Path(stage['target'])
        if target.exists():
            raise RuntimeError('Retaining existing stage output; inspect before retry: '+str(target))
        durable_json(out/'stage-probe-status.json', dict(status='running', stage=name,
                     epoch=time.time(), paired_to_original=False))
        started = time.time()
        with (out/(name+'-probe-command-output.txt')).open('w') as stream:
            try:
                p = subprocess.run(stage['command'], stdout=stream, stderr=subprocess.STDOUT,
                                   timeout=stage['timeout_seconds'])
                result = dict(returncode=p.returncode)
            except subprocess.TimeoutExpired:
                result = dict(error='Stage exceeded its explicit feasibility time budget')
        result.update(elapsed_seconds=time.time()-started, command=stage['command'],
                      events_requested=stage['events'], paired_to_original=False)
        durable_json(out/(name+'-probe-result.json'), result)
        if result.get('returncode') != 0 or not target.is_file() or not target.stat().st_size:
            durable_json(out/'stage-probe-status.json', dict(status='failed', stage=name,
                         result=result, epoch=time.time(), outputs_retained=True))
            return 1
        h = hashlib.sha256()
        with target.open('rb') as f:
            for block in iter(lambda: f.read(8*1024**2), b''):
                h.update(block)
        durable_json(out/(name+'-probe-output.json'), dict(path=str(target),
                     bytes=target.stat().st_size, sha256=h.hexdigest(),
                     note='Local checksum; not a durable remote-transfer proof'))
    durable_json(out/'stage-probe-status.json', dict(status='completed', epoch=time.time(),
                 paired_to_original=False, note='Only GEN/SIM feasibility, not a full-chain fidelity result'))
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    try:
        code = run(config)
    except Exception as error:
        durable_json(Path(config['output'])/'stage-probe-status.json',
                     dict(status='failed', error=repr(error), epoch=time.time(), outputs_retained=True))
        raise
    raise SystemExit(code)
