"""Monitor one existing Rucio rule, then reuse the stable input verifier.

No rule creation, deletion, EOS uploads, or production submission. Site paths,
the byte admission budget, and commands are explicit in the JSON configuration.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time
from replay_scratch_budget import durable_json


def run(config):
    out = Path(config['evidence'])
    out.mkdir(parents=True, exist_ok=True)
    request = json.loads(Path(config['request']).read_text())
    def cli(*args):
        p = subprocess.run(config['rucio_command'] + list(args),
                           capture_output=True, text=True, timeout=120)
        if p.returncode:
            raise RuntimeError(p.stderr[-2000:])
        return p.stdout
    def status(value, **fields):
        record = dict(status=value, epoch=time.time(), pid=os.getpid(),
                      rule_id=request['rule_id'], **fields)
        durable_json(out/'status.json', record)
        return record
    with (out/'monitor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not Path(config['stop_file']).exists():
            try:
                raw = cli('rule', 'show', request['rule_id'])
                info = {line.split(':', 1)[0].strip(): line.split(':', 1)[1].strip()
                        for line in raw.splitlines() if ':' in line}
                durable_json(out/'rule.json', dict(epoch=time.time(), fields=info))
                if info.get('State') != 'OK':
                    status('waiting-for-rucio', rucio_state=info.get('State'),
                           locks=info.get('Locks OK/REPLICATING/STUCK'))
                else:
                    urls = cli('replica', 'list', 'file', request['did'],
                               '--rses', request['destination'], '--pfns')
                    urls = re.findall(r'davs://[^\s|]+', urls)
                    if len(urls) != 1:
                        raise RuntimeError('Expected exactly one destination DAVS URL')
                    request['url'] = urls[0]
                    durable_json(out/'verification-request.json', request)
                    status('verifying-two-full-readbacks')
                    command = config['verify_command'] + [
                        '--request', str(out/'verification-request.json'),
                        '--destination', config['cache'], '--evidence', str(out/'verification'),
                        '--max-file-bytes', str(config['max_file_bytes'])]
                    p = subprocess.run(command, capture_output=True, text=True, timeout=3600)
                    durable_json(out/'verification-command.json', dict(returncode=p.returncode,
                        stdout=p.stdout[-2000:], stderr=p.stderr[-2000:], epoch=time.time()))
                    if p.returncode:
                        raise RuntimeError('Input verification failed: '+p.stderr[-1000:])
                    status('verified-input-ready', cache=config['cache'])
                    if config.get('ready_command'):
                        status('running-bounded-followup', cache=config['cache'])
                        try:
                            p = subprocess.run(config['ready_command'], capture_output=True,
                                               text=True, timeout=config.get('ready_timeout_seconds', 1800))
                        except subprocess.TimeoutExpired:
                            return status('followup-failed', cache=config['cache'],
                                          error='Bounded followup timed out; partial science outputs retained')
                        durable_json(out/'followup-command.json', dict(returncode=p.returncode,
                            stdout=p.stdout[-4000:], stderr=p.stderr[-4000:], epoch=time.time()))
                        # A science failure must not repeat the transfer or rerun partial outputs.
                        return status('followup-completed' if p.returncode == 0 else 'followup-failed',
                                      cache=config['cache'], returncode=p.returncode)
                    return status('verified-input-ready', cache=config['cache'])
            except Exception as error:
                status('retrying', error=repr(error), files_deleted=False)
            time.sleep(60)
        return status('stopped', files_deleted=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(run(json.loads(args.config.read_text()))))
