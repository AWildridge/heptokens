"""Independent process watchdog; restart a crashed or unresponsive controller.

The controller's source transactions remain on workers under their own leases.
Restarting the controller resumes from receipts and defers busy transactions.
"""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import time
from run_generator_backfill import atomic


def watch(config_path):
    config=json.loads(Path(config_path).read_text());out=Path(config['output'])
    command=[config.get('python','/work/pixi/global/.pixi/envs/default/bin/python'),
             str(Path(config['tools_dir'],'supervise_generator_backfill.py')),'--config',str(config_path)]
    restarts=0
    with open(out/'recovery-watchdog.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (out/'READY.json').exists():
            began=time.time();reason=None
            with (out/'recovery-controller.log').open('a') as log:
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                while process.poll() is None and not (out/'READY.json').exists():
                    status=out/'recovery-controller-status.json'
                    state=json.loads(status.read_text()) if status.exists() else {}
                    age=time.time()-max(began,state.get('heartbeat',began))
                    atomic(out/'watchdog-status.json',dict(state='supervising',heartbeat=time.time(),controller_pid=process.pid,
                           controller_heartbeat_age_seconds=age,restarts=restarts,controller_state=state.get('state')))
                    if state.get('state')=='integrity-failure' and state.get('heartbeat',0)>=began:
                        reason='integrity-failure';break
                    if age>300:
                        reason='controller-heartbeat-stale'
                        process.send_signal(signal.SIGTERM)
                        try:process.wait(timeout=15)
                        except subprocess.TimeoutExpired:process.kill();process.wait()
                        break
                    time.sleep(10)
                if (out/'READY.json').exists():
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:process.terminate()
                    atomic(out/'watchdog-status.json',dict(state='complete',heartbeat=time.time(),restarts=restarts));return
                if reason=='integrity-failure':
                    atomic(out/'watchdog-status.json',dict(state='integrity-failure-needs-attention',heartbeat=time.time(),restarts=restarts));return
                restarts+=1
                atomic(out/'watchdog-status.json',dict(state='restarting-controller',heartbeat=time.time(),restarts=restarts,
                       reason=reason or 'controller-exited',exit_code=process.poll()))
                time.sleep(15)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    watch(parser.parse_args().config)
