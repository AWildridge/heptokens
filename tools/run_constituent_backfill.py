"""Restartable Dask controller and independent wake-up watchdog for truth links."""
import argparse
from collections import deque
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback
import uuid

from constituent_backfill_worker import atomic, digest, proxy_lifetime, verified_group


def dispatch(config, source, pilot=False):
    # Keep the Dask process in the maintained global environment and isolate ROOT.
    import json, os, signal, subprocess, time, uuid
    from pathlib import Path
    token = '%05d-' % source['task'] + uuid.uuid4().hex
    directory = Path(config['output']) / 'dispatch'
    directory.mkdir(exist_ok=True)
    request = directory / (token + '.json')
    result = directory / (token + '-result.json')
    request.write_text(json.dumps(dict(config=config, source=source, pilot=pilot, result=str(result))))
    env = dict(os.environ, PYTHONPATH=config['analysis_pythonpath'], OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    with (directory / (token + '.log')).open('w') as log:
        process = subprocess.Popen([config['analysis_python'],
                                    str(Path(config['tools_dir']) / 'constituent_backfill_worker.py'), str(request)],
                                   env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=config.get('source_timeout_seconds', 14400))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return dict(status='failed', task=source['task'], retryable=True, error='Source subprocess timeout')
    if not result.exists():
        return dict(status='failed', task=source['task'], retryable=True,
                    error='Source subprocess exited without result: ' + str(code), log=str(directory / (token + '.log')))
    return json.loads(result.read_text())


def one_source_plugin():
    from distributed import WorkerPlugin
    class OneSource(WorkerPlugin):
        def setup(self, worker):
            async def publish_when_registered():
                import asyncio
                for attempt in range(600):
                    status = worker.status.name
                    if status in ['closing', 'closed', 'failed']:
                        return
                    if status == 'running':
                        try:
                            await worker.set_resources(constituent_source=1)
                            return
                        except (KeyError, OSError, TimeoutError):
                            pass
                    await asyncio.sleep(.5)
                raise RuntimeError('Worker never published constituent-source resource')
            # A plugin may run before scheduler registration finishes. Blocking
            # setup on set_resources then fails new workers during scale-out.
            worker.loop.add_callback(publish_when_registered)
    return OneSource()


def resource_wakeup_plugin():
    from distributed import SchedulerPlugin
    class ResourceWakeup(SchedulerPlugin):
        def start(self, scheduler):
            from tornado.ioloop import PeriodicCallback
            def wake():
                recommendations = {}
                for task in list(scheduler.unrunnable):
                    if (task.resource_restrictions or {}).get('constituent_source'):
                        if scheduler.valid_workers(task):
                            recommendations[task.key] = 'processing'
                if recommendations:
                    scheduler.transitions(recommendations, stimulus_id='constituent-resources-ready')
            self.timer = PeriodicCallback(wake, 2000)
            self.timer.start()
        async def close(self):
            self.timer.stop()
    return ResourceWakeup()


def load_sources(config):
    inventory = Path(config['inventory'])
    if digest(inventory) != config['inventory_sha256']:
        raise ValueError('Frozen EOS inventory changed')
    data = json.loads(inventory.read_text())
    manifest = json.loads(Path(config['manifest']).read_text())
    expected = {s['task']: s for s in manifest['files']}
    sources = {s['task']: s for s in data['sources']}
    if len(sources) != len(data['sources']):
        raise ValueError('Duplicate input source')
    for task, source in sources.items():
        if source['source'] != expected[task] or source['events'] % 1000:
            raise ValueError('Input source identity/event chunking differs')
    missing = sorted(set(expected) - set(sources))
    return sources, missing, manifest


def completed_source(config, source):
    out = Path(config['output']) / 'sources' / ('%05d' % source['task'])
    path = out / 'complete.json'
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if (result['status'] != 'complete' or result['config_identity'] != config['identity']
            or result['task'] != source['task'] or result['events'] != source['events']):
        raise ValueError('Source receipt identity differs')
    count = source['events'] // 1000
    groups = []
    for begin in range(0, count, config['chunks_per_group']):
        chunks = ['chunk-%06d' % i for i in range(begin, min(begin + config['chunks_per_group'], count))]
        group = out / 'groups' / ('group-%06d' % (begin // config['chunks_per_group'])) / 'complete.json'
        value = verified_group(group, config, source, chunks)
        if value is None:
            raise ValueError('Source missing committed group')
        groups.append(value)
    if sum(g['events'] for g in groups) != source['events']:
        raise ValueError('Source group event conservation failed')
    return result


def connect(config):
    from dask_gateway import Gateway
    from distributed import Client
    gateway = Gateway()
    reports = gateway.list_clusters()
    names = set(reports) if isinstance(reports, dict) else {r.name for r in reports}
    if config.get('cluster') not in names:
        if names:
            raise RuntimeError('Another Dask cluster exists; refusing to replace it')
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   CUDA_VISIBLE_DEVICES='', X509_USER_PROXY=config['proxy'], PYTHONPATH=config['tools_dir'])
        cluster = gateway.new_cluster(pixi_project='/work/pixi/global', worker_cores=1, worker_memory=4, env=env)
        config['cluster'] = cluster.name
        atomic(config['config_path'], config)
    else:
        cluster = gateway.connect(config['cluster'], shutdown_on_close=False)
    client = Client(cluster)
    client.register_plugin(one_source_plugin(), name='constituent-one-source')
    client.register_plugin(resource_wakeup_plugin(), name='constituent-resource-wakeup')
    return cluster, client


def pilot(config):
    from distributed import as_completed
    sources, missing, manifest = load_sources(config)
    out = Path(config['output'])
    cluster, client = connect(config)
    try:
        futures = {client.submit(dispatch, config, sources[task], True, pure=False,
                                 resources={'constituent_source': 1}): task for task in config['pilot_tasks']}
        results = []
        for future in as_completed(futures):
            value = future.result()
            results.append(value)
            atomic(out / 'pilot-status.json', dict(state='running', updated=time.time(), results=results))
            print(json.dumps(dict(task=futures[future], status=value['status'])), flush=True)
        if any(r['status'] != 'pilot-passed-local-only' for r in results):
            raise ValueError('Pilot failed; full production remains gated')
        atomic(out / 'PILOT-LOCAL-PASSED.json', dict(status='all-first-chunk-parity-and-links-passed',
               identity=config['identity'], events=sum(r['events'] for r in results),
               results=results, limitations='Local pilot only, no new EOS durability proof or full-source completion.',
               missing_Purdue_source_tasks=missing, timestamp=time.time()))
    finally:
        client.close()


def remote_preflight(config):
    from constituent_backfill_worker import publish
    out = Path(config['output'])
    local = json.loads((out / 'PILOT-LOCAL-PASSED.json').read_text())
    if local['identity'] != config['identity']:
        raise ValueError('Pilot used a different frozen runtime')
    # Every bin/layout pilot payload receives the same canonical publication checks.
    proofs = []
    for result in local['results']:
        directory = out / 'pilot-sources' / ('%05d' % result['source_task'])
        proof = directory / 'eos-verified.json'
        if proof.exists():
            remote = json.loads(proof.read_text())
            if (remote['status'] != 'two-stable-full-readbacks-passed'
                    or remote['first']['sha256'] != result['sha256']
                    or remote['second']['sha256'] != result['sha256']):
                raise ValueError('Pilot EOS proof differs')
        else:
            remote = publish(config, Path(result['archive']), directory, result['source_task'], 'pilot-first-chunk')
        proofs.append(dict(task=result['source_task'], remote=remote, sha256=result['sha256']))
        atomic(out / 'controller-status.json', dict(state='checking-EOS-preflight', heartbeat=time.time(),
               verified_pilot_payloads=len(proofs), total_pilot_payloads=len(local['results'])))
    atomic(out / 'PREFLIGHT-PASSED.json', dict(status='all-local-checks-and-two-EOS-readbacks-passed',
           identity=config['identity'], proofs=proofs, timestamp=time.time()))


def controller(config):
    from distributed import as_completed
    sources, missing, manifest = load_sources(config)
    out = Path(config['output'])
    ready = {task: r for task, s in sources.items() if (r := completed_source(config, s))}
    groups = {}
    # Interleave bins so a problematic prefix cannot starve the other samples.
    bins = {}
    for task, s in sources.items():
        if task not in ready:
            bins.setdefault(str(s['source']['record']), deque()).append(task)
    queue = deque()
    while any(bins.values()):
        for name in sorted(bins):
            if bins[name]:
                queue.append(bins[name].popleft())
    retries_path = out / 'retries.json'
    retries = json.loads(retries_path.read_text()) if retries_path.exists() else {}
    quarantined = {int(k) for k, v in retries.items() if not v.get('retryable', True) and int(k) not in ready}
    cluster, client = connect(config)
    running, completed = {}, as_completed()
    last_heartbeat, last_scan, last_progress = 0, 0, time.time()
    history = []
    previous_durable_events = 0
    dispatch_pause_until = 0
    recent_failures = deque()
    desired = None
    began = time.time()
    try:
        while len(ready) < len(sources):
            now = time.time()
            lifetime = proxy_lifetime(config)
            minimum = config.get('minimum_proxy_seconds', 3600)
            can_dispatch = lifetime >= minimum and now >= dispatch_pause_until
            if not (out / 'PREFLIGHT-PASSED.json').exists():
                # Overnight start requires enough credential life to cover the requested absence.
                if lifetime >= config.get('initial_proxy_seconds', 18 * 3600):
                    atomic(out / 'controller-status.json', dict(state='checking-EOS-preflight', heartbeat=now))
                    remote_preflight(config)
                else:
                    can_dispatch = False
            if can_dispatch and (out / 'PREFLIGHT-PASSED.json').exists():
                target = config.get('workers', 1000)
            else:
                # Do not scale down workers while validated, unpublished payloads
                # remain in their scratch. No bulk workers exist before preflight.
                target = desired if desired == config.get('workers', 1000) else config.get('pilot_workers', 4)
            if desired != target:
                cluster.scale(target)
                desired = target
            for _ in range(len(queue)):
                if not can_dispatch or len(running) >= config.get('inflight', 1000):
                    break
                task = queue.popleft()
                if task in ready or task in quarantined:
                    continue
                if now < retries.get(str(task), {}).get('next_retry', 0):
                    queue.append(task)
                    continue
                value = completed_source(config, sources[task])
                if value:
                    ready[task] = value
                    continue
                future = client.submit(dispatch, config, sources[task], False, pure=False,
                                       retries=0, resources={'constituent_source': 1})
                running[future] = task
                completed.add(future)
            for future in completed.next_batch(block=False) if running else []:
                task = running.pop(future)
                try:
                    value = future.result()
                except Exception:
                    value = dict(status='failed', retryable=True, error=traceback.format_exc())
                if value['status'] == 'complete':
                    ready[task] = completed_source(config, sources[task])
                    retries.pop(str(task), None)
                    last_progress = now
                else:
                    busy = value['status'] in ['source-busy', 'scratch-busy']
                    count = retries.get(str(task), {}).get('attempts', 0) + (not busy)
                    retryable = value.get('retryable', True)
                    retries[str(task)] = dict(attempts=count, retryable=retryable, updated=now,
                        next_retry=now + min(600, 15 * 2**min(count, 5)) * random.uniform(.8, 1.2),
                        error=value.get('error'), status=value['status'])
                    if retryable:
                        queue.append(task)
                    else:
                        quarantined.add(task)
                    if not busy:
                        recent_failures.append(now)
                        while recent_failures and now - recent_failures[0] > 300:
                            recent_failures.popleft()
                        if len(recent_failures) >= 20:
                            dispatch_pause_until = now + 600
                future.release()
            if now - last_scan >= 60:
                for path in (out / 'sources').glob('*/groups/*/complete.json'):
                    if str(path) not in groups:
                        value = json.loads(path.read_text())
                        verified_group(path, config, sources[value['source_task']], value['chunks'])
                        groups[str(path)] = value
                last_scan = now
            durable_events = sum(g['events'] for g in groups.values())
            if durable_events > previous_durable_events:
                last_progress = now
                previous_durable_events = durable_events
            history.append((now, durable_events))
            history = [r for r in history if now - r[0] < 1260]
            elapsed = now - history[0][0]
            rate = (durable_events - history[0][1]) / elapsed if elapsed >= 600 else None
            if now - last_heartbeat >= 15:
                state = 'running' if can_dispatch else ('waiting-for-proxy-renewal' if lifetime < minimum or not (out / 'PREFLIGHT-PASSED.json').exists() else 'transport-failure-backoff')
                if quarantined and not running and not queue:
                    state = 'integrity-failures-quarantined'
                snapshot = dict(state=state, heartbeat=now, started=began, cluster=config['cluster'],
                    requested_workers=desired, running_sources=len(running), queued_sources=len(queue),
                    verified_sources=len(ready), available_sources=len(sources), campaign_sources=len(manifest['files']),
                    verified_groups=len(groups), durable_events=durable_events,
                    available_events=sum(s['events'] for s in sources.values()), campaign_events=manifest['total_events'],
                    proxy_seconds_remaining=lifetime, initial_proxy_seconds_required=config.get('initial_proxy_seconds', 64800),
                    missing_Purdue_source_tasks=missing, quarantined=sorted(quarantined), retries=retries,
                    last_durable_progress=last_progress, recent_durable_events_per_second=rate,
                    projected_remaining_hours=(sum(s['events'] for s in sources.values()) - durable_events) / rate / 3600 if rate and rate > 0 else None,
                    eta_limit='Measured group-receipt rate after >=10 minutes; assumes sustained capacity/I/O.',
                    preflight_passed=(out / 'PREFLIGHT-PASSED.json').exists())
                snapshot['dispatch_pause_until'] = dispatch_pause_until
                atomic(out / 'controller-status.json', snapshot)
                atomic(retries_path, retries)
                with (out / 'history.jsonl').open('a') as stream:
                    stream.write(json.dumps({k: v for k, v in snapshot.items() if k != 'retries'}) + '\n')
                last_heartbeat = now
            if client.status != 'running':
                raise RuntimeError('Dask client disconnected; supervisor will reconnect')
            time.sleep(2)
        atomic(out / 'AVAILABLE-SOURCES-COMPLETE.json', dict(status='all-accessible-sources-complete',
               sources=len(ready), events=sum(r['events'] for r in ready.values()),
               missing_Purdue_source_tasks=missing, entire_campaign_complete=not missing,
               config_identity=config['identity'], finished=time.time()))
        if not missing:
            atomic(out / 'READY.json', dict(status='entire-campaign-complete', events=manifest['total_events'],
                   sources=len(ready), config_identity=config['identity'], finished=time.time()))
        cluster.scale(0)
    finally:
        client.close()


def watch(config):
    out = Path(config['output'])
    with (out / 'watchdog.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        process = None
        restarts = 0
        last_notify = time.time()
        last_alert = ''
        deadline = time.time() + config.get('monitor_hours', 20) * 3600
        while time.time() < deadline:
            now = time.time()
            status_path = out / 'controller-status.json'
            state = json.loads(status_path.read_text()) if status_path.exists() else {}
            terminal = (out / 'AVAILABLE-SOURCES-COMPLETE.json').exists()
            stale = now - state.get('heartbeat', now) > 300
            if not terminal and (process is None or process.poll() is not None or stale):
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                if state.get('state') == 'integrity-failure':
                    terminal = True
                else:
                    with (out / 'controller.log').open('a') as log:
                        process = subprocess.Popen([config['controller_python'],
                            str(Path(config['tools_dir']) / 'run_constituent_backfill.py'),
                            '--config', config['config_path'], 'controller'], stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
                    restarts += 1
                    atomic(out / 'controller-status.json', dict(state='starting-controller', heartbeat=time.time(),
                           controller_pid=process.pid, watchdog_restarts=restarts))
            reasons = []
            if restarts > 1:
                reasons.append('controller-restarted-review-required')
            if state.get('state') == 'waiting-for-proxy-renewal':
                reasons.append('proxy-renewal-required')
            if state.get('state') == 'transport-failure-backoff':
                reasons.append('repeated-source-failures-review-required')
            if state.get('quarantined') or state.get('state') == 'integrity-failure':
                reasons.append('integrity-review-required')
            if state.get('state') == 'running' and now - state.get('last_durable_progress', now) > 1800:
                reasons.append('no-durable-progress-for-30-minutes')
            if stale:
                reasons.append('controller-heartbeat-stale')
            if terminal:
                reasons.append('accessible-source-run-finished-check-missing-sources')
            alert = ','.join(reasons)
            due = now - last_notify >= config.get('agent_check_seconds', 1800)
            if (alert and alert != last_alert and now - last_notify >= 60) or due or terminal:
                message = ('Authorized overnight constituent-link backfill monitor: inspect ' + str(out)
                    + '/controller-status.json, watchdog-status.json and group receipts. '
                    + 'Fix recoverable failures autonomously, keep exact compact parity and two EOS readbacks; '
                    + 'do not overwrite originals or claim full completion while sources are missing. '
                    + 'Current trigger: ' + (alert or 'scheduled 30-minute check') + '.')
                try:
                    result = subprocess.run(['codex', 'queue', '--thread', config['thread_id'], '--message', message],
                                            capture_output=True, text=True, timeout=90)
                    delivery = dict(time=time.time(), returncode=result.returncode,
                                    stdout=result.stdout[-1500:], stderr=result.stderr[-1500:], trigger=alert or 'periodic')
                except Exception:
                    delivery = dict(time=time.time(), error=traceback.format_exc(), trigger=alert or 'periodic')
                atomic(out / 'last-agent-trigger.json', delivery)
                with (out / 'agent-triggers.jsonl').open('a') as stream:
                    stream.write(json.dumps(delivery) + '\n')
                if delivery.get('returncode') == 0:
                    last_notify = time.time()
                    last_alert = alert
            atomic(out / 'watchdog-status.json', dict(state='complete' if terminal else 'supervising',
                   heartbeat=time.time(), controller_pid=process.pid if process else None,
                   controller_state=state.get('state'), restarts=restarts,
                   last_agent_trigger=last_notify, alert=alert,
                   persistence_limit='Survives terminal disconnect; requires this AF pod and Codex daemon to remain alive.'))
            if terminal:
                return
            time.sleep(30)
        atomic(out / 'MONITOR-WINDOW-ENDED.json', dict(time=time.time(), note='20h monitoring window ended; inspect controller.'))
        subprocess.run(['codex', 'queue', '--thread', config['thread_id'], '--message',
                        'Constituent-link backfill: the configured 20-hour monitor window ended. Inspect '
                        + str(out) + ' and renew monitoring if any work remains.'],
                       capture_output=True, text=True, timeout=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('mode', choices=['pilot', 'controller', 'watch'])
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.mode == 'pilot':
        pilot(config)
    elif args.mode == 'watch':
        watch(config)
    else:
        out = Path(config['output'])
        with (out / 'controller.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                controller(config)
            except (ValueError, AssertionError):
                atomic(out / 'controller-status.json', dict(state='integrity-failure', heartbeat=time.time(), error=traceback.format_exc()))
                raise


if __name__ == '__main__':
    main()
