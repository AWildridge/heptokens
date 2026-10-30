"""Restartable campaign controller with source-local retries and live heartbeat.

Worker loss/lease waits never count toward a global failure limit. A durable
source receipt is authoritative. Scientific integrity failures are quarantined
visibly while unrelated sources continue. A controller crash is restarted by
the supervisor; completed sources are recounted on each restart.
"""
import argparse
from collections import defaultdict,deque
import fcntl
import hashlib
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
from run_generator_backfill import atomic, checked_receipt, digest


def dispatch_task(config,source):
    # Deliberately self-contained: workers need no module installation or /home.
    import json,os,signal,subprocess,time,uuid
    from pathlib import Path
    base=Path(config['output'],'dispatch');base.mkdir(exist_ok=True)
    token=str(source['task'])+'-'+uuid.uuid4().hex
    result=base/(token+'-result.json')
    request=base/(token+'.json')
    request.write_text(json.dumps(dict(config=config,source=source,result=str(result))))
    driver=str(Path(config['tools_dir'],'run_generator_source.py'))
    with (base/(token+'.log')).open('w') as log:
        proc=subprocess.Popen([config.get('python','/work/pixi/global/.pixi/envs/default/bin/python'),driver,str(request)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=config.get('source_timeout_seconds',14400))
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            return dict(status='source-failed',task=int(source['task']),retryable=True,error='Source process timeout',request=str(request))
    if not result.exists():
        return dict(status='source-failed',task=int(source['task']),retryable=True,error='Source child exited without a result: '+str(code),request=str(request))
    return json.loads(result.read_text())


def sys_executable(config):
    return config.get('python','/work/pixi/global/.pixi/envs/default/bin/python')


def interleave_sources(sources):
    groups=defaultdict(deque)
    for source in sources:groups[str(source['record'])].append(source)
    result=[]
    while any(groups.values()):
        for record in sorted(groups):
            if groups[record]:result.append(groups[record].popleft())
    return result


def worker_plugin():
    from distributed import WorkerPlugin
    class OneSource(WorkerPlugin):
        async def setup(self,worker):
            import os
            from tornado.ioloop import PeriodicCallback
            async def refresh():
                st=os.statvfs('/tmp')
                healthy=st.f_bavail*st.f_frsize>=12*1024**3
                await worker.set_resources(generator_source=int(healthy))
            await refresh()
            self.monitor=PeriodicCallback(refresh,15000)
            self.monitor.start()
        def teardown(self,worker):
            self.monitor.stop()
    return OneSource()


def completed_index(config,manifest):
    ready={}
    for source in manifest['files']:
        receipt=Path(config['output'],'sources','%05d'%int(source['task']),'complete.json')
        if receipt.exists():ready[int(source['task'])]=checked_receipt(receipt,source)
    return ready


def run_controller(config):
    from dask_gateway import Gateway
    from distributed import Client,as_completed
    out=Path(config['output'])
    manifest=json.loads(Path(config['manifest']).read_text())
    ready=completed_index(config,manifest)
    all_sources={int(s['task']):s for s in manifest['files']}
    ordered=interleave_sources(s for s in manifest['files'] if int(s['task']) not in ready)
    queue=deque(int(s['task']) for s in ordered)
    retry_file=out/'recovery-retries.json'
    retries=json.loads(retry_file.read_text()) if retry_file.exists() else {}
    quarantined={int(k) for k,v in retries.items() if not v.get('retryable',True) and int(k) not in ready}
    last_error=None
    gateway=Gateway()
    reports=gateway.list_clusters()
    available=set(reports) if isinstance(reports,dict) else {report.name for report in reports}
    name=config.get('cluster')
    if name not in available:
        # Recreate only this campaign's cluster when it no longer exists. The
        # facility enforces one cluster/user; never replace another cluster.
        if available:raise RuntimeError('Another Dask cluster exists; refusing to replace it')
        cluster=gateway.new_cluster(pixi_project='/work/pixi/global',worker_cores=1,worker_memory=8,env=config['worker_env'])
        config['cluster']=cluster.name
        atomic(out/'campaign-recovery.json',config)
    else:cluster=gateway.connect(name,shutdown_on_close=False)
    client=Client(cluster)
    client.register_plugin(worker_plugin(),name='generator-recovery-one-source')
    cluster.scale(config.get('workers',1000))
    running={}
    ac=as_completed()
    started=time.time();last_heartbeat=0;last_index=0
    def heartbeat(state='running'):
        nonlocal last_heartbeat,last_index
        now=time.time()
        failures={k:v for k,v in retries.items() if int(k) not in ready}
        atomic(out/'recovery-controller-status.json',dict(state=state,heartbeat=now,started=started,
            cluster=config['cluster'],requested_workers=config.get('workers',1000),running=len(running),
            queued=len(queue),verified_sources=len(ready),verified_events=sum(x['validation']['events'] for x in ready.values()),
            total_sources=len(all_sources),total_events=manifest['total_events'],quarantined=sorted(quarantined),
            failures=failures,last_controller_error=last_error))
        atomic(retry_file,retries)
        if now-last_index>=60 or state=='complete':
            atomic(out/'ready-sources.json',dict(schema='heptokens-generator-backfill-index-v1',
                entire_campaign_complete=state=='complete',updated=now,
                sources={str(k):dict(task=k,source=x['source'],events=x['validation']['events'],bytes=x['bytes'],sha256=x['sha256'],url=x['remote']['second']['url'],receipt=str(out/'sources'/('%05d'%k)/'complete.json')) for k,x in ready.items()},
                events=sum(x['validation']['events'] for x in ready.values()),manifest_sha256=digest(config['manifest'])))
            last_index=now
        last_heartbeat=now
    heartbeat('starting')
    while len(ready)<len(all_sources):
        now=time.time()
        # Bound in-flight submissions and stagger retries, without starving
        # fresh work in the other pThat bins behind a failed source.
        for _ in range(len(queue)):
            if len(running)>=config.get('inflight',1000):break
            task=queue.popleft()
            if task in ready or task in quarantined:continue
            info=retries.get(str(task),{})
            if now<info.get('next_retry',0):queue.append(task);continue
            receipt=out/'sources'/('%05d'%task)/'complete.json'
            if receipt.exists():ready[task]=checked_receipt(receipt,all_sources[task]);continue
            future=client.submit(dispatch_task,config,all_sources[task],pure=False,retries=0,resources={'generator_source':1})
            running[future]=task;ac.add(future)
            if time.time()-last_heartbeat>=15:heartbeat()
        if running:
            finished=ac.next_batch(block=False)
        else:finished=[]
        for future in finished:
            task=running.pop(future)
            try:result=future.result()
            except Exception:
                result=dict(status='source-failed',retryable=True,error=traceback.format_exc())
            status=result.get('status')
            if status=='complete':
                receipt=out/'sources'/('%05d'%task)/'complete.json'
                ready[task]=checked_receipt(receipt,all_sources[task])
                retries.pop(str(task),None)
                print(json.dumps(dict(task=task,status='verified',events=ready[task]['validation']['events'])),flush=True)
            else:
                prior=retries.get(str(task),{})
                busy=status in ['source-busy','scratch-busy']
                count=prior.get('attempts',0)+(0 if busy else 1)
                retryable=busy or result.get('retryable',True)
                delay=random.uniform(10,30) if busy else min(300,10*2**min(count,5))*random.uniform(.8,1.2)
                retries[str(task)]=dict(attempts=count,retryable=retryable,status=status,error=result.get('error'),updated=time.time(),next_retry=time.time()+delay)
                if retryable:queue.append(task)
                else:quarantined.add(task)
                print(json.dumps(dict(task=task,status=status,retryable=retryable,attempts=count)),flush=True)
        if time.time()-last_heartbeat>=15:heartbeat('needs-attention' if quarantined and not running and not queue else 'running')
        if client.status!='running':raise RuntimeError('Dask client disconnected; supervisor will reconnect')
        time.sleep(1)
    counts={str(record['record']):sum(x['validation']['events'] for x in ready.values() if str(x['source']['record'])==str(record['record'])) for record in manifest['records']}
    if sum(counts.values())!=manifest['total_events'] or any(counts[str(r['record'])]!=r['events'] for r in manifest['records']):
        raise ValueError('Campaign or per-bin event total differs from frozen catalog')
    heartbeat('complete')
    atomic(out/'READY.json',dict(schema='heptokens-generator-backfill-ready-v1',entire_campaign_complete=True,sources=len(ready),events=manifest['total_events'],per_record=counts,manifest_sha256=digest(config['manifest']),index_sha256=digest(out/'ready-sources.json'),software=config['software'],finished=time.time()))
    cluster.scale(0)
    client.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    config=json.loads(Path(args.config).read_text());out=Path(config['output'])
    for name,expected in config['software'].items():
        if digest(Path(config['root'])/name)!=expected:raise ValueError('Frozen campaign software changed: '+name)
    with open(out/'recovery-controller.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (out/'READY.json').exists():
            try:run_controller(config)
            except ValueError:
                atomic(out/'recovery-controller-status.json',dict(state='integrity-failure',heartbeat=time.time(),error=traceback.format_exc()))
                raise
            except Exception:
                atomic(out/'recovery-controller-status.json',dict(state='reconnecting',heartbeat=time.time(),error=traceback.format_exc()))
                time.sleep(30)
            finally:
                from distributed import default_client
                try:default_client().close()
                except ValueError:pass


if __name__=='__main__':main()
