"""User-run Purdue->NERSC handoff of existing tarballs, with bounded RAM.

SSH authentication is interactive and occurs only when the user invokes this
script. Worker originals are never removed. Expected hashes and destination
acknowledgements are persisted on WORK before releasing each source lease.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time

from replay_scratch_budget import durable_json
from replay_import_transfer import load_snapshot,sha


def begin_export(root,row,expected,dask_worker=None):
    import fcntl,threading,time,uuid
    from pathlib import Path
    from distributed import get_worker
    from replay_bundle import verify
    import json
    worker=get_worker();project=Path(root);scratch=Path(row['scratch'])
    lease=(project/'purdue-setup/production/worker-local/source-leases'/('%05d.lock'%row['task'])).open('a+')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        archive=scratch/'data/committed/bundles'/('%05d.tar'%row['task']);receipt=Path(row['source'])/'BUNDLE-MANIFEST.json'
        if archive.is_symlink() or not archive.resolve().is_relative_to(scratch.resolve()):raise ValueError('Unsafe owned archive path')
        report=json.loads(receipt.read_text())
        if report['archive']!={k:expected[k] for k in ['bytes','sha256']}:raise ValueError('Worker archive differs from frozen snapshot')
        verify(archive,report)
        stream=archive.open('rb');token=uuid.uuid4().hex
        if not hasattr(worker,'_heptokens_exports'):worker._heptokens_exports={}
        item={'lease':lease,'stream':stream,'last':time.monotonic()};worker._heptokens_exports[token]=item
        def expire():
            while token in worker._heptokens_exports:
                time.sleep(15)
                if time.monotonic()-item['last']>300:
                    worker._heptokens_exports.pop(token,None);item['stream'].close();item['lease'].close();return
        threading.Thread(target=expire,daemon=True).start()
        return token
    except BaseException:lease.close();raise


def export_chunk(token,size=8*1024**2,dask_worker=None):
    import time
    if not 1<=size<=8*1024**2:raise ValueError('Transfer buffer exceeds bound')
    item=dask_worker._heptokens_exports[token];item['last']=time.monotonic()
    return item['stream'].read(size)


def touch_export(token,dask_worker=None):
    import time
    dask_worker._heptokens_exports[token]['last']=time.monotonic()


def end_export(token,dask_worker=None):
    item=getattr(dask_worker,'_heptokens_exports',{}).pop(token,None)
    if item:item['stream'].close();item['lease'].close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kit',type=Path,required=True);p.add_argument('--host',required=True);p.add_argument('--destination',required=True);p.add_argument('--identity',type=Path);p.add_argument('--tasks',type=int,nargs='+');p.add_argument('--stable-seconds',type=int,default=30);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if not re.fullmatch('[A-Za-z0-9_.@-]+',a.host) or not a.destination.startswith('/') or '\n' in a.destination:p.error('Use SSH host/alias and an absolute destination path')
    if a.stable_seconds<1:p.error('Positive stability interval required')
    kit=a.kit.resolve();snapshot=kit/'local-inventory.json';manifest=load_snapshot(snapshot);rows=manifest['sources'];root=Path(json.loads((kit/'purdue-config.json').read_text())['project'])
    if a.tasks:
        selected=set(a.tasks)
        if selected-set(r['task'] for r in rows):p.error('Unknown source task')
        rows=[r for r in rows if r['task'] in selected]
    remote_kit=a.destination+'/.transfer-kit/'+sha(kit/'SHA256SUMS')[:16]
    if a.dry_run:
        print(json.dumps({'sources':len(rows),'archive_bytes':sum(r['objects']['source.tar']['bytes'] for r in rows),'host':a.host,'destination':a.destination,'work_payload_staging_bytes':0,'max_buffer_bytes':8*1024**2,'source_files_deleted':False,'network_connections':0},indent=2));return
    # The agent never executes this branch. The user authenticates their NERSC
    # account once; all transfers reuse that connection without embedding tokens.
    from dask_gateway import Gateway
    from distributed import Client
    sys.path.insert(0,str(root/'purdue-setup'))
    from storage_watchdog import inspect_scratch
    receipts=root/'purdue-setup/results/nersc-handoff-records'/hashlib.sha256((a.host+a.destination).encode()).hexdigest()[:16]
    receipts.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='heptokens-ssh-') as temp:
        sock=str(Path(temp)/'control');ssh=['ssh','-S',sock,'-o','ControlMaster=auto','-o','ControlPersist=30m']
        if a.identity:ssh+=['-i',str(a.identity)]
        subprocess.run(ssh+['-M','-fN',a.host],check=True)
        print('NERSC verification command: bash '+shlex.quote(remote_kit+'/verify-local.sh')+' '+shlex.quote(a.destination),flush=True)
        try:
            exists=subprocess.run(ssh+[a.host,'test -e '+shlex.quote(remote_kit+'/SHA256SUMS')],check=False).returncode==0
            if not exists:
                command='mkdir -p '+shlex.quote(remote_kit)+' && tar --keep-old-files -xzf - -C '+shlex.quote(remote_kit)
                with (kit/'kit-content.tar.gz').open('rb') as data:subprocess.run(ssh+[a.host,command],stdin=data,check=True)
            check=subprocess.run(ssh+[a.host,'sha256sum '+shlex.quote(remote_kit+'/SHA256SUMS')],capture_output=True,text=True,check=True)
            if check.stdout.split()[0]!=sha(kit/'SHA256SUMS'):raise ValueError('Remote kit identity differs')
            subprocess.run(ssh+[a.host,'cd '+shlex.quote(remote_kit)+' && sha256sum -c SHA256SUMS >&2'],check=True)
            # Metadata caches only; all are bound by the frozen snapshot hashes.
            seed=['python3',remote_kit+'/tools/replay_import_transfer.py','seed-metadata','--manifest',remote_kit+'/local-inventory.json','--root',a.destination]
            subprocess.run(ssh+[a.host,shlex.join(seed)],check=True)
            gateway=Gateway();cluster=gateway.connect(json.loads((kit/'purdue-config.json').read_text())['cluster'],shutdown_on_close=False);client=Client(cluster)
            try:
                visible=client.run(inspect_scratch)
                candidates={r['task']:dict(r,worker=w) for w,v in visible.items() for r in v['complete']}
                for row in rows:
                    task=row['task'];expected=row['objects']['source.tar']
                    base=['python3',remote_kit+'/tools/replay_import_transfer.py','import-one','--manifest',remote_kit+'/local-inventory.json','--root',a.destination,'--task',str(task),'--backend','none','--workers','1','--stable-seconds',str(a.stable_seconds)]
                    check=subprocess.run(ssh+[a.host,shlex.join(base)],capture_output=True,text=True)
                    if check.returncode not in [0,3]:raise RuntimeError(check.stderr)
                    if check.returncode==3:
                        if task not in candidates:raise RuntimeError('Retained source unavailable; refresh EOS inventory if it has since committed: '+str(task))
                        current=candidates[task]
                        if current['attempt']!=row['attempt']:raise ValueError('Retained attempt differs; refresh frozen local snapshot')
                        worker=current['worker'];token=client.submit(begin_export,str(root),current,expected,workers=[worker],allow_other_workers=False,resources={'replay_source':1},pure=False).result()
                        finished=threading.Event()
                        def heartbeat():
                            while not finished.wait(30):
                                try:client.run(touch_export,token,workers=[worker])
                                except Exception:return
                        keeper=threading.Thread(target=heartbeat,daemon=True);keeper.start()
                        try:
                            receive=['python3',remote_kit+'/tools/replay_import_transfer.py','receive','--manifest',remote_kit+'/local-inventory.json','--root',a.destination,'--task',str(task)]
                            with subprocess.Popen(ssh+[a.host,shlex.join(receive)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
                                h=hashlib.sha256();size=0
                                while True:
                                    block=client.run(export_chunk,token,workers=[worker])[worker]
                                    if not block:break
                                    proc.stdin.write(block);h.update(block);size+=len(block)
                                proc.stdin.close();proc.stdin=None
                                stdout,stderr=proc.communicate()
                                if proc.returncode:raise RuntimeError(stderr.decode())
                                if {'bytes':size,'sha256':h.hexdigest()}!={k:expected[k] for k in ['bytes','sha256']}:raise ValueError('Streamed worker bytes differ')
                            check=subprocess.run(ssh+[a.host,shlex.join(base)],capture_output=True,text=True)
                            if check.returncode:raise RuntimeError(check.stderr)
                            store_ack(check.stdout,row,snapshot,receipts)
                        finally:
                            finished.set();keeper.join(timeout=5)
                            client.run(end_export,token,workers=[worker])
                    else:store_ack(check.stdout,row,snapshot,receipts)
                    print('Validated NERSC source %05d (%d events)'%(task,row['events']),flush=True)
            finally:client.close();cluster.close(shutdown=False);gateway.close()
        finally:subprocess.run(ssh+['-O','exit',a.host],check=False)


def store_ack(stdout,row,snapshot,receipts):
    messages=[json.loads(line) for line in stdout.splitlines() if line.strip()]
    ack=next((m for m in messages if m.get('status')=='verified-import' and m.get('task')==row['task']),None)
    if (ack is None or ack['snapshot_sha256']!=sha(snapshot) or ack['events']!=row['events']
        or ack['object_sha256']!={n:v['sha256'] for n,v in row['objects'].items()}):raise ValueError('Destination acknowledgement differs from expected transfer')
    durable_json(receipts/('source-%05d.json'%row['task']),ack)


if __name__=='__main__':main()
