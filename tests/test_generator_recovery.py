"""Failures observed overnight: duplicate leases, lost acknowledgements, retries."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_generator_backfill as recovery
from supervise_generator_backfill import interleave_sources


def source():
    return dict(task=7,record='63193',uri='root://example/input',bytes=10,checksum='adler32:00000001')


def receipt():
    item=dict(source=source(),task=7,status='complete',bytes=123,sha256='a'*64,
              validation=dict(events=10),finished=1,started=0)
    readback=dict(bytes=123,sha256='a'*64,readback='passed',url='davs://example/output')
    item['remote']=dict(status='two-stable-full-readbacks-passed',first=readback,second=dict(readback))
    return item


def config(tmp_path):
    return dict(output=str(tmp_path),remote='davs://example/campaign',root=str(tmp_path),remote_cap_bytes=1000)


def test_duplicate_lease_waits_instead_of_failing_campaign(tmp_path):
    cfg=config(tmp_path);shared=tmp_path/'sources/00007';shared.mkdir(parents=True)
    with (shared/'task.lock').open('a+') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert recovery.run_task(cfg,source())==dict(status='source-busy',task=7)
        # A receipt is usable even while the original process finishes cleanup.
        recovery.atomic(shared/'complete.json',receipt())
        assert recovery.run_task(cfg,source())['status']=='complete'


def test_different_root_bytes_get_distinct_immutable_reservations(tmp_path):
    cfg=config(tmp_path)
    # The original upload reservation is preserved during regeneration.
    recovery.reserve_remote(cfg,'sources/00007',123,'a'*64)
    urls=[]
    for sha in ['b'*64,'c'*64]:
        recovery.reserve_remote(cfg,'sources/00007/attempts/'+sha,123,sha)
        urls.append(recovery.attempt_url(cfg,'sources','00007',sha))
    assert len(set(urls))==2
    ledger=json.loads((tmp_path/'remote-ledger.json').read_text())
    assert len(ledger)==3 and ledger['sources/00007']['sha256']=='a'*64
    with pytest.raises(ValueError,match='Conflicting'):
        recovery.reserve_remote(cfg,'sources/00007',123,'b'*64)


def test_reservation_cap_includes_abandoned_attempts(tmp_path):
    cfg=config(tmp_path);cfg['remote_cap_bytes']=150
    recovery.reserve_remote(cfg,'old',100,'a')
    with pytest.raises(ValueError,match='budget'):
        recovery.reserve_remote(cfg,'new',100,'b')
    assert json.loads((tmp_path/'remote-ledger.json').read_text())=={'old':dict(bytes=100,sha256='a')}


def test_lost_commit_ack_adopts_existing_proof_without_rerunning(tmp_path):
    cfg=config(tmp_path);shared=tmp_path/'sources/00007';attempt=shared/'attempt-old';attempt.mkdir(parents=True)
    item=receipt();proof=item.pop('remote');item.pop('status')
    recovery.atomic(attempt/'validated.json',item)
    recovery.atomic(attempt/'remote-proof.json',proof)
    scratch=tmp_path/'scratch';scratch.mkdir()
    def forbidden(*args):raise AssertionError('Proven output must not be regenerated or recopied')
    result=recovery.recover_source(cfg,source(),shared,scratch,forbidden)
    assert result['status']=='complete'
    recovery.checked_receipt(shared/'complete.json',source())


def test_mismatched_receipt_is_never_adopted(tmp_path):
    item=receipt();item['remote']['second']['sha256']='bad'
    path=tmp_path/'complete.json';recovery.atomic(path,item)
    with pytest.raises(ValueError,match='proof differs'):
        recovery.checked_receipt(path,source())


def test_partial_legacy_orphan_is_not_overwritten_or_committed(tmp_path):
    cfg=config(tmp_path);shared=tmp_path/'sources/00007';attempt=shared/'attempt-old';attempt.mkdir(parents=True)
    item=receipt();item.pop('remote');item.pop('status');recovery.atomic(attempt/'validated.json',item)
    recovery.reserve_remote(cfg,'sources/00007',123,'a'*64)
    scratch=tmp_path/'scratch';scratch.mkdir();requests=[]
    def partial(args,*unused):
        request=json.loads(Path(args[-1]).read_text());requests.append(request)
        Path(request['source']).write_bytes(b'incomplete readback')
        raise subprocess.CalledProcessError(1,args)
    assert recovery.recover_source(cfg,source(),shared,scratch,partial) is None
    assert requests[0]['action']=='restore'
    assert not (shared/'complete.json').exists()
    assert (attempt/'validated.json').exists()
    assert not list(scratch.glob('restore-*.root'))


def test_restored_orphan_requires_native_validation_and_two_readbacks(tmp_path):
    cfg=config(tmp_path);shared=tmp_path/'sources/00007';attempt=shared/'attempt-old';attempt.mkdir(parents=True)
    item=receipt();proof=item.pop('remote');item.pop('status');item['remote_url']='davs://example/orphan'
    recovery.atomic(attempt/'validated.json',item)
    scratch=tmp_path/'scratch';scratch.mkdir();actions=[]
    def transport(args,*unused):
        request=json.loads(Path(args[-1]).read_text());actions.append(request['action'])
        if request['action']=='restore':Path(request['source']).write_bytes(b'restored')
        else:recovery.atomic(Path(request['proof']),proof)
    with patch.object(recovery,'validate',return_value=dict(events=10)) as validation:
        restored=recovery.recover_source(cfg,source(),shared,scratch,transport)
    assert actions==['restore','publish']
    assert validation.call_args.kwargs['expected']==10
    assert restored['status']=='complete'
    recovery.checked_receipt(shared/'complete.json',source())


def test_admission_interleaves_bins_instead_of_starving_later_samples():
    original=[dict(task=i,record=r) for r,i in [('a',1),('a',2),('b',3),('b',4),('c',5)]]
    assert [s['task'] for s in interleave_sources(original)]==[1,3,5,2,4]


def test_more_than_twenty_transient_failures_do_not_stop_unrelated_work(tmp_path,monkeypatch):
    import supervise_generator_backfill as controller
    original=[dict(source(),task=i,record='x') for i in range(24)]
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(files=original,total_events=24,records=[dict(record='x',events=24)])))
    cfg=dict(config(tmp_path),manifest=str(manifest),cluster='test',workers=1,inflight=24,software={})
    calls={}
    class Future:
        def __init__(self,value):self.value=value
        def result(self):return self.value
    class Cluster:
        def scale(self,n):pass
    class Gateway:
        def list_clusters(self):
            from types import SimpleNamespace
            return [SimpleNamespace(name='test')]
        def connect(self,*args,**kwargs):return Cluster()
    class Client:
        status='running'
        def __init__(self,*args):pass
        def register_plugin(self,*args,**kwargs):pass
        def close(self):pass
        def submit(self,function,config,item,**kwargs):
            task=item['task'];calls[task]=calls.get(task,0)+1
            if calls[task]==1:return Future(dict(status='source-failed',retryable=True,error='simulated timeout'))
            result=receipt();result.update(source=item,task=task,validation=dict(events=1))
            recovery.atomic(tmp_path/'sources'/('%05d'%task)/'complete.json',result)
            return Future(result)
    class Completed:
        def __init__(self):self.values=[]
        def add(self,future):self.values.append(future)
        def next_batch(self,block):values=self.values;self.values=[];return values
    monkeypatch.setattr('dask_gateway.Gateway',Gateway)
    monkeypatch.setattr('distributed.Client',Client)
    monkeypatch.setattr('distributed.as_completed',Completed)
    monkeypatch.setattr(controller.random,'uniform',lambda *args:0)
    monkeypatch.setattr(controller.time,'sleep',lambda *args:None)
    controller.run_controller(cfg)
    assert all(calls[i]==2 for i in range(24))
    assert json.loads((tmp_path/'READY.json').read_text())['events']==24


def test_independent_watchdog_restarts_crashed_controller(tmp_path,monkeypatch):
    import watch_generator_backfill as watchdog
    import time
    tools=tmp_path/'tools';tools.mkdir()
    # Two complete process exits without READY, then a successful third run.
    script="""from pathlib import Path
import json,sys
c=json.load(open(sys.argv[2]));p=Path(c['output']);f=p/'starts'
n=int(f.read_text())+1 if f.exists() else 1
f.write_text(str(n))
if n<3:sys.exit(86)
(p/'READY.json').write_text('{}')
"""
    (tools/'supervise_generator_backfill.py').write_text(script)
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(dict(output=str(tmp_path),tools_dir=str(tools),python=sys.executable)))
    sleep=time.sleep
    monkeypatch.setattr(watchdog.time,'sleep',lambda delay:sleep(.01))
    watchdog.watch(cfg)
    assert (tmp_path/'starts').read_text()=='3'
    assert json.loads((tmp_path/'watchdog-status.json').read_text())['restarts']==2


def test_node_scratch_reservations_prevent_simultaneous_overadmission(tmp_path,monkeypatch):
    import generator_scratch
    from types import SimpleNamespace
    # Reserve one GiB per source on a simulated 1.5 GiB free backing disk.
    cfg=dict(config(tmp_path),max_output_bytes=0,scratch_safety_bytes=0)
    monkeypatch.setattr(generator_scratch.os,'statvfs',lambda path:SimpleNamespace(f_bavail=1536*1024**2,f_frsize=1))
    with generator_scratch.admit(cfg) as first:
        assert first
        with generator_scratch.admit(cfg) as second:assert not second
    with generator_scratch.admit(cfg) as after_release:assert after_release


def test_node_scratch_refuses_full_disk_before_generating_payload(tmp_path,monkeypatch):
    import generator_scratch
    from types import SimpleNamespace
    monkeypatch.setattr(generator_scratch.os,'statvfs',lambda path:SimpleNamespace(f_bavail=0,f_frsize=1))
    with generator_scratch.admit(config(tmp_path)) as admitted:assert not admitted


def test_node_scratch_tolerates_an_older_owner_finishing_cleanup(tmp_path,monkeypatch):
    import generator_scratch
    import hashlib
    node=hashlib.sha256(Path('/proc/sys/kernel/random/boot_id').read_bytes()).hexdigest()[:24]
    directory=tmp_path/'scratch-reservations'/node;directory.mkdir(parents=True)
    stale=directory/'old-owner.json';stale.write_text(json.dumps(dict(bytes=999999999999)))
    unlink=Path.unlink
    raced=[]
    def finishing_owner(path,**kwargs):
        if path==stale and not raced:
            raced.append(True)
            unlink(path,missing_ok=True)
        return unlink(path,**kwargs)
    monkeypatch.setattr(Path,'unlink',finishing_owner)
    with generator_scratch.admit(dict(config(tmp_path),max_output_bytes=0,scratch_safety_bytes=0)) as admitted:assert admitted
    assert raced
