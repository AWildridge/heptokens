"""Failure injection for bounded retries, process cleanup and remote quarantine."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from replay_recovery import RecoveryQueue
from replay_runtime import run_cpu_process
import quarantine_replay as quarantine
import commit_replay_gfal as remote
from test_replay_remote_source import source


def test_detector_crash_retries_same_command_and_preserves_partial_outputs(tmp_path):
    from simulate_delphes_pair import simulate_detector
    marker=tmp_path/'attempts'
    code="""from pathlib import Path
import sys
p=Path(sys.argv[1]);n=int(p.read_text())+1 if p.exists() else 1;p.write_text(str(n))
Path(sys.argv[2]).write_bytes(b'partial' if n==1 else b'complete')
print('seed='+sys.argv[3],flush=True)
sys.exit(129 if n==1 else 0)
"""
    command=[sys.executable,'-c',code,str(marker),str(tmp_path/'ATLAS.root'),'3205000']
    attempts=simulate_detector(command,tmp_path,'ATLAS',os.environ.copy())
    assert [a['returncode'] for a in attempts]==[129,0]
    assert (tmp_path/'ATLAS-failed-attempt-1.root').read_bytes()==b'partial'
    assert (tmp_path/'ATLAS.root').read_bytes()==b'complete'
    assert 'seed=3205000' in attempts[0]['log_tail']
    assert 'seed=3205000' in (tmp_path/'ATLAS.log').read_text()


def test_deterministic_detector_error_is_not_retried(tmp_path):
    from simulate_delphes_pair import simulate_detector
    with pytest.raises(subprocess.CalledProcessError) as result:
        simulate_detector([sys.executable,'-c','raise SystemExit(2)'],tmp_path,'ATLAS',os.environ.copy())
    assert len(result.value.detector_attempts)==1


def test_permanent_failure_does_not_prevent_healthy_sources_or_reset_on_restart(tmp_path):
    path=tmp_path/'retries.json'
    queue=RecoveryQueue(path,delay=1)
    pending=[4,5,6];completed=[];now=0
    while pending:
        task=pending.pop(0)
        if not queue.ready(task,now):
            pending.append(task);now+=1;continue
        if task==4:
            if queue.failed(task,'replay',RuntimeError('segfault'),now)!='quarantined':pending.append(task)
        else:
            completed.append(task);queue.completed(task)
        now+=1
        queue=RecoveryQueue(path,delay=1)
    assert completed==[5,6] and queue.quarantined()==[4]
    assert len(queue.records['4']['history'])==3
    assert not queue.ready(4,100000)


def test_timeout_releases_descendant_lock_even_if_parent_exits_on_term(tmp_path):
    import fcntl
    ready=tmp_path/'ready';lock=tmp_path/'lock'
    child="import fcntl,time,pathlib;f=open(%r,'w');fcntl.lockf(f,fcntl.LOCK_EX);pathlib.Path(%r).touch();time.sleep(60)"%(str(lock),str(ready))
    parent="import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',%r]);time.sleep(60)"%child
    with pytest.raises(subprocess.TimeoutExpired):
        run_cpu_process([sys.executable,'-c',parent],timeout=2,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    assert ready.exists()
    with lock.open('a') as f:fcntl.lockf(f,fcntl.LOCK_EX|fcntl.LOCK_NB)


@pytest.fixture
def failed_source(source,tmp_path,monkeypatch):
    directory,storage,_=source
    for name in ['stage_file','verify_staged_file','remote_adler','publish_file']:
        monkeypatch.setattr(quarantine,name,getattr(remote,name))
    work=tmp_path/'work';work.mkdir();(work/'input.root').write_bytes(b'original-input')
    roots={'source':directory,'work':work}
    return roots,storage,tmp_path/'quarantine.json'


def test_failed_source_eviction_requires_two_full_remote_passes(failed_source,monkeypatch):
    roots,storage,evidence=failed_source
    original=quarantine.verify_staged_file;calls=[]
    def verify(path,item,seconds):
        assert (roots['work']/'input.root').exists()
        calls.append(item['url']);return original(path,item,seconds)
    monkeypatch.setattr(quarantine,'verify_staged_file',verify)
    result=quarantine.quarantine(roots,'davs://example.invalid/failed/4',evidence,{'error':'segfault'},1)
    assert result['status']=='verified-incomplete-quarantine'
    assert result['eviction']=='verified-and-pruned'
    assert not (roots['work']/'input.root').exists()
    assert (roots['source']/'chunk-000000/receipt.json').exists()
    assert not (roots['source']/'chunk-000000/CMS.h5').exists()
    assert all(calls.count(item['url'])==2 for item in result['files'].values())
    assert json.loads((roots['source'].parent.parent/'storage-ledger.json').read_text())['reservations']['00004/chunk-000000']==0


def test_corrupt_quarantine_keeps_all_local_data_and_reservations(failed_source,monkeypatch):
    roots,storage,evidence=failed_source
    original=quarantine.verify_staged_file;calls={}
    def corrupt_second_pass(path,item,seconds):
        url=item['url'];calls[url]=calls.get(url,0)+1
        if calls[url]==2 and url.endswith('/input.root'):storage[url]=b'corrupt'
        return original(path,item,seconds)
    monkeypatch.setattr(quarantine,'verify_staged_file',corrupt_second_pass)
    with pytest.raises(ValueError,match='readback'):
        quarantine.quarantine(roots,'davs://example.invalid/failed/4',evidence,{'error':'segfault'},1)
    assert (roots['work']/'input.root').read_bytes()==b'original-input'
    assert (roots['source']/'chunk-000000/CMS.h5').exists()
    assert json.loads((roots['source'].parent.parent/'storage-ledger.json').read_text())['reservations']['00004/chunk-000000']>0


def test_interrupted_quarantine_eviction_resumes_from_remote_proof(failed_source,monkeypatch):
    roots,storage,evidence=failed_source
    original=Path.unlink
    def interrupted(path,*args,**kwargs):
        if path.name=='input.root':raise RuntimeError('power loss')
        return original(path,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path,'unlink',interrupted)
        with pytest.raises(RuntimeError,match='power loss'):
            quarantine.quarantine(roots,'davs://example.invalid/failed/4',evidence,{'error':'segfault'},1)
    assert not (roots['source']/'chunk-000000/CMS.h5').exists()
    proof=quarantine.quarantine(roots,'davs://example.invalid/failed/4',evidence,{'error':'segfault'},1)
    assert proof['eviction']=='verified-and-pruned' and not (roots['work']/'input.root').exists()
