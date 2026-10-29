"""Management recovery must respect a user stop and retain single ownership."""
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'purdue-setup'),str(Path(__file__).resolve().parents[1]/'tools')]
if not (ROOT/'purdue-setup/storage_watchdog.py').is_file():
    pytest.skip('Requires the separate Purdue AF site checkout', allow_module_level=True)
spec=importlib.util.spec_from_file_location('storage_recovery_controls',ROOT/'purdue-setup/storage_watchdog.py')
watch=importlib.util.module_from_spec(spec);spec.loader.exec_module(watch)


@pytest.fixture
def sandbox(tmp_path,monkeypatch):
    site=tmp_path/'site';out=site/'results/outage';out.mkdir(parents=True);(site/'production').mkdir()
    monkeypatch.setattr(watch,'SITE',site);monkeypatch.setattr(watch,'OUT',out);monkeypatch.setattr(watch,'STOP',site/'production/STOP-tmp')
    calls=[];monkeypatch.setattr(watch.subprocess,'run',lambda command,**kwargs:calls.append(command))
    return site,calls


def test_user_stop_is_not_overwritten_or_released(sandbox):
    site,calls=sandbox;watch.STOP.write_text(json.dumps({'reason':'user-request'}));before=watch.STOP.read_bytes()
    assert not watch.hold() and not watch.restart_production()
    assert watch.STOP.read_bytes()==before and not calls


def test_restart_uses_guaranteed_cpu_budget(sandbox):
    site,calls=sandbox;assert watch.hold();assert watch.restart_production()
    assert not watch.STOP.exists() and len(calls)==2
    assert calls[0][calls[0].index('--max-active')+1]=='25'
    assert calls[1][calls[1].index('--max-workers')+1]=='25'


def test_running_supervisor_prevents_duplicate_launcher(sandbox):
    site,calls=sandbox;assert watch.hold()
    with (site/'production/worker-local-supervisor.lock').open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        assert not watch.restart_production()
    assert watch.STOP.exists() and not calls


def attempt(site,worker):
    row={'task':7,'attempt':'a'*32,'scratch':'/tmp/owned-attempt'}
    p=site/'production/worker-local/attempts/00007'/row['attempt']/'status.json';p.parent.mkdir(parents=True)
    (site/'production/worker-local/source-leases').mkdir()
    p.write_text(json.dumps(dict(row,phase=watch.PHASE,worker=worker)))
    return row,p


def test_rebinding_cannot_take_a_live_worker(sandbox):
    site,calls=sandbox;row,p=attempt(site,'old');before=p.read_bytes()
    with pytest.raises(ValueError,match='still live'):watch.bind_retained(row,'new',['old','new'])
    assert p.read_bytes()==before


def test_rebinding_cannot_take_an_inherited_source_lease(sandbox):
    site,calls=sandbox;row,p=attempt(site,'old');before=p.read_bytes()
    with (site/'production/worker-local/source-leases/00007.lock').open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        with pytest.raises(BlockingIOError):watch.bind_retained(row,'new',['new'])
    assert p.read_bytes()==before


def test_rebinding_retains_old_provenance(sandbox):
    site,calls=sandbox;row,p=attempt(site,'old');watch.bind_retained(row,'new',['new'])
    assert json.loads(p.read_text())['worker']=='new'
    audit=json.loads((watch.OUT/'worker-rebindings'/('00007-'+row['attempt']+'.json')).read_text())
    assert audit['previous']['worker']=='old' and audit['new_worker']=='new'
