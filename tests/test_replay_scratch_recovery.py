"""Orphan bookkeeping recovery must not evict data or steal a source lease."""
import fcntl,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from replay_scratch_recovery import reconcile_absent_attempts


def setup(tmp_path,validation=None,phase='test',started=1):
    attempt='a'*32;node='b'*64;key='00007-'+attempt
    row={'phase':phase,'task':7,'attempt':attempt,'validation_limit':validation,'started':started,
         'scratch':'/tmp/owned-attempt','reservation':{'key':key,'node':node,'bytes':10}}
    p=tmp_path/'attempts/00007'/attempt/'status.json';p.parent.mkdir(parents=True);p.write_text(json.dumps(row))
    ledger=tmp_path/'node-disk'/node/'reservations.json';ledger.parent.mkdir(parents=True);ledger.write_text(json.dumps({key:10}))
    (tmp_path/'source-leases').mkdir();calls=[]
    return row,calls,lambda shared,reservation:calls.append((shared,reservation))


def test_absent_unlocked_old_attempt_releases_only_booking(tmp_path):
    row,calls,release=setup(tmp_path)
    physical=tmp_path/'retained-payload';physical.write_bytes(b'keep')
    r=reconcile_absent_attempts(tmp_path,'test',[],release,now=1000)
    assert len(r['released'])==len(calls)==1 and not r['physical_files_deleted']
    assert physical.read_bytes()==b'keep'


def test_visible_attempt_keeps_budget_for_canonical_resume(tmp_path):
    row,calls,release=setup(tmp_path)
    r=reconcile_absent_attempts(tmp_path,'test',[row['scratch']],release,now=1000)
    assert not calls and r['retained'][0]['reason']=='scratch-still-visible'


def test_inherited_lease_prevents_release(tmp_path):
    row,calls,release=setup(tmp_path)
    with (tmp_path/'source-leases/00007.lock').open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        r=reconcile_absent_attempts(tmp_path,'test',[],release,now=1000)
    assert not calls and r['retained'][0]['reason']=='source-lease-held'


def test_recent_attempt_is_not_reclaimed(tmp_path):
    row,calls,release=setup(tmp_path,started=950)
    r=reconcile_absent_attempts(tmp_path,'test',[],release,now=1000)
    assert not calls and r['retained'][0]['reason']=='recent-attempt'


def test_validation_is_not_a_production_booking(tmp_path):
    row,calls,release=setup(tmp_path,validation=1000)
    r=reconcile_absent_attempts(tmp_path,'test',[],release,now=1000)
    assert not calls and not r['released']


def test_other_phase_is_untouched(tmp_path):
    row,calls,release=setup(tmp_path,phase='other')
    r=reconcile_absent_attempts(tmp_path,'test',[],release,now=1000)
    assert not calls and not r['released']
