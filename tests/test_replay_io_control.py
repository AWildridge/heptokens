from replay_io_control import govern
import json
import multiprocessing
import time
from pathlib import Path

POLICY={'window_seconds':300,'minimum_completed_sources':2,'minimum_workers':25,'maximum_workers':249,
        'minimum_transfer_slots':2,'maximum_transfer_slots':32,'transfer_step':4,'regression_ratio':.8}


def state():
    return {'max_active':249,'max_uploads':8,'max_readbacks':16,'last_evaluation':0,
            'baseline':{'durable_mib_per_second':100,'upload_seconds_per_gib':5,'readback_seconds_per_gib':5}}


def observation(**updates):
    row={'time':400,'completed_sources':4,'durable_mib_per_second':100,'errors':0,'archive_backlog':20,
         'waiting_readbacks':0,'waiting_uploads':0,'registered_workers':100,'network_saturated':False,
         'upload_seconds_per_gib':5,'readback_seconds_per_gib':5}
    row.update(updates);return row


def test_source_completion_burstiness_alone_does_not_throttle():
    new,action=govern(state(),observation(durable_mib_per_second=40),POLICY)
    assert new['max_active']==249 and action is None


def test_real_io_regression_throttles_admission_without_killing_tasks():
    new,action=govern(state(),observation(durable_mib_per_second=60,upload_seconds_per_gib=9,network_saturated=True),POLICY)
    assert new['max_active']==80 and action.startswith('backoff')


def test_two_saturated_plateaus_throttle_and_respect_guaranteed_floor():
    new,action=govern(state(),observation(network_saturated=True,registered_workers=28),POLICY)
    assert action is None
    new,action=govern(new,observation(time=800,network_saturated=True,registered_workers=28),POLICY)
    assert new['max_active']==25 and action.startswith('throttle')
    cooled,action=govern(new,observation(time=1200,network_saturated=True,registered_workers=28,waiting_readbacks=3),POLICY)
    assert action is None and cooled['max_readbacks']==new['max_readbacks']


def test_waiting_readbacks_probe_readback_limit_only():
    new,action=govern(state(),observation(waiting_readbacks=3),POLICY)
    assert new['max_readbacks']==20 and new['max_uploads']==8 and action=='probe-max_readbacks'


def test_insufficient_sample_leaves_full_request_unchanged():
    new,action=govern(state(),observation(completed_sources=1,errors=1),POLICY)
    assert new==state() and action is None


def hold_slot(directory,limits,queue,seconds):
    from replay_io_control import controlled_slot
    with controlled_slot(directory,limits,'max_uploads',1):
        queue.put(time.monotonic());time.sleep(seconds)


def test_slots_serialize_independent_processes(tmp_path):
    context=multiprocessing.get_context('spawn');queue=context.Queue()
    limits=tmp_path/'limits.json';limits.write_text(json.dumps({'max_uploads':1}))
    a=context.Process(target=hold_slot,args=(str(tmp_path/'slots'),str(limits),queue,.5));a.start();first=queue.get(timeout=10)
    b=context.Process(target=hold_slot,args=(str(tmp_path/'slots'),str(limits),queue,.1));b.start();second=queue.get(timeout=10)
    a.join(10);b.join(10)
    assert a.exitcode==b.exitcode==0 and second-first>=.45


def test_waiter_observes_raised_limit_without_worker_restart(tmp_path):
    context=multiprocessing.get_context('spawn');queue=context.Queue()
    limits=tmp_path/'limits.json';limits.write_text(json.dumps({'max_uploads':1}))
    a=context.Process(target=hold_slot,args=(str(tmp_path/'slots'),str(limits),queue,2));a.start();first=queue.get(timeout=10)
    b=context.Process(target=hold_slot,args=(str(tmp_path/'slots'),str(limits),queue,.1));b.start()
    time.sleep(.4);temporary=tmp_path/'new.json';temporary.write_text(json.dumps({'max_uploads':2}));temporary.replace(limits)
    second=queue.get(timeout=10);a.join(10);b.join(10)
    assert a.exitcode==b.exitcode==0 and second-first<1.8
