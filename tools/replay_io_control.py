"""Shared dynamic I/O slots and a conservative, evidence-driven governor."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import time


@contextmanager
def controlled_slot(directory, limits, key, default):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    handles={}
    try:
        while True:
            config=json.loads(Path(limits).read_text()) if limits and Path(limits).exists() else {}
            count=int(config.get(key,default))
            if not 1<=count<=250:raise ValueError('Invalid I/O slot count')
            for number in range(count):
                if number not in handles:handles[number]=(directory/('slot-%d'%number)).open('a+')
                handle=handles[number]
                try:fcntl.lockf(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                try:yield
                finally:fcntl.lockf(handle,fcntl.LOCK_UN)
                return
            time.sleep(.25)
    finally:
        for handle in handles.values():handle.close()


def govern(state, observation, policy):
    """Return new limits; never kill tasks or infer saturation from idle CPUs."""
    state=dict(state)
    now=observation['time']
    if now-state.get('last_evaluation',0)<policy['window_seconds']:return state,None
    if observation['completed_sources']<policy['minimum_completed_sources']:
        return state,None
    prior=state.get('baseline')
    goodput=observation['durable_mib_per_second']
    action=None
    if prior and prior['durable_mib_per_second']>0:
        ratio=goodput/prior['durable_mib_per_second']
        slower=any(observation.get(k,0)>prior.get(k,0)*1.25 for k in ['upload_seconds_per_gib','readback_seconds_per_gib'] if prior.get(k,0)>0)
        if observation['errors'] or (ratio<policy['regression_ratio'] and slower and observation.get('network_saturated',False)):
            effective=min(state['max_active'],observation['registered_workers'])
            state['max_active']=max(policy['minimum_workers'],int(effective*.8))
            last=state.get('last_good_limits',{})
            for key in ['max_uploads','max_readbacks']:
                state[key]=max(policy['minimum_transfer_slots'],min(state[key],last.get(key,state[key])))
            state['plateau_windows']=0
            state['io_probe_cooldown_until']=now+3*policy['window_seconds']
            action='backoff-regression-or-transfer-errors'
        elif observation.get('network_saturated',False) and observation['archive_backlog']>=state['max_uploads'] and ratio<1.05:
            state['plateau_windows']=state.get('plateau_windows',0)+1
            if state['plateau_windows']>=2:
                effective=min(state['max_active'],observation['registered_workers'])
                state['max_active']=max(policy['minimum_workers'],int(effective*.8))
                for key,value in state.get('last_good_limits',{}).items():
                    if key in ['max_uploads','max_readbacks']:state[key]=value
                state['plateau_windows']=0
                state['io_probe_cooldown_until']=now+3*policy['window_seconds']
                action='throttle-source-admission-at-io-plateau'
        else:
            state['plateau_windows']=0
    if (action is None and now>=state.get('io_probe_cooldown_until',0) and
        (not prior or goodput>=prior['durable_mib_per_second']*.95)):
        state['last_good_limits']={k:state[k] for k in ['max_active','max_uploads','max_readbacks']}
        for key,waiter in [('max_readbacks','waiting_readbacks'),('max_uploads','waiting_uploads')]:
            if observation[waiter]>0 and state[key]<policy['maximum_transfer_slots']:
                state[key]=min(policy['maximum_transfer_slots'],state[key]+policy['transfer_step'])
                action='probe-'+key
                break
        if action is None and observation['archive_backlog']==0 and state['max_active']<policy['maximum_workers']:
            state['max_active']=min(policy['maximum_workers'],max(state['max_active']+1,int(state['max_active']*1.25)))
            action='recover-source-admission'
    state['last_evaluation']=now
    state['baseline']=observation
    return state,action
