"""Release future-growth bookkeeping only for absent, unlocked replay attempts.

Physical leftovers remain in statvfs. This never removes source or output files.
Visible attempts retain their full booking so existing completed work can resume
its canonical transfer even when a fresh source cannot be admitted.
"""
import fcntl
import json
from pathlib import Path
import re
import time


def reconcile_absent_attempts(shared, phase, visible_scratch, release, grace_seconds=120, now=None):
    shared=Path(shared)
    visible=set(map(str,visible_scratch))
    now=time.time() if now is None else now
    report={'released':[],'retained':[],'physical_files_deleted':False}
    for path in sorted((shared/'attempts').glob('*/*/status.json')):
        record=json.loads(path.read_text())
        if record.get('phase')!=phase or record.get('validation_limit') is not None:
            continue
        task=record['task'];attempt=record['attempt'];reservation=record['reservation']
        if (type(task) is not int or not 0<=task<4546 or not re.fullmatch('[0-9a-f]{32}',attempt)
            or reservation['key']!='%05d-%s'%(task,attempt)
            or not re.fullmatch('[0-9a-f]{64}',reservation['node'])):
            raise ValueError('Unexpected replay reservation identity')
        ledger=shared/'node-disk'/reservation['node']/'reservations.json'
        if not ledger.exists() or reservation['key'] not in json.loads(ledger.read_text()):
            continue
        reason=None
        if record['scratch'] in visible:reason='scratch-still-visible'
        elif now-max(record.get('started',0),record.get('finished',0))<grace_seconds:reason='recent-attempt'
        if reason:
            report['retained'].append({'task':task,'attempt':attempt,'reason':reason});continue
        lease=shared/'source-leases'/('%05d.lock'%task)
        with lease.open('a+') as stream:
            try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                report['retained'].append({'task':task,'attempt':attempt,'reason':'source-lease-held'});continue
            # Recheck mutable record after acquiring its source lease.
            current=json.loads(path.read_text())
            if current!=record:
                report['retained'].append({'task':task,'attempt':attempt,'reason':'record-changed'});continue
            release(shared/'node-disk',reservation)
            report['released'].append({'task':task,'attempt':attempt,'reservation':reservation,
                'reason':'not-visible-on-any-current-worker; source lease acquired'})
    return report
