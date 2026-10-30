"""Conservative per-node admission for worker ephemeral scratch, not user quotas.

Workers on a node share the backing disk. Boot identity groups their reservations;
each reservation is backed by a process-owned flock and vanishes after a crash.
Physical free bytes are checked in addition to all outstanding reservations,
conservatively double-counting bytes already written by active jobs.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import uuid


@contextmanager
def admit(config):
    boot=Path('/proc/sys/kernel/random/boot_id').read_bytes()
    node=hashlib.sha256(boot).hexdigest()[:24]
    directory=Path(config['output'],'scratch-reservations',node)
    directory.mkdir(parents=True,exist_ok=True)
    token=uuid.uuid4().hex
    amount=2*config.get('max_output_bytes',4000000000)+1024**3
    safety=config.get('scratch_safety_bytes',4*1024**3)
    lease_path=directory/(token+'.lock');metadata=directory/(token+'.json')
    admitted=False
    with lease_path.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        with (directory/'admission.lock').open('a+') as admission:
            fcntl.flock(admission,fcntl.LOCK_EX)
            reserved=0
            for path in directory.glob('*.json'):
                other=path.with_suffix('.lock')
                with other.open('a+') as probe:
                    try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        try:reserved+=json.loads(path.read_text())['bytes']
                        except FileNotFoundError:pass  # older owner finished cleanup
                    else:
                        path.unlink(missing_ok=True);other.unlink(missing_ok=True)
            st=os.statvfs('/tmp')
            free=st.f_bavail*st.f_frsize
            if free>=reserved+amount+safety:
                metadata.write_text(json.dumps(dict(bytes=amount)))
                admitted=True
        try:yield admitted
        finally:
            with (directory/'admission.lock').open('a+') as admission:
                fcntl.flock(admission,fcntl.LOCK_EX)
                metadata.unlink(missing_ok=True)
                lease_path.unlink(missing_ok=True)
