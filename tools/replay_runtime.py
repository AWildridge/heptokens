"""Execution identity and shared transfer bounds for replay backends."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import resource
import time
import threading
import signal
import subprocess

_ledger_threads = {}
_ledger_registry_lock = threading.Lock()
_cpu_processes = set()
_cpu_process_lock = threading.Lock()


def terminate_cpu_children(signum, frame):
    """A replay timeout must also terminate the detectors' separate groups."""
    with _cpu_process_lock:
        for process in _cpu_processes:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    raise SystemExit(128 + signum)


def run_cpu_process(command, *, timeout, **kwargs):
    """Bound a whole subprocess tree, including children holding source locks."""
    if timeout <= 0:
        raise ValueError('A positive CPU subprocess timeout is required')
    with subprocess.Popen(command, start_new_session=True, **kwargs) as process:
        with _cpu_process_lock:
            _cpu_processes.add(process)
        try:
            returncode = process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            # A child can survive after the parent exits on SIGTERM.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
        finally:
            with _cpu_process_lock:
                _cpu_processes.discard(process)
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)
    return returncode


@contextmanager
def storage_ledger_lock(root):
    """Serialize threads as well as processes using the existing POSIX lock."""
    path = (Path(root) / 'storage.lock').resolve()
    with _ledger_registry_lock:
        thread_lock = _ledger_threads.setdefault(str(path), threading.Lock())
    with thread_lock, open(path, 'a+') as handle:
        fcntl.lockf(handle, fcntl.LOCK_EX)
        yield


def execution_identity():
    if os.environ.get('SLURM_JOB_ID'):
        return {'backend': 'slurm', 'id': os.environ['SLURM_JOB_ID']}
    # The launcher passes the real worker identity to its child processes.
    if os.environ.get('HEPTOKENS_EXECUTION'):
        identity = json.loads(os.environ['HEPTOKENS_EXECUTION'])
        if identity['backend'] not in ('dask', 'session') or identity['reserved_cores'] < 2:
            raise RuntimeError('Invalid CPU worker execution context')
        return identity
    from distributed import get_worker
    worker = get_worker()
    if worker.nthreads < 2:
        raise RuntimeError('Paired detector execution requires at least two reserved CPU cores')
    return {'backend': 'dask', 'id': worker.address, 'reserved_cores': worker.nthreads}


@contextmanager
def transfer_slot(directory, count=2):
    """POSIX record locks on worker-visible storage bound downloads globally."""
    directory = Path(directory)
    if count < 1:
        raise ValueError('At least one transfer slot is required')
    directory.mkdir(parents=True, exist_ok=True)
    handles = [open(directory / ('slot-%d' % i), 'a+') for i in range(count)]
    try:
        while True:
            for handle in handles:
                try:
                    fcntl.lockf(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                try:
                    yield
                finally:
                    fcntl.lockf(handle, fcntl.LOCK_UN)
                return
            time.sleep(1)
    finally:
        for handle in handles:
            handle.close()


def usage_clock():
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.monotonic(), own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


@contextmanager
def measure_stage(work, stage):
    """Record elapsed/CPU time; child RSS is a high-water mark, not total RSS."""
    started, cpu_started = usage_clock()
    status = 'failed'
    try:
        yield
        status = 'passed'
    finally:
        finished, cpu_finished = usage_clock()
        path = Path(work) / 'performance.json'
        rows = json.loads(path.read_text()) if path.exists() else []
        rows.append({'stage': stage, 'status': status,
                     'wall_seconds': finished-started, 'cpu_seconds': cpu_finished-cpu_started,
                     'largest_child_rss_kib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                     'note': 'RSS is cumulative largest child high-water, not simultaneous process-tree peak'})
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(rows, indent=2))
        temporary.replace(path)
