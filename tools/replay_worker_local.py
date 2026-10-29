"""Run the canonical replay on disposable worker disk, retaining proof on shared storage."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from replay_cms_collection import atomic_json, checked_copy, reserve, sha
from replay_runtime import run_cpu_process, storage_ledger_lock

LOADED_SOURCE_SHA256 = sha(__file__)

GIB = 1024**3


def disk_snapshot(path='/tmp'):
    stat = os.statvfs(path)
    return {'node': hashlib.sha256(Path('/proc/sys/kernel/random/boot_id').read_bytes()).hexdigest(),
            'total': stat.f_blocks*stat.f_frsize, 'available': stat.f_bavail*stat.f_frsize}


def admit_disk(shared, key, booking, snapshot=None, pinned_credits=False):
    snapshot = disk_snapshot() if snapshot is None else snapshot
    root = Path(shared)/snapshot['node']; root.mkdir(parents=True, exist_ok=True)
    with storage_ledger_lock(root):
        path = root/'reservations.json'
        data = json.loads(path.read_text()) if path.exists() else {}
        if pinned_credits:
            # Credits and free space must describe the same locked ledger state.
            snapshot = disk_snapshot()
            if snapshot['node'] != root.name:
                raise ValueError('Scratch node changed during admission')
        floor = max(15*GIB, snapshot['total']//10)
        # Legacy mode charges the full reservation. Pinned mode credits only
        # immutable allocated bytes retained until coordinated cleanup.
        credit = 0
        if pinned_credits:
            from replay_scratch_budget import credit_total
            credit = credit_total(root, data)
        if key not in data and sum(data.values())-credit+booking+floor > snapshot['available']:
            return None
        if key in data and data[key] != booking:
            raise ValueError('Worker scratch reservation changed')
        data[key] = booking; atomic_json(path, data)
        return {'node': snapshot['node'], 'key': key, 'bytes': booking, 'free_floor': floor,
                'available_at_admission': snapshot['available']}


def release_disk(shared, reservation):
    root = Path(shared)/reservation['node']
    with storage_ledger_lock(root):
        path = root/'reservations.json'; data = json.loads(path.read_text())
        from replay_scratch_budget import release_credit_locked
        release_credit_locked(root, reservation['key'])
        data.pop(reservation['key'], None); atomic_json(path, data)


def transfer_source(source, destination, evidence, prefix=False, workers=2):
    """Executed in the transfer environment, separate from ROOT libraries."""
    from commit_replay_gfal import commit_source, prune_source_payloads
    source, evidence = Path(source), Path(evidence)
    evidence.mkdir(parents=True, exist_ok=True)
    expected = {str(p.relative_to(source)): {'bytes': p.stat().st_size, 'sha256': sha(p),
                'url': destination.rstrip('/')+'/'+str(p.relative_to(source))}
                for p in source.rglob('*') if p.is_file() and p.name != 'task.lock'}
    if not (evidence/'expected-files.json').exists():
        atomic_json(evidence/'expected-files.json', expected)
    proof_path = source/'remote-source-copy.json'
    proof = json.loads(proof_path.read_text()) if proof_path.exists() else commit_source(
        source, destination, allow_prefix=prefix, file_workers=workers)
    # Persist expected hashes and the first full readback before any eviction.
    atomic_json(evidence/'remote-source-copy.json', proof)
    metadata = ['COMPLETE.json', 'export-index.json', 'SOURCE-COMMIT.json']
    metadata += [str(p.relative_to(source)) for p in source.glob('chunk-*/receipt.json')]
    metadata += [str(p.relative_to(source)) for p in source.glob('export-*.log')]
    for name in metadata:
        checked_copy(source/name, evidence/name)
    # Canonical pruning writes the second proof on WORK before deleting data.
    eviction = prune_source_payloads(source, file_workers=workers,
                                    audit_path=evidence/'second-readback.json')
    atomic_json(evidence/'local-eviction.json', eviction)
    return proof


def source_lease_name(name, validation_limit, namespace=None):
    if not validation_limit:return name
    if namespace:
        if not isinstance(namespace,str) or len(namespace)>32 or any(not(c.isascii() and (c.isalnum() or c in '_-')) for c in namespace):
            raise ValueError('Invalid validation lease namespace')
        return 'validation-'+namespace+'-'+name
    return 'validation-'+name


def run_worker_source(task, settings, identity, validation_limit=None):
    """Site-neutral orchestration; export, physics and schema stay canonical."""
    from distributed import get_worker
    worker = get_worker()
    root, tools, campaign = map(Path, [settings['root'], settings['tools'], settings['campaign']])
    if worker.state.nthreads < 4:
        raise RuntimeError('Four reserved CPU cores required')
    expected_helper = identity.get(str(Path(__file__).relative_to(root)))
    if expected_helper and expected_helper != LOADED_SOURCE_SHA256:
        raise ValueError('Worker helper module cache differs; reload validated module')
    for name, expected in identity.items():
        if sha(root/name) != expected:
            raise ValueError('Production identity changed: '+name)
    phase = settings['phase']; attempt = uuid.uuid4().hex
    name = '%05d' % task
    shared = Path(settings['shared']); shared.mkdir(parents=True, exist_ok=True)
    leases = shared/'source-leases'; leases.mkdir(exist_ok=True)
    with (leases/(source_lease_name(name,validation_limit,settings.get('validation_namespace'))+'.lock')).open('a+') as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'status': 'source-lease-busy', 'task': task}
        # Reuse intact local attempts. Different workers get unique EOS attempt
        # URLs, so reproduction never overwrites an interrupted upload.
        for prior in sorted((shared/'attempts'/name).glob('*/status.json'), key=lambda p:p.stat().st_mtime, reverse=True):
            saved = json.loads(prior.read_text())
            if saved.get('phase') != phase or saved.get('validation_limit') != validation_limit:
                continue
            if saved['worker'] == worker.address and Path(saved['scratch']).exists():
                attempt = saved['attempt']; break
            if saved['worker'] not in settings.get('live_workers', [saved['worker']]):
                # We hold the inherited source lease: no old replay/archive child
                # can still own this source. Physical leftovers remain reflected
                # in statvfs even after a conservative virtual booking is freed.
                release_disk(shared/'node-disk', saved['reservation'])
        key = name+'-'+attempt
        manifest = json.loads((campaign/'cms-qcd-source-manifest.json').read_text())
        entry = manifest['files'][task]
        booking = (6 if settings.get('transfer_format') == 'bundle' else 4)*entry['bytes']+GIB
        reservation = admit_disk(shared/'node-disk', key, booking, pinned_credits=settings.get('pinned_scratch_credits', False))
        if reservation is None:
            return {'status': 'scratch-busy', 'task': task, 'node': disk_snapshot()['node']}
        scratch = Path('/tmp')/('heptokens-'+phase+'-'+key)
        scratch.mkdir(mode=0o700, exist_ok=True)
        attempt_dir = shared/'attempts'/name/attempt; attempt_dir.mkdir(parents=True, exist_ok=True)
        record = {'status': 'running', 'phase': phase, 'task': task, 'attempt': attempt, 'worker': worker.address,
                  'scratch': str(scratch), 'reservation': reservation, 'started': time.time(),
                  'validation_limit': validation_limit, 'production_events_claimed': 0}
        atomic_json(attempt_dir/'status.json', record)
        try:
            layout = dict(settings['layout'], durable_replay_root=str(scratch/'data/committed'),
                          scratch_replay_root=str(scratch/'data/scratch'), storage_cap_bytes=booking,
                          max_source_staging_bytes=booking, max_export_bytes=3*entry['bytes'],
                          source_overhead_bytes=GIB, transfer_slots=str(shared/'download-slots'),
                          max_downloads=settings['max_downloads'])
            command = list(layout['cmssw_command'])
            # Make /tmp visible inside the pinned CMSSW userspace.
            command[command.index('-w'):command.index('-w')] = ['-b', '/tmp']
            command[command.index('-w')+1] = str(scratch)
            command = [('HEPTOKENS_CMSSW_WORK='+str(scratch)) if arg.startswith('HEPTOKENS_CMSSW_WORK=') else arg for arg in command]
            layout['cmssw_command'] = command
            if settings.get('pinned_scratch_credits'):
                layout['scratch_accounting'] = {'scratch': str(scratch), 'key': key,
                    'node_root': str(shared/'node-disk'/reservation['node']),
                    'manifest': str(attempt_dir/'evidence/pinned-files.json')}
            layout_path = attempt_dir/'layout.json'; atomic_json(layout_path, layout)
            run_id = phase+'-'+attempt
            durable = Path(layout['durable_replay_root'])
            if validation_limit:
                durable = durable.parent/('pilot-'+run_id+'-limit'+str(validation_limit))
            source = durable/'sources'/name
            env = dict(os.environ, HEPTOKENS_EXECUTION=json.dumps({'backend': 'dask', 'id': worker.address,
                       'reserved_cores': worker.state.nthreads}), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='', TMPDIR=str(scratch))
            env['PATH'] = str(Path(layout['python']).parent)+':'+env['PATH']
            env['LD_LIBRARY_PATH'] = ''; env['PYTHONPATH'] = ''
            replay = [layout['python'], str(tools/'replay_cms_collection.py'), '--campaign', str(campaign),
                      '--layout', str(layout_path), '--task', str(task), '--run-id', run_id]
            if validation_limit:
                replay += ['--validation-limit', str(validation_limit)]
            if not (source/'remote-source-copy.json').exists():
                with (attempt_dir/'replay.log').open('a') as log:
                    run_cpu_process(replay, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=12*3600, pass_fds=(lease.fileno(),))
            record['status'] = 'archiving'; atomic_json(attempt_dir/'status.json', record)
            complete = json.loads((source/'COMPLETE.json').read_text())
            payload = sum(p.stat().st_size for p in source.rglob('*') if p.is_file())
            budget = Path(settings['remote_budget']); budget.mkdir(parents=True, exist_ok=True)
            with storage_ledger_lock(budget):
                ledger = budget/'storage-ledger.json'
                prior_budget = json.loads(ledger.read_text())['reservations'].get(key) if ledger.exists() else None
            reserve(budget, key, prior_budget if prior_budget is not None else payload+4*1024**2, settings['remote_cap_bytes'])
            destination = settings['remote'].rstrip('/')+'/'+('validation' if validation_limit else 'attempts')+'/'+phase+'/'+name+'/'+attempt
            if settings.get('pinned_scratch_credits'):
                from replay_scratch_budget import physical_bytes
                archive_bound = physical_bytes(scratch)+2*(payload+4*1024**2)
                if archive_bound > booking:
                    raise ValueError('Archive peak exceeds reserved scratch; retained without transfer')
                record['archive_peak_bound_bytes'] = archive_bound
            record.update(destination=destination, events=complete['exported_events'], payload_bytes=payload)
            atomic_json(attempt_dir/'status.json', record)
            transfer_code = 'import sys;sys.path.insert(0,sys.argv[1]);from replay_worker_local import transfer_source;from replay_runtime import transfer_slot\nwith transfer_slot(sys.argv[6],int(sys.argv[7])):transfer_source(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]=="True")'
            if settings.get('transfer_format') == 'bundle':
                transfer_code = 'import sys;sys.path.insert(0,sys.argv[1]);from replay_remote_bundle import transfer_source;from replay_io_control import controlled_slot\nwith controlled_slot(sys.argv[6],sys.argv[8],"max_archives",int(sys.argv[7])):transfer_source(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]=="True")'
            transfer = ['bash', '-c', 'source "$1" || exit; shift; exec "$@"', 'worker-local-transfer',
                        settings['transfer_env'], settings['transfer_python'], '-c', transfer_code,
                        str(tools), str(source), destination, str(attempt_dir/'evidence'), str(bool(validation_limit)), str(shared/'eos-slots'), str(settings['max_archives']), settings.get('io_control','')]
            evidence = attempt_dir/'evidence'; evidence.mkdir(parents=True, exist_ok=True)
            performance = Path(layout['scratch_replay_root'])/'work'/name/'performance.json'
            if validation_limit:
                performance = Path(layout['scratch_replay_root']).parent/('pilot-'+run_id+'-limit'+str(validation_limit))/'work'/name/'performance.json'
            if performance.exists(): checked_copy(performance, evidence/'performance.json')
            archive_started = time.time()
            with (attempt_dir/'transfer.log').open('a') as log:
                run_cpu_process(transfer, env=dict(os.environ, TMPDIR=str(scratch), HEPTOKENS_GFAL_BACKEND=settings.get('gfal_backend','cli'), HEPTOKENS_IO_LIMITS=settings.get('io_control',''), HEPTOKENS_IO_SLOTS=str(shared/'bundle-io-slots'), HEPTOKENS_IO_METRICS=str(attempt_dir/'evidence/io-operations.jsonl')), stdout=log,
                                stderr=subprocess.STDOUT, timeout=6*3600, pass_fds=(lease.fileno(),))
            proof = json.loads((attempt_dir/'evidence/remote-source-copy.json').read_text())
            second = json.loads((attempt_dir/'evidence/second-readback.json').read_text())
            if second['status'] != 'verified-before-local-eviction':
                raise ValueError('Second readback audit missing')
            record.update(archive_wall_seconds=time.time()-archive_started, archive_started=archive_started, status='validated-prefix-on-eos' if validation_limit else 'verified-on-eos',
                          remote_manifest=proof['remote_manifest'], commit_wall_seconds=proof['commit_wall_seconds'],
                          eviction=json.loads((attempt_dir/'evidence/local-eviction.json').read_text()),
                          finished=time.time(), production_events_claimed=0 if validation_limit else complete['exported_events'])
            # All per-file expected hashes and both readback passes are now on WORK.
            atomic_json(attempt_dir/'status.json', record)
            if not validation_limit:
                target = Path(settings['committed'])/'sources'/name
                for path in (attempt_dir/'evidence').rglob('*'):
                    if path.is_file(): checked_copy(path, target/path.relative_to(attempt_dir/'evidence'))
                atomic_json(Path(settings['results'])/('production-archive-'+name+'.json'), record)
            if settings.get('pinned_scratch_credits'):
                from replay_scratch_budget import cleanup
                cleanup(shared/'node-disk', reservation, scratch)
            else:
                shutil.rmtree(scratch)
                release_disk(shared/'node-disk', reservation)
            record['scratch_cleanup'] = 'after WORK proof and two complete remote readbacks'
            atomic_json(attempt_dir/'status.json', record)
            return record
        except BaseException as error:
            record.update(status='failed-retained-scratch', error=repr(error), finished=time.time())
            atomic_json(attempt_dir/'status.json', record)
            # Keep the booking while this pod still contains failed intermediates.
            raise
