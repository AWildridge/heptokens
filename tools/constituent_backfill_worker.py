"""Source-leased, bounded-scratch native contributor recovery transactions.

Original detector inputs and compact payloads are immutable. Groups become
complete only after every compact bit matches and two full EOS readbacks pass.
"""
import argparse
from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import tarfile
import tempfile
import time
import traceback
import ssl
import urllib.request
import uuid

from replay_io_control import controlled_slot
from replay_runtime import run_cpu_process, terminate_cpu_children


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked(path, expected):
    if Path(path).stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
        raise ValueError('Payload hash/size differs: ' + str(path))


def safe_member(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or str(path) != name:
        raise ValueError('Unsafe archive path: ' + name)
    return name


def proxy_lifetime(config):
    if not shutil.which('voms-proxy-info'):
        # Minimal worker images have no grid CLI. The AF controller additionally
        # checks the VOMS AC; workers independently guard the proxy certificate.
        try:
            value = ssl._ssl._test_decode_cert(config['proxy'])
            return max(0, int(ssl.cert_time_to_seconds(value['notAfter']) - time.time()))
        except (OSError, ValueError, ssl.SSLError):
            return 0
    result = subprocess.run(['voms-proxy-info', '-file', config['proxy'], '-timeleft', '-actimeleft'],
                            capture_output=True, text=True, timeout=15)
    try:
        values = [int(value) for value in result.stdout.split()]
        return min(values) if result.returncode == 0 and len(values) == 2 else 0
    except ValueError:
        return 0


def download(config, record, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with controlled_slot(Path(config['output']) / 'io-slots/download', config['io_limits'], 'max_downloads', 64):
        context = ssl.create_default_context(cafile=config['ca_bundle'])
        for attempt in range(4):
            try:
                with urllib.request.urlopen(record['url'].replace('davs://', 'https://', 1),
                                            context=context, timeout=180) as stream, target.open('wb') as output:
                    size = 0
                    for block in iter(lambda: stream.read(4 * 1024**2), b''):
                        size += len(block)
                        if size > record['bytes']:
                            raise ValueError('Remote input exceeds frozen size')
                        output.write(block)
                break
            except (OSError, TimeoutError):
                if attempt == 3:
                    raise
                time.sleep(3 * 2**attempt)
    checked(target, record)


def source_chunks(files, events):
    chunks = sorted({name.split('/')[0] for name in files
                     if name.startswith('chunk-') and name.endswith('/receipt.json')})
    expected = ['chunk-%06d' % i for i in range((events + 999) // 1000)]
    if chunks != expected:
        raise ValueError('Source chunk inventory is not complete and contiguous')
    return chunks


def verified_group(path, config, source, chunks):
    if not Path(path).exists():
        return None
    record = json.loads(Path(path).read_text())
    if (record['status'] != 'complete' or record['source_task'] != source['task']
            or record['chunks'] != chunks or record['config_identity'] != config['identity']):
        raise ValueError('Completed group identity differs: ' + str(path))
    proof = record['remote']
    if proof['status'] != 'two-stable-full-readbacks-passed':
        raise ValueError('Group has no stable full EOS verification')
    for key in ['sha256', 'bytes', 'adler32']:
        if proof['first'][key] != proof['second'][key] or (key in record and proof['first'][key] != record[key]):
            raise ValueError('Group remote digests differ')
    if any(len(item['checked_compact_datasets']) != 18 for item in record['views']):
        raise ValueError('Group lacks complete detector parity evidence')
    if sum(item['events'] for item in record['views']) != 2 * record['events']:
        raise ValueError('Group detector/event accounting differs')
    return record


def reserve_remote(config, key, size):
    out = Path(config['output'])
    path = out / 'reservations' / (key + '.json')
    with (out / 'remote-cap.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if json.loads(path.read_text())['bytes'] != size:
                raise ValueError('Conflicting remote byte reservation')
            return
        total_path = out / 'reserved-remote-bytes.json'
        total = json.loads(total_path.read_text())['bytes'] if total_path.exists() else 0
        if total + size > config['remote_cap_bytes']:
            raise ValueError('Configured EOS output budget exhausted')
        # An interrupted reservation may conservatively count extra bytes.
        atomic(total_path, dict(bytes=total + size))
        atomic(path, dict(bytes=size))


def publish(config, archive, directory, task, group):
    if proxy_lifetime(config) < config.get('minimum_proxy_seconds', 3600):
        raise RuntimeError('CMS proxy has less than one hour; publication waits for renewal')
    sha = digest(archive)
    url = config['remote'].rstrip('/') + '/sources/%05d/%s/%s/links.tar' % (task, group, sha)
    reserve_remote(config, '%05d-%s-%s' % (task, group, sha), archive.stat().st_size)
    request = directory / 'publish-request.json'
    proof = directory / 'eos-verified.json'
    atomic(request, dict(source=str(archive), url=url, proof=str(proof),
                         limits=config['io_limits'], slots=str(Path(config['output']) / 'io-slots')))
    env = dict(os.environ, HEPTOKENS_BACKFILL_TOOLS=config['tools_dir'])
    with (directory / 'publish.log').open('a') as log:
        run_cpu_process(['bash', config['transfer_script'], str(request)], env=env,
                        stdout=log, stderr=subprocess.STDOUT, timeout=7200)
    result = json.loads(proof.read_text())
    if result['status'] != 'two-stable-full-readbacks-passed':
        raise ValueError('Incomplete group publication')
    for stage in ['first', 'second']:
        if result[stage]['sha256'] != sha or result[stage]['bytes'] != archive.stat().st_size:
            raise ValueError('Group publication hash differs')
    return result


def replay_chunk(config, source, chunk, native, sidecars):
    from recover_delphes_constituent_links import recover
    from simulate_delphes_pair import simulate_detector
    from audit_delphes_constituent_links import audit
    receipt = json.loads((chunk / 'receipt.json').read_text())
    index = int(chunk.name.split('-')[-1])
    if (receipt['source'] != source['source'] or receipt['events'] != 1000
            or receipt['seed'] != 420000 + source['task'] * 1000 + index):
        raise ValueError('Frozen source, chunk, event count or seed differs')
    for name, expected in receipt['sha256'].items():
        if Path(name).name != name or digest(chunk / name) != expected:
            raise ValueError('Frozen chunk payload differs: ' + name)
    native.mkdir(parents=True)
    truth = native / 'input.hepmc'
    with gzip.open(chunk / 'truth.hepmc.gz', 'rb') as stream, truth.open('xb') as output:
        shutil.copyfileobj(stream, output, 4 * 1024**2)
    if digest(truth) != receipt['input_sha256']:
        raise ValueError('Decoded original detector input digest differs')
    original_validation = json.loads((chunk / 'validation.json').read_text())
    original_binaries = original_validation.get('binaries')
    runtime = select_runtime(config, original_binaries)
    validation = dict(status='simulated', input_sha256=receipt['input_sha256'],
                      seed=receipt['seed'], events=receipt['events'], detectors={},
                      binaries=runtime['binaries'], runtime=runtime['path'],
                      runtime_selection='Exact archived binary hashes' if original_binaries else 'Legacy original build; archived validation predates binary-hash recording')
    start = time.monotonic()
    records = []
    for detector in ['CMS', 'ATLAS']:
        card = 'delphes_card_' + detector + '.tcl'
        wrapper = '%s-seed%d.tcl' % (detector, receipt['seed'])
        shutil.copyfile(chunk / card, native / card)
        shutil.copyfile(chunk / wrapper, native / wrapper)
        output = native / (detector + '.root')
        command = [str(Path(runtime['path']) / 'DelphesHepMC2'), str(native / wrapper), str(output), str(truth)]
        # The canonical retry routine retains failed outputs/logs and uses the
        # same command (including input, card and seed) for every attempt.
        prior = Path.cwd()
        try:
            os.chdir(native)
            env = dict(os.environ, ROOT_INCLUDE_PATH=runtime['path'] + ':' + runtime['path'] + '/external')
            attempts = simulate_detector(command, native, detector, env)
        finally:
            os.chdir(prior)
        validation['detectors'][detector] = dict(card_sha256=digest(native / card),
                                               output_sha256=digest(output), attempts=attempts)
    atomic(native / 'validation.json', validation)
    for detector in ['CMS', 'ATLAS']:
        record = recover(native / (detector + '.root'), chunk / (detector + '.h5'),
                         chunk, detector, Path(config['delphes']), sidecars / detector)
        record['independent_serialization_audit'] = audit(sidecars / detector / 'constituent-links.root',
                                                        chunk / (detector + '.h5'))
        atomic(sidecars / detector / 'recovery.json', record)
        # Recompaction is a checked temporary; compact originals stay on EOS.
        (sidecars / detector / 'recompacted.h5').unlink()
        records.append(record)
    shutil.copyfile(native / 'validation.json', sidecars / 'native-validation.json')
    shutil.copyfile(chunk / 'receipt.json', sidecars / 'original-chunk-receipt.json')
    return records, time.monotonic() - start


def select_runtime(config, recorded):
    if recorded is None:
        return config['delphes_runtimes'][0]
    for runtime in config['delphes_runtimes']:
        if runtime['binaries'] == recorded:
            return runtime
    raise ValueError('Archived Delphes binary hashes do not match a validated available runtime')


def run_source(config, source, pilot=False):
    from replay_worker_local import admit_disk, release_disk
    task = int(source['task'])
    out = Path(config['output'])
    state_dir = out / ('pilot-sources' if pilot else 'sources') / ('%05d' % task)
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / 'source.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(status='source-busy', task=task, retryable=True)
        # Reuse the canonical shared per-node admission ledger rather than
        # letting 1000 pods independently consume the same apparent free space.
        shared_disk = out / 'node-disk'
        booking = source['objects'].get('source.tar', {}).get('bytes', 0) + 4 * 1024**3
        reservation = admit_disk(shared_disk, '%05d-' % task + uuid.uuid4().hex, booking)
        if reservation is None:
            return dict(status='scratch-busy', task=task, retryable=True)
        scratch = Path(tempfile.mkdtemp(prefix='heptokens-links-%05d-' % task))
        def phase(name, **kwargs):
            atomic(state_dir / 'progress.json', dict(phase=name, heartbeat=time.time(),
                   source_task=task, scratch=str(scratch), reservation=reservation,
                   worker=os.environ.get('HOSTNAME'), **kwargs))
        phase('admitted')
        try:
            stat = os.statvfs(scratch)
            if stat.f_bavail * stat.f_frsize < config.get('minimum_scratch_bytes', 8 * 1024**3):
                raise RuntimeError('Worker scratch has less than required free capacity')
            for path, expected in config['software'].items():
                if digest(path) != expected:
                    raise ValueError('Frozen recovery runtime differs: ' + path)
            objects = source['objects']
            archive = None
            if 'source.tar' in objects:
                phase('downloading-bundle')
                archive_path = scratch / 'source.tar'
                download(config, objects['source.tar'], archive_path)
                manifest = scratch / 'BUNDLE-MANIFEST.json'
                download(config, objects['BUNDLE-MANIFEST.json'], manifest)
                bundle = json.loads(manifest.read_text())
                if bundle['archive'] != {k: objects['source.tar'][k] for k in ['bytes', 'sha256']}:
                    raise ValueError('Bundle and trusted inventory hashes differ')
                files = bundle['files']
                archive = tarfile.open(archive_path, 'r:')
                members = {member.name: member for member in archive.getmembers()}
                if len(members) != len(archive.getmembers()) or set(members) != set(files) | {'BUNDLE-MANIFEST.json'}:
                    raise ValueError('Bundle member inventory differs')
                if any(not m.isfile() or safe_member(m.name) != m.name for m in members.values()):
                    raise ValueError('Non-regular or unsafe bundle member')
            else:
                files = objects
            chunks = source_chunks(files, source['events'])
            if pilot:
                chunks = chunks[:1]
            group_size = config.get('chunks_per_group', 5)
            completed = []
            for begin in range(0, len(chunks), group_size):
                group_chunks = chunks[begin:begin + group_size]
                group = 'group-%06d' % (begin // group_size)
                durable = state_dir / 'groups' / group / 'complete.json'
                if not pilot:
                    previous = verified_group(durable, config, source, group_chunks)
                    if previous:
                        completed.append(previous)
                        continue
                    if proxy_lifetime(config) < config.get('minimum_proxy_seconds', 3600):
                        raise RuntimeError('Proxy expired or expiring; source will resume after renewal')
                group_dir = scratch / group
                payload = group_dir / 'payload'
                payload.mkdir(parents=True)
                views = []
                phase('replaying', group=group, completed_groups=len(completed))
                for name in group_chunks:
                    chunk = group_dir / 'input' / name
                    chunk.mkdir(parents=True)
                    for filename, expected in files.items():
                        if not filename.startswith(name + '/'):
                            continue
                        relative = safe_member(filename)
                        destination = group_dir / 'input' / relative
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if archive:
                            with archive.extractfile(members[filename]) as stream, destination.open('xb') as output:
                                shutil.copyfileobj(stream, output, 4 * 1024**2)
                            checked(destination, expected)
                        else:
                            download(config, expected, destination)
                    records, wall = replay_chunk(config, source, chunk,
                                                 group_dir / 'native' / name, payload / name)
                    for record in records:
                        views.append(dict(chunk=name, detector=record['detector'], events=record['counters']['events'],
                                          counters=record['counters'], sidecar_sha256=record['sidecar']['sha256'],
                                          sidecar_bytes=record['sidecar']['bytes'],
                                          independent_serialization_audit=record['independent_serialization_audit'],
                                          checked_compact_datasets=record['checked_compact_datasets']))
                    phase('replaying', group=group, latest_chunk=name, chunk_wall_seconds=wall,
                          completed_groups=len(completed))
                events = sum(v['events'] for v in views) // 2
                record = dict(status='locally-validated', source_task=task, chunks=group_chunks,
                              events=events, views=views, config_identity=config['identity'],
                              input_source_identity=source['source'],
                              input_inventory_sha256=config['inventory_sha256'])
                atomic(payload / 'GROUP.json', record)
                target = group_dir / 'links.tar'
                with tarfile.open(target, 'x:') as tar:
                    for path in sorted(payload.rglob('*')):
                        if path.is_file():
                            tar.add(path, arcname=str(path.relative_to(payload)), recursive=False)
                if pilot:
                    destination = state_dir / ('pilot-links-' + config['identity'][:12] + '.tar')
                    shutil.copyfile(target, destination)
                    checked(destination, dict(bytes=target.stat().st_size, sha256=digest(target)))
                    record.update(status='pilot-passed-local-only', archive=str(destination),
                                  sha256=digest(destination), bytes=destination.stat().st_size)
                    atomic(state_dir / 'pilot-complete.json', record)
                    # Keep native pilot outputs until sidecars pass EOS verification.
                    phase('pilot-passed-local-only', proof=str(state_dir / 'pilot-complete.json'))
                    return record
                phase('publishing', group=group, completed_groups=len(completed))
                remote = publish(config, target, group_dir, task, group)
                record.update(status='complete', remote=remote, sha256=digest(target), bytes=target.stat().st_size,
                              completed_at=time.time())
                atomic(durable, record)
                verified_group(durable, config, source, group_chunks)
                completed.append(record)
                # Only now are native ROOTs, readback scratch and local sidecars redundant.
                shutil.rmtree(group_dir)
            if archive:
                archive.close()
            if sum(g['events'] for g in completed) != source['events']:
                raise ValueError('Full source event coverage differs')
            result = dict(status='complete', task=task, events=source['events'],
                          groups=len(completed), config_identity=config['identity'],
                          source=source['source'], completed_at=time.time(),
                          group_receipts=[str(state_dir / 'groups' / ('group-%06d' % i) / 'complete.json')
                                          for i in range(len(completed))])
            atomic(state_dir / 'complete.json', result)
            phase('complete', events=source['events'])
            shutil.rmtree(scratch)
            release_disk(shared_disk, reservation)
            return result
        except BaseException as error:
            phase('failed', error_type=type(error).__name__, error=str(error))
            atomic(state_dir / ('failure-' + uuid.uuid4().hex + '.json'),
                   dict(time=time.time(), scratch=str(scratch), error=traceback.format_exc()))
            # Never remove unverified native or sidecar payloads on failure.
            raise


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, terminate_cpu_children)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    try:
        result = run_source(request['config'], request['source'], request.get('pilot', False))
    except Exception as error:
        result = dict(status='failed', task=request['source']['task'],
                      retryable=not isinstance(error, (ValueError, AssertionError)),
                      error_type=type(error).__name__, error=traceback.format_exc())
    atomic(request['result'], result)
