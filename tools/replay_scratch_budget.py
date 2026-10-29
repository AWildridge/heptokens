"""Credit pinned, immutable local bytes against a source's future disk growth.

The full peak reservation is retained. Hardlinks prevent credited bytes from
disappearing while a writer can grow. Final cleanup revokes credit under the
node ledger lock before removing files or releasing the reservation.
"""
import hashlib
import json
import os
from pathlib import Path
import uuid

from replay_runtime import storage_ledger_lock


def durable_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def physical_bytes(root):
    seen = set()
    total = 0
    for path in Path(root).rglob('*'):
        if not path.is_file():
            continue
        stat = path.stat()
        key = stat.st_dev, stat.st_ino
        if key not in seen:
            seen.add(key)
            total += stat.st_size
    return total


def credit_total(node_root, reservations):
    path = Path(node_root) / 'pinned-credits.json'
    credits = json.loads(path.read_text()) if path.exists() else {}
    total = 0
    for key, size in reservations.items():
        value = credits.get(key, 0)
        if not isinstance(value, int) or not 0 <= value <= size:
            raise ValueError('Invalid pinned scratch credit: ' + key)
        total += value
    return total


def pin_files(config, paths, expected_hashes=None):
    """Only call after canonical input validation or immutable chunk commit."""
    if not config:
        return
    expected_hashes = {str(Path(p).resolve()): value for p, value in (expected_hashes or {}).items()}
    root = Path(config['scratch']).resolve()
    pins = root / 'immutable-pins'
    pins.mkdir(exist_ok=True, mode=0o700)
    manifest_path = Path(config['manifest'])
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'files': {}}
    for source in map(Path, paths):
        if source.is_symlink():
            raise ValueError('Cannot credit a symlink')
        source = source.resolve()
        if not source.is_relative_to(root) or pins in source.parents:
            raise ValueError('Scratch credit is limited to this attempt')
        relative = str(source.relative_to(root))
        target = pins / hashlib.sha256(relative.encode()).hexdigest()
        expected = digest(source)
        if str(source) in expected_hashes and expected_hashes[str(source)] != expected:
            raise ValueError('Export checksum differs before scratch credit')
        if relative in manifest['files'] and manifest['files'][relative]['sha256'] != expected:
            raise ValueError('Pinned file differs on restart: ' + relative)
        if target.exists():
            if digest(target) != expected:
                raise ValueError('Pinned file differs on restart: ' + relative)
        else:
            os.link(source, target)
        with target.open('rb') as stream:
            os.fsync(stream.fileno())
        if digest(target) != expected:
            raise ValueError('File changed while pinning: ' + relative)
        stat = target.stat()
        manifest['files'][relative] = {'pin': str(target), 'sha256': expected,
                                      'bytes': stat.st_size, 'device': stat.st_dev, 'inode': stat.st_ino}
    unique = {}
    for item in manifest['files'].values():
        stat = Path(item['pin']).stat()
        if (stat.st_dev, stat.st_ino, stat.st_size) != (item['device'], item['inode'], item['bytes']):
            raise ValueError('Pinned file identity changed')
        # Sparse holes consume no disk; never grant credit for logical holes.
        unique[(stat.st_dev, stat.st_ino)] = min(stat.st_size, stat.st_blocks * 512)
    manifest.update(credit_bytes=sum(unique.values()), reservation=config['key'])
    node_root = Path(config['node_root'])
    with storage_ledger_lock(node_root):
        reservations = json.loads((node_root / 'reservations.json').read_text())
        if manifest['credit_bytes'] > reservations[config['key']]:
            raise ValueError('Pinned data exceeds source peak budget')
        # Hashes are durably on WORK before the admission ledger receives credit.
        durable_json(manifest_path, manifest)
        path = node_root / 'pinned-credits.json'
        credits = json.loads(path.read_text()) if path.exists() else {}
        credits[config['key']] = manifest['credit_bytes']
        durable_json(path, credits)


def release_credit_locked(node_root, key):
    path = Path(node_root) / 'pinned-credits.json'
    if path.exists():
        credits = json.loads(path.read_text())
        credits.pop(key, None)
        durable_json(path, credits)


def cleanup(shared, reservation, scratch):
    """All replay and transfer children must have finished before this call."""
    import shutil
    node_root = Path(shared) / reservation['node']
    with storage_ledger_lock(node_root):
        # A crash after revocation is conservative, including partial rmtree.
        release_credit_locked(node_root, reservation['key'])
        shutil.rmtree(scratch)
        path = node_root / 'reservations.json'
        reservations = json.loads(path.read_text())
        reservations.pop(reservation['key'], None)
        durable_json(path, reservations)
