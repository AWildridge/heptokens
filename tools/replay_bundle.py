"""Pack immutable replay files and verify both tar bytes and individual members.

The external JSON receipt is the trust anchor: retain it independently of the tar.
This utility does not change replay commits or delete source files.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import os

MANIFEST = 'BUNDLE-MANIFEST.json'


def digest(stream):
    sha = hashlib.sha256()
    size = 0
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        sha.update(block)
        size += len(block)
    return {'bytes': size, 'sha256': sha.hexdigest()}


def checked_name(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or str(p) != name or name == MANIFEST:
        raise ValueError('Unsafe or reserved bundle member: ' + name)
    return name


def pack(source, inventory, archive, receipt):
    source, archive, receipt = Path(source).resolve(), Path(archive), Path(receipt)
    if archive.exists() or receipt.exists():
        raise FileExistsError('Bundle or receipt already exists')
    if not inventory:
        raise ValueError('Empty inventory')
    members = {}
    for name, expected in sorted(inventory.items()):
        checked_name(name)
        path = source / name
        if not path.resolve().is_relative_to(source) or any(p.is_symlink() for p in [path, *path.parents] if p != source):
            raise ValueError('Bundle input is a symbolic link or escapes source')
        with path.open('rb') as f:
            actual = digest(f)
        if any(actual[k] != expected[k] for k in ['bytes', 'sha256']):
            raise ValueError('Input differs from validated inventory: ' + name)
        members[name] = actual
    manifest = {'format': 'heptokens-replay-tar-v1', 'files': members}
    # Exclusive creation prevents replacement of an existing validated bundle.
    temporary = Path(str(archive) + '.packing')
    with temporary.open('xb') as output, tarfile.open(fileobj=output, mode='w') as tar:
        for name, expected in members.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = expected['bytes'], 0o644, 0
            with (source / name).open('rb') as f:
                tar.addfile(info, f)
        data = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
        info = tarfile.TarInfo(MANIFEST)
        info.size, info.mode, info.mtime = len(data), 0o644, 0
        tar.addfile(info, io.BytesIO(data))
    with temporary.open('rb') as f:
        report = dict(manifest, archive=digest(f))
    # Read back every member before publishing the receipt as a valid bundle.
    verify(temporary, report)
    os.link(temporary, archive)
    temporary.unlink()
    receipt_temporary = Path(str(receipt) + '.packing')
    with receipt_temporary.open('x') as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.link(receipt_temporary, receipt)
    receipt_temporary.unlink()
    return report


def verify(archive, receipt):
    archive = Path(archive)
    if receipt.get('format') != 'heptokens-replay-tar-v1':
        raise ValueError('Unsupported bundle format')
    with archive.open('rb') as f:
        if digest(f) != receipt['archive']:
            raise ValueError('Whole archive checksum or size differs')
    expected = receipt['files']
    for name in expected:
        checked_name(name)
    seen = set()
    internal = None
    with tarfile.open(archive, 'r:') as tar:
        for member in tar:
            name = member.name
            if name in seen or not member.isfile():
                raise ValueError('Duplicate or non-regular tar member')
            seen.add(name)
            if name == MANIFEST:
                if member.size > 16 * 1024 * 1024:
                    raise ValueError('Oversized manifest')
                internal = json.load(tar.extractfile(member))
            elif checked_name(name) not in expected:
                raise ValueError('Unexpected tar member')
            elif digest(tar.extractfile(member)) != expected[name]:
                raise ValueError('Member checksum or size differs: ' + name)
    if seen != set(expected) | {MANIFEST}:
        raise ValueError('Missing tar members')
    if internal != {'format': receipt['format'], 'files': expected}:
        raise ValueError('Internal and external manifests differ')
    return {'status': 'passed', 'members': len(expected), 'archive': receipt['archive']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['pack', 'verify'])
    parser.add_argument('--archive', required=True)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--source')
    parser.add_argument('--inventory')
    args = parser.parse_args()
    if args.mode == 'pack':
        if not args.source or not args.inventory:
            parser.error('pack requires --source and --inventory')
        result = pack(args.source, json.loads(Path(args.inventory).read_text()), args.archive, args.receipt)
    else:
        result = verify(args.archive, json.loads(Path(args.receipt).read_text()))
    print(json.dumps(result, indent=2))
