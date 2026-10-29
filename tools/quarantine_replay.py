"""Preserve failed replay staging on EOS before freeing its admission booking.

These diagnostic commits are explicitly incomplete, never production receipts.
Reuse the production GFAL checksum/stability/full-readback implementation.
"""
import fcntl
import json
from pathlib import Path
import time

from commit_replay_gfal import digest, file_map, stage_file, verify_staged_file, publish_file, remote_adler
from replay_runtime import storage_ledger_lock


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def quarantine(roots, destination, evidence, failure, stability_seconds=30, file_workers=2):
    if stability_seconds < 1:
        raise ValueError('Positive remote stability interval required')
    roots = {key: Path(path).resolve() for key, path in roots.items()}
    if any('/' in key or key in ('', '.', '..') for key in roots):
        raise ValueError('Invalid quarantine root name')
    source = roots['source']
    source.mkdir(parents=True, exist_ok=True)
    evidence = Path(evidence)
    manifest_path = evidence.with_suffix('.manifest.json')
    with open(source/'task.lock', 'a+') as lock:
        fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest['destination'] != destination or manifest['roots'] != {k:str(v) for k,v in roots.items()}:
                raise ValueError('Quarantine identity changed')
        else:
            files = {}
            for key, root in roots.items():
                for path in sorted(root.rglob('*')):
                    if path.is_symlink():
                        raise ValueError('Symlink in failed-source staging')
                    if path.is_file() and path.name != 'task.lock':
                        name = key + '/' + path.relative_to(root).as_posix()
                        files[name] = {'bytes': path.stat().st_size, 'sha256': digest(path, 'sha256')}
            manifest = {'schema':'heptokens-failed-source-v1', 'status':'incomplete-quarantined',
                        'destination':destination, 'roots':{k:str(v) for k,v in roots.items()},
                        'failure':failure, 'files':files}
            atomic(manifest_path, manifest)
        def local(name):
            key, relative = name.split('/', 1)
            path = (roots[key]/relative).resolve()
            if not path.is_relative_to(roots[key]):
                raise ValueError('Quarantine path escaped its source root')
            return path
        if evidence.exists():
            proof = json.loads(evidence.read_text())
            if proof['manifest_sha256'] != digest(manifest_path, 'sha256'):
                raise ValueError('Quarantine proof changed')
        else:
            def upload(name):
                path = local(name)
                expected = manifest['files'][name]
                if path.stat().st_size != expected['bytes'] or digest(path, 'sha256') != expected['sha256']:
                    raise ValueError('Failed-source file changed before backup')
                return stage_file(path, destination.rstrip('/')+'/'+name)
            staged = dict(zip(manifest['files'], file_map(upload, list(manifest['files']), file_workers)))
            time.sleep(stability_seconds)
            verified = dict(zip(staged, file_map(lambda n: verify_staged_file(local(n), staged[n], stability_seconds), list(staged), file_workers)))
            final = publish_file(manifest_path, destination.rstrip('/')+'/QUARANTINE-COMMIT.json', stability_seconds)
            proof = {'status':'verified-incomplete-quarantine', 'manifest_sha256':digest(manifest_path,'sha256'),
                     'files':verified, 'remote_manifest':final}
            atomic(evidence, proof)
        # A second full remote pass is mandatory before any local deletion.
        all_remote = dict(proof['files'], **{'QUARANTINE-COMMIT.json':proof['remote_manifest']})
        if set(proof['files']) != set(manifest['files']):
            raise ValueError('Quarantine file inventory differs')
        for name, item in proof['files'].items():
            if any(item[key] != manifest['files'][name][key] for key in ['bytes','sha256']):
                raise ValueError('Quarantine payload proof differs')
        for item in all_remote.values():
            if remote_adler(item['url']) != item['adler32']:
                raise ValueError('Unstable quarantine remote file')
        time.sleep(stability_seconds)
        file_map(lambda pair: verify_staged_file(manifest_path if pair[0]=='QUARANTINE-COMMIT.json' else local(pair[0]), pair[1], stability_seconds), list(all_remote.items()), file_workers)
        # Validate all extant local files before deleting even one of them.
        for name, item in manifest['files'].items():
            path = local(name)
            if path.exists() and (path.stat().st_size != item['bytes'] or digest(path,'sha256') != item['sha256']):
                raise ValueError('Failed-source file changed before eviction')
        for name in manifest['files']:
            key, _ = name.split('/',1)
            path = local(name)
            if path.exists() and not (key == 'source' and path.suffix == '.json'):
                path.unlink()
        # Zero only this source's payload reservations, retaining remote proof.
        root = source.parent.parent
        with storage_ledger_lock(root):
            ledger_path = root/'storage-ledger.json'
            if ledger_path.exists():
                ledger = json.loads(ledger_path.read_text())
                for key in ledger['reservations']:
                    if key.split('/')[0] == source.name:
                        ledger['reservations'][key] = 0
                        ledger.setdefault('quarantined', {})[key] = proof['remote_manifest']
                atomic(ledger_path, ledger)
        proof.update(eviction='verified-and-pruned', verified_utc_epoch=time.time())
        atomic(evidence, proof)
        return proof
