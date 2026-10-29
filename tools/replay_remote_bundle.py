"""Immutable source tar commits with two audited remote readbacks before eviction.

Physics receipts and source validation are shared with the file-based transport.
The archive is a transport envelope; extracted payload bytes remain unchanged.
"""
import fcntl
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import time

import commit_replay_gfal as remote
from replay_bundle import pack, verify, checked_name, digest as stream_digest
from replay_source_validation import validate_source_inventory
from replay_runtime import storage_ledger_lock

SCHEMA = 'heptokens-remote-source-bundle-v1'


def atomic(path, value):
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def objects(source, proof):
    manifest_path = source / 'SOURCE-COMMIT.json'
    if proof['status'] != 'verified-remote-source' or remote.digest(manifest_path, 'sha256') != proof['remote_manifest']['sha256']:
        raise ValueError('Invalid bundled source commit proof')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != SCHEMA or manifest['source']['task'] != int(source.name):
        raise ValueError('Bundle source identity differs')
    if manifest['files'] != proof['files']:
        raise ValueError('Bundle payload inventory differs from proof')
    complete=json.loads((source/'COMPLETE.json').read_text())
    for key in ['source','source_events','exported_events','validation_limit','manifest_sha256','chunks']:
        if complete[key]!=manifest[key]:raise ValueError('Bundle source accounting changed')
    for name,item in manifest['files'].items():
        checked_name(name)
        if '/' not in name or name.endswith('/receipt.json'):
            path=source/name
            if not path.is_file() or path.stat().st_size!=item['bytes'] or remote.digest(path,'sha256')!=item['sha256']:
                raise ValueError('Local source metadata or receipt differs from bundle')
    for key in ['archive', 'bundle_receipt']:
        expected = {k: proof[key][k] for k in ['url', 'bytes', 'sha256', 'adler32']}
        if manifest[key] != expected:
            raise ValueError('Bundle object differs from final manifest')
    receipt = source / 'BUNDLE-MANIFEST.json'
    if remote.digest(receipt, 'sha256') != proof['bundle_receipt']['sha256']:
        raise ValueError('Local bundle receipt changed')
    bundle = json.loads(receipt.read_text())
    if bundle['files'] != {name: {k: item[k] for k in ['bytes', 'sha256']} for name, item in manifest['files'].items()}:
        raise ValueError('Bundle member hashes differ from source commit')
    if bundle['archive'] != {k: proof['archive'][k] for k in ['bytes', 'sha256']}:
        raise ValueError('Bundle archive hash differs from source commit')
    return manifest, bundle, {'source.tar': proof['archive'], 'BUNDLE-MANIFEST.json': proof['bundle_receipt'], 'SOURCE-COMMIT.json': proof['remote_manifest']}


def commit_bundle(source, destination, evidence, stability_seconds=30, allow_prefix=False):
    source, evidence = Path(source), Path(evidence)
    if stability_seconds < 1:
        raise ValueError('Positive stability interval required')
    began = time.monotonic()
    with (source / 'task.lock').open('a+') as lock:
        fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        complete, index, names, count = validate_source_inventory(source, allow_prefix, remote.digest)
        inventory = {name: {'bytes': (source/name).stat().st_size, 'sha256': remote.digest(source/name, 'sha256'),
                            'adler32': remote.digest(source/name, 'adler32')} for name in names}
        # Expected member hashes are on shared WORK before any network write.
        atomic(evidence/'expected-files.json', inventory)
        archives = source.parent.parent / 'bundles'
        archives.mkdir(exist_ok=True)
        archive = archives / (source.name + '.tar')
        receipt = source / 'BUNDLE-MANIFEST.json'
        # A dead pack process can leave only disposable envelope bytes. The
        # original source has just passed complete receipt validation above.
        for leftover in [Path(str(archive)+'.packing'),Path(str(receipt)+'.packing')]:
            leftover.unlink(missing_ok=True)
        if archive.exists() and not receipt.exists():
            with archive.open('rb') as stream:archive_digest=stream_digest(stream)
            recovered={'format':'heptokens-replay-tar-v1','files':{n:{k:v[k] for k in ['bytes','sha256']} for n,v in inventory.items()},'archive':archive_digest}
            verify(archive,recovered)
            atomic(receipt,recovered)
        if archive.exists() or receipt.exists():
            if not archive.exists() or not receipt.exists():
                raise ValueError('Interrupted pack retained; inspect incomplete bundle')
            bundle = json.loads(receipt.read_text())
            verify(archive, bundle)
            if bundle['files'] != {n: {k: v[k] for k in ['bytes', 'sha256']} for n,v in inventory.items()}:
                raise ValueError('Prior bundle inventory differs')
        else:
            bundle = pack(source, inventory, archive, receipt)
        atomic(evidence/'BUNDLE-MANIFEST.json', bundle)
        urls = {archive: destination.rstrip('/')+'/source.tar', receipt: destination.rstrip('/')+'/BUNDLE-MANIFEST.json'}
        expected_objects = {url: {'bytes': path.stat().st_size, 'sha256': remote.digest(path,'sha256')} for path,url in urls.items()}
        atomic(evidence/'expected-objects.json', expected_objects)
        staged = {path: remote.stage_file(path,url) for path,url in urls.items()}
        time.sleep(stability_seconds)
        first = {path: remote.verify_staged_file(path,p,stability_seconds) for path,p in staged.items()}
        manifest = {'schema': SCHEMA, 'status': 'complete', 'source': complete['source'],
                    'source_events': complete['source_events'], 'exported_events': count,
                    'validation_limit': complete['validation_limit'], 'manifest_sha256': complete['manifest_sha256'],
                    'chunks': complete['chunks'], 'files': inventory,
                    'archive': {k: first[archive][k] for k in ['url','bytes','sha256','adler32']},
                    'bundle_receipt': {k: first[receipt][k] for k in ['url','bytes','sha256','adler32']}}
        path = source/'SOURCE-COMMIT.json'
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError('Existing source commit differs')
        atomic(path, manifest)
        expected_objects[destination.rstrip('/')+'/SOURCE-COMMIT.json'] = {'bytes': path.stat().st_size, 'sha256': remote.digest(path,'sha256')}
        atomic(evidence/'expected-objects.json', expected_objects)
        final = remote.publish_file(path,destination.rstrip('/')+'/SOURCE-COMMIT.json',stability_seconds)
        proof = {'status': 'verified-remote-source', 'schema': SCHEMA, 'files': inventory,
                 'archive': first[archive], 'bundle_receipt': first[receipt], 'remote_manifest': final,
                 'events': count, 'validation_limit': complete['validation_limit'],
                 'commit_wall_seconds': time.monotonic()-began, 'verified_utc_epoch': time.time(),
                 'local_deletion': False}
        atomic(source/'remote-source-copy.json',proof)
        atomic(evidence/'remote-source-copy.json',proof)
        return proof


def prune_bundle(source, evidence, stability_seconds=30):
    source, evidence = Path(source), Path(evidence)
    if stability_seconds < 1:
        raise ValueError('Positive stability interval required')
    with (source/'task.lock').open('a+') as lock:
        fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        proof = json.loads((source/'remote-source-copy.json').read_text())
        manifest, bundle, items = objects(source,proof)
        for item in items.values():
            if remote.remote_adler(item['url']) != item['adler32']:
                raise ValueError('Remote bundle changed; local output retained')
        time.sleep(stability_seconds)
        second = {name: remote.verify_staged_file(source/name,item,stability_seconds) for name,item in items.items()}
        removals = []
        for chunk in manifest['chunks']:
            receipt = json.loads((source/chunk['chunk']/'receipt.json').read_text())
            for name,expected in receipt['sha256'].items():
                relative = chunk['chunk']+'/'+name
                checked_name(relative)
                item = manifest['files'][relative]
                if item['sha256'] != expected or item['bytes'] != receipt['bytes'][name]:
                    raise ValueError('Chunk receipt differs from bundle inventory')
                path = source/relative
                if path.exists() and (path.stat().st_size != item['bytes'] or remote.digest(path,'sha256') != expected):
                    raise ValueError('Local output changed; all output retained')
                removals.append((path,item['bytes']))
        remote.archive_ledger(source,manifest,proof)
        # Failed WORK writes abort before the first local unlink.
        expected={name:{k:item[k] for k in ['bytes','sha256','adler32']} for name,item in manifest['files'].items()}
        existing=evidence/'expected-files.json'
        if existing.exists() and json.loads(existing.read_text())!=expected:
            raise ValueError('WORK expected member hashes changed')
        atomic(existing,expected)
        atomic(evidence/'expected-objects.json',{item['url']:{k:item[k] for k in ['bytes','sha256']} for item in items.values()})
        atomic(evidence/'remote-source-copy.json',proof)
        atomic(evidence/'second-readback.json',{'status':'verified-before-local-eviction','files':second,
               'remote_manifest_sha256':proof['remote_manifest']['sha256'],'verified_utc_epoch':time.time()})
        for path,_ in removals:
            path.unlink(missing_ok=True)
        remote.archive_ledger(source,manifest,proof,apply=True)
        result = {'status':'local-payloads-pruned-after-remote-reverification',
                  'removed_bytes':sum(size for _,size in removals), 'verified_utc_epoch':time.time(),
                  'remote_manifest_sha256':proof['remote_manifest']['sha256']}
        atomic(source/'local-eviction.json',result)
        atomic(evidence/'local-eviction.json',result)
        return result


def restore_bundle(source, stability_seconds=30):
    source = Path(source)
    with (source/'task.lock').open('a+') as lock:
        fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        proof = json.loads((source/'remote-source-copy.json').read_text())
        manifest,bundle,items = objects(source,proof)
        if stability_seconds < 1:
            raise ValueError('Positive stability interval required')
        for item in items.values():
            if remote.remote_adler(item['url']) != item['adler32']:
                raise ValueError('Remote bundle changed')
        time.sleep(stability_seconds)
        for name in ['SOURCE-COMMIT.json','BUNDLE-MANIFEST.json']:
            remote.verify_staged_file(source/name,items[name],stability_seconds)
        root = source.parent.parent
        admission = root/'admission-ledger.json'
        if admission.exists() and json.loads(admission.read_text()).get('reservations'):
            raise RuntimeError('Stop active producers before restoring archived source')
        ledger = root/'storage-ledger.json'
        if ledger.exists():
            with storage_ledger_lock(root):
                data=json.loads(ledger.read_text())
                changes={source.name+'/'+c['chunk']:sum(json.loads((source/c['chunk']/'receipt.json').read_text())['bytes'].values()) for c in manifest['chunks']}
                total=sum(data['reservations'].values())+sum(v-data['reservations'].get(k,0) for k,v in changes.items())
                if total>data['cap_bytes']:
                    raise RuntimeError('Restoration exceeds local ledger cap')
                data['reservations'].update(changes)
                for key,size in changes.items():
                    if key in data.get('remote_archived',{}):data['remote_archived'][key]['remaining_local_bytes']=size
                atomic(ledger,data)
        restored=[]
        with tempfile.TemporaryDirectory(prefix='replay-bundle-restore-') as temp:
            archive=Path(temp)/'source.tar'
            remote.run('gfal-copy','--checksum','ADLER32',items['source.tar']['url'],archive.as_uri())
            verify(archive,bundle)
            # Only regular, checked names from a verified tar are copied. Never
            # extract links, devices, or paths chosen by untrusted tar metadata.
            with tarfile.open(archive,'r:') as tar:
                for name,item in manifest['files'].items():
                    checked_name(name);path=source/name
                    if path.exists():
                        if path.stat().st_size!=item['bytes'] or remote.digest(path,'sha256')!=item['sha256']:
                            raise ValueError('Existing restored file differs')
                        continue
                    if any(p.is_symlink() for p in [path,*path.parents] if p!=source):
                        raise ValueError('Restore destination contains a symbolic link')
                    path.parent.mkdir(parents=True,exist_ok=True)
                    partial=path.with_suffix(path.suffix+'.restore-partial')
                    with partial.open('xb') as output,tar.extractfile(name) as inp:shutil.copyfileobj(inp,output)
                    if partial.stat().st_size!=item['bytes'] or remote.digest(partial,'sha256')!=item['sha256']:
                        raise ValueError('Restored member differs; partial retained')
                    partial.replace(path);restored.append(name)
        complete,_,_,count=validate_source_inventory(source,manifest['validation_limit'] is not None,remote.digest)
        if count!=manifest['exported_events'] or complete['source']!=manifest['source']:
            raise ValueError('Restored source accounting differs')
        result={'status':'restored-verified-source','files':restored,'verified_utc_epoch':time.time(),
                'remote_manifest_sha256':proof['remote_manifest']['sha256']}
        atomic(source/'local-restoration.json',result)
        return result


def audit_bundle(source, expected_source, manifest_sha256, stability_seconds=30):
    source=Path(source)
    final=json.loads((source/'SOURCE-COMMIT.json').read_text())
    if final['source']!=expected_source or final['manifest_sha256']!=manifest_sha256 or final['validation_limit'] is not None:
        raise ValueError('Invalid full-source bundle identity')
    # Audit in disposable staging. The original archived source and its ledger
    # remain unchanged; reuse the canonical inventory validator after restore.
    with tempfile.TemporaryDirectory(prefix='replay-bundle-audit-') as temp:
        clone=Path(temp)/'sources'/source.name
        clone.mkdir(parents=True)
        metadata=set(['COMPLETE.json','export-index.json','SOURCE-COMMIT.json','BUNDLE-MANIFEST.json','remote-source-copy.json'])
        metadata.update(name for name in final['files'] if '/' not in name)
        for name in sorted(metadata):
            shutil.copyfile(source/name,clone/name)
        for path in source.glob('chunk-*/receipt.json'):
            target=clone/path.relative_to(source);target.parent.mkdir();shutil.copyfile(path,target)
        restore_bundle(clone,stability_seconds)
        _,_,_,count=validate_source_inventory(clone,False,remote.digest)
        return count,len(final['chunks'])


def transfer_source(source, destination, evidence, prefix=False):
    """Worker transport entry point, retaining small replay metadata on WORK."""
    from gfal_transport import run
    from replay_cms_collection import checked_copy
    remote.run = run
    source,evidence=Path(source),Path(evidence)
    proof_path=source/'remote-source-copy.json'
    proof=json.loads(proof_path.read_text()) if proof_path.exists() else commit_bundle(
        source,destination,evidence,allow_prefix=prefix)
    if proof.get('schema')!=SCHEMA:
        raise ValueError('Transfer mode differs from existing source proof')
    names=['COMPLETE.json','export-index.json','SOURCE-COMMIT.json','BUNDLE-MANIFEST.json']
    names += [str(p.relative_to(source)) for p in source.glob('chunk-*/receipt.json')]
    names += [p.name for p in source.glob('export-*.log')]
    for name in names:checked_copy(source/name,evidence/name)
    prune_bundle(source,evidence)
    return proof
