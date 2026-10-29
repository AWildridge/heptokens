"""Publish and verify immutable replay sources or chunks on remote storage.

The POSIX staging receipt remains authoritative until every payload has passed
remote checksum stability and a SHA256 readback. Explicit eviction requires a
second complete remote verification; restoration checks every downloaded file.
Requires a single source owner and an exclusive POSIX staging lock.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
import zlib
from replay_runtime import storage_ledger_lock


def file_map(function,items,workers):
    if workers<1 or workers>8:raise ValueError('File transfer workers must be between 1 and 8')
    if workers==1:return [function(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function,items))


def digest(path, algorithm):
    value = hashlib.sha256() if algorithm == 'sha256' else 1
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            if algorithm == 'sha256':
                value.update(block)
            else:
                value = zlib.adler32(block, value)
    return value.hexdigest() if algorithm == 'sha256' else '%08x' % (value & 0xffffffff)


def run(*command):
    from gfal_transport import run as transport_run
    return transport_run(*command)


def remote_adler(url):
    return run('gfal-sum', url, 'ADLER32').split()[-1].lower().zfill(8)


def stage_file(source, url):
    started = time.monotonic()
    checksum = digest(source, 'adler32')
    # Never overwrite an existing object. An interrupted partial upload requires
    # inspection, not an automatic force-copy over potentially committed work.
    try:
        existing = remote_adler(url)
    except subprocess.CalledProcessError as exc:
        if 'No such file' not in exc.stderr and 'not found' not in exc.stderr.lower():
            raise
        run('gfal-copy', '-p', '--checksum', 'ADLER32', source.resolve().as_uri(), url)
        existing = remote_adler(url)
    if existing != checksum:
        raise ValueError('Remote checksum differs: ' + url)
    return {'url':url,'bytes':source.stat().st_size,'sha256':digest(source,'sha256'),
            'adler32':checksum,'upload_or_existing_check_seconds':time.monotonic()-started}


def verify_staged_file(source, staged, stability_seconds):
    url = staged['url']
    if remote_adler(url) != staged['adler32']:
        raise ValueError('Remote file is unstable: ' + url)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='replay-readback-') as directory:
        target = Path(directory) / source.name
        run('gfal-copy', '--checksum', 'ADLER32', url, target.as_uri())
        if target.stat().st_size != staged['bytes'] or digest(target, 'sha256') != staged['sha256']:
            raise ValueError('Remote SHA256 readback differs: ' + url)
    return dict(staged,stable_interval_seconds=stability_seconds,readback='passed',
                readback_seconds=time.monotonic()-started)


def publish_file(source, url, stability_seconds):
    staged = stage_file(source,url)
    time.sleep(stability_seconds)
    return verify_staged_file(source,staged,stability_seconds)


def commit(chunk, destination, stability_seconds=30):
    started = time.monotonic()
    if stability_seconds < 1:
        raise ValueError('Stability interval must be positive')
    with open(chunk.parent / 'task.lock', 'a+') as lock:
        fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = chunk / 'receipt.json'
        receipt = json.loads(receipt_path.read_text())
        if receipt['status'] != 'complete':
            raise ValueError('Chunk is not committed')
        for name, expected in receipt['sha256'].items():
            if Path(name).name != name or digest(chunk / name, 'sha256') != expected:
                raise ValueError('Invalid local receipt payload: ' + name)
        staged = {name:stage_file(chunk/name,destination.rstrip('/')+'/'+name)
                  for name in receipt['sha256']}
        # Every payload is idle for at least this interval. Batch the wait so
        # adding small provenance files does not serialize many 30-second naps.
        time.sleep(stability_seconds)
        evidence = {name:verify_staged_file(chunk/name,item,stability_seconds)
                    for name,item in staged.items()}
        evidence['receipt.json'] = publish_file(receipt_path,destination.rstrip('/')+'/receipt.json',stability_seconds)
        # Readers of remote storage must require this final immutable manifest,
        # rather than inferring completion from a directory listing.
        manifest = {'schema':'heptokens-remote-chunk-v1','status':'complete',
                    'receipt_sha256':digest(receipt_path,'sha256'),
                    'files':{name:{key:item[key] for key in ['url','bytes','sha256']}
                             for name,item in evidence.items()}}
        manifest_path = chunk/'REMOTE-COMMIT.json'
        manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True))
        remote_manifest = publish_file(manifest_path,destination.rstrip('/')+'/REMOTE-COMMIT.json',stability_seconds)
        result = {'status': 'verified-remote-copy', 'files': evidence,
                  'remote_manifest':remote_manifest,
                  'receipt_sha256': digest(receipt_path, 'sha256'), 'verified_utc_epoch': time.time(),
                  'commit_wall_seconds':time.monotonic()-started,
                  'local_deletion': False}
        temporary = chunk / 'remote-copy.json.tmp'
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(chunk / 'remote-copy.json')
        return result


def commit_source(source, destination, stability_seconds=30, allow_prefix=False, file_workers=1, bundle=False, evidence=None):
    """Commit a complete source tree with one immutable final manifest.

    All existing physics receipts stay unchanged. Remote readers must require
    SOURCE-COMMIT.json, including for prefix-only pilot directories.
    """
    if bundle:
        if evidence is None:raise ValueError('Bundle commits require shared evidence storage')
        from replay_remote_bundle import commit_bundle
        return commit_bundle(source,destination,evidence,stability_seconds,allow_prefix)
    if stability_seconds<1:
        raise ValueError('Stability interval must be positive')
    started=time.monotonic()
    with open(source/'task.lock','a+') as lock:
        fcntl.lockf(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from replay_source_validation import validate_source_inventory
        complete,index,files,count=validate_source_inventory(source,allow_prefix,digest)
        staged=dict(zip(files,file_map(lambda name:stage_file(source/name,destination.rstrip('/')+'/'+name),files,file_workers)))
        time.sleep(stability_seconds)
        evidence=dict(zip(staged,file_map(lambda pair:verify_staged_file(source/pair[0],pair[1],stability_seconds),
                                         list(staged.items()),file_workers)))
        manifest={'schema':'heptokens-remote-source-v1','status':'complete',
                  'source':complete['source'],'source_events':complete['source_events'],
                  'exported_events':count,'validation_limit':complete['validation_limit'],
                  'manifest_sha256':complete['manifest_sha256'],'chunks':complete['chunks'],
                  'files':{name:{key:item[key] for key in ['url','bytes','sha256','adler32']} for name,item in evidence.items()}}
        path=source/'SOURCE-COMMIT.json'
        path.write_text(json.dumps(manifest,indent=2,sort_keys=True))
        remote_manifest=publish_file(path,destination.rstrip('/')+'/SOURCE-COMMIT.json',stability_seconds)
        result={'status':'verified-remote-source','files':evidence,'remote_manifest':remote_manifest,
                'events':count,'validation_limit':complete['validation_limit'],
                'commit_wall_seconds':time.monotonic()-started,'verified_utc_epoch':time.time(),'file_workers':file_workers,
                'local_deletion':False}
        temporary=source/'remote-source-copy.json.tmp'
        temporary.write_text(json.dumps(result,indent=2))
        temporary.replace(source/'remote-source-copy.json')
        return result


def archive_ledger(source,manifest,proof,apply=False):
    root=source.parent.parent
    ledger=root/'storage-ledger.json'
    if not ledger.exists():
        return
    with storage_ledger_lock(root):
        data=json.loads(ledger.read_text())
        for item in manifest['chunks']:
            name=item['chunk'];receipt=json.loads((source/name/'receipt.json').read_text())
            key=source.name+'/'+name
            original=sum(receipt['bytes'].values())
            retained=0
            archived={'original_bytes':original,'remaining_local_bytes':retained,
                      'remote_manifest_sha256':proof['remote_manifest']['sha256'],
                      'remote_manifest_url':proof['remote_manifest']['url']}
            prior=data.get('remote_archived',{}).get(key)
            if prior is not None and any(prior[field]!=archived[field] for field in ['original_bytes','remote_manifest_sha256','remote_manifest_url']):
                raise ValueError('Prior remote archive differs')
            current=data['reservations'].get(key)
            if current!=original and not (prior is not None and current==retained):
                raise ValueError('Local storage ledger differs; local outputs retained')
            if apply:
                data['reservations'][key]=retained
                data.setdefault('remote_archived',{})[key]=archived
        if apply:
            temporary=ledger.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(data,indent=2));temporary.replace(ledger)


def prune_source_payloads(source, stability_seconds=30, file_workers=1, audit_path=None):
    """Reverify the remote source in full, then evict large local payloads.

    Keep source/chunk receipts, export accounting and final remote manifests.
    Validation reports and cards remain available in the verified remote copy.
    Retry after interruption is safe even when some payloads are already gone.
    """
    if json.loads((source/'SOURCE-COMMIT.json').read_text()).get('schema')=='heptokens-remote-source-bundle-v1':
        if audit_path is None:raise ValueError('Bundle pruning requires shared evidence storage')
        from replay_remote_bundle import prune_bundle
        return prune_bundle(source,Path(audit_path).parent,stability_seconds)
    if stability_seconds<1:
        raise ValueError('Stability interval must be positive')
    with open(source/'task.lock','a+') as lock:
        fcntl.lockf(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        proof=json.loads((source/'remote-source-copy.json').read_text())
        manifest=json.loads((source/'SOURCE-COMMIT.json').read_text())
        if proof['status']!='verified-remote-source' or digest(source/'SOURCE-COMMIT.json','sha256')!=proof['remote_manifest']['sha256']:
            raise ValueError('Remote source proof is incomplete or changed')
        if manifest['source']['task']!=int(source.name):
            raise ValueError('Source ownership identity differs')
        projected={name:{key:item[key] for key in ['url','bytes','sha256','adler32']} for name,item in proof['files'].items()}
        if projected!=manifest['files']:
            raise ValueError('Remote payload evidence differs from the final manifest')
        evidence=dict(proof['files'],**{'SOURCE-COMMIT.json':proof['remote_manifest']})
        def check_remote(pair):
            name,item=pair
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Unsafe source-relative path')
            if remote_adler(item['url'])!=item['adler32']:
                raise ValueError('Remote payload changed; local output retained')
        file_map(check_remote,list(evidence.items()),file_workers)
        time.sleep(stability_seconds)
        second_readbacks = dict(zip(evidence, file_map(
            lambda pair:verify_staged_file(source/pair[0],pair[1],stability_seconds),
            list(evidence.items()),file_workers)))
        removable=set()
        for chunk in manifest['chunks']:
            receipt=json.loads((source/chunk['chunk']/'receipt.json').read_text())
            removable.update(chunk['chunk']+'/'+name for name in receipt['sha256'])
        removals=[]
        for name,item in proof['files'].items():
            path=source/name
            if name in removable:
                if path.exists() and digest(path,'sha256')!=item['sha256']:
                    raise ValueError('Local payload changed; all local output retained')
                removals.append((path,item['bytes']))
        archive_ledger(source,manifest,proof)
        if audit_path is not None:
            audit_path = Path(audit_path)
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = audit_path.with_suffix(audit_path.suffix + '.tmp')
            temporary.write_text(json.dumps({
                'status': 'verified-before-local-eviction',
                'files': second_readbacks,
                'remote_manifest_sha256': proof['remote_manifest']['sha256'],
                'verified_utc_epoch': time.time()}, indent=2))
            temporary.replace(audit_path)
        # Every remote payload and the final manifest have just passed a second
        # stability interval and full readback. No unverified file is removed.
        for path,_ in removals:
            path.unlink(missing_ok=True)
        archive_ledger(source,manifest,proof,apply=True)
        result={'status':'local-payloads-pruned-after-remote-reverification',
                'removed_bytes':sum(size for _,size in removals),'verified_utc_epoch':time.time(),
                'remote_manifest_sha256':proof['remote_manifest']['sha256']}
        temporary=source/'local-eviction.json.tmp'
        temporary.write_text(json.dumps(result,indent=2));temporary.replace(source/'local-eviction.json')
        return result


def restore_source_payloads(source, stability_seconds=30):
    """Rehydrate an archived source into POSIX staging for canonical verification."""
    if json.loads((source/'SOURCE-COMMIT.json').read_text()).get('schema')=='heptokens-remote-source-bundle-v1':
        from replay_remote_bundle import restore_bundle
        return restore_bundle(source,stability_seconds)
    if stability_seconds<1:
        raise ValueError('Stability interval must be positive')
    with open(source/'task.lock','a+') as lock:
        fcntl.lockf(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        proof=json.loads((source/'remote-source-copy.json').read_text())
        manifest_path=source/'SOURCE-COMMIT.json'
        if proof['status']!='verified-remote-source' or digest(manifest_path,'sha256')!=proof['remote_manifest']['sha256']:
            raise ValueError('Invalid source archive proof')
        remote_manifest=proof['remote_manifest']
        if remote_adler(remote_manifest['url'])!=remote_manifest['adler32']:
            raise ValueError('Remote source manifest changed')
        time.sleep(stability_seconds)
        verify_staged_file(manifest_path,remote_manifest,stability_seconds)
        manifest=json.loads(manifest_path.read_text())
        root=source.parent.parent
        admission=root/'admission-ledger.json'
        if admission.exists() and json.loads(admission.read_text()).get('reservations'):
            raise RuntimeError('Stop active producers before restoring archived source payloads')
        ledger=root/'storage-ledger.json'
        if ledger.exists():
            with storage_ledger_lock(root):
                data=json.loads(ledger.read_text())
                changes={}
                for chunk in manifest['chunks']:
                    key=source.name+'/'+chunk['chunk']
                    receipt=json.loads((source/chunk['chunk']/'receipt.json').read_text())
                    changes[key]=sum(receipt['bytes'].values())
                total=sum(data['reservations'].values())+sum(size-data['reservations'].get(key,0) for key,size in changes.items())
                if total>data['cap_bytes']:
                    raise RuntimeError('Restoration exceeds the local staging ledger cap')
                data['reservations'].update(changes)
                for key,size in changes.items():
                    if key in data.get('remote_archived',{}):
                        data['remote_archived'][key]['remaining_local_bytes']=size
                temporary=ledger.with_suffix('.json.tmp')
                temporary.write_text(json.dumps(data,indent=2));temporary.replace(ledger)
        restored=[]
        for name,item in manifest['files'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Unsafe source-relative path')
            path=source/name
            if path.exists():
                if digest(path,'sha256')!=item['sha256']:
                    raise ValueError('Existing local archive file differs')
                continue
            path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_suffix(path.suffix+'.restore-partial')
            if temporary.exists():
                raise ValueError('Prior incomplete restore needs inspection: '+str(temporary))
            run('gfal-copy','--checksum','ADLER32',item['url'],temporary.resolve().as_uri())
            if temporary.stat().st_size!=item['bytes'] or digest(temporary,'sha256')!=item['sha256']:
                raise ValueError('Restored remote payload differs; partial retained')
            temporary.replace(path);restored.append(name)
        result={'status':'restored-verified-source','files':restored,'verified_utc_epoch':time.time(),
                'remote_manifest_sha256':remote_manifest['sha256']}
        temporary=source/'local-restoration.json.tmp'
        temporary.write_text(json.dumps(result,indent=2));temporary.replace(source/'local-restoration.json')
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    selection=parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--chunk',type=Path)
    selection.add_argument('--source',type=Path)
    selection.add_argument('--prune-source',type=Path)
    selection.add_argument('--restore-source',type=Path)
    parser.add_argument('--destination')
    parser.add_argument('--allow-prefix',action='store_true')
    parser.add_argument('--bundle',action='store_true')
    parser.add_argument('--evidence',type=Path)
    parser.add_argument('--stability-seconds', type=int, default=30)
    parser.add_argument('--file-workers',type=int,default=1,help='Concurrent files per source; default preserves serial transfers')
    args = parser.parse_args()
    if args.restore_source:
        result=restore_source_payloads(args.restore_source,args.stability_seconds)
    elif args.prune_source:
        result=prune_source_payloads(args.prune_source,args.stability_seconds,args.file_workers,audit_path=(args.evidence/'second-readback.json') if args.evidence else None)
    else:
        if not args.destination:parser.error('--destination is required for a commit')
        result=commit_source(args.source,args.destination,args.stability_seconds,args.allow_prefix,args.file_workers,bundle=args.bundle,evidence=args.evidence) if args.source else commit(args.chunk,args.destination,args.stability_seconds)
    print(json.dumps(result,indent=2))
