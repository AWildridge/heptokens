"""Verified imports of existing replay outputs; never produce campaign READY.

Reuses canonical tar and source validators. Destination receipts are independent
of EOS receipts and never authorize deletion of the original source copies.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
from urllib.parse import unquote,urlsplit

from replay_bundle import verify as verify_bundle, checked_name, digest
from replay_source_validation import validate_source_inventory
from replay_scratch_budget import durable_json

CONTROL={'source.tar','BUNDLE-MANIFEST.json','SOURCE-COMMIT.json','LOCAL-SOURCE.json'}


def sha(path):
    with Path(path).open('rb') as stream:return digest(stream)['sha256']


def name_ok(name):
    if name in CONTROL:return name
    return checked_name(name)


def safe_path(root,name):
    name_ok(name);root=Path(root).resolve();path=root/name
    # Path.is_relative_to was added in Python 3.9; NERSC system Python may
    # be older. relative_to has the same containment semantics on Python 3.7+.
    try:path.resolve().relative_to(root)
    except ValueError:raise ValueError('Symlink or escaped destination: '+name)
    if any(p.is_symlink() for p in [path,*path.parents] if p!=root):
        raise ValueError('Symlink or escaped destination: '+name)
    return path


def matches(path,expected):
    return path.is_file() and not path.is_symlink() and path.stat().st_size==expected['bytes'] and sha(path)==expected['sha256']


def fetch(item,path,backend):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if not matches(path,item):raise ValueError('Existing destination differs; retained: '+str(path))
        return
    partial=path.with_name(path.name+'.download-partial')
    for candidate in [partial,path.with_name(path.name+'.receive-partial')]:
        if candidate.exists() and matches(candidate,item):
            with candidate.open('rb') as stream:os.fsync(stream.fileno())
            os.link(candidate,path);candidate.unlink();return
    parsed=urlsplit(item['url'])
    if parsed.scheme=='file':
        with open(unquote(parsed.path),'rb') as src,partial.open('wb') as out:shutil.copyfileobj(src,out,8*1024**2)
    elif backend=='none':raise FileNotFoundError('Missing cached object: '+str(path))
    elif backend=='gfal':
        if partial.exists():partial.rename(partial.with_name(partial.name+'.retained-'+uuid.uuid4().hex))
        subprocess.run(['gfal-copy','--checksum','ADLER32',item['url'],partial.resolve().as_uri()],check=True,timeout=7200)
    elif backend=='curl':
        if parsed.scheme not in ['https','davs']:raise ValueError('curl backend requires HTTPS/DAVS')
        url='https:'+item['url'].split(':',1)[1]
        command=['curl','--fail','--location','--retry','3','--connect-timeout','30','--max-time','7200','--continue-at','-','--output',str(partial)]
        proxy=os.environ.get('X509_USER_PROXY')
        if proxy:command+=['--cert',proxy,'--key',proxy]
        ca_bundle=os.environ.get('CURL_CA_BUNDLE') or os.environ.get('SSL_CERT_FILE')
        if ca_bundle:command+=['--cacert',ca_bundle]
        elif os.environ.get('X509_CERT_DIR'):command+=['--capath',os.environ['X509_CERT_DIR']]
        subprocess.run(command+[url],check=True,timeout=7300)
    else:raise ValueError('Unsupported copy backend')
    if not matches(partial,item):raise ValueError('Transferred SHA256/size differs; partial retained: '+str(partial))
    with partial.open('rb') as stream:os.fsync(stream.fileno())
    os.link(partial,path);partial.unlink()


def load_snapshot(path):
    snapshot=json.loads(Path(path).read_text());tasks=[r['task'] for r in snapshot['sources']]
    if len(tasks)!=len(set(tasks)) or any(type(t) is not int or not 0<=t<4546 for t in tasks):raise ValueError('Invalid/duplicate source task')
    if snapshot.get('cleanup_authorized') is not False:raise ValueError('Snapshot must explicitly forbid source cleanup')
    for row in snapshot['sources']:
        if row['source']['task']!=row['task']:raise ValueError('Source identity differs')
        for name,item in row['objects'].items():
            name_ok(name)
            if type(item['bytes']) is not int or item['bytes']<0 or len(item['sha256'])!=64 or any(c not in '0123456789abcdef' for c in item['sha256']):raise ValueError('Invalid expected object')
    return snapshot


def validate_source(source,row,snapshot):
    def checksum(path,algorithm):
        if algorithm!='sha256':raise ValueError('Only SHA256 accepted')
        return sha(path)
    complete,index,files,events=validate_source_inventory(source,False,checksum)
    if (complete['source']!=row['source'] or events!=row['events'] or complete['source_events']!=events
        or complete['manifest_sha256']!=snapshot['source_manifest_sha256']):raise ValueError('Imported source/event/manifest identity differs')
    return complete,files


def transport_metadata(cache,row,snapshot):
    metadata='LOCAL-SOURCE.json' if 'LOCAL-SOURCE.json' in row['objects'] else 'SOURCE-COMMIT.json'
    final=json.loads((cache/metadata).read_text())
    if (final['source']!=row['source'] or final['exported_events']!=row['events'] or final['validation_limit'] is not None
        or final['manifest_sha256']!=snapshot['source_manifest_sha256']):raise ValueError('Transport source metadata differs')
    members={n:{k:v[k] for k in ['bytes','sha256']} for n,v in final['files'].items()}
    receipt=None
    if 'source.tar' in row['objects']:
        receipt=json.loads((cache/'BUNDLE-MANIFEST.json').read_text())
        if receipt['archive']!={k:row['objects']['source.tar'][k] for k in ['bytes','sha256']}:raise ValueError('Archive receipt differs from snapshot')
        if receipt['files']!=members:raise ValueError('Tar files differ from transport metadata')
    elif members!={n:{k:v[k] for k in ['bytes','sha256']} for n,v in row['objects'].items() if n!=metadata}:
        raise ValueError('Legacy file inventory differs from final commit')
    return metadata,final,receipt


def eviction_path(root,row):
    return safe_path(root,'transfer-records/archive-cache-%05d.json'%row['task'])


def archive_eviction_proof(root,row,snapshot,snapshot_path):
    path=eviction_path(root,row)
    if not path.is_file():raise ValueError('Cached archive missing without verified eviction evidence')
    proof=json.loads(path.read_text());verification=proof.get('verification',{})
    if (proof.get('schema')!='heptokens-import-archive-eviction-v1'
        or proof.get('status') not in ['verified-before-cache-eviction','archive-cache-pruned']
        or proof.get('root')!=str(Path(root).resolve()) or proof.get('task')!=row['task']
        or proof.get('archive')!=row['objects']['source.tar']
        or proof.get('source_cleanup_authorized') is not False
        or verification.get('status')!='verified-import'
        or verification.get('task')!=row['task'] or verification.get('root')!=str(Path(root).resolve())
        or verification.get('archive_cache')!='retained'
        or verification.get('source_cleanup_authorized') is not False
        or verification.get('source')!=row['source'] or verification.get('events')!=row['events']
        or verification.get('snapshot_sha256')!=sha(snapshot_path)
        or verification.get('source_manifest_sha256')!=snapshot['source_manifest_sha256']
        or verification.get('object_sha256')!={n:v['sha256'] for n,v in row['objects'].items()}
        or verification.get('stable_interval_seconds',0)<1):
        raise ValueError('Archive eviction evidence differs from frozen inventory')
    return proof


def begin_verification(root,row,snapshot_path):
    # A failed retry or recheck must not leave an earlier success counted as current.
    durable_json(Path(root)/'transfer-records'/('source-%05d.json'%row['task']),
                 {'status':'verifying-import','task':row['task'],'snapshot_sha256':sha(snapshot_path),
                  'time':time.time(),'source_cleanup_authorized':False,'dataset_complete':False})


def verify_import(root,row,snapshot,snapshot_path,stable_seconds=0):
    root=Path(root).resolve();source=safe_path(root,'sources/%05d'%row['task'])
    cache=safe_path(root,'objects/%05d'%row['task'])
    begin_verification(root,row,snapshot_path)
    archive_state='retained' if 'source.tar' in row['objects'] else 'not-applicable'
    for name,item in row['objects'].items():
        path=safe_path(cache,name)
        if name=='source.tar' and not path.exists():
            archive_eviction_proof(root,row,snapshot,snapshot_path)
            archive_state='pruned-after-verified-import'
        elif not matches(path,item):raise ValueError('Cached object differs from frozen inventory: '+name)
    metadata,final,bundle_receipt=transport_metadata(cache,row,snapshot)
    for name,item in final['files'].items():
        if not matches(safe_path(source,name),item):raise ValueError('Imported file differs from frozen transport manifest: '+name)
    for name in [metadata]+(['BUNDLE-MANIFEST.json'] if 'source.tar' in row['objects'] else []):
        if sha(safe_path(source,name))!=sha(safe_path(cache,name)):raise ValueError('Source transport metadata changed')
    complete,files=validate_source(source,row,snapshot)
    for chunk in complete['chunks']:
        for detector in ['CMS','ATLAS']:
            original=source/chunk['chunk']/(detector+'.h5');training=safe_path(root,'training/%05d/%s/%s.h5'%(row['task'],chunk['chunk'],detector))
            if not training.exists() or sha(training)!=sha(original):raise ValueError('Training shard differs')
    if stable_seconds:
        time.sleep(stable_seconds)
        for name,item in final['files'].items():
            if not matches(safe_path(source,name),item):raise ValueError('Destination file is unstable: '+name)
        validate_source(source,row,snapshot)
        for name,item in row['objects'].items():
            path=safe_path(cache,name)
            if name=='source.tar' and archive_state=='pruned-after-verified-import':
                archive_eviction_proof(root,row,snapshot,snapshot_path)
            elif not matches(path,item):raise ValueError('Cached object is unstable: '+name)
        for name in [metadata]+(['BUNDLE-MANIFEST.json'] if 'source.tar' in row['objects'] else []):
            if sha(safe_path(source,name))!=sha(safe_path(cache,name)):raise ValueError('Source transport metadata is unstable')
        for chunk in complete['chunks']:
            for detector in ['CMS','ATLAS']:
                original=source/chunk['chunk']/(detector+'.h5');training=safe_path(root,'training/%05d/%s/%s.h5'%(row['task'],chunk['chunk'],detector))
                if not training.exists() or sha(training)!=sha(original):raise ValueError('Training shard is unstable')
    receipt={'status':'verified-import','task':row['task'],'events':row['events'],'source':row['source'],
             'transfer_tool_sha256':sha(Path(__file__)),'snapshot_sha256':sha(snapshot_path),'source_manifest_sha256':snapshot['source_manifest_sha256'],
             'complete_sha256':sha(source/'COMPLETE.json'),'object_sha256':{n:v['sha256'] for n,v in row['objects'].items()},
             'verified_files':len(files),'root':str(root),'verified_utc_epoch':time.time(),
             'stable_interval_seconds':stable_seconds,'archive_cache':archive_state,
             'source_cleanup_authorized':False,'dataset_complete':False}
    durable_json(root/'transfer-records'/('source-%05d.json'%row['task']),receipt)
    return receipt


def prune_archive_cache(root,row,snapshot,snapshot_path,stable_seconds):
    """Reopen and verify an imported source, durably record proof, then unlink only its cached tar."""
    if 'source.tar' not in row['objects']:
        return verify_import(root,row,snapshot,snapshot_path,stable_seconds)
    if stable_seconds<1:raise ValueError('Archive eviction requires a positive stability interval')
    verification=verify_import(root,row,snapshot,snapshot_path,stable_seconds)
    archive=safe_path(root,'objects/%05d/source.tar'%row['task'])
    if archive.exists():
        before=archive.stat()
        proof={'schema':'heptokens-import-archive-eviction-v1','status':'verified-before-cache-eviction',
               'task':row['task'],'root':str(Path(root).resolve()),'archive':row['objects']['source.tar'],
               'verification':verification,'recorded_utc_epoch':time.time(),
               'source_cleanup_authorized':False,'scope':'Only this destination source.tar cache; all source originals and imported files retained'}
        durable_json(eviction_path(root,row),proof)
        after=archive.stat()
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
            raise ValueError('Cached archive changed before eviction; retained')
        archive.unlink()
        fd=os.open(archive.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        proof.update(status='archive-cache-pruned',pruned_utc_epoch=time.time())
        durable_json(eviction_path(root,row),proof)
    verification['archive_cache']='pruned-after-verified-import'
    durable_json(Path(root)/'transfer-records'/('source-%05d.json'%row['task']),verification)
    return verification


def import_one(snapshot_path,root,row,backend='none',stable_seconds=30,prune_cache=False):
    snapshot=load_snapshot(snapshot_path);root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    if prune_cache and stable_seconds<1:raise ValueError('Archive eviction requires a positive stability interval')
    with (root/('.source-%05d.lock'%row['task'])).open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        begin_verification(root,row,snapshot_path)
        cache=safe_path(root,'objects/%05d'%row['task']);cache.mkdir(parents=True,exist_ok=True)
        if 'source.tar' in row['objects'] and not safe_path(cache,'source.tar').exists() and eviction_path(root,row).exists():
            # A durable pre-unlink proof also covers interruption immediately after unlink.
            # Recheck all retained data; never redownload an intentionally evicted archive.
            return verify_import(root,row,snapshot,snapshot_path,stable_seconds)
        for name,item in row['objects'].items():fetch(item,safe_path(cache,name),backend)
        bundle='source.tar' in row['objects']
        metadata,final,receipt=transport_metadata(cache,row,snapshot)
        destination=safe_path(root,'sources/%05d'%row['task'])
        if bundle:
            verify_bundle(cache/'source.tar',receipt)
        if not destination.exists():
            staging=safe_path(root,'.import-partials/%05d/sources/%05d'%(row['task'],row['task']));staging.mkdir(parents=True,exist_ok=True)
            if bundle:
                with tarfile.open(cache/'source.tar','r:') as tar:
                    for name,item in receipt['files'].items():
                        target=safe_path(staging,name);target.parent.mkdir(parents=True,exist_ok=True)
                        if target.exists():
                            if not matches(target,item):raise ValueError('Partial extracted member differs; retained')
                            continue
                        partial=target.with_name(target.name+'.extract-partial')
                        with tar.extractfile(name) as src,partial.open('wb') as out:shutil.copyfileobj(src,out,8*1024**2)
                        if not matches(partial,item):raise ValueError('Extracted member SHA256 differs')
                        with partial.open('rb') as stream:os.fsync(stream.fileno())
                        os.link(partial,target);partial.unlink()
            else:
                for name in final['files']:
                    target=safe_path(staging,name);target.parent.mkdir(parents=True,exist_ok=True)
                    if not target.exists():os.link(safe_path(cache,name),target)
                    elif not matches(target,row['objects'][name]):raise ValueError('Partial legacy source differs')
            validate_source(staging,row,snapshot)
            for name in [metadata]+(['BUNDLE-MANIFEST.json'] if bundle else []):
                target=safe_path(staging,name)
                if not target.exists():os.link(cache/name,target)
                elif sha(target)!=sha(cache/name):raise ValueError('Existing source metadata differs')
            for directory,_,_ in os.walk(staging,topdown=False):
                fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
                try:os.fsync(fd)
                finally:os.close(fd)
            destination.parent.mkdir(parents=True,exist_ok=True);os.rename(staging,destination)
            fd=os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
        validate_source(destination,row,snapshot)
        for chunk in json.loads((destination/'COMPLETE.json').read_text())['chunks']:
            for detector in ['CMS','ATLAS']:
                name='training/%05d/%s/%s.h5'%(row['task'],chunk['chunk'],detector);target=safe_path(root,name);target.parent.mkdir(parents=True,exist_ok=True)
                src=destination/chunk['chunk']/(detector+'.h5')
                if not target.exists():os.link(src,target)
                elif sha(target)!=sha(src):raise ValueError('Existing training data differs; never overwritten')
        verified=verify_import(root,row,snapshot,snapshot_path,stable_seconds)
        return prune_archive_cache(root,row,snapshot,snapshot_path,stable_seconds) if prune_cache else verified


def summary(root,snapshot_path,failed_tasks=()):
    root=Path(root).resolve();snapshot=load_snapshot(snapshot_path);expected={r['task']:r for r in snapshot['sources']};verified=[]
    for p in (root/'transfer-records').glob('source-*.json'):
        r=json.loads(p.read_text())
        if r['task'] in expected and r['task'] not in failed_tasks and r['snapshot_sha256']==sha(snapshot_path) and r['status']=='verified-import':verified.append(r)
    result={'status':'verified-partial-import' if len(verified)==len(expected) else 'incomplete-transfer','verified_sources':len(verified),
            'expected_sources':len(expected),'verified_events':sum(r['events'] for r in verified),'snapshot_sha256':sha(snapshot_path),
            'archive_cache_pruned_sources':sum(r.get('archive_cache')=='pruned-after-verified-import' for r in verified),
            'failed_tasks':sorted(failed_tasks),
            'source_cleanup_authorized':False,'dataset_complete':False,'root':str(root),'time':time.time()}
    durable_json(root/'TRANSFER-STATUS.json',result)
    marker=root/('TRANSFER-VERIFIED-'+sha(snapshot_path)[:16]+'.json')
    if result['status']=='verified-partial-import':durable_json(marker,result)
    elif marker.exists():marker.unlink()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['pull','import-one','receive','verify','summary','seed-metadata']);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--tasks',type=int,nargs='+');p.add_argument('--task',type=int);p.add_argument('--name',default='source.tar');p.add_argument('--backend',choices=['auto','gfal','curl','none'],default='auto');p.add_argument('--workers',type=int,default=4);p.add_argument('--stable-seconds',type=int,default=30);p.add_argument('--prune-archive-cache',action='store_true',help='After independent stable source verification, evict only destination source.tar caches; retain originals and data');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if a.workers<1 or a.workers>16 or a.stable_seconds<1:p.error('workers1..16 and positive stability interval required')
    if a.prune_archive_cache and a.mode not in ['pull','import-one']:p.error('Archive eviction is available only during pull/import-one')
    if a.backend=='auto':a.backend='gfal' if shutil.which('gfal-copy') else 'curl'
    snapshot=load_snapshot(a.manifest);rows=snapshot['sources'];bytask={r['task']:r for r in rows}
    if a.task is not None:
        if a.task not in bytask:p.error('Task absent from frozen snapshot')
        rows=[bytask[a.task]]
    if a.tasks:
        if any(t not in bytask for t in a.tasks):p.error('Task absent from frozen snapshot')
        rows=[bytask[t] for t in a.tasks]
    if a.dry_run:
        print(json.dumps({'mode':a.mode,'sources':len(rows),'object_bytes':sum(v['bytes'] for r in rows for v in r['objects'].values()),'prune_destination_archive_cache':a.prune_archive_cache,'root':str(a.root),'network_connections':0,'source_cleanup_authorized':False},indent=2));return
    if a.mode=='seed-metadata':
        for row in rows:
            for name,item in row['objects'].items():
                if name=='source.tar':continue
                src=safe_path(a.manifest.parent,'local-objects/%05d/%s'%(row['task'],name))
                if not matches(src,item):raise ValueError('Kit metadata differs from frozen inventory')
                target=safe_path(a.root,'objects/%05d/%s'%(row['task'],name));target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():
                    if not matches(target,item):raise ValueError('Existing destination metadata differs; never overwritten')
                else:os.link(src,target)
        print(json.dumps({'status':'metadata-seeded','sources':len(rows)}));return
    if a.mode=='receive':
        if a.task is None or a.name not in bytask[a.task]['objects']:p.error('Receive requires a known task/object')
        item=bytask[a.task]['objects'][a.name];target=safe_path(a.root,'objects/%05d/%s'%(a.task,a.name));target.parent.mkdir(parents=True,exist_ok=True);partial=target.with_name(target.name+'.receive-partial')
        with partial.open('wb') as out:
            result=digest_copy(sys.stdin.buffer,out,item['bytes']);out.flush();os.fsync(out.fileno())
        if result!={k:item[k] for k in ['bytes','sha256']}:raise ValueError('Received stream SHA256/size differs; partial retained')
        if target.exists():
            if not matches(target,item):raise ValueError('Existing cached object differs; never overwritten')
            partial.unlink()
        else:os.link(partial,target);partial.unlink()
        print(json.dumps({'status':'received-verified-object','task':a.task,'name':a.name,**result}));return
    if a.mode=='summary':print(json.dumps(summary(a.root,a.manifest)));return
    def process(row):
        if a.mode=='verify':
            with (a.root/('.source-%05d.lock'%row['task'])).open('a+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                return verify_import(a.root,row,snapshot,a.manifest,a.stable_seconds)
        return import_one(a.manifest,a.root,row,a.backend,a.stable_seconds,prune_cache=a.prune_archive_cache)
    failures=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[(row,pool.submit(process,row)) for row in rows]
        for row,future in futures:
            try:print(json.dumps(future.result()),flush=True)
            except FileNotFoundError as error:failures.append((row['task'],str(error),'missing'))
            except Exception as error:failures.append((row['task'],repr(error),'invalid'))
    print(json.dumps(summary(a.root,a.manifest,failed_tasks={row[0] for row in failures})),flush=True)
    if failures:
        durable_json(a.root/'transfer-records/failures.json',{'time':time.time(),'failures':failures,'source_cleanup_authorized':False})
        print(json.dumps({'failures':failures}),file=sys.stderr)
        raise SystemExit(3 if all(x[2]=='missing' for x in failures) else 1)


def digest_copy(src,dst,expected_bytes):
    h=hashlib.sha256();size=0
    while True:
        block=src.read(8*1024**2)
        if not block:break
        size+=len(block)
        if size>expected_bytes:raise ValueError('Incoming stream exceeds expected size')
        dst.write(block);h.update(block)
    return {'bytes':size,'sha256':h.hexdigest()}


if __name__=='__main__':main()
