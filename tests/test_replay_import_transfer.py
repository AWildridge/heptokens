"""Offline transfer tests: preserve data, reject corruption, record partial imports."""
import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'heptokens/tools'))
from replay_bundle import pack,digest
from replay_import_transfer import import_one,load_snapshot,verify_import,summary,sha,safe_path,digest_copy
from replay_local_handoff import begin_export,export_chunk,end_export,store_ack


def fixture(tmp_path,bundle=True):
    source=tmp_path/'origin/sources/00007';chunk=source/'chunk-000000';chunk.mkdir(parents=True)
    original={'task':7,'dataset':'original-only'};truthsha='d'*64;frozen='e'*64
    payloads={'CMS.h5':b'cms-bytes','ATLAS.h5':b'atlas-bytes','validation.json':json.dumps({'status':'passed','detectors':{'CMS':{'status':'passed'},'ATLAS':{'status':'passed'}}}).encode()}
    for n,data in payloads.items():(chunk/n).write_bytes(data)
    receipt={'status':'complete','source':original,'events':1,'input_sha256':truthsha,'seed':427000,
             'sha256':{n:hashlib.sha256(v).hexdigest() for n,v in payloads.items()},'bytes':{n:len(v) for n,v in payloads.items()}}
    (chunk/'receipt.json').write_text(json.dumps(receipt))
    complete={'status':'complete','source':original,'source_events':1,'exported_events':1,'validation_limit':None,'manifest_sha256':frozen,'chunks':[{'chunk':'chunk-000000','events':1}]}
    (source/'COMPLETE.json').write_text(json.dumps(complete));(source/'export-index.json').write_text(json.dumps({'exported_events':1,'chunks':[{'sha256':truthsha}]}))
    inventory={str(p.relative_to(source)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in source.rglob('*') if p.is_file()}
    objects={};final=dict(complete,schema='heptokens-remote-source-bundle-v1' if bundle else 'heptokens-remote-source-v1',files=inventory)
    if bundle:
        archive=tmp_path/'origin/source.tar';receiptpath=source/'BUNDLE-MANIFEST.json';report=pack(source,inventory,archive,receiptpath)
        final['archive']=report['archive'];final['bundle_receipt']={'bytes':receiptpath.stat().st_size,'sha256':sha(receiptpath)}
        objects['source.tar']={'url':archive.as_uri(),**report['archive']};objects['BUNDLE-MANIFEST.json']={'url':receiptpath.as_uri(),'bytes':receiptpath.stat().st_size,'sha256':sha(receiptpath)}
    else:
        objects.update({n:{'url':(source/n).as_uri(),**v} for n,v in inventory.items()})
    commit=source/'SOURCE-COMMIT.json';commit.write_text(json.dumps(final));objects['SOURCE-COMMIT.json']={'url':commit.as_uri(),'bytes':commit.stat().st_size,'sha256':sha(commit)}
    row={'task':7,'events':1,'source':original,'objects':objects};snapshot={'sources':[row],'cleanup_authorized':False,'source_manifest_sha256':frozen};manifest=tmp_path/'snapshot.json';manifest.write_text(json.dumps(snapshot));return source,row,manifest,snapshot


@pytest.mark.parametrize('bundle',[True,False])
def test_import_and_independent_verify_preserve_originals(tmp_path,bundle):
    origin,row,path,snapshot=fixture(tmp_path,bundle);root=tmp_path/'imported'
    result=import_one(path,root,row,backend='none',stable_seconds=0)
    assert result['status']=='verified-import' and not result['source_cleanup_authorized']
    assert (origin/'chunk-000000/CMS.h5').read_bytes()==b'cms-bytes'
    assert (root/'training/00007/chunk-000000/CMS.h5').stat().st_ino==(root/'sources/00007/chunk-000000/CMS.h5').stat().st_ino
    assert verify_import(root,row,snapshot,path)['events']==1
    assert not (root/'READY.json').exists() and summary(root,path)['status']=='verified-partial-import'


def test_resume_revalidates_completed_source(tmp_path):
    origin,row,path,snapshot=fixture(tmp_path);root=tmp_path/'imported'
    import_one(path,root,row,stable_seconds=0);import_one(path,root,row,stable_seconds=0)
    assert len(list((root/'transfer-records').glob('source-*.json')))==1


def test_corrupt_cached_archive_is_not_overwritten(tmp_path):
    origin,row,path,snapshot=fixture(tmp_path);root=tmp_path/'imported';import_one(path,root,row,stable_seconds=0)
    archive=root/'objects/00007/source.tar';archive.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Existing destination differs'):import_one(path,root,row,stable_seconds=0)
    assert archive.read_bytes()==b'corrupt'


def test_tampered_payload_and_receipt_cannot_rebind_expected_hash(tmp_path):
    origin,row,path,snapshot=fixture(tmp_path);root=tmp_path/'imported';import_one(path,root,row,stable_seconds=0)
    chunk=root/'sources/00007/chunk-000000';(chunk/'CMS.h5').write_bytes(b'changed')
    r=json.loads((chunk/'receipt.json').read_text());r['sha256']['CMS.h5']=sha(chunk/'CMS.h5');r['bytes']['CMS.h5']=7;(chunk/'receipt.json').write_text(json.dumps(r))
    with pytest.raises(ValueError,match='frozen transport'):verify_import(root,row,snapshot,path)


def test_symlink_destination_refused(tmp_path):
    root=tmp_path/'root';root.mkdir();outside=tmp_path/'outside';outside.mkdir();(root/'sources').symlink_to(outside)
    with pytest.raises(ValueError,match='Symlink'):safe_path(root,'sources/00007')


@pytest.mark.parametrize('bundle',[True,False])
def test_import_without_python39_path_api(tmp_path,monkeypatch,bundle):
    origin,row,path,snapshot=fixture(tmp_path,bundle)
    def unavailable(*args,**kwargs):raise AttributeError('is_relative_to unavailable before Python 3.9')
    monkeypatch.setattr(Path,'is_relative_to',unavailable)
    root=tmp_path/'older-python-import'
    assert import_one(path,root,row,backend='none',stable_seconds=0)['status']=='verified-import'
    assert verify_import(root,row,snapshot,path)['events']==1
    outside=tmp_path/'outside';outside.mkdir();(root/'escape').symlink_to(outside)
    with pytest.raises(ValueError,match='Symlink'):safe_path(root,'escape/data')


def test_duplicate_tasks_refused(tmp_path):
    origin,row,path,snapshot=fixture(tmp_path);snapshot['sources'].append(row);path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError,match='duplicate'):load_snapshot(path)


def test_stream_size_limit(tmp_path):
    import io
    with pytest.raises(ValueError,match='exceeds'):digest_copy(io.BytesIO(b'too many'),io.BytesIO(),1)


def test_acknowledgement_must_match_expected_source(tmp_path):
    origin,row,path,snapshot=fixture(tmp_path);root=tmp_path/'imported';r=import_one(path,root,row,stable_seconds=0)
    store_ack(json.dumps(r),row,path,tmp_path/'work-records');r['events']=2
    with pytest.raises(ValueError,match='acknowledgement'):store_ack(json.dumps(r),row,path,tmp_path/'work-records')


def test_worker_lease_protects_stream_and_original_is_retained(tmp_path,monkeypatch):
    import fcntl
    origin,row,path,snapshot=fixture(tmp_path);scratch=tmp_path/'scratch';archive=scratch/'data/committed/bundles/00007.tar';archive.parent.mkdir(parents=True)
    archive.write_bytes((tmp_path/'origin/source.tar').read_bytes());lease=tmp_path/'purdue-setup/production/worker-local/source-leases/00007.lock';lease.parent.mkdir(parents=True)
    worker=SimpleNamespace();monkeypatch.setitem(sys.modules,'distributed',SimpleNamespace(get_worker=lambda:worker))
    current={'scratch':str(scratch),'source':str(origin),'task':7};token=begin_export(str(tmp_path),current,row['objects']['source.tar'])
    try:
        with lease.open('a+') as f:
            with pytest.raises(BlockingIOError):fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert export_chunk(token,dask_worker=worker)==archive.read_bytes()
    finally:end_export(token,dask_worker=worker)
    with lease.open('a+') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert archive.exists()


def test_offline_receive_seed_import_and_reverify_cli(tmp_path):
    import subprocess
    origin,row,path,snapshot=fixture(tmp_path);root=tmp_path/'imported';tools=ROOT/'heptokens/tools/replay_import_transfer.py'
    for name,item in row['objects'].items():
        if name=='source.tar':item['url']='worker-scratch://attempt/source.tar';continue
        folder=tmp_path/'local-objects/00007';folder.mkdir(parents=True,exist_ok=True)
        (folder/name).write_bytes((origin/name).read_bytes());item['url']='kit-local://00007/'+name
    path.write_text(json.dumps(snapshot))
    base=[sys.executable,str(tools),'--manifest',str(path),'--root',str(root),'--task','7','--workers','1','--stable-seconds','1']
    def run(mode,**kwargs):
        return subprocess.run([base[0],base[1],mode,*base[2:]],capture_output=True,check=True,**kwargs)
    run('seed-metadata');run('receive',input=(tmp_path/'origin/source.tar').read_bytes());run('import-one');run('verify')
    assert summary(root,path)['verified_sources']==1 and not (root/'READY.json').exists()


def test_dry_run_does_not_create_destination(tmp_path):
    import subprocess
    origin,row,path,snapshot=fixture(tmp_path);target=tmp_path/'absent'
    p=subprocess.run([sys.executable,str(ROOT/'heptokens/tools/replay_import_transfer.py'),'pull','--manifest',str(path),'--root',str(target),'--dry-run'],capture_output=True,text=True,check=True)
    assert json.loads(p.stdout)['network_connections']==0 and not target.exists()


def test_pruned_import_keeps_originals_and_verifies_without_archive(tmp_path,monkeypatch):
    import replay_import_transfer as transfer
    monkeypatch.setattr(transfer.time,'sleep',lambda seconds:None)
    origin,row,path,snapshot=fixture(tmp_path);root=tmp_path/'imported'
    result=import_one(path,root,row,stable_seconds=1,prune_cache=True)
    assert result['archive_cache']=='pruned-after-verified-import'
    assert not (root/'objects/00007/source.tar').exists()
    assert (tmp_path/'origin/source.tar').exists()
    assert (origin/'chunk-000000/CMS.h5').exists()
    proof=json.loads((root/'transfer-records/archive-cache-00007.json').read_text())
    assert proof['status']=='archive-cache-pruned'
    assert proof['archive']==row['objects']['source.tar']
    assert proof['verification']['stable_interval_seconds']==1
    assert not proof['source_cleanup_authorized']
    assert verify_import(root,row,snapshot,path,1)['events']==1
    assert summary(root,path)['archive_cache_pruned_sources']==1
    monkeypatch.setattr(transfer,'fetch',lambda *args:pytest.fail('Pruned resume must not fetch objects'))
    assert import_one(path,root,row,stable_seconds=1,prune_cache=True)['events']==1
    assert import_one(path,root,row,stable_seconds=1)['events']==1


@pytest.mark.parametrize('bundle',[True,False])
def test_pruning_requires_positive_stability_and_preserves_legacy_objects(tmp_path,monkeypatch,bundle):
    import replay_import_transfer as transfer
    monkeypatch.setattr(transfer.time,'sleep',lambda seconds:None)
    origin,row,path,snapshot=fixture(tmp_path,bundle);root=tmp_path/'imported'
    with pytest.raises(ValueError,match='positive stability'):
        import_one(path,root,row,stable_seconds=0,prune_cache=True)
    result=import_one(path,root,row,stable_seconds=1,prune_cache=True)
    if not bundle:
        assert result['archive_cache']=='not-applicable'
        for name in row['objects']:assert (root/'objects/00007'/name).exists()
        assert not (root/'transfer-records/archive-cache-00007.json').exists()


def test_missing_archive_requires_eviction_evidence(tmp_path):
    origin,row,path,snapshot=fixture(tmp_path);root=tmp_path/'imported'
    import_one(path,root,row,stable_seconds=0)
    (root/'objects/00007/source.tar').unlink()
    with pytest.raises(ValueError,match='without verified eviction evidence'):
        verify_import(root,row,snapshot,path)
    assert summary(root,path)['status']=='incomplete-transfer'


def test_corrupt_payload_after_pruning_is_rejected_and_success_invalidated(tmp_path,monkeypatch):
    import replay_import_transfer as transfer
    monkeypatch.setattr(transfer.time,'sleep',lambda seconds:None)
    origin,row,path,snapshot=fixture(tmp_path);root=tmp_path/'imported'
    import_one(path,root,row,stable_seconds=1,prune_cache=True)
    assert summary(root,path)['status']=='verified-partial-import'
    marker=root/('TRANSFER-VERIFIED-'+sha(path)[:16]+'.json');assert marker.exists()
    (root/'sources/00007/chunk-000000/CMS.h5').write_bytes(b'bad')
    with pytest.raises(ValueError,match='frozen transport'):
        verify_import(root,row,snapshot,path)
    assert summary(root,path)['status']=='incomplete-transfer' and not marker.exists()


def test_failed_independent_readback_retains_archive(tmp_path,monkeypatch):
    import replay_import_transfer as transfer
    calls=[]
    def corrupt_during_stability(seconds):
        calls.append(seconds)
        if len(calls)==2:(root/'sources/00007/chunk-000000/CMS.h5').write_bytes(b'bad')
    monkeypatch.setattr(transfer.time,'sleep',corrupt_during_stability)
    origin,row,path,snapshot=fixture(tmp_path);root=tmp_path/'imported'
    with pytest.raises(ValueError,match='unstable'):
        import_one(path,root,row,stable_seconds=1,prune_cache=True)
    assert len(calls)==2 and (root/'objects/00007/source.tar').exists()
    assert not (root/'transfer-records/archive-cache-00007.json').exists()
    assert summary(root,path)['status']=='incomplete-transfer'


def test_proof_is_durable_before_unlink_and_interruption_can_resume(tmp_path,monkeypatch):
    import replay_import_transfer as transfer
    monkeypatch.setattr(transfer.time,'sleep',lambda seconds:None)
    origin,row,path,snapshot=fixture(tmp_path);root=tmp_path/'imported'
    real=transfer.durable_json
    def interrupt_after_unlink(target,data):
        if data.get('status')=='archive-cache-pruned':
            assert not (root/'objects/00007/source.tar').exists()
            proof=json.loads((root/'transfer-records/archive-cache-00007.json').read_text())
            assert proof['status']=='verified-before-cache-eviction'
            raise OSError('simulated loss after unlink')
        real(target,data)
    monkeypatch.setattr(transfer,'durable_json',interrupt_after_unlink)
    with pytest.raises(OSError,match='simulated loss'):
        import_one(path,root,row,stable_seconds=1,prune_cache=True)
    monkeypatch.setattr(transfer,'durable_json',real)
    monkeypatch.setattr(transfer,'fetch',lambda *args:pytest.fail('Interrupted pruning must not redownload'))
    assert import_one(path,root,row,stable_seconds=1,prune_cache=True)['events']==1


@pytest.mark.parametrize('field,value',[
    ('root','/another-import'),('task',8),('status','unverified'),
    ('archive',{'bytes':0,'sha256':'f'*64,'url':'wrong'}),
])
def test_eviction_evidence_cannot_rebind_to_another_source(tmp_path,monkeypatch,field,value):
    import replay_import_transfer as transfer
    monkeypatch.setattr(transfer.time,'sleep',lambda seconds:None)
    origin,row,path,snapshot=fixture(tmp_path);root=tmp_path/'imported'
    import_one(path,root,row,stable_seconds=1,prune_cache=True)
    p=root/'transfer-records/archive-cache-00007.json';d=json.loads(p.read_text());d[field]=value;p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='eviction evidence differs'):
        verify_import(root,row,snapshot,path)


def test_cli_prune_resume_and_independent_verification(tmp_path):
    import subprocess
    origin,row,path,snapshot=fixture(tmp_path);root=tmp_path/'imported'
    command=[sys.executable,str(ROOT/'heptokens/tools/replay_import_transfer.py'),
             '--manifest',str(path),'--root',str(root),'--workers','1','--stable-seconds','1']
    def run(mode,*extra):
        return subprocess.run([command[0],command[1],mode,*command[2:],*extra],capture_output=True,text=True,check=True)
    run('pull','--prune-archive-cache')
    assert not (root/'objects/00007/source.tar').exists()
    run('pull','--prune-archive-cache');run('verify')
    assert summary(root,path)['archive_cache_pruned_sources']==1


def test_cli_verifier_respects_source_lock(tmp_path):
    import subprocess,fcntl
    origin,row,path,snapshot=fixture(tmp_path);root=tmp_path/'imported'
    import_one(path,root,row,stable_seconds=0)
    with (root/'.source-00007.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        result=subprocess.run([sys.executable,str(ROOT/'heptokens/tools/replay_import_transfer.py'),
            'verify','--manifest',str(path),'--root',str(root),'--stable-seconds','1'],capture_output=True,text=True)
    assert result.returncode!=0
    assert (root/'objects/00007/source.tar').exists()
    assert json.loads((root/'TRANSFER-STATUS.json').read_text())['status']=='incomplete-transfer'


def test_existing_retained_archive_can_be_pruned_without_redownload(tmp_path,monkeypatch):
    import replay_import_transfer as transfer
    monkeypatch.setattr(transfer.time,'sleep',lambda seconds:None)
    origin,row,path,snapshot=fixture(tmp_path);root=tmp_path/'imported'
    import_one(path,root,row,stable_seconds=0)
    assert (root/'objects/00007/source.tar').exists()
    assert import_one(path,root,row,stable_seconds=1,prune_cache=True)['archive_cache']=='pruned-after-verified-import'
    assert not (root/'objects/00007/source.tar').exists()


def test_failed_durable_proof_retains_archive(tmp_path,monkeypatch):
    import replay_import_transfer as transfer
    monkeypatch.setattr(transfer.time,'sleep',lambda seconds:None)
    origin,row,path,snapshot=fixture(tmp_path);root=tmp_path/'imported'
    real=transfer.durable_json
    def fail_proof(target,data):
        if data.get('schema')=='heptokens-import-archive-eviction-v1':raise OSError('proof disk full')
        real(target,data)
    monkeypatch.setattr(transfer,'durable_json',fail_proof)
    with pytest.raises(OSError,match='proof disk full'):
        import_one(path,root,row,stable_seconds=1,prune_cache=True)
    assert (root/'objects/00007/source.tar').exists()
    assert (root/'sources/00007/chunk-000000/CMS.h5').exists()


def test_interruption_before_unlink_retains_verified_archive_and_can_resume(tmp_path,monkeypatch):
    import replay_import_transfer as transfer
    monkeypatch.setattr(transfer.time,'sleep',lambda seconds:None)
    origin,row,path,snapshot=fixture(tmp_path);root=tmp_path/'imported'
    real=Path.unlink
    def fail_unlink(target,*args,**kwargs):
        if target==root/'objects/00007/source.tar':
            assert (root/'transfer-records/archive-cache-00007.json').exists()
            raise OSError('simulated interruption before unlink')
        return real(target,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',fail_unlink)
    with pytest.raises(OSError,match='before unlink'):
        import_one(path,root,row,stable_seconds=1,prune_cache=True)
    assert (root/'objects/00007/source.tar').exists()
    monkeypatch.setattr(Path,'unlink',real)
    assert import_one(path,root,row,stable_seconds=1,prune_cache=True)['events']==1


def test_pruned_source_still_requires_frozen_metadata_and_snapshot(tmp_path,monkeypatch):
    import replay_import_transfer as transfer
    monkeypatch.setattr(transfer.time,'sleep',lambda seconds:None)
    origin,row,path,snapshot=fixture(tmp_path);root=tmp_path/'imported'
    import_one(path,root,row,stable_seconds=1,prune_cache=True)
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='eviction evidence differs'):
        verify_import(root,row,snapshot,path)


def test_missing_metadata_is_not_covered_by_archive_eviction(tmp_path,monkeypatch):
    import replay_import_transfer as transfer
    monkeypatch.setattr(transfer.time,'sleep',lambda seconds:None)
    origin,row,path,snapshot=fixture(tmp_path);root=tmp_path/'imported'
    import_one(path,root,row,stable_seconds=1,prune_cache=True)
    (root/'objects/00007/BUNDLE-MANIFEST.json').unlink()
    with pytest.raises(ValueError,match='Cached object differs'):
        verify_import(root,row,snapshot,path)


@pytest.mark.parametrize('relative',[
    'objects/00007/BUNDLE-MANIFEST.json',
    'objects/00007/source.tar',
    'training/00007/chunk-000000/CMS.h5',
])
def test_corruption_during_final_stability_interval_prevents_eviction(tmp_path,monkeypatch,relative):
    import replay_import_transfer as transfer
    origin,row,path,snapshot=fixture(tmp_path);root=tmp_path/'imported';calls=[]
    def corrupt(seconds):
        calls.append(seconds)
        if len(calls)==2:
            target=root/relative
            if relative.startswith('training/'):
                target.unlink()  # Break the training hardlink without changing source payload.
            target.write_bytes(b'corrupt')
    monkeypatch.setattr(transfer.time,'sleep',corrupt)
    with pytest.raises(ValueError,match='unstable'):
        import_one(path,root,row,stable_seconds=1,prune_cache=True)
    assert (root/'objects/00007/source.tar').exists()
    assert not (root/'transfer-records/archive-cache-00007.json').exists()


def test_archive_replacement_during_proof_write_is_not_deleted(tmp_path,monkeypatch):
    import replay_import_transfer as transfer
    monkeypatch.setattr(transfer.time,'sleep',lambda seconds:None)
    origin,row,path,snapshot=fixture(tmp_path);root=tmp_path/'imported';real=transfer.durable_json
    def replace(target,data):
        real(target,data)
        if data.get('status')=='verified-before-cache-eviction':
            archive=root/'objects/00007/source.tar';archive.unlink();archive.write_bytes(b'new unexpected file')
    monkeypatch.setattr(transfer,'durable_json',replace)
    with pytest.raises(ValueError,match='changed before eviction'):
        import_one(path,root,row,stable_seconds=1,prune_cache=True)
    assert (root/'objects/00007/source.tar').read_bytes()==b'new unexpected file'


@pytest.mark.parametrize('setting',['CURL_CA_BUNDLE','SSL_CERT_FILE'])
def test_curl_uses_explicit_ca_bundle_instead_of_ambient_directory(tmp_path,monkeypatch,setting):
    import replay_import_transfer as transfer
    monkeypatch.delenv('CURL_CA_BUNDLE',raising=False)
    monkeypatch.delenv('SSL_CERT_FILE',raising=False)
    bundle=tmp_path/'trusted-grid-cas.pem';bundle.write_text('public CA test fixture')
    monkeypatch.setenv(setting,str(bundle));monkeypatch.setenv('X509_CERT_DIR','/unrelated/ambient/grid-cas')
    payload=b'verified metadata';item={'url':'https://example.invalid/metadata','bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()};calls=[]
    def copy(command,**kwargs):
        calls.append(command)
        Path(command[command.index('--output')+1]).write_bytes(payload)
    monkeypatch.setattr(transfer.subprocess,'run',copy)
    target=tmp_path/'copy.json';transfer.fetch(item,target,'curl')
    assert target.read_bytes()==payload
    assert calls[0][calls[0].index('--cacert')+1]==str(bundle)
    assert '--capath' not in calls[0] and '--insecure' not in calls[0]
