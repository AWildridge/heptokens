"""Source-level archive, eviction, and restore failure/recovery checks."""
import json
from pathlib import Path
import sys
from urllib.parse import urlparse
import zlib

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import commit_replay_gfal as remote


@pytest.fixture
def source(tmp_path,monkeypatch):
    root=tmp_path/'staging';directory=root/'sources/00004';chunk=directory/'chunk-000000'
    chunk.mkdir(parents=True)
    entry={'task':4,'uri':'root://example.invalid/original','bytes':10,'checksum':'adler32:00000001'}
    payloads={'CMS.h5':b'compact','truth.hepmc.gz':b'truth',
              'validation.json':json.dumps({'status':'passed','detectors':{'CMS':{'status':'passed'},'ATLAS':{'status':'passed'}}}).encode()}
    for name,data in payloads.items():(chunk/name).write_bytes(data)
    receipt={'status':'complete','source':entry,'events':1,'input_sha256':'original-truth-sha',
             'seed':424000,'sha256':{name:remote.digest(chunk/name,'sha256') for name in payloads},
             'bytes':{name:len(data) for name,data in payloads.items()}}
    (chunk/'receipt.json').write_text(json.dumps(receipt))
    (directory/'export-index.json').write_text(json.dumps({'exported_events':1,'chunks':[{'events':1,'sha256':'original-truth-sha'}]}))
    (directory/'COMPLETE.json').write_text(json.dumps({'status':'complete','source':entry,'source_events':1,
        'exported_events':1,'validation_limit':None,'manifest_sha256':'frozen-source-manifest',
        'chunks':[{'chunk':'chunk-000000','events':1}]}))
    (root/'storage-ledger.json').write_text(json.dumps({'cap_bytes':10000,'reservations':{'00004/chunk-000000':sum(receipt['bytes'].values())}}))
    storage={}
    def adler(data):return '%08x'%(zlib.adler32(data)&0xffffffff)
    def stage(path,url):
        data=path.read_bytes()
        if url in storage and storage[url]!=data:raise ValueError('Existing remote bytes differ')
        storage[url]=data
        return {'url':url,'sha256':remote.digest(path,'sha256'),'bytes':len(data),'adler32':adler(data)}
    def verify(path,item,seconds):
        import hashlib
        data=storage[item['url']]
        if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:
            raise ValueError('Remote readback differs')
        return dict(item,readback='passed')
    def run(*command):
        assert command[0]=='gfal-copy'
        Path(urlparse(command[-1]).path).write_bytes(storage[command[-2]])
    monkeypatch.setattr(remote,'stage_file',stage)
    monkeypatch.setattr(remote,'verify_staged_file',verify)
    monkeypatch.setattr(remote,'remote_adler',lambda url:adler(storage[url]))
    monkeypatch.setattr(remote,'run',run)
    monkeypatch.setattr(remote.time,'sleep',lambda _:None)
    return directory,storage,receipt


@pytest.mark.parametrize('file_workers',[1,2])
def test_source_roundtrip_prunes_only_after_verification_and_restores(source,file_workers):
    directory,storage,receipt=source
    proof=remote.commit_source(directory,'davs://example.invalid/source',file_workers=file_workers)
    assert proof['events']==1 and (directory/'chunk-000000/CMS.h5').exists()
    remote.prune_source_payloads(directory,file_workers=file_workers)
    assert not (directory/'chunk-000000/CMS.h5').exists()
    assert (directory/'chunk-000000/receipt.json').exists()
    ledger=json.loads((directory.parent.parent/'storage-ledger.json').read_text())
    assert ledger['reservations']['00004/chunk-000000']==0
    remote.restore_source_payloads(directory)
    assert (directory/'chunk-000000/CMS.h5').read_bytes()==b'compact'
    assert json.loads((directory.parent.parent/'storage-ledger.json').read_text())['reservations']['00004/chunk-000000']==sum(receipt['bytes'].values())
    remote.prune_source_payloads(directory)
    assert not (directory/'chunk-000000/CMS.h5').exists()


@pytest.mark.parametrize('file_workers',[1,2])
def test_changed_remote_payload_prevents_any_local_deletion(source,file_workers):
    directory,storage,_=source
    remote.commit_source(directory,'davs://example.invalid/source')
    storage['davs://example.invalid/source/chunk-000000/CMS.h5']=b'corrupt'
    with pytest.raises(ValueError,match='Remote payload changed'):
        remote.prune_source_payloads(directory,file_workers=file_workers)
    assert (directory/'chunk-000000/CMS.h5').exists()
    assert (directory/'chunk-000000/truth.hepmc.gz').exists()


def test_interrupted_eviction_is_resumable_without_missing_remote_evidence(source,monkeypatch):
    directory,_,_=source
    remote.commit_source(directory,'davs://example.invalid/source')
    original=Path.unlink
    def interrupt(path,*args,**kwargs):
        if path.name=='truth.hepmc.gz':raise RuntimeError('interrupted eviction')
        return original(path,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path,'unlink',interrupt)
        with pytest.raises(RuntimeError,match='interrupted eviction'):remote.prune_source_payloads(directory)
    remote.prune_source_payloads(directory)
    assert not (directory/'chunk-000000/CMS.h5').exists()
    assert not (directory/'chunk-000000/truth.hepmc.gz').exists()


def test_corrupt_local_ledger_stops_before_deletion(source):
    directory,_,_=source
    remote.commit_source(directory,'davs://example.invalid/source')
    ledger=directory.parent.parent/'storage-ledger.json'
    data=json.loads(ledger.read_text());data['reservations']['00004/chunk-000000']=0;ledger.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='ledger differs'):remote.prune_source_payloads(directory)
    assert (directory/'chunk-000000/CMS.h5').exists()


def test_remote_collection_audit_accepts_pruned_verified_source(source,monkeypatch):
    import verify_cms_replay_collection as audit
    directory,_,receipt=source
    remote.commit_source(directory,'davs://example.invalid/source')
    remote.prune_source_payloads(directory)
    for name in ['remote_adler','verify_staged_file','run']:
        monkeypatch.setattr(audit,name,getattr(remote,name))
    assert audit.verify_source(directory,receipt['source'],'frozen-source-manifest',remote=True)==(1,1)


def test_remote_collection_audit_rejects_changed_committed_bytes(source,monkeypatch):
    import verify_cms_replay_collection as audit
    directory,storage,receipt=source
    remote.commit_source(directory,'davs://example.invalid/source')
    remote.prune_source_payloads(directory)
    for name in ['remote_adler','verify_staged_file','run']:
        monkeypatch.setattr(audit,name,getattr(remote,name))
    storage['davs://example.invalid/source/chunk-000000/CMS.h5']=b'changed'
    with pytest.raises(ValueError,match='Remote file changed'):
        audit.verify_source(directory,receipt['source'],'frozen-source-manifest',remote=True)


def test_prefix_can_never_pass_full_source_audit(source):
    import verify_cms_replay_collection as audit
    directory,_,receipt=source
    path=directory/'COMPLETE.json'
    data=json.loads(path.read_text());data['validation_limit']=1;path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Invalid full-source completion'):
        audit.verify_source(directory,receipt['source'],'frozen-source-manifest',remote=True)


def test_parallel_source_file_operations_respect_the_bound(source,monkeypatch):
    import threading
    directory,_,_=source
    original=remote.verify_staged_file;lock=threading.Lock();active=peak=0
    def measured(*args):
        nonlocal active,peak
        with lock:active+=1;peak=max(peak,active)
        try:
            threading.Event().wait(.01)
            return original(*args)
        finally:
            with lock:active-=1
    monkeypatch.setattr(remote,'verify_staged_file',measured)
    remote.commit_source(directory,'davs://example.invalid/source',file_workers=2)
    assert peak==2
    remote.prune_source_payloads(directory,file_workers=2)
    assert peak==2 and active==0
