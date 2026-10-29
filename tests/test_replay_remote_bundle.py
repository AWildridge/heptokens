import json
from pathlib import Path
import pytest
from test_replay_remote_source import source
import commit_replay_gfal as remote
import replay_remote_bundle as bundled


def test_bundle_roundtrip_audit_and_idempotent_eviction(source,tmp_path):
    directory,storage,receipt=source;evidence=tmp_path/'WORK'
    proof=bundled.commit_bundle(directory,'davs://example.invalid/bundle',evidence)
    assert len(storage)==3
    assert proof['files']['chunk-000000/CMS.h5']['sha256']==receipt['sha256']['CMS.h5']
    bundled.prune_bundle(directory,evidence)
    assert not (directory/'chunk-000000/CMS.h5').exists()
    bundled.prune_bundle(directory,evidence)
    assert bundled.audit_bundle(directory,receipt['source'],'frozen-source-manifest')==(1,1)
    bundled.restore_bundle(directory)
    assert (directory/'chunk-000000/CMS.h5').read_bytes()==b'compact'
    assert json.loads((directory.parent.parent/'storage-ledger.json').read_text())['reservations']['00004/chunk-000000']==sum(receipt['bytes'].values())


def test_corrupt_remote_tar_retains_every_payload(source,tmp_path):
    directory,storage,_=source;evidence=tmp_path/'WORK'
    bundled.commit_bundle(directory,'davs://example.invalid/bundle',evidence)
    storage['davs://example.invalid/bundle/source.tar']=b'corrupt'
    with pytest.raises(ValueError,match='Remote bundle changed'):
        bundled.prune_bundle(directory,evidence)
    assert (directory/'chunk-000000/CMS.h5').exists()


def test_work_proofs_precede_first_unlink(source,tmp_path,monkeypatch):
    directory,_,_=source;evidence=tmp_path/'WORK'
    bundled.commit_bundle(directory,'davs://example.invalid/bundle',evidence)
    unlink=Path.unlink
    def checked(path,*args,**kwargs):
        if path.name=='CMS.h5':
            assert json.loads((evidence/'remote-source-copy.json').read_text())['archive']['readback']=='passed'
            assert json.loads((evidence/'second-readback.json').read_text())['files']['source.tar']['readback']=='passed'
            assert (evidence/'expected-objects.json').exists()
        return unlink(path,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',checked)
    bundled.prune_bundle(directory,evidence)


def test_failed_work_proof_write_retains_payload(source,tmp_path,monkeypatch):
    directory,_,_=source;evidence=tmp_path/'WORK'
    bundled.commit_bundle(directory,'davs://example.invalid/bundle',evidence)
    write=bundled.atomic
    def fail(path,value):
        if Path(path).name=='second-readback.json':raise OSError('WORK unavailable')
        return write(path,value)
    monkeypatch.setattr(bundled,'atomic',fail)
    with pytest.raises(OSError):bundled.prune_bundle(directory,evidence)
    assert (directory/'chunk-000000/CMS.h5').exists()


def test_interrupted_bundle_eviction_resumes(source,tmp_path,monkeypatch):
    directory,_,_=source;evidence=tmp_path/'WORK'
    bundled.commit_bundle(directory,'davs://example.invalid/bundle',evidence)
    unlink=Path.unlink
    def interrupt(path,*args,**kwargs):
        result=unlink(path,*args,**kwargs)
        if path.name=='CMS.h5':raise RuntimeError('worker interrupted')
        return result
    with monkeypatch.context() as patch:
        patch.setattr(Path,'unlink',interrupt)
        with pytest.raises(RuntimeError):bundled.prune_bundle(directory,evidence)
    bundled.prune_bundle(directory,evidence)
    assert not (directory/'chunk-000000/truth.hepmc.gz').exists()


def test_changed_receipt_blocks_eviction(source,tmp_path):
    directory,_,_=source;evidence=tmp_path/'WORK'
    bundled.commit_bundle(directory,'davs://example.invalid/bundle',evidence)
    path=directory/'chunk-000000/receipt.json';receipt=json.loads(path.read_text());receipt['sha256']['CMS.h5']='0'*64;path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='receipt differs'):bundled.prune_bundle(directory,evidence)
    assert (directory/'chunk-000000/CMS.h5').exists()


def test_corrupt_ledger_blocks_bundle_eviction(source,tmp_path):
    directory,_,_=source;evidence=tmp_path/'WORK'
    bundled.commit_bundle(directory,'davs://example.invalid/bundle',evidence)
    ledger=directory.parent.parent/'storage-ledger.json';d=json.loads(ledger.read_text());d['reservations']['00004/chunk-000000']=0;ledger.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='ledger differs'):bundled.prune_bundle(directory,evidence)
    assert (directory/'chunk-000000/CMS.h5').exists()


def test_prefix_cannot_become_full_bundle(source,tmp_path):
    directory,_,_=source;path=directory/'COMPLETE.json';d=json.loads(path.read_text());d['validation_limit']=1;path.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='Full-source'):bundled.commit_bundle(directory,'davs://example.invalid/bundle',tmp_path/'WORK')
    bundled.commit_bundle(directory,'davs://example.invalid/bundle',tmp_path/'WORK',allow_prefix=True)
    with pytest.raises(ValueError,match='full-source'):bundled.audit_bundle(directory,d['source'],'frozen-source-manifest')
