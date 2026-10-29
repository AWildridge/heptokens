"""Fail-closed checks for portable staging and remote commit behavior."""
import json
import multiprocessing
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import commit_replay_gfal as remote
import replay_runtime as runtime


def lock_attempt(path, connection):
    import fcntl
    with open(path, 'a+') as stream:
        try:
            fcntl.lockf(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            connection.send('acquired')
        except BlockingIOError:
            connection.send('locked')


def test_transfer_slot_is_shared_between_processes(tmp_path):
    context = multiprocessing.get_context('spawn')
    reader, writer = context.Pipe(duplex=False)
    with runtime.transfer_slot(tmp_path, 1):
        child = context.Process(target=lock_attempt, args=(str(tmp_path/'slot-0'), writer))
        child.start()
        assert reader.poll(30)
        assert reader.recv() == 'locked'
        child.join(30)
        assert child.exitcode == 0


def test_site_storage_budget_is_enforced(tmp_path):
    import replay_cms_collection as replay
    replay.reserve(tmp_path, 'first', 10, cap_bytes=10)
    with pytest.raises(RuntimeError, match='capacity guard'):
        replay.reserve(tmp_path, 'second', 1, cap_bytes=10)
    with pytest.raises(ValueError, match='cap changed'):
        replay.reserve(tmp_path, 'first', 10, cap_bytes=20)


def test_failed_stage_records_failure_before_propagating(tmp_path):
    with pytest.raises(RuntimeError, match='injected'):
        with runtime.measure_stage(tmp_path, 'interrupted-commit'):
            raise RuntimeError('injected')
    result = json.loads((tmp_path/'performance.json').read_text())[0]
    assert result['status'] == 'failed' and result['wall_seconds'] >= 0


def test_missing_worker_context_is_not_a_fake_slurm_job(monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    monkeypatch.delenv('HEPTOKENS_EXECUTION', raising=False)
    with pytest.raises(ValueError, match='No worker'):
        runtime.execution_identity()


def test_invalid_cpu_context_is_rejected(monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    monkeypatch.setenv('HEPTOKENS_EXECUTION', json.dumps({'backend': 'dask', 'reserved_cores': 1}))
    with pytest.raises(RuntimeError):
        runtime.execution_identity()


def test_remote_instability_prevents_readback_and_commit(tmp_path, monkeypatch):
    source = tmp_path / 'truth.gz'
    source.write_bytes(b'truth')
    checksums = iter([remote.digest(source, 'adler32'), '00000000'])
    monkeypatch.setattr(remote, 'remote_adler', lambda _: next(checksums))
    monkeypatch.setattr(remote.time, 'sleep', lambda _: None)
    monkeypatch.setattr(remote, 'run', lambda *args: pytest.fail('Must not read back unstable remote data'))
    with pytest.raises(ValueError, match='unstable'):
        remote.publish_file(source, 'root://example.invalid/truth.gz', 1)
    assert source.read_bytes() == b'truth'


def test_corrupt_readback_preserves_local_copy(tmp_path, monkeypatch):
    source = tmp_path / 'truth.gz'
    source.write_bytes(b'truth')
    monkeypatch.setattr(remote, 'remote_adler', lambda _: remote.digest(source, 'adler32'))
    monkeypatch.setattr(remote.time, 'sleep', lambda _: None)
    def corrupt(*command):
        from urllib.parse import urlparse
        Path(urlparse(command[-1]).path).write_bytes(b'other')
    monkeypatch.setattr(remote, 'run', corrupt)
    with pytest.raises(ValueError, match='readback'):
        remote.publish_file(source, 'root://example.invalid/truth.gz', 1)
    assert source.read_bytes() == b'truth'


def test_payloads_precede_remote_receipt(tmp_path, monkeypatch):
    chunk = tmp_path / 'source/chunk-000000'
    chunk.mkdir(parents=True)
    (chunk / 'truth.gz').write_bytes(b'truth')
    (chunk / 'receipt.json').write_text(json.dumps({'status': 'complete', 'sha256': {'truth.gz': remote.digest(chunk/'truth.gz', 'sha256')}}))
    seen = []
    monkeypatch.setattr(remote.time,'sleep',lambda _:None)
    monkeypatch.setattr(remote,'stage_file',lambda path,url:seen.append(path.name) or {'url':url,'bytes':path.stat().st_size,'sha256':remote.digest(path,'sha256')})
    monkeypatch.setattr(remote,'verify_staged_file',lambda path,item,seconds:item)
    monkeypatch.setattr(remote, 'publish_file', lambda path, url, seconds: seen.append(path.name) or {'url': url,'bytes':path.stat().st_size,'sha256':remote.digest(path,'sha256')})
    result = remote.commit(chunk, 'root://example.invalid/chunk')
    assert seen == ['truth.gz', 'receipt.json', 'REMOTE-COMMIT.json']
    assert result['status'] == 'verified-remote-copy'
    assert (chunk/'truth.gz').is_file()


def test_preferred_xrootd_uses_original_uri_and_checks_checksum(tmp_path,monkeypatch):
    import zlib
    import replay_cms_collection as replay
    data=b'original-source'
    entry={'uri':'root://example.invalid/original','url':'https://example.invalid/original',
           'bytes':len(data),'checksum':'adler32:%08x'%(zlib.adler32(data)&0xffffffff)}
    def copy(command,**kwargs):
        assert command[-2]==entry['uri']
        Path(command[-1]).write_bytes(data)
    monkeypatch.setattr(replay.subprocess,'run',copy)
    monkeypatch.setattr(replay.requests,'get',lambda *args,**kwargs:pytest.fail('XRootD succeeded'))
    target=tmp_path/'input.root'
    replay.download(entry,target,prefer_xrootd=True)
    assert target.read_bytes()==data


def test_corrupt_preferred_xrootd_falls_back_to_checked_https(tmp_path,monkeypatch):
    import zlib
    import replay_cms_collection as replay
    data=b'original-source'
    entry={'uri':'root://example.invalid/original','url':'https://example.invalid/original',
           'bytes':len(data),'checksum':'adler32:%08x'%(zlib.adler32(data)&0xffffffff)}
    def corrupt(command,**kwargs):Path(command[-1]).write_bytes(b'corrupt')
    class Response:
        status_code=200
        headers={}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def raise_for_status(self):pass
        def iter_content(self,size):yield data
    monkeypatch.setattr(replay.subprocess,'run',corrupt)
    monkeypatch.setattr(replay.requests,'get',lambda *args,**kwargs:Response())
    target=tmp_path/'input.root'
    replay.download(entry,target,prefer_xrootd=True)
    assert target.read_bytes()==data
    assert len(list(tmp_path.glob('input.root.invalid-*')))==1


def test_storage_cap_serializes_threads_and_process_locks(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import replay_cms_collection as replay
    first_write=threading.Event();release=threading.Event();second_write=threading.Event()
    original=replay.atomic_json
    def delayed(path,data):
        if 'first' in data['reservations'] and 'second' not in data['reservations']:
            first_write.set()
            assert release.wait(5)
        else:second_write.set()
        original(path,data)
    monkeypatch.setattr(replay,'atomic_json',delayed)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first=pool.submit(replay.reserve,tmp_path,'first',10,15)
        assert first_write.wait(5)
        second=pool.submit(replay.reserve,tmp_path,'second',10,15)
        try:assert not second_write.wait(.1)
        finally:release.set()
        first.result()
        with pytest.raises(RuntimeError,match='capacity guard'):second.result()
    assert json.loads((tmp_path/'storage-ledger.json').read_text())['reservations']=={'first':10}
