"""Shared node admission and durable transfer evidence for disposable workers."""
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import replay_worker_local as local
import commit_replay_gfal as remote
from test_replay_remote_source import source


def test_workers_on_one_node_share_disk_admission(tmp_path):
    snapshot = {'node': 'node-a', 'total': 300*local.GIB, 'available': 55*local.GIB}
    first = local.admit_disk(tmp_path, 'worker-a', 15*local.GIB, snapshot)
    assert first
    assert local.admit_disk(tmp_path, 'worker-b', 15*local.GIB, snapshot) is None
    other = dict(snapshot, node='node-b')
    assert local.admit_disk(tmp_path, 'worker-b', 15*local.GIB, other)
    local.release_disk(tmp_path, first)
    assert local.admit_disk(tmp_path, 'worker-c', 15*local.GIB, snapshot)


def test_both_readback_proofs_exist_before_payload_deletion(source, tmp_path, monkeypatch):
    directory, _, receipt = source
    evidence = tmp_path/'WORK-evidence'
    unlink = Path.unlink
    def checked_unlink(path, *args, **kwargs):
        if path.name in receipt['sha256']:
            first = json.loads((evidence/'remote-source-copy.json').read_text())
            second = json.loads((evidence/'second-readback.json').read_text())
            assert second['status'] == 'verified-before-local-eviction'
            assert second['files']['chunk-000000/'+path.name]['sha256'] == receipt['sha256'][path.name]
            assert first['files']['chunk-000000/'+path.name]['sha256'] == receipt['sha256'][path.name]
            assert (evidence/'expected-files.json').is_file()
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', checked_unlink)
    local.transfer_source(directory, 'davs://example.invalid/attempt-1', evidence)
    assert not (directory/'chunk-000000/CMS.h5').exists()
    assert (evidence/'SOURCE-COMMIT.json').is_file()
    assert (evidence/'chunk-000000/receipt.json').is_file()


def test_second_readback_failure_keeps_payload_and_expected_hashes(source, tmp_path, monkeypatch):
    directory, _, _ = source
    evidence = tmp_path/'WORK-evidence'
    verify = remote.verify_staged_file
    def fail_second(path, item, seconds):
        if (evidence/'remote-source-copy.json').exists():
            raise ValueError('injected second readback failure')
        return verify(path, item, seconds)
    monkeypatch.setattr(remote, 'verify_staged_file', fail_second)
    with pytest.raises(ValueError, match='second readback'):
        local.transfer_source(directory, 'davs://example.invalid/attempt-2', evidence)
    assert (directory/'chunk-000000/CMS.h5').exists()
    assert (directory/'chunk-000000/truth.hepmc.gz').exists()
    assert (evidence/'expected-files.json').is_file()
    assert (evidence/'remote-source-copy.json').is_file()


def test_interrupted_prune_reuses_remote_proof_and_preserves_expected_inventory(source, tmp_path, monkeypatch):
    directory, _, receipt = source
    evidence = tmp_path/'WORK-evidence'
    unlink = Path.unlink
    def interrupt(path, *args, **kwargs):
        result = unlink(path, *args, **kwargs)
        if path.name == 'CMS.h5': raise RuntimeError('injected loss during prune')
        return result
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'unlink', interrupt)
        with pytest.raises(RuntimeError, match='during prune'):
            local.transfer_source(directory, 'davs://example.invalid/attempt-3', evidence)
    expected_before = (evidence/'expected-files.json').read_bytes()
    local.transfer_source(directory, 'davs://example.invalid/attempt-3', evidence)
    assert (evidence/'expected-files.json').read_bytes() == expected_before
    assert json.loads(expected_before)['chunk-000000/CMS.h5']['sha256'] == receipt['sha256']['CMS.h5']
    assert not (directory/'chunk-000000/truth.hepmc.gz').exists()
