import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import replay_scratch_budget as budget


def load_helper():
    path = ROOT / 'tools/replay_worker_local.py'
    spec = importlib.util.spec_from_file_location('scratch_budget_subject', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


@pytest.fixture
def setup(tmp_path):
    scratch = tmp_path / 'attempt'
    scratch.mkdir()
    node = tmp_path / 'ledger/node'
    node.mkdir(parents=True)
    budget.durable_json(node / 'reservations.json', {'owner': 60})
    return {'scratch': str(scratch), 'key': 'owner', 'node_root': str(node),
            'manifest': str(tmp_path / 'work/pinned-files.json')}


def make_file(config, name='input.root', size=20):
    p = Path(config['scratch']) / name
    p.write_bytes(b'x' * size)
    return p


def test_credit_bytes_remain_after_original_unlink(setup):
    source = make_file(setup)
    budget.pin_files(setup, [source])
    manifest = json.loads(Path(setup['manifest']).read_text())
    pin = Path(next(iter(manifest['files'].values()))['pin'])
    assert os.stat(source).st_ino == os.stat(pin).st_ino
    source.unlink()
    assert pin.read_bytes() == b'x' * 20
    assert budget.credit_total(setup['node_root'], {'owner': 60}) == 20
    assert budget.physical_bytes(setup['scratch']) == 20


def test_restart_does_not_duplicate_credit(setup):
    source = make_file(setup)
    budget.pin_files(setup, [source])
    budget.pin_files(setup, [source])
    assert budget.credit_total(setup['node_root'], {'owner': 60}) == 20
    assert budget.physical_bytes(setup['scratch']) == 20


def test_hardlink_aliases_count_once(setup):
    source = make_file(setup)
    alias = source.with_name('alias')
    os.link(source, alias)
    budget.pin_files(setup, [source, alias])
    assert budget.credit_total(setup['node_root'], {'owner': 60}) == 20
    assert budget.physical_bytes(setup['scratch']) == 20


def test_sparse_holes_do_not_receive_credit(setup):
    source = Path(setup['scratch']) / 'sparse'
    with source.open('wb') as stream:
        stream.seek(1024 * 1024)
        stream.write(b'x')
    node = Path(setup['node_root'])
    budget.durable_json(node / 'reservations.json', {'owner': 2 * 1024 * 1024})
    budget.pin_files(setup, [source])
    assert budget.credit_total(node, {'owner': 2 * 1024 * 1024}) == min(source.stat().st_size, source.stat().st_blocks * 512)
    assert budget.credit_total(node, {'owner': 2 * 1024 * 1024}) < source.stat().st_size


def test_changed_restart_is_rejected_without_more_credit(setup):
    source = make_file(setup)
    budget.pin_files(setup, [source])
    source.unlink()
    source.write_bytes(b'y' * 20)
    with pytest.raises(ValueError, match='differs on restart'):
        budget.pin_files(setup, [source])
    assert budget.credit_total(setup['node_root'], {'owner': 60}) == 20


def test_in_place_mutation_is_rejected(setup):
    source = make_file(setup)
    budget.pin_files(setup, [source])
    source.write_bytes(b'z' * 20)
    with pytest.raises(ValueError, match='differs on restart'):
        budget.pin_files(setup, [source])


def test_export_checksum_mismatch_cannot_grant_credit(setup):
    source = make_file(setup)
    with pytest.raises(ValueError, match='Export checksum'):
        budget.pin_files(setup, [source], expected_hashes={str(source): '0'*64})
    assert budget.credit_total(setup['node_root'], {'owner': 60}) == 0


def test_foreign_paths_and_symlinks_are_rejected(setup, tmp_path):
    foreign = tmp_path / 'outside'
    foreign.write_bytes(b'outside')
    with pytest.raises(ValueError, match='this attempt'):
        budget.pin_files(setup, [foreign])
    link = Path(setup['scratch']) / 'link'
    link.symlink_to(foreign)
    with pytest.raises(ValueError, match='symlink'):
        budget.pin_files(setup, [link])
    assert foreign.read_bytes() == b'outside'


def test_cleanup_interruption_revokes_credit_keeps_budget(setup, monkeypatch):
    source = make_file(setup)
    budget.pin_files(setup, [source])
    import shutil
    def interrupt(path):
        raise OSError('injected interrupted cleanup')
    with monkeypatch.context() as m:
        m.setattr(shutil, 'rmtree', interrupt)
        with pytest.raises(OSError):
            budget.cleanup(Path(setup['node_root']).parent, {'node': 'node', 'key': 'owner'}, setup['scratch'])
    assert budget.credit_total(setup['node_root'], {'owner': 60}) == 0
    assert json.loads((Path(setup['node_root']) / 'reservations.json').read_text()) == {'owner': 60}
    assert source.exists()


def test_successful_cleanup_releases_only_owner(setup):
    source = make_file(setup)
    node = Path(setup['node_root'])
    budget.durable_json(node / 'reservations.json', {'owner': 60, 'another': 10})
    budget.pin_files(setup, [source])
    budget.cleanup(node.parent, {'node': 'node', 'key': 'owner'}, setup['scratch'])
    assert not source.exists()
    assert json.loads((node / 'reservations.json').read_text()) == {'another': 10}
    assert budget.credit_total(node, {'another': 10}) == 0


def test_invalid_credit_fails_closed_and_orphans_do_not_count(setup):
    node = Path(setup['node_root'])
    budget.durable_json(node / 'pinned-credits.json', {'owner': 61})
    with pytest.raises(ValueError, match='Invalid pinned'):
        budget.credit_total(node, {'owner': 60})
    budget.durable_json(node / 'pinned-credits.json', {'absent': 1000})
    assert budget.credit_total(node, {'owner': 60}) == 0


def test_credits_admit_growth_that_already_fits_but_keep_full_peak(setup, monkeypatch):
    source = make_file(setup)
    budget.pin_files(setup, [source])
    helper = load_helper()
    monkeypatch.setattr(helper, 'GIB', 1)
    snapshot = {'node': 'node', 'total': 100, 'available': 80}
    monkeypatch.setattr(helper, 'disk_snapshot', lambda: snapshot)
    node = Path(setup['node_root'])
    assert helper.admit_disk(node.parent, 'next', 20, snapshot=snapshot) is None
    admitted = helper.admit_disk(node.parent, 'next', 20, snapshot=snapshot, pinned_credits=True)
    assert admitted['bytes'] == 20
    assert json.loads((node / 'reservations.json').read_text()) == {'owner': 60, 'next': 20}
    assert helper.admit_disk(node.parent, 'too-much', 10, snapshot=snapshot, pinned_credits=True) is None


def test_pinned_admission_refreshes_free_space_under_ledger_lock(setup, monkeypatch):
    budget.pin_files(setup, [make_file(setup)])
    helper = load_helper()
    monkeypatch.setattr(helper, 'GIB', 1)
    snapshots = iter([{'node': 'node', 'total': 100, 'available': 100},
                      {'node': 'node', 'total': 100, 'available': 80}])
    monkeypatch.setattr(helper, 'disk_snapshot', lambda: next(snapshots))
    # The stale snapshot would admit 35; the current locked snapshot rejects it.
    assert helper.admit_disk(Path(setup['node_root']).parent, 'next', 35, pinned_credits=True) is None
