import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from replay_bundle import pack, verify, digest


def sample(tmp_path):
    source = tmp_path / 'source'
    (source / 'chunk').mkdir(parents=True)
    (source / 'chunk/data').write_bytes(b'validated physics bytes' * 100)
    with (source / 'chunk/data').open('rb') as f:
        inventory = {'chunk/data': digest(f)}
    archive, receipt = tmp_path / 'source.tar', tmp_path / 'receipt.json'
    return source, inventory, archive, receipt


def test_round_trip_deterministic(tmp_path):
    source, inventory, archive, receipt = sample(tmp_path)
    report = pack(source, inventory, archive, receipt)
    assert verify(archive, json.loads(receipt.read_text()))['members'] == 1
    second = pack(source, inventory, tmp_path / 'second.tar', tmp_path / 'second.json')
    assert second == report


def test_wrong_input_hash_not_published(tmp_path):
    source, inventory, archive, receipt = sample(tmp_path)
    inventory['chunk/data']['sha256'] = '0' * 64
    with pytest.raises(ValueError):
        pack(source, inventory, archive, receipt)
    assert not archive.exists() and not receipt.exists()


def test_corrupt_archive(tmp_path):
    source, inventory, archive, receipt = sample(tmp_path)
    report = pack(source, inventory, archive, receipt)
    data = bytearray(archive.read_bytes()); data[512] ^= 1; archive.write_bytes(data)
    with pytest.raises(ValueError, match='Whole archive'):
        verify(archive, report)


@pytest.mark.parametrize('name', ['../outside', '/absolute', 'chunk/../data', 'BUNDLE-MANIFEST.json'])
def test_unsafe_inventory_names(tmp_path, name):
    source, inventory, archive, receipt = sample(tmp_path)
    with pytest.raises(ValueError):
        pack(source, {name: inventory['chunk/data']}, archive, receipt)


def test_symlink_rejected(tmp_path):
    source, inventory, archive, receipt = sample(tmp_path)
    (source / 'link').symlink_to(source / 'chunk/data')
    with pytest.raises(ValueError):
        pack(source, {'link': inventory['chunk/data']}, archive, receipt)


def test_duplicate_member_even_with_matching_outer_digest(tmp_path):
    source, inventory, archive, receipt = sample(tmp_path)
    report = pack(source, inventory, archive, receipt)
    with tarfile.open(archive, 'a') as tar:
        tar.add(source / 'chunk/data', arcname='chunk/data')
    with archive.open('rb') as f:
        report['archive'] = digest(f)
    with pytest.raises(ValueError, match='Duplicate'):
        verify(archive, report)
