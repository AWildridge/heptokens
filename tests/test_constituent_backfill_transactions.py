"""Campaign guards: malicious archives, incomplete inputs and false completion."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from constituent_backfill_worker import atomic, safe_member, source_chunks, verified_group, select_runtime


@pytest.mark.parametrize('name', ['../x', '/x', 'chunk/../../x', './x', 'a//b', ''])
def test_archive_escape_rejected(name):
    with pytest.raises(ValueError):
        safe_member(name)


def test_missing_or_gapped_source_chunk_rejected():
    files = {'chunk-000000/receipt.json': {}, 'chunk-000002/receipt.json': {}}
    with pytest.raises(ValueError):
        source_chunks(files, 3000)
    assert source_chunks({'chunk-000000/receipt.json': {}}, 1000) == ['chunk-000000']


def group():
    proof = dict(sha256='sha', bytes=10, adler32='adler')
    return dict(status='complete', source_task=4, chunks=['chunk-000000'], config_identity='frozen',
                sha256='sha', bytes=10, events=1000,
                views=[dict(events=1000, checked_compact_datasets=list(range(18))) for _ in range(2)],
                remote=dict(status='two-stable-full-readbacks-passed', first=dict(proof), second=dict(proof)))


@pytest.mark.parametrize('mutation', ['digest', 'views', 'parity', 'identity', 'durability'])
def test_false_group_completion_rejected(tmp_path, mutation):
    path = tmp_path / 'complete.json'
    record = group()
    if mutation == 'digest':
        record['remote']['second']['sha256'] = 'other'
    elif mutation == 'views':
        record['views'].pop()
    elif mutation == 'parity':
        record['views'][0]['checked_compact_datasets'].pop()
    elif mutation == 'identity':
        record['config_identity'] = 'other'
    else:
        record['remote']['status'] = 'uploaded-only'
    atomic(path, record)
    with pytest.raises(ValueError):
        verified_group(path, {'identity': 'frozen'}, {'task': 4}, ['chunk-000000'])


def test_verified_group_and_atomic_publication(tmp_path):
    path = tmp_path / 'groups' / 'complete.json'
    atomic(path, group())
    assert verified_group(path, {'identity': 'frozen'}, {'task': 4}, ['chunk-000000'])['events'] == 1000
    assert list(path.parent.iterdir()) == [path]


def test_archived_binary_hashes_choose_runtime_without_substitution():
    original = dict(path='original', binaries=dict(DelphesHepMC2='old-binary', libDelphes='old-library'))
    corrected = dict(path='bounds-v1', binaries=dict(DelphesHepMC2='fixed-binary', libDelphes='fixed-library'))
    config = dict(delphes_runtimes=[original, corrected])
    assert select_runtime(config, None) == original
    assert select_runtime(config, original['binaries']) == original
    assert select_runtime(config, corrected['binaries']) == corrected
    with pytest.raises(ValueError):
        select_runtime(config, dict(DelphesHepMC2='unknown'))
