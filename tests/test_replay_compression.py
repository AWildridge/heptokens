"""Compression must preserve truth and retain the input after any failure."""
import gzip
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import replay_compression as compression


@pytest.fixture
def truth(tmp_path):
    path = tmp_path/'truth.hepmc'
    data = bytes(range(256))*4096
    path.write_bytes(data)
    return path, data, hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize('backend', ['zlib', 'libdeflate'])
def test_decoded_truth_is_exact_and_input_is_retained(truth, tmp_path, backend):
    source, raw, expected = truth
    library = Path(sys.prefix)/'lib/libdeflate.so.0'
    if backend == 'libdeflate' and not library.exists():
        pytest.skip('Optional native library is not installed')
    target = tmp_path/'truth.hepmc.gz'
    report = compression.compress_gzip(source, target, expected, backend=backend,
        library_path=library, library_sha256=compression.digest(library) if library.exists() else None)
    assert gzip.decompress(target.read_bytes()) == raw
    assert source.read_bytes() == raw
    assert report['decoded_sha256'] == expected


def test_bad_input_hash_never_publishes_output(truth, tmp_path):
    source, raw, _ = truth
    with pytest.raises(ValueError, match='before compression'):
        compression.compress_gzip(source, tmp_path/'result.gz', '0'*64)
    assert source.read_bytes() == raw
    assert not (tmp_path/'result.gz').exists()


def test_oversized_native_input_uses_bounded_streaming_fallback(truth, tmp_path):
    source, raw, expected = truth
    target = tmp_path/'result.gz'
    report = compression.compress_gzip(source, target, expected, backend='libdeflate',
                                       max_input_bytes=1)
    assert report['backend'] == 'zlib-stream-fallback'
    assert gzip.decompress(target.read_bytes()) == raw


def test_corrupt_native_output_is_not_published(truth, tmp_path, monkeypatch):
    source, raw, expected = truth
    monkeypatch.setattr(compression, 'pinned_library', lambda *args: object())
    monkeypatch.setattr(compression, 'native_gzip', lambda source, target, *args: target.write_bytes(b'corrupt'))
    with pytest.raises(gzip.BadGzipFile):
        compression.compress_gzip(source, tmp_path/'result.gz', expected,
            backend='libdeflate', library_path=tmp_path/'fake.so', library_sha256='0'*64)
    assert source.read_bytes() == raw
    assert not (tmp_path/'result.gz').exists()


def test_existing_valid_output_cannot_be_overwritten(truth, tmp_path):
    source, _, expected = truth
    target = tmp_path/'result.gz'
    target.write_bytes(b'preserved')
    with pytest.raises(FileExistsError):
        compression.compress_gzip(source, target, expected)
    assert target.read_bytes() == b'preserved'
