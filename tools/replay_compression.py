"""Checked gzip compression with an optional, explicitly pinned fast backend.

This module is not enabled by the current replay driver.  The native backend
keeps the gzip format and bounds its in-memory input; larger chunks stream
through zlib.  Neither backend removes its input.
"""
import ctypes
from functools import lru_cache
import gzip
import hashlib
import os
from pathlib import Path
import shutil
import tempfile


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            result.update(block)
    return result.hexdigest()


@lru_cache(maxsize=8)
def pinned_library(path, expected_sha256):
    if digest(path) != expected_sha256:
        raise ValueError('Compression library SHA256 differs')
    library = ctypes.CDLL(path)
    library.libdeflate_alloc_compressor.argtypes = [ctypes.c_int]
    library.libdeflate_alloc_compressor.restype = ctypes.c_void_p
    library.libdeflate_free_compressor.argtypes = [ctypes.c_void_p]
    library.libdeflate_free_compressor.restype = None
    library.libdeflate_gzip_compress_bound.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    library.libdeflate_gzip_compress_bound.restype = ctypes.c_size_t
    library.libdeflate_gzip_compress.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_size_t, ctypes.c_void_p,
                                              ctypes.c_size_t]
    library.libdeflate_gzip_compress.restype = ctypes.c_size_t
    return library


def native_gzip(source, destination, library, level):
    raw = source.read_bytes()
    compressor = library.libdeflate_alloc_compressor(level)
    if not compressor:
        raise MemoryError('Cannot allocate gzip compressor')
    try:
        bound = library.libdeflate_gzip_compress_bound(compressor, len(raw))
        output = ctypes.create_string_buffer(bound)
        size = library.libdeflate_gzip_compress(
            compressor, ctypes.c_char_p(raw), len(raw), output, bound)
        if not size:
            raise ValueError('Native gzip output buffer is too small')
        with destination.open('wb') as stream:
            stream.write(memoryview(output).cast('B')[:size])
    finally:
        library.libdeflate_free_compressor(compressor)


def compress_gzip(source, target, expected_sha256, *, backend='zlib', level=6,
                  library_path=None, library_sha256=None,
                  max_input_bytes=128*1024**2):
    """Atomically publish only a gzip whose decoded bytes match the input hash."""
    source, target = Path(source), Path(target)
    if backend not in ('zlib', 'libdeflate') or level not in range(1, 10):
        raise ValueError('Unsupported gzip backend or level')
    if max_input_bytes < 1:
        raise ValueError('A positive in-memory compression limit is required')
    if target.exists():
        raise FileExistsError(str(target))
    if digest(source) != expected_sha256:
        raise ValueError('Generator record SHA256 differs before compression')
    actual = backend
    if backend == 'libdeflate' and source.stat().st_size > max_input_bytes:
        actual = 'zlib-stream-fallback'
    library = None
    if actual == 'libdeflate':
        if not library_path or not library_sha256:
            raise ValueError('Native gzip requires an explicit library path and SHA256')
        library_path = str(Path(library_path).resolve())
        library = pinned_library(library_path, library_sha256)
    handle, name = tempfile.mkstemp(prefix=target.name+'.partial-', dir=target.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        if library is not None:
            native_gzip(source, temporary, library, level)
        else:
            with source.open('rb') as inp, temporary.open('wb') as output:
                with gzip.GzipFile(filename=target.name, mode='wb', fileobj=output,
                                   compresslevel=level) as compressed:
                    shutil.copyfileobj(inp, compressed)
        decoded = hashlib.sha256()
        with gzip.open(temporary, 'rb') as stream:
            for block in iter(lambda: stream.read(1024**2), b''):
                decoded.update(block)
        if decoded.hexdigest() != expected_sha256:
            raise ValueError('Compressed generator record failed decoded SHA256 verification')
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        # Atomic no-overwrite publication also protects against concurrent writers.
        os.link(temporary, target)
        report = {'codec': 'gzip', 'backend': actual, 'level': level,
                  'decoded_sha256': expected_sha256, 'sha256': digest(target),
                  'bytes': target.stat().st_size, 'input_bytes': source.stat().st_size}
        if library is not None:
            report.update(library=library_path, library_sha256=library_sha256)
        return report
    finally:
        temporary.unlink(missing_ok=True)
