"""Seekable HTTP byte-range stream used to inspect/sample large public HDF5 files."""
import io
import requests

class HTTPRange(io.RawIOBase):
    def __init__(self, url):
        super().__init__()
        self.url, self.pos = url, 0
        self.session = requests.Session()
        r = self.session.get(url, headers={'Range': 'bytes=0-0'}, timeout=60)
        r.raise_for_status()
        if r.status_code != 206:
            raise ValueError('Server must support Range')
        self.size = int(r.headers['Content-Range'].split('/')[-1])
        self.etag = r.headers.get('ETag')
        self.last_modified = r.headers.get('Last-Modified')
        self.cache = {}
        self.block_size = 1024 * 1024

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError('Invalid whence')
        pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if pos < 0:
            raise ValueError('Negative offset')
        self.pos = pos
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0: return b''
        pieces = []
        while n:
            block = self.pos // self.block_size
            start = block * self.block_size
            end = min(self.size, start + self.block_size) - 1
            if block not in self.cache:
                headers = {'Range': f'bytes={start}-{end}'}
                if self.etag: headers['If-Match'] = self.etag
                r = self.session.get(self.url, headers=headers, timeout=120)
                r.raise_for_status()
                if r.status_code != 206 or len(r.content) != end-start+1 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                    raise ValueError('Invalid range response')
                if len(self.cache) >= 16:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[block] = r.content
            offset = self.pos - start
            piece = self.cache[block][offset:offset+n]
            pieces.append(piece)
            self.pos += len(piece)
            n -= len(piece)
        return b''.join(pieces)

    def readinto(self, b):
        data = self.read(len(b)); b[:len(data)] = data; return len(data)

    def close(self):
        self.session.close()
        super().close()
