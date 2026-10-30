"""Separate GFAL process; two canonical stable full readbacks before cleanup."""
import json
from pathlib import Path
import sys
import time
import os
from commit_replay_gfal import stage_file, verify_staged_file, remote_adler, run, digest
from run_generator_backfill import atomic

request = json.loads(Path(sys.argv[1]).read_text())
if 'limits' in request:
    os.environ['HEPTOKENS_IO_LIMITS'] = request['limits']
    os.environ['HEPTOKENS_IO_SLOTS'] = request['slots']
source = Path(request['source'])
if request.get('action')=='restore':
    run('gfal-copy','--checksum','ADLER32',request['url'],source.resolve().as_uri())
    if source.stat().st_size!=request['bytes'] or digest(source,'sha256')!=request['sha256']:
        raise ValueError('Orphan payload differs from recorded validated SHA256')
    atomic(request['proof'],dict(status='restored-exact-validated-bytes',url=request['url'],sha256=request['sha256']))
    sys.exit(0)
staged = stage_file(source, request['url'])
if request.get('checkpoint'):atomic(request['checkpoint'],dict(stage='uploaded',payload=staged))
time.sleep(30)
first = verify_staged_file(source, staged, 30)
if request.get('checkpoint'):atomic(request['checkpoint'],dict(stage='first-readback-passed',payload=staged,first=first))
time.sleep(30)
second = verify_staged_file(source, staged, 30)
atomic(request['proof'], dict(status='two-stable-full-readbacks-passed', first=first, second=second))
