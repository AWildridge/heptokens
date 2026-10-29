"""Separate GFAL process; two canonical stable full readbacks before cleanup."""
import json
from pathlib import Path
import sys
import time
import os
from commit_replay_gfal import stage_file, verify_staged_file
from run_generator_backfill import atomic

request = json.loads(Path(sys.argv[1]).read_text())
if 'limits' in request:
    os.environ['HEPTOKENS_IO_LIMITS'] = request['limits']
    os.environ['HEPTOKENS_IO_SLOTS'] = request['slots']
source = Path(request['source'])
staged = stage_file(source, request['url'])
time.sleep(30)
first = verify_staged_file(source, staged, 30)
time.sleep(30)
second = verify_staged_file(source, staged, 30)
atomic(request['proof'], dict(status='two-stable-full-readbacks-passed', first=first, second=second))
