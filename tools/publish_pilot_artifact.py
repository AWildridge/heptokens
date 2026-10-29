"""Publish a checksummed scientific artifact with two stable full readbacks."""
import argparse
import json
from pathlib import Path
import time
from commit_replay_gfal import digest, publish_file, verify_staged_file
from replay_scratch_budget import durable_json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--destination', required=True)
p.add_argument('--evidence', type=Path, required=True)
a = p.parse_args()
a.evidence.mkdir(exist_ok=True, parents=True)
expected = dict(sha256=digest(a.source,'sha256'), bytes=a.source.stat().st_size, url=a.destination)
durable_json(a.evidence/'expected.json',expected)
first = publish_file(a.source,a.destination,30)
durable_json(a.evidence/'first-readback.json',first)
time.sleep(30)
second = verify_staged_file(a.source,first,30)
if digest(a.source,'sha256') != expected['sha256']:
    raise ValueError('Source changed while publishing; retained')
durable_json(a.evidence/'verified.json',dict(status='stable-and-read-back-twice',
             first=first,second=second,local_deletion=False))
print(json.dumps(expected))
