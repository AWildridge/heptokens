"""Freeze pilot evidence/plots with inventory hashes and canonical EOS proofs."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time

from commit_replay_gfal import digest, publish_file, verify_staged_file
from replay_scratch_budget import durable_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--destination',required=True)
    a=p.parse_args()
    directory=a.directory.resolve()
    archive=directory/'evidence-and-plots.tar'
    proof=directory/'evidence-archive-verified.json'
    if proof.exists():
        if json.loads(proof.read_text())['status']!='stable-and-read-back-twice':
            raise ValueError('Invalid evidence archive proof')
        return
    # Status/heartbeat files can change while the watcher is packaging; immutable
    # release-summary.json records the completed scientific checks instead.
    excluded={'status.json','finish-status.json','history.jsonl','run.lock','finish.lock',
              'evidence-and-plots.tar','evidence-and-plots.tar.partial','evidence-archive-expected.json',
              'evidence-archive-first-readback.json','evidence-archive-verified.json',
              'evidence-archive-command.json','evidence-archive-command.log'}
    paths=[]
    for path in sorted(directory.rglob('*')):
        if not path.is_file() or path.is_symlink() or 'scratch' in path.relative_to(directory).parts:
            continue
        if path.parent==directory and path.name in excluded:
            continue
        if path.suffix in {'.root','.pileup','.tar'} or path.name.endswith('.partial'):
            raise ValueError('Unarchived bulk or incomplete payload retained: '+str(path))
        paths.append(path)
    members={str(path.relative_to(directory)):digest(path,'sha256') for path in paths}
    if 'RELEASE-MANIFEST.json' in members:
        raise ValueError('Preserve prior release manifest before rebuilding')
    durable_json(directory/'RELEASE-MANIFEST.json',dict(members_sha256=members))
    paths.append(directory/'RELEASE-MANIFEST.json')
    if archive.exists() or Path(str(archive)+'.partial').exists():
        raise ValueError('Preserve prior evidence archive before rebuilding')
    total=sum(path.stat().st_size for path in paths)
    used=sum(path.stat().st_size for path in directory.rglob('*') if path.is_file())
    if used+2*total>7_500_000_000:
        raise ValueError('Evidence archive/readback exceeds local admission cap')
    with tarfile.open(str(archive)+'.partial','w') as tar:
        for path in paths:
            tar.add(path,arcname=str(path.relative_to(directory)),recursive=False)
    Path(str(archive)+'.partial').rename(archive)
    with tarfile.open(archive,'r') as tar:
        if set(tar.getnames())!={*members,'RELEASE-MANIFEST.json'}:
            raise ValueError('Evidence tar inventory differs')
        for name,expected in members.items():
            with tar.extractfile(name) as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=expected:
                    raise ValueError('Evidence tar member differs')
    expected=dict(sha256=digest(archive,'sha256'),bytes=archive.stat().st_size,url=a.destination,members_sha256=members)
    durable_json(directory/'evidence-archive-expected.json',expected)
    first=publish_file(archive,a.destination,30)
    durable_json(directory/'evidence-archive-first-readback.json',first)
    time.sleep(30)
    second=verify_staged_file(archive,first,30)
    durable_json(proof,dict(status='stable-and-read-back-twice',first=first,second=second,members_sha256=members))
    if digest(archive,'sha256')!=expected['sha256']:
        raise ValueError('Evidence archive changed before redundant cleanup')
    archive.unlink()
    print('Complete evidence and plots twice read back:',a.destination,flush=True)


if __name__=='__main__':
    main()
