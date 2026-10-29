"""Independently audit early scheduled-bank archives with one extra readback.

Only the pre-existing archives without the new full-slot proof are read. Their
complete SHA256 must match both earlier remote proofs before payload extraction.
All original archives are retained. Failed local readbacks remain for inspection.
"""
import argparse
import fcntl
import json
from pathlib import Path
import re
import shutil
import tarfile
import tempfile

from audit_scheduled_pileup_mapping import audit_group,sha
from gfal_transport import run
from replay_scratch_budget import durable_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--schedule',type=Path,required=True)
    a=p.parse_args();pilot=a.pilot.resolve()
    results=[]
    with (pilot/'mapping-backfill.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for arm in ['generic','cms']:
            for proof_path in sorted((pilot/arm).glob('groups/*/archive-verified.json')):
                group=proof_path.parent;output=group/'scheduled-sharing-audit.json'
                if output.exists():
                    if json.loads(output.read_text())['status']!='every-slot-matches-scheduled-physical-payload':
                        raise ValueError('Invalid existing mapping proof')
                    continue
                proof=json.loads(proof_path.read_text())
                first,second=proof['first'],proof['second']
                if proof['status']!='stable-and-read-back-twice' or first['sha256']!=second['sha256']:
                    raise ValueError('Earlier archive proof differs')
                used=0
                for path in pilot.rglob('*'):
                    try:
                        if path.is_file():used+=path.stat().st_size
                    except FileNotFoundError:pass
                if used+2*first['bytes']>7_500_000_000:
                    raise ValueError('Mapping readback exceeds local admission cap')
                work=Path(tempfile.mkdtemp(prefix='mapping-backfill-',dir=str(pilot/'scratch')))
                archive=work/'archive.tar'
                run('gfal-copy','--checksum','ADLER32',first['url'],archive.as_uri())
                if archive.stat().st_size!=first['bytes'] or sha(archive)!=first['sha256']:
                    raise ValueError('Third archive readback differs; local copy retained')
                extracted=work/'group';extracted.mkdir()
                with tarfile.open(archive,'r') as tar:
                    for member in tar:
                        if not re.fullmatch(r'sequence-\d{3}/bank\.pileup(?:\.json)?',member.name):
                            continue
                        if not member.isfile():
                            raise ValueError('Scheduled bank archive member is not a regular file')
                        target=extracted/member.name;target.parent.mkdir(exist_ok=True)
                        with tar.extractfile(member) as source,target.open('wb') as destination:
                            shutil.copyfileobj(source,destination,1024*1024)
                        if sha(target)!=proof['members_sha256'][member.name]:
                            raise ValueError('Extracted particle-bank member differs')
                result=audit_group(extracted,a.schedule,pilot/arm/'interaction-ledger.sqlite',output)
                result['additional_full_archive_readback']=dict(url=first['url'],sha256=first['sha256'],bytes=first['bytes'],previous_full_readbacks=2)
                durable_json(output,result)
                results.append(dict(arm=arm,group=group.name,interaction_slots=result['interaction_slots']))
                shutil.rmtree(work)  # Only these redundant, fully checked copies.
                print(json.dumps(results[-1]),flush=True)
        durable_json(pilot/'mapping-backfill-result.json',dict(status='completed',groups=results))


if __name__=='__main__':
    main()
