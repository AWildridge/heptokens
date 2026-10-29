"""Archive a finished pilot group using the canonical stable GFAL readbacks.

Expected archive/member SHA256 and both remote proofs remain on WORK. Remove
only the exact newly generated pilot ROOTs after two full stable readbacks.
"""
import argparse
import json
from pathlib import Path
import tarfile
import time

from commit_replay_gfal import digest, publish_file, verify_staged_file
from replay_scratch_budget import durable_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--destination', required=True)
    p.add_argument('--include-pileup', action='store_true', help='Also preserve each scheduled bank and its mapping proof')
    a = p.parse_args()
    proof_path = a.group / 'archive-verified.json'
    if proof_path.exists():
        proof = json.loads(proof_path.read_text())
        if proof['status'] != 'stable-and-read-back-twice':
            raise ValueError('Invalid previous archive proof')
        return
    sharing_proof = a.group / 'scheduled-sharing-audit.json'
    if a.include_pileup:
        from audit_scheduled_pileup_mapping import audit_group
        audit_group(a.group,
                    a.group.parents[2] / 'provenance/matched-schedule/index.json',
                    a.group.parents[1] / 'interaction-ledger.sqlite', sharing_proof)
    roots = sorted(a.group.glob('*/delphes.root'))
    if not roots:
        raise ValueError('No finished pilot ROOTs to archive')
    members = {}
    if a.include_pileup:
        members[sharing_proof.name] = digest(sharing_proof, 'sha256')
    for root in roots:
        summary = json.loads((root.parent / 'summary.json').read_text())
        if digest(root, 'sha256') != summary['root_sha256']:
            raise ValueError('Native pilot ROOT identity differs')
        for path in [root, root.parent / 'summary.json', root.parent / 'card.tcl']:
            members[str(path.relative_to(a.group))] = digest(path, 'sha256')
        if a.include_pileup:
            bank = root.parent / 'bank.pileup'
            mapping = root.parent / 'bank.pileup.json'
            proof = json.loads(mapping.read_text())
            if digest(bank, 'sha256') != proof['sha256'] or proof['sha256'] != summary['identity']['library_sha256']:
                raise ValueError('Scheduled bank identity differs from native run')
            for path in [bank, mapping]:
                members[str(path.relative_to(a.group))] = digest(path, 'sha256')
    durable_json(a.group / 'ARCHIVE-MANIFEST.json', {'members_sha256': members, 'events': sum(json.loads((r.parent / 'summary.json').read_text())['events'] for r in roots)})
    if not a.archive.exists():
        a.archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(str(a.archive) + '.partial', 'w') as archive:
            for name in [*members, 'ARCHIVE-MANIFEST.json']:
                archive.add(a.group / name, arcname=name, recursive=False)
        Path(str(a.archive) + '.partial').rename(a.archive)
    with tarfile.open(a.archive, 'r') as archive:
        actual = {x.name for x in archive.getmembers()}
        if actual != {*members, 'ARCHIVE-MANIFEST.json'}:
            raise ValueError('Archive member inventory differs')
        import hashlib
        for name, expected in members.items():
            with archive.extractfile(name) as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                    raise ValueError('Archive member bytes differ')
    expected = {'sha256': digest(a.archive, 'sha256'), 'bytes': a.archive.stat().st_size,
                'url': a.destination, 'members_sha256': members}
    durable_json(a.group / 'archive-expected.json', expected)
    first = publish_file(a.archive, a.destination, 30)
    durable_json(a.group / 'archive-first-readback.json', first)
    time.sleep(30)
    second = verify_staged_file(a.archive, first, 30)
    proof = {'status': 'stable-and-read-back-twice', 'time': time.time(),
             'first': first, 'second': second, 'members_sha256': members,
             'scheduled_banks_included': a.include_pileup,
             'local_cleanup': 'Only exact new pilot ROOT members, optional scheduled-bank members, and redundant tar after both verified readbacks'}
    durable_json(proof_path, proof)
    for root in roots:
        if digest(root, 'sha256') != members[str(root.relative_to(a.group))]:
            raise ValueError('Pilot ROOT changed before cleanup')
        root.unlink()
        if a.include_pileup:
            bank = root.parent / 'bank.pileup'
            if digest(bank, 'sha256') != members[str(bank.relative_to(a.group))]:
                raise ValueError('Scheduled bank changed before cleanup')
            bank.unlink()
    if digest(a.archive, 'sha256') != expected['sha256']:
        raise ValueError('Archive changed before cleanup')
    a.archive.unlink()
    print('Archived and twice verified:', a.destination, flush=True)


if __name__ == '__main__':
    main()
