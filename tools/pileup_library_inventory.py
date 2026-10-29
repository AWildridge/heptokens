"""Exact payload/storage accounting for a completed scheduled pileup arm."""
import argparse
import json
from pathlib import Path
import sqlite3
import struct

from replay_scratch_budget import durable_json


def inventory(directory):
    complete=json.loads((directory/'complete.json').read_text())
    if complete['status']!='complete-native-runs-and-durable-archives-validated':
        raise ValueError('Require a completed arm')
    bank_paths=sorted(directory.glob('groups/*/*/bank.pileup.json'))
    banks=[json.loads(p.read_text()) for p in bank_paths]
    summaries=[json.loads(p.read_text()) for p in directory.glob('groups/*/*/summary.json')]
    proofs=[json.loads(p.read_text()) for p in directory.glob('groups/*/archive-verified.json')]
    if len(banks)!=100 or len(summaries)!=100 or len(proofs)!=20:
        raise ValueError('Incomplete bank/root/archive inventory')
    particles=sum(v['particles'] for v in banks)
    slots=sum(v['events'] for v in banks)
    stored=sum(v['bytes'] for v in banks)
    if slots!=complete['interaction_slots'] or stored!=36*particles+12*slots+8*len(banks):
        raise ValueError('Serialized native particle/index byte conservation failed')
    duplicate_particles=duplicate_slots=shared_ids=0
    with sqlite3.connect('file:'+str((directory/'interaction-ledger.sqlite').resolve())+'?mode=ro',uri=True) as db:
        for payload,uses in db.execute('select r.payload,i.uses from repeated r join interactions i using(idx)'):
            count=struct.unpack_from('>i',payload)[0]
            if len(payload)!=4+36*count or uses<2:
                raise ValueError('Repeated-payload ledger differs')
            duplicate_particles+=(uses-1)*count
            duplicate_slots+=uses-1
            shared_ids+=1
    if duplicate_slots!=complete['repeat_slots']:
        raise ValueError('Repeated native-payload slot accounting failed')
    unique=complete['distinct_physical_interactions']
    unique_particles=particles-duplicate_particles
    result=dict(status='verified-particle-and-byte-conservation',
                distinct_interactions=unique,scheduled_slots=slots,repeated_slots=duplicate_slots,
                interaction_ids_used_more_than_once=shared_ids,
                distinct_interaction_particle_records=unique_particles,
                scheduled_particle_records=particles,repeated_particle_records=duplicate_particles,
                stored_scheduled_bank_bytes=stored,scheduled_bank_shards=100,
                computed_deduplicated_single_native_bank_bytes=36*unique_particles+12*unique+8,
                deduplicated_bank_note='Computed logical size from exact record counts; no monolithic deduplicated file was generated',
                native_root_bytes=sum(v['root_bytes'] for v in summaries),
                durable_tar_bytes=sum(v['first']['bytes'] for v in proofs),durable_tar_archives=20,
                format='4byte particle count +36bytes/particle;8byte offset/interaction and8byte entry-count footer per native shard',
                source_particle_selection='All final/status1 particles, no kinematic cuts; reconstructed selections are separate')
    if directory.name=='generic':
        audits=[json.loads(p.read_text()) for p in directory.glob('sources/*/audit.json')]
        if sum(v['events'] for v in audits)!=unique or sum(v['particles'] for v in audits)!=unique_particles:
            raise ValueError('Independent generic source-to-scheduled particle conservation failed')
        result['independent_generic_source_particle_conservation']='passed'
    schedule=json.loads((directory.parent/'provenance/matched-schedule/index.json').read_text())
    shards=[]
    for path,bank in zip(bank_paths,banks):
        sequence=int(path.parent.name.split('-')[1]);chunk=schedule['chunks'][sequence]
        proof=json.loads((path.parent.parent/'archive-verified.json').read_text())
        stem=path.parent.name
        shards.append(dict(sequence=sequence,hard_source=chunk['task'],hard_chunk=chunk['chunk'],
                           hard_events=1000,interaction_slots=bank['events'],particles=bank['particles'],bytes=bank['bytes'],
                           archive_url=proof['first']['url'],archive_sha256=proof['first']['sha256'],
                           bank_member=stem+'/bank.pileup',bank_sha256=bank['sha256'],
                           root_member=stem+'/delphes.root',root_sha256=proof['members_sha256'][stem+'/delphes.root'],
                           card_member=stem+'/card.tcl',
                           schedule_in_evidence='provenance/matched-schedule/%05d/chunk-%06d.npz'%(chunk['task'],chunk['chunk']),
                           counts_in_evidence='provenance/matched-schedule/%05d/chunk-%06d.counts'%(chunk['task'],chunk['chunk'])))
    durable_json(directory/'library-index.json',dict(
        status='complete-sharded-native-library-with-exact-identity-schedule',
        distinct_interactions=unique,stored_interaction_slots=slots,
        consumption='Use the saved per-hard-event counts with OrderedPileUp=true; read each shard sequentially. Random resampling would not preserve the target reuse.',
        identity_ledger=directory.name+'/interaction-ledger.sqlite',shards=shards))
    result['library_index']='library-index.json'
    durable_json(directory/'library-inventory.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    a=p.parse_args();print(json.dumps(inventory(a.directory),indent=2))


if __name__=='__main__':main()
