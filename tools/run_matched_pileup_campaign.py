"""Bounded CPU pilot with an exact CMS identity-sharing schedule.

Particle replacement is statistical. The frozen default CMS response is retained;
only the separately built ordered merger is added. Durable archives contain ALL
scheduled particle records, native ROOTs, cards and mapping hashes. A SQLite
ledger proves the bijection and exact use counts without retaining bulk banks.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import mmap
import os
from pathlib import Path
import sqlite3
import stat
import struct
import subprocess
import sys
import time
import traceback

import numpy as np
from replay_scratch_budget import durable_json


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Campaign:
    def __init__(self, a):
        self.a=a
        self.root=a.root.resolve()
        self.site=self.root/'purdue-setup'
        self.tools=self.root/'heptokens/tools'
        self.out=a.output.resolve()
        self.out.mkdir(parents=True,exist_ok=True)
        self.schedule=json.loads((a.schedule/'index.json').read_text())
        if sha(self.schedule['identity_file'])!=self.schedule['identity_sha256']:
            raise ValueError('Identity schedule changed')
        self.uses=np.load(self.schedule['identity_file'])['uses']
        self.target=len(self.uses)
        self.env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                      MKL_NUM_THREADS='1',HEPTOKENS_EXECUTION=json.dumps(dict(backend='session',id=os.environ.get('HOSTNAME','purdue'),reserved_cores=16)))
        self.env['TMPDIR']=str(self.out/'scratch')
        (self.out/'scratch').mkdir(exist_ok=True)
        self.global_python='/work/pixi/global/.pixi/envs/default/bin/python'
        self.native_python=str(self.site/'.pixi/envs/default/bin/python')
        self.transfer_python=str(self.site/'transfer-pilot/.pixi/envs/default/bin/python')
        self.delphes=self.site/'software/delphes-28658365-ordered-pileup-v1'
        self.base=self.site/'results/fullsim-delphes-validation-100k-20261021'
        self.parents=json.loads(a.selection.read_text())['files']
        self.pool=ThreadPoolExecutor(max_workers=3)
        self.pending={}
        self.current=None
        self.sources=[]
        self.native_cache={}

    def status(self, phase, **extra):
        result=dict(phase=phase,updated_epoch=time.time(),**extra)
        durable_json(self.out/'status.json',result)
        with (self.out/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(result)+'\n')
        print(json.dumps(result),flush=True)

    def budget(self, reserve=0):
        # Local file accounting is an admission guard, not a quota query. The
        # centrally queried WORK baseline is recorded separately before launch.
        used=0
        for path in self.out.rglob('*'):
            try:
                info=path.stat()
            except FileNotFoundError:
                continue  # A verified redundant readback may have just closed.
            if stat.S_ISREG(info.st_mode):
                used+=info.st_size
        if used+reserve>7_500_000_000:
            raise RuntimeError('7.5GB new-file admission cap reached; retain outputs')
        return used

    def command(self,args,evidence,timeout=3600,transfer=False):
        evidence=Path(evidence)
        evidence.parent.mkdir(parents=True,exist_ok=True)
        args=list(map(str,args))
        if transfer:
            args=['bash','-c','source "$1" || exit; shift; exec "$@"','matched-pilot',
                  str(self.site/'transfer-pilot/env.sh'),*args]
        started=time.time()
        with evidence.with_suffix('.log').open('w') as stream:
            result=subprocess.run(args,env=dict(self.env,HEPTOKENS_GFAL_BACKEND='context'),
                                  stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
        durable_json(evidence,dict(command=args,returncode=result.returncode,seconds=time.time()-started))
        if result.returncode:
            raise RuntimeError('Command failed: '+str(evidence))

    def audit(self,bank,evidence):
        if evidence.exists():
            result=json.loads(evidence.read_text())
            if result['status']!='verified-all-records-and-native-roundtrip' or sha(bank)!=result['sha256']:
                raise ValueError('Saved bank audit changed')
            return result
        self.budget(bank.stat().st_size)
        self.command([self.native_python,self.tools/'audit_delphes_pileup_library.py','--input',bank,
                      '--delphes',self.delphes,'--output',evidence],evidence.with_name(evidence.stem+'-command.json'))
        return json.loads(evidence.read_text())

    def generate(self,index):
        directory=self.out/'generic/sources'/('%03d'%index)
        directory.mkdir(parents=True,exist_ok=True)
        bank=directory/'physical.pileup'
        count=min(50600,self.target-index*50600)
        seed=202610280+index
        reference=self.site/'results/pileup-libraries-equal-size-20261027'
        if not bank.exists():
            self.command([reference/'generate-generic-pileup',reference/'generic.cmnd',bank,count,seed,
                          '/cvmfs/cms.cern.ch/slc7_amd64_gcc700/external/pythia8/240-916c58331a54d89d4e45de43937b7b09/share/Pythia8/xmldoc',
                          directory/'effective-settings.cmnd'],directory/'generation-command.json')
        audit=self.audit(bank,directory/'audit.json')
        generation=json.loads(Path(str(bank)+'.generation.json').read_text())
        if audit['events']!=count or not generation['all_generator_weights_one'] or generation['seed']!=seed:
            raise ValueError('Generic source recipe/count differs')
        metadata=dict(index=index,events=count,seed=seed,bank=str(bank),sha256=audit['sha256'],
                      source_identity='Pythia8.240 seed%d'%seed,physical_definition='13TeV SoftQCD:all Tune:pp5; all final particles; native production vertices/times')
        durable_json(directory/'source.json',metadata)
        return metadata

    def cms(self,index):
        source=self.parents[index]
        directory=self.out/'cms/sources'/('%03d'%index)
        directory.mkdir(parents=True,exist_ok=True)
        bank=directory/'physical.pileup'
        if index==0:
            # The accelerated converter's complete byte comparison and weight
            # audit must pass before this original bank is admitted.
            verified=json.loads((self.a.validation/'comparison.json').read_text())
            if verified['status']!='byte-identical-all-50600-interactions':
                raise ValueError('Accelerated conversion preflight missing')
            original=self.site/'results/pileup-libraries-equal-size-20261027/cms-minbias-50600.pileup'
            bank=original
            summary=json.loads((self.a.validation/'compiled-50600.pileup.summary.json').read_text())
            audit=json.loads((self.site/'results/pileup-libraries-equal-size-20261027/cms-audit.json').read_text())
            if sha(bank)!=summary['sha256'] or sha(bank)!=audit['sha256']:
                raise ValueError('Original verified CMS bank changed')
        else:
            cached=directory/'input.root'
            proof=directory/'verification/verified-input.json'
            if not bank.exists():
                if not cached.exists():
                    self.budget(2*source['bytes']+700_000_000)
                request=dict(url='davs://eos.cms.rcac.purdue.edu:9000'+source['name'],
                             bytes=source['bytes'],adler32=source['adler32'])
                durable_json(directory/'request.json',request)
                if not proof.exists() or not cached.exists():
                    self.command([self.transfer_python,self.tools/'verify_rucio_input.py','--request',directory/'request.json',
                                  '--destination',cached,'--evidence',directory/'verification'],directory/'verification-command.json',transfer=True)
                verified=json.loads(proof.read_text())
                if sha(cached)!=verified['sha256']:
                    raise ValueError('Verified GEN cache changed')
                old=json.loads((self.site/'results/pileup-pilot-100k-20261022/gen-preflight-command.json').read_text())['command']
                command=old[:old.index('python')+1]
                command[3:3]=['-b','/etc/resolv.conf','-b','/etc/hosts']
                command += [self.tools/'export_cms_pileup_native.py','--input',cached,'--output',bank,'--delphes',self.delphes]
                self.command(command,directory/'extraction-command.json',timeout=3600)
            summary=json.loads(Path(str(bank)+'.summary.json').read_text())
            audit=self.audit(bank,directory/'audit.json')
            if cached.exists():
                verified=json.loads(proof.read_text())
                if sha(cached)!=verified['sha256'] or verified['full_readbacks']!=2:
                    raise ValueError('Input cleanup lacks two full remote proofs')
                durable_json(directory/'input-cleanup.json',dict(source_retained=request['url'] if 'request' in locals() else json.loads((directory/'request.json').read_text())['url'],sha256=verified['sha256'],reason='Redundant local cache; original remote GEN verified by two full readbacks; native payload fully audited'))
                cached.unlink()
        if summary['events']!=source['events'] or audit['events']!=source['events'] or not summary['all_gen_weights_one']:
            raise ValueError('CMS GEN catalog/count/unit weights differ')
        metadata=dict(index=index,events=source['events'],bank=str(bank),sha256=audit['sha256'],
                      source_identity=source['did'],event_ids=summary['event_ids'],
                      physical_definition='CMS CP5 status1 GEN, modal xyz removed; unavailable ct=0')
        durable_json(directory/'source.json',metadata)
        return metadata

    def open_source(self,arm,index,entry):
        if self.current is not None:
            self.current[0].close(); self.current[1].close(); self.current=None
        if arm=='generic':
            total=(self.target+50599)//50600
            # At most three generated source files ahead, bounded by admission.
            for j in range(index,min(total,index+3)):
                if j not in self.pending:
                    self.budget(1_200_000_000)
                    self.pending[j]=self.pool.submit(self.generate,j)
            metadata=self.pending.pop(index).result()
        else:
            metadata=self.cms(index)
        stream=Path(metadata['bank']).open('rb')
        bank=mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ)
        events=struct.unpack_from('>q',bank,len(bank)-8)[0]
        if events!=metadata['events']:
            raise ValueError('Source bank event count changed')
        self.current=(bank,stream,metadata,len(bank)-8*(events+1))
        self.native_cache[index]=metadata
        return entry

    def next_record(self,arm,cursor):
        index,entry=cursor['source'],cursor['entry']
        if self.current is None or self.current[2]['index']!=index:
            self.open_source(arm,index,entry)
        bank,stream,metadata,index_start=self.current
        if entry==metadata['events']:
            cursor.update(source=index+1,entry=0)
            return self.next_record(arm,cursor)
        start=struct.unpack_from('>q',bank,index_start+8*entry)[0]
        count=struct.unpack_from('>i',bank,start)[0]
        stop=start+4+36*count
        if not 0<=start<stop<=index_start:
            raise ValueError('Invalid native source boundary')
        record=bank[start:stop]
        if arm=='cms':
            key=':'.join(map(str,metadata['event_ids'][entry]))
        else:
            key=str(metadata['seed'])+':'+str(entry)
        cursor['entry']+=1
        return record,index,entry,key

    def materialize(self,db,arm,chunk,cursor,folder):
        if sha(chunk['schedule'])!=chunk['schedule_sha256'] or sha(chunk['counts'])!=chunk['counts_sha256']:
            raise ValueError('Schedule/counts changed')
        source=np.load(chunk['schedule'])
        indices=source['indices']
        offsets=source['offsets']
        hard=self.base/('%05d'%chunk['task'])/'export'/('chunk-%06d.hepmc'%chunk['chunk'])
        refs=json.loads(Path(str(hard)+'.json').read_text())['events']
        np.testing.assert_array_equal(source['hard_ids'],[[r[k] for k in ['run','lumi','event']] for r in refs])
        np.testing.assert_array_equal(np.diff(offsets),[int(v) for v in Path(chunk['counts']).read_text().split()])
        bank=folder/'bank.pileup'
        temporary=folder/'bank.pileup.partial'
        if temporary.exists():
            raise ValueError('Incomplete bank retained; inspect before retry')
        positions=[];position=particles=0
        mapping_hash=hashlib.sha256()
        new_before=cursor['unique']
        with temporary.open('wb') as output:
            for idx_raw in indices:
                idx=int(idx_raw)
                if idx==cursor['unique']:
                    record,source_index,entry,key=self.next_record(arm,cursor)
                    payload_hash=hashlib.sha256(record).digest()
                    db.execute('insert into interactions values(?,?,?,?,?,?,?)',(idx,source_index,entry,key,payload_hash,1,int(self.uses[idx])))
                    if self.uses[idx]>1:
                        db.execute('insert into repeated values(?,?)',(idx,record))
                    cursor['unique']+=1
                elif idx<cursor['unique']:
                    saved=db.execute('select payload from repeated where idx=?',(idx,)).fetchone()
                    if saved is None:
                        raise ValueError('Unexpected reuse without saved payload')
                    record=saved[0]
                    payload_hash=hashlib.sha256(record).digest()
                    db.execute('update interactions set uses=uses+1 where idx=? and uses<expected_uses',(idx,))
                    if db.execute('select changes()').fetchone()[0]!=1:
                        raise ValueError('Exceeded planned interaction reuse')
                else:
                    raise ValueError('Identity schedule skips a new ordinal')
                positions.append(position)
                output.write(record)
                position+=len(record)
                particles+=struct.unpack_from('>i',record)[0]
                mapping_hash.update(struct.pack('>q',idx)+payload_hash)
            for value in [*positions,len(positions)]:
                output.write(struct.pack('>q',value))
        if cursor['unique']-new_before!=chunk['new_identities']:
            raise ValueError('New identities per chunk differ')
        digest=sha(temporary)
        if bank.exists():
            if sha(bank)!=digest:
                raise ValueError('Preserved bank differs from replayed ledger')
            temporary.unlink()
        else:
            temporary.rename(bank)
        durable_json(folder/'bank.pileup.json',dict(status='all-source-records-verified-schedule-materialized',
                     sha256=digest,bytes=bank.stat().st_size,events=len(indices),particles=particles,
                     unique_new_interactions=chunk['new_identities'],schedule_sha256=chunk['schedule_sha256'],
                     mapping_sha256=mapping_hash.hexdigest(),mapping='In scheduled order hash int64 ordinal followed by SHA256 of exact source event bytes; no particle transformations in assembly'))
        return hard,bank

    def archive(self,arm,group):
        folder=self.out/arm/'groups'/('group-%03d'%group)
        proof=folder/'archive-verified.json'
        if not proof.exists():
            expected=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
            self.budget(2*expected)
            self.command([self.transfer_python,self.tools/'archive_pileup_pilot.py','--group',folder,
                          '--archive',folder.with_suffix('.tar'),'--destination',self.a.destination.rstrip('/')+'/'+arm+'/'+folder.name+'.tar',
                          '--include-pileup'],folder/'archive-command.json',transfer=True)
        verified=json.loads(proof.read_text())
        if verified['status']!='stable-and-read-back-twice':
            raise ValueError('Durable archive proof missing')
        # Closed physical segments are redundant only after all their used
        # records are in the twice-read-back scheduled archives.
        for index,metadata in list(self.native_cache.items()):
            if self.current is not None and index==self.current[2]['index']:
                continue
            path=Path(metadata['bank'])
            if self.out not in path.parents:
                continue  # Never remove the original 50600 banks.
            if path.exists():
                if sha(path)!=metadata['sha256']:
                    raise ValueError('Source segment changed before cleanup')
                durable_json(path.parent/'payload-cleanup.json',dict(reason='Every consumed record now preserved in verified scheduled archives; source identity and audit retained',last_archive=str(proof),sha256=metadata['sha256']))
                path.unlink()

    def run_arm(self,arm):
        directory=self.out/arm
        directory.mkdir(exist_ok=True)
        if (directory/'complete.json').exists():
            return
        self.current=None;self.native_cache={};self.pending={}
        db=sqlite3.connect(str(directory/'interaction-ledger.sqlite'))
        db.execute('pragma synchronous=FULL')
        db.execute('create table if not exists interactions(idx integer primary key, source integer, entry integer, physical_id text unique, payload_sha256 blob unique, uses integer, expected_uses integer)')
        db.execute('create table if not exists repeated(idx integer primary key,payload blob)')
        db.execute('create table if not exists chunks(sequence integer primary key,cursor text,bank_sha256 text)')
        db.commit()
        previous=db.execute('select sequence,cursor from chunks order by sequence desc limit 1').fetchone()
        start=previous[0]+1 if previous else 0
        cursor=json.loads(previous[1]) if previous else dict(source=0,entry=0,unique=0)
        if start and start%5==0:
            self.archive(arm,(start-1)//5)
        for chunk in self.schedule['chunks'][start:]:
            sequence=chunk['sequence']
            group=sequence//5
            folder=directory/'groups'/('group-%03d'%group)/('sequence-%03d'%sequence)
            folder.mkdir(parents=True,exist_ok=True)
            self.status('running',arm=arm,completed_hard_events=sequence*1000,unique_interactions=cursor['unique'],local_bytes=self.budget())
            try:
                hard,bank=self.materialize(db,arm,chunk,cursor,folder)
                # The first complete shard in each arm independently checks all
                # record/index/finite fields AND native round-trip, in addition
                # to the full source-bank audits performed for every segment.
                if sequence==0:
                    self.audit(bank,folder/'scheduled-bank-audit.json')
                self.command([self.global_python,self.tools/'run_delphes_pileup_pilot.py','--input',hard,
                              '--delphes',self.delphes,'--library',bank,'--output',folder,
                              '--mean','22.36408','--sigma-z-m','0.03649213','--mean-z-m','0.009275728',
                              '--seed',str(202610280+sequence),'--base-card','default','--counts',chunk['counts']],folder/'detector-command.json')
                summary=json.loads((folder/'summary.json').read_text())
                if summary['events']!=1000 or summary['status']!='validated':
                    raise ValueError('Native chunk failed validation')
                db.execute('insert into chunks values(?,?,?)',(sequence,json.dumps(cursor),sha(bank)))
                db.commit()
            except BaseException:
                db.rollback()
                raise
            if (sequence+1)%5==0:
                self.archive(arm,group)
        frequency=dict(db.execute('select uses,count(*) from interactions group by uses'))
        unique,draws,mismatches=db.execute('select count(*),sum(uses),sum(uses!=expected_uses) from interactions').fetchone()
        if unique!=self.target or draws!=self.schedule['draws'] or mismatches or frequency!={int(k):v for k,v in self.schedule['identity_use_frequency_histogram'].items()}:
            raise ValueError('Full physical occupancy differs from target')
        if db.execute('pragma integrity_check').fetchone()[0]!='ok':
            raise ValueError('Physical ledger integrity failure')
        result=dict(status='complete-native-runs-and-durable-archives-validated',hard_events=100000,
                    interaction_slots=draws,distinct_physical_interactions=unique,repeat_slots=draws-unique,
                    repeat_slot_fraction=(draws-unique)/draws,identity_use_frequency_histogram=frequency,
                    physical_id_and_payload_uniqueness='SQL UNIQUE constraints over entire bank',
                    expected_uses_mismatches=mismatches,ledger_sha256=sha(directory/'interaction-ledger.sqlite'),
                    limitation='Statistical replacement, not original particles or CMS detector premixing; modelling uncertainties excluded')
        durable_json(directory/'complete.json',result)
        self.status('arm-complete',arm=arm,**result)
        db.close()
        if self.current is not None:
            bank,stream,metadata,_=self.current
            bank.close();stream.close();self.current=None
            path=Path(metadata['bank'])
            if self.out in path.parents and path.exists() and sha(path)==metadata['sha256']:
                durable_json(path.parent/'payload-cleanup.json',dict(reason='All required particle records are in the complete twice-verified archive series',sha256=metadata['sha256']))
                path.unlink()

    def run(self):
        self.status('starting',selected_target='Actual100k FullSim identity-sharing graph',schedule_sha256=sha(self.a.schedule/'index.json'),cpu_workers_max=4,storage_cap_bytes=7_500_000_000)
        for arm in self.a.arms:
            self.run_arm(arm)
        self.pool.shutdown(wait=True)
        self.status('simulation-complete-analysis-pending',arms=self.a.arms)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','output','schedule','selection','validation']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--destination',required=True)
    p.add_argument('--arms',nargs='+',choices=['generic','cms'],default=['generic','cms'])
    a=p.parse_args()
    campaign=Campaign(a)
    with (campaign.out/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            campaign.run()
        except BaseException as error:
            campaign.status('failed-output-preserved',error=str(error),traceback=traceback.format_exc())
            raise


if __name__=='__main__':
    main()
