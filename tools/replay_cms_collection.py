"""Replay one frozen original CMS MiniAOD source in resumable chunks.

Compressed truth and compact training copies are durable; full detector ROOT
files are temporary. A shared POSIX ledger enforces the configured storage cap.
"""
import argparse, fcntl, gzip, hashlib, json, os, shutil, subprocess, time, uuid, zlib
from pathlib import Path
import requests
from replay_runtime import execution_identity, transfer_slot, measure_stage, storage_ledger_lock

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()

def atomic_json(path,data):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def reserve(root,key,size,cap_bytes=7*1024**4,ledger_name='storage-ledger.json'):
    root.mkdir(parents=True,exist_ok=True)
    with storage_ledger_lock(root):
        ledger=root/ledger_name;data=json.loads(ledger.read_text()) if ledger.exists() else {'cap_bytes':cap_bytes,'reservations':{}}
        if data['cap_bytes'] != cap_bytes: raise ValueError('Storage cap changed; inspect ledger')
        old=data['reservations'].get(key)
        if old is not None and old!=size: raise ValueError('Reservation changed; inspect prior chunk')
        if old is None:
            if sum(data['reservations'].values())+size>data['cap_bytes']:raise RuntimeError('Storage capacity guard reached; retain validated inputs and inspect the staging budget')
            data['reservations'][key]=size;atomic_json(ledger,data)

def checked_copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        if sha(src)!=sha(dst):raise ValueError('Existing durable copy differs: '+str(dst))
        return
    tmp=Path(str(dst)+'.partial');shutil.copyfile(src,tmp)
    if sha(src)!=sha(tmp):raise ValueError('Copy checksum mismatch')
    tmp.replace(dst)

def download(entry,path,prefer_xrootd=False):
    def validate():
        if path.stat().st_size!=entry['bytes']:return False
        checksum=1
        with open(path,'rb') as f:
            for block in iter(lambda:f.read(1048576),b''):checksum=zlib.adler32(block,checksum)
        return '%08x'%(checksum&0xffffffff)==entry['checksum'].split(':',1)[1].lower()
    if path.exists():
        if not validate():raise ValueError('Existing source checksum mismatch')
        return
    def xrootd():
        xrdpartial=Path(str(path)+'.xrootd.partial')
        env=os.environ.copy()
        env.update(XRD_CONNECTIONWINDOW='60',XRD_REQUESTTIMEOUT='120',XRD_STREAMTIMEOUT='120')
        command=['xrdcp','--force','--nopbar','--retry','2',entry['uri'],str(xrdpartial)]
        try:
            print('Trying published XRootD source',entry['uri'],flush=True)
            subprocess.run(command,check=True,timeout=900,env=env)
            xrdpartial.replace(path)
            if not validate():
                path.rename(Path(str(path)+'.invalid-'+uuid.uuid4().hex))
                raise ValueError('XRootD source size/adler32 mismatch')
            return True
        except Exception as error:
            print('XRootD transfer failed:',repr(error),flush=True)
            return False
    if prefer_xrootd and xrootd():return
    partial=Path(str(path)+'.partial')
    for attempt in range(6):
        try:
            offset=partial.stat().st_size if partial.exists() else 0
            if offset==entry['bytes']: partial.replace(path)
            else:
                with requests.get(entry['url'],headers={'Range':'bytes=%d-'%offset} if offset else {},stream=True,timeout=(30,180)) as response:
                    response.raise_for_status()
                    if offset and (response.status_code!=206 or not response.headers.get('Content-Range','').startswith('bytes %d-'%offset)):
                        raise ValueError('Server did not honor resume offset')
                    with open(partial,'ab' if offset else 'wb') as f:
                        for block in response.iter_content(4*1024*1024):f.write(block)
                partial.replace(path)
            if not validate():path.rename(Path(str(path)+'.invalid-'+uuid.uuid4().hex));raise ValueError('Source size/adler32 mismatch')
            return
        except Exception as error:
            print('HTTP source transfer failed:',repr(error),flush=True)
            # Use the identical published EOS URI when CERN's HTTPS gateway
            # is unavailable. The source identity and Adler32 gate stay fixed.
            if attempt==1:
                if xrootd():return
            if attempt==5:raise
            time.sleep(min(30,2**attempt))

def main():
    from compact_delphes import convert
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--task',type=int,required=True)
    p.add_argument('--validation-limit',type=int)
    p.add_argument('--layout',type=Path)
    p.add_argument('--run-id')
    a=p.parse_args();campaign=a.campaign.resolve();config=json.loads((a.layout or campaign/'layout.json').read_text())
    detector_workers=config.get('detector_workers',2)
    if detector_workers not in [1,2]:raise ValueError('detector_workers must be one or two')
    tools=Path(__file__).resolve().parent
    execution=execution_identity()
    os.environ['HEPTOKENS_EXECUTION']=json.dumps(execution)
    run_id=a.run_id or execution['id']
    if '/' in run_id or ':' in run_id: p.error('Supply a filesystem-safe --run-id')
    if a.validation_limit is not None and a.validation_limit <= 0: p.error('Validation limit must be positive')
    if not a.validation_limit:
        ready_path=Path(config['upstream_prepared_receipt']) if config.get('upstream_prepared_receipt') else Path(config['prepared_root'])/'cms_qcd/READY.json'
        ready=json.loads(ready_path.read_text())
        if ready.get('status')!='complete' or ready.get('selection')!='cms_qcd' or ready['inventory_sha256']!=sha(campaign/'prepared-inventory.json'):raise ValueError('CMS QCD prepared inventory not verified')
    manifest=json.loads((campaign/'cms-qcd-source-manifest.json').read_text());entry=manifest['files'][a.task]
    if any('://' in config[key] for key in ['durable_replay_root','scratch_replay_root']):
        raise ValueError('Replay requires POSIX staging roots; publish EOS copies with commit_replay_gfal.py')
    durable=Path(config['durable_replay_root']);scratch=Path(config['scratch_replay_root'])
    if a.validation_limit:
        pilot='pilot-'+run_id+'-limit'+str(a.validation_limit)
        durable=durable.parent/pilot;scratch=scratch.parent/pilot
    work=scratch/'work'/('%05d'%a.task);work.mkdir(parents=True,exist_ok=True)
    taskroot=durable/'sources'/('%05d'%a.task);taskroot.mkdir(parents=True,exist_ok=True)
    trainroot=scratch/'training'/('%05d'%a.task)
    def staging_bytes():
        if config.get('scratch_accounting'):
            from replay_scratch_budget import physical_bytes
            return physical_bytes(config['scratch_accounting']['scratch'])
        return sum(path.stat().st_size for root in [work,taskroot,trainroot] if root.exists() for path in root.rglob('*') if path.is_file())
    def staging_guard(extra=0):
        if config.get('max_source_staging_bytes') and staging_bytes()+extra>config['max_source_staging_bytes']:
            raise RuntimeError('Source staging budget reached; validated outputs retained')
    # Serialize retries for this source without locking independent tasks.
    with open(taskroot/'task.lock','a+') as lock:
        fcntl.lockf(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        complete=taskroot/'COMPLETE.json'
        if complete.exists():
            result=json.loads(complete.read_text())
            if result['validation_limit']!=a.validation_limit or result['manifest_sha256']!=sha(campaign/'cms-qcd-source-manifest.json'):raise ValueError('Completed source configuration changed')
            for item in result['chunks']:
                directory=taskroot/item['chunk'];receipt=json.loads((directory/'receipt.json').read_text())
                for name,digest in receipt['sha256'].items():
                    if sha(directory/name)!=digest:raise ValueError('Durable completed chunk corrupted')
                    if name.endswith('.h5'):checked_copy(directory/name,scratch/'training'/('%05d'%a.task)/item['chunk']/name)
            print('Already complete',a.task,flush=True);return
        source=work/'input.root'
        with transfer_slot(config.get('transfer_slots',scratch/'transfer-slots'),config.get('max_downloads',2)), measure_stage(work,'source-download-and-validation'):
            download(entry,source,prefer_xrootd=config.get('source_transport')=='xrootd')
        if config.get('scratch_accounting'):
            from replay_scratch_budget import pin_files
            pin_files(config['scratch_accounting'], [source])
        staging_guard()
        exported=work/'export';indexfile=exported/'index.json'
        if not indexfile.exists():
            if exported.exists():exported.rename(work/('failed-export-'+uuid.uuid4().hex))
            if config.get('cmssw_command'):
                command=config['cmssw_command']+['python',str(tools/'export_cms_chunks.py'),'--input',str(source),'--output',str(exported),'--chunk-events','1000']
            else:
                command=['shifter','--image='+config['cmssw_image'],'--module=cvmfs','bash',config['cmssw_wrapper'],'python',str(tools/'export_cms_chunks.py'),'--input',str(source),'--output',str(exported),'--chunk-events','1000']
            if a.validation_limit:command+=['--limit',str(a.validation_limit)]
            if config.get('max_source_staging_bytes'):
                available=config['max_source_staging_bytes']-staging_bytes()-config.get('source_overhead_bytes',1024**3)
                export_budget=min(config['max_export_bytes'],available)
                if export_budget<=0:raise RuntimeError('No safe export storage budget remains; inspect interrupted exports')
                command+=['--max-output-bytes',str(export_budget)]
            exportlog=work/('export-'+run_id+'-'+uuid.uuid4().hex+'.log')
            try:
                with measure_stage(work,'cmssw-export'),open(exportlog,'w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            finally:checked_copy(exportlog,taskroot/exportlog.name)
        index=json.loads(indexfile.read_text())
        checked_copy(indexfile,taskroot/'export-index.json')
        if not a.validation_limit and (index['validation_limit'] is not None or index['source_events']!=index['exported_events']):raise ValueError('Not full source replay')
        if config.get('scratch_accounting'):
            from replay_scratch_budget import pin_files
            immutable_exports=[];expected_exports={}
            for chunk in index['chunks']:
                truth_path=exported/chunk['file']
                if truth_path.exists():
                    immutable_exports.append(truth_path);expected_exports[str(truth_path)]=chunk['sha256']
                reference_path=Path(str(truth_path)+'.json')
                if reference_path.exists():immutable_exports.append(reference_path)
            pin_files(config['scratch_accounting'], immutable_exports, expected_hashes=expected_exports)
        completed=[]
        for number,chunk in enumerate(index['chunks']):
            name='chunk-%06d'%number;dest=taskroot/name;receiptpath=dest/'receipt.json'
            train=scratch/'training'/('%05d'%a.task)/name
            if receiptpath.exists():
                receipt=json.loads(receiptpath.read_text())
                if receipt['input_sha256']!=chunk['sha256']:raise ValueError('Chunk input changed')
                for filename,digest in receipt['sha256'].items():
                    if sha(dest/filename)!=digest:raise ValueError('Durable checksum mismatch')
                    if filename.endswith('.h5'):checked_copy(dest/filename,train/filename)
            else:
                truth=exported/chunk['file'];refpath=Path(str(truth)+'.json')
                if sha(truth)!=chunk['sha256']:raise ValueError('Export changed')
                seed=420000+a.task*1000+number
                output=None
                for candidate in sorted(work.glob(name+'-attempt-*')):
                    record=candidate/'validation.json'
                    if not record.is_file():continue
                    validation=json.loads(record.read_text())
                    if validation.get('status')!='passed' or validation.get('input_sha256')!=chunk['sha256'] or validation.get('seed')!=seed:continue
                    if all((candidate/(d+'.root')).is_file() and validation['detectors'][d]['card_sha256']==sha(Path(config['delphes'])/'cards'/('delphes_card_'+d+'.tcl')) and sha(candidate/(d+'.root'))==validation['detectors'][d]['output_sha256'] for d in ['CMS','ATLAS']):
                        output=candidate;print('Reuse validated detector ROOT pair',str(output),flush=True);break
                if output is None:output=work/(name+'-attempt-'+uuid.uuid4().hex)
                cmd=[config['python'],str(tools/'simulate_delphes_pair.py'),'--input',str(truth),'--delphes',config['delphes'],'--output',str(output),'--seed',str(seed),'--detector-workers',str(detector_workers)]
                if not output.exists():
                    with measure_stage(work,'paired-simulation-validation/'+name):subprocess.run(cmd,check=True)
                staging_guard()
                reference=json.loads(refpath.read_text());reference['source_key']=entry['uri'];counts={}
                for detector in ['CMS','ATLAS']:
                    target=output/(detector+'.h5');temporary=output/(detector+'-recovery-'+uuid.uuid4().hex+'.h5')
                    with measure_stage(work,'compact-'+detector+'/'+name):
                        counts[detector]=convert(output/(detector+'.root'),temporary,Path(config['delphes']),reference)
                    if target.exists():target.rename(output/(detector+'-previous-'+uuid.uuid4().hex+'.h5'))
                    temporary.replace(target)
                with measure_stage(work,'compress-truth/'+name),open(truth,'rb') as inp,gzip.open(output/'truth.hepmc.gz','wb',compresslevel=6) as out:shutil.copyfileobj(inp,out)
                metadata={k:v for k,v in reference.items() if k!='events'}
                metadata['events']=[{k:v for k,v in event.items() if k!='particles'} for event in reference['events']]
                metadata['particle_reference']='Exact particle P4, PDG and production vertices retained in truth.hepmc.gz; reconstruct particles in HepMC order for the validation tool'
                with measure_stage(work,'compress-metadata/'+name),gzip.open(output/'truth-reference.json.gz','wt',compresslevel=6) as out:json.dump(metadata,out)
                files=['CMS.h5','ATLAS.h5','truth.hepmc.gz','truth-reference.json.gz','validation.json','CMS.log','ATLAS.log','delphes_card_CMS.tcl','delphes_card_ATLAS.tcl','CMS-seed%d.tcl'%seed,'ATLAS-seed%d.tcl'%seed]
                staging_guard(sum((output/f).stat().st_size for f in files)+sum((output/(d+'.h5')).stat().st_size for d in ['CMS','ATLAS']))
                reserve(durable,'%05d/%s'%(a.task,name),sum((output/f).stat().st_size for f in files),config.get('storage_cap_bytes',7*1024**4))
                with measure_stage(work,'local-checked-commit/'+name):
                    for filename in files:checked_copy(output/filename,dest/filename)
                    for detector in ['CMS','ATLAS']:checked_copy(output/(detector+'.h5'),train/(detector+'.h5'))
                receipt={'status':'complete','events':chunk['events'],'input_sha256':chunk['sha256'],'seed':seed,'source':entry,'counts':counts,'sha256':{f:sha(dest/f) for f in files},'bytes':{f:(dest/f).stat().st_size for f in files},'execution':execution,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'precision':'float32 jet/constituent P4; exact memberships; no cuts; compressed original export retained'}
                atomic_json(receiptpath,receipt)
                if config.get('scratch_accounting'):
                    from replay_scratch_budget import pin_files
                    pin_files(config['scratch_accounting'], [dest/f for f in files if f.endswith(('.h5','.gz'))])
                # Only new disposable intermediates, after validated durable commit.
                shutil.rmtree(output);truth.unlink();refpath.unlink()
            completed.append({'chunk':name,'events':chunk['events']})
            print('Committed',a.task,name,chunk['events'],'events',flush=True)
        atomic_json(complete,{'status':'complete','source':entry,'source_events':index['source_events'],'exported_events':index['exported_events'],'validation_limit':a.validation_limit,'chunks':completed,'manifest_sha256':sha(campaign/'cms-qcd-source-manifest.json')})
        source.unlink()

if __name__=='__main__':main()
