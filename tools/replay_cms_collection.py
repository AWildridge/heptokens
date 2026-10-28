"""One frozen original CMS MiniAOD source per Slurm task; resumable chunks.

Compressed truth and compact training copies are durable; full detector ROOT
files are temporary. A shared CFS ledger caps this campaign at 7 TiB.
"""
import argparse, fcntl, gzip, hashlib, json, os, shutil, subprocess, time, uuid, zlib
from pathlib import Path
import requests
from compact_delphes import convert

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()

def atomic_json(path,data):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def reserve(root,key,size):
    root.mkdir(parents=True,exist_ok=True)
    with open(root/'storage.lock','a+') as lock:
        fcntl.lockf(lock,fcntl.LOCK_EX)
        ledger=root/'storage-ledger.json';data=json.loads(ledger.read_text()) if ledger.exists() else {'cap_bytes':7*1024**4,'reservations':{}}
        old=data['reservations'].get(key)
        if old is not None and old!=size: raise ValueError('Reservation changed; inspect prior chunk')
        if old is None:
            if sum(data['reservations'].values())+size>data['cap_bytes']:raise RuntimeError('CFS campaign 7 TiB capacity guard reached; retain scratch inputs and expand storage before resuming')
            data['reservations'][key]=size;atomic_json(ledger,data)

def checked_copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        if sha(src)!=sha(dst):raise ValueError('Existing durable copy differs: '+str(dst))
        return
    tmp=Path(str(dst)+'.partial');shutil.copyfile(src,tmp)
    if sha(src)!=sha(tmp):raise ValueError('Copy checksum mismatch')
    tmp.replace(dst)

def download(entry,path):
    def validate():
        if path.stat().st_size!=entry['bytes']:return False
        checksum=1
        with open(path,'rb') as f:
            for block in iter(lambda:f.read(1048576),b''):checksum=zlib.adler32(block,checksum)
        return '%08x'%(checksum&0xffffffff)==entry['checksum'].split(':',1)[1].lower()
    if path.exists():
        if not validate():raise ValueError('Existing source checksum mismatch')
        return
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
                    return
                except Exception as xrd_error:print('XRootD transfer failed:',repr(xrd_error),flush=True)
            if attempt==5:raise
            time.sleep(min(30,2**attempt))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--task',type=int,required=True)
    p.add_argument('--validation-limit',type=int)
    a=p.parse_args();campaign=a.campaign.resolve();config=json.loads((campaign/'layout.json').read_text())
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    if not a.validation_limit:
        ready=json.loads((Path(config['prepared_root'])/'cms_qcd/READY.json').read_text())
        if ready.get('status')!='complete' or ready.get('selection')!='cms_qcd' or ready['inventory_sha256']!=sha(campaign/'prepared-inventory.json'):raise ValueError('CMS QCD prepared inventory not verified')
    manifest=json.loads((campaign/'cms-qcd-source-manifest.json').read_text());entry=manifest['files'][a.task]
    durable=Path(config['durable_replay_root']);scratch=Path(config['scratch_replay_root'])
    if a.validation_limit:
        pilot='pilot-'+str(os.environ['SLURM_JOB_ID'])+'-limit'+str(a.validation_limit)
        durable=durable.parent/pilot;scratch=scratch.parent/pilot
    work=scratch/'work'/('%05d'%a.task);work.mkdir(parents=True,exist_ok=True)
    taskroot=durable/'sources'/('%05d'%a.task);taskroot.mkdir(parents=True,exist_ok=True)
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
        source=work/'input.root';download(entry,source)
        exported=work/'export';indexfile=exported/'index.json'
        if not indexfile.exists():
            if exported.exists():exported.rename(work/('failed-export-'+uuid.uuid4().hex))
            command=['shifter','--image='+config['cmssw_image'],'--module=cvmfs','bash',config['cmssw_wrapper'],'python',str(campaign/'tools/export_cms_chunks.py'),'--input',str(source),'--output',str(exported),'--chunk-events','1000']
            if a.validation_limit:command+=['--limit',str(a.validation_limit)]
            exportlog=work/('export-'+os.environ['SLURM_JOB_ID']+'-'+uuid.uuid4().hex+'.log')
            try:
                with open(exportlog,'w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            finally:checked_copy(exportlog,taskroot/exportlog.name)
        index=json.loads(indexfile.read_text())
        checked_copy(indexfile,taskroot/'export-index.json')
        if not a.validation_limit and (index['validation_limit'] is not None or index['source_events']!=index['exported_events']):raise ValueError('Not full source replay')
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
                cmd=[config['python'],str(campaign/'tools/simulate_delphes_pair.py'),'--input',str(truth),'--delphes',config['delphes'],'--output',str(output),'--seed',str(seed)]
                if not output.exists():subprocess.run(cmd,check=True)
                reference=json.loads(refpath.read_text());reference['source_key']=entry['uri'];counts={}
                for detector in ['CMS','ATLAS']:
                    target=output/(detector+'.h5');temporary=output/(detector+'-recovery-'+uuid.uuid4().hex+'.h5')
                    counts[detector]=convert(output/(detector+'.root'),temporary,Path(config['delphes']),reference)
                    if target.exists():target.rename(output/(detector+'-previous-'+uuid.uuid4().hex+'.h5'))
                    temporary.replace(target)
                with open(truth,'rb') as inp,gzip.open(output/'truth.hepmc.gz','wb',compresslevel=6) as out:shutil.copyfileobj(inp,out)
                metadata={k:v for k,v in reference.items() if k!='events'}
                metadata['events']=[{k:v for k,v in event.items() if k!='particles'} for event in reference['events']]
                metadata['particle_reference']='Exact particle P4, PDG and production vertices retained in truth.hepmc.gz; reconstruct particles in HepMC order for the validation tool'
                with gzip.open(output/'truth-reference.json.gz','wt',compresslevel=6) as out:json.dump(metadata,out)
                files=['CMS.h5','ATLAS.h5','truth.hepmc.gz','truth-reference.json.gz','validation.json','CMS.log','ATLAS.log','delphes_card_CMS.tcl','delphes_card_ATLAS.tcl','CMS-seed%d.tcl'%seed,'ATLAS-seed%d.tcl'%seed]
                reserve(durable,'%05d/%s'%(a.task,name),sum((output/f).stat().st_size for f in files))
                for filename in files:checked_copy(output/filename,dest/filename)
                for detector in ['CMS','ATLAS']:checked_copy(output/(detector+'.h5'),train/(detector+'.h5'))
                receipt={'status':'complete','events':chunk['events'],'input_sha256':chunk['sha256'],'seed':seed,'source':entry,'counts':counts,'sha256':{f:sha(dest/f) for f in files},'bytes':{f:(dest/f).stat().st_size for f in files},'slurm_job_id':os.environ['SLURM_JOB_ID'],'precision':'float32 jet/constituent P4; exact memberships; no cuts; compressed original export retained'}
                atomic_json(receiptpath,receipt)
                # Only new disposable intermediates, after validated durable commit.
                shutil.rmtree(output);truth.unlink();refpath.unlink()
            completed.append({'chunk':name,'events':chunk['events']})
            print('Committed',a.task,name,chunk['events'],'events',flush=True)
        atomic_json(complete,{'status':'complete','source':entry,'source_events':index['source_events'],'exported_events':index['exported_events'],'validation_limit':a.validation_limit,'chunks':completed,'manifest_sha256':sha(campaign/'cms-qcd-source-manifest.json')})
        source.unlink()

if __name__=='__main__':main()
