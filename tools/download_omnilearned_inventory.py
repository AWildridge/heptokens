"""Resume complete HDF5 transfers from a frozen inventory; one Slurm array task.

Completion receipts are atomic and bind size, SHA256, source ETag and schema.
Partial files are never exposed as .h5 inputs. No sampling or preprocessing.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def atomic_json(path,record):
 temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(record,indent=2));temporary.replace(path)


def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as stream:
  for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()


def fetch(entry,root):
 directory=root/entry['dataset']/entry['split'];directory.mkdir(parents=True,exist_ok=True)
 output=directory/entry['filename'];receipt=directory/(entry['filename']+'.receipt.json')
 identity={key:entry.get(key) for key in ['url','bytes','etag','last_modified']}
 if output.exists():
  if not receipt.exists():raise ValueError('Unverified existing file: '+str(output))
  saved=json.loads(receipt.read_text())
  if saved['source']!=identity or output.stat().st_size!=entry['bytes'] or digest(output)!=saved['sha256']:
   raise ValueError('Existing input does not match frozen source: '+str(output))
  return saved
 partial=directory/(entry['filename']+'.partial');metadata=directory/(entry['filename']+'.partial.json')
 if partial.exists() and (not metadata.exists() or json.loads(metadata.read_text())!=identity):
  raise ValueError('Partial source identity changed; preserve and use a fresh root: '+str(partial))
 atomic_json(metadata,identity)
 session=requests.Session()
 session.mount('https://',HTTPAdapter(max_retries=Retry(total=5,backoff_factor=1,status_forcelist=[429,500,502,503,504],allowed_methods=['GET','HEAD'])))
 try:
  for attempt in range(6):
   offset=partial.stat().st_size if partial.exists() else 0
   if offset>entry['bytes']:raise ValueError('Oversized partial file')
   if offset==entry['bytes']:break
   headers={}
   if entry.get('etag'):headers['If-Match']=entry['etag']
   if offset:headers['Range']='bytes=%d-'%offset
   try:
    with session.get(entry['url'],headers=headers,stream=True,timeout=(30,180)) as response:
     response.raise_for_status()
     if entry.get('etag') and response.headers.get('ETag')!=entry['etag']:raise ValueError('Source ETag changed')
     if offset:
      if response.status_code!=206 or not response.headers.get('Content-Range','').startswith('bytes %d-'%offset):raise ValueError('Server did not honor resume range')
     elif response.status_code!=200:raise ValueError('Unexpected full response')
     with open(partial,'ab' if offset else 'wb') as stream:
      for block in response.iter_content(4*1024*1024):stream.write(block)
    if partial.stat().st_size!=entry['bytes']:raise IOError('Transfer size mismatch')
    break
   except (requests.RequestException,IOError):
    if attempt==5:raise
    time.sleep(min(30,2**attempt))
  if not partial.exists() or partial.stat().st_size!=entry['bytes']:raise ValueError('Incomplete transfer')
  with h5py.File(partial,'r') as f:
   if bool(f.attrs.get('sample_only',False)):raise ValueError('Row sample in full-data root')
   n=len(f['data'])
   if f['data'].ndim!=3 or f['data'].shape[-1]<4 or len(f['global'])!=n or len(f['pid'])!=n:raise ValueError('Invalid schema')
   for key in ['data','global','pid']:
    if not np.isfinite(f[key][:min(64,n)]).all():raise ValueError('Nonfinite validation rows')
   schema={key:{'shape':list(f[key].shape),'dtype':str(f[key].dtype)} for key in f}
  record={'source':identity,'sha256':digest(partial),'rows':n,'schema':schema,'completed_utc':datetime.now(timezone.utc).isoformat(),'slurm_job_id':os.environ['SLURM_JOB_ID'],'status':'complete'}
  # Receipt first permits safe recovery if interrupted between these two renames.
  atomic_json(receipt,record);partial.replace(output);metadata.unlink(missing_ok=True)
  print('Complete',entry['dataset'],entry['split'],entry['filename'],entry['bytes'],flush=True)
  return record
 finally:session.close()


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--task',type=int,required=True);p.add_argument('--workers',type=int,default=3)
 args=p.parse_args()
 if not os.environ.get('SLURM_JOB_ID'):p.error('Run in Slurm')
 inventory=json.loads(args.inventory.read_text());root=Path(inventory['root'])
 entries=[e for e in inventory['files'] if e['task']==args.task]
 if not entries:raise ValueError('No files for task')
 journal=root/'status'/('task-%02d-%s-%s.json'%(args.task,os.environ['SLURM_JOB_ID'],datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')));journal.parent.mkdir(parents=True,exist_ok=True)
 state={'task':args.task,'dataset':entries[0]['dataset'],'split':entries[0]['split'],'expected_files':len(entries),'completed_files':0,'expected_bytes':sum(e['bytes'] for e in entries),'status':'running','slurm_job_id':os.environ['SLURM_JOB_ID'],'inventory_sha256':digest(args.inventory)}
 atomic_json(journal,state)
 try:
  with ThreadPoolExecutor(max_workers=args.workers) as pool:
   # Bounded request concurrency. Each file has its own session and receipt.
   from concurrent.futures import as_completed
   futures=[pool.submit(fetch,entry,root) for entry in entries]
   for future in as_completed(futures):
    future.result();state['completed_files']+=1;atomic_json(journal,state)
  state['status']='complete'
 except Exception as error:state.update(status='failed',error=repr(error));raise
 finally:atomic_json(journal,state)

if __name__=='__main__':main()
