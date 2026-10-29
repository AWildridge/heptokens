"""Cache one Rucio input and record two full, stable checksum readbacks.

Never uploads, overwrites a conflicting local file, or deletes remote originals.
Run in the existing isolated GFAL environment.
"""
import argparse,json,os,time
from pathlib import Path
from commit_replay_gfal import digest,remote_adler,verify_staged_file
from gfal_transport import run
from replay_scratch_budget import durable_json

def verify_input(request,destination,evidence,stability_seconds=30,max_file_bytes=2*1024**3):
 if stability_seconds<1:raise ValueError('Positive stability interval required')
 destination=Path(destination)
 if destination.is_symlink() or destination.with_name(destination.name+'.download-partial').is_symlink():raise ValueError('Symlink cache paths are not accepted')
 request=dict(request);destination=destination.resolve();evidence=Path(evidence).resolve()
 if not 0<request['bytes']<=max_file_bytes:raise ValueError('Input exceeds the explicit single-file byte budget')
 destination.parent.mkdir(parents=True,exist_ok=True);evidence.mkdir(parents=True,exist_ok=True)
 expected=request['adler32'].lower().zfill(8);url=request['url']
 def matches(path):return path.is_file() and path.stat().st_size==request['bytes'] and digest(path,'adler32')==expected
 if remote_adler(url)!=expected:raise ValueError('Purdue checksum differs from Rucio catalog')
 if destination.exists():
  if not matches(destination):raise ValueError('Existing cached file differs; retained without overwrite')
 else:
  partial=destination.with_name(destination.name+'.download-partial')
  if partial.exists():
   if not matches(partial):raise ValueError('Incomplete or conflicting partial retained; inspect before retry')
  else:run('gfal-copy','--checksum','ADLER32',url,partial.as_uri())
  if not matches(partial):raise ValueError('First full readback size/Adler32 differs; partial retained')
  with partial.open('rb') as stream:os.fsync(stream.fileno())
  os.link(partial,destination);partial.unlink()
  fd=os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)
 first={'url':url,'bytes':request['bytes'],'adler32':expected,'sha256':digest(destination,'sha256'),'rule_id':request.get('rule_id'),'first_readback_epoch':time.time()}
 durable_json(evidence/'expected-input.json',first);durable_json(evidence/'first-readback.json',dict(first,status='catalog-checksum-and-full-readback-passed'))
 time.sleep(stability_seconds)
 # Reuse the production verifier; its independent temporary copy is removed
 # only after the full readback has been hashed and compared successfully.
 second=verify_staged_file(destination,first,stability_seconds)
 if not matches(destination) or digest(destination,'sha256')!=first['sha256']:raise ValueError('Cached input changed during verification')
 durable_json(evidence/'second-readback.json',dict(second,status='second-full-sha256-readback-passed'))
 result=dict(first,status='verified-input-ready',local_path=str(destination),second_readback_epoch=time.time(),stable_interval_seconds=stability_seconds,full_readbacks=2,remote_original_deleted=False)
 durable_json(evidence/'verified-input.json',result);(evidence/'SHA256SUMS').write_text(first['sha256']+'  '+str(destination)+'\n');return result

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--stability-seconds',type=int,default=30);p.add_argument('--max-file-bytes',type=int,default=2*1024**3);a=p.parse_args();print(json.dumps(verify_input(json.loads(a.request.read_text()),a.destination,a.evidence,a.stability_seconds,a.max_file_bytes)))
