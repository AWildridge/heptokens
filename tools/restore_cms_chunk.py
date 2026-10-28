"""Restore a durable chunk for the existing paired detector validation runner."""
import argparse,gzip,hashlib,json,os,shutil
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--chunk',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
if a.output.exists() or Path(str(a.output)+'.json').exists():p.error('Use fresh output')
a.output.parent.mkdir(parents=True,exist_ok=True)
with gzip.open(a.chunk/'truth.hepmc.gz','rb') as inp,open(a.output,'wb') as out:shutil.copyfileobj(inp,out)
with gzip.open(a.chunk/'truth-reference.json.gz','rt') as inp:reference=json.load(inp)
digest=hashlib.sha256()
with open(a.output,'rb') as inp:
 for block in iter(lambda:inp.read(1048576),b''):digest.update(block)
if digest.hexdigest()!=reference['sha256']:raise ValueError('Truth checksum mismatch')
index=-1;position=None
with open(a.output) as inp:
 for line in inp:
  tokens=line.split()
  if not tokens:continue
  if tokens[0]=='E':
   index+=1
   if int(tokens[1])!=reference['events'][index]['export_event']:raise ValueError('Event ordering')
   reference['events'][index]['particles']=[]
  elif tokens[0]=='V':position=list(map(float,tokens[3:7]))
  elif tokens[0]=='P':
   if position is None:raise ValueError('Missing vertex')
   reference['events'][index]['particles'].append([int(tokens[2])]+list(map(float,tokens[3:8]))+position)
if index+1!=len(reference['events']):raise ValueError('Event count')
Path(str(a.output)+'.json').write_text(json.dumps(reference))
print(a.output)
