"""Build an exact public shard inventory before bulk transfer (Slurm only)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
import os
from pathlib import Path
from urllib.parse import urljoin
import requests

BASE='https://portal.nersc.gov/cfs/dasrepo/omnilearned/'
DATASETS=['aspen','jetclass','jetclass2','cms_qcd']
class Links(HTMLParser):
 def __init__(self): super().__init__(); self.links=[]
 def handle_starttag(self,tag,attrs):
  if tag=='a':
   href=dict(attrs).get('href','')
   if href.endswith('.h5') and '/' not in href and Path(href).name==href:
    self.links.append(href)

def head(entry):
 r=requests.head(entry['url'],timeout=90); r.raise_for_status()
 if 'Content-Length' not in r.headers: raise ValueError('Missing source size: '+entry['url'])
 return entry|{'bytes':int(r.headers['Content-Length']),'etag':r.headers.get('ETag'),'last_modified':r.headers.get('Last-Modified')}

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--root',type=Path,required=True)
args=p.parse_args()
if not os.environ.get('SLURM_JOB_ID'): p.error('Run in Slurm')
if args.output.exists(): raise FileExistsError(args.output)
entries=[]; tasks=[]
for dataset in DATASETS:
 for split in ['train','val','test']:
  url=urljoin(BASE,f'{dataset}/{split}/')
  r=requests.get(url,timeout=90);r.raise_for_status();parser=Links();parser.feed(r.text)
  names=sorted(set(parser.links))
  if not names: raise ValueError('No public shards: '+url)
  task=len(tasks)
  tasks.append({'task':task,'dataset':dataset,'split':split,'files':len(names)})
  entries += [{'task':task,'dataset':dataset,'split':split,'filename':name,'url':urljoin(url,name)} for name in names]
with ThreadPoolExecutor(max_workers=12) as pool: entries=list(pool.map(head,entries))
summary={name:{'files':sum(e['dataset']==name for e in entries),'bytes':sum(e['bytes'] for e in entries if e['dataset']==name)} for name in DATASETS}
record={'created_utc':datetime.now(timezone.utc).isoformat(),'root':str(args.root),'slurm_job_id':os.environ['SLURM_JOB_ID'],'datasets':summary,'total_bytes':sum(e['bytes'] for e in entries),'tasks':tasks,'files':entries,'scope':'all listed prepared HDF5 shards in all three splits; no partial row samples'}
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(record,indent=2))
print(json.dumps({k:v for k,v in record.items() if k!='files'},indent=2))
