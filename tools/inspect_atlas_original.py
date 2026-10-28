"""Inspect a checksum-verified original ATLAS nominal release shard."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import zlib
import h5py
import requests
import os

metadata=Path('/global/u2/a/anwildri/code/perlmutter-setup/qcd-top-delphes/atlas-top-record.json')
record=json.loads(metadata.read_text())['metadata']
index=next(item for item in record['_file_indices'] if 'train_nominal' in item['description'])
source=min(index['files'],key=lambda item:item['size'])
root=Path(os.environ['RESEARCH_SCRATCH'])/'runs'/('atlas-original-inspection-'+os.environ['SLURM_JOB_ID'])
root.mkdir(parents=True,exist_ok=False)
url='https://opendata.cern.ch'+source['uri'].split('root://eospublic.cern.ch/')[1]
compressed=root/source['filename']
with requests.get(url,stream=True,timeout=120) as response:
 response.raise_for_status()
 with open(compressed,'wb') as output:
  for chunk in response.iter_content(1024*1024): output.write(chunk)
a=1; digest=hashlib.sha256()
with open(compressed,'rb') as stream:
 for chunk in iter(lambda:stream.read(1024*1024),b''):
  a=zlib.adler32(chunk,a); digest.update(chunk)
assert 'adler32:%08x'%(a&0xffffffff)==source['checksum']
assert compressed.stat().st_size==source['size']
raw=root/compressed.stem
with gzip.open(compressed,'rb') as stream,open(raw,'wb') as output: shutil.copyfileobj(stream,output)
with h5py.File(raw,'r') as f:
 fields={}
 def collect(name,node):
  if isinstance(node,h5py.Dataset):
   fields[name]={'shape':list(node.shape),'dtype':str(node.dtype)}
 f.visititems(collect)
summary={'source_record':80030,'source':source,'download_url':url,'download_sha256':digest.hexdigest(),
         'source_code_revision':'5bc3c8d70cf8736d24b969972a3dd365017a9e24',
         'question':'Does the original nominal ATLAS Top Tagging release retain full generator events?',
         'slurm_job_id':os.environ['SLURM_JOB_ID'],'fields':fields,
         'outcome':'schema inspected; see fields alongside official documentation',
         'scope':'original nominal shard, metadata/schema inspection only; no new event generation'}
(root/'inspection.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
