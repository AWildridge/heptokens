"""Publish READY only when every frozen shard is complete and locally usable."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import h5py

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inventory',type=Path,required=True)
p.add_argument('--dataset',choices=['aspen','jetclass','jetclass2','cms_qcd'],help='Verify one dataset independently; omit to verify all four')
args=p.parse_args()
inventory=json.loads(args.inventory.read_text());root=Path(inventory['root']);counts={}
entries=[e for e in inventory['files'] if args.dataset is None or e['dataset']==args.dataset]
if not entries:raise ValueError('Empty verification selection')
for entry in entries:
 path=root/entry['dataset']/entry['split']/entry['filename'];receipt=Path(str(path)+'.receipt.json')
 if not path.is_file() or not receipt.is_file():raise ValueError('Missing completed shard: '+str(path))
 saved=json.loads(receipt.read_text());identity={k:entry.get(k) for k in ['url','bytes','etag','last_modified']}
 if saved['status']!='complete' or saved['source']!=identity or path.stat().st_size!=entry['bytes']:raise ValueError('Receipt/size mismatch: '+str(path))
 with h5py.File(path,'r') as f:
  if bool(f.attrs.get('sample_only',False)) or len(f['data'])!=saved['rows']:raise ValueError('Sample/row mismatch')
  if len(f['pid'])!=saved['rows'] or len(f['global'])!=saved['rows']:raise ValueError('Invalid aligned schema')
 key=entry['dataset']+'/'+entry['split'];s=counts.setdefault(key,{'files':0,'rows':0,'bytes':0});s['files']+=1;s['rows']+=saved['rows'];s['bytes']+=entry['bytes']
ready={'status':'complete','verified_utc':datetime.now(timezone.utc).isoformat(),'inventory_sha256':hashlib.sha256(args.inventory.read_bytes()).hexdigest(),'root':str(root),'selection':args.dataset or 'all','datasets':counts,'total_bytes':sum(e['bytes'] for e in entries),'verification':'completed transfer size, pinned ETag, recorded full-file SHA256, HDF5 schema and rows; no sampled shards'}
destination=root/args.dataset if args.dataset else root
(destination/'READY.json.tmp').write_text(json.dumps(ready,indent=2));(destination/'READY.json.tmp').replace(destination/'READY.json')
print(json.dumps(ready,indent=2))
