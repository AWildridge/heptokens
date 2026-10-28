"""Publish campaign readiness only after every source event has both views."""
import argparse,hashlib,json,os
from pathlib import Path
from datetime import datetime,timezone

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);a=p.parse_args()
if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
layout=json.loads((a.campaign/'layout.json').read_text());manifestpath=a.campaign/'cms-qcd-source-manifest.json'
manifest=json.loads(manifestpath.read_text());digest=hashlib.sha256(manifestpath.read_bytes()).hexdigest()
root=Path(layout['durable_replay_root']);scratch=Path(layout['scratch_replay_root']);totals={str(r['record']):0 for r in manifest['records']};chunks=0
for source in manifest['files']:
 directory=root/'sources'/('%05d'%source['task']);result=json.loads((directory/'COMPLETE.json').read_text())
 if result['manifest_sha256']!=digest or result['validation_limit'] is not None or result['source_events']!=result['exported_events'] or result['source']!=source:raise ValueError('Invalid full-source completion')
 count=0
 for item in result['chunks']:
  chunk=directory/item['chunk'];receipt=json.loads((chunk/'receipt.json').read_text())
  if receipt['status']!='complete' or receipt['events']!=item['events']:raise ValueError('Incomplete chunk')
  for filename,size in receipt['bytes'].items():
   if (chunk/filename).stat().st_size!=size:raise ValueError('Durable file missing/truncated')
   if filename.endswith('.h5') and (scratch/'training'/('%05d'%source['task'])/item['chunk']/filename).stat().st_size!=size:raise ValueError('Scratch training file missing/truncated')
  validation=json.loads((chunk/'validation.json').read_text())
  if validation['status']!='passed' or any(validation['detectors'][d]['status']!='passed' for d in ['CMS','ATLAS']):raise ValueError('Failed detector validation')
  count+=receipt['events'];chunks+=1
 if count!=result['source_events']:raise ValueError('Source event accounting mismatch')
 totals[str(source['record'])]+=count
for record in manifest['records']:
 if totals[str(record['record'])]!=record['events']:raise ValueError('Catalogue event accounting mismatch')
ready={'status':'complete','verified_utc':datetime.now(timezone.utc).isoformat(),'source_files':len(manifest['files']),'events_by_record':totals,'total_events':sum(totals.values()),'chunks':chunks,'manifest_sha256':digest,'scope':manifest['scope'],'detectors':['default CMS','default ATLAS'],'scratch_training_root':str(scratch),'durable_root':str(root),'normalization':'raw weights; no cross-section or luminosity scaling','verification':'source/chunk/event accounting; both detector truth/weight checks; committed file sizes and checksums recorded at creation'}
for path in [root/'READY.json',scratch/'READY.json']:
 path.with_suffix('.json.tmp').write_text(json.dumps(ready,indent=2));path.with_suffix('.json.tmp').replace(path)
print(json.dumps(ready,indent=2))
