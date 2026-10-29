"""Original-event end-to-end test per pT bin, plus idempotent restart checks."""
import argparse,json,os,subprocess,sys,time,uuid
from pathlib import Path
import awkward as ak
import h5py
import requests

tools=Path(__file__).resolve().parent;sys.path.insert(0,str(tools))
from replay_cms_collection import download,sha,checked_copy,atomic_json
from replay_runtime import transfer_slot
p=argparse.ArgumentParser();p.add_argument('--bin',type=int,required=True);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--layout',type=Path,required=True);p.add_argument('--run-id',required=True);p.add_argument('--limit',type=int,default=2001);p.add_argument('--report-dir',type=Path,required=True);p.add_argument('--interrupt-after-first-commit',action='store_true');a=p.parse_args()
campaign=a.campaign.resolve()
job=a.run_id;sources=json.loads((campaign/'test-bin-sources.json').read_text());selected=sources[a.bin]
config=json.loads(a.layout.read_text());manifest=json.loads((campaign/'cms-qcd-source-manifest.json').read_text());entry=manifest['files'][selected['task']]
name='pilot-'+job+'-limit'+str(a.limit);scratch=Path(config['scratch_replay_root']).parent/name;durable=Path(config['durable_replay_root']).parent/name
work=scratch/'work'/('%05d'%entry['task']);work.mkdir(parents=True,exist_ok=True)
# One live fault-injection test: bypass HTTPS to force the actual XRootD
# fallback. The bytes still come from the original published CERN source.
fallback_proof=None
if a.bin==1:
    proofpath=work/'live-xrootd-proof.json'
    if proofpath.exists():
        fallback_proof=json.loads(proofpath.read_text())
        if fallback_proof['source']!=entry or sha(Path(fallback_proof['validated_file']))!=fallback_proof['sha256']:
            raise ValueError('Prior live XRootD evidence changed')
if a.bin==1 and fallback_proof is None:
    original=requests.get
    def fail_gateway(*args,**kwargs):raise requests.HTTPError('503 injected to test live XRootD fallback')
    requests.get=fail_gateway
    fallback_path=work/('live-xrootd-'+uuid.uuid4().hex+'.root')
    try:
        with transfer_slot(config['transfer_slots'],config.get('max_downloads',2)):
            download(entry,fallback_path)
    finally:requests.get=original
    checked_copy(fallback_path,work/'input.root')
    fallback_proof={'source':entry,'validated_file':str(fallback_path),'sha256':sha(fallback_path),
                    'verified_utc_epoch':time.time(),'transport':'published XRootD URI after injected HTTPS 503'}
    atomic_json(proofpath,fallback_proof)
command=[config['python'],str(tools/'replay_cms_collection.py'),'--campaign',str(campaign),'--task',str(entry['task']),'--validation-limit',str(a.limit),'--layout',str(a.layout),'--run-id',a.run_id]
interrupted=False
if a.interrupt_after_first_commit and not (durable/'sources'/('%05d'%entry['task'])/'COMPLETE.json').exists():
    injection="""import sys
sys.path.insert(0, sys.argv.pop(1))
import replay_cms_collection as replay
original = replay.atomic_json
def interrupt(path, data):
    original(path, data)
    if path.name == 'receipt.json': raise SystemExit(75)
replay.atomic_json = interrupt
replay.main()
"""
    attempt=subprocess.run([config['python'],'-c',injection,str(tools),*command[2:]])
    if attempt.returncode!=75:raise ValueError('Expected interruption immediately after durable receipt commit')
    interrupted=True
subprocess.run(command,check=True)
root=durable/'sources'/('%05d'%entry['task']);complete=json.loads((root/'COMPLETE.json').read_text())
if complete['exported_events']!=min(a.limit,complete['source_events']):raise ValueError('Prefix event count mismatch')
before={str(path.relative_to(root)):sha(path) for path in root.glob('chunk-*/*.h5')}
for chunk in complete['chunks']:
    directory=root/chunk['chunk']
    with h5py.File(directory/'CMS.h5','r') as cms,h5py.File(directory/'ATLAS.h5','r') as atlas:
        for key in ['run','lumi','event','export_event','weight','split']:
            if ak.to_list(ak.Array(cms['events/'+key][:]))!=ak.to_list(ak.Array(atlas['events/'+key][:])):raise ValueError('Detector event metadata mismatch')
subprocess.run(command,check=True)
after={str(path.relative_to(root)):sha(path) for path in root.glob('chunk-*/*.h5')}
if before!=after:raise ValueError('Restart changed committed detector outputs')
report={'status':'passed','record':entry['record'],'task':entry['task'],'events':complete['exported_events'],'chunk_events':[c['events'] for c in complete['chunks']],'execution':json.loads(os.environ['HEPTOKENS_EXECUTION']),'checks':['original source size and Adler32','retained generator collections','both stock-card truth/weight checks','compact memberships/P4 round trip','matched detector event IDs, weights and splits','idempotent completed-task restart'],'forced_live_xrootd_fallback':a.bin==1,'durable_output':str(root),'scratch_output':str(scratch)}
report['test_harness_sha256']=sha(Path(__file__))
report['interrupted_after_commit_and_recovered']=interrupted
report['storage_status']='POSIX staging only; remote durability has not been established by this test'
report['live_xrootd_proof']=fallback_proof
destination=a.report_dir/('test-bin-%d-%s.json'%(a.bin,job));destination.write_text(json.dumps(report,indent=2));print(destination,flush=True)
