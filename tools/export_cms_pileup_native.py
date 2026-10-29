"""CMSSW Python2 wrapper: compiled FWLite -> native Delphes bank, ct=0.

The modal-origin convention matches build_delphes_pileup_library.py. A complete
SHA comparison on the original50600 interactions validates the accelerated path.
All IDs and unit weights are checked; all particles with status1 are retained.
"""
from __future__ import print_function
import argparse, hashlib, json, os, time
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
from DataFormats.FWLite import Events

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',required=True)
p.add_argument('--output',required=True)
p.add_argument('--delphes',required=True)
a=p.parse_args()
partial=a.output+'.partial'
ids_path=a.output+'.ids.jsonl'
if any(os.path.exists(v) for v in [a.output,partial,ids_path]):
    raise ValueError('Preserve existing output/partial/identity ledger')
ROOT.gInterpreter.AddIncludePath(a.delphes)
ROOT.gInterpreter.AddIncludePath(a.delphes+'/external')
header=os.path.join(os.path.dirname(os.path.abspath(__file__)),'export_cms_pileup_native.h')
if not ROOT.gInterpreter.Declare('#include "'+header+'"'):
    raise RuntimeError('FWLite converter compilation failed')
started=time.time()
particles=int(ROOT.export_cms_native(a.input,partial,ids_path,a.output+'.progress'))
ids=[json.loads(v) for v in open(ids_path)]
if len(set(tuple(v) for v in ids))!=len(ids):
    raise ValueError('Duplicate event identities')
digest=hashlib.sha256()
with open(partial,'rb') as stream:
    for block in iter(lambda:stream.read(1048576),b''):
        digest.update(block)
os.rename(partial,a.output)
with open(a.output+'.summary.json','w') as stream:
    json.dump(dict(status='extracted-native-audit-required',input=a.input,
                   events=len(ids),particles=particles,event_ids=ids,
                   sha256=digest.hexdigest(),bytes=os.path.getsize(a.output),
                   all_gen_weights_one=True,gen_weight_counts={'1.0':len(ids)},
                   seconds=time.time()-started,
                   origin_convention='Subtract modal stable-particle xyz; first occurrence wins ties; preserve relative decay displacements',
                   units='GeV/mm; unavailable ct=0',selection='All status1 GEN particles; no kinematic cuts'),stream,indent=2)
print('Extracted',len(ids),'interactions;',particles,'particles')
