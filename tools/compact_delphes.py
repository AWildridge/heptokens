"""Lossless constituent membership and float32 P4 training shards from Delphes.

All Jet and FatJet objects, without cuts, sorting, truncation or truth labels.
Ragged offsets retain event->jet->constituent associations. ROOT intermediates
are disposable only after both detectors pass validation and durable copies.
"""
import argparse, hashlib, json, os
from pathlib import Path
import h5py
import numpy as np
import ROOT

def convert(source, target, delphes, reference):
    ROOT.gInterpreter.AddIncludePath(str(delphes))
    ROOT.gInterpreter.AddIncludePath(str(delphes/'external'))
    ROOT.gSystem.Load(str(delphes/'libDelphes.so'))
    f = ROOT.TFile.Open(str(source)); tree = f.Get('Delphes')
    if not tree or tree.GetEntries() != len(reference['events']): raise ValueError('Event count')
    collections = {name: {'event_offsets':[0], 'jet_p4':[], 'constituent_offsets':[0], 'constituent_p4':[], 'constituent_type':[], 'charge':[]} for name in ['Jet','FatJet']}
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        for name, data in collections.items():
            if not tree.GetBranch(name):
                data['event_offsets'].append(len(data['jet_p4']))
                continue
            for jet in getattr(tree, name):
                p4 = jet.P4(); data['jet_p4'].append([p4.Px(),p4.Py(),p4.Pz(),p4.E()])
                if jet.Constituents.GetEntriesFast() == 0: raise ValueError('Empty jet')
                for j in range(jet.Constituents.GetEntriesFast()):
                    obj = jet.Constituents.At(j)
                    if not obj: raise ValueError('Unresolved constituent reference')
                    p4 = obj.P4(); data['constituent_p4'].append([p4.Px(),p4.Py(),p4.Pz(),p4.E()])
                    kind = obj.ClassName()
                    kinds={'Track':1,'Tower':2,'GenParticle':3,'Muon':4,'Electron':5,'Photon':6}
                    if kind not in kinds: raise ValueError('Unknown constituent class '+kind)
                    data['constituent_type'].append(kinds[kind])
                    data['charge'].append(int(obj.Charge) if hasattr(obj,'Charge') else 0)
                data['constituent_offsets'].append(len(data['constituent_p4']))
            data['event_offsets'].append(len(data['jet_p4']))
    f.Close()
    with h5py.File(target, 'w') as out:
        out.attrs.update(schema='delphes-ragged-p4-v1', units='GeV', detector=source.stem,
                         selection='all default-card Jet and FatJet objects; no cuts or truncation',
                         precision='float32 P4; integer offsets exact',
                         constituent_type='1=Track,2=Tower,3=GenParticle,4=Muon,5=Electron,6=Photon; not PDG labels',
                         truth_labels='unavailable from stable-only replay')
        ev = out.create_group('events')
        for key in ['run','lumi','event','export_event']:
            ev.create_dataset(key, data=np.array([r[key] for r in reference['events']], dtype=np.uint64), compression='gzip',shuffle=True)
        ev.create_dataset('weight',data=np.array([r['weight'] for r in reference['events']],dtype=np.float64),compression='gzip')
        # Source + event identity binds both detector views to the same split.
        split=[]
        for r in reference['events']:
            identity=reference.get('source_key',reference.get('input',''))+':'+':'.join(str(r[k]) for k in ['run','lumi','event'])
            bucket=int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8],'big')%100
            split.append(0 if bucket<80 else 1 if bucket<90 else 2)
        ev.create_dataset('split',data=np.array(split,dtype=np.uint8))
        out.attrs['split_rule']='SHA256(source URI:run:lumi:event) uint64 big-endian mod100; 0=train<80,1=val<90,2=test'
        for name, data in collections.items():
            group = out.create_group(name)
            for key, values in data.items():
                dtype = np.float32 if key.endswith('p4') else (np.int8 if key in ['charge','constituent_type'] else np.int64)
                array = np.asarray(values,dtype=dtype)
                if key.endswith('p4'):
                    array = array.reshape(-1,4)
                    if not np.isfinite(array).all(): raise ValueError('Nonfinite P4')
                group.create_dataset(key,data=array,compression='gzip',shuffle=True)
    # Verify the saved representation against the ROOT-derived values, including
    # every integer membership offset and all P4 components to float32 precision.
    with h5py.File(target,'r') as out:
        for name, data in collections.items():
            for key, values in data.items():
                actual = out[name][key][:]
                expected = np.asarray(values,dtype=actual.dtype).reshape(actual.shape)
                np.testing.assert_array_equal(actual,expected)
    return {name:{'jets':len(data['jet_p4']),'constituents':len(data['constituent_p4'])} for name,data in collections.items()}

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source','output','delphes','reference']: p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): p.error('Use Slurm')
    if a.output.exists(): p.error('Output exists')
    print(json.dumps(convert(a.source,a.output,a.delphes,json.loads(a.reference.read_text()))))
