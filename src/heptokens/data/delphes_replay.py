"""Lazy jet training access to the complete ragged paired detector replay.

Features here are Cartesian GeV P4, unlike OmniLearned's prepared features.
The adapter sorts by constituent pT and pads/truncates for batching; stored
shards retain every constituent and can be reused with a different length.
"""
from bisect import bisect_right
from pathlib import Path
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from heptokens.data.base import BaseMapModule

class DelphesReplayDataset(Dataset):
    def __init__(self,root,detector,split='train',collection='Jet',num_elements=128,num_samples=None,require_ready=True):
        if detector not in ['CMS','ATLAS'] or split not in ['train','val','test'] or collection not in ['Jet','FatJet']:raise ValueError('Invalid replay selection')
        if num_elements<=0 or (num_samples is not None and num_samples<=0):raise ValueError('Positive limits required')
        if require_ready and not (Path(root)/'READY.json').is_file():raise FileNotFoundError('Full replay not verified; READY.json is required')
        self.files=sorted(Path(root).glob('training/*/chunk-*/'+detector+'.h5'))
        if not self.files:raise FileNotFoundError('No completed replay training shards')
        self.split=['train','val','test'].index(split);self.collection=collection;self.num_elements=num_elements
        self.ends=[];total=0;self._handles={};self._pid=os.getpid()
        for path in self.files:
            with h5py.File(path,'r') as f:
                offsets=f[collection+'/event_offsets'][:];mask=f['events/split'][:]==self.split
                total+=int(np.diff(offsets)[mask].sum());self.ends.append(total)
        self.length=min(total,num_samples) if num_samples else total

    def __len__(self):return self.length

    def __getitem__(self,index):
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        if self._pid!=os.getpid():self.close();self._pid=os.getpid()
        shard=bisect_right(self.ends,index);local=index-(self.ends[shard-1] if shard else 0)
        if shard not in self._handles:self._handles[shard]=h5py.File(self.files[shard],'r')
        f=self._handles[shard];g=f[self.collection];offsets=g['event_offsets'][:]
        events=np.flatnonzero(f['events/split'][:]==self.split);ends=np.cumsum(np.diff(offsets)[events])
        position=int(np.searchsorted(ends,local,side='right'));event=int(events[position])
        jet=int(offsets[event]+local-(ends[position-1] if position else 0))
        start,stop=g['constituent_offsets'][jet:jet+2];raw=g['constituent_p4'][start:stop]
        order=np.argsort(-(raw[:,0]**2+raw[:,1]**2),kind='stable')[:self.num_elements]
        data=np.zeros((self.num_elements,4),dtype=np.float32);data[:len(order)]=raw[order]
        mask=np.arange(self.num_elements)<len(order)
        return {'csts':torch.from_numpy(data),'mask':torch.from_numpy(mask),'jets':torch.from_numpy(g['jet_p4'][jet]),
                'weight':torch.tensor(f['events/weight'][event],dtype=torch.float64),
                'event_id':torch.tensor([int(f['events/'+k][event]) for k in ['run','lumi','event']],dtype=torch.int64)}

    def close(self):
        for f in self._handles.values():f.close()
        self._handles={}
    def __getstate__(self):
        state=self.__dict__.copy();state['_handles']={};return state
    def __del__(self):
        if hasattr(self,'_handles'):self.close()

class DelphesReplayModule(BaseMapModule):
    def __init__(self,*,root,detector,collection='Jet',num_elements=128,num_samples=None,require_ready=True,**kwargs):
        super().__init__(n_classes=None,**kwargs)
        self.options=dict(root=root,detector=detector,collection=collection,num_elements=num_elements,num_samples=num_samples,require_ready=require_ready)
    def setup(self,stage=None):
        for name,split in [('train_set','train'),('valid_set','val'),('test_set','test')]:
            if not hasattr(self,name):setattr(self,name,DelphesReplayDataset(split=split,**self.options))
    def get_data_sample(self):
        if not hasattr(self,'train_set'):self.setup()
        return self.train_set[0]['csts'].clone()
