import h5py
import numpy as np
import pytest
from torch.utils.data import DataLoader
from heptokens.data.delphes_replay import DelphesReplayDataset

def test_matched_event_splits_membership_weights_and_batching(tmp_path):
    folder=tmp_path/'training/00000/chunk-000000';folder.mkdir(parents=True)
    for detector in ['CMS','ATLAS']:
        with h5py.File(folder/(detector+'.h5'),'w') as f:
            e=f.create_group('events')
            e['split']=[0,1,2];e['weight']=[1.5,-2.,3.]
            for key in ['run','lumi','event']:e[key]=[1,2,3]
            g=f.create_group('Jet');g['event_offsets']=[0,2,3,4]
            g['jet_p4']=np.arange(16,dtype=np.float32).reshape(4,4)
            g['constituent_offsets']=[0,3,4,5,6]
            g['constituent_p4']=np.array([[1,0,0,1],[3,0,0,3],[2,0,0,2],[4,0,0,4],[5,0,0,5],[6,0,0,6]],dtype=np.float32)
    with pytest.raises(FileNotFoundError,match='READY'):
        DelphesReplayDataset(tmp_path,'CMS')
    train=DelphesReplayDataset(tmp_path,'CMS',collection='Jet',num_elements=2,require_ready=False)
    assert len(train)==2
    assert train[0]['csts'][:,0].tolist()==[3,2]
    assert train[1]['mask'].tolist()==[True,False]
    assert train[0]['weight'].item()==1.5
    batch=next(iter(DataLoader(train,batch_size=2,num_workers=2)))
    assert batch['csts'].shape==(2,2,4)
    for detector in ['CMS','ATLAS']:
        val=DelphesReplayDataset(tmp_path,detector,split='val',collection='Jet',require_ready=False)
        assert len(val)==1 and val[0]['event_id'].tolist()==[2,2,2]
        assert val[0]['weight'].item()==-2.
        assert val[0]['csts'][0,0].item()==5
