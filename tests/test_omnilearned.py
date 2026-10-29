from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from heptokens.data.omnilearned import OmniLearnedDataset, OmniLearnedModule


def shard(root, dataset, split, filename, labels):
    directory = root / dataset / split
    directory.mkdir(parents=True, exist_ok=True)
    data = np.zeros((len(labels), 5, 9), dtype=np.float32)
    data[:, :2, :4] = [0.1, -0.2, 2.0, 2.5]
    data[:, :2, 4] = 3
    with h5py.File(directory / filename, 'w') as f:
        f['data'] = data
        f['global'] = np.ones((len(labels), 3), dtype=np.float32)
        f['pid'] = labels


def test_shards_fixed_labels_and_limits(tmp_path):
    shard(tmp_path, 'jetclass', 'train', 'b.h5', [11, 2])
    shard(tmp_path, 'jetclass', 'train', 'a.h5', [8, 8])
    ds = OmniLearnedDataset(tmp_path, 'jetclass', num_samples=3, num_elements=3)
    assert len(ds) == 3
    assert [ds[i]['labels'].item() for i in range(3)] == [6, 6, 9]
    assert ds[0]['csts'].shape == (3, 4)
    assert ds[0]['mask'].tolist() == [True, True, False]
    assert ds[0]['pid_codes'].tolist() == [3, 3, 0]
    assert ds[-1]['labels'].item() == 9
    with pytest.raises(IndexError):
        ds[3]
    batch = next(iter(DataLoader(ds, batch_size=2, num_workers=2)))
    assert batch['csts'].shape == (2, 3, 4)
    assert batch['mask'].dtype == torch.bool
    ds.close()


def test_official_splits_and_no_aspen_truth(tmp_path):
    for split, labels in [('train', [12, 13]), ('val', [199, 199]), ('test', [17, 17])]:
        shard(tmp_path, 'jetclass2', split, 'shard.h5', labels)
    module = OmniLearnedModule(root=tmp_path, dataset='jetclass2', batch_size=2, num_workers=0)
    module.setup()
    assert module.get_n_classes() == 188
    assert next(iter(module.val_dataloader()))['labels'].tolist() == [187, 187]
    assert next(iter(module.test_dataloader()))['labels'].tolist() == [5, 5]
    assert module.get_data_sample().shape == (5, 4)
    shard(tmp_path, 'aspen', 'train', 'shard.h5', [200, 200])
    assert 'labels' not in OmniLearnedDataset(tmp_path, 'aspen')[0]
    shard(tmp_path, 'cms_qcd', 'train', 'shard.h5', [201, 201])
    assert 'labels' not in OmniLearnedDataset(tmp_path, 'cms_qcd')[0]


def test_invalid_labels_fail(tmp_path):
    shard(tmp_path, 'jetclass', 'train', 'shard.h5', [12])
    with pytest.raises(ValueError, match='Label outside'):
        OmniLearnedDataset(tmp_path, 'jetclass')[0]


def test_nonfinite_fails(tmp_path):
    shard(tmp_path, 'jetclass', 'train', 'shard.h5', [2])
    with h5py.File(tmp_path / 'jetclass/train/shard.h5', 'r+') as f:
        f['data'][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        OmniLearnedDataset(tmp_path, 'jetclass')[0]
