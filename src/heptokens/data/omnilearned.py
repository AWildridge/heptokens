"""Lazy access to OmniLearned's prepared HDF5 shards, preserving official splits.

The first four data columns are used verbatim, in the upstream representation.
Particle PID codes are categorical upstream codes, NOT PDG identifiers.
"""
from bisect import bisect_right
from pathlib import Path
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from heptokens.data.base import BaseMapModule

DATASETS = {"aspen": (200, None), "jetclass": (2, 10), "jetclass2": (12, 188), "cms_qcd": (201, None)}


class OmniLearnedDataset(Dataset):
    """Local shards under root/dataset/{train,val,test}; no automatic downloads.

    Limits apply to the entire split, in sorted shard order. Validation and test
    labels use the same fixed offsets as training, even if some classes are absent.
    Masks reproduce upstream's column-2 nonzero contract. All other stored
    columns remain accessible as pid_codes/add_info, without claiming PDG IDs.
    """
    def __init__(self, root, dataset, split="train", num_samples=None, num_elements=None):
        if dataset not in DATASETS or split not in {"train", "val", "test"}:
            raise ValueError("Unsupported dataset or split")
        if num_samples is not None and num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if num_elements is not None and num_elements <= 0:
            raise ValueError("num_elements must be positive")
        self.dataset, self.split = dataset, split
        self.num_elements = num_elements
        directory = Path(root) / dataset / split
        self.files = sorted(set(directory.glob("*.h5")) | set(directory.glob("*.hdf5")))
        if not self.files:
            raise FileNotFoundError(f"No shards in {directory}; stage files first")
        self.ends, self._handles, self._pid = [], {}, os.getpid()
        total, shape = 0, None
        for path in self.files:
            with h5py.File(path, "r") as f:
                data = f["data"]
                if data.ndim != 3 or data.shape[-1] < 4 or len(f["global"]) != len(data) or len(f["pid"]) != len(data):
                    raise ValueError(f"Invalid OmniLearned schema: {path}")
                if shape is not None and data.shape[1:] != shape:
                    raise ValueError("Shard particle/feature shapes must match within a split")
                shape = data.shape[1:]
                total += len(data)
                self.ends.append(total)
        self.length = min(total, num_samples) if num_samples is not None else total
        self.num_elements = min(shape[0], num_elements) if num_elements else shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        shard = bisect_right(self.ends, index)
        row = index - (self.ends[shard - 1] if shard else 0)
        if shard not in self._handles:
            self._handles[shard] = h5py.File(self.files[shard], "r")
        f = self._handles[shard]
        data = np.asarray(f["data"][row, :self.num_elements], dtype=np.float32)
        jets = np.asarray(f["global"][row], dtype=np.float32)
        if not np.isfinite(data).all() or not np.isfinite(jets).all():
            raise ValueError(f"Nonfinite input in {self.files[shard]} row {row}")
        sample = {"csts": torch.from_numpy(data[:, :4].copy()),
                  "mask": torch.from_numpy(data[:, 2] != 0),
                  "jets": torch.from_numpy(jets.copy())}
        if data.shape[1] > 4:
            sample["pid_codes"] = torch.from_numpy(data[:, 4].astype(np.int64))
            sample["add_info"] = torch.from_numpy(data[:, 5:].copy())
        offset, nclasses = DATASETS[self.dataset]
        if nclasses is not None:
            raw = np.asarray(f["pid"][row])
            if raw.size != 1 or not np.isfinite(raw).all() or raw.item() != int(raw.item()):
                raise ValueError("Expected scalar integer upstream label")
            label = int(raw.item()) - offset
            if not 0 <= label < nclasses:
                raise ValueError(f"Label outside {self.dataset} range: {raw}")
            sample["labels"] = torch.tensor(label, dtype=torch.long)
        # Aspen's upstream bucket 200 is a source identifier, not a truth label.
        return sample

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __del__(self):
        if hasattr(self, "_handles"):
            self.close()


class OmniLearnedModule(BaseMapModule):
    """Use supplied splits directly; no reshuffling between train/val/test."""
    def __init__(self, *, root, dataset, num_samples=None, num_elements=None, **kwargs):
        super().__init__(n_classes=DATASETS[dataset][1], **kwargs)
        self.root, self.dataset = root, dataset
        self.num_samples, self.num_elements = num_samples, num_elements

    def setup(self, stage=None):
        for attribute, split in (("train_set", "train"), ("valid_set", "val"), ("test_set", "test")):
            if not hasattr(self, attribute):
                setattr(self, attribute, OmniLearnedDataset(self.root, self.dataset, split,
                                                           self.num_samples, self.num_elements))

    def get_data_sample(self):
        if not hasattr(self, "train_set"):
            self.setup()
        return self.train_set[0]["csts"].clone()
