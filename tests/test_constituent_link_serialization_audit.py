"""A serialized link table must preserve event identity, membership and bounds."""
import json
import sys
from pathlib import Path

import awkward as ak
import h5py
import numpy as np
import pytest
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from audit_delphes_constituent_links import audit


def fixture(tmp_path, change=None):
    compact, sidecar = tmp_path / 'compact.h5', tmp_path / 'links.root'
    event = {name: np.array([1], dtype='uint64') for name in ['run', 'lumi', 'event', 'export_event']}
    event['weight'] = np.array([-1.0], dtype='float64')
    with h5py.File(compact, 'w') as h5:
        for key, values in event.items():
            h5['events/' + key] = values
        for collection, count in [('Jet', 1), ('FatJet', 0)]:
            h5[collection + '/constituent_p4'] = np.zeros((count, 4), dtype='float32')
            h5[collection + '/constituent_offsets'] = np.array([0, 1] if count else [0], dtype='int64')
            h5[collection + '/event_offsets'] = np.array([0, count], dtype='int64')
    event['truth_count'] = np.array([2], dtype='int64')
    fields = dict(event_index='int64', collection='string', jet_index='int64', constituent_index='int64',
                  compact_constituent_row='int64', truth_indices='var * int64', truth_barcodes='var * int64')
    rows = dict(event_index=np.array([0]), collection=ak.Array(['Jet']), jet_index=np.array([0]),
                constituent_index=np.array([0]), compact_constituent_row=np.array([0]),
                truth_indices=ak.Array([[0, 1]]), truth_barcodes=ak.Array([[1, 2]]))
    if change == 'identity':
        event['event'] = np.array([2], dtype='uint64')
    elif change == 'membership':
        rows['constituent_index'] = np.array([1])
    elif change == 'bounds':
        rows['truth_indices'] = ak.Array([[0, 2]])
    elif change == 'duplicates':
        rows['truth_indices'] = ak.Array([[0, 0]])
    elif change == 'alignment':
        rows['truth_barcodes'] = ak.Array([[1]])
    with uproot.recreate(sidecar) as root:
        root['metadata'] = json.dumps(dict(detector='CMS'))
        root.mktree('events', {k: v.dtype for k, v in event.items()}).extend(event)
        root.mktree('links', fields).extend(rows)
    return sidecar, compact


def test_independent_serialized_audit(tmp_path):
    result = audit(*fixture(tmp_path))
    assert result['events'] == 1 and result['contributor_edges'] == 2


@pytest.mark.parametrize('change', ['identity', 'membership', 'bounds', 'duplicates', 'alignment'])
def test_corrupt_serialized_links_rejected(tmp_path, change):
    with pytest.raises((ValueError, AssertionError)):
        audit(*fixture(tmp_path, change))
