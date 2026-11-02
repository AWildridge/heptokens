"""Independent serialized-link coverage and reference-bound checks using uproot."""
from pathlib import Path
import json


def audit(sidecar, compact):
    import awkward as ak
    import h5py
    import numpy as np
    import uproot
    from uproot.source.file import MemmapSource
    with uproot.open(sidecar, handler=MemmapSource) as root:
        if root['links'].classname != 'TTree' or root['events'].classname != 'TTree':
            raise ValueError('Links/events must be ordinary TTrees')
        links = root['links'].arrays(library='ak')
        events = root['events'].arrays(library='np')
        metadata = json.loads(str(root['metadata']))
    with h5py.File(compact, 'r') as h5:
        for key in ['run', 'lumi', 'event', 'export_event', 'weight']:
            a, b = events[key], h5['events'][key][:]
            if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                raise ValueError('Serialized sidecar event identity/weight differs: ' + key)
        covered = 0
        for collection in ['Jet', 'FatJet']:
            rows = links[links['collection'] == collection]
            count = len(h5[collection]['constituent_p4'])
            stored = ak.to_numpy(rows['compact_constituent_row'])
            np.testing.assert_array_equal(stored, np.arange(count))
            covered += len(rows)
            if len(rows):
                e, j, c = [ak.to_numpy(rows[name]) for name in ['event_index', 'jet_index', 'constituent_index']]
                if np.any(e < 0) or np.any(e >= len(events['event'])) or np.any(j < 0) or np.any(c < 0):
                    raise ValueError('Negative or out-of-range object index')
                event_offsets = h5[collection]['event_offsets'][:]
                if np.any(j >= event_offsets[e+1] - event_offsets[e]):
                    raise ValueError('Jet index outside its event')
                jet = event_offsets[e] + j
                offsets = h5[collection]['constituent_offsets'][:]
                if np.any(offsets[jet] + c != stored) or np.any(stored >= offsets[jet+1]):
                    raise ValueError('Serialized constituent membership differs')
        if covered != len(links):
            raise ValueError('Unknown collection in serialized sidecar')
    index, barcode = links['truth_indices'], links['truth_barcodes']
    counts = ak.num(index)
    if not bool(ak.all((counts > 0) & (counts == ak.num(barcode)))):
        raise ValueError('Empty or misaligned contributor lists')
    bounds = events['truth_count'][ak.to_numpy(links['event_index'])]
    if not bool(ak.all((index >= 0) & (index < bounds))):
        raise ValueError('Truth reference outside corresponding event')
    if not bool(ak.all(barcode > 0)):
        raise ValueError('Invalid stable HepMC barcode')
    ordered = ak.sort(index, axis=-1)
    if not bool(ak.all(ordered[:, 1:] != ordered[:, :-1])):
        raise ValueError('Duplicate contributor reference')
    return dict(status='passed', method='Independent uproot serialization/event/membership/reference audit',
                events=len(events['event']), constituent_rows=len(links),
                contributor_edges=int(ak.sum(counts)), detector=metadata['detector'])
