"""Materialize a scheduled shard from one verified physical PU bank segment.

Indices in the schedule identify distinct underlying particle events; repeated
indices copy the SAME source record. Copy payload bytes without changing any
particle/vertex convention. Segment start is explicit; requests outside it fail.
"""
import argparse
import hashlib
import json
import mmap
from pathlib import Path
import struct

import numpy as np


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--schedule', type=Path, required=True)
    p.add_argument('--physical-library', type=Path, required=True)
    p.add_argument('--source-sha256', required=True)
    p.add_argument('--segment-start', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    partial = Path(str(a.output) + '.partial')
    if a.output.exists() or partial.exists():
        raise ValueError('Preserve existing output/partial')
    if digest(a.physical_library) != a.source_sha256:
        raise ValueError('Physical bank changed from verified identity')
    with np.load(a.schedule) as source:
        indices = source['indices'].astype(np.int64)
        hard_offsets = source['offsets']
        if hard_offsets[0] != 0 or hard_offsets[-1] != len(indices) or (np.diff(hard_offsets) < 0).any():
            raise ValueError('Invalid hard-event offsets')
    offsets, particles, position = [], 0, 0
    a.output.parent.mkdir(parents=True, exist_ok=True)
    expected = hashlib.sha256()
    with a.physical_library.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as bank:
        events = struct.unpack_from('>q', bank, len(bank) - 8)[0]
        index_start = len(bank) - 8 * (events + 1)
        relative = indices - a.segment_start
        if ((relative < 0) | (relative >= events)).any():
            raise ValueError('Schedule requests identities outside this bank segment')
        with partial.open('wb') as output:
            for entry in relative:
                start = struct.unpack_from('>q', bank, index_start + int(entry) * 8)[0]
                count = struct.unpack_from('>i', bank, start)[0]
                stop = start + 4 + 36 * count
                if start < 0 or count < 0 or stop > index_start:
                    raise ValueError('Invalid source event boundary')
                record = bank[start:stop]
                output.write(record)
                expected.update(record)
                offsets.append(position)
                position += len(record)
                particles += count
            for value in [*offsets, len(offsets)]:
                record = struct.pack('>q', value)
                output.write(record)
                expected.update(record)
    actual = digest(partial)
    if actual != expected.hexdigest():
        raise ValueError('Scheduled bank bytes differ')
    partial.rename(a.output)
    result = {'status': 'materialized-byte-identical-source-records-native-audit-required',
              'events': len(indices), 'particles': particles, 'hard_events': len(hard_offsets) - 1,
              'distinct_physical_interaction_indices': len(set(indices.tolist())),
              'source_segment_start': a.segment_start, 'source': str(a.physical_library.resolve()),
              'source_sha256': a.source_sha256, 'schedule': str(a.schedule.resolve()),
              'schedule_sha256': digest(a.schedule), 'sha256': actual, 'bytes': a.output.stat().st_size,
              'mapping': 'Schedule unique index maps bijectively to one physical source event; repeated indices copy identical payload bytes. This is a statistical replacement, not original-particle matching.'}
    Path(str(a.output) + '.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
