"""Convert extracted CMS GEN particles with the native Delphes pileup writer.

Remove each interaction's modal stable-particle vertex before Delphes samples
a new beam spot. This approximates the primary interaction origin while
retaining relative decay displacements; unavailable production ct is zero.
Independently verify every binary byte against big-endian float32 records.
"""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import struct

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--delphes', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Preserve existing library; use a fresh output')
    ROOT.gInterpreter.AddIncludePath(str(a.delphes))
    ROOT.gInterpreter.AddIncludePath(str(a.delphes / 'external'))
    if ROOT.gSystem.Load(str(a.delphes / 'libDelphes.so')) < 0:
        raise RuntimeError('Cannot load validated Delphes runtime')
    ROOT.gInterpreter.Declare('#include "classes/DelphesPileUpWriter.h"')
    partial = Path(str(a.output) + '.partial')
    writer = ROOT.DelphesPileUpWriter(str(partial))
    expected = hashlib.sha256()
    offsets, ids = [], []
    offset = particles = 0
    origin_min = [float('inf')] * 3
    origin_max = [-float('inf')] * 3
    with gzip.open(a.input, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            values = row['particles']
            origin = collections.Counter(tuple(v[6:9]) for v in values).most_common(1)[0][0]
            offsets.append(offset)
            ids.append(row['key'])
            expected.update(struct.pack('>i', len(values)))
            for v in values:
                xyz = [v[6+i] - origin[i] for i in range(3)]
                args = [int(v[0]), *xyz, v[9], *v[1:5]]
                writer.WriteParticle(*args)
                expected.update(struct.pack('>i8f', *args))
            writer.WriteEntry()
            particles += len(values)
            offset += 4 + 36 * len(values)
            origin_min = [min(x, y) for x, y in zip(origin_min, origin)]
            origin_max = [max(x, y) for x, y in zip(origin_max, origin)]
    writer.WriteIndex()
    writer.__destruct__()
    for value in offsets:
        expected.update(struct.pack('>q', value))
    expected.update(struct.pack('>q', len(offsets)))
    actual = hashlib.file_digest(partial.open('rb'), 'sha256').hexdigest()
    if actual != expected.hexdigest():
        raise ValueError('Native writer bytes differ from independent particle/index encoding')
    partial.rename(a.output)
    Path(str(a.output) + '.json').write_text(json.dumps({
        'status': 'verified-native-library', 'events': len(offsets), 'particles': particles,
        'bytes': a.output.stat().st_size, 'sha256': actual, 'event_ids': ids,
        'source_sha256': hashlib.file_digest(a.input.open('rb'), 'sha256').hexdigest(),
        'origin_convention': 'Subtract modal stable-particle xyz per interaction; preserve relative displaced decays',
        'original_origin_range_mm': [origin_min, origin_max],
        'units': 'GeV/mm, unavailable ct=0; native float32 library',
        'verification': 'All binary particle records and index independently encoded and SHA256 compared',
    }, indent=2) + '\n')
    print('Verified native library:', len(offsets), 'interactions,', particles, 'particles', flush=True)


if __name__ == '__main__':
    main()
