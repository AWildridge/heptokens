"""Independently audit binary records, then round-trip every native record.

No reconstruction or particle selection. Preserve original and unverified data.
The redundant round-trip is removed only after full SHA256 equality is recorded.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--delphes', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Preserve existing evidence')
    size, offset, particles = a.input.stat().st_size, 0, 0
    unique, ids, multiplicities = set(), Counter(), []
    record = np.dtype([('pid', '>i4'), ('values', '>f4', (8,))])
    with a.input.open('rb') as stream:
        stream.seek(-8, 2)
        events = struct.unpack('>q', stream.read(8))[0]
        if not 0 < events < 10000000:
            raise ValueError('Invalid native event count')
        index_start = size - 8 * (events + 1)
        stream.seek(index_start)
        offsets = struct.unpack('>' + str(events) + 'q', stream.read(8 * events))
        for entry in offsets:
            if entry != offset:
                raise ValueError('Noncontiguous or invalid index')
            stream.seek(entry)
            count = struct.unpack('>i', stream.read(4))[0]
            if not 0 <= count < 1000000:
                raise ValueError('Invalid native particle count')
            body = stream.read(count * 36)
            if len(body) != count * 36:
                raise ValueError('Truncated records')
            rows = np.frombuffer(body, dtype=record)
            if not np.isfinite(rows['values']).all() or (rows['values'][:, 7] < 0).any():
                raise ValueError('Nonfinite particles or negative energy')
            unique.add(hashlib.sha256(body).digest())
            ids.update(Counter(map(int, rows['pid'])))
            multiplicities.append(count)
            particles += count
            offset += 4 + 36 * count
    if offset != index_start:
        raise ValueError('Unexpected bytes before index')
    # Compile an adapter around the unmodified native reader AND writer.
    import ROOT
    ROOT.PyConfig.IgnoreCommandLineOptions = True
    ROOT.gInterpreter.AddIncludePath(str(a.delphes.resolve()))
    if ROOT.gSystem.Load(str((a.delphes / 'libDelphes.so').resolve())) < 0:
        raise ValueError('Cannot load native Delphes')
    ROOT.gInterpreter.Declare(r'''
    #include "classes/DelphesPileUpReader.h"
    #include "classes/DelphesPileUpWriter.h"
    long long audit_roundtrip(const char *src, const char *dst) {
      DelphesPileUpReader reader(src);
      DelphesPileUpWriter writer(dst);
      long long total = 0;
      for (long long i=0; i<reader.GetEntries(); ++i) {
        if (!reader.ReadEntry(i)) throw std::runtime_error("ReadEntry failed");
        int32_t pid; float x,y,z,t,px,py,pz,e;
        while(reader.ReadParticle(pid,x,y,z,t,px,py,pz,e)) {
          writer.WriteParticle(pid,x,y,z,t,px,py,pz,e); ++total;
        }
        writer.WriteEntry();
      }
      writer.WriteIndex(); return total;
    }
    ''')
    roundtrip = Path(str(a.input) + '.audit-roundtrip')
    if roundtrip.exists():
        raise ValueError('Preserve earlier roundtrip for inspection')
    total = ROOT.audit_roundtrip(str(a.input.resolve()), str(roundtrip.resolve()))
    original_sha, native_sha = sha(a.input), sha(roundtrip)
    if total != particles or original_sha != native_sha:
        raise ValueError('Native roundtrip differs from independent audit')
    result = {
        'status': 'verified-all-records-and-native-roundtrip',
        'path': str(a.input.resolve()), 'bytes': size, 'events': events,
        'particles': particles, 'sha256': original_sha,
        'unique_particle_record_payloads': len(unique),
        'duplicate_particle_record_payloads': events - len(unique),
        'particle_count_min_max': [min(multiplicities), max(multiplicities)],
        'pdg_id_particle_counts': dict(sorted(ids.items())),
        'native_roundtrip_sha256': native_sha,
        'native_roundtrip_particles': total,
        'validation': 'Every big-endian record/index checked independently; all float fields finite; native reader/writer roundtrip SHA256 identical',
        'units': 'GeV, mm, ct in mm; no selection or weighting in this audit'}
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    roundtrip.unlink()  # redundant; both full hashes already recorded, original kept
    print(json.dumps({k: result[k] for k in ('status', 'events', 'particles', 'bytes', 'sha256')}))


if __name__ == '__main__':
    main()
