"""Assign fresh PU identities using the original100k CMS in-time reuse pattern.

This builds an identity/count ledger, NOT particle data or a detector simulation.
New generic/CMS particles must be mapped one-to-one to its unique indices.
Chunks are interleaved across hard sources to avoid source-file/bin alignment.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Preserve existing schedule; choose a fresh directory')
    sources, inputs = {}, []
    for path in sorted(a.metadata.glob('metadata-*.jsonl.gz')):
        task = int(path.name.split('-')[1].split('.')[0])
        with gzip.open(path, 'rt') as stream:
            sources[task] = [json.loads(line) for line in stream]
        inputs.append({'path': str(path.resolve()), 'sha256': digest(path)})
    if set(sources) != {84, 1373, 2785, 4151} or any(len(v) != 25000 for v in sources.values()):
        raise ValueError('Require frozen four-source100k sample')
    a.output.mkdir(parents=True)
    identity_to_index, identities, uses, chunks = {}, [], [], []
    within_repeats = total = 0
    for chunk in range(25):
        for task in sorted(sources):
            rows = sources[task][chunk * 1000:(chunk + 1) * 1000]
            counts, offsets, indices, hard_ids = [], [0], [], []
            first_new = len(identities)
            for row in rows:
                ids = [tuple(v) for v in row['ids']]
                if len(ids) != row['n']:
                    raise ValueError('N does not match retained in-time IDs')
                within_repeats += int(len(set(ids)) != len(ids))
                counts.append(len(ids))
                hard_ids.append(row['key'])
                for key in ids:
                    index = identity_to_index.get(key)
                    if index is None:
                        index = len(identities)
                        identity_to_index[key] = index
                        identities.append(key)
                        uses.append(0)
                    uses[index] += 1
                    indices.append(index)
                offsets.append(len(indices))
            directory = a.output / ('%05d' % task)
            directory.mkdir(exist_ok=True)
            stem = directory / ('chunk-%06d' % chunk)
            data_path, counts_path = Path(str(stem) + '.npz'), Path(str(stem) + '.counts')
            np.savez_compressed(data_path, offsets=np.asarray(offsets, dtype=np.int64),
                                indices=np.asarray(indices, dtype=np.int32),
                                hard_ids=np.asarray(hard_ids, dtype=np.uint64))
            counts_path.write_text(''.join(str(n) + '\n' for n in counts))
            chunks.append({'sequence': len(chunks), 'task': task, 'chunk': chunk,
                           'hard_events': len(rows), 'draws': len(indices),
                           'first_new_identity': first_new,
                           'new_identities': len(identities) - first_new,
                           'schedule': str(data_path.resolve()), 'schedule_sha256': digest(data_path),
                           'counts': str(counts_path.resolve()), 'counts_sha256': digest(counts_path)})
            total += len(indices)
    frequency = dict(sorted(Counter(uses).items()))
    identity_path = a.output / 'unique-original-identities.npz'
    np.savez_compressed(identity_path, original_ids=np.asarray(identities, dtype=np.uint64),
                        uses=np.asarray(uses, dtype=np.uint8))
    expected = json.loads((a.metadata.parent / 'cms-pileup-sampling-20261027/interaction-reuse.json').read_text())
    if (total != expected['interaction_slots'] or len(identities) != expected['unique_interaction_ids']
            or frequency != {int(k): v for k, v in expected['interaction_id_use_frequency_histogram'].items()}
            or within_repeats != 0):
        raise ValueError('Allocation does not reproduce original reuse statistics')
    result = {'status': 'validated-identity-schedule-particles-not-yet-attached',
              'hard_events': 100000, 'draws': total, 'unique_particle_events_required': len(identities),
              'repeat_slots': total - len(identities),
              'repeat_slot_fraction': (total - len(identities)) / total,
              'identity_use_frequency_histogram': frequency, 'within_event_repeat_count': within_repeats,
              'identity_file': str(identity_path.resolve()), 'identity_sha256': digest(identity_path),
              'inputs': inputs, 'chunks': chunks,
              'method': 'One unique new particle event per original CMS interaction ID; preserve the complete in-time reuse graph and exact N for every hard event. Interleave1000-event chunks across sources.',
              'limits': ['Does not attach or generate particles; no physical library or100k simulation yet.',
                         'New particle mapping is statistical; original CMS particles are not claimed matched.',
                         'Preserves original cross-source/shared-ID structure; does not enforce new disjoint ML splits.',
                         'No out-of-time detector premixing or response closure.']}
    (a.output / 'index.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['status', 'hard_events', 'draws',
          'unique_particle_events_required', 'repeat_slots', 'repeat_slot_fraction']}, indent=2))


if __name__ == '__main__':
    main()
