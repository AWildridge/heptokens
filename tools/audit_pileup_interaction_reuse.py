"""Measure original in-time CMS interaction-ID reuse in frozen pilot metadata.

Treat run/lumi/event as an interaction identity within the identified CMS
MinBias parent production. Counts are unweighted generated-event statistics.
Only retained in-time records are audited; no OOT or particle-level completeness
claim follows. Hard-event file/chunk boundaries are NOT production-job boundaries.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Preserve existing audit')
    all_ids, frames = Counter(), Counter()
    sources, hashes, total_within_repeats = {}, [], 0
    source_sets = {}
    for path in sorted(a.pilot.glob('metadata-*.jsonl.gz')):
        counts = Counter()
        hard_ids, previous = set(), set()
        within_events, within_slots, adjacent_overlap = 0, 0, 0
        events, slots, empty = 0, 0, 0
        chunks, chunk = [], set()
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                hard = tuple(row['key'])
                if hard in hard_ids:
                    raise ValueError('Duplicate hard-event identity within source')
                hard_ids.add(hard)
                ids = [tuple(key) for key in row['ids']]
                if len(ids) != row['n']:
                    raise ValueError('Retained in-time ID count differs from recorded N')
                current = set(ids)
                if len(ids) != len(current):
                    within_events += 1
                    within_slots += len(ids) - len(current)
                adjacent_overlap += len(current & previous)
                previous = current
                counts.update(ids)
                chunk.update(ids)
                events += 1
                slots += len(ids)
                if not ids:
                    empty += 1
                else:
                    # Canonical unordered interaction-set fingerprint. A repeated
                    # set alone does not prove identical detector-stage premix.
                    payload = json.dumps(sorted(ids), separators=(',', ':')).encode()
                    frames[(len(ids), hashlib.sha256(payload).hexdigest())] += 1
                if events % 1000 == 0:
                    chunks.append({'hard_event_start': events - 1000,
                                   'hard_event_stop': events,
                                   'unique_interaction_ids': len(chunk)})
                    chunk = set()
        if chunk:
            chunks.append({'hard_event_stop': events, 'unique_interaction_ids': len(chunk)})
        source_sets[path.name] = set(counts)
        sources[path.name] = {
            'hard_events': events, 'interaction_slots': slots,
            'unique_interaction_ids': len(counts),
            'repeated_slots_beyond_first_use': slots - len(counts),
            'hard_events_with_within_event_repetition': within_events,
            'within_event_repeated_slots': within_slots,
            'overlap_count_between_adjacent_hard_events': adjacent_overlap,
            'zero_pileup_hard_events': empty, 'chunks_of_1000': chunks,
        }
        total_within_repeats += within_events
        all_ids.update(counts)
        with path.open('rb') as stream:
            hashes.append({'path': str(path.resolve()),
                           'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()})
        print(path.name, events, slots, len(counts), flush=True)
    total_events = sum(v['hard_events'] for v in sources.values())
    if total_events != 100000:
        raise ValueError('Expected complete matched100k prefix sample')
    cross = {}
    names = sorted(source_sets)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            cross[name + ' vs ' + other] = len(source_sets[name] & source_sets[other])
    result = {
        'status': 'complete-in-time-ID-audit', 'hard_events': total_events,
        'interaction_slots': sum(all_ids.values()),
        'unique_interaction_ids': len(all_ids),
        'repeated_slots_beyond_first_use': sum(all_ids.values()) - len(all_ids),
        'unique_ids_used_more_than_once': sum(n > 1 for n in all_ids.values()),
        'maximum_interaction_id_uses': max(all_ids.values()),
        'interaction_id_use_frequency_histogram': dict(sorted(Counter(all_ids.values()).items())),
        'hard_events_with_within_event_repetition': total_within_repeats,
        'distinct_nonempty_interaction_sets': len(frames),
        'repeated_nonempty_interaction_set_slots': sum(frames.values()) - len(frames),
        'repeated_nonempty_set_sizes': dict(sorted(Counter(
            {n: sum(count - 1 for (size, _), count in frames.items() if size == n)
             for n in {size for size, _ in frames} if any(
                 count > 1 and size == n for (size, _), count in frames.items())}).items())),
        'per_source': sources, 'cross_source_shared_unique_ids': cross, 'inputs': hashes,
        'method': 'Exact run/lumi/event identity counting, equal weight per hard event; unordered nonempty sets SHA256 fingerprinted.',
        'limits': ['Only stored in-time interaction IDs; excludes OOT and detector response.',
                   'Assumes IDs identify interactions within the common MinBias parent production.',
                   'One first25k-event file prefix per pThat bin; no campaign-wide representativeness claim.',
                   'Hard-event adjacency and1000-event chunks do not identify original production jobs.',
                   'Interaction-set repetition does not prove the entire premix record or vertices are identical.']}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('hard_events', 'interaction_slots',
        'unique_interaction_ids', 'repeated_slots_beyond_first_use',
        'maximum_interaction_id_uses', 'hard_events_with_within_event_repetition',
        'repeated_nonempty_interaction_set_slots')}, indent=2))


if __name__ == '__main__':
    main()
