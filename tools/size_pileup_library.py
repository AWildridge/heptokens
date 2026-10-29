"""Size a bank by the probability of a duplicate inside ONE hard event.

Reference assumptions: Poisson(200) overlays, bank=30000, independent draws.
Pilot inputs: every recorded in-time count from the frozen 100k metadata.
No physics/cross-section weighting: this describes generated-event reuse.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path

from scipy.stats import poisson


def duplicate_probability(n, m, native=False):
    if n > m:
        return 1.0
    if n < 2:
        return 0.0
    if not native:
        return -math.expm1(sum(math.log1p(-i / m) for i in range(n)))
    # Native Nint(U*M), reject M: first entry has half the other weights.
    denom = m - 0.5
    half = math.log(n * 0.5 / denom) + sum(
        math.log((m - 1 - i) / denom) for i in range(n - 1))
    ordinary = (sum(math.log((m - 1 - i) / denom) for i in range(n))
                if n < m else -math.inf)
    maximum = max(half, ordinary)
    log_no = maximum + math.log(math.exp(half - maximum) + math.exp(ordinary - maximum))
    return max(0.0, -math.expm1(log_no))


def minimum_bank(hist, target, native):
    def probability(m):
        total = sum(hist.values())
        return sum(w * duplicate_probability(n, m, native) for n, w in hist.items()) / total
    low, high = 1, 30000
    while probability(high) > target:
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if probability(middle) <= target:
            high = middle
        else:
            low = middle + 1
    return {'minimum_entries': low, 'probability_at_minimum': probability(low),
            'probability_one_entry_below': probability(low - 1) if low > 1 else None,
            'probabilities': {str(m): probability(m) for m in (1000, 30000, 50600)}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Preserve existing evidence')
    samples, inputs = {}, []
    for path in sorted(a.pilot.glob('metadata-*.jsonl.gz')):
        with gzip.open(path, 'rt') as stream:
            samples[path.name] = Counter(json.loads(line)['n'] for line in stream)
        inputs.append({'path': str(path.resolve()), 'sha256': hashlib.file_digest(
            path.open('rb'), 'sha256').hexdigest()})
    pooled = sum(samples.values(), Counter())
    if sum(pooled.values()) != 100000:
        raise ValueError('Expected complete frozen 100k pilot')
    results = {}
    for native in (False, True):
        reference = sum(float(poisson.pmf(n, 200)) * duplicate_probability(n, 30000, native)
                        for n in range(501))
        results['native_sampler' if native else 'uniform_sampler'] = {
            'reference_duplicate_probability': reference,
            'pooled': minimum_bank(pooled, reference, native),
            'per_source': {k: minimum_bank(v, reference, native) for k, v in samples.items()}}
    profile = json.loads((a.pilot / 'target-profile.json').read_text())
    poisson_arms = {}
    for source, values in profile['samples'].items():
        mean = values['mean_true_interactions']
        histogram = {n: float(poisson.pmf(n, mean)) for n in range(151)}
        poisson_arms[source] = {
            'mean': mean, 'omitted_probability_above_150': float(poisson.sf(150, mean)),
            **minimum_bank(histogram, results['native_sampler']['reference_duplicate_probability'], True)}
    result = {
        'definition': 'At least one library identity repeated inside one reconstructed hard event',
        'reference': {'mean_pileup': 200, 'library_entries': 30000,
                      'multiplicity_assumption': 'Poisson; terms 0..500; omitted mass <1e-60'},
        'pilot_hard_events': sum(pooled.values()),
        'pilot_recorded_draws': sum(n * w for n, w in pooled.items()),
        'recorded_N_histogram': dict(sorted(pooled.items())), 'inputs': inputs,
        'results': results,
        'poisson_arm_results_native_sampler': poisson_arms,
        'poisson_arm_expected_total_draws': sum(
            v['events'] * v['mean_true_interactions'] for v in profile['samples'].values()),
        'chosen_bank_entries': 50600,
        'choice': 'Conservative equal-size banks; retains existing CMS diversity. NOT an inferred minimum.',
        'limitations': ['Assumed reference sizing rationale, not established author intent.',
                       'No constraint on reuse between hard events or across ML splits.',
                       'No finite-library physics convergence or ML leakage test.',
                       'Equal event weighting; measured pilot prefix, not campaign-wide guarantee.']}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
