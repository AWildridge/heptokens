"""Validate linked tables and aggregate reusable weighted distributions."""

import json
import math
from pathlib import Path

import hist
import numpy as np
import pyarrow.parquet as pq

from .physics import event_kinematics
from .source import atomic_json, read_json, sha256


AXES = dict(ht=(100, 0, 6000), jet_pt=(100, 0, 3000), jet_mass=(80, 0, 800),
            n_jets=(25, -0.5, 24.5), dijet_mass=(100, 0, 8000),
            dijet_delta_y=(80, -6, 6), dijet_delta_phi=(64, -math.pi, math.pi),
            psi=(64, -math.pi, math.pi), cos2psi=(50, -1, 1), sin2psi=(50, -1, 1),
            response=(80, 0, 4), matching_dr=(50, 0, .5))


def rows(partition, name):
    return pq.read_table(Path(partition)/(name+'.parquet')).to_pylist()


class Summary:
    def __init__(self):
        self.histograms = {}
        self.counts = {}

    def count(self, key, amount=1):
        self.counts[key] = self.counts.get(key, 0)+amount

    def fill(self, name, value, weight):
        if value is None or weight is None:
            return
        if not math.isfinite(value) or not math.isfinite(weight):
            raise ValueError(f'Nonfinite distribution input {name}')
        if name not in self.histograms:
            self.histograms[name] = hist.Hist(hist.axis.Regular(*AXES[name]), storage=hist.storage.Weight())
        self.histograms[name].fill(value, weight=weight)

    def export(self):
        return dict(counts=self.counts, distributions={k:dict(edges=h.axes[0].edges.tolist(),
            sumw=h.values(flow=True).tolist(), sumw2=h.variances(flow=True).tolist(),
            flow_bins='first=underflow,last=overflow') for k,h in self.histograms.items()})


def check_tree(nodes, jets):
    by_jet = {}
    for n in nodes:
        by_jet.setdefault(n['jet_id'], {})[n['node_id']] = n
    for j in jets:
        group = by_jet.get(j['jet_id'], {})
        if not group:
            raise ValueError('Missing selected-jet tree')
        roots = [n for n in group.values() if n['parent_id'] is None]
        if len(roots) != 1 or sorted(roots[0]['constituent_indices']) != sorted(j['constituent_indices']):
            raise ValueError('Tree root membership mismatch')
        if len(group) != 2*len(j['constituent_indices'])-1:
            raise ValueError('Tree cardinality mismatch')
        for n in group.values():
            if n['is_leaf']:
                if len(n['constituent_indices']) != 1:
                    raise ValueError('Non-singleton tree leaf')
                continue
            b, c = group[n['child_hard_id']], group[n['child_soft_id']]
            if b['parent_id'] != n['node_id'] or c['parent_id'] != n['node_id']:
                raise ValueError('Tree parent/child mismatch')
            if set(b['constituent_indices']) & set(c['constituent_indices']):
                raise ValueError('Tree daughters share constituents')
            if sorted(b['constituent_indices']+c['constituent_indices']) != sorted(n['constituent_indices']):
                raise ValueError('Tree daughter union mismatch')
            for k in ['px','py','pz','energy']:
                if not math.isclose(n[k], b[k]+c[k], rel_tol=1e-10, abs_tol=1e-8):
                    raise ValueError('Tree four-momentum conservation failed')


def validate_run(output, make_plots=False):
    output = Path(output)
    metadata = read_json(output/'metadata.json')
    cfg = metadata['contract']['config']
    summaries, commits = {}, []
    seen_groups = {}
    total_events = 0
    for partition in sorted(output.glob('source-*')):
        if not (partition/'COMMIT.json').is_file():
            raise ValueError(f'Uncommitted partition: {partition}')
        commit = read_json(partition/'COMMIT.json'); commits.append(commit)
        for name, info in commit['tables'].items():
            if sha256(partition/name) != info['sha256']:
                raise ValueError(f'Changed sidecar {partition/name}')
        book = rows(partition, 'bookkeeping/events')
        by_event = {r['event_id']: r for r in book}
        if len(by_event) != len(book):
            raise ValueError('Duplicate stable-event rows within partition')
        duplicates = set()
        for b in book:
            group = b['generator_event_id']
            previous = seen_groups.get(group)
            if previous is not None:
                if previous != (b['truth_fingerprint'], b['split'], b['sample_id'], b['original_weight']):
                    raise ValueError('Conflicting event replica fingerprint, split, sample or weight')
                duplicates.add(b['event_id'])
            else:
                seen_groups[group] = (b['truth_fingerprint'], b['split'], b['sample_id'], b['original_weight'])
        total_events += len(book)-len(duplicates)
        truth_events = rows(partition, 'truth/events')
        truth_jets = rows(partition, 'truth/jets')
        truth_nodes = rows(partition, 'truth/nodes')
        truth_angles = rows(partition, 'truth/angles')
        truth_particles = rows(partition, 'truth/particles')
        truth_jet_ids = {j['jet_id'] for j in truth_jets}
        if cfg['include_trees']:
            check_tree(truth_nodes, truth_jets)
        truth_by_event = {}
        for j in truth_jets:
            truth_by_event.setdefault(j['event_id'], []).append(j)
        for detector in ['CMS', 'ATLAS']:
            reco = rows(partition, f'inputs/{detector}/jets')
            selected = [j for j in reco if j['pt'] >= cfg['jet_min_pt'] and j['eta'] is not None
                        and abs(j['eta']) < cfg['jet_abs_eta']]
            reco_ids = {j['jet_id'] for j in selected}
            if cfg['include_trees']:
                check_tree(rows(partition, f'inputs/{detector}/nodes'), selected)
            matches = rows(partition, f'targets/{detector}/jet_matches')
            if {m['reco_jet_id'] for m in matches} != reco_ids or len(matches) != len(reco_ids):
                raise ValueError('Selected reco-jet coverage mismatch')
            linked = [m['truth_jet_id'] for m in matches if m['matched']]
            if len(linked) != len(set(linked)) or not set(linked) <= truth_jet_ids:
                raise ValueError('Jet matching is not a valid one-to-one truth join')
            if any(j['event_id'] not in by_event for j in reco):
                raise ValueError('Dangling reco event ID')
            object_truth = rows(partition, f'targets/{detector}/object_truth')
            coverage = rows(partition, f'targets/{detector}/truth_jet_coverage')
            reco_angles = rows(partition, f'inputs/{detector}/angles')
            reco_by_event = {}
            for j in selected:
                reco_by_event.setdefault(j['event_id'], []).append(j)
            def summary(eid, level, mode='raw_generator_weight'):
                key = f"{by_event[eid]['sample_id']}/{detector}/{level}/{mode}"
                return summaries.setdefault(key, Summary())
            def weight(eid, mode):
                return by_event[eid]['original_weight' if mode == 'raw_generator_weight' else 'physical_weight_pb']
            for eid in by_event:
                if eid in duplicates:
                    continue
                s = summary(eid, 'bookkeeping')
                s.count('events'); s.count('normalization_missing', not by_event[eid]['normalization_valid'])
                s.count('synchronization_valid', by_event[eid]['synchronization_valid'])
            for mode in ['raw_generator_weight', 'physical_pb']:
                for e in truth_events:
                    eid = e['event_id']
                    if eid in duplicates:
                        continue
                    for level, kin in [('truth', e), ('reco', event_kinematics(reco_by_event.get(eid, [])))]:
                        s = summary(eid, level, mode)
                        for field in ['ht', 'n_jets', 'dijet_mass', 'dijet_delta_y', 'dijet_delta_phi']:
                            s.fill(field, kin[field], weight(eid, mode))
                for level, jets in [('truth', truth_jets), ('reco', selected)]:
                    for j in jets:
                        eid = j['event_id']
                        if eid in duplicates:
                            continue
                        s = summary(eid, level, mode)
                        s.fill('jet_pt', j['pt'], weight(eid, mode)); s.fill('jet_mass', j['mass'], weight(eid, mode))
                        if mode == 'raw_generator_weight':
                            s.count('jets')
                            if level == 'truth':
                                s.count('heavy_hadron_flavor_missing', not j['heavy_hadron_flavor_valid'])
                                s.count('parton_flavor_missing', not j['parton_flavor_valid'])
                                s.count('heavy_flavor_unknown_sumw', weight(eid, mode))
                                s.count('parton_flavor_unknown_sumw', weight(eid, mode))
                for level, angles in [('truth', truth_angles), ('reco', reco_angles)]:
                    for a in angles:
                        eid = a['event_id']
                        if eid in duplicates:
                            continue
                        s = summary(eid, level, mode)
                        if mode == 'raw_generator_weight':
                            s.count('angles'); s.count('angles_degenerate', not a['geometric_valid'])
                            s.count('angles_unresolved', not a['resolution_valid']); s.count('angles_valid', a['valid'])
                        if a['valid']:
                            for field in ['psi','cos2psi','sin2psi']:
                                s.fill(field, a[field], weight(eid, mode))
                for m in matches:
                    eid = m['event_id']
                    if eid in duplicates:
                        continue
                    s = summary(eid, 'matching', mode)
                    if mode == 'raw_generator_weight':
                        s.count('reco_jets'); s.count('matched', m['matched']); s.count('ambiguous',m['ambiguous'])
                    s.fill('response', m['pt_response'], weight(eid, mode))
                    s.fill('matching_dr', m['delta_r'], weight(eid, mode))
            for row in coverage:
                if row['event_id'] not in duplicates:
                    s = summary(row['event_id'], 'matching'); s.count('truth_jets'); s.count('truth_matched',row['matched'])
            for row in object_truth:
                if row['event_id'] not in duplicates:
                    s = summary(row['event_id'], 'objects'); s.count('objects'); s.count('association_missing',not row['association_valid'])
            for row in truth_particles:
                if row['event_id'] not in duplicates:
                    s = summary(row['event_id'], 'truth_particles'); s.count('particles'); s.count('charge_missing',not row['charge_valid'])
    exported = {k:v.export() for k,v in sorted(summaries.items())}
    for value in exported.values():
        c = value['counts']; fractions = {}
        for num, den in [('matched','reco_jets'),('truth_matched','truth_jets'),('ambiguous','reco_jets'),
            ('association_missing','objects'),('heavy_hadron_flavor_missing','jets'),('parton_flavor_missing','jets'),
            ('angles_degenerate','angles'),('angles_unresolved','angles'),('angles_valid','angles'),('charge_missing','particles')]:
            if c.get(den, 0):
                fractions[num+'/'+den] = c.get(num, 0)/c[den]
        value['fractions'] = fractions
    elapsed = sum(c['started_elapsed_seconds'] for c in commits)
    n_produced = sum(c['events'] for c in commits)
    n_expected = sum(s['expected_events'] for s in metadata['samples'].values())
    report = dict(status='passed', validation_scope='committed partitions only; no claims for unprocessed data',
        unique_events=total_events, produced_rows=n_produced, committed_sources=len(commits),
        event_split_consistent=True, truth_reco_joins_valid=True, trees_valid=True if cfg['include_trees'] else None,
        deferred_targets=metadata.get('deferred_targets',[]),
        campaign_expected_events=n_expected, campaign_complete=n_produced == n_expected
            and len(commits) == metadata['source_files'] and all(c['complete_source'] for c in commits),
        selection_variable='generator_pthat (NOT HT)', summaries=exported,
        unavailable=metadata['unavailable'],
        synchronization='archived independent ROOT content comparison, freshly hash-bound to retained payload; stable-state fingerprints and identity joins recomputed',
        normalization='Raw weighted distributions are diagnostic. Physical-pb distributions are empty until measured full-sample sumw is provided. No equal-bin weighting.',
        resources=dict(total_worker_seconds=elapsed, produced_events=n_produced,
            worker_seconds_per_event=elapsed/n_produced if n_produced else None,
            projected_core_hours=elapsed/n_produced*n_expected/3600 if n_produced else None,
            bytes=sum(t['bytes'] for c in commits for t in c['tables'].values()),
            peak_worker_rss_MiB=max((c['max_rss_MiB'] for c in commits),default=0)),
        source_partitions=[c['source_id'] for c in commits])
    atomic_json(output/'validation.json', report)
    lines = ['# QCD label validation', '', f"Status: passed for {total_events:,} unique events in {len(commits)} source partitions.",
        '', report['validation_scope'], '', 'Samples are generator pthat bins, not HT bins.', '',
        '| Sample | Detector | Reco matched | Truth covered | Ambiguous reco |', '|---|---|---:|---:|---:|']
    for key, v in exported.items():
        if '/matching/raw_generator_weight' in key:
            a, d, *_ = key.split('/'); f = v['fractions']
            lines.append(f"| {a} | {d} | {f.get('matched/reco_jets',0):.2%} | {f.get('truth_matched/truth_jets',0):.2%} | {f.get('ambiguous/reco_jets',0):.2%} |")
    lines.extend(['', '## Unavailable targets', ''])
    lines.extend(f'- **{k}**: {v}' for k,v in report['unavailable'].items())
    lines.extend(['', '## Normalization and synchronization', '',report['normalization'],'',report['synchronization'],
        '', '## Resource estimate', '', f"Measured worker time: {elapsed:.1f} s. Projected full-campaign CPU time: {report['resources']['projected_core_hours']:.1f} core-hours.",
        'The estimate includes small-sample I/O and initialization and is not a production throughput guarantee. Use source partitions on batch workers; keep login-node work small.'])
    (output/'validation.md').write_text('\n'.join(lines)+'\n')
    if make_plots:
        plot_report(output/'validation.json', output/'plots')
    return report


def plot_report(report_path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)
    report = read_json(report_path); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for sample in sorted({k.split('/')[0] for k in report['summaries']}):
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        for ax, variable, level in zip(axes.flat,['ht','jet_pt','cos2psi','response'],['reco','reco','reco','matching']):
            for detector in ['CMS','ATLAS']:
                key = f'{sample}/{detector}/{level}/raw_generator_weight'
                h = report['summaries'].get(key,{}).get('distributions',{}).get(variable)
                if h:
                    values = np.asarray(h['sumw'][1:-1]); total = values.sum()
                    ax.stairs(values/total if total else values, h['edges'], label=detector)
            ax.set_xlabel(variable+(' [GeV]' if variable in ['ht','jet_pt'] else ''))
            ax.set_ylabel('Raw-weighted shape / bin'); ax.legend(fontsize=11)
            hep.label.exp_label(exp='TREASURE Simulation', text='', data=False, ax=ax, fontsize=12)
        fig.suptitle(f'QCD source {sample}; validation subset',fontsize=16)
        fig.tight_layout(); fig.savefig(output/f'sample-{sample}.png',dpi=130)
        fig.savefig(output/f'sample-{sample}.pdf'); plt.close(fig)
