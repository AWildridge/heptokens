"""CLI entry point. Raw inputs are immutable; each output pins code and cuts."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields
import json
import os
from pathlib import Path
import sqlite3
import sys

from .physics import Config, event_identity
from .produce import produce_source, utc_now
from .source import atomic_json, read_json, sample_records, sha256, source_root


def configure(path):
    if path is None:
        return Config()
    data = read_json(path)
    unknown = set(data)-{f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f'Unknown config keys: {unknown}')
    if 'invisible_pids' in data:
        data['invisible_pids'] = tuple(data['invisible_pids'])
    return Config(**data)


def organize(campaign, destination):
    """One directory rename, a compatibility alias, and small detector catalogs."""
    campaign, destination = Path(campaign).absolute(), Path(destination).absolute()
    if campaign.is_symlink() and campaign.resolve() == (destination/'raw').resolve():
        return dict(status='already organized', destination=str(destination))
    if campaign.is_symlink() or not campaign.is_dir() or destination.exists():
        raise ValueError('Use the original campaign and a fresh destination')
    manifest = read_json(campaign/'provenance/cms-qcd-source-manifest.json')
    accounting = read_json(campaign/'provenance/revisions/source-00006-recovery-20261028/CAMPAIGN-ACCOUNTING.json')
    if accounting['status'] != 'complete' or accounting['source_files'] != len(manifest['files']):
        raise ValueError('Campaign accounting is incomplete')
    # Verify all small source completion records exist before moving anything.
    locations = {entry['task']: source_root(campaign, entry['task']).relative_to(campaign)
                 for entry in manifest['files']}
    destination.mkdir()
    try:
        campaign.rename(destination/'raw')
        campaign.symlink_to(os.path.relpath(destination/'raw', campaign.parent), target_is_directory=True)
    except BaseException:
        if not campaign.exists() and (destination/'raw').exists():
            (destination/'raw').rename(campaign)
        raise
    samples = sample_records(manifest)
    for detector in ['CMS','ATLAS']:
        directory = destination/detector; directory.mkdir()
        with (directory/'sources.jsonl').open('w') as f:
            for entry in manifest['files']:
                row = dict(detector=detector, source_id=entry['task'], sample_id=str(entry['record']),
                    source_directory=str(Path('../raw')/locations[entry['task']]),
                    relative_to='this catalog directory', file_pattern=f'chunk-*/{detector}.h5',
                    original_uri=entry['uri'])
                # Catalog is one level deeper than destination; ../raw is correct.
                f.write(json.dumps(row)+'\n')
        (directory/'README.txt').write_text(
            f'{detector} reconstructed inputs\n\n'
            'sources.jsonl indexes all source directories and the detector-specific HDF5 filename.\n'
            'Physical paired shards and original provenance live under ../raw/.\n'
            'Use labels/VERSION/inputs/'+detector+'/ via the per-source partitions after label production.\n'
            'Do not feed truth/, targets/, or bookkeeping/ to a representation model.\n')
    (destination/'labels').mkdir()
    atomic_json(destination/'DATASET.json', dict(schema='paired-qcd-layout-v1', created_utc=utc_now(),
        original_path=str(campaign), raw='raw', detectors=['CMS','ATLAS'],
        events=accounting['total_events'], sources=accounting['source_files'], samples=samples,
        original_files_modified=False, compatibility_alias=str(campaign),
        source_manifest_sha256=sha256(destination/'raw/provenance/cms-qcd-source-manifest.json')))
    (destination/'README.txt').write_text(
        'Generator-synchronized CMS / ATLAS QCD multijet dataset\n\n'
        'CMS/ and ATLAS/: explicit detector catalogs (4,546 sources each).\n'
        'raw/: original paired shards, compressed stable truth and provenance, unchanged.\n'
        'labels/: versioned event/object/splitting sidecars; raw inputs are never overwritten.\n'
        'The previous campaign path is a compatibility symlink into raw/.\n'
        'Selection is generator pthat in 300-470, 470-600, 600-800, 800-1000 GeV bins, NOT HT.\n'
        'Truth is quantized, rapidity-limited stable MiniAOD truth, with no original pileup or genealogy.\n')
    return dict(status='organized', destination=str(destination), detectors=['CMS','ATLAS'],
                source_catalog_entries=len(locations), compatibility_alias=str(campaign))


def account_weights(campaign, output, tasks=None):
    """Disk-backed, replica-aware accounting; never infer sumw from event counts."""
    campaign, output = Path(campaign), Path(output)
    if output.exists():
        raise ValueError('Use a fresh accounting output')
    output.mkdir(parents=True)
    manifest_path = campaign/'provenance/cms-qcd-source-manifest.json'
    manifest = read_json(manifest_path); samples = sample_records(manifest)
    tasks = list(range(len(manifest['files']))) if tasks is None else tasks
    db = sqlite3.connect(output/'event-identities.sqlite')
    db.execute('CREATE TABLE events (id BLOB PRIMARY KEY, sample TEXT, weight REAL) WITHOUT ROWID')
    processed, duplicates = [], 0
    try:
        for task in tasks:
            source = source_root(campaign,task); complete = read_json(source/'COMPLETE.json')
            count = 0
            for chunk in complete['chunks']:
                root = source/chunk['chunk']; receipt = read_json(root/'receipt.json')
                if receipt['source']['uri'] != manifest['files'][task]['uri']:
                    raise ValueError('Accounting source identity mismatch')
                if sha256(root/'truth-reference.json.gz') != receipt['sha256']['truth-reference.json.gz']:
                    raise ValueError('Accounting reference checksum mismatch')
                reference = read_json(root/'truth-reference.json.gz')
                if reference['source_key'] != receipt['source']['uri'] or len(reference['events']) != chunk['events']:
                    raise ValueError('Accounting event count/source mismatch')
                sample = str(receipt['source']['record'])
                for e in reference['events']:
                    eid = bytes.fromhex(event_identity(sample,e['run'],e['lumi'],e['event']))
                    weight = float(e['weight'])
                    import math
                    if not math.isfinite(weight):
                        raise ValueError('Nonfinite accounting weight')
                    try:
                        db.execute('INSERT INTO events VALUES (?,?,?)',(eid,sample,weight))
                    except sqlite3.IntegrityError:
                        old = db.execute('SELECT sample,weight FROM events WHERE id=?',(eid,)).fetchone()
                        if old != (sample,weight):
                            raise ValueError('Conflicting replica weights')
                        duplicates += 1
                    count += 1
            if count != complete['exported_events']:
                raise ValueError('Source accounting count mismatch')
            db.commit(); processed.append(task)
        totals = {s:(int(n),float(w),float(a)) for s,n,w,a in db.execute(
            'SELECT sample,COUNT(*),SUM(weight),SUM(ABS(weight)) FROM events GROUP BY sample')}
        result = dict(created_utc=utc_now(), source_manifest_sha256=sha256(manifest_path),
            sources=len(processed), expected_sources=len(manifest['files']), duplicates=duplicates,
            source_tasks=processed, samples={})
        for sample, info in samples.items():
            n,w,a = totals.get(sample,(0,0.,0.))
            expected_tasks = {e['task'] for e in manifest['files'] if str(e['record']) == sample}
            valid = expected_tasks <= set(processed) and n == info['expected_events'] and w != 0
            result['samples'][sample] = dict(valid=valid, events=n, expected_events=info['expected_events'],
                sum_weights=w, sum_abs_weights=a, method='measured retained generator weights; deduplicated sample/run/lumi/event')
        result['valid'] = all(r['valid'] for r in result['samples'].values())
        atomic_json(output/'normalization.json', result)
        return result
    finally:
        db.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    o = sub.add_parser('organize'); o.add_argument('--campaign',type=Path,required=True);o.add_argument('--destination',type=Path,required=True)
    r = sub.add_parser('produce')
    r.add_argument('--campaign',type=Path,required=True); r.add_argument('--output',type=Path,required=True)
    r.add_argument('--source-task',type=int,action='append'); r.add_argument('--all-sources',action='store_true')
    r.add_argument('--max-chunks',type=int); r.add_argument('--events-per-chunk',type=int)
    r.add_argument('--workers',type=int,default=1); r.add_argument('--config',type=Path)
    r.add_argument('--normalization',type=Path); r.add_argument('--dask-address')
    v = sub.add_parser('validate');v.add_argument('--output',type=Path,required=True);v.add_argument('--plots',action='store_true')
    a = sub.add_parser('account-weights');a.add_argument('--campaign',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--source-task',type=int,action='append')
    args = p.parse_args()
    if args.command == 'organize':
        result = organize(args.campaign,args.destination)
    elif args.command == 'validate':
        from .validation import validate_run
        report = validate_run(args.output,args.plots)
        result = {k:v for k,v in report.items() if k not in ['summaries','unavailable']}
    elif args.command == 'account-weights':
        if not os.environ.get('SLURM_JOB_ID') and args.source_task is None:
            p.error('Full weight accounting requires a batch allocation; use selected sources for a local smoke test')
        result = account_weights(args.campaign,args.output,args.source_task)
    else:
        cfg = configure(args.config)
        manifest_path = args.campaign/'provenance/cms-qcd-source-manifest.json'
        manifest = read_json(manifest_path)
        if args.all_sources == bool(args.source_task):
            p.error('Choose --all-sources OR one or more --source-task values')
        tasks = list(range(len(manifest['files']))) if args.all_sources else sorted(set(args.source_task))
        if args.workers < 1 or any(t < 0 or t >= len(manifest['files']) for t in tasks):
            p.error('Invalid workers/source-task')
        if not os.environ.get('SLURM_JOB_ID') and not args.dask_address:
            if (args.max_chunks != 1 or args.events_per_chunk is None or args.events_per_chunk > 1000
                    or len(tasks) > 4 or args.workers > 2):
                p.error('Login-node mode is limited to four sources, one chunk each, <=1000 events/chunk and <=2 workers. Use SLURM or Dask for production.')
        normalization = read_json(args.normalization) if args.normalization else None
        if normalization and normalization['source_manifest_sha256'] != sha256(manifest_path):
            p.error('Normalization belongs to a different source manifest')
        common = dict(campaign=args.campaign, output=args.output,cfg=cfg,max_chunks=args.max_chunks,
                      events_per_chunk=args.events_per_chunk,normalization=normalization)
        results = []
        if args.dask_address:
            from distributed import Client, as_completed as dask_completed
            with Client(args.dask_address) as client:
                futures = [client.submit(produce_source,task=t,**common,pure=False) for t in tasks]
                for future in dask_completed(futures):
                    result = future.result(); results.append(result)
                    print(json.dumps({k:result[k] for k in ['source_id','events','started_elapsed_seconds']}),flush=True)
        elif args.workers == 1:
            for task in tasks:
                result = produce_source(task=task,**common); results.append(result)
                print(json.dumps({k:result[k] for k in ['source_id','events','started_elapsed_seconds']}),flush=True)
        else:
            # BLAS threads are limited by the launcher; parallelism is across sources.
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(produce_source,task=t,**common) for t in tasks]
                for future in as_completed(futures):
                    result = future.result(); results.append(result)
                    print(json.dumps({k:result[k] for k in ['source_id','events','started_elapsed_seconds']}),flush=True)
        result = dict(status='complete',partitions=len(results),events=sum(r['events'] for r in results))
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
