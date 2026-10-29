"""Audit replay sources; publish READY only after the entire frozen manifest passes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time

from replay_cms_collection import atomic_json, sha
from commit_replay_gfal import remote_adler, verify_staged_file, run


def verify_source(directory, source, manifest_sha256, scratch=None, remote=False, stability_seconds=30):
    result=json.loads((directory/'COMPLETE.json').read_text())
    if (result['status']!='complete' or result['manifest_sha256']!=manifest_sha256 or
        result['validation_limit'] is not None or result['source_events']!=result['exported_events'] or
        result['source']!=source):
        raise ValueError('Invalid full-source completion')
    index=json.loads((directory/'export-index.json').read_text())
    if index['exported_events']!=result['exported_events'] or len(index['chunks'])!=len(result['chunks']):
        raise ValueError('Export index accounting differs')
    evidence=None
    if remote and json.loads((directory/'SOURCE-COMMIT.json').read_text()).get('schema')=='heptokens-remote-source-bundle-v1':
        from replay_remote_bundle import audit_bundle
        return audit_bundle(directory,source,manifest_sha256,stability_seconds)
    if remote:
        proof=json.loads((directory/'remote-source-copy.json').read_text())
        final=json.loads((directory/'SOURCE-COMMIT.json').read_text())
        if (proof['status']!='verified-remote-source' or
            sha(directory/'SOURCE-COMMIT.json')!=proof['remote_manifest']['sha256'] or
            final['source']!=source or final['validation_limit'] is not None or
            final['manifest_sha256']!=manifest_sha256 or final['chunks']!=result['chunks'] or
            final['source_events']!=result['source_events'] or final['exported_events']!=result['exported_events']):
            raise ValueError('Remote final manifest differs from source accounting')
        evidence={name:{key:item[key] for key in ['url','bytes','sha256','adler32']}
                  for name,item in proof['files'].items()}
        if evidence!=final['files']:
            raise ValueError('Remote proof differs from final manifest')
        evidence=dict(evidence, **{'SOURCE-COMMIT.json':proof['remote_manifest']})
        for name,item in evidence.items():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Unsafe source-relative path')
            if remote_adler(item['url'])!=item['adler32']:
                raise ValueError('Remote file changed')
        time.sleep(stability_seconds)
        for name,item in evidence.items():
            verify_staged_file(directory/name,item,stability_seconds)
        for name in ['COMPLETE.json','export-index.json']:
            if sha(directory/name)!=evidence[name]['sha256']:
                raise ValueError('Local source metadata differs from remote commit')
    count=0
    for number,item in enumerate(result['chunks']):
        if item['chunk']!='chunk-%06d'%number:
            raise ValueError('Noncanonical chunk sequence')
        chunk=directory/item['chunk']
        receipt=json.loads((chunk/'receipt.json').read_text())
        if (receipt['status']!='complete' or receipt['source']!=source or
            receipt['events']!=item['events'] or receipt['seed']!=420000+source['task']*1000+number or
            receipt['input_sha256']!=index['chunks'][number]['sha256']):
            raise ValueError('Invalid chunk identity or accounting')
        if remote and sha(chunk/'receipt.json')!=evidence[item['chunk']+'/receipt.json']['sha256']:
            raise ValueError('Local receipt differs from remote commit')
        for filename,size in receipt['bytes'].items():
            if Path(filename).name!=filename:
                raise ValueError('Unsafe payload path')
            if remote:
                committed=evidence[item['chunk']+'/'+filename]
                if committed['bytes']!=size or committed['sha256']!=receipt['sha256'][filename]:
                    raise ValueError('Payload differs from receipt')
            else:
                if (chunk/filename).stat().st_size!=size or sha(chunk/filename)!=receipt['sha256'][filename]:
                    raise ValueError('Durable payload missing or corrupted')
                if filename.endswith('.h5'):
                    training=scratch/'training'/('%05d'%source['task'])/item['chunk']/filename
                    if training.stat().st_size!=size or sha(training)!=receipt['sha256'][filename]:
                        raise ValueError('Scratch training payload missing or corrupted')
        if remote:
            with tempfile.TemporaryDirectory(prefix='heptokens-audit-') as temporary:
                validation_path=Path(temporary)/'validation.json'
                committed=evidence[item['chunk']+'/validation.json']
                run('gfal-copy','--checksum','ADLER32',committed['url'],validation_path.as_uri())
                if sha(validation_path)!=committed['sha256']:
                    raise ValueError('Validation readback changed')
                validation=json.loads(validation_path.read_text())
        else:
            validation=json.loads((chunk/'validation.json').read_text())
        if (validation['status']!='passed' or
            any(validation['detectors'][d]['status']!='passed' for d in ['CMS','ATLAS'])):
            raise ValueError('Failed paired detector validation')
        count+=receipt['events']
    if count!=result['source_events']:
        raise ValueError('Source event accounting mismatch')
    return count,len(result['chunks'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--layout',type=Path)
    parser.add_argument('--remote',action='store_true',help='Verify immutable EOS source commits with full readbacks')
    parser.add_argument('--audit-only',action='store_true',help='Write a report without publishing READY')
    parser.add_argument('--tasks',type=int,nargs='+')
    parser.add_argument('--report',type=Path)
    parser.add_argument('--stability-seconds',type=int,default=30)
    args=parser.parse_args()
    if args.stability_seconds<1:parser.error('Stability interval must be positive')
    if args.tasks is not None and not args.audit_only:parser.error('Source subsets require --audit-only')
    layout=json.loads((args.layout or args.campaign/'layout.json').read_text())
    manifest_path=args.campaign/'cms-qcd-source-manifest.json'
    manifest=json.loads(manifest_path.read_text());manifest_sha256=sha(manifest_path)
    root=Path(layout['durable_replay_root']);scratch=Path(layout['scratch_replay_root'])
    totals={str(record['record']):0 for record in manifest['records']}
    selected=manifest['files']
    if args.tasks is not None:
        if len(args.tasks)!=len(set(args.tasks)) or any(task<0 or task>=len(selected) for task in args.tasks):
            parser.error('Invalid or duplicate source tasks')
        selected=[selected[task] for task in args.tasks]
    chunks=0;passed=[];failures=[]
    for source in selected:
        try:
            events,source_chunks=verify_source(root/'sources'/('%05d'%source['task']),source,manifest_sha256,
                                               scratch,args.remote,args.stability_seconds)
            totals[str(source['record'])]+=events;chunks+=source_chunks;passed.append(source['task'])
        except Exception as error:
            if not args.audit_only:raise
            failures.append({'task':source['task'],'error':repr(error)})
    complete=len(passed)==len(manifest['files']) and not failures
    if complete:
        for record in manifest['records']:
            if totals[str(record['record'])]!=record['events']:
                raise ValueError('Catalogue event accounting mismatch')
    report={'status':'complete' if complete else 'incomplete','verified_utc':datetime.now(timezone.utc).isoformat(),
            'source_files':len(passed),'expected_source_files':len(manifest['files']),
            'events_by_record':totals,'total_events':sum(totals.values()),'chunks':chunks,
            'manifest_sha256':manifest_sha256,'passed_tasks':passed,'failures':failures,
            'scope':manifest['scope'],'detectors':['default CMS','default ATLAS'],
            'durable_root':str(root),'scratch_training_root':None if args.remote else str(scratch),
            'normalization':'raw weights; no cross-section or luminosity scaling',
            'verification':'full source/chunk/event accounting; paired validation; current size/SHA256 checks'+
                           ('; stable remote Adler32 and complete readbacks' if args.remote else '')}
    if args.report:atomic_json(args.report,report)
    if not args.audit_only:
        if not complete:raise ValueError('Entire frozen collection is required for READY')
        for path in ([root/'READY.json'] if args.remote else [root/'READY.json',scratch/'READY.json']):
            atomic_json(path,report)
    print(json.dumps(report,indent=2))
    return 1 if failures else 0


if __name__=='__main__':
    raise SystemExit(main())
