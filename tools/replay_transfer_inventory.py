"""Export expected object hashes for committed EOS-to-NERSC transfers.

This is a partial campaign snapshot, never a dataset READY marker. Only sources
with a completed archive record and matching local commit proof are exported.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(committed, records):
    sources=[]
    for record_path in sorted(records):
        record=json.loads(Path(record_path).read_text())
        if record['status']!='verified-on-eos':continue
        task=record['task'];source=Path(committed)/'sources'/('%05d'%task)
        final_path=source/'SOURCE-COMMIT.json';final=json.loads(final_path.read_text())
        proof=json.loads((source/'remote-source-copy.json').read_text())
        if sha(final_path)!=record['remote_manifest']['sha256'] or record['remote_manifest']!=proof['remote_manifest']:
            raise ValueError('Source commit proof differs: '+str(task))
        if final['source']['task']!=task or final['exported_events']!=record['events'] or final['validation_limit'] is not None:
            raise ValueError('Invalid production source accounting')
        objects={final_path.name:proof['remote_manifest']}
        if final['schema']=='heptokens-remote-source-bundle-v1':
            objects.update({'source.tar':final['archive'],'BUNDLE-MANIFEST.json':final['bundle_receipt']})
        else:objects.update(final['files'])
        sources.append({'task':task,'events':record['events'],'phase':record.get('phase'),
            'source':final['source'],'objects':{name:{k:item[k] for k in ['url','bytes','sha256']} for name,item in objects.items()}})
    tasks=[r['task'] for r in sources]
    if len(tasks)!=len(set(tasks)):raise ValueError('Duplicate source records')
    return {'status':'partial-campaign-transfer-snapshot','sources':sources,'source_count':len(sources),
        'events':sum(r['events'] for r in sources),'object_bytes':sum(v['bytes'] for r in sources for v in r['objects'].values()),
        'cleanup_authorized':False,'destination_acknowledgements':[]}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--committed',type=Path,required=True)
    parser.add_argument('--records-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=inventory(args.committed,args.records_directory.glob('production-archive-*.json'))
    temporary=args.output.with_suffix(args.output.suffix+'.tmp')
    temporary.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');temporary.replace(args.output)
    print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2))
