"""Shared receipt and source-accounting validation for replay transports."""
import json
from pathlib import Path


def validate_source_inventory(source, allow_prefix, digest):
    complete=json.loads((source/'COMPLETE.json').read_text())
    if complete['status']!='complete' or (complete['validation_limit'] is not None and not allow_prefix):
        raise ValueError('Full-source completion is required; prefixes need explicit pilot mode')
    if source.name!='%05d'%complete['source']['task']:
        raise ValueError('Source task identity differs')
    index=json.loads((source/'export-index.json').read_text())
    if index['exported_events']!=complete['exported_events'] or len(index['chunks'])!=len(complete['chunks']):
        raise ValueError('Export accounting differs')
    files=['COMPLETE.json','export-index.json']
    files += [path.name for path in sorted(source.glob('export-*.log'))]
    count=0
    for number,item in enumerate(complete['chunks']):
        chunk=item['chunk']
        if chunk!='chunk-%06d'%number:
            raise ValueError('Noncanonical chunk identity')
        directory=source/chunk
        receipt=json.loads((directory/'receipt.json').read_text())
        validation=json.loads((directory/'validation.json').read_text())
        if receipt['status']!='complete' or receipt['source']!=complete['source'] or receipt['events']!=item['events'] or receipt['input_sha256']!=index['chunks'][number]['sha256']:
            raise ValueError('Source/chunk identity or event accounting differs')
        if receipt['seed']!=420000+complete['source']['task']*1000+number:
            raise ValueError('Seed rule differs')
        if validation['status']!='passed' or any(validation['detectors'][detector]['status']!='passed' for detector in ['CMS','ATLAS']):
            raise ValueError('Detector validation is incomplete')
        for name,expected in receipt['sha256'].items():
            if Path(name).name!=name or digest(directory/name,'sha256')!=expected or (directory/name).stat().st_size!=receipt['bytes'][name]:
                raise ValueError('Invalid source payload: '+name)
            files.append(chunk+'/'+name)
        files.append(chunk+'/receipt.json')
        count+=receipt['events']
    if count!=complete['exported_events'] or (complete['validation_limit'] is None and count!=complete['source_events']):
        raise ValueError('Source event count differs')
    return complete, index, files, count
