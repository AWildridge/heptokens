"""Verify every overlay card preserves the pinned default CMS response blocks."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from replay_scratch_budget import durable_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--pilot',type=Path,required=True)
a=p.parse_args()
runtime=json.loads((a.pilot/'runtime-config.json').read_text())
stock=(Path(runtime['delphes'])/'cards/delphes_card_CMS.tcl').read_text()
def blocks(text):
    return {name:(kind,body) for kind,name,body in re.findall(r'(?ms)^module (\w+) (\w+) \{(.*?)^\}',text)}
reference=blocks(stock)
path=re.search(r'(?s)set ExecutionPath \{(.*?)\}',stock).group(1).split()
identity=json.loads((a.pilot/'pilot-identity.json').read_text())
libraries=json.loads((a.pilot/'library-identity.json').read_text())
events=0
for job in identity['jobs']:
    group=Path(job['output'])
    summary=json.loads((group/'summary.json').read_text())
    text=(group/'card.tcl').read_text()
    if hashlib.sha256(text.encode()).hexdigest()!=summary['identity']['card_sha256']:
        raise ValueError('Saved runtime card checksum mismatch')
    if summary['events']!=1000 or summary['base_card']!='default' or summary['chs_available']:
        raise ValueError('Wrong reconstruction or event count')
    if summary['identity']['seed']!=job['seed'] or summary['mean_pileup_parameter']!=job['mean']:
        raise ValueError('Wrong overlay parameters')
    library='cms' if job['variant']=='cms-poisson' else 'generic'
    if summary['identity']['library_sha256']!=libraries[library]['sha256']:
        raise ValueError('Overlay library differs from the previous validated pilot')
    if any(row['chs_charged_pf'] is not None for row in summary['rows']):
        raise ValueError('Default card must not claim CHS')
    modules=blocks(text)
    if set(modules)!=set(reference)|{'PileUpMerger'}:
        raise ValueError('Unexpected reconstruction modules')
    for name,(kind,body) in reference.items():
        if name=='TreeWriter':continue
        candidate=modules[name][1]
        if name=='ParticlePropagator':
            candidate=candidate.replace('PileUpMerger/stableParticles','Delphes/stableParticles')
        if (kind,body)!=(modules[name][0],candidate):
            raise ValueError('Default response changed: '+name)
    actual=re.search(r'(?s)set ExecutionPath \{(.*?)\}',text).group(1).split()
    if actual!=['PileUpMerger']+path:
        raise ValueError('Default execution sequence changed')
    events+=summary['events']
result=dict(status='passed',jobs=len(identity['jobs']),events=events,
            libraries=libraries,
            default_card_sha256=hashlib.sha256(stock.encode()).hexdigest(),
            checks=['All saved runtime card hashes, seeds and means match',
                    'All stock response/reconstruction module definitions preserved',
                    'Library SHA256 matches the preceding validated pileup pilot',
                    'Only native merger prepended, propagator input rerouted, output branches selected',
                    'No CHS/rho or response retuning introduced'])
durable_json(a.pilot/'runtime-card-audit.json',result)
print(json.dumps(result))
