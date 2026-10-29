"""Pinned CMSSW Python2 launcher/build entrypoint for the native gen view."""
from __future__ import print_function
import argparse
import json
import os
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
p=argparse.ArgumentParser()
p.add_argument('--build',action='store_true')
p.add_argument('--source')
p.add_argument('--build-dir')
p.add_argument('--library')
p.add_argument('--input')
p.add_argument('--summary')
p.add_argument('--metadata')
a=p.parse_args()
for library in ['libFWCoreFWLite.so','libDataFormatsFWLite.so','libDataFormatsPatCandidates.so','libDataFormatsHepMCCandidate.so','libSimDataFormatsGeneratorProducts.so']:
    if ROOT.gSystem.Load(library)<0:raise RuntimeError('Cannot load '+library)
ROOT.FWLiteEnabler.enable()
if a.build:
    if not os.path.isdir(a.build_dir):os.makedirs(a.build_dir)
    libraries=['DataFormatsFWLite','FWCoreFWLite','DataFormatsPatCandidates','DataFormatsHepMCCandidate','SimDataFormatsGeneratorProducts']
    ROOT.gSystem.AddLinkedLibs(' '+' '.join(str(ROOT.gSystem.DynamicPathName('lib'+name+'.so',True)) for name in libraries))
    if not ROOT.gSystem.CompileMacro(a.source,'kO','gen_backfill_native',a.build_dir):
        raise RuntimeError('Native generator exporter failed to compile')
    print('Native generator exporter compiled')
else:
    if ROOT.gSystem.Load(a.library)<0:raise RuntimeError('Cannot load validated generator exporter')
    meta=open(a.metadata).read()
    count=ROOT.export_generator_view(a.input,a.summary,meta)
    print(json.dumps(dict(events=int(count))))
