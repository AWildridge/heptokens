"""Prepare an isolated Delphes fork with optional ordered PU consumption.

Default reconstruction and default random entry selection are unchanged.
Ordered mode requires one recorded N per hard event and consumes the next N
particle-bank records exactly once. Exhaustion/unused records fail closed.
This utility prepares source only; compile and audit the separate runtime.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Source does not match pinned patch anchor: ' + old[:70])
    return text.replace(old, new, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Preserve existing runtime')
    paths = ['modules/PileUpMerger.h', 'modules/PileUpMerger.cc']
    original = {name: (a.source / name).read_text() for name in paths}
    header, body = original[paths[0]], original[paths[1]]
    header = replace_once(header, '#include "classes/DelphesModule.h"',
        '#include "classes/DelphesModule.h"\n#include <fstream>')
    header = replace_once(header, '  DelphesPileUpReader *fReader; //!', '''  DelphesPileUpReader *fReader; //!
  Bool_t fOrderedPileUp; //!
  Long64_t fOrderedEntry; //!
  std::ifstream *fOrderedCounts; //!''')
    header = replace_once(header, 'ClassDef(PileUpMerger, 1)', 'ClassDef(PileUpMerger, 2)')
    body = replace_once(body, 'fFunction(0), fReader(0), fItInputArray(0)',
        'fFunction(0), fReader(0), fOrderedPileUp(false), fOrderedEntry(0), fOrderedCounts(0), fItInputArray(0)')
    body = replace_once(body, '  fReader = new DelphesPileUpReader(fileName);', '''  fReader = new DelphesPileUpReader(fileName);
  fOrderedPileUp = GetBool("OrderedPileUp", false);
  fOrderedEntry = 0;
  if(fOrderedPileUp)
  {
    fOrderedCounts = new std::ifstream(GetString("PileUpCountsFile", ""));
    if(!fOrderedCounts->is_open()) throw runtime_error("Cannot open ordered pileup count file");
  }''')
    body = replace_once(body, '  if(fReader) delete fReader;', '''  if(fOrderedPileUp)
  {
    if(fOrderedEntry != fReader->GetEntries()) throw runtime_error("Unused ordered pileup records");
    *fOrderedCounts >> std::ws;
    if(!fOrderedCounts->eof()) throw runtime_error("Unused ordered pileup event counts");
    delete fOrderedCounts;
    fOrderedCounts = 0;
  }
  if(fReader) delete fReader;''')
    body = replace_once(body, '  switch(fPileUpDistribution)', '''  if(fOrderedPileUp)
  {
    if(!(*fOrderedCounts >> numberOfEvents) || numberOfEvents < 0 || numberOfEvents > 100000)
      throw runtime_error("Missing or invalid ordered pileup count");
    if(fOrderedEntry + numberOfEvents > fReader->GetEntries())
      throw runtime_error("Ordered pileup bank exhausted");
  }
  else switch(fPileUpDistribution)''')
    body = replace_once(body, '''    do
    {
      entry = TMath::Nint(gRandom->Rndm() * allEntries);
    } while(entry >= allEntries);''', '''    if(fOrderedPileUp)
    {
      entry = fOrderedEntry++;
    }
    else
    {
      do
      {
        entry = TMath::Nint(gRandom->Rndm() * allEntries);
      } while(entry >= allEntries);
    }''')
    shutil.copytree(a.source, a.output, symlinks=True)
    changed = {paths[0]: header, paths[1]: body}
    identities = {}
    for name, content in changed.items():
        (a.output / name).write_text(content)
        identities[name] = {
            'base_sha256': hashlib.sha256(original[name].encode()).hexdigest(),
            'patched_sha256': hashlib.sha256(content.encode()).hexdigest()}
        if (a.source / name).read_text() != original[name]:
            raise ValueError('Original runtime unexpectedly changed')
    (a.output / 'ORDERED-PILEUP-PROVENANCE.json').write_text(json.dumps({
        'status': 'source-prepared-not-yet-built', 'source': str(a.source.resolve()),
        'files': identities, 'semantics': 'Optional ordered particle record consumption with explicit per-hard-event N; all native detector, rotation and vertex response retained.',
        'default_mode': 'Original independent random entry selection retained verbatim in else branch',
    }, indent=2) + '\n')
    print('Prepared isolated ordered-pileup runtime:', a.output)


if __name__ == '__main__':
    main()
