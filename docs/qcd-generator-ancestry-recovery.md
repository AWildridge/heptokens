# Generator ancestry recovery without repeating detector simulation

Status: investigation and proposed preprocessing change, 2026-10-29. The ROOT
label producer still masks flavor targets. No bulk recovery, production flavor
assignment, detector rerun or response change was performed for this study.

## Where the information was lost

The earliest retained exporter implementation already selects
`packedGenParticles`, requires every particle to have status 1, and serializes
PDG/P4/mass/xyz with ct=0. The production implementation has the same selection:
[`export_cms_chunks.py`](../tools/export_cms_chunks.py). It neither reads
`prunedGenParticles` nor serializes mother/daughter references or status flags.
It groups particles by xyz into synthetic HepMC vertices and writes zero decay
end-vertex references. This is a stable-particle detector input, not an archival
generator graph. The earlier exploratory
[`export_cms_miniaod_hepmc.py`](../tools/export_cms_miniaod_hepmc.py) retrieves
pruned particles but saves only their count.

[`replay_cms_collection.py`](../tools/replay_cms_collection.py) gzip-compresses
that existing HepMC payload; this step does not remove ancestry. Its compressed
reference removes the duplicate particle-value array because the same values
are retained in HepMC, but neither representation has the ancestry fields.

There is a second preservation limit:
[`compact_delphes.py`](../tools/compact_delphes.py) saves reconstructed
jet/constituent P4, offsets, object class and charge, omitting generator particles,
truth TRefs, flavor/tag fields and other detector observables. Native ROOTs are
then eligible for cleanup after the recorded validation/durable commits.
Those validation checks establish the retained representation's integrity,
not preservation of the original generator history. The design error was
using the minimal detector-input representation as the reusable truth archive.

## What the original inputs retain: measured checks

Evidence is in the separate Purdue workspace directory
`purdue-setup/results/gen-ancestry-recovery-20261029/`:

- `findings.json`, five native audit payloads and their command/log records;
- `ghost-feasibility.json` and the standalone diagnostic script;
- `source-code/` and `primary-source-identities.json`, containing pinned CMS
  definitions and hashes. Original source-84 PAT pruning selections are also
  recorded in the prior `cms-chain-pilot-20261022/PAT-actual-pset.txt`.

Method: CMSSW_10_6_30 FWLite reads the existing cached original MiniAOD files,
using the first 10 frozen replay reference IDs from each of sources 84, 1373,
2785 and 4151 (40 events, four pThat bins). It also reads the ten exact source-84
IDs in the already restored original AOD parent. IDs and generator weights must
match. Counts below are unweighted and use no particle acceptance cut; they
are not cross-section estimates or campaign-wide completeness measurements.

| Observed quantity | Result |
|---|---:|
| MiniAOD events audited | 40 |
| Retained packed stable particles | 18,770 |
| Retained pruned graph nodes | 3,488 |
| Pruned mother/daughter reference entries checked | 21,136 |
| Unavailable or mismatched checked reference targets | 0 |
| Terminal b hadrons | 8 |
| Terminal c hadrons | 52 |
| Terminal c hadrons with a b ancestor | 11 |
| Events reproducing saved stable particles in exact order | 40/40 |

All packed mother and last-pruned references in these 40 events resolve to the
saved pruned collection. This observation is sample-specific; valid null or
unavailable references must still have explicit representations in a backfill.
Terminal heavy hadrons follow the inspected CMSSW_10_6_25
`HadronAndPartonSelector` and `CandMCTagUtils` definitions: select a hadron
containing b/c with no daughter containing the same heavy flavor.

In the ten matched source-84 events, AOD has **16,967** `genParticles`, while
MiniAOD has **941** pruned nodes and **5,176** packed stable particles. The
packed/pruned collections can overlap in represented particles; their counts
must not be interpreted as a deduplicated full graph. All six terminal b and
16 terminal c hadrons agree exactly between AOD and MiniAOD in PDG ID, status,
P4, mass, xyz and the 15 saved status flags. Charm-from-b classifications also
agree. Time was not compared: the audit's ct=0 is an explicit placeholder.

This demonstrates recoverable flavor information in these original files.
It also demonstrates that MiniAOD is not the full original generator graph.
The standard MiniAOD contract deliberately retains a subset of intermediate
particles, plus packed stable particles and links into the pruned collection.
See [CMS MiniAOD truth documentation](https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookMiniAOD2016#MC_Truth).
The actual saved PAT configuration and native checks take precedence over
generic documentation for this campaign.

## Proposed recovery and archival contract

Perform a generator-only pass over the exact original input events. This needs
I/O and truth processing, with no Delphes invocation.

1. **Recover and store all available generator information.** At minimum,
   export every packed and pruned particle and every retained reference from
   the original MiniAOD. To meet the stronger unpruned-record goal, resolve
   and read the original AOD/GEN parents and export their `genParticles` or
   native generator record where available. The original cached AOD already
   demonstrates this route on ten matched events. Parent accessibility, file
   coverage and retained product inventories still need auditing at scale.
   Label the scope explicitly (`miniaod_retained`, `aod_genparticles`, or a
   verified native GEN scope). An AOD `genParticles` export must not silently
   claim to reproduce every internal Pythia/HepMC field.

2. **Store the graph with the reconstructed events in new ROOT outputs.**
   Proposed additional TTrees in each versioned per-source `labels.root`:
   `gen/events` for event/source identities, weight/scales/PDF/LHE metadata
   when present; `gen/particles` for collection/index, PDG, original status,
   status flags, P4/mass/charge and vertex fields; `gen/relations` for all
   saved mother/daughter/ancestor/reference links; and
   `gen/replay_particle_map` for the stable HepMC barcode-to-original packed
   index mapping. Preserve ProductID/key, reference availability and relation
   role. `motherRef()` and `lastPrunedRef()` differ: the latter can identify
   a particle's own pruned representation, so it is not always a decay edge.
   Record pruned ancestor links as such, without inventing deleted vertices.
   A stable-to-full-AOD identity mapping requires an independently validated
   bridge; mask ambiguous mappings rather than using nearest kinematics.

   Archive all recovered graph nodes, independently of jet selection. Give
   unavailable times/metadata explicit validity masks. Saving a packed value
   in float64 does not restore its upstream precision. One shared generator
   graph should serve both detector views. Include source/catalog checksums,
   processing history, extraction software, units, graph scope and a separate
   graph hash in the manifest. Verify durable EOS readbacks before releasing
   bounded staging. Keep the frozen replay payloads and receipts unchanged.

3. **Preserve current event and jet identities.** Match source/sample plus
   run/lumi/event and nominal weight, then reproduce the saved ordered stable
   payload and its fingerprint. The 40-event audit already does the ordered
   comparison and derives its packed-index mapping. Preserve the original
   packed-particle clustering inputs, IDs, split membership, and reconstructed
   arrays; store enriched truth hashes separately. Replacing those particles
   with higher-precision/wider-acceptance AOD stable particles would define a
   different truth target and requires a separate versioned study.

4. **Assign heavy-hadron truth labels in preprocessing.** Select terminal b/c
   hadrons using a pinned CMS-compatible prescription; ghost-associate their
   directions by reclustering the existing truth particles with negligible
   heavy-hadron momenta. Associate results back to existing truth jets through
   exact physical constituent membership, preserving original P4 and IDs.
   Store lists and counts of associated b and c hadrons, with b precedence
   for the categorical 5/4/0 label. Record charm from b separately; a
   prompt-charm count excludes b-descended charm, while the inclusive
   hadron-based charm label need not. A missing/incomplete generator graph
   means unknown, not light. Parton flavor requires its own versioned selector
   and association; do not replace the hadron definition with a leading quark.

5. **Attach targets using the existing truth-to-reco joins.** Propagate the
   common truth jet's flavor to its matched CMS/ATLAS reconstructed jet, retaining
   unmatched and ambiguous flags. This is the flavor of the matched common
   R=0.5 truth jet: CMS reconstructed jets have R=0.5 and ATLAS R=0.6. If native
   R=0.6 truth flavor is needed, construct an additional explicitly defined
   R=0.6 truth association during preprocessing, rather than relabeling the
   R=0.5 target. Original FullSim CMS AK4 jet flavor fields can help validate
   definitions but must not be copied onto different Delphes jets by rank.

## Truth-only feasibility result and remaining checks

An isolated FastJet 3.5.2.0 diagnostic used these same 40 events, the existing
common anti-kt R=0.5 definition, particle pT>0/|eta|<5 with neutrinos excluded
and muons included, and jet pT>=30 GeV/|eta|<2.5. It tested ghost rescalings
1e-18 (the inspected CMS default), 1e-20 and 1e-22. All 126 selected truth jets
retain exactly the same physical constituent memberships. Maximum observed
absolute jet-P4 change was 0 GeV, within the predeclared 1e-12 relative/absolute
test tolerances. Ghost associations agree across all three rescalings. The
diagnostic finds five jets with a retained b ghost and ten with a c ghost and
no b ghost; the remaining 111 have neither. These are small-sample diagnostic
counts, not published production labels or evidence of campaign completeness.

Before scaling, implement and validate the versioned ROOT graph exporter and
label joins; test duplicate/self/cross-collection/null references, cycles and
unavailable products; compare AOD/MiniAOD heavy-hadron retention on a broader
representative sample; audit exact event/weight/stable-payload and detector-array
conservation; measure ROOT output size, I/O and CPU cost; and validate durable
EOS archival with bounded input/output admission. No total campaign cost or
complete ancestry closure follows from this prefix check.

## Flavor labels and detector tagging are separate

The pinned Delphes HepMC reader sends only non-status-1 quarks/gluons/taus to
`Delphes/partons`. Our exported status-1-only input therefore supplies no such
partons. The pinned CMS card's `JetFlavorAssociation` reads that array, and
`BTagging` chooses its response formula from `jet->Flavor`. Consequently the
old native flavor/tag output cannot be treated as a valid heavy-flavor truth
reference; this conclusion follows from the inspected input/reader/card/module
code, rather than a campaign-wide audit of saved tag bits. The compact data
also omit those native flavor/tag fields.

A generator backfill restores **truth labels** without detector resimulation.
It does not restore discarded detector tag scores, track impact parameters or
original stochastic tag decisions. If tag-response emulation is needed, define
a separately validated postprocessing step on fixed saved jets with explicit
new RNG provenance. The existing reconstructed P4/features can remain the
input to that study.
