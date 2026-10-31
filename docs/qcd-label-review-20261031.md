# Corrections to the label pilot interpretation (2026-10-31)

The 30 October pilot's measurements and immutable ROOT outputs remain unchanged.
Its discussion overstated implementation choices as missing information. The
user confirmed the NERSC transfer is complete, requested light/strange/gluon
labels, and accepted the retained-generator/time/spin limits for this Delphes
study. This note corrects the production plan; it does not claim the additional
labels have already been produced or validated.

## 1. Input locations

The user confirms the completed input transfer and completion records are on
NERSC. The 4,540-source inventory describes the Purdue-owned portion, not an
incomplete full dataset. Six sources were already owned at Perlmutter. Checking
for those sources at the expected Purdue URLs did not test their NERSC presence.
The remaining operational task is to read the completed NERSC inventory and
receipts, then connect all 4,546 source identities to the preprocessing input
catalog. No replacement event generation or new Rucio transfer follows from
this inventory-location mismatch. The present session has not independently
re-read the NERSC receipts; the prior SSH attempt stopped at host verification.

## 2. Reconstructed constituent truth links

This question asks: **which generated stable particle(s) produced this detector
constituent?** Delphes stores that relation in its native ROOT output:

- `Track.Particle`: a ROOT `TRef` to the generated particle;
- `Tower.Particles`: a ROOT `TRefArray` of contributing generated particles;
- `Jet.Constituents`: references to reconstructed constituents;
- `Jet.Particles`: generated-particle references collected through candidates.

The pinned upstream `DelphesClasses.h` declares these members, and
`TreeWriter::ProcessTracks`, `ProcessTowers` and `ProcessJets` fill them. A tower
can have several contributors; even native truth does not give it a single
unique particle identity. These are Delphes truth-contributor relations, not
CMS FullSim hit-to-track associations or a complete detector energy attribution.

Our `tools/compact_delphes.py:19` retains event/jet/constituent offsets, P4,
constituent class and charge, but does not serialize those generated-particle
references. It preserves jet-to-reco-constituent membership. Consequently:

- Native Delphes ROOT, if retained, is the exact source for recovering the links.
- The generator backfill restores particle identities and ancestry, but does not
  say which particles the already-run detector reconstruction selected/merged.
- Angular matching can supply an approximate association, with charge/P4 tests,
  multiplicity/ambiguity diagnostics and explicit provenance. It cannot restore
  discarded TRefs, particularly for merged calorimeter constituents.

This does not justify claiming that all jet-flavor labels are unavailable.
Geometric reco-to-truth jet matching is already used by the producer. The
reason for using that matching is to associate two different clustered jet
collections, which can merge/split/respond differently; it is a different
relation from native constituent contributor bookkeeping.

For future compaction, preserve event-local generated-particle indices behind
these TRefs and list-valued tower contributors before discarding native ROOT.
This is an output bookkeeping change, not a detector-response retuning. Recovering
old exact links still requires the old ROOT or replay of detector bookkeeping;
we have not established another lossless recovery from the compact files.

## 3. Jet flavor, including strange and gluon labels

This question asks: **which generated hadron/parton flavor is associated with a
jet?** It does not require the per-reconstructed-constituent links in item 2.
We can cluster common truth jets from saved stable particles, associate retained
hadrons/partons to them, then use the existing reco-to-truth jet match to attach
the resulting truth targets to CMS and ATLAS jets.

The inspected CMSSW_10_6_25 sources provide a concrete baseline:

- `Pythia8PartonSelector.cc` selects quarks 1..5 and gluon 21, skipping statuses
  1/2, with no parton daughters; when applied here this is a retained-graph
  prescription, not proof of equivalence to running on unpruned GEN.
- `JetFlavourClustering.cc:20` documents b/c hadron tagging and parton ghost
  association. For light partons it uses the hardest associated candidate to
  distinguish PDG 1, 2, 3 and 21. It also specifies hadron/parton priority rules.
- The inspected MiniAOD pruning configuration explicitly retains light quarks
  and gluons for parton-based jet flavor (`prunedGenParticles_cfi.py:25`).

A bounded check of the first 1,000 generator entries in each of the four pilot
sources finds candidates for every requested quark/gluon species, including
7,790 strange partons and 107,616 gluons before jet association or cuts. These
are retained-parton candidate counts, not jet counts, purity or tagging results.
Evidence: `purdue-setup/results/qcd-label-review-20261031/retained-parton-candidates.json`.

The next version should preserve independent hadron and parton information:

- heavy-hadron b/c associations and charm-from-b metadata;
- signed associated parton PDG IDs, indices, P4/ranking and chosen candidate;
- separate u/d, strange and gluon classes, with a combined u/d "light" category;
- unmatched and association/conflict diagnostics, rather than assigning every
  jet without a b/c association to a generic light category.

Use the inspected CMS prescription as an explicit baseline and test it in the
pilot. No request for the user to invent a flavor convention is necessary.
The association choice still matters: a gluon can split into strange quarks,
and different hard-process versus shower-parton prescriptions can then label
the same jet differently. A kaon in a jet alone is not an initiating-s label.
Preserve candidate information so alternative definitions can be compared.

Shower-node categories are another target: mapping a geometric C/A branching
to a generator branching is not determined by assigning one flavor to a jet.
C/A targets remain deferred, as already requested; this is not a reason to
hold light/strange/gluon jet tags.

## 4. Accepted simulation scope

The user accepts retained MiniAOD genealogy, missing production times/spin and
original pileup-particle limits for this Delphes study. Keep those facts in
provenance; they are not approval gates for producing the requested labels.

## 5. HT

The scalar sum is straightforward. `event_kinematics` already computes HT as the
sum of selected truth or reconstructed jet pT (`physics.py:365`). It is saved
for truth, CMS and ATLAS events. Existing jet cuts are pT>=30 GeV, |eta|<2.5.
That observable was not missing from the pilot.

The extra field called `generator_ht` was masked because no separate stage or
object collection had been attached to that name. That is a naming/definition
issue, not an inability to calculate hadronic activity:

- jet HT: scalar sum of the selected jets' pT, already available;
- stable-hadron HT: scalar sum of retained status-1 hadrons' pT, calculated once
  per stable hadron, with inherited MiniAOD coverage and no additional cuts;
- hard-parton/LHE HT: a different scalar sum over a selected parton stage.

The next pilot should add the literal stable-hadron sum under an explicit name,
while retaining the existing truth/reco jet HT and event pThat separately.
Do not sum every intermediate generator particle: counting a parent and its
daughters double-counts the shower/decay history. Also, a sum of final-hadron
pT is not identical to a jet-pT sum: vector clustering changes the sum and
photons from hadron decays are not final-state hadrons.

The user need not choose an exotic new observable. The producer should record
which object collection and cuts a scalar sum uses. Physical cross-section
normalization is an independent sample-weight accounting task, not an HT
ambiguity and not a prerequisite for raw per-event label computation.

## Inspectable sources

Pinned downloaded sources and this bounded check are saved under
`purdue-setup/results/qcd-label-review-20261031/`.

- [CMS Pythia8 parton selector](https://github.com/cms-sw/cmssw/blob/CMSSW_10_6_25/PhysicsTools/JetMCAlgos/src/Pythia8PartonSelector.cc)
- [CMS jet flavor clustering](https://github.com/cms-sw/cmssw/blob/CMSSW_10_6_25/PhysicsTools/JetMCAlgos/plugins/JetFlavourClustering.cc)
- [MiniAOD pruning](https://github.com/cms-sw/cmssw/blob/CMSSW_10_6_25/PhysicsTools/PatAlgos/python/slimming/prunedGenParticles_cfi.py)
- [Pinned Delphes output classes](https://github.com/delphes/delphes/blob/28658365abeb71ee36dfc739f9670c1514c0cb10/classes/DelphesClasses.h)
- [Pinned Delphes ROOT writer](https://github.com/delphes/delphes/blob/28658365abeb71ee36dfc739f9670c1514c0cb10/modules/TreeWriter.cc)
