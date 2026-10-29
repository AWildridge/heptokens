# Generator backfill (2026-10-29)

The backfill joins existing detector events to their original CMS generator
information. It does not run Delphes, generate replacement events, assign jet
flavour, or alter the frozen replay. The source identity is the frozen manifest
task/dataset plus run, luminosity block, event, and original entry ordinal.

## Content and limits

Use already accessible MiniAODSIM inputs; do not wait for an AOD restoration.
MiniAOD is pruned and quantized. All generator information **retained in that
tier** is preserved; the original pre-pruning generator history cannot be
reconstructed from MiniAOD. The campaign is 4,546 frozen original files and
210,542,000 catalogued events in four pThat bins, 300--1000 GeV. There are no
particle selections and no cross-section normalization in this export.

Each `generator.root` contains native CMS Events/Runs/LuminosityBlocks,
provenance and generator products, including packed/pruned particles, generator
weights, available GEN/LHE metadata, generator jets/flavour associations, GEN
MET, pileup summaries and retained generator event vertex/time products. Native
products are the archival authority, including fields not duplicated in the
plain view. A branch inventory check rejects dropped generator products.

`gen/events` is an ordinary TTree readable with uproot, without CMSSW. Every
retained packed and pruned particle is included, in original collection order.
Packed particles store double-precision getter values for pt, eta, phi, mass;
pruned particles store px, py, pz, energy and mass. Derived coordinates should
be calculated in the reader. Momentum is in GeV and spatial coordinates in mm
(native cm multiplied by ten). Per-particle time is not invented. A retained
event-origin time product, if present, stays in the native products and must not
be confused with a per-particle production time.

Both collections include pdgId, status, threeCharge, all 15 status-flag bits,
longLived, massConstraint and spatial vertices; pruned particles also include
collisionId. GenEventInfo convenience fields include nominal/all weights,
binning values, DJR values, qScale, couplings, process ID, matrix-element parton
counts and PDF information with an explicit availability flag.

Relations are ragged adjacency arrays: `motherOffset`, `motherIndex`,
`motherState`, `motherProcess`, `motherProduct`, with analogous daughter fields
for pruned particles and lastPruned fields for packed particles. Offsets have
particle_count+1 entries. State 0 means null, 1 unavailable, 2 available. Nonnull
references must resolve to the retained pruned collection; unavailable or
misdirected links fail the export. Index -1 accompanies a null reference.
No missing edge, time, or weight is silently replaced with physical zero.

## Execution and validation

`tools/skim_generator_products.py` runs a generator-only PoolOutputModule in
CMSSW_10_6_30/slc7_amd64_gcc700. `export_generator_view.cc`, compiled once in
that runtime, appends the plain TTree. `run_generator_backfill.py` dispatches
CPU tasks through the existing Purdue Dask Gateway cluster. ROOT and GFAL run
in separate subprocess environments. Inputs are downloaded into worker-local
scratch over XRootD and checked against frozen size/Adler32; SHA256 is recorded.
The validated streaming mode instead reads selected generator baskets directly
over XRootD. Before reading it checks the authoritative server's reported size
and Adler32 against the frozen catalog, and records ROOT UUID, event count and
generator branch inventory. This is a server-reported source checksum, **not a
fresh full-file source SHA256**; the receipt explicitly records that distinction.
Every generated output still receives fresh complete hash/readback checks.
Workers do not rely on an EOS mount or on the session's `/tmp` or `/home`.

Every output is checked for native/view/source event-count agreement, event ID
uniqueness, entry order, finite particle coordinates, graph offsets and valid
link indices. Every plain-view basket is read. Representative native FWLite
reopening and original-event comparisons are required before bulk admission.
These checks establish data preservation within their inspected scope, not
physics closure or validity of a future flavour algorithm.

Outputs are immutable in a separate EOS campaign. Canonical GFAL helpers check
Adler32 and perform two stable complete SHA256 readbacks before generated local
payload cleanup. WORK holds manifests, hashes, logs, reservations and receipts.
Only per-source `complete.json` receipts with both readback proofs are ready for
preprocessing. Failed attempts retain their unverified generated output. A
remote byte-reservation ledger enforces the campaign cap before uploading;
shared I/O slots bound downloads/uploads/readbacks. No original source or
frozen production payload is removed.

The active site configuration, immutable software hashes, per-source receipts,
measurements and status live under
`../purdue-setup/results/gen-backfill-20261029/` relative to the Git repository.
Prefix-based size and throughput projections are estimates; full-source and
concurrent measurements are needed to judge the five-hour target. Dispatch is
not evidence that all 210,542,000 events have completed.
