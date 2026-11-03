# Constituent-link backfill

The user authorized the full constituent-association backfill on 2026-11-02,
with up to 1,000 single-core CPU Dask workers and unattended monitoring.
It adds ordinary ROOT TTrees containing native contributor references to the
existing immutable detector campaign. It does not generate replacement hard
events, rerun CMS FullSim, change cards or replace compact detector data.

Site configuration and current evidence are external to Git at
`purdue-setup/results/constituent-link-backfill-20261102/`.
Inspect `campaign.json`, `controller-status.json`, `watchdog-status.json` and
`last-agent-trigger.json`. Dispatch and local parity are not durable completion.

On2026-11-03 the expired CMS child proxy was successfully renewed for48h from
the still-valid standard temporary parent proxy. The earlier claim that renewal
required the encrypted long-term key was incorrect. All five pilot archives
then passed two complete EOS readbacks and bulk execution started. Six NERSC
sources remain outside the Purdue input snapshot. Check live receipts for the
current completed scope; this paragraph is not a full-campaign READY marker.

## Recovery and integrity

`tools/constituent_backfill_worker.py` downloads the original input over HTTPS
with trusted grid CAs and frozen size/SHA256 checks, supporting both source tar
bundles and legacy individual objects. It reconstructs the original stable-only
HepMC input, cards and seed wrappers, preserving the saved chunk boundaries,
IDs, raw weights and seed `420000 + task*1000 + chunk`. No cuts or cross-section
normalization are introduced. The expanded generator backfill remains separate;
feeding that different graph to Delphes would change the replay input.

Simulation selects the archived executable/library hashes. The campaign includes
both the original build and the existing `bounds-v1` build from the documented
October19 SimpleCalorimeter correction. Legacy validations without binary hashes
use the retained original build; that is a provenance limitation. The pinned
stock response cards are unchanged. Unknown recorded hashes fail explicitly.
The common original Delphes class dictionary reads both builds' unchanged ROOT
object schema; the simulation binary selection is recorded independently.

Each view passes the existing recovery tool's checks: all18 compact datasets
bitwise equal, every stable particle checked against the archived input and
every native contributor resolved to that event's Particle array. The separate
`tools/audit_delphes_constituent_links.py` rereads ordinary `events`/`links`
TTrees through uproot, checking event IDs/weight bits, complete constituent row
coverage, event/jet membership and list/reference bounds. Multi-contributor
towers retain lists; neither energy fractions nor a unique tower species are
invented. Preserved barcodes allow later joins to the generator backfill.

Groups of five chunks hold one ROOT file per detector/chunk plus provenance in
a transport tar. The ROOT files are read with uproot after bounded staging and
unpacking. Only new sidecars/proofs are published under the user's EOS prefix
`heptokens-constituent-links-20261102`. Content-addressed URLs avoid overwrites.
Every group requires upload plus two full stable SHA256/Adler32/size readbacks
before its durable receipt is committed and temporary native ROOTs are removed.
Sources become complete only when all expected group receipts reconcile.

## Execution and monitoring

The resource plugin schedules its scheduler RPC after worker startup rather
than blocking the startup hook. A scheduler plugin also wakes source tasks that
were submitted before their required worker resources appeared. A regression
test creates an initially empty local cluster, submits constrained work, and
grows to one then two workers. This fixes the scale-out startup race and pending
`no-worker` tasks without changing recovery physics or integrity requirements.
The live frozen campaign receives the same operational repair through
`operational_controller.py`, preserving its pinned scientific file hashes.

`tools/run_constituent_backfill.py` provides `pilot`, `controller` and `watch`
modes with `--config PATH`. The Dask scheduler/workers use the maintained global
environment. Each worker launches the frozen ROOT project interpreter in a
separate source process, avoiding the project environment's Bokeh/NumPy startup
incompatibility. A source lease prevents duplicate writers. Canonical node-disk
reservations bound shared scratch; transfer slots and a2TB remote cap bound I/O.

The controller resumes from independently checked group/source receipts. Worker
loss and transport errors retry with backoff; scientific mismatches quarantine
the affected source. Repeated failures pause admission for review. The watchdog
checks every30s, restarts dead/stale controllers, and uses `codex queue` to wake
the same task every30min or on actionable failures. Its20h window covers the
requested16h absence. This survives terminal disconnect while the AF pod and
Codex daemon remain alive; it is not a hosted cloud automation.

At launch the CMS proxy had expired. The controller automatically waits for
renewal at the existing worker-visible path. It requires at least18h remaining
proxy and VOMS attribute lifetime before its initial EOS publication checks,
then requests1,000 workers only after all pilot payloads pass those checks.
Admission later pauses below1h. Actual available workers and I/O throughput
determine completion time; the deadline is not guaranteed by a scale request.

## Measured pilot and current limits

The final frozen `runtime-v6` pilot uses the first1,000 events in each of
sources4,84,1373,2785,4151, covering all four pThat bins and both input layouts.
All ten views passed:5,000 stored source-event entries,1,252,486 stored constituent
rows and1,564,117 contributor edges. Counts include repeated objects across jet
collections and detector views. Eight retained-versus-replayed comparisons for
the four previously audited sources passed every serialized field and unordered
contributor-set comparison. This is a representative first-chunk test, not a
full-source/group publication test or campaign-wide determinism guarantee.
Evidence: `pilot-summary.json`, `independent-pilot-comparison.json` and
`PILOT-LOCAL-PASSED.json`. Twenty-eight targeted tests passed (`final-tests.txt`).

The current Purdue input snapshot covers4,540 sources/210,266,000 events.
Six separately NERSC-owned sources (0,1,2,3,5,6) are absent. They account for
276,000 source-event entries relative to the210,542,000 campaign total.
Noninteractive NERSC SSH remains denied even with the documented username.
`AVAILABLE-SOURCES-COMPLETE.json` can describe the accessible subset; full
`READY.json` is forbidden while sources remain missing. The user's NERSC report
establishes replay readiness there, not access to those files from this AF pod.

The generator backfill and subsequent flavor/HT label preprocessing are separate
steps. Completing this sidecar run does not establish completed full label
production, detector closure or ML validation.
