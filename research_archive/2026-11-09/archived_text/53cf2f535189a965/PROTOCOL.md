# Bench real RAW local repair pilot V1

Status: frozen before target construction and new geometry scoring, 2026-10-24.

## Scope and assets

The user authorized actual execution and subsequent bounded training **if the
data feasibility result warrants it**. This phase uses exactly the 12 saved
Coverage Native RAW requests in `data/manifest.json`: a8, 27, 838c, 91e, fd355,
2d, each noise 0 and 1. They are deliberately selected development examples,
not an independent success-rate estimate. No request replacement, new noise,
generator call or D0 call is allowed in this data phase. Existing inputs,
weights, scoring implementations, reports and results remain read-only.

## Prior evidence and substantive change

RELDEF500 did not improve the SHAPE500 visual-positive identities and incurred
DEV FF costs. Its loss is not extended or tuned here. R05 bounded anchoring and
R27 bounded restoration already demonstrate that arbitrary local attraction
can create costs and lose connections. This pilot instead tests whether a
small, auditable set of **real RAW -> local coordinate repair labels** can be
made, with explicit KEEP / EDIT / UNKNOWN. It is neither a new globally applied
inference attraction algorithm nor a claim that a lower-FF output is ground
truth. Source complete geometry is not used to choose a target completion.

## Target contract

- One fixed finite repair strategy, one candidate target per request. Its
  constants and implementation hash are recorded in `targets/strategy.json`
  before executing proposals. There is no radius sweep, best-result selection
  or iterative search over alternative repair methods.
- Inspect existing RAW illegal CF pairs and the finite C surface to propose
  only small, geometrically supported wall/rim repairs. A local wall that barely
  crosses the known surface can be projected back to that finite surface; a
  missing support, arbitrary double layer or displaced whole body cannot be
  declared repaired this way. Apply the same eligibility rule to all 12 inputs;
  retain explicit reasons when no eligible patch is found.
- Coordinate displacement is bounded by 0.005 L for **every** affected corner,
  where L is the original normalized model-domain diagonal. Do not enlarge the
  bound on failure. If an existing GEO corner cluster is used, enumerate and
  check all of its original free corners; do not silently substitute a radius
  neighborhood for that cluster or move only one member while claiming a
  coordinated edit.
- C, N, generated slot IDs and original per-slot corner order stay exact.
  No deletion, new faces, re-triangulation or collapse into padding slots.
  Target shape is float32 N x 3 x 3. KEEP coordinates equal RAW byte-for-byte.
  UNKNOWN coordinates are merely RAW placeholders and have zero loss mask.
  C is excluded from free-coordinate supervision and restored by contract.
- A proposed patch is accepted only if the original finite-triangle detector
  removes a designated illegal relationship and creates no new illegal CF/FF
  pair; all affected faces and relations with the full mesh are checked. Reject
  degeneration, unacceptable normal/shape change, loss of existing legitimate
  local connections or compression of trusted thin support. Record the failed
  proposal and rejection reason; never quietly replace it with a successful
  different strategy.
- KEEP must have a specific structural basis. Absence of a detected collision
  alone does not certify an entire generated object as correct. DELETE_NEEDED,
  REBUILD_NEEDED and UNKNOWN remain separate action annotations, not hidden
  coordinate training targets.

## Original postprocessing and evaluation

Each final local target (including identity placeholders on failed requests)
is treated as a candidate D0 **output**, then sent directly through the existing
GEO_C, source orientation and HINGE_OPP015 -> ALPHA -> HOLE cascade. D0 is not
called again. Exactly 12 target geometry jobs and 24 original BASE/FINAL score
rows are planned. Existing RAW, official, SHAPE, RELDEF and GEO_ONLY evidence
is reused. A raw finite-pair scan for target validation is additional CPU
diagnostic work and is counted separately from full metric evaluations.

The unchanged 177-field evaluation, frozen queries and full per-output values
are retained. Track edited original slots through GEO using the saved slot and
corner mappings. Report deleted slots and original areas separately; disappearance
of a face is not an acceptable explanation for a coordinate repair. Verify
KEEP/C in target coordinates and their survival/geometry again after GEO and
FINAL. Use identical camera bounds and saved section positions for actual
AI_PROXY image review; inspect rejected examples too.

## Decision before any training

The attachment's “majority of examples yield credible KEEP+EDIT pairs” is
operationalized as at least **7 of these 12** requests with a nonempty, validated
EDIT that remains effective after the original processing chain, with no
material loss of the protected structure. KEEP-only or unchanged UNKNOWN
requests do not count. Parent identities and patch/action counts must be shown,
not just corner totals. This threshold is a data-readiness gate for this pilot,
not a statistical claim of general efficacy.

If this gate fails, do not train on untrusted placeholders or relax acceptance
after seeing results. Deliver the actual partial labels, failed proposals and
evidence. Such a failure applies to this bounded construction method; it does
not establish an upper bound for every local repair method or for D0.

If it passes, the user has authorized continuing: first lock a separate training
manifest of real RAW repair pairs from permitted training parents, protect these
12 development parents from training, and register a finite training/evaluation
budget using the original model and coordinate loss. Keep an identity auxiliary
task. Resource-measure the already authorized task before dispatch and preserve
the original deterministic runtime. Do not automatically reuse these 12 as
training while calling them backend-unseen. No broader training or modified
loss follows merely from passing this data gate.

## Execution and recovery

CPU stages use one thread per independent worker. Initial measurements found
24 CPU cores, approximately 61 GiB available RAM and no GPU workload from this
task. Up to three independent original geometry/scoring jobs may run concurrently
after a fresh resource measurement; each owns separate outputs. No GPU is needed
for this phase. Completed stages are reused only after hash verification.
Incomplete attempts are retained and reported; no silent overwrite or replay.
The final report includes actual operation counts, timing, resource snapshots,
source hashes, code.diff, original-asset integrity and a recovery entry point.

No model-default change, cleanup, commit or push is authorized by this pilot.
