# R2_BENCH_EARLY_FM_V1 — frozen 1500-update test

User authorized continuation of the next test after the actual-trajectory crossover.
Question: can emphasizing early-time conditional FM improve real RAW C adoption,
while preserving familiar positives and postprocessed body quality?

## Single variable and closest prior evidence
Reuse USER v2's exact COMMON12000/A step0 generator, fresh AdamW, RNG, 25-parent /
81-condition data, 1500-update schedule, batch8/micro1, xyz augmentation, restricted
OT followed by HYBRID .25, original SEAM masks, BF16 forward and FP32 integration.
Only multiply each sample's original coordinate-normalized FM contribution by
w(t)=1.5 for t<.5 and .5 otherwise. Keep the original whole-batch free-coordinate
denominator; do NOT normalize away the weight per microbatch. Original t is
sigmoid(N(0,1)), NOT uniform. Its symmetry implies population E[w]=1; the actual
frozen stream's counts and scalar-weighted mean are reported, not assumed exact.
This is an experimental allocation of supervision, not a causal conclusion from
the crossover and not a new loss/architecture/OT/solver. Late supervision remains.

Closest evidence: USER v2 ordinary1500 (direct same-stream comparator), the
actual-trajectory crossover (USER prefix retained familiar gains, DEV zero),
R02 injection/time settings (no stable gain), R07/R19 auxiliary-loss schedules
(different objectives), and Bench asynchronous SEAM completion (changed trajectory
and corner clocks, distinct from weighting original FM). None establishes that
this weighting will work. No sweep or second candidate arm is authorized here.

## Fixed computation and evaluation
One new arm, 1500 optimizer updates, 12000 forward/backward microbatches. Endpoint
1500 only, no best-checkpoint selection. Safe full-cursor checkpoints every250;
no silent replay after a partial update, numerical error or OOM. No extra neural
probes. Log original SSE/counts and weighted objective separately, time-bin errors,
gradient clipping, exposures, allocator peaks, process resources and all attempts.

Same crossover16 requests: four CURRENT_TRAIN and four DEV parents, two old frozen
noises each. Same C/N/production layouts, source-free inference, fixed queries and
positions of front/side sections. New800 Euler50 calls and16 single official Bench
D0 calls; no occupancy/head calls, no new noise and no source/oracle layouts.
Save all RAW before D0. Same C/seam restore, GEO_C, source winding, HINGE_OPP015,
ALPHA/HOLE final CPU chain. Preserve RAW, BASE_POST, CLEAN_FINAL_POST and all original
scorer fields. Reuse COMMON and USER old results read-only; new scoring48 outputs.
This is a repeatedly observed small development panel, not a generalization test
on fresh parents/noises. No unique-source reconstruction ranking.

## Decision and limits
Review actual identical-view/section images for all16 RAW and finalPOST. Record
AI_PROXY YES/PARTIAL/NO/UNCERTAIN separately from exact topology. Primary evidence
is paired adoption, particularly fixed failures and DEV; keep prior good support /
assembly outputs, body completeness and local surface damage visible. Report CF,
FF affected-face fraction AND illegal-pair counts, I with support status, small
angles, actual free count/area, deletions and legal bulk reach. An improvement in
training MSE/I alone does not pass. Do not trade away body surfaces to lower FF.
If no repeatable paired adoption signal, do not extend this weighting recipe.

## Resource and preservation policy
Resource audit before each GPU wave. One authorized training arm means one training
process; use measured actual backward/optimizer peak, not inference allocation.
Leave at least max(2GiB,10%VRAM) extra free headroom. Independent CPU scoring/render
jobs may overlap and are thread-capped. Incremental sampling workers only if actual
resources and throughput warrant. Never stop user processes or change precision /
batch to add workers. Keep all old data/weights/results and scoring definitions.
After report and recoverable artifacts stop; no default change, cleanup, commit/push.
