# R2_BENCH_FOUR_ROUTES_V1

User-authorized 2026-10-27: execute four bounded routes, independently, then stop.
The user accepted the scope through the asynchronous budget question. No cleanup,
default replacement, commit, push, additional arms or automatic training extension.

| Route | Fixed intervention | Budget |
| --- | --- | --- |
| 1.1 HYBRID | Frozen M RAW/D0 local binary selection, original geometry backend | 46 existing requests, CPU only; zero model calls |
| 2.1 EXPOSURE | Continue M with matching live AdamW/RNG and same sampling/objective policy | 3,000 new updates, batch8/micro1; 46 Euler50 RAW and 46 single D0 |
| 2.2 BESTOF4 | Frozen narrow B, four preregistered noises per C, fixed geometric selector | 23 conditions, reuse two existing candidates and generate two new each; 46 new Euler50 RAW; at most23 new single D0 |
| 3.1 SEAM | Continue narrow B with original source-owner SEAM training and C-only production layout | 1,500 updates, batch8/micro1; frozen inherited head; 46 Euler50 RAW and46 single D0 |

Each route receives an independent immutable registration and execution ledger.
All original data, scores, queries, RAW/POST, weights and metadata are read-only.
The same 46-request panel remains fixed; OLD/FOCUS/DEV/HOLDOUT/HIGH/NEW_WIDE groups
are not pooled to hide regressions. NEW_WIDE parents are held out of narrow B
and its SEAM continuation, but are trained by M and its exposure continuation.
No unique source-style reconstruction success criterion.

Route1.1 uses seed faces with valid RAW area >1e-12 L² and positive edge lengths,
and D0/RAW area<.35 OR minimum-edge ratio<.30. Expand exactly one ring through
the original GEO(RAW) corner clusters, not recursively. Selected whole free
faces take RAW coordinates; others take restored-C D0 coordinates. Preserve C.
No oracle semantic labels or scoring-driven rollback. Mixed boundary damage is
measured explicitly. Same GEO_C, source winding, HINGE_OPP015, ALPHA, HOLE afterward.

Route2.1 is an additional-compute test, not an isolated proof that more independent
parents generalize. HIGH six parents receive500 additional instances each,
reaching750 cumulative; OLD/FOCUS exposures also increase. Reuse M's count-only
group-MSE rule, without interpreting it as equal gradient contributions.
Fresh registered noise/time streams, preserved parent-exposure policy and source
optimizer. Fixed final checkpoint; no best selection.

Route2.2 tests a selection policy, not a better generator. Preserve every candidate
and the selection components including NA. Geometric selection is not certified
semantic adoption. Report cost versus one-draw references and actual image review.
No hard-noise training, reward updates or source-Y-dependent selection.

Route3.1 introduces no shared-noise relation, couple loss, new head, or mid-trajectory
handover. Original SEAM is adapted to the already audited Native N128..799 contract;
N800 remains forbidden. Source topology may be used for training supervision only.
Closed C has no required open interface and receives NA for that requirement.
Any incompatible target is reported without silently changing or dropping it.

Strict deterministic runtime: CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA, strict
deterministic algorithms, math SDPA, TF32 off, BF16 forward/FP32 parameters and
integration. Read actual resources before each GPU dispatch. Incremental workers,
at least max(2GiB,10%total) extra headroom; no simultaneous loading spikes.
CPU geometry may overlap under bounded threads. No silent OOM/scientific retries;
exact safe-state continuation only, without replaying counted updates.

Preserve full original metric fields, denominators, failures and NA. RAW/BASE/FINAL
remain separate. Actual multi-view and fixed-position section review is AI_PROXY,
not an independent human assessment. No labels inferred from scalar metrics.
