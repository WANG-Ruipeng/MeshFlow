# CHAIR_MULTIPART_SEAM_V1 — paired numerical closeout draft

Status: **PENDING complete frozen evaluation caches**. This document contains a reporting contract and empty table structure, not experimental conclusions. No model, scorer or geometry detector is called by this preparation.

## 1. What is being compared

The new `MULTIPART_S1500` endpoint starts from the retained live `PLAIN3000_step1000` state at generator global step 14000 and finishes at global step 15500 after exactly 1500 updates. It uses the 78-condition pool (47 SEAT_FULL + 31 BACK_FULL), equal SEAT/BACK effective-batch exposure, and a newly initialized C-only contextual occupancy head trained alongside the generator. The generator objective remains the original free-coordinate FM with OT then HYBRID 0.25; the head has standard unweighted BCE, fixed logit threshold 0, and no DEV calibration.

The historical S2000 reference is global step 15000. This comparison therefore changes the condition pool, the layout head, and the accumulated training budget together. It can show the observed effect of this complete recipe; it cannot isolate a causal benefit of BACK data, context attention, unweighted BCE, or an extra 500 updates.

Every comparison must use the same request identity, C, N, frozen noise and frozen queries. Match processing stage to processing stage: RAW vs RAW; BASE_POST vs BASE_POST; GUARD_GEO vs GUARD_GEO; UNIFIED_FINAL vs UNIFIED_FINAL. The old previously maintained final result may be shown as `PRIOR_FINAL`, separately from the matched unified pipeline. Never count a new postprocess benefit as a generator-only benefit.

## 2. Panel and aggregation

| Task | Original split label | Conditions / parents | Noises per C | Outputs | Interpretation |
|---|---|---:|---:|---:|---|
| SEAT_FULL | TRAIN | 4 | 4 | 16 | Familiar seat regression |
| SEAT_FULL | DEV | 3 | 4 | 12 | Seat transfer within the frozen development panel |
| BACK_FULL | FAMILIAR | 4 | 2 | 8 | Previously familiar parents; these BACK conditions are now explicitly included in the new mixed training pool |
| BACK_FULL | DEV | 4 | 2 | 8 | Held-out BACK development panel |
| Total | Separate strata above | 15 C identities | — | 44 | No pooled success claim that offsets SEAT regression with BACK improvement |

Use the frozen comparison definition: noise mean within condition, condition mean within parent, then equal-parent mean within each part/split. Preserve the output mean and all denominators alongside it. This panel has one C per parent within each part/split; do not double-count the two noises as independent parents. Keep descriptive threshold counts as request counts and identify their denominators.

All numeric fields are retained in `comparison/per_output.csv`, `per_condition.csv`, `per_parent.csv`, `summary_parent_equal.csv`, and `paired_per_output.csv`. The report emphasizes the predeclared key fields but does not discard conflicting metrics or NA results.

## 3. Same-stage quality table

Populate one table for each of the four part/split strata. Each cell is `old S2000 → MULTIPART (new − old)`; rate deltas are percentage points. Show both the original field identifier and readable label. `ConflictCov_C` must not be relabelled as whole-mesh intersection coverage. A field absent or inapplicable at a stage remains NA with its valid-output count.

| Stage | CF illegal affected-free % | FF illegal affected-free % | ConflictCov_C % | Free min-angle <5° % | Isolated one-face components (legal graph) | Isolated one-face components (unfiltered graph) | Nonmanifold edges | Valid outputs |
|---|---|---|---|---|---|---|---|---|
| RAW | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| BASE_POST | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| GUARD_GEO | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| UNIFIED_FINAL | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

## 4. Same-stage connection and surface-retention table

Again use four independent part/split tables, not one favorable overall mean. Reachability is measured on the frozen scorer's legal shared-edge graph; it does not certify watertightness, mechanical usability or whole-body completeness. A request with ≥30% reached bulk area is a descriptive threshold outcome, not an independent visual adoption verdict.

| Stage | Legal C/F edges | Legal C/F edge length (model units) | Legal seam/bulk edges | Reached bulk area % | Requests ≥30% reached bulk area / valid | Actual free faces | Deleted free faces | Free surface area (model units²) |
|---|---|---|---|---|---|---|---|---|
| RAW | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| BASE_POST | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| GUARD_GEO | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| UNIFIED_FINAL | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

A smaller deletion count or larger retained area alone is not an improvement: it may retain defective small faces. Cross-reference the independent real image review, without deriving image labels from numeric scores.

## 5. Layout diagnostic (source completion reference only)

Run the already prepared `layout_diagnostic --predicted` only after all 15 actual predicted layouts/logit receipts exist. It must read saved predictions and source reference caches; it must not perform another head/model forward or send source owner labels into inference.

| Part / split | Edges | Source owner | Source alias | Source unsupported Null | TP | FN | FP (alias / Null) | TN | Owner recall | Predicted active |
|---|---:|---:|---:|---:|---|---|---|---|---|---|
| SEAT / TRAIN | 66 | 62 | 0 | 4 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| SEAT / DEV | 24 | 20 | 3 | 1 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| BACK / FAMILIAR | 72 | 62 | 0 | 10 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| BACK / DEV | 65 | 64 | 1 | 0 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

These are identities in one source completion, not a complete taxonomy of all valid completions. In particular, these caches do not support a universal rule that BACK upper edges are Null. Report all-off conditions, alias activation and individual conditions where layout changes accompany generation differences, without claiming the layout change alone caused them.

## 6. Paired gains, losses and tails

For every part/split and stage retain the per-request deltas, the per-parent means, the largest gains and costs, and all NA reasons. Name newly gained and lost ≥30%-reach requests separately. Print SEAT regressions even if BACK improves. Do not select a best checkpoint or change the fixed endpoint based on these results.

Expected table:

| Part / split | Stage | Metric | Largest gain request / delta | Largest cost request / delta | New threshold positives | Lost threshold positives | NA / failure |
|---|---|---|---|---|---|---|---|
| Each frozen stratum | Each applicable stage | Predeclared key metric | PENDING | PENDING | PENDING | PENDING | PENDING |

## 7. Actual completion and compute accounting

PENDING: authenticate final training endpoint, 44 RAW, 15 saved C-only layout predictions, 44 single official D0 results, base-score receipts, both matched unified pipelines and comparison completion. Read existing receipts; no redundant scoring or generation is authorized by this report.

The numerical summary adds zero model forwards, zero optimizer updates and zero scorer calls. Any engineering failure of the read-only aggregation must be recorded separately and must not trigger a model rerun.

## 8. Bounded conclusion

PENDING full numerical caches and separate actual image reviews (SEAT28 by root; BACK16 by the assigned reviewer). Final interpretation will answer: does this combined multipart recipe improve BACK on the frozen panel, what happens to SEAT, and do quality/retention costs make it worth retaining as a candidate? No default model change, causal single-component claim or universal generalization claim follows automatically.
