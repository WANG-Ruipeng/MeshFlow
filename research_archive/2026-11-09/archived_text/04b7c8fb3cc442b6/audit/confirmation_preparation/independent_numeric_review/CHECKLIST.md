# Confirmation independent numerical audit: prepared checklist

Status: PREPARED_ONLY. No confirmation quality table, private method mapping, actual image or model has been read/run for this preparation. This checklist is based only on frozen source code. Execution must wait until schedule40_train has actually inspected all assigned 48 original PNGs and frozen its 24 request notes / 48 output observations and three cross-noise notes.

## 1. Mandatory gate before any numerical read

Read audit/visual_review/confirm/schedule40_train/initial_freeze_receipt.json first. Require schema recipe_confirmation_initial_visual_freeze_v1, status FROZEN_INITIAL_REVIEW, reviewer schedule40_train, private_mapping_read=false, confirmation_quality_read=false, prior_main_and_course_knowledge=true, AI_reviewer=true; counts requests=24, outputs=48, images=48. Verify all original note and image hashes against the freeze receipt/safe manifest, including five-view declarations, image timestamps no later than freeze time and all three parent cross-noise entries. Do not fabricate missing notes. If incomplete, return PENDING without reading quality tables. The root still controls actual execution.

Then use analysis/confirmation/latest.json to locate the completed immutable analysis. Require analysis_receipt COMPLETE and verify all output hashes, provenance input hashes and the three analyzer source hashes before and after reading. Record each file actually read, SHA256, exact script/tool command, exit code, elapsed CPU-audit wall time, exceptions and zero new model/scorer/geometry/RNG calls. Never reopen weights, raw meshes or invoke the analyzer/scorer to repair an inconsistency.

## 2. Exact panel and table counts

Candidate minus control is C20_40_MIX minus R2_MIX_H_ALL. Confirmation uses n8/9/10/11; first-stage effect references use n0/1/6/7. Both use the same eight DEV parents and two ratios. Read registered request identities, not model completion order.

| Table or unit | Expected count | Meaning |
|---|---:|---|
| Registered confirmation requests | 64 | 8 parents x 2 ratios x 4 noises |
| all_metrics.json / per_output.csv | 128 | 2 labels x 64 requests |
| metric_catalog.csv | 49 | 47 fixed numeric fields + 2 nonnumeric fields |
| per_parent_metrics.csv | 1504 | 2 labels x 16 parent-ratio groups x 47 |
| model_summary.csv | 564 | 2 labels x 2 ratios x 3 strata x 47 |
| paired_noise_metrics.csv | 3008 | 64 paired requests x 47 |
| paired_noise_summary.csv | 282 | 2 ratios x 3 strata x 47 |
| paired_parent_metrics.csv | 752 | 16 paired parent-ratio groups x 47 |
| paired_parent_summary.csv | 282 | 2 ratios x 3 strata x 47 |
| leave_one_parent_out.csv | 1504 | 2 ratios x (8+6+2 omissions) x 47 |
| metric_status_summary.csv | 588 | 2 labels x 2 ratios x 3 strata x 49 |
| diversity_pairs.csv | 192 | 2 labels x 16 conditions x 6 within-condition pairs |
| diversity_per_condition.csv | 32 | 2 labels x 16 conditions |
| diversity_summary.csv | 12 | 2 labels x 2 ratios x 3 strata |
| diversity_paired_parents.csv | 16 | one candidate-minus-control effect per parent-ratio |
| diversity_paired_summary.csv | 6 | 2 ratios x 3 strata |
| diversity_leave_one_parent_out.csv | 32 | 2 ratios x (8+6+2 omissions) |
| cross_repeat_parent_effects.csv | 752 | 16 parent-ratio groups x 47 |
| cross_repeat_group_effects.csv | 282 | 2 ratios x 3 strata x 47 |
| cross_repeat_diversity_parent_effects.csv | 16 | matched parent-ratio effects |
| cross_repeat_diversity_group_effects.csv | 6 | 2 ratios x 3 strata |

First-stage pair tables have the same corresponding counts. The 128 selected first-stage rows come from main R2 only plus course C20 only; do not accidentally include main R1 or course C40. Summary and LOO rows repeat underlying evidence across strata and must not be counted as extra independent samples.

## 3. Source identity and denominator checks

- Exactly two unique label/request cells for every registered request, no duplicate or unknown cell. Each source row has PASS and all contract booleans true.
- Match parent_id, condition_id, ratio, candidate, condition_cohort, N and K to the frozen request. Match row.identity.request_id/label/input_sha256 to that request and row.identity.query_sha256 to both outer row query_sha256 and the frozen 16-condition query registration.
- Query support is 2048 C samples per condition. Match query_weight_denominator and L0; registered_free_faces=N-K. The two labels share identical C/input/Gaussian/query identities within each request. All four noises share each condition's frozen query bank.
- Keep all 49 fields, including ConflictCov_C_status and EXACT_COORD_closed_two_faces_per_edge. The latter two are nonnumeric; booleans must never be treated as 0/1 numeric metrics. Preserve declared status and NA/missing reasons, including all-NA numeric fields.
- Derive ORIGINAL6/EXTENDED2 from each parent's C40 membership and use it also for C20. Require exactly six plus two parents at each ratio; do not substitute the C20 condition_cohort for parent membership.

## 4. Independent arithmetic from original saved scalar records

Use stdlib-only arithmetic modeled on audit/course_independent_review/audit_scalars.py, not analyzer.tables() to reproduce its own result. That old script has top-level real data reads and therefore MUST NOT be imported. Reuse only inspected pure helpers by copying their arithmetic into a new independent adapter if needed. Comparison tolerance abs_tol=1e-12, rel_tol=1e-10; exact identity/count/sign comparisons remain exact.

For every numeric field and all 128 outputs, independently construct label/parent/ratio/noise lookup. First average exactly four registered noises per parent, then equally weight parents within ALL (8), ORIGINAL6 (6), EXTENDED2 (2). Calculate every paired noise delta candidate-control; paired parent delta must equal both the difference of complete parent means and the complete four-noise mean of deltas.

A missing/nonfinite/nonnumeric expected observation makes the complete mean NA; never silently reduce the denominator. Verify expected/available/missing/status and explicitly separate available_mean. Check median, linearly interpolated p05/p95, min/max, population SD and negative/zero/positive counts using the frozen definition. Check all 1504 parent means, 3008 noise deltas, 752 parent deltas, 564 model summaries, 282 paired-noise summaries and 282 paired-parent summaries against their saved tables.

Recompute each summary's minimum/maximum parent or request IDs with the same registered order/tie handling. Recompute all 1504 explicit leave-one-parent-out rows independently and their summary min/max, omitting one parent only within the current stratum. The EXTENDED2 LOO leaves one parent; these ranges are not confidence intervals. Preserve valid descriptive sign counts for incomplete groups without falsely declaring a complete mean.

For metrics with a defined direction, recompute top1/top2 shares of gross beneficial changes, with denominator sum(max(0, signed benefit)); zero total => NA. No direction is imposed on descriptive metrics or D_free. Report concentration together with opposite tails; do not select only favorable fields.

## 5. CF, FF, I, triangle quality and costs

Check all 128 I union identities: ConflictCov_C = ConflictCov_C_cross + ConflictCov_C_dup - ConflictCov_C_cross_and_dup. Retain lower/upper, uncertain fraction, interior support, unresolved pairs and segment-extraction failures. Confirm union fields are finite/in expected [0,1] range when status permits; do not invent an unregistered upper-bound formula or rerun geometry.

Explicitly retain CF proper affected-free fraction/count, CF coplanar and unresolved counts; FF illegal and proper fractions/counts, coplanar and unresolved counts; I cross/dup/overlap components; free angle<5 fraction, p05 angle, q>10, q mean, exact zero/near-degenerate fractions and exact-duplicate excess. CF and FF rates use their existing affected-face definitions, not interchangeable pair-count denominators. For percentage displays use fraction x100 and identify deltas as percentage points. Exact-coordinate topology diagnostics remain their original identity, not a claim of original vertex-ID preservation.

Produce complete 20/40 and original6/extended2 comparisons, parent improvement/worsening counts only for metrics with directions, largest reverse parent/noise tails and gross-benefit concentration. Interpret costs jointly with the already-frozen image notes. No human scores, metric redefinition, semantic usability claim, significance claim or automatic model promotion.

## 6. D_free six-pair checks

For each of 32 label/condition groups require exactly the six unique pairs combinations(8,9,10,11,2), no self-pair, duplicate or reversed substitute. Recompute each D_free_mean from all six PASS finite saved values; expected_pairs=6 and available_pairs match observed. If any pair is unavailable the complete mean is NA, not a survivor mean.

Recompute the 12 model/stratum summaries, 16 candidate-control parent deltas, six paired summaries and 32 LOO rows from those input means. Preserve all 192 pair records. Relative change uses candidate/control-1 only when control is finite and nonzero; group relative change (if added descriptively) is ratio of group means and must not be mislabeled as mean of per-parent relative changes. D_free measures variation; larger is not automatically better.

## 7. Cross-repeat effects and conclusion limits

Independently recompute first-stage selected-pair parent/stratum effects from the saved first-stage scalar rows, and compare with repeat effects by parent+ratio+metric or ratio+stratum+metric. Verify effect_change=repeat_effect-first_stage_effect and sign_relation including zero/NA rules. Verify 752 parent and 282 group rows plus diversity16+6. Never pair n0 with n8 as output-level matches.

Both training input order/randomness and evaluation noise changed while official/model initialization and Geo1010 stayed fixed; this is independent input-stream replication, not an independent model-initialization trial or pure training-seed effect. The same eight previously seen DEV parents do not establish new-parent generalization. Preserve condition-pool confounding. Report whether the proposed candidate's signal repeats, countervailing I/CF/FF/angle/diversity costs and visual tails, without automatic adoption or additional experiments.

## 8. Future audit delivery

Write only a fresh audit/confirmation_independent_review/ directory: audit.json (actual checks, all exceptions and source hashes), complete independently computed per-noise/per-parent/stratum tables as necessary, core summary and a concise findings document. Preserve original scalar files, initial visual notes, mappings, weights, RAW and frozen analysis. A detected error is reported first; no silent mutation or rescoring. Missing completed analysis => PENDING rather than partial COMPLETE.