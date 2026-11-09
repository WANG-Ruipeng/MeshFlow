# R2 USER mix: complete AI_PROXY visual-review summary

All 24 original colored/neutral PNGs were reviewed: 12 fixed requests, 6 parents, 4 models, 48 output judgments and 36 new-versus-historical paired judgments. The two camera views are not independent samples.

These are AI_PROXY observations, not human ratings. **Similar does not mean successful or geometrically clean.** Whole shape and C use remain separate. The root reviewer had prior numerical aggregate context; this is not a wholly blinded review.

## Per-baseline paired counts

| Baseline | Dimension | Improved | Similar | Degraded | Uncertain | Requests |
|---|---|---:|---:|---:|---:|---:|
| R2_FROZEN | overall | 1 | 7 | 3 | 1 | 12 |
| R2_FROZEN | C_use | 3 | 7 | 1 | 1 | 12 |
| R2_CONT_OLD | overall | 1 | 8 | 3 | 0 | 12 |
| R2_CONT_OLD | C_use | 3 | 7 | 1 | 1 | 12 |
| R2_CONT_USER | overall | 1 | 8 | 2 | 1 | 12 |
| R2_CONT_USER | C_use | 1 | 9 | 1 | 1 | 12 |

## Region counts

| Baseline | Region | Dimension | Improved | Similar | Degraded | Uncertain | Requests / parents |
|---|---|---|---:|---:|---:|---:|---:|
| R2_FROZEN | seat | overall | 0 | 3 | 1 | 0 | 4 / 2 |
| R2_FROZEN | seat | C_use | 3 | 0 | 1 | 0 | 4 / 2 |
| R2_FROZEN | back | overall | 1 | 4 | 2 | 1 | 8 / 4 |
| R2_FROZEN | back | C_use | 0 | 7 | 0 | 1 | 8 / 4 |
| R2_CONT_OLD | seat | overall | 1 | 2 | 1 | 0 | 4 / 2 |
| R2_CONT_OLD | seat | C_use | 3 | 0 | 1 | 0 | 4 / 2 |
| R2_CONT_OLD | back | overall | 0 | 6 | 2 | 0 | 8 / 4 |
| R2_CONT_OLD | back | C_use | 0 | 7 | 0 | 1 | 8 / 4 |
| R2_CONT_USER | seat | overall | 0 | 3 | 0 | 1 | 4 / 2 |
| R2_CONT_USER | seat | C_use | 1 | 2 | 1 | 0 | 4 / 2 |
| R2_CONT_USER | back | overall | 1 | 5 | 2 | 0 | 8 / 4 |
| R2_CONT_USER | back | C_use | 0 | 7 | 0 | 1 | 8 / 4 |

## Parent consistency across the two fixed noises

The table preserves both noise judgments. Only `both_improved` supports a consistent improvement for that one parent/condition; it remains development evidence.

| Baseline | Parent | Region/tier | Overall n0 / n1 | C use n0 / n1 |
|---|---|---|---|---|
| R2_FROZEN | 33ec1e64a02ff718faaadcdc0221eb72 | seat/medium | similar / degraded | degraded / improved |
| R2_FROZEN | 534b0e6e2c53a8e1a7cdfd3874ed3723 | back/full | similar / degraded | similar / uncertain |
| R2_FROZEN | 866e7248434968051cc7750d0afca796 | seat/full | similar / similar | improved / improved |
| R2_FROZEN | 86ec9c796a6c1789e792598ed58dd1cc | back/full | similar / similar | similar / similar |
| R2_FROZEN | bdd29e651e5f6fb2b079317292bdc5d4 | back/medium | degraded / improved | similar / similar |
| R2_FROZEN | f59627a077dcc2566c63b8b48495261a | back/small | uncertain / similar | similar / similar |
| R2_CONT_OLD | 33ec1e64a02ff718faaadcdc0221eb72 | seat/medium | similar / degraded | degraded / improved |
| R2_CONT_OLD | 534b0e6e2c53a8e1a7cdfd3874ed3723 | back/full | similar / degraded | similar / uncertain |
| R2_CONT_OLD | 866e7248434968051cc7750d0afca796 | seat/full | improved / similar | improved / improved |
| R2_CONT_OLD | 86ec9c796a6c1789e792598ed58dd1cc | back/full | similar / similar | similar / similar |
| R2_CONT_OLD | bdd29e651e5f6fb2b079317292bdc5d4 | back/medium | degraded / similar | similar / similar |
| R2_CONT_OLD | f59627a077dcc2566c63b8b48495261a | back/small | similar / similar | similar / similar |
| R2_CONT_USER | 33ec1e64a02ff718faaadcdc0221eb72 | seat/medium | similar / uncertain | degraded / similar |
| R2_CONT_USER | 534b0e6e2c53a8e1a7cdfd3874ed3723 | back/full | improved / degraded | similar / uncertain |
| R2_CONT_USER | 866e7248434968051cc7750d0afca796 | seat/full | similar / similar | improved / similar |
| R2_CONT_USER | 86ec9c796a6c1789e792598ed58dd1cc | back/full | similar / similar | similar / similar |
| R2_CONT_USER | bdd29e651e5f6fb2b079317292bdc5d4 | back/medium | degraded / similar | similar / similar |
| R2_CONT_USER | f59627a077dcc2566c63b8b48495261a | back/small | similar / similar | similar / similar |

## Coverage and interpretation

- All 48 outputs are explicitly marked actually viewed; all 36 pairs and all 24 original images are present once, with no missing or duplicate records.
- Original source text and categories are preserved exactly in `AI_PROXY_per_output.csv` and `AI_PROXY_paired.csv`; their source file/hash and input metadata are attached.
- `AI_PROXY_summary.json` retains per-image hashes, source review context, region counts and all six-parent/two-noise comparisons.
- Different chair style does not itself establish better quality. Pixels cannot certify hidden geometry, connectivity or illegal intersections.
- These historical checkpoints differ in training history. This small, repeatedly viewed development panel does not establish generalization or sole-factor causality.
- This merge used no model, scorer or experiment audit runner.
