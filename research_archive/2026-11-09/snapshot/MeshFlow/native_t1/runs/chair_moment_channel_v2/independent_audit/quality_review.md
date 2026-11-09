# Independent endpoint quality review

Arithmetic audit: **PASS**, 906/906 checks. Existing CSV data only; no model calls or score recomputation.

The label **NO_CLEAR_LEARNING_SIGNAL** is supported. Keep the simpler HYBRID/START research reference and do not expand relation-channel training on this evidence. This does not change the maintained default model.

| Primary A20 validation | Mean fixed-query MSE |
|---|---:|
| START | 0.0015310739 |
| H_CONT_END | 0.0015961439 |
| MOMENT_XY_END | 0.0016433661 |

MOMENT versus H_CONT worsens the parent mean by 2.96% (3/8 parents improve, 5/8 worsen); both noise-specific means worsen. H_CONT and MOMENT versus START worsen by 4.25% and 7.33%. The parent medians and noise-specific signs must also be retained: improvements in some cases do not establish a stable overall benefit.

| Primary paired comparison | Mean delta | Median parent delta | Improved parents |
|---|---:|---:|---:|
| H_CONT_END-START | +6.50700038e-05 | -6.25293333e-06 | 4/8 |
| MOMENT_XY_END-H_CONT_END | +4.72222232e-05 | +3.46263117e-05 | 3/8 |
| MOMENT_XY_END-START | +0.000112292227 | +2.62320657e-05 | 3/8 |

| A20 validation metric | START | H_CONT | MOMENT |
|---|---:|---:|---:|
| GT_to_F_original_RMS_L | 0.035490676 | 0.036146048 | 0.036522966 |
| F_to_GT_original_RMS_L | 0.037902224 | 0.036897682 | 0.037286714 |
| fixed_query_p95_L | 0.068844734 | 0.070568342 | 0.071017968 |
| coverage_fraction | 0.32299805 | 0.34960938 | 0.34411621 |
| Gamma_to_free_edges_RMS_bbox_pct | 3.2538945 | 3.5423643 | 3.6998947 |
| Gamma_to_free_edges_coverage_0.001_fraction | 0.0035608139 | 0.00374091 | 0.00289293 |
| CF_proper_affected_free_fraction | 0.060134912 | 0.064275341 | 0.063516916 |
| FF_illegal_affected_free_fraction | 0.92627031 | 0.91208338 | 0.91131698 |
| free_near_degenerate_fraction | 0 | 0 | 0 |
| free_minimum_angle_lt5_fraction | 0.14675664 | 0.10344472 | 0.10665415 |
| small_angle_abs_gap_to_same_GT | 0.13389986 | 0.10011092 | 0.10103748 |
| free_angle_p05_degrees | 5.6727129 | 6.344456 | 6.3874092 |

Both directional distances, p95, coverage and interface RMS worsen for MOMENT versus H_CONT on A20 validation. CF/FF affected fractions improve slightly, but FF remains about 91%. No new nonfinite/C/N preservation failures appear in the saved output table. There is no single all-metric win. A20 training geometry improves, while A40 validation geometry worsens; observed holdout has mixed directions and essentially flat MSE. A40 denominators remain 7/8 train and 6/8 validation.

Angles require matched GT context: the GT sub-5-degree fraction averages 0.1782926. The H/M fractions are lower than GT, which is not proof of cleaner geometry. MOMENT has a slightly worse mean per-output absolute GT fraction gap than H_CONT, despite a slightly larger angle p05; versus START that gap improves. Neither increasing nor decreasing small-angle counts alone defines quality.

The actual 3x3 entries use generated-input alpha as rows and target beta as columns. All saved column advantage/strict-win calculations were independently reproduced from the full matrices. A_column is mean(off-diagonal column entries) minus the diagonal; row diagnostics answer a different question. No-response correctly has zero column advantage and no strict wins.

Validation MODEL strict correct-column choices are START 8/24, H_CONT 9/24, MOMENT 9/24; none achieves all three targets on any of the 8 curves. Width response errors improve slightly in the aggregate from H to M, but worsen on noise0 and improve on noise1. Baseline width ratios remain about 0.70-0.72. The identity b_GT = r_base * b_self holds per curve: baseline bias and relative response remain distinct. Post-scale reaches near-unit relative width response and 10/24 strict choices for H/M; it is an offline own-F1+C diagnostic, not equivalent model conditioning.

| A20 mean MSE contribution | both_in | enter | leave | both_out |
|---|---:|---:|---:|---:|
| H_CONT_END-START | -5.83282425e-08 | -0.000205571375 | +0.000101216694 | +0.000169483014 |
| MOMENT_XY_END-H_CONT_END | -3.30863645e-08 | -2.2263142e-06 | +1.66528501e-05 | +3.28287737e-05 |
| MOMENT_XY_END-START | +6.18340162e-08 | -0.000205149722 | +9.49363215e-05 | +0.000222443794 |

Four-part contributions use the full Q denominator and sum to the paired MSE delta. both_out is threshold-defined; it must not be relabeled an anatomical far end.

The dominant parent is `c1b8c2f0c4f3cc43aff7c08b060f5ed6`. Noise0 improves slightly for both continuations; noise1 deteriorates in both.

| c1b... comparison | noise | MSE delta | both_out contribution |
|---|---:|---:|---:|
| H_CONT_END-START | 0 | -4.75251923e-05 | -5.19765424e-05 |
| H_CONT_END-START | 1 | +0.00332859907 | +0.00319966393 |
| MOMENT_XY_END-H_CONT_END | 0 | -7.72102613e-06 | -8.14429403e-06 |
| MOMENT_XY_END-H_CONT_END | 1 | +0.000389547579 | +0.000389495854 |
| MOMENT_XY_END-START | 0 | -5.52462184e-05 | -5.99334104e-05 |
| MOMENT_XY_END-START | 1 | +0.00371814665 | +0.00358401153 |

On noise1, both_out contributes 96.13% of the H-START net increase and 96.39% of the M-START increase; it explains nearly all of M-H. Thus the saved tables already answer whether regression concentrates in the same parent/noise and threshold region. They do not show spatial overlap of the erroneous queries or explain the cause.

One priority next step, proposed only: use these existing raw outputs and frozen queries to spatially inspect c1b... noise1, retaining the same parent/noise0 as the fixed control. The falsifiable hypothesis is that both continuations lose or misplace the same local GT region, beyond a simple global width bias. Disjoint affected query regions or an adequate single affine width explanation would contradict it. Do not launch more training to select a favorable subset on the same evidence.

Limits: this is one training stream with two sampling noises, eight primary parents, and a post-hoc diagnostic hypothesis. No statistical significance or independent replication is claimed. All original NA rows and metric definitions remain unchanged. Full selected numeric evidence and SHA256 inputs are in `quality_review.json`; source tables are in `../paired_results/`.
