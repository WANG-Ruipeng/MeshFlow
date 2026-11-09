# Independent numerical crosscheck

Read-only arithmetic on the existing 252 score rows and 84 GEO receipts. No mesh scoring, model call, visual relabeling, or promotion decision.

`receipt.json` reports PASS for all 177 fields across output → condition → parent → macro: 49,914 metric means and 49,914 valid-count cells match. All 252 face-accounting identities and all 84 existing GEO receipt counts match. The original published table unit audit also passed 918 cells. Fraction columns in CSV remain 0–1; percentages below are display conversions only.

## Applicable denominators

- OLD_TRAIN (machine label CURRENT_TRAIN): 8 parents, one condition each, two fixed noises each, 16 requests per model and stage. NEW_TRAIN: 2 parents / 4 requests. NEW_PARENT_HOLDOUT: only 1 parent / 2 requests. DEV: 4 parents / 8 requests. SEAM_S3000 has no evaluation on the three new conditions; these comparisons are NA.
- Means give equal weight to parents after averaging each condition's noises. These are not pooled affected-face ratios. CF, FF and small-angle fractions use the currently retained free-face denominator at each stage; the raw denominators and post denominators differ materially.
- All 252 I statuses are PASS. `C_query_weight_denominator` is 1.0. The auxiliary `I_queries` column in this audit's macro CSV copies **ConflictCov_C_interior_samples**, not total query count. For a8 it is 69 (interior weight 0.03369140625); for 27 it is 1707 (0.83349609375). Do not call this column a total number of C queries or infer a new support normalization from it.
- `original_free_retained = N - K - deleted_free_faces`. `actual_free_faces = original_free_retained + new_bridge_faces` holds in every row. Native and candidate have no new bridge faces. Old SEAM adds two bridge faces total across OLD_TRAIN FINAL.
- `free_total_surface_area` is the current generated free surface area in model-coordinate squared units, not GT coverage, not preserved source-surface area, and not a verified subset of RAW area. Original-free retention counts are slot accounting, not evidence that the subject remains complete.
- Native and candidate SEAM-only metrics remain NA. Ordinary C/F legal edges are applicable to all methods; a legal edge alone does not certify C use, complete support, or a large connected subject.

## FINAL comparison

Every cell below is parent-equal. Legal-edge outputs are actual request counts.

| Group | Model | CF % | FF % | I % | <5° % | Original free retained | Deleted free | Free area | Legal-edge outputs | Mean legal C/F edges |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OLD_TRAIN | Native | 9.335 | 14.569 | 1.868 | 33.973 | 89.75 | 106.875 | 29.233 | 9/16 | 1.625 |
| OLD_TRAIN | Candidate | 8.696 | 15.801 | 5.069 | 33.275 | 87.50 | 109.125 | 31.470 | 8/16 | 1.000 |
| OLD_TRAIN | Old SEAM | 2.086 | 10.250 | 3.366 | 34.744 | 90.75 | 105.875 | 26.711 | 16/16 | 16.875 |
| DEV | Native | 7.288 | 24.052 | 12.854 | 58.013 | 68.75 | 105.25 | 27.799 | 0/8 | 0 |
| DEV | Candidate | 6.476 | 22.259 | 11.420 | 65.322 | 70.00 | 104.00 | 22.196 | 0/8 | 0 |
| DEV | Old SEAM | 7.903 | 27.209 | 13.147 | 55.882 | 68.875 | 105.125 | 27.699 | 8/8 | 2.5 |
| NEW_TRAIN | Native | 4.714 | 20.867 | 2.734 | 57.278 | 57.75 | 130.25 | 20.516 | 2/4 | 0.5 |
| NEW_TRAIN | Candidate | 0 | 1.786 | 0 | 71.142 | 30.75 | 157.25 | 3.795 | 3/4 | 0.75 |
| NEW_PARENT_HOLDOUT | Native | 20.324 | 15.917 | 9.253 | 19.137 | 125.5 | 91.5 | 9.543 | 0/2 | 0 |
| NEW_PARENT_HOLDOUT | Candidate | 24.195 | 16.939 | 11.304 | 17.691 | 126.5 | 90.5 | 9.912 | 0/2 | 0 |

The new-condition low conflict scores coexist with a much smaller remaining free surface and increased small-angle rates. The holdout has no new legal interface and worse CF, FF and I. OLD_TRAIN trades slightly lower CF/small-angle rates for higher FF/I and fewer legal-edge outputs. DEV lowers average CF/FF/I, but loses free surface area and raises small-angle rate; this is not an across-metric upgrade.

## NEW_TRAIN RAW → BASE → FINAL

| Model / stage | CF % | FF % | I % | <5° % | Free faces | Deleted free | Free area |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native RAW | 7.159 | 91.285 | 6.848 | 34.218 | 188.00 | 0 | 28.919 |
| Native BASE | 5.296 | 25.423 | 3.101 | 60.730 | 57.75 | 130.25 | 20.233 |
| Native FINAL | 4.714 | 20.867 | 2.734 | 57.278 | 57.75 | 130.25 | 20.516 |
| Candidate RAW | 15.037 | 86.111 | 10.022 | 49.832 | 188.00 | 0 | 16.381 |
| Candidate BASE | 0 | 1.786 | 0 | 71.142 | 30.75 | 157.25 | 3.665 |
| Candidate FINAL | 0 | 1.786 | 0 | 71.142 | 30.75 | 157.25 | 3.795 |

The decline in area is already visible in RAW relative to Native, and substantial further loss accompanies the D0/GEO chain. No original free faces are deleted between BASE and FINAL here. This table does not isolate the continuous D0 effect from subsequent GEO consolidation; the already recorded GEO ledger locates the actual face deletions without proving which upstream stage made the faces narrow.

Per parent, averaged only over its two fixed noises:

| Parent / model | RAW free area | BASE retained / requested free | BASE free area | FINAL free area | FINAL <5° % |
|---|---:|---:|---:|---:|---:|
| a8 / Native | 35.645 | 63.5 / 150 | 25.598 | 26.044 | 55.150 |
| a8 / Candidate | 20.088 | 35.5 / 150 | 2.713 | 2.888 | 88.712 |
| 27 / Native | 22.193 | 52 / 226 | 14.869 | 14.989 | 59.407 |
| 27 / Candidate | 12.674 | 26 / 226 | 4.617 | 4.702 | 53.571 |

Candidate a8 n0/n1 retains 37/34 faces; 27 n0/n1 retains 28/24 faces. The collapse is present in all four requests, not just one outlier. 27's small-angle percentage is lower than Native but its face count and free area are also much lower; a8 worsens strongly in small angles.

Existing GEO deletion reasons, NEW_TRAIN per request means:

- Native: 124.00 repeated-vertex faces + 6.25 exact duplicate free faces = 130.25 deleted.
- Candidate: 156.75 repeated-vertex faces + 0.50 exact duplicate free faces = 157.25 deleted.

Thus extra deletion is not explained by simply removing more exact duplicate triangles. “Repeated vertex” is the saved GEO consolidated-index reason, not proof that each pre-GEO triangle was already degenerate or that every deleted face was semantically useful.

## Parent-level costs hidden by the means

- OLD_TRAIN 6b: FINAL FF 36.751% → 54.918%, I 0.659% → 27.588%, legal edges 2 → 0 (two-noise means). This is a substantial local cost even though several other OLD_TRAIN FF values improve.
- DEV 2d: FINAL FF 57.664% → 47.425%; this supplies most of the average DEV FF improvement. The other DEV parents' FF is unchanged or higher.
- DEV 459f: FINAL I 2.832% → 0 and CF 5.833% → 0, while free area 23.170 → 3.142 and small-angle rate 61.25% → 94.262%. Do not count the conflict reduction without the area/shape cost.
- df6 holdout: FINAL CF 20.324% → 24.195%, FF 15.917% → 16.939%, I 9.253% → 11.304%; both models have zero legal C/F edges. This one-parent observation cannot establish broad generalization.

The complete per-parent, per-request and stage numbers are retained in the adjacent CSVs. Actual image-review conclusions must be taken from the separate completed AI_PROXY review, not inferred from this audit.
