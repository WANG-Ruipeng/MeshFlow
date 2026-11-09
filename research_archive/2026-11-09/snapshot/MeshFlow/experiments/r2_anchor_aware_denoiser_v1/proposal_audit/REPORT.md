# Archived official proposal: C restoration diagnostic

All 48 existing official proposals were read on CPU. F_hat is identical on both sides and is before GEO_C. Original C source-ID boundary, fixed L, all eight thresholds, strict < and per-output macro averaging are unchanged.

**Moved C is illegal for the completion task. These are explanatory distances, not candidate-model scores, training whitelist rules, or stopping gates.**

The denoiser separates corner copies of the same original source vertex. C_hat therefore has no unique source-vertex coordinate. The canonical diagnostic uses the LAST occurrence, following the inherited topology dictionary overwrite order. Every occurrence and per-ID minimum/maximum distance bounds are saved; no corner averaging, weld, geometry change, or hidden vertex-count denominator change occurs.

| generator | C query | mean distance (model) | median distance macro | p95 distance macro | A-VMR@1%L (%) | PMR@1%L (%) |
|---|---|---:|---:|---:|---:|---:|
| R2_CONT_OLD | ORIGINAL_C | 0.314119 | 0.296664 | 0.475968 | 18.022 | 0.000 |
| R2_CONT_OLD | PROPOSAL_C_LAST_CORNER | 0.268140 | 0.246600 | 0.425983 | 40.112 | 16.667 |
| R2_CONT_USER | ORIGINAL_C | 0.435384 | 0.427130 | 0.609675 | 21.429 | 8.333 |
| R2_CONT_USER | PROPOSAL_C_LAST_CORNER | 0.400933 | 0.385315 | 0.580277 | 32.937 | 16.667 |
| R2_FROZEN | ORIGINAL_C | 0.318054 | 0.309208 | 0.489323 | 20.073 | 0.000 |
| R2_FROZEN | PROPOSAL_C_LAST_CORNER | 0.282393 | 0.273846 | 0.449282 | 41.270 | 33.333 |
| R2_USER_MIX | ORIGINAL_C | 0.453472 | 0.436071 | 0.703349 | 5.556 | 0.000 |
| R2_USER_MIX | PROPOSAL_C_LAST_CORNER | 0.433705 | 0.409687 | 0.691328 | 14.550 | 0.000 |

Canonical restoration raises the per-output mean boundary-to-free-vertex distance in 37/48 outputs, lowers it in 11/48; aggregate original minus proposal mean distance = 0.033965 model units. The sign describes local proximity only and does not identify the principal cause of failed attachment.

Files: per_boundary_vertex.csv (496 IDs), per_boundary_corner_occurrence.csv (all copies), per_output_distances.csv (48 × 4 states), per_output_thresholds.csv (48 × 4 × 8), grouped_summary.csv and restore_effect_paired_summary.csv (generator/region/parent/noise). Minimum/maximum copy-distance states are sensitivity bounds, not realizable alternate meshes.

Independent cKDTree check: 992 distances, maximum absolute disagreement 0. Protected input/source files unchanged: 165. No torch import, model load, GPU work, noise or geometry generation.
