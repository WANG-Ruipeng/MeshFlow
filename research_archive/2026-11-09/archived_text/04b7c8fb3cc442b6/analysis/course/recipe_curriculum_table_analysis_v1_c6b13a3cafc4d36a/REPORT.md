# Curriculum comparison from saved results

192 outputs: the two curriculum endpoints plus 64 existing R2 cells. No model or scoring calls; no automatic winner.
Four noises are averaged per parent and parents are weighted equally. C20 and C40 remain separate. Original6/Extended2 at C20 is a parent-membership diagnostic, not a new evaluation cohort.
All curricula use the registered H_ALL recipe. Reordering the same multiset changes update grouping and the whole-batch free denominator; this is a complete curriculum comparison, not constant per-scalar optimizer weighting. Historical condition-pool construction differences remain.
Single training realizations on previously observed DEV parents; signs, tails and leave-one-parent-out summaries are descriptive, not statistical significance. Reference GT/S/E diagnostics are kept separately and never used to choose a winner.

## C=20%, ALL

| Model | CF proper % | FF illegal % | I lower % | I upper % | Angle<5° % | Near-degenerate % |
|---|---:|---:|---:|---:|---:|---:|
| R2_MIX_H_ALL | 4.7949 | 89.2799 | 3.3066 | 3.3066 | 11.2974 | 0.0000 |
| C20_40_MIX | 5.0103 | 88.1928 | 3.8818 | 3.8818 | 11.3425 | 0.0000 |
| C40_20_MIX | 4.6299 | 88.9728 | 3.6514 | 3.6514 | 10.9550 | 0.0000 |

## C=20%, ORIGINAL6

| Model | CF proper % | FF illegal % | I lower % | I upper % | Angle<5° % | Near-degenerate % |
|---|---:|---:|---:|---:|---:|---:|
| R2_MIX_H_ALL | 4.8227 | 88.1241 | 3.1942 | 3.1942 | 7.5174 | 0.0000 |
| C20_40_MIX | 4.5742 | 87.0455 | 3.2694 | 3.2694 | 8.1315 | 0.0000 |
| C40_20_MIX | 4.7629 | 87.6642 | 3.1413 | 3.1413 | 7.6058 | 0.0000 |

## C=20%, EXTENDED2

| Model | CF proper % | FF illegal % | I lower % | I upper % | Angle<5° % | Near-degenerate % |
|---|---:|---:|---:|---:|---:|---:|
| R2_MIX_H_ALL | 4.7115 | 92.7473 | 3.6438 | 3.6438 | 22.6374 | 0.0000 |
| C20_40_MIX | 6.3187 | 91.6346 | 5.7190 | 5.7190 | 20.9753 | 0.0000 |
| C40_20_MIX | 4.2308 | 92.8984 | 5.1819 | 5.1819 | 21.0027 | 0.0000 |

## C=40%, ALL

| Model | CF proper % | FF illegal % | I lower % | I upper % | Angle<5° % | Near-degenerate % |
|---|---:|---:|---:|---:|---:|---:|
| R2_MIX_H_ALL | 10.1780 | 84.5693 | 1.9394 | 1.9394 | 9.7137 | 0.0000 |
| C20_40_MIX | 9.7777 | 83.9154 | 2.0569 | 2.0569 | 8.8355 | 0.0000 |
| C40_20_MIX | 10.8164 | 84.7174 | 2.3758 | 2.3758 | 9.1165 | 0.0000 |

## C=40%, ORIGINAL6

| Model | CF proper % | FF illegal % | I lower % | I upper % | Angle<5° % | Near-degenerate % |
|---|---:|---:|---:|---:|---:|---:|
| R2_MIX_H_ALL | 9.0310 | 83.0114 | 1.9531 | 1.9531 | 7.2426 | 0.0000 |
| C20_40_MIX | 8.7583 | 82.4251 | 1.9246 | 1.9246 | 6.2057 | 0.0000 |
| C40_20_MIX | 10.1673 | 83.3424 | 2.5045 | 2.5045 | 6.8292 | 0.0000 |

## C=40%, EXTENDED2

| Model | CF proper % | FF illegal % | I lower % | I upper % | Angle<5° % | Near-degenerate % |
|---|---:|---:|---:|---:|---:|---:|
| R2_MIX_H_ALL | 13.6191 | 89.2429 | 1.8982 | 1.8982 | 17.1272 | 0.0000 |
| C20_40_MIX | 12.8361 | 88.3863 | 2.4536 | 2.4536 | 16.7250 | 0.0000 |
| C40_20_MIX | 12.7635 | 88.8425 | 1.9897 | 1.9897 | 15.9783 | 0.0000 |

## Complete paired evidence

All 49 metric fields and statuses are retained, including nonnumeric or all-NA fields. Every numeric metric has model, parent, noise-pair, parent-pair, distribution, tail and applicable concentration/leave-one-parent-out tables. Missing values preserve their fixed denominators; no survivor mean replaces the main mean.
paired_noise_metrics.csv and paired_parent_metrics.csv use first minus second, explicitly comparing C20_40_MIX minus R2, C40_20_MIX minus R2, and C20_40_MIX minus C40_20_MIX. No factorial coefficient from the earlier 2x2 study is reused.
Diversity preserves all six noise pairs per condition. More variation is not automatically better. Geometry cost reductions do not certify coherent chairs, attachment or successful semantic use of C. Full actual visual review and completed training accounting remain independent prerequisites for any later decision.
