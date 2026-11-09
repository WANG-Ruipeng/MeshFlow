# Frozen seam-tip / bulk cluster binding

Completed 112 registered CPU variants: LONG and A_COUPLE, 28 requests each, at 1.5%L and 2%L maximum actual PRE_GEO corner displacement. Formal execution took 145.258 seconds with two CPU workers; zero model/GPU calls, new queries, retries or additional deleted faces relative to the original POST. All 302 protected input hashes remain unchanged. The original 124 numeric metric definitions are retained in `per_output.csv`, `summary.csv`, and `paired_metrics.csv`.

This is a combined, deterministically sorted queue of point-cluster midpoint merges and complete lateral-edge correspondences. C and seam base anchors stay fixed. Both free clusters move to their midpoint, subject to every original cluster member's displacement cap. The 1.5%/2% numbers are per-corner motion caps, not search radii: representative gaps may be at most twice those values before the full-member guard. Point sharing itself receives no shared-edge or reachable-area credit. All connectivity values below come from the original actual legal-edge graph.

| Split / arm | Baseline bulk area reachable | 1.5% binding | 2% binding | Baseline / bound >=30% outputs |
|---|---:|---:|---:|---:|
| TRAIN LONG | 51.331% | 68.240% | 68.239% | 11/16 -> 16/16 |
| TRAIN A_COUPLE | 51.625% | 63.700% | 63.700% | 11/16 -> 14/16 |
| DEV LONG | 0.118% | 0.854% | 0.854% | 0/12 -> 0/12 |
| DEV A_COUPLE | 0.210% | 0.294% | 0.293% | 0/12 -> 0/12 |

LONG gains the >=30% marker on all four 73 noises and 85/n0. A gains it on 73/n0, 73/n2, and 85/n0. These are geometric graph outcomes on familiar TRAIN conditions, not newly assessed semantic success. For DEV, only 2d/n0 has nonzero final reachable bulk: LONG 10.253%, A 3.529%/3.521%. All 608/610 outputs remain at zero despite accepted point merges.

| Split / arm | FF baseline -> 1.5% / 2% | I baseline -> 1.5% / 2% |
|---|---:|---:|
| TRAIN LONG | 14.3508% -> 13.9376% / 13.9376% | 0.2655% -> 0.2319% / 0.2319% |
| TRAIN A | 13.1089% -> 12.7335% / 12.7335% | 0.4242% -> 0.3540% / 0.3540% |
| DEV LONG | 16.1266% -> 15.5688% / 16.3002% | 2.3153% -> 2.3193% / 2.4292% |
| DEV A | 19.1198% -> 19.1514% / 19.2734% | 3.1250% -> 3.1576% / 3.1738% |

Across all 112 outputs, original legal C/F and F/F edges are preserved, no new illegal C/F pairs appear, and no faces are removed. Nonmanifold edge counts do not increase. This does **not** mean no new conflict: per threshold, TRAIN LONG adds 11 illegal FF pairs while removing 24; TRAIN A adds 10 while removing 25. DEV LONG adds 19/35 while removing 19/29; DEV A adds 13/17 while removing 16/16. These paired face identities and rejection reasons are saved per cell. In particular, several 8c point merges introduce FF pairs that are not obvious in the views.

The separate array audit passes for all 112 outputs: fixed C bytes/order/indices, full-member displacement limits, final affected-face angle/q/normal guards, no lost original faces, and correct indexed/source-corner transport. The 24 cells accepting no binding reproduce all 124 original numeric fields (2,976 comparisons). This verification performs no additional geometric scoring scan.

All 32 TRAIN comparison pages (64 bound variants) were actually opened and reviewed as AI_PROXY_NONBLIND. The visible overall structures and use of blue C are preserved relative to each baseline; no conspicuous newly added tear or fragment was found at this display scale. Seven pages show a local rim/section joining improvement; several other graph gains are too small to certify visually. Existing irregular back plates, under-seat openings, and the LONG 85/n1 hanging plate remain. This is not a claim that every original chair was complete or every microscopic seam is closed. Actual labels and image hashes are in `../reviews/binding_train.csv`; DEV review is separate.

The result supports a bounded postprocessing improvement on the four familiar TRAIN conditions. It does not resolve the difficult DEV seam-to-bulk gap. The 2% setting adds DEV FF/I costs without adding >=30% outputs beyond 1.5%; neither setting is automatically promoted or applied to historical outputs.

Recovery: `python -B -m r2.pilots.seam_followup_v1.binding --stage run` verifies completed receipts and returns existing results. An incomplete attempted cell refuses silent replay. No default change, training, commit or push is part of this result.
