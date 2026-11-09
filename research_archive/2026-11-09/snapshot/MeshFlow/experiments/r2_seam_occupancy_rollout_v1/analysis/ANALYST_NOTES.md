# Frozen occupancy rollout: analyst notes

Read-only interpretation of the completed 192 score cells and their derived CSVs. No additional geometry scans, model calls, or source-code changes. Percentages below are per-output macro means unless a count is explicitly stated. Four seen TRAIN conditions, four fixed noises each; SOURCE_ORACLE is a source-label-assisted diagnostic.

## What the layout intervention establishes

The occupancy layout matters for connectivity, but it does not explain all residual FF or the 3f/n2 I tail. Both new layouts add the same one nonzero seam-to-bulk request, **3f/n1**, changing nonzero coverage 13/16 to 14/16 and >=30% coverage 11/16 to 12/16. All layouts already have at least one legal CF edge in 16/16 outputs; additional edge count is not additional output-level CF coverage.

Original GEO results:

| Quantity | PREDICTED | ALL_ON | SOURCE_ORACLE |
|---|---:|---:|---:|
| Legal CF edges | 175 | 231 | 210 |
| Seam-to-bulk area macro mean | 49.2483% | 56.5023% | 56.8345% |
| FF affected-free fraction | 14.5629% | 14.1557% | 13.3980% |
| Illegal FF pairs | 343 | 345 | 316 |
| Common-BULK illegal FF pairs | 297 | 304 | 278 |
| Common-BULK affected fraction | 14.2615% | 13.5189% | 12.7719% |
| I / ConflictCov_C | 3.3020% | 3.1433% | 3.1891% |
| C/F nonmanifold edges | 18 | 27 | 32 |

The ALL_ON decrease in FF affected fraction does not mean the number of illegal pairs decreased: 343 to 345. Its fixed common-BULK pair count increases 297 to 304. The dynamic SEAM/BULK relabeling cannot be interpreted as physical removal of old bulk collisions. Common-BULK is the intersection of BULK RAW slots across all three layouts; face-count denominators after cleanup are separately recorded.

## Parent and noise structure

- **3f:** Both alternatives restore large seam-to-bulk reach at n1: 0 to 100% ALL_ON / 99.6412% SOURCE_ORACLE. Parent mean reach is 73.0953% to 98.6458% / 99.5051%. This is the clearest paired positive. Parent FF is 4.9553% to 4.3070% / 5.1499%; the lower ALL_ON mean includes n0/n1 gains and n2/n3 costs. At n3, the original FF=0 becomes 7.2368% / 6.7568%.
- **8c:** FF improves from 33.2946% to 31.6093% ALL_ON / 27.7357% SOURCE_ORACLE. SOURCE_ORACLE improves FF in all four noises, while ALL_ON improves two and worsens two. Common-BULK FF pairs total 204 to 219 / 185. Parent reach only rises 63.4904% to 65.4502% / 65.9198%, and all four already had nonzero reach.
- **85:** Both new layouts are identical to one another. Mean FF increases 17.0089% to 17.7136%, while legal CF edges increase 28 to 32 total and mean reach rises 33.0570% to 34.5629%. This is another tradeoff, not uniform improvement.
- **73:** Its layout was already fully occupied; unchanged results are an identity control, not another independent success. Across the panel, all 64 same-layout stage comparisons are byte-identical; the other 128 comparisons have intentionally different layouts.

## The 3f/n2 I tail remains

POST I is 42.5293% PREDICTED, 44.3848% ALL_ON, and 43.1152% SOURCE_ORACLE. SOURCE_ORACLE has zero exact illegal CF pairs at this endpoint, yet I remains 43.1152%, illustrating again that near-surface conflict coverage and exact illegal triangle pairs are different quantities.

RAW I is 24.3652% / 18.0664% / 12.3535%; PRE_GEO is 43.2129% / 49.3164% / 47.1680%. Thus the lower RAW I of the layout variants is not retained through the fixed D0-plus-anchor-restoration stage. These three endpoint snapshots do not identify a particular integration-time event or prove that D0 is the sole cause.

## Converted true-seam and source-Null slots

The 26 previously missed true edges give **104 slot/request observations**, not 104 independent conditions. The four source-Null edges give 16 observations. Numbers below concern the original GEO endpoint. Counts of incident FF pairs can count one pair twice if both converted slots participate; they are not unique whole-mesh pair counts.

| Fixed slot subset / layout | Retained | CF-involved retained slots | FF-involved retained slots | FF pair incidences |
|---|---:|---:|---:|---:|
| 104 true-seam slots, PREDICTED bulk role | 103 | 0 | 22 | 39 |
| 104 true-seam slots, ALL_ON seam role | 99 | 3 | 17 | 30 |
| 104 true-seam slots, SOURCE_ORACLE seam role | 100 | 3 | 15 | 27 |
| 16 source-Null slots, PREDICTED bulk role | 16 | 1 | 3 | 5 |
| 16 source-Null slots, ALL_ON seam role | 15 | 0 | 2 | 6 |
| 16 source-Null slots, SOURCE_ORACLE bulk role | 16 | 0 | 3 | 5 |

Restored true-seam slots are not uniformly collision-free or uniformly retained. ALL_ON deletes five of those 104 observations, SOURCE_ORACLE four, versus one when they were ordinary bulk slots. Of the 16 ALL_ON source-Null slot observations, the 3f/n1 slot18 is deleted as exact_collinear; the other 15 remain. The 12 3f Null observations have no exact illegal CF/FF pair at POST (one deleted). The 8c Null slot participates in one illegal FF pair at n0 and five at n2. This does not prove the source-Null extensions are semantically necessary or harmful: Null means absent from the source interface, not forbidden in every possible completion.

## Capacity backend: retain as a tradeoff candidate, not a default upgrade

For ALL_ON and SOURCE_ORACLE, the capacity backend removes all 27 / 32 C/F nonmanifold edges and retains all 231 / 210 original legal CF edges, adding 27 / 31 legal CF edges. It adds no new face deletions and retains six / seven extra faces.

The costs are material:

| Same-pre-GEO capacity versus original | ALL_ON | SOURCE_ORACLE |
|---|---:|---:|
| Original legal FF edges not reproduced | 20 | 27 |
| Lost original legal FF length, model units | 18.2349 | 24.7570 |
| New illegal CF pairs | 19 | 20 |
| Resolved illegal CF pairs | 2 | 0 |
| New illegal FF pairs | 12 | 17 |
| Resolved illegal FF pairs | 0 | 0 |
| FF macro increase | +0.7688 pp | +1.0526 pp |
| I macro increase | +0.2899 pp | +0.2289 pp |
| Mean seam-to-bulk reach change | -0.6474 pp | -0.7130 pp |

Coverage remains 14/16 nonzero and 12/16 >=30% for each new layout. The capacity rule succeeds at its incidence objective but is not a general geometry safeguard; it can add exact CF penetration even as it removes nonmanifold edges. These two backends start from the same pre-GEO geometry; missing terminal FF edges are not evidence of literally cutting an already indexed pre-GEO mesh.

## Actual visual review

The evaluator actually viewed eight 3f/8c RAW/POST contact sheets (two views each), plus the full 3f/n2 POST and 8c/n1 RAW pages. The authoritative 72-row review is `gallery/review_3f_8c_AI_PROXY_v2.csv`; V1 only differs in descriptive enum formatting and evidence-field naming, and must not be counted as a second reviewer. Every row includes image paths and hashes. This is nonblind AI_PROXY, not human evaluation.

All reviewed variants visibly retain C in the seat role and retain major chair parts. ALL_ON 3f n1/n2/n3 has broader diagonal/triangular sheets through the back opening than SOURCE_ORACLE; n2 has a narrow hanging strip beside a support. ALL_ON 8c/n1 RAW has a thin dangling strip near a leg, not obvious after POST. No obvious large-scale original-versus-capacity difference is visible in these views. Exact local joins and hidden coplanar layers remain UNKNOWN; visual fields are not inferred from numeric interface counts.

The new evidence supports investigating occupancy-layout consistency because it can change actual connectivity and the 8c FF result. It does not support replacing the classifier by unconditional all-on, treating source labels as deployment inputs, declaring the I tail resolved, or promoting the capacity backend by its edge-count gain alone.
