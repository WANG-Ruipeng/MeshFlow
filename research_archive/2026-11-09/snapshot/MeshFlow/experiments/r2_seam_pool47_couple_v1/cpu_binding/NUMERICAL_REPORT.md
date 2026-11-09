# B_LONG_LAYOUT CPU binding: numerical result

Completed 56/56 cells in 65.63 s with 2 CPU workers; zero model calls, zero new noise/query draws. All 196 protected inputs verified. Original 124 numeric metric fields retained. Visual review remains pending.

The 1.5% and 2.0% caps add local TRAIN bulk connectivity but do not create a DEV output with >=30% reachable bulk. Point aliases alone are not interface success.

| Cap / split | Bulk >=30% before→after | Mean bulk reach | ΔFF pp | ΔI pp | Accepted tip / lateral | New / removed illegal FF pairs |
|---|---:|---:|---:|---:|---:|---:|
| 015 / TRAIN | 12→14/16 | 62.4904% | -0.1950 | -0.0061 | 18 / 10 | 6 / 15 |
| 015 / DEV | 0→0/12 | 0.3910% | +0.6812 | +0.0122 | 21 / 0 | 19 / 6 |
| 020 / TRAIN | 12→14/16 | 62.4904% | -0.1950 | -0.0061 | 18 / 10 | 6 / 15 |
| 020 / DEV | 0→0/12 | 0.3910% | +0.7347 | +0.0081 | 27 / 0 | 20 / 7 |

All 56 cells preserve every old legal C/F and F/F edge. No additional source face is deleted, and no new illegal C/F pair is introduced. These safeguards do not prohibit new illegal F/F pairs: DEV has 19 and 20 new pairs at the two caps. Mean CF/I improvements elsewhere cannot cancel this cost.

TRAIN additions crossing 30% are 73/n0 and 73/n3. 73/n1 remains 21.89%; 85/n0 remains zero. DEV 608 and 610 remain zero reachable bulk; 2d/n0 and n2 have only 0.983% and 3.709%. Both DEV caps accept point aliases only, no complete lateral-edge candidate survives.

DEV has 16/4/4 activated and retained seam faces per 2d/610/608 request, not eight per object. All 96 retained tips are currently FREE; zero C_ANCHORED tips is an observed count, not a skipped category.

| Parent | Tips | Nearest free bulk vertex ≤2%L | Finite bulk edge ≤2%L | Distinct tip-cluster minimum motion ≤2%L |
|---|---:|---:|---:|---:|
| 2d08a64e | 64 | 64 | 64 | 63 |
| 608b46b0 | 16 | 16 | 16 | 7 |
| 610b1621 | 16 | 13 | 15 | 13 |

Nearest point distances can be zero because the seam tip already shares a bulk boundary vertex. They do not establish a shared edge. Minimum distinct-cluster motion excludes the already identical vertex and additionally includes every original PRE_GEO cluster member; its denominator therefore answers a different question.

The three 2d source aliases produce 12 raw slots across four noises. 12 survive original POST. Their source-alias designation is not a true Null label. Detailed illegal-pair and legal-free-component records are in dev_diagnostic/alias_slot_fates.csv and alias_illegal_pairs.csv; this association is not evidence that aliases caused failed bulk connection.

The full threshold funnel separates closed lateral edges, axis mismatch, C/identity restrictions, representative gap and original-corner displacement. The finite-segment and endpoint distances use finite geometry, not infinite lines. Supplementary side-to-vertex observations add a missing requested diagnostic without changing, rerunning or selecting binding cells.

Interpretation: the saved B_LONG_LAYOUT endpoint benefits locally on TRAIN; this fixed binding does not solve DEV seam-to-bulk continuity. Widening from 1.5% to 2.0% adds DEV aliases and FF cost with essentially unchanged reach. No model or default is promoted by this CPU result.
