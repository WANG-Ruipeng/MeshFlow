# Independent validation and DEV rejection detail

The independent NumPy replay in `../cpu_binding_independent_review.json` checked all 56 final root maps and coordinates against 1,511 recorded candidates and 104 accepted candidates. It reports PASS for actual original-cluster membership, sequential target coordinates, full-member PRE_GEO displacement, C bytes and source-face identities. The newly tightened current-root C guard rejects nine hazardous queued candidates in this experiment. This is an exercised safeguard here, unlike the old 112-cell panel where that branch was not reached.

The DEV lateral-edge funnel is more specific than “nearby but cannot connect”:

| Parent / cap | Side × bulk-edge pairs | Axis rejected | Representative gap rejected | Full-member displacement rejected | Geometric candidates before later shape guards |
|---|---:|---:|---:|---:|---:|
| 608 / 1.5% | 2,232 | 1,750 | 477 | 5 | 0 |
| 608 / 2.0% | 2,232 | 1,750 | 457 | 25 | 0 |
| 610 / 1.5% | 3,288 | 2,656 | 632 | 0 | 0 |
| 610 / 2.0% | 3,288 | 2,656 | 632 | 0 | 0 |

For 2d, there are 19,712 pairs per cap. 7,747 involve an already closed seam lateral edge and are ineligible; 8,848 fail the axis check; six have no permissible endpoint identity mapping. The remaining funnel gives six geometric candidates at 1.5% and twelve at 2.0%. Their actual sequential rejections are two collapsed-vertex-ID cases, three/nine new-or-worse small-angle cases, and one new-or-worse q case. No lateral candidate is accepted. The row counts above are candidate pairs, not independent objects or successful interfaces.

All twelve RAW slots corresponding to 2d's three source-alias edges across four noises survive POST. Exactly one slot, n0 / edge 12, participates in two illegal F/F pairs; all twelve have zero illegal C/F pairs. These observations do not support calling every alias a collision or claiming aliases caused the failure to reach bulk. The source aliases are not true Null labels and were never used by the binding selector.

The main numerical report remains the original immutable artifact. This addendum reports later independent validation and detailed aggregation only; it changes no geometry, scoring or experimental selection.
