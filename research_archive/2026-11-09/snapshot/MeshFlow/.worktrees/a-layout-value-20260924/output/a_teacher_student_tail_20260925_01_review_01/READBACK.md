# Independent saved-array readback

PASS: 8 fixed pairs x 9 registered lambdas = 72 saved geometries. All 16 teacher/student endpoint arrays match their original saved u/r/vertices exactly. Field identity, face connectivity and file/typed-array hashes agree with the original endpoint ledger.

Minimum angle, linear p05, all per-face areas, area totals, and direction determinants were independently recomputed from saved arrays. No radius queries, network forward, training, GPU calls, Monte Carlo sampling, exact intersection audit, or new output selection were performed.

All 72 have positive direction determinants and no recorded area degeneracy; 37/72 pass both angle conditions. These conditions do not certify full geometric validity.

| UID / target | first failed sampled lambda | failed sampled lambdas | final min / p05 margins (deg) | teacher bottom26 retained |
|---|---:|---|---:|---:|
| 7f1be3e7 / old | None | [] | +1.106597 / +1.332672 | 24 |
| 7f1be3e7 / margin | None | [] | +1.726461 / +0.890152 | 21 |
| 2379915c / old | 0.375 | [0.375, 0.5, 0.625, 0.75, 0.875, 1.0] | -0.263871 / -0.098375 | 24 |
| 2379915c / margin | 0.75 | [0.75, 0.875, 1.0] | -0.276827 / +0.320662 | 24 |
| 8c919e99 / old | 0.125 | [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0] | -0.046446 / -0.040201 | 26 |
| 8c919e99 / margin | 0.375 | [0.375, 0.5, 0.625, 0.75, 0.875, 1.0] | +2.705824 / -0.286813 | 24 |
| 736820a8 / old | 0.125 | [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0] | +0.016937 / -0.007723 | 26 |
| 736820a8 / margin | 0.625 | [0.625, 0.75, 0.875, 1.0] | +0.512854 / -0.071287 | 24 |

The p05 definition is the linear interpolation of zero-based sorted ranks25 and26, with weights0.45 and0.55. The bottom26 membership set is recorded separately; it does not include all p05 support faces. Face identities can change along the path.

The first failed lambda is a statement about the fixed nine-point diagnostic grid, not the exact continuous first crossing or a universal admissible network error. Angle trends can be nonmonotonic. Global RMS and local tail enrichment are descriptive: neither alone certifies quality or identifies causal vertices. The original qualitative conclusion is consistent with the saved data, provided it is stated as lack of a demonstrated quality guarantee from global error, not the mathematical impossibility of any norm/geometry-dependent bound.

The interpolants are diagnostic geometries, not additional method outputs. Existing endpoint radius identity was verified by exact saved-array comparison; intermediate radii were not independently queried in this review. Full validity is NOT_TESTED for these interpolants. No scientific performance claim follows from their angle pass counts.

Evidence: READBACK.json and readback.py. Original run files were unchanged.
