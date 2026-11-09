# Independent cached layer audit

Status: PASS

- Passed cells: 50/50 (44 model-stage cells and 6 source references).
- Independent checks: 1270325.
- Frozen source files: 210 checked before and after.
- Queries: 102400; finite query-triangle pairs: 17119232.
- No model, generation, training or original quality scorer calls.
- All original dup masks are independently reproduced when present.
- 85b/n7 retains its registered meaning: POST interface loss with stable RAW adoption.

Method: Independent dominant-axis finite-triangle distance/ray calculation for all 2048 queries x every free face; exact adjacency, sidewall, signed-gap clusters and old dup parity recomputed.

- Result-selected explanatory subset; not a population success-rate estimate.
- Normal-line clusters are geometric observations, not semantic layers.
- Source normal side is not inside/outside.
- Observed exact adjacency may include nonmanifold or illegal contacts; not the previous valid-edge graph.
- References are local thickness context only, not unique-GT success labels.
- This numerical audit does not itself perform visual review or certify semantic structural thickness.
