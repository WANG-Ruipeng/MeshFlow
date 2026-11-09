# Idea1B fixed terminal affine CPU diagnostic

All 24 frozen FULL_S6000 requests completed; 48 original scorer calls. No model, GPU, training or new noise.

Inherited RAW adoption is used only for screening. New POST usage/body labels remain NOT_REVIEWED.

|Stage|Split|Interfaces|Screening joint|New/lost screened|Legal CF edges|Lost FF edges|New illegal FF pairs|
|---|---|---:|---:|---:|---:|---:|---:|
|AFFINE|ALL|8/24|6/24|0/0|62|0|0|
|AFFINE|TRAIN|6/12|6/12|0/0|58|0|0|
|AFFINE|DEV|2/12|0/12|0/0|4|0|0|
|AFFINE_B2|ALL|13/24|9/24|3/0|90|0|3|
|AFFINE_B2|TRAIN|10/12|8/12|2/0|83|0|2|
|AFFINE_B2|DEV|3/12|1/12|1/0|7|0|1|

The objective is fixed finite-edge distance, not the reward. One identity-start bounded Powell run uses at most 300 evaluations/request; a budget limit is not proof of no better affine solution. Shared C anchors eliminate infeasible affine degrees of freedom.

Affine displacement is capped at 3.5%L on retained F representatives and all original pre_geo F corners. B2 is exactly the old decoder with 2%L relative to transformed pre_geo; the separate total displacement includes original GEO representative consolidation and is NOT claimed <=2%L.

An invertible global affine preserves exact F/F intersections. Classification changes under AFFINE alone may reflect FP32 geometry/predicate tolerances; all actual pair differences are retained rather than called self-intersection repair. Additional B2 changes are separately paired in vs_affine_ledger.json.

Execution wall time before report: 106.53s, one CPU process, one BLAS/OMP thread. Full rows: per_output.csv; before/after endpoint reachability and objective trace per request; fixed-selection CPU images in gallery/. No old assets were modified.
