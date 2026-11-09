# Training diagnostic review

| Arm | FM last100 mean | weighted seam/FM median | final boundary norm | boundary grad median | boundary update median | marked C-corner mean |
|---|---|---|---|---|---|---|
| A_FM | 0.10906756525859236 | 0.0 | None | None | None | None |
| B_BOUNDARY | 0.1090501472807955 | 0.0 | 0.01398006733506918 | 0.00023229133512359113 | 6.093967203923967e-05 | 0.23596414906159044 |
| C_SEAM | 0.10911144084326224 | 0.010025581840993957 | None | None | None | None |
| D_BOUNDARY_SEAM | 0.10909317591984291 | 0.00997984224360382 | 0.015337633900344372 | 0.0002427039507892914 | 6.141310586826876e-05 | 0.23596414906159044 |

All four arms have 1000 exact ordered UPDATE_DONE rows and 8000 sample diagnostics. Shared and coupled inputs, supervised target identities and B/D C-only marker statistics match.
Lower training loss does not establish generated attachment quality. Existing marked corners are a varying subset, not an all-known constant feature.
Full details, quantiles, first/last100 means, gate/formula checks and source hashes are in review_training_diagnostics.json.
Final gallery must independently cover 160 P1 + 40 P5 outputs and 412 PNG files; current START preview is not final.
