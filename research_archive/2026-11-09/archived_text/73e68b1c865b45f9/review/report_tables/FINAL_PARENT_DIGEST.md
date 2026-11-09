# Final DEV parent digest

All differences are candidate minus reference, in percentage points (pp). Higher S/E is favorable; lower I is favorable. Each parent retains both frozen noises; no best-noise selection. Medians are over eight parent means. All figures are read from existing comparison CSVs and checked against gate JSON plus the existing evaluation parent CSVs; no scoring or model calls.

## Severe parent I/E costs

Registered severe threshold: I increase >10pp or E decrease >10pp. Conservative bounds are preserved.

| Candidate | Reference | Parent | I delta pp | E delta pp | Finding |
|---|---|---|---:|---:|---|
| TRAJECTORY_REPLAY_500 | START | None | NA | NA | No confirmed or possible severe I/E cost |
| TRAJECTORY_REPLAY_500 | REPLAY_FM_CONTROL_500 | None | NA | NA | No confirmed or possible severe I/E cost |
| TRAJECTORY_REPLAY_1000 | START | None | NA | NA | No confirmed or possible severe I/E cost |
| TRAJECTORY_REPLAY_1000 | REPLAY_FM_CONTROL_1000 | None | NA | NA | No confirmed or possible severe I/E cost |
| TRAJECTORY_REPLAY_CAPPED_500 | START | None | NA | NA | No confirmed or possible severe I/E cost |
| TRAJECTORY_REPLAY_CAPPED_500 | REPLAY_FM_CONTROL_500 | 534b0e6e2c53a8e1a7cdfd3874ed3723 | +13.916016 | +9.813790 | Confirmed I increase >10pp; no severe E drop |
| TRAJECTORY_REPLAY_CAPPED_1000 | START | None | NA | NA | No confirmed or possible severe I/E cost |
| TRAJECTORY_REPLAY_CAPPED_1000 | REPLAY_FM_CONTROL_1000 | 534b0e6e2c53a8e1a7cdfd3874ed3723 | +17.700195 | +5.510820 | Confirmed I increase >10pp; no severe E drop |

No E decrease greater than10pp, confirmed or possible, occurs in any of these eight comparisons. The two flagged I intervals are exact (lower=upper). These flags are cost caveats from the existing gate, not newly introduced vetoes.

## Complete 1000-step parent deltas

### explore_TRAJECTORY_REPLAY_1000 versus START

| Parent UID | S delta pp | I delta pp | E delta pp |
|---|---:|---:|---:|
| 33ec1e64a02ff718faaadcdc0221eb72 | +1.842506 | +8.740234 | -1.007049 |
| 341bcae1e7f366a233d93f57729688db | -0.857724 | +1.293945 | -1.951343 |
| 534b0e6e2c53a8e1a7cdfd3874ed3723 | -3.910688 | -9.667969 | +8.127831 |
| 866e7248434968051cc7750d0afca796 | +13.592791 | -3.271484 | +2.465820 |
| 86ec9c796a6c1789e792598ed58dd1cc | -22.563386 | +0.122070 | -0.024414 |
| bdd29e651e5f6fb2b079317292bdc5d4 | -4.152489 | -1.782227 | -0.683594 |
| c1b8c2f0c4f3cc43aff7c08b060f5ed6 | -9.060312 | -2.832031 | +2.197266 |
| f59627a077dcc2566c63b8b48495261a | -16.933271 | -0.170898 | +68.427971 |

| Metric | Mean delta pp | Median delta pp | Opposite-noise parents /8 | Noise0 mean delta pp | Noise1 mean delta pp |
|---|---:|---:|---:|---:|---:|
| S | -5.255322 | -4.031588 | 1 | -2.839350 | -7.671293 |
| I | -0.946045 | -0.976562 | 2 | -3.887939 | +1.995850 |
| E | +9.694061 | +1.086426 | 2 | +9.901635 | +9.486487 |

### explore_TRAJECTORY_REPLAY_1000 versus explore_REPLAY_FM_CONTROL_1000

| Parent UID | S delta pp | I delta pp | E delta pp |
|---|---:|---:|---:|
| 33ec1e64a02ff718faaadcdc0221eb72 | +4.966446 | +9.643555 | -1.435045 |
| 341bcae1e7f366a233d93f57729688db | +0.000000 | +0.854492 | -1.951343 |
| 534b0e6e2c53a8e1a7cdfd3874ed3723 | -2.516658 | -0.537109 | +5.888274 |
| 866e7248434968051cc7750d0afca796 | -3.745472 | -2.563477 | +0.610352 |
| 86ec9c796a6c1789e792598ed58dd1cc | -15.425796 | -1.416016 | +0.415039 |
| bdd29e651e5f6fb2b079317292bdc5d4 | +6.974753 | -4.370117 | +0.878906 |
| c1b8c2f0c4f3cc43aff7c08b060f5ed6 | -6.989863 | -6.640625 | -0.610352 |
| f59627a077dcc2566c63b8b48495261a | -20.679699 | -2.661133 | +66.666667 |

| Metric | Mean delta pp | Median delta pp | Opposite-noise parents /8 | Noise0 mean delta pp | Noise1 mean delta pp |
|---|---:|---:|---:|---:|---:|
| S | -4.677036 | -3.131065 | 0 | -6.982049 | -2.372023 |
| I | -0.961304 | -1.989746 | 3 | -2.905273 | +0.982666 |
| E | +8.807812 | +0.512695 | 3 | +12.724241 | +4.891383 |

### explore_TRAJECTORY_REPLAY_CAPPED_1000 versus START

| Parent UID | S delta pp | I delta pp | E delta pp |
|---|---:|---:|---:|
| 33ec1e64a02ff718faaadcdc0221eb72 | -3.123940 | -0.878906 | -0.478348 |
| 341bcae1e7f366a233d93f57729688db | -0.857724 | +0.000000 | +0.000000 |
| 534b0e6e2c53a8e1a7cdfd3874ed3723 | -2.576426 | +8.569336 | +7.750377 |
| 866e7248434968051cc7750d0afca796 | +12.318790 | -2.124023 | +0.122070 |
| 86ec9c796a6c1789e792598ed58dd1cc | -7.239226 | +3.247070 | -5.883789 |
| bdd29e651e5f6fb2b079317292bdc5d4 | -0.306187 | -0.537109 | -0.317383 |
| c1b8c2f0c4f3cc43aff7c08b060f5ed6 | -3.261623 | -1.806641 | +0.390625 |
| f59627a077dcc2566c63b8b48495261a | -6.107131 | +0.146484 | +54.863302 |

| Metric | Mean delta pp | Median delta pp | Opposite-noise parents /8 | Noise0 mean delta pp | Noise1 mean delta pp |
|---|---:|---:|---:|---:|---:|
| S | -1.394183 | -2.850183 | 3 | +0.657796 | -3.446163 |
| I | +0.827026 | -0.268555 | 5 | -2.307129 | +3.961182 |
| E | +7.055857 | +0.061035 | 2 | +9.202918 | +4.908796 |

### explore_TRAJECTORY_REPLAY_CAPPED_1000 versus explore_REPLAY_FM_CONTROL_1000

| Parent UID | S delta pp | I delta pp | E delta pp |
|---|---:|---:|---:|
| 33ec1e64a02ff718faaadcdc0221eb72 | +0.000000 | +0.024414 | -0.906344 |
| 341bcae1e7f366a233d93f57729688db | +0.000000 | -0.439453 | +0.000000 |
| 534b0e6e2c53a8e1a7cdfd3874ed3723 | -1.182396 | +17.700195 | +5.510820 |
| 866e7248434968051cc7750d0afca796 | -5.019474 | -1.416016 | -1.733398 |
| 86ec9c796a6c1789e792598ed58dd1cc | -0.101637 | +1.708984 | -5.444336 |
| bdd29e651e5f6fb2b079317292bdc5d4 | +10.821055 | -3.125000 | +1.245117 |
| c1b8c2f0c4f3cc43aff7c08b060f5ed6 | -1.191174 | -5.615234 | -2.416992 |
| f59627a077dcc2566c63b8b48495261a | -9.853558 | -2.343750 | +53.101998 |

| Metric | Mean delta pp | Median delta pp | Opposite-noise parents /8 | Noise0 mean delta pp | Noise1 mean delta pp |
|---|---:|---:|---:|---:|---:|
| S | -0.815898 | -0.642016 | 4 | -3.484903 | +1.853107 |
| I | +0.811768 | -0.927734 | 3 | -1.324463 | +2.947998 |
| E | +6.169608 | -0.453172 | 1 | +12.025523 | +0.313693 |

Opposite noise means strictly positive delta for one frozen noise and strictly negative delta for the other. Zero is not opposite. All listed opposite signs also remain opposite under the saved conservative lower/upper bounds. CSVs include each noise delta and all parent/median bounds.

Positive mean E is concentrated: for TR1000 versus START, parent f59627a0 contributes +68.427971pp while the eight-parent median is +1.086426pp; for capped1000 it contributes +54.863302pp while the median is +0.061035pp. Against teacher, capped1000 has positive mean E +6.169608pp but negative median -0.453172pp. These descriptive distributions do not establish population generalization.

Sources: `comparisons/DEV/explore_{TRAJECTORY_REPLAY,TRAJECTORY_REPLAY_CAPPED}_{500,1000}__vs__explore_REPLAY_FM_CONTROL_{500,1000}/{gate.json,vs_START_per_parent.csv,vs_START_per_output.csv,vs_START_per_noise.csv,vs_matched_per_parent.csv,vs_matched_per_output.csv,vs_matched_per_noise.csv}` and matching `evaluation/{label}/DEV/per_parent_metrics.csv`. Exact resolved-relative paths and SHA256 are in `FINAL_PARENT_DIGEST_receipt.json`.
