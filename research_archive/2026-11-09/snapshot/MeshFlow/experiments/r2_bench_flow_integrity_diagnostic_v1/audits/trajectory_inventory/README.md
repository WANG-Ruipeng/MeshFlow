# Existing complete trajectory inventory

All requested production trajectories are complete Euler50 records, not sparse snapshots. The CPU inventory loaded every NPZ and verified its actual 51×N×3×3 FP32 finite states, initial registered Gaussian with anchor writeback, all-step exact fixed corners, the free mask, and equality of the last canonical state to saved RAW. Existing trajectory hashes in receipts were checked where recorded. No model, OT, generation or scoring ran.

| Assets | Requests | Saved arrays | Use |
|---|---:|---|---|
| Curriculum COMMON_S12000 | 24 | `state[51,N,3,3]` | Same old panel, plain Native input |
| Curriculum NATIVE_S6000 | 24 | `state[51,N,3,3]` | Inherited current-panel requests |
| Current NATIVE_S6000 new inputs | 6 | `state[51,N,3,3]` | a8 / 27 / df6, two noises each |
| Current COVERAGE_FINAL | 30 | `state[51,N,3,3]` | All current paired inputs |
| Curriculum SEAM_S3000 | 24 | `state[51,N,3,3]` | Same old C/N/noise; distinct SEAM fixed mask |
| Historical COMMON / USER | 16 each | `states[51,N,3,3]`, `velocities[50,N,3,3]`, `times[51]` | Pure CC/UU single-model records |
| Historical CU / UC | 16 each | suffix `states[26,N,3,3]`, `velocities[25,N,3,3]`, `times[26]` | Intentional model switch at t=.5; exclude from single-model smoothness |

The 108 current/curriculum trajectories are sufficient for the authorized CPU analysis. Current trajectories lack separately saved network velocity, but successive states provide the realized discrete velocity `(X[k+1]-X[k])/.02`; this is not an exact recovery of the original BF16 forward output. Historical 32 pure-model records also passed every stored-velocity Euler recurrence exactly.

Source locations:

- Current manifest: `experiments/r2_bench_seat_coverage_v1/eval_manifest.json`.
- Inherited24 manifest: `experiments/r2_bench_full_seat_curriculum_v1/eval_manifest.json`.
- Full records: the corresponding `evaluation/<label>/<request_id>/raw/trajectory.npz`.
- Layout, anchor, owner-edge, known/free mask and role identity: `evaluation/<label>/layouts/<condition_id>/layout.npz`.
- Historical pure records: `experiments/r2_bench_trajectory_diagnostic_v1/evaluation/{COMMON,USER}/<request_id>/raw/trajectory.npz`; shared layouts under that experiment's `layouts/`.
- Historical crossover `registration.json` maps CC/UU directly to those original full records. CU/UC suffixes must not be mistaken for a new pure full-model run.
- Per-condition manifest rows carry separate `inference_npz`, `source_npz`, `metadata_npz`, `noise_npz`, `case_path`, hashes and query descriptors. All concrete paths and hashes are in `per_trajectory.json` and CSV.

All four current/curriculum generator files and the historical COMMON/USER files exist. Checkpoint hashes and local/global-step identity are copied from their original frozen registrations in `checkpoint_identities.json`; weights were not loaded or reread just for hashing. Current labels mean COMMON global12000, Native local6000/global18000, Coverage local1500/global19500, and SEAM local3000/global21000.

All 30 current Native/Coverage pairs have exactly the same C input hash, noise hash, layout hash, N/K and free-coordinate count. Native/SEAM share C/N and addressable noise arrays but SEAM fixes two additional corners in active slots; whole-state tensor distances would mix different contracts. All states use the original model/world coordinates, not condition-reference coordinates. Frozen L is 9.208138264587333 in both the current and historical records.

`recommended_panel.json` was an inventory-stage proposal of a8/27, old 838 good / 6b bad, f495 DEV and df6 holdout, with both noises. It is **not** the final probe registration or an authorization for extra forwards; the root PROTOCOL fixes the actual source-supervised probe panel separately. Selection is informed by previous outputs, not a new unbiased benchmark.

Reusable code: `bench_control_transfer_v1/sample.py:rollout` saves every state; `bench_full_seat_curriculum_v1/audit_all_raw_cache.py` authenticates full records; `bench_trajectory_diagnostic_v1/gpu.py:Runner.rollout` saves states and velocities; `bench_trajectory_crossover_v1/prepare.py` verifies Euler recurrence without a model. Source-informed `fixed/prediction_t*.npz`, `continuation_t*.npz` and source-owner oracle trajectories have different scientific scope and must not replace production noise-to-output evidence.
