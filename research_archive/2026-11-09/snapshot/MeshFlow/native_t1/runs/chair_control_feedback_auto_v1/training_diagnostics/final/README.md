# Saved training diagnostics

Status: PASS

CPU-only review; no model, backward, update, matching, or new generation calls.

| Phase / arm | Steps | FM mean | Weighted aux / FM mean ratio | Clip fraction | Hook observations | Input + update median s |
|---|---:|---:|---:|---:|---:|---:|
| explore / FM_CONTROL | 1000 | 0.109428 | 0 | 0.0180 | 0 | 1.4278 |
| explore / REPLAY_FM_CONTROL | 1000 | 0.109389 | 0.0112061 | 0.0180 | 41 | 1.5267 |
| explore / TRAJECTORY_REPLAY_CAPPED | 1000 | 0.113675 | 0.842323 | 0.2800 | 41 | 1.5501 |
| explore / TRAJECTORY_REPLAY | 1000 | 0.12323 | 4.21624 | 0.8550 | 41 | 1.5321 |

Checks: 662034; failures: 0; identical primary batch comparisons: 6000.

Auxiliary gradients are actual weighted extra-backward hook measurements. Cosines are observed only at step 1 / every 25 steps; missing observations are not zeros.

Trajectory replay uses a fixed START feedback pool; it is not an on-policy rollout of the changing model. Main-stream repeat seeds are independent while this supervision pool is intentionally shared.

These values describe the learning process. Different auxiliary state distributions make raw loss magnitudes incomparable as measures of generated-control quality. No DEV effect, recipe change, or training extension is inferred.

Files: per_step.csv, primary_samples.csv, primary_time_strata.csv, aux_time_strata.csv, gradient_observations.csv, aux_audit.csv, auxiliary_request_coverage.csv, shared_input_pairs.csv, ledger_by_step.csv, learning_diagnostics.png, summary.json. The registered capped arm additionally has loss_balance_diagnostics.png.

Actual auxiliary coefficient and nominal source lambda are separate. The fixed START pool contains 16 trajectories; consult auxiliary_coverage for how many actually contributed extra backward inputs.
