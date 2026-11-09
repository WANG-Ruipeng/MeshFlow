# Four-arm saved training diagnostics

Final audit: **PASS**, 662,034 checks, zero errors. All four exploration arms completed 1,000 optimizer updates from the same START. All 6,000 same-step arm-pair primary input comparisons agree. The completed-training scope contains 74,000 matched call pairs: 35,000 forward, 35,000 backward and 4,000 updates. This is not the final experiment-wide inference/accounting audit.

| Arm | Mean primary FM | Mean actual weighted auxiliary loss | Clip fraction | Weighted auxiliary gradient norm, median | Auxiliary / FM gradient norm, median | Negative gradient cosines |
|---|---:|---:|---:|---:|---:|---:|
| FM_CONTROL | 0.109428 | 0 | 1.8% | NA | NA | NA |
| REPLAY_FM_CONTROL | 0.109389 | 0.001226 | 1.8% | 0.028387 | 0.072897 | 9 / 41 |
| TRAJECTORY_REPLAY | 0.123230 | 0.519569 | 85.5% | 1.915497 | 2.822769 | 23 / 41 |
| TRAJECTORY_REPLAY_CAPPED | 0.113675 | 0.095751 | 28.0% | 0.511772 | 0.850085 | 25 / 41 |

Gradient values come from actual weighted auxiliary backward leaf hooks, observed only at step 1 and every 25 steps (41 observations per auxiliary arm). They are not differences of gradient norms. The mean gradient cosine is +0.111605 for teacher replay, -0.013744 for trajectory replay and -0.029258 for capped trajectory replay.

The cap was active at 751 / 1,000 H3 steps (75.1%). Its actual detached coefficient had mean 0.04524070, median 0.03073394 and range [0.000568566, 0.1]. All logged coefficients and actual FP32 products passed the audit. The largest FP32 loss-value excess above the Python FM sum was 1.70e-8, consistent with the registered multiplication-rounding treatment. Capping the loss value does not cap the gradient norm: the observed H3 auxiliary / FM gradient-norm ratio reached 9.3146.

All auxiliary arms selected 12 of the 16 saved trajectories, covering eight parents and 24 distinct request/t/Xt/u target identities. The other four trajectories were generated and scored but did not contribute auxiliary backward inputs. S/I/E queues contributed 334/333/333 updates; t=0.6/0.8 contributed 504/496. Original and capped trajectory replay used exactly the same 24 Xt/t/u/target identities.

The first 96 steps (1-96) and last 96 steps (905-1000) each cover all 48 queue/parent/t positions twice. These positions map to 24 distinct states. The table below gives each of those 24 states equal weight, rather than weighting duplicated queue selections again.

| Arm | Early auxiliary MSE | Late auxiliary MSE | Reduction | States with lower late MSE |
|---|---:|---:|---:|---:|
| REPLAY_FM_CONTROL | 0.014502 | 0.009767 | 32.65% | 24 / 24 |
| TRAJECTORY_REPLAY | 15.610978 | 2.054055 | 86.84% | 24 / 24 |
| TRAJECTORY_REPLAY_CAPPED | 16.704400 | 4.880052 | 70.79% | 24 / 24 |

Under actual schedule weighting, H3 MSE fell from 16.052696 to 3.753658. The t=0.6 component fell from 6.549519 to 1.487651; t=0.8 fell from 25.555874 to 6.019665. Its cap activation fell from 97.92% to 51.04%, while the coefficient mean rose from 0.0178053 to 0.0704289 and the clipping fraction rose from 8.33% to 48.96%. Thus reduced target error and less frequent cap activation did not imply progressively smaller gradients. These are fixed START-state fitting diagnostics, not proof of improved generated control or generalization.

The median input-plus-update times were 1.4278 / 1.5267 / 1.5321 / 1.5501 seconds for FM / teacher / trajectory / capped. They exclude checkpoint I/O, DEV generation, loading and idle time, and can include concurrent CPU interference. FM performed the registered 8,000 OT matches; the other three arms read all 8,000 matching cache entries each, with no new primary OT calls.

Both final PNGs were actually viewed and passed layout checks. Exact tables, source hashes and plotting receipts are in `summary.json`, `full1000_matched_learning_summary.json`, `matched_24_target_learning.csv`, and the per-step CSVs. The original frozen TR1000 learning summary and original three-arm final snapshot remain preserved. A new auxiliary summary helper initially rejected a CSV boolean string; its original source, one engineering failure and exact reader-only correction are retained under `engineering/matched_learning_bool_reader/`. No model failure or new scientific call occurred.

The repeat verifier remains actual **NOT_RUN**; its earlier 28 CPU tests do not constitute a repeat result. No repeat, CONFIRM or further training is authorized by these diagnostics.
