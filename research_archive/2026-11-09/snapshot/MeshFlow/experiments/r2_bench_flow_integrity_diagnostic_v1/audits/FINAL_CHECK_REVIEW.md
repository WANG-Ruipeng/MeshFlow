# Independent final artifact check

Status: **PASS**. The CPU-only receipt is [final_check.json](final_check.json); reproducible audit code is [final_check.py](final_check.py). This audit performed no model forward, backward, OT, geometric rescoring, or checkpoint load. It checked 245 small-source/cached-artifact hashes, 1,286 summary/delta values, and 12 independent saved-prediction reductions.

- Frozen forwards: COMMON 60, Native 60, Coverage 60, SEAM 20, total **200**. Every registered call has exactly one START and one COMPLETE entry; four dispatch exits succeeded. No unresolved attempt/failure is present.
- Backward, optimizer updates, head forwards, new rollout and D0: **0**. CPU production OT: **16**. Saved complete trajectories analyzed: **108**. Generation noise was reused; the protocol separately authorized and produced **12 diagnostic epsilon arrays**.
- All 40 prediction NPZ files match their completion hashes. Registered original sources, case/noise files, transported pair maps and analyzed outputs match recorded hashes. The `COVERAGE_FINAL` trajectory label and `COVERAGE_S1500` probe label refer to the same checkpoint path and SHA-256.
- Exact executed probe code asserts unchanged CPU/CUDA RNG and unchanged generator tensor hash before writing COMPLETE. The audit verifies this code's attempt hash and all completion records; it does not independently reconstruct unsaved RNG tensors. Original checkpoint SHA/state was checked by the loader, with current small sidecars and file sizes checked here, without reading the large checkpoints again.
- The fixed four-request risk proxy is 0.002208–0.003442, below 0.005. No Euler100/D0 branch ran. This does **not** exclude numerical integration error or establish a converged reference.

## Numeric interpretation checked

The independent reduction reproduces the equal-time → equal-noise → equal-condition → equal-parent calculation and all Native paired parent deltas. Three Native models use the same 60 saved bridge inputs. Their six-parent all-free velocity MSE is COMMON 0.850953, Native 0.855071, Coverage 0.809181; both-free adjacent-source-pair differential MSE is 0.359919, 0.357348, 0.325491. These aggregates are descriptive of this source bridge panel.

Coverage's largest fitting gains occur on the newly trained complete-seat parents:

| Parent | All-free velocity MSE, Native → Coverage | Both-free differential MSE | One-fixed differential MSE |
|---|---:|---:|---:|
| 27eb79 | 0.310596 → 0.132247 | 0.333979 → 0.202796 | 0.151993 → 0.083636 |
| a8e987 | 0.252702 → 0.157050 | 0.287492 → 0.246928 | 0.313529 → 0.221223 |
| 6b894c | 0.152139 → 0.153970 | 0.236335 → 0.233363 | 0.184811 → 0.194307 |
| df6fd3 | 0.246210 → 0.256882 | 0.283558 → 0.284612 | 0.225232 → 0.231363 |

Thus the new-parent bridge fitting improved while the held-out df6 and old difficult 6b do not share that pattern. This is compatible with the prior free-generation quality costs; reduced supervised error is not sufficient evidence of improved free completion. The unusually large fd355 all-free MSE (about 4.02) materially affects the six-parent mean, so the per-parent table should remain visible.

SEAM uses only two parents and a target-derived source-owner layout. Its lower aggregate MSE must not be ranked against the six-parent Native totals as a same-input architecture result. Fixed corners are projected to zero for motion comparisons, and their raw network outputs are logged separately. Shared-source errors use the filtered legal source-edge endpoint pairs, not all generated edges.

Metadata limitation: `split` in the probe analysis CSV is `UNSPECIFIED`, because the frozen package supplies `eval_group`. The exact parent-to-`eval_group` mapping is retained in `final_check.json`; group interpretation must join that mapping rather than treating the CSV column as a valid grouping.

This audit changes no method, metric, checkpoint or original output, and does not authorize further model calls.
