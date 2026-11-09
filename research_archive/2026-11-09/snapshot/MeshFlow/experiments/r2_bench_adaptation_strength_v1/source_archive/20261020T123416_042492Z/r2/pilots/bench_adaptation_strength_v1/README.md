# Bench initialization and adaptation strength, fixed three-arm pilot

Authorization: user approved the reviewed three-arm plan on 2026-10-20.
All outputs are isolated in `experiments/r2_bench_adaptation_strength_v1/`.
Original checkpoints, data, scorers and reports are read-only.

| Arm | Generator initialization | Backbone LR | Role/Geo LR |
|---|---|---:|---:|
| A_COMMON_LR1E5 | COMMON12000 live | 1e-5 | 1e-5 |
| B_COMMON_LR2E6 | Same COMMON12000 | 2e-6 | 1e-5 |
| C_OFFICIAL_LR2E6 | Official Bench EMA + original condition initialization | 2e-6 | 1e-5 |

All use fresh grouped AdamW and the same existing CLEAN 58-parent / 195-condition
1500-update stream, batch8/micro1, source-supervised SEAM, HYBRID.25, original FM.
No occupancy-head update, head forward, new loss or new data. Inference reuses
the frozen Bench SEAM_S15000 C-only predicted layouts.

The formal budget is 4500 updates, 36000 training sample forwards/backwards,
1056 fixed-probe forwards (A/B share step0), 224 Euler50 outputs, and 224 single
Bench D0 forwards. All RAW must be saved before D0. No checkpoint selection.
Actual main results use local1500; original global histories remain explicit.

Read-only status:

```bash
python -B -m r2.pilots.bench_adaptation_strength_v1.resume
```

CPU registration and preparation:

```bash
python -B -m r2.pilots.bench_adaptation_strength_v1.register
python -B -m r2.pilots.bench_adaptation_strength_v1.register --training
python -B -m r2.pilots.bench_adaptation_strength_v1.training --preflight-cpu
```

Dispatch requires a matching PASS gate and current resource measurements.
Start A, observe real forward/backward/optimizer peaks, then dispatch an
additional authorized arm only when resource headroom permits:

```bash
python -B -m r2.pilots.bench_adaptation_strength_v1.resume --stage train --identity A_COMMON_LR1E5 --observed-peak-mib 4500
```

For safe interruptions only, the same command accepts `--resume-training`.
Failed or unsaved neural work is not replayed automatically. Do not kill
unrelated user processes or alter batch/precision to add workers.

Comparison boundaries: B−A tests backbone LR under reset-optimizer COMMON;
C−B tests complete initial packages at equal future budget. This is not a full
factorial or proof of catastrophic forgetting. Probe improvements do not replace
C adoption, intact shape/supports, actual connectivity and geometry costs.
Existing DEV remains a development panel. No automatic Chair change, cleanup,
default switch, commit or push.
