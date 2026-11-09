# Execution / recovery

Use WSL Ubuntu, working directory `/mnt/e/MeshFlow`:

```bash
export PYTHONPATH=/mnt/e/MeshFlow/r2/src:/mnt/e/MeshFlow
PY=/mnt/e/MeshFlow-Control-artifacts/venv/bin/python
$PY -B -m r2.pilots.bench_state_schedule_v1.resume --stage status
```

Training source, all original inputs and scientific code are frozen in `training_registration.json`. Each arm owns model/optimizer/RNG, call ledger and status. Do not copy another experiment's status or `--resume` checkpoint. Final checkpoint is fixed1500. The source A0 is COMMON12000 plus fresh AdamW, not COMMON optimizer momentum.

If and only if status is `INTERRUPTED_SAFE` and call counts exactly match its checkpoint, the original authorized budget can continue without repeated calls:

```bash
$PY -B -m r2.pilots.bench_state_schedule_v1.resume --stage train --arm PERTURBED_STATE
$PY -B -m r2.pilots.bench_state_schedule_v1.resume --stage train --arm ASYNC_LEAD_SEAM
```

`TRAINING`, `FAILED`, `OOM`, pending calls or a checkpoint behind actual updates are not implicit replay permission. Inspect logs and preserve them. Do not replace the running supervisor/process.

After both training endpoints, sampling is exactly48 per arm:

```bash
$PY -B -m r2.pilots.bench_state_schedule_v1.resume --stage raw --arm PERTURBED_STATE
$PY -B -m r2.pilots.bench_state_schedule_v1.resume --stage raw --arm ASYNC_LEAD_SEAM
```

Only when all96 RAW receipts pass may D0 run, once per output:

```bash
$PY -B -m r2.pilots.bench_state_schedule_v1.resume --stage d0 --arm PERTURBED_STATE
$PY -B -m r2.pilots.bench_state_schedule_v1.resume --stage d0 --arm ASYNC_LEAD_SEAM
$PY -B -m r2.pilots.bench_state_schedule_v1.resume --stage score-raw --workers 4
$PY -B -m r2.pilots.bench_state_schedule_v1.resume --stage score-post --workers 4
$PY -B -m r2.pilots.bench_state_schedule_v1.resume --stage collect
$PY -B -m r2.pilots.bench_state_schedule_v1.resume --stage training_summary
$PY -B -m r2.pilots.bench_state_schedule_v1.resume --stage gallery --workers 4
```

CPU scoring modules hide CUDA at import and must stay in separate processes from training/sampling. Remeasure RAM/CPU/VRAM before dispatching another worker. Worker count examples are not fixed authorization to oversubscribe.

Actual visual review uses `visual_review/protocol.json` and frozen assignments. Inspect each recorded PNG, write honest `AI_PROXY` observations with image hashes and then call `visual_review collect`. A file being rendered is not proof it was inspected.

No command in this experiment changes the maintained default model or publishes to GitHub.
