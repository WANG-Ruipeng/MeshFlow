# Control MVP48 — Session B

Status: COMPLETE

Actual optimizer updates: 2500 / 2500.
Executed phases: {"T1": 1000, "Geo_shared": 500, "Geo": 500, "Full": 500}

Implementation: one variable-N Native model, original Geo/Edge/L5 functions, effective batch8/microbatch1.
Official Chair EMA only. Geo/Full restore the same complete shared step500 model, AdamW, RNG and data cursor.
Formal GPU concurrency is one under /tmp/wrp_meshflow_jedge5_diag_gpu0.lock; CPU numerical threads are capped at two.
Timing and GPU allocated/reserved peaks are phase scoped; nvidia-smi global/process observations are separate.
Concurrent observations are not isolated compute benchmarks.

Historical BF16 permutation boundary remains FAIL: atol=.015625, rtol=.03125; 10/1008 coordinates; max_abs=.064453125.
Evidence: native_t1/runs/geometry_context_revised_20260930_153000/gates.json. No threshold changed or repair claimed.

CPU checks: training/cpu_tests.json (synthetic fixtures only, zero optimizer steps).
Real-data/official-weight prechecks: PASS.
Training/checkpoint implementation has not been end-to-end exercised when dependencies are blocked.
Checkpoints and sidecars: checkpoints/. Ledger: training/training_log.csv and training/update_events.jsonl.
Unified API: experiments.mf_h1_local0.control_mvp48.model_mvp.load_mvp_checkpoint(path, device)
Returns (model, actual_metadata, hashes). Model construction/inference must not be reimplemented by Session C.
Inference helpers: velocity_mvp and integrate_mvp_step. The caller owns its registered sampling budget and GPU lock.

No TRAIN_READY is published unless all three final endpoints and strict reloading succeed.

Recovery before any attempted GPU precheck/training:

```bash
cd /mnt/e/MeshFlow && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.train_run --official-checkpoint /ABSOLUTE/PATH/TO/CHAIR/last.pt
```

After any numerical failure or optimizer attempt, inspect status.json and update_events.jsonl first.
This runner refuses automatic restarts; no failure replay budget was granted. Completed checkpoints remain preserved.
