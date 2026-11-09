# Readout Precision Path and Rollout V1 — resume

This is a fixed-weight, frozen-input experiment. Run from the independent installed WSL environment; no original historical experiments are imported. Original artifacts remain read-only. No training or new Gaussian is permitted.

The source entry point is `/mnt/e/MeshFlow-Control/tools/run_readout_precision.py`. The artifact root is `/mnt/e/MeshFlow-Control-artifacts/followups/readout_precision_path_v1`.

```bash
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B /mnt/e/MeshFlow-Control/tools/run_readout_precision.py --stage prepare
```

Use the same command with these stages in order:

1. `prepare`: verify checkpoint/input/cached output identities, freeze four observed requests. CPU only.
2. `preflight`: four original BF16 ON replay probes, eight actual-zero-addition probes. Four planned FP32 OFF fixed cells are evaluated as references here and reused later. 12 extra + 4 planned forwards; no warmups.
3. `fixed-state`: complete the remaining 44/48 FP32 fixed calls. Reuse all 48 old BF16 calls. Store only the registered 16 full FP32 layer captures; all96 new layer statistics remain.
4. `rollout`: 24 new Euler50 attempts, each on its own state, plus eight old BF16_ON caches. Actual frozen z arrays are read; seeds are metadata only. Each returned step is journaled to support CPU-only IO recovery.
5. `score`: 24 new CPU scores using the original migrated scorer, eight verified score caches. Max2 processes ×2 threads.
6. `render`: all32 outputs, four views +wire, condition-wide shared frame, inputC and separate P1 reference pages. Raw geometry unchanged. Rendering is not visual review.
7. `summarize`: CPU four-grid/paired metrics and descriptive outputs. The final Markdown report and visual-review rows require actual evidence inspection.
8. `verify`: verify closed call receipts, fixed budgets and protected dependencies; no model computation.

A completed GPU stage verifies identities and skips its completed calls. The ledger is never reset. A reserved attempt lacking its return/commit is not automatically re-executed. Preserve returned arrays and recover IO on CPU; real failed calls remain counted. Source changes after execution freeze require a recorded engineering revision, not silently overwriting hashes.

Hard caps: fixed48F, preflight16F, sampling1600F/32attempts, total1664F, cumulative GPU owner2700s. Normal completed route1260F=48+12+1200; 24 new rollouts plus8 cache. Single GPU worker. Training/backward/updates/newC/newGaussian all0. Spare caps are not extra experiment authorization.

`plan.json` defines selection, sources, zero-denominator rules and storage. `checkpoint_and_input_manifest.json` binds actual arrays and checkpoints. `budget.json` is regenerated from the durable budget/calls.jsonl and gpu_occupancy.jsonl. Re-running CPU summarize can recompute tables from immutable captures without a model.

The independent package has no Git HEAD yet; existing untracked source is preserved. No commit/push or default model/precision change occurs.