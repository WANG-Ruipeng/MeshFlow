# Manual continuation and recovery

**Final state, 2026-10-25: COMPLETE.** Both training endpoints, all 60 RAW and 60 D0 outputs, 270 score rows and 270 actual image observations have finished. No task remains to relaunch. Verify the saved evidence with `python -B experiments/r2_bench_exposure_allocation_v1/closeout.py`; this performs no model calls. Current results are in `REPORT.md`. The following dispatch/recovery contract is retained as execution documentation, not an instruction to repeat completed work. No new training, cleanup, model promotion or Git action is authorized by this file.

This experiment has two independent, authorized 1,500-update branches from the same Coverage global-step-19,500 checkpoint. Both endpoints are local step 1,500 / global step 21,000. Each arm has 12,000 training forwards, 12,000 backwards and 1,500 optimizer updates. No extra probes, heads, training arms or replay are authorized by this document.

## Current execution contract

The two initial training jobs were dispatched through detached WSL supervisors and both exited successfully. Training had no failed or pending scientific calls and no replay. Do not restart training merely because the interactive terminal or an agent turn ended.

Run commands from `/mnt/e/MeshFlow`, using `/mnt/e/MeshFlow-Control-artifacts/venv/bin/python`. The examples below abbreviate this interpreter as `python`.

Read status without starting work:

```bash
python -B -m r2.pilots.bench_exposure_allocation_v1.workflow
```

The authoritative training states are `training/A_50_50/status.json` and `training/B_75_25/status.json`. The supervisor state, PID plus creation timestamp, command, stdout and traceback are under `workflow/<job>/`. A bare PID is insufficient proof that the original process is still alive. `STALE_PROCESS_REQUIRES_INSPECTION`, `LAUNCH_INCOMPLETE_REQUIRES_INSPECTION` and `STOPPED_REQUIRES_INSPECTION` never authorize relaunch.

## Start only genuinely missing stages

The following are manual stage names, not a script to run blindly. Each invocation launches exactly one stage. Inspect status and dependencies first; `--launch` returns an existing attempt's status without rerunning it. The launcher does not schedule later stages or decide concurrency.

| Stage | Manual command suffix | Prerequisite |
| --- | --- | --- |
| A training | `--launch train-a` | Already dispatched; do not replay. Root-issued matching A GO required. |
| B training | `--launch train-b` | Already dispatched; do not replay. Root-issued matching B GO required. |
| A fixed evaluation | `--launch sample-a` | A endpoint complete; fresh resource review |
| B fixed evaluation | `--launch sample-b` | B endpoint complete; fresh resource review |
| A official D0 | `--launch d0-a` | All 60 new RAW outputs verified; fresh resource review |
| B official D0 | `--launch d0-b` | All 60 new RAW outputs verified; fresh resource review |
| Freeze scoring | `--launch prepare-scores` | Scoring implementation ready and inputs fixed |
| RAW scoring | `--launch score-raw` | Both RAW panels complete; scoring frozen |
| POST scoring | `--launch score-post` | Both D0/POST panels complete; scoring frozen |
| Aggregate | `--launch collect` | Both scoring stages complete |

Append a suffix to:

```bash
python -B -m r2.pilots.bench_exposure_allocation_v1.workflow
```

Sampling uses the frozen 30 requests per new arm and Euler50, with no new queries or noise. P0 is read-only cache reuse. Every new D0 waits for all 60 RAW results and uses the original single official D0 plus GEO_C/source orientation. SHAPE, RELDEF and RAW-only GEO are not substituted. GPU sampling, D0 and training require separate resource measurements; the two-worker training observation is not an inference concurrency authorization.

## Failure handling

1. Stop issuing additional jobs. Preserve the current process and its logs; do not kill unrelated GPU processes.
2. Inspect the supervisor and scientific status, call ledger, complete checkpoint receipt, optimizer/RNG identities, exposure log and pending-call state. Do not delete a workflow directory to bypass its attempt marker.
3. A failed or partially completed scientific update cannot be silently replayed. The initial full checkpoint and an older rolling checkpoint do not by themselves authorize recomputation.
4. The training implementation supports explicit `--resume` only at an exactly matching safe saved boundary. This is an engineering capability, not permission to resume automatically. Any actual interruption requires a documented review confirming checkpoint, call ledger and logs share the same cursor and that no update/call would be replayed. Non-boundary failure requires a new decision.
5. Keep the failed attempt and its hashes. No implicit fallback, smaller batch, precision change, budget extension, new checkpoint selection or retry is permitted.

## Artifacts and archive scope

- `training_registration.json`, its matching `registration.json`, the fixed manifests and schedules remain authoritative.
- Each arm keeps rolling `checkpoint_latest.pt` at the prescribed 250-step boundary and its immutable final `checkpoint_step1500.pt` plus `generator_step1500.pt` and receipts. Do not delete an active recovery checkpoint.
- `source_archive/manifest.json` verifies copies of every source file in the frozen training registration and the CPU-validated manual launcher. It contains no weight copies, dataset copies or running-state snapshots. Evaluation code is frozen separately before its first model call and is not implicitly certified by the training archive.
- `dispatch/A_first25_gpu_observations.json` and `dispatch/AB_steady_observations.json` record actual resource observations. WDDM per-process memory is unavailable; reported peaks are aggregate device memory. Short concurrent throughput measurements must not be reported as isolated full-run compute benchmarks.

This file changes no default model, data, scientific source, registration, Git state or training process.
