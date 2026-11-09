# Exposure-allocation experiment: manual runbook

**All registered work is complete (2026-10-25):** both 1,500-update training arms, 60 new RAW / 60 single-D0 outputs, all 270 score rows, 90 image pages and 270 actual AI_PROXY observations. Data preparation produced two valid source C/Y candidates but zero new training-ready parents under N128..256. **Do not train, sample, score or review again merely to resume this task.** Read `REPORT.md` and the stage receipts. After final sealing, `COMPLETE.json` is the authoritative closeout; older source snapshots preserve their then-current PENDING text.

Read-only final evidence authentication (no model/scoring calls):

```bash
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B experiments/r2_bench_exposure_allocation_v1/closeout.py
```

The manual stage commands below document the historical execution mechanism. They do not authorize another run. Some CPU stages were dispatched directly and their own completion receipts take precedence over stale launcher placeholders.

Use the registered WSL interpreter from `/mnt/e/MeshFlow`:

```bash
PY=/mnt/e/MeshFlow-Control-artifacts/venv/bin/python
$PY -B -m r2.pilots.bench_exposure_allocation_v1.workflow
```

This command only reads workflow state. Compare it with the stage's actual completion receipt. A COMPLETE stage is verified from its saved outputs; it is not regenerated. A live RUNNING/LAUNCHED stage continues in its detached supervisor. Do not delete attempt directories or relaunch stale/failed scientific work.

Only after checking the current status, dependencies and resources may the root operator manually launch a genuinely missing stage:

```bash
$PY -B -m r2.pilots.bench_exposure_allocation_v1.workflow --launch <stage>
```

Stages: `sample-a`, `sample-b`, `d0-a`, `d0-b`, `prepare-scores`, `score-raw`, `score-post`, `collect`. All 60 new RAW outputs must exist and verify before either D0 stage. P0 uses existing Coverage outputs; do not regenerate it. GPU resource decisions are made separately for sampling and D0. The launcher never starts a successor stage automatically.

Training audit verification is CPU-only and returns the saved receipt once complete:

```bash
$PY -B -m r2.pilots.bench_exposure_allocation_v1.training_closeout
```

Gallery/tables use `report_results` with actions `prepare`, `render`, `tables`, `visual`. These are CPU-only, but repeated rendering/scoring is still unnecessary. Inspect their completion receipts first. `visual` requires complete **actual image-review records**; do not infer or manufacture labels from numerical metrics. Keep all RAW, BASE_POST and CLEAN_FINAL_POST results and all parent groups separate.

On a scientific failure, stop additional dispatch and review the saved checkpoint, call ledger and attempt identity. There is no automatic retry or budget extension. A training `--resume` exists only for an explicitly reviewed exact safe boundary; it does not authorize recomputing completed updates. See `RECOVERY_README.md` for the full contract.

`source_archive/` is the original training snapshot. `final_source_archive/` is an additional, independently hashed source snapshot, including evaluation/reporting and the documented JSON-idempotence fix. Its `SOURCE_SNAPSHOT_ONLY` status does not mean the experiment or image review is complete. It copies no weights, large arrays or training logs, and it changes no Git index, default profile, scorer or model implementation.
