# Exposure-allocation experiment: manual runbook

Training A and B is complete: each used 1,500 updates, 12,000 forwards and 12,000 backwards, ending at global step 21,000. `training_closeout/complete.json` is PASS. **Do not train or replay these stages again.** The full evaluation is still in progress when this source snapshot is created.

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
