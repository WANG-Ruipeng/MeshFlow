# R2 USER data scaling: finite local pilot

This directory is the isolated implementation for `R2_USER_DATA_SCALING_12H_V1`. All run artifacts belong to `experiments/r2_user_data_scaling_12h_v1`. Historical data, scorers, weights and outputs remain read-only. The task clock in `registration.json` remains authoritative across restarts.

The intended comparison is nested 32-parent USER_SMALL versus an actual 128–256-parent USER_LARGE_M pool, from the same official Chair generator EMA and fresh AdamW. Both use USER SEAT/BACK FULL/HALF, R2 mixed augmentation and HYBRID from step zero. The maintained default is not changed. No B, READOUT, RBF, new loss, student or repeated denoising is added.

**Current result: DATA_SCALE_BLOCKED.** Actual source qualification retained 99 independent parents / 286 conditions. The new DEV panel uses 32 parents / 64 conditions, leaving 67 training parents / 194 conditions, 61 below the task minimum of 128. The nested 32-parent SMALL draft is retained only as data evidence. Task section 2 therefore stops before initialization, preflight, training or generation. No new weights or model outputs exist for this scaling experiment.

**GPU implementation status: IMPLEMENTED_NOT_GPU_VALIDATED.** The training, sampling, D0 and scoring adapters are prepared but have not run on this experiment. Only finite CPU preparation, input review and controller-gate checks were executed. Read the data freeze, ledger and report for actual completion; this block is not a negative model result.

## Status and bounded verification

Use the existing WSL environment; no package installation or download is required:

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B r2/pilots/user_data_scaling_12h_v1/resume_all.py
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B r2/pilots/user_data_scaling_12h_v1/resume_all.py --stage verify
```

No arguments means read-only `status`. `start` is an alias of `run`; both enforce the same data block and original clock. `verify` hashes source and frozen manifest files, with a 16 MiB per-file and 256 MiB total cap; it saves a verification receipt. Neither path imports Torch, runs OT, creates new noise, calls a model, or rehashes multi-gigabyte weights. A limited hash pass does not mean the data gate passed or that GPU execution was validated.

## Required data handoff

`data/data_freeze.json` must contain:

```json
{
  "status": "FROZEN",
  "train_manifest": {"path": "/mnt/e/MeshFlow/experiments/r2_user_data_scaling_12h_v1/data/train_manifest.json", "sha256": "..."},
  "panel_manifest": {"path": "/mnt/e/MeshFlow/experiments/r2_user_data_scaling_12h_v1/data/panel_manifest.json", "sha256": "..."},
  "large_label": "USER_LARGE_256",
  "train_parent_count": 256,
  "dev_parent_count": 32
}
```

The count and LARGE suffix must reflect verified independent parents, never repeated variants. The train manifest uses `r2_user_scaling_data_v1`, with nested `pools`, `parents` and exact original-face `tasks`; `training_data.py` defines the schema and exact C/free checks. The panel has 64 `conditions`, two distinct C per new parent, with `parent_id`, `condition_id`, `region_type`, `tier`, `N`, `K`, `condition_source`, `condition_source_sha256`, structure and quality metadata. The evaluation adapter additionally excludes historical conditional-training parents and the old six parents.

`DATA_SCALE_BLOCKED`, missing hashes, fewer than 128 LARGE parents, missing 32 new DEV parents, train/DEV overlap or an inconsistent nested subset stop before GPU dispatch. There is no force, continue-through-block, auto-expansion, automatic repair, deadline extension or substitute-old-data option. A blocked draft manifest can still document the real available parents and conditions.

## Finite execution and recovery

After genuine data qualification, the same entrypoint can execute the authorized chain:

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B r2/pilots/user_data_scaling_12h_v1/resume_all.py --stage preflight
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B r2/pilots/user_data_scaling_12h_v1/resume_all.py --stage run
```

`preflight` freezes evaluation inputs and actual Gaussian arrays, creates the common initialization, prepares the 16 fixed FM fixtures and runs the task's single authorized preflight (at most 160 F, 128 B, 16 isolated updates). It discards that update state. A shared 6000/4000 budget is then selected once using measured conservative update time, including OT wait, plus explicit checkpoint/reload/diagnostic allowances. The allowances are recorded, not presented as a new benchmark. If even both 4000-step arms cannot fit before the original nine-hour training cutoff, it writes `BUDGET_BLOCKED` and stops. `run` performs this preparation if absent, but cannot enter formal training before data and budget locks pass.

Formal training rotates USER_SMALL then USER_LARGE_M at 2000, 4000 and, if registered, 6000. Each arm restores its own model, AdamW, RNG and input cursor. `latest` and fixed 4000/final checkpoints are owned by `training.py`; no best-checkpoint selection occurs. At the training cutoff a registered 6000 run may use both previously saved fixed4000 exports only with a `TIME_BUDGET_FALLBACK` receipt. Extra consumed updates remain recorded. No common endpoint means `TRAINING_PARTIAL`.

All GPU phases are separate child processes. Resources are recorded before and after each dispatch. One GPU worker is a task requirement; the initialized generator and official D0 never coexist in this controller. The same `gpu_owner.lock` protects sampling/D0 and the training adapter. There is no additional memory probe or benchmark. CPU OT/scoring retain the registered two-worker/two-thread limits.

After training, the chain binds the fixed exports and frozen historical USER_MIX, samples every registered request with unchanged Euler50, releases each generator process, runs one resident official D0 across all pending RAW, then original GEO_C and source-C restoration/free orientation. The original twelve historical RAW/D0 proposals may be reused only after their identities pass. All RAW finish before D0 starts. Canonical/raw evidence and the final source-oriented result are retained separately. Scoring uses original full-pair CF/FF/I, all eight A-VMR/PMR settings and the actual-index legal/directed connection diagnostics. Rendering uses shared bounds and four views including wireframe.

Each child writes `controller/dispatch/<stage>/started.json`, a log and a completion receipt. Completed exact-identity stages are skipped. Failed, incomplete or changed-identity stages stop; the controller never changes an attempt name to silently retry. The training adapter itself permits exact safe-checkpoint restoration, but an interrupted controller receipt requires explicit engineering inspection before reconnecting it. Preserve the failed receipt and consumed-call ledger. Do not erase an attempt to replay scientific work.

The original deadline stops new scientific tasks and terminates only this controller's owned process group if necessary; unrelated user processes are never killed. No command commits, pushes, switches defaults, changes the model, or launches a follow-up experiment.

## Actual visual review and aggregation

The renderer creates the full gallery and an **unreviewed** CSV template. The controller then stops at `WAITING_ACTUAL_VISUAL_REVIEW`; it cannot view pictures or manufacture review labels. The operator/agent must inspect the actual images, save all registered rows to `evaluation/gallery/AI_PROXY_review.csv` (including explicitly unreviewed rows), and add `output_review_receipt.json` beside it:

```json
{
  "status": "COMPLETE",
  "review_kind": "AI_PROXY",
  "review_file": "/mnt/e/MeshFlow/experiments/r2_user_data_scaling_12h_v1/evaluation/gallery/AI_PROXY_review.csv",
  "review_file_sha256": "...",
  "actually_reviewed": 840,
  "numeric_ranking_seen_before_review": false
}
```

`840` applies only to the full three-model × 140-request × RAW/POST panel; record the actual count. Accepted viewed-row statuses are `REVIEWED`, `REVIEWED_AI_PROXY` and `REVIEWED_HUMAN`; each needs observations in `notes` and the actual image hash. `PARTIAL` is allowed only with honest reviewed counts and remaining `NOT_REVIEWED` rows. It is not a completed visual evaluation. Do not call an AI review human or blind when the mapping/rankings were known.

Once review observations are saved, complete the objective aggregation within the original window:

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B r2/pilots/user_data_scaling_12h_v1/resume_all.py --stage aggregate
```

Per-output, per-condition, per-parent, noise-paired and stratified CSVs remain separate for NEW32 and OLD6. The controller does not write a scientific verdict. The final report must decide whether LARGE merits expanded coverage using all preserved costs and actual observations, state partial/blocked work, and stop without automatic promotion.
