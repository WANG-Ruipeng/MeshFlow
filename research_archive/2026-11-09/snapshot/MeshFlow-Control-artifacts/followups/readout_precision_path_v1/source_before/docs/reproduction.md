# Reproduction and execution contract

Commands below document implemented entry points. They do not assert that a run has finished and do not authorize extra experiments. Keep all weights, datasets, matching caches, meshes and logs outside the source checkout. The original repository is an import source only and remains read-only.

## Environment and paths

Use a compatible CUDA build of PyTorch 2.7.x for the GPU stages. CPU preparation and evaluation also require the dependencies in `pyproject.toml`. For an already provisioned environment:

```bash
python -m pip install --no-build-isolation --no-deps -e .
python -m pytest -q -m "not gpu"

# Replace these placeholders with your own external paths.
export LEGACY_ROOT=/path/to/read-only/MeshFlow
export ARTIFACTS=/path/to/MeshFlow-Control-artifacts
export START=/path/to/chair_hybrid_start_total5000.pt
export RUN="$ARTIFACTS"
export LEDGER="$ARTIFACTS/audit/budget"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

This offline install does not fetch missing dependencies. Configure paths explicitly; do not append the legacy checkout to `PYTHONPATH`. General public commands can be invoked from an installed package, while `tools/` and `configs/` examples assume the source checkout as the working directory.

## Inspect and audit the external START

```bash
python -m meshflow_control inspect --checkpoint "$START"
python tools/import_start.py --checkpoint "$START" \
  --out "$ARTIFACTS/audit/start_import.json"
```

`inspect` reads checkpoint metadata on CPU. `import_start.py` performs the strict CPU model/optimizer import and records parameter identity; it runs no forward and initializes no CUDA. It does not make a new copy of the large checkpoint. The loader pins the file and model-state SHA256 listed in the README, the original schema and the total-5000 optimizer state.

## Import data and freeze the paired stream

```bash
python -m meshflow_control prepare \
  --source-root "$LEGACY_ROOT" --out "$ARTIFACTS/data"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m meshflow_control.data.stream \
  --data "$ARTIFACTS/data" --out "$ARTIFACTS/training_stream" \
  --precompute --workers 4
```

The public importer expects these historical asset directories beneath the read-only source root:

- `native_t1/runs/control_mvp48_v1`
- `native_t1/runs/chair_domain_control_v1`

A custom layout can use `tools/import_legacy_data.py --source-mvp ... --source-chair ... --out ...`. Import copies necessary arrays and produces a portable manifest; runtime training uses the new imported data only. It does not execute legacy modules.

The importer and stream preparation are separate stages. `prepare` does not run OT. The second command freezes the exact 1,000-batch plan and 8,000 matching maps, preserving original matching mathematics. The CPU worker count can be selected from 1 to 8 after measuring resources; it changes scheduling, not seeds or samples. Four workers with one BLAS thread each were used for the recorded preparation.

Required external evidence includes:

| Artifact | Purpose |
| --- | --- |
| `data/train_manifest.json` | 32 parents, 453 READY conditions and relative data paths |
| `data/import_receipt.json`, `data/source_identity.json` | Original dependency hashes and import checks |
| `training_stream/train_plan.json` | Frozen per-example identities and random streams |
| `training_stream/paired_inputs.json` | Actual shared input/label hashes |
| `training_stream/ot_cache/` | Reserved/completed CPU matching evidence |
| `training_stream/precompute_receipt.json` | Actual attempts, returns, cache hits and elapsed time |

A completed map may be reused; a recorded attempt without a complete map blocks silent rematching. Do not remove its record to force a retry. CPU tests and these receipts establish data preparation, not model quality.

## Register the fixed evaluation inputs

The separate CPU API migrates the eight original DEV conditions and verified START caches, freezes NEW noise 4/5 and the four P2 controls, and writes `data/evaluation/requests.json`:

```bash
python -c 'import os; from meshflow_control.evaluation.reference import prepare_panels; print(prepare_panels(os.environ["LEGACY_ROOT"], os.environ["ARTIFACTS"]))'
```

It requires the retained `native_t1/runs/chair_capped500_half_v1` manifests, reference assets and their explicitly referenced START cache files. It reads the historical 0..3 noise identities only to reject collisions; it does not score historical candidate methods or load their weights. The resulting evaluation inputs/references are copied to the external artifact directory and do not require legacy Python at runtime. If the required retained assets are unavailable, this exact registered study is blocked; a replacement panel would be a different experiment.

## Validate configurations and preflight

```bash
python -m meshflow_control check-config --config configs/a_baseline.yaml
python -m meshflow_control check-config --config configs/b_mixed_readout.yaml
python -m meshflow_control check-config --config configs/c_clean_readout.yaml
```

The configuration validator enforces the registered model/training settings. Before formal GPU work, use the experiment controller's frozen identity, initialization, gradient, precision, cost and budget checks. The public `train` command is a training entry point; it does not automatically execute the entire study or substitute for those preflight receipts.

Measure GPU name, total/used/free memory, utilization and active processes, as well as CPU/RAM and CPU matching load before GPU dispatch. This study fixes one GPU worker, using the shared lock `/tmp/wrp_meshflow_jedge5_diag_gpu0.lock`. Retain at least the larger of 2 GiB or 15% of device memory as free headroom. Do not start another GPU job or change scientific settings to fit memory.

The finite GPU preflight also requires two already registered official-model reference fixtures, including their `official_reference.json` with PASS status. That reference directory is an external provenance prerequisite, not something the public `prepare` command creates.

```bash
export OFFICIAL_REFS=/path/to/frozen_official_reference_directory
python tools/preflight_gpu.py \
  --start "$START" --references "$OFFICIAL_REFS" \
  --training-data "$ARTIFACTS/data" \
  --stream-root "$ARTIFACTS/training_stream" \
  --out "$ARTIFACTS/preflight/real" --ledger "$LEDGER"
```

The script has a fixed schedule of 43 additional one-step forwards, 11 zero-update backwards and four Euler50 replays. The already registered official references consume another four one-step forwards and two Euler50 replays. All actual calls, including any failures, remain in the same budget; this schedule is not a permission to regenerate missing references or retry incomplete cells. `--resume` may reuse completed immutable cells with matching identities, but refuses silent partial-cell replay.

## Fixed experiment controller

After actual preflight PASS, the implemented total entry point is:

```bash
python tools/run_experiment.py \
  --artifacts "$ARTIFACTS" --start "$START" \
  --preflight "$ARTIFACTS/preflight/real/preflight.json"
```

It uses `$ARTIFACTS/audit/budget`, trains A/B/C sequentially to 1,000 while saving both checkpoints, samples the seven registered labels, and constructs `outputs/all_completed.json` with 288 cells. It then runs CPU evaluation and gallery rendering. Its logs, resource snapshots and status are under `orchestration/`; actual outputs are under `outputs/`, `evaluation/` and `gallery/`.

To continue a reviewed interruption, pass `--resume` to the same controller command. Completed checkpoints and outputs are hash-checked; ambiguous training state and unreceipted sample attempts are refused. The controller does not perform final visual review, independent audit, scientific interpretation or report writing. Its COMPLETE status alone does not establish research success.

The next sections document lower-level commands. Do not run fresh manual training in addition to the controller's same registered work.

## Train and resume

Use the same external ledger, output root, imported data and paired stream for all arms:

```bash
python -m meshflow_control train \
  --config configs/a_baseline.yaml \
  --data-manifest "$ARTIFACTS/data/train_manifest.json" \
  --stream-root "$ARTIFACTS/training_stream" \
  --start-checkpoint "$START" --out "$RUN" --ledger "$LEDGER" \
  --stop-step 500
```

Run B/C with `configs/b_mixed_readout.yaml` and `configs/c_clean_readout.yaml` respectively. They use independent model/optimizer instances and identical registered examples. GPU work remains sequential. Continue an existing arm explicitly:

```bash
python -m meshflow_control train \
  --config configs/a_baseline.yaml \
  --data-manifest "$ARTIFACTS/data/train_manifest.json" \
  --stream-root "$ARTIFACTS/training_stream" \
  --start-checkpoint "$START" --out "$RUN" --ledger "$LEDGER" \
  --resume "$RUN/checkpoints/A_step0500.pt" --stop-step 1000
```

The analogous B/C checkpoints are `B_step0500.pt` and `C_step0500.pt`. A direct `--stop-step 1000` run also saves the fixed 500 and 1,000 checkpoints. Do not invoke both routes as separate fresh runs.

Original AdamW parameters advance from step 5,000; new readout parameters start at zero. The final conditional total is 6,000. Parameters are FP32, the backbone forward is BF16, the readout calculation and integration are FP32. Training uses eval mode with autograd, free-only FM, eight microbatches and one shared valid-free scalar denominator. There is no auxiliary loss, scheduler or EMA.

Per-arm evidence is under `training/A`, `training/B` and `training/C`: status, input/update events, exposure CSV, loss/cost CSV, readout gradient/update diagnostics and the latest checkpoint pointer. Checkpoints include optimizer moments, RNG state, stream cursor, configuration and data identity.

SIGINT/SIGTERM requests stop at a completed update boundary and save a recoverable checkpoint with status `INTERRUPTED_SAFE`. Resume with that exact path and unchanged arguments. Configuration/data/stream identities and durable update logs must agree. An incomplete update, uncovered execution or prior `BLOCKED` numerical failure must not be silently replayed. Do not delete ledger entries or restart from START to hide a failed attempt.

## Sample a supplied control mesh

The condition file is either an FP32 `.npy` or a `.npz` containing an FP32 `C` array. Accepted shapes are `[K,3,3]` or `[K,9]`; all values must be finite. The explicit `N` must be in [128,256], with 0 < K < N.

```bash
python -m meshflow_control sample \
  --checkpoint "$START" --start \
  --condition /path/to/condition.npz --condition-key C \
  --num-faces 192 --seed 12345 \
  --out "$RUN/example_START" --ledger "$LEDGER"

# For a newly trained checkpoint, omit --start.
python -m meshflow_control sample \
  --checkpoint "$RUN/checkpoints/C_step1000.pt" \
  --condition /path/to/condition.npz --condition-key C \
  --num-faces 192 --seed 12345 \
  --out "$RUN/example_C1000" --ledger "$LEDGER"
```

These illustrate the API; do not add them to the fixed study as extra samples. Its controller uses only registered requests and frozen noise identities. `--candidates n` is a generic convenience that uses seed+j; it does not construct the study's OLD/NEW panels.

Sampling is Euler50 with one model evaluation per step and exact C clamping at every state. The output directory contains:

- `raw.npz`: raw triangles, free triangles, C, initial noise, all 51 states and known-corner mappings.
- `raw.ply`: binary little-endian FP32 triangle soup with explicit sequential face indices.
- `run.json`: executed timing/call facts and export/condition-retention checks.

There is no welding, hole filling, quantization, smoothing or geometry repair. Known triangle geometry is preserved, but original shared-vertex topology is not claimed. Existing output directories with a completed run receipt are refused rather than overwritten.

## Evaluate and render the frozen requests

The evaluation manifest is a JSON list or an object containing `rows`. Each row identifies `request_id`, `label`, `panel`, `cohort`, `parent_id`, `condition_id`, `candidate`, `N`, `K`, the input bank and hash, the reference manifest and hash, and the actual RAW path. The reference manifest binds separate condition/reference assets and their hashes. Ground-truth/reference arrays are never added to the model input bank. Use the controller's populated manifest; do not infer missing paths or replace missing outputs with a different candidate.

```bash
python -m meshflow_control evaluate \
  --manifest "$ARTIFACTS/outputs/all_completed.json" \
  --out "$RUN/evaluation" --workers 2

python -m meshflow_control render \
  --manifest "$ARTIFACTS/outputs/all_completed.json" \
  --out "$RUN/visuals"
```

Evaluation is CPU-only and records per-output metrics, metric status/denominators, parent aggregates and diversity diagnostics. Worker count is clamped to 1..4, with two CPU threads per worker. Historical START metrics may be reused only with their exact cache identities; new outputs require new scoring. Missing/unsupported metrics stay explicit as NA or NOT_IMPLEMENTED, and uncertainty bounds must accompany PARTIAL values.

Rendering is a single CPU worker using Matplotlib Agg. It preserves original vertices, uses a shared condition frame across labels/noises, displays C blue and free faces orange, and produces four views plus wireframe. P1 also has a separate source-reference page. The complete registered gallery, camera/geometry identities and index are retained. `FINAL_COMPLETE` means rendering coverage is complete; `AI_VISUAL_REVIEW=PENDING` remains until somebody actually examines the images. Source-hidden pages are not evidence of a double-blind human study.

## Fixed panels, budgets and interpretation

P1 has eight original DEV parents, each with OLD noise 0/1 and NEW noise 4/5. Its seven labels are START and A/B/C at 500/1,000: 224 cells. The two old noises and two new noises are reported separately and together; the parent, not each noise, is the independent averaging unit.

P2 has four fixed controls, four noises per control and START/A1000/B1000/C1000: 64 cells. P2 has no source completion GT. Do not compute source MSE or reference-completion S/E there. All 288 cells remain in the manifest, including explicit missing/failed entries. With all 16 OLD START cache entries verified, the normal new sampling requirement is 272 rollouts.

| Budget | Study limit |
| --- | --- |
| Formal optimizer updates | 3,000 total; 1,000 per arm |
| Formal training forward/backward calls | 24,000 / 24,000 |
| Preflight forward/backward calls | At most 64 / 24; no optimizer updates |
| Initialization Euler replay | At most 6 |
| All complete Euler attempts | At most 320, including replay/failure allowances |
| Sampling function evaluations (NFE) | At most 16,000; 50 per complete rollout |
| Cumulative GPU lock ownership | At most 6 hours, including loading, saving and waits |

Use **one shared ledger** for these GPU stages. Its durable reserve-before-call entries count attempts, returns and errors; changing the ledger directory would evade accounting and is not a valid continuation. Limits are ceilings, not targets for extra calls. No additional retries, probes, arms or threshold searches are implied.

A versus START includes 1,000 extra ordinary training updates. B versus A includes additional capacity and a direct readout. C versus B changes the memory source and gradient path; it cannot establish that any difference is solely caused by free-token contamination. The task is plausible controlled completion, so source MSE, Chamfer and reference S/I/E are diagnostics rather than a demand for one exact source completion. Retained C does not establish attachment, and diversity does not establish quality. Report complete multi-view evidence, costs, failed attempts and uncertainty; do not automatically promote a checkpoint or alter the maintained START default.
