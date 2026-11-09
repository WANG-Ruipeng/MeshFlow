# Window policy pilot: unchanged postprocessing contract

This adapter reuses the official single D0, maintained GEO_C, source C corner
restoration/free orientation, and original RAW/POST scorers. It does not add
binding, RBF, iterations, queries, semantic labels or a learned quality metric.
The nearest prior experiment is R29: its geometry and scoring implementation
are reused; its closed NFT/SELF endpoint learning recipe is not resumed here.

`postprocess.py` changes only process-local `D.OUT` and `E.OUT` to
`experiments/r2_window_policy_rl_v1`. Original source, inputs and evidence are
read-only. Importing the adapter does not import torch or perform a model call.

## Controller API

```python
from r2.pilots.window_policy_rl_v1 import postprocess as P

jobs = [
    {"stage_dir": str(OUT / "calibration" / label),
     "label": label, "requests": manifest["calibration_requests"]}
    for label in ("FULL_DET", "S005", "S010", "S020")
]

# Read-only CPU validation; returns normalized per-request rows and identity.
rows, identity = P.prepare_jobs(
    jobs, manifest_path=manifest_path, registration_path=registration_path)

# Explicit GPU worker only, after the controller records resource admission.
# All 256 calibration outputs can share one resident official D0 instance.
P.run_d0(jobs, manifest_path=manifest_path,
         registration_path=registration_path,
         resource_snapshot_path=resource_snapshot_path,
         ledger_dir=OUT / "calibration" / "d0_ledger")

# Run in a fresh, model-free process, after every required proposal exists.
P.run_cpu(jobs, manifest_path=manifest_path,
          registration_path=registration_path,
          ledger_dir=OUT / "calibration" / "cpu_ledger",
          stages=("post", "score"), workers=2)
```

For training, pass one job with the explicitly frozen eight requests and
`stage_dir=OUT/training/batch000`. For final evaluation, pass the three registered
model labels and each label's 44 requests. Request counts and authorization are
owned by the frozen controller; this wrapper never expands or samples requests.

CPU `stages` can be `("post",)`, `("score",)` or `("post","score")`. Use a
separate ledger directory for each independently executed phase. Scoring always
keeps both RAW and POST and does not assign review labels or rewards.

## JSON worker entrypoints

The job file contains `jobs`, `manifest_path`, `registration_path`, `ledger_dir`,
and, for D0, `resource_snapshot_path`. CPU optionally reads `workers` and `stages`.

```text
python -B -m r2.pilots.window_policy_rl_v1.postprocess --phase verify-inputs --job JOB.json
python -B -m r2.pilots.window_policy_rl_v1.postprocess --phase d0 --job D0_JOB.json
python -B -m r2.pilots.window_policy_rl_v1.postprocess --phase cpu --job CPU_JOB.json --workers 2
```

The D0 phase and CPU phase must be separate processes. Original CPU metrics hide
CUDA and reject a process that has imported torch. The GPU phase rejects an
already hidden CUDA environment; it is not a CPU fallback.

## Inputs and output layout

Requests follow the existing `_freeze_request` schema: `request_id`,
`condition_id`, `N`, `K`, frozen input NPZ/hash, canonical C/hash, initial noise
hash, `condition_source` and `condition_source_sha256`. `condition_path` and
`condition_sha256` are accepted aliases. Metadata must carry the validated
source C topology and reversible corner order; no parent GT is needed.

Query descriptors are read from `request.query` or manifest `queries`,
`evaluation_queries`, or `training_queries`, and must match the frozen C/hash.
When a job omits `requests`, legacy `manifest.evaluation_requests` is accepted.
No new query bank is generated.

Each stage directory contains:

```text
raw/<request_id>/                 # owned by the sampler; read-only here
  raw.npz, raw.ply, run.json, receipt.json, window_buffer.npz
proposal/<request_id>/
  started.json, proposal.npz, receipt.json
post/<request_id>/
  proposal.npz, proposal_receipt.json
  denoised.npz, pre_geo.npz        # identical bytes: D0 proposal with exact C restored
  post_canonical.npz/.ply, post_restored_C.npz/.ply, post.npz/.ply
  source_restoration.json, orientation.json, geometry.json, denoise.json
  receipt.json                   # full file hashes, RAW/source/proposal identity
scores/<request_id>/RAW/ and POST/
  score.json, vectors.npz, pairs.json, shape_per_face.json, connectivity.json
```

The original sampler validator checks all 51 C states, initial noise, endpoint,
NPZ/PLY agreement and finite FP32 arrays. It does not require intermediate steps
to be deterministic Euler states, so it remains valid for the frozen stochastic
window. RAW receipts retain `result_path` for `run.json`, `nfe=50`, and
`known_state_checks=51`.

## Cache and failure rules

- Every RAW in the supplied stage jobs is validated before D0 or CPU work.
- A proposal receipt must authenticate exactly one official D0 call, the RAW
  hash, the pinned official checkpoint hash and the proposal hash.
- Completed workers verify the full cached identity and artifacts, then return
  `cache_verified=True` and `model_calls_this_invocation=0`.
- Any incomplete worker ledger, proposal directory, POST cell or score cell
  raises an explicit error. No redraw, new model attempt, overwrite or silent
  recovery is performed.
- All wrapper writes and ledgers stay below the new experiment root. A supplied
  path outside that root, or containing symlink components, is rejected.
- `pre_geo.npz` is a byte-identical convenience alias; it is not a second D0 or a
  second geometric operation. Source face/corner IDs and cleanup outcomes are
  propagated by the maintained exporter, not reconstructed from proximity.

Source winding expresses recorded C order; it does not certify global outward
orientation, manifoldness, water-tightness or that all open C edges must seal.
Legal C/F adjacency, legal edge length, CF/FF/I and deletion/degeneracy remain
separate original observations. Image adoption and body labels require actual
review outside this adapter.
