# MF-H1-LOCAL0 — isolated Phase 0–2

This directory adds S_static, Q_center, G_closest and P_point to the frozen
official qiisun/MeshFlow chair EMA at commit
8cc74da013d3d2feec1853cccbb1f9ac37814b95. No upstream tracked source is changed.

Only Phase 0–2 are authorized. There is no automatic Phase 3 launcher.
The real run directory is outputs/run_20260928_01. Read its final PRECHECK_REPORT.md
and decision.json before interpreting any result.

## Environment and assets

Use the existing WSL interpreter:
`/home/warpwang/miniforge3/envs/meshflow/bin/python`.
Run from `/mnt/e/MeshFlow`. No dependency installation or driver change is needed.
The original checkpoint stays at `/mnt/c/Users/warpwang/Downloads/last.pt`.
Expected SHA256:
`bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d`.

Author chair data were safely extracted from shapenet-cls.tar.gz into
`experiments/mf_h1_local0/assets/shapenet-03001627`.
The source archive and checkpoint are never rewritten. Author UID is the
available parent identity; the archive carries no stronger ancestry metadata.

## Reproduction order

Use fresh output paths; scripts refuse existing run/report files. A rerun is
additional compute and training, so do not launch it automatically after the
authorized run. The commands below document entry points, not further permission.

1. `python -m experiments.mf_h1_local0.tests.run_units --output <new-json>`
2. `python -m experiments.mf_h1_local0.prepare --data <chair-root> --output <fresh-run>`
3. `python -m experiments.mf_h1_local0.run_precheck --checkpoint <last.pt> --output <run>`
4. `python -m experiments.mf_h1_local0.run_sampling_precheck --checkpoint <last.pt> --output <run>`
5. The reference arithmetic diagnostics are `diagnose_sampler.py` and
   `diagnose_sampler_reparam.py`; these record additional forwards/rollouts.
   Their current entry points resolve the recorded run_20260928_01 directory.
6. `python -m experiments.mf_h1_local0.run_phase2 --checkpoint <last.pt> --output <run>`
   requires all gate reports and runs exactly Q/G 200 updates each, seed10,
   then 96 actual conditional rollouts and 2 cached unconditional rollouts.

The current runner resolves the completed CPU test and environment reports from
their recorded paths. For another run, provide matching fresh audit reports
before starting. Phase 2 is not intended as a configurable architecture search.

## Implementation

- `geometry.py`: finite segments/triangles; symmetric deterministic degeneracy handling.
- `adapters.py`: shared point encoder, grouped tokens, four residual modules.
  Q/G/P have exactly matched state keys/shapes. Only output projections are zero.
- `official_wrapper.py`: calls the original official forward once; residual hooks
  after layers 3/6/9/12. No target/teacher endpoint argument and no mesh repair.
- `prompt_dataset.py`: reuses upstream normalization, Gaussian noise and nested
  OT directly; targets are doubled after OT exactly as in the official code.
- `train_adapter.py`: masked error sums divided by the complete eight-sample
  valid-coordinate count, frozen eval backbone with autograd through it, full
  optimizer/RNG checkpoints, strict paired input hashes and counted forwards.
- `sample.py`: 50 explicit Euler updates, constant dt=1/50, face/geometry CFG=1;
  the cached condition encoder is shared fairly and current-Xt relations update
  every step. Raw NPZ/OBJ stay in final target coordinates, with no quantization.
- `evaluate.py`: independent Trimesh/CPU triangle distance evaluation; no training
  geometry helper reuse, no projection, topology certification or mesh repairs.

## Evidence boundaries

The single-token test proves a property of the control branch, not useful
generation or architecture superiority. Phase 2 can establish implementation
and condition response; only Q/G seed10 are trained here. S/P training and the
pre-registered two-seed Phase 3 comparisons remain NOT_RUN / NOT_AUTHORIZED.

The original float32 linspace sampler and a fixed dt=1/50 loop differ at a few
floating-point time nodes/steps. Their observed rollout discrepancy remains a
recorded failed numerical comparison. Exact node/dt matching reproduces the
official transport bitwise, and an equivalent integer-clock parameterization
of the unmodified official integrator reproduces the authorized constant-dt
loop bitwise. This is reported as numerical sensitivity, not hidden with a
looser threshold.

All 96 conditional rollouts are actual executions; cyclic mismatches repeat
another parent's correct prompt at the same noise, so there are at most 48
distinct arm/prompt/noise inputs. The four diagnostic author-train parents,
rather than rollout count, are the statistical units. The two base outputs
are reused for 24 prompt scores and are not 24 independent base samples.


The completed run is IMPLEMENTATION_PASS_ONLY. Identical BF16 prompt/noise
repeats vary on the scale of condition changes; causal condition response and
architecture advantage are NOT_ESTABLISHED. Read condition_response.json and
components_benchmark.json. Numerical and learning diagnosis should precede any
separately authorized expansion. No additional run is authorized by this README.
