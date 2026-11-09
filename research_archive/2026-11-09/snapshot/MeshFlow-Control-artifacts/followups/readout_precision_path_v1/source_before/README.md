# MeshFlow-Control

An independent research implementation of triangle-conditioned MeshFlow completion and the **Native Mesh Direct Context Readout V1** study. This is not an official MeshFlow author release.

The model receives an FP32 control mesh `C`, an explicit total face count `N`, and Gaussian noise. It returns a RAW triangle soup `C + F`: known triangles retain their exact coordinates, order, multiplicity and winding, while the free triangles are generated. Exact retention of `C` does **not** guarantee welded interfaces, watertightness or a unique reconstruction.

The fixed study compares:

| Arm | Configuration | Additional readout |
| --- | --- | --- |
| A: BASELINE_CONT | `configs/a_baseline.yaml` | None |
| B: MIXED_READOUT | `configs/b_mixed_readout.yaml` | Free queries read known tokens after the same backbone block |
| C: CLEAN_READOUT | `configs/c_clean_readout.yaml` | Free queries read the known part of the initial token embedding |

B/C add the same 2,359,296 parameters at blocks 3, 6, 9 and 12. Their output projections start at zero. All original and added parameters are trained; C's memory is not detached. B versus A includes additional capacity, and C versus B includes differences in memory depth, state and gradient paths. These comparisons do not isolate a single causal mechanism.

## Install

Python 3.10+ and the dependencies declared in [pyproject.toml](pyproject.toml) are required, including PyTorch 2.7.x. Training and sampling require a compatible CUDA environment on Linux/WSL; the GPU worker uses the shared POSIX `fcntl` lock. Native Windows GPU dispatch is not supported by that lock implementation. Dataset preparation, scoring and rendering use CPU and can be installed independently of the original repository.

With the dependencies already installed, install the source without downloading packages:

```bash
python -m pip install --no-build-isolation --no-deps -e .
python -m meshflow_control --help
python -m meshflow_control check-config --config configs/a_baseline.yaml
python -m pytest -q -m "not gpu"
```

The `meshflow-control` console command and `python -m meshflow_control` expose the same CLI. No old repository is required on `PYTHONPATH`; historical Python is not imported at runtime. Offline installation requires the build and runtime dependencies to exist already.

## Keep artifacts outside this repository

Provide the checkpoint and data separately. This repository contains no model weights, private datasets, generated meshes or experiment logs.

The strict START loader expects the retained HYBRID total-5000 checkpoint:

- File SHA256: `d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80`
- Model-state SHA256: `166c71fef888f26494fc4a0af1776b77a484efa72a6f3ce06552c604fbecc972`

It restores the original AdamW state; arbitrary MeshFlow checkpoints are not interchangeable. The import tool audits the external file without copying it or running a model forward.

The fixed-run controller is `tools/run_experiment.py`. It requires a successful bounded `tools/preflight_gpu.py` receipt, frozen training/evaluation inputs and the shared external ledger. It schedules the three arms and registered outputs; final report writing and actual visual review remain separate.

See [reproduction.md](docs/reproduction.md) for actual prepare, preflight, controller, training, recovery, sample, evaluate and render commands. See [data_contract.md](docs/data_contract.md) for the frozen training data and HYBRID coupling, and [metrics.md](docs/metrics.md) for metric definitions and interpretation.

## Research scope and completion evidence

Each arm has 1,000 updates, effective batch eight and microbatch one. Step 500 is secondary; step 1,000 is primary. The study retains the original START and does not automatically replace the maintained model.

The fixed study completed all three 1,000-update arms, all six step-500/1,000 checkpoints and 272 new Euler50 outputs plus 16 verified START cached outputs. All 288 results were scored and passed exact C/trajectory/NPZ/PLY checks. The independent installed package passed 28 CPU tests; real GPU execution is recorded separately. Training used 32 parents, 453 READY conditions and 8,000 byte-paired CPU OT inputs per arm. See [actual results and limits](docs/results.md); external artifacts retain the full per-output tables, gallery, actual AI review, call ledger and EXPERIMENT_REPORT.md.

Generation is conditional completion, not necessarily reproduction of the sole source mesh. Source-target error and reference-conditioned S/E, together with source-independent C/F conflict I, are useful diagnostics with stated uncertainty, not proof that a different plausible completion is invalid. Shape defects, conflicts, condition retention and complete multi-view evidence must be assessed together. P2 controls have no source completion ground truth.

## Sources and licensing

Necessary components were migrated from the maintained MeshFlow fork, source revision `ae91eaa6f882a7d311233fa32053b7cd5de7ebe0`, based on the original [MeshFlow repository](https://github.com/qiisun/MeshFlow). This repository is a research extension; it does not claim affiliation or endorsement.

The repository retains the original [MIT license](LICENSE), but this must not be read as covering every vendored file. In particular, `src/meshflow_control/models/backbone/utils.py` retains an NVIDIA 2024 notice requiring an express license agreement for use, reproduction, disclosure or distribution; the separate permission underlying that notice has not been verified here. This provenance limitation remains explicit and does not change the local experiment implementation.

Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [model source map](docs/model_source_map.json) and [evaluation source map](docs/evaluation_source_map.json). Dataset import and training migration receipts retain additional file identities in the external artifact directory. Model weights, source datasets and dependencies retain their separate terms; the repository license does not authorize their redistribution.
