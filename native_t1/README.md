# Native T1 + Geo: FM baseline and optional losses

The main workflow is **Native In-Context T1 + C-only T_geo + the original free-face FM loss**. The local sampling default is the pure-FM L0 cumulative3000 endpoint. **JEdge5 is an optional training-loss recipe**, alongside surface-only, so later candidates can share the same architecture and inference path.

Known triangles use time1; free triangles use sampled time t. The unchanged 50-step clamped Euler sampler preserves C in all51 states. The recipe remains a two-chair, N112, mixed K4/8/12 sandbox; there is no claim of industrial or new-object generalization. See the [baseline identity, complete training lineage and limits](docs/fm_baseline.md).

## Environment and assets

Run from the repository root on Linux or WSL with Python3.10 and a CUDA GPU supporting BF16. The tested environment uses PyTorch2.7.1+cu128:

~~~bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r native_t1/requirements.txt
~~~

Execution uses strict deterministic algorithms, math SDPA, TF32 off, BF16 backbone autocast and FP32 parameters/integration coordinates. Geometry features and optional objectives use FP32. CUBLAS configuration is set before CUDA initialization; unsupported deterministic operations stop execution. FlashAttention is unnecessary.

Download the official assets separately:

- [Official chair EMA checkpoint, last.pt](https://huggingface.co/datasets/qsun2001/meshflow/resolve/main/v1/120m-ot-v-chair/checkpoints/last.pt)
- [Official ShapeNet archive, shapenet.tar.gz](https://huggingface.co/datasets/qsun2001/meshflow/resolve/main/obj_data/shapenet.tar.gz)

The official checkpoint SHA256 is `bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d`. Only its EMA is loaded, strictly; missing assets never trigger random backbone initialization.

Extract the archive outside Git. Preparation accepts the directory containing `objaverse_occ_v5_ids/`, or that NPZ directory itself:

~~~bash
python -B -m native_t1 prepare \
  --data-root /datasets/shapenet \
  --output native_t1/data/chair_n112.npz
~~~

The [recipe](recipes/chair_n112.json) fixes two source objects, hashes, face indices and the original coordinate transform. Preparation preserves real faces and complements without padding or truncation. It needs no historical output. Data, weights, caches and generated artifacts are excluded from Git.

## Train the FM baseline

With the local G0 cumulative2500 checkpoint, this explicit-budget command produces a pure-FM cumulative3000 endpoint:

~~~bash
python -B -m native_t1 train \
  --inputs native_t1/data/chair_n112.npz \
  --init-geo native_t1/checkpoints/tgeo_g0_cumulative2500.pt \
  --context-encoder geo --loss-recipe fm \
  --continue-stream --updates 500 \
  --out native_t1/runs/fm_3000
~~~

A fresh clone contains no local weights. Follow the [official EMA-to-FM3000 commands](docs/fm_baseline.md#build-from-official-assets) to create your own checkpoint. The historical pure-T1 trainer default remains `--context-encoder none`; the main Geo workflow explicitly passes `--context-encoder geo` as above.

Every invocation constructs fresh AdamW: lr1e-5, betas(0.9,0.95), weight_decay0, clip1, effective batch8/microbatch1. All generator and encoder parameters train; eval mode suppresses dropout while retaining gradients. There is no scheduler or new EMA. `--continue-stream` restores input RNG only, requires the same prepared archive and excludes `--seed`. It does not restore optimizer momentum.

Every output directory must be new. Training requires an explicit update budget and never starts sampling automatically. FM has no auxiliary-loss execution. It is distinct from unconditional official MeshFlow: the native time/role construction and Geo encoder remain present.

## Optional JEdge5

The existing JEdge5 implementation and checkpoint remain available:

~~~bash
python -B -m native_t1 train-edge5 \
  --inputs native_t1/data/chair_n112.npz \
  --init-geo native_t1/checkpoints/tgeo_g0_cumulative2500.pt \
  --continue-stream --updates 500 \
  --out native_t1/runs/optional_jedge5_3000
~~~

This shorthand selects `train --context-encoder geo --loss-recipe edge5`. Fixed gated/ramped coefficients are retained without recalibration. Use matched starting weights, saved stream and update budgets when comparing a new loss with FM. See [JEdge5's formula and mixed evidence](docs/jedge5.md). No sweep or candidate selection is automatic.

All maintained recipes use the same `--loss-recipe` selector. See the [optional-loss interface and extension contract](docs/loss_recipes.md) for adding future candidates without changing the FM default or sampler.

## Sample from C only

Without `--profile`, sampling uses `fm-geo` and the retained local L0 weight. Fresh clones provide their own trained checkpoint explicitly:

~~~bash
python -B -m native_t1 sample --profile fm-geo \
  --checkpoint native_t1/runs/fm_3000/additional_step500_cumulative_step3000.pt \
  --condition native_t1/data/chair_n112.npz \
  --condition-key parent0_K12_constraints \
  --seed 10101 --out native_t1/runs/sample_fm
~~~

To select JEdge5 explicitly, use `sample --profile edge5` with its checkpoint. `fm-geo` accepts only pure-FM Geo identity; `edge5` accepts only the edge5 objective. `trained-geo` remains a generic explicit-checkpoint compatibility route.

Standalone FP32 `[K,3,3]` or `[K,9]` NPY can replace NPZ; omit `--condition-key`. Coordinates must already use the trained scale. Total N112; supported K2/4/8/12, with K4/8/12 used in training. Invalid/degenerate known triangles are rejected.

Inference reads only learned weights, C and noise. It receives no GT, source IDs or losses and performs no guidance, projection, welding or repair. `raw.npz` stores the output, all51 states and original Gaussian; `run.json` records identity, runtime, C preservation and call counts.

## Checks and code boundaries

~~~bash
python -B -m unittest discover -s native_t1/tests -v
~~~

`train`/`train-edge5 --check-only` with a new output directory performs one microbatch forward/backward and zero updates. It is an explicitly requested GPU check, not an import side effect. Help/imports start no training.

`model.py`, `context_geometry.py` and `sampling.py` define the shared model and sampler. `training.py` retains FM; `objectives.py` and `losses/` hold optional training objectives. `geometry_checkpoint.py` validates recipe identity. Portable execution imports no historical `experiments/` package.

Known-coordinate preservation does not guarantee seam connectivity or watertightness. Full historical collision/topology audits are separate from portable lightweight metrics; unexecuted audits remain `NOT_RUN`. See [dependencies and checkpoint boundaries](DEPENDENCIES.md). Failed-candidate source is archived locally with its evidence, not part of the main training path.
