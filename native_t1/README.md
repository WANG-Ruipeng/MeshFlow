# Native T1 + Geo: Chair HYBRID START

The maintained sampling default is **Chair HYBRID START**, the original OT_HYBRID pilot1000/conditional_total5000 endpoint. It uses Native T1 + C-only Geo + FM with the previously trained HYBRID coupling (lambda=.25). MOMENT is retired. The START model supports total N128..256; it is a separate identity from the explicit legacy N112 FM and JEdge5 profiles.

See [START identity, sampling and retirement boundaries](docs/chair_start.md). Weights remain external to Git. The complete original START checkpoint is preserved locally, including saved AdamW/RNG state; inference does not resume training.

```bash
python -B -m native_t1 sample --condition /path/to/C.npy \
  --num-faces 163 --seed 2452526625 --out native_t1/runs/chair_start_sample
```

C is FP32 known geometry in the trained coordinate scale; total N must be explicit. The sampler preserves C in all51 states of the original Euler50 dynamics. It does not take GT, add guidance, blend new noise or repair meshes. This rollback establishes the maintained baseline, not a claim of universal quality or watertightness.

The older two-chair N112 training sandbox and optional JEdge5/surface losses remain available below. These are explicit legacy recipes, not an implicit continuation of START.

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

## Train the legacy N112 FM baseline

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

All maintained recipes use the same `--loss-recipe` selector. See the [optional-loss interface and extension contract](docs/loss_recipes.md) for adding future candidates without changing the FM training objective or legacy sampler.

## Sample from C only

Without `--profile`, sampling uses `chair-hybrid` and the retained START weight. The following command explicitly selects the legacy N112 L0 profile. Fresh clones provide their own trained checkpoint explicitly:

~~~bash
python -B -m native_t1 sample --profile fm-geo \
  --checkpoint native_t1/runs/fm_3000/additional_step500_cumulative_step3000.pt \
  --condition native_t1/data/chair_n112.npz \
  --condition-key parent0_K12_constraints \
  --seed 10101 --out native_t1/runs/sample_fm
~~~

To select JEdge5 explicitly, use `sample --profile edge5` with its checkpoint. `fm-geo` accepts only pure-FM Geo identity; `edge5` accepts only the edge5 objective. `trained-geo` remains a generic explicit-checkpoint compatibility route.

Standalone FP32 `[K,3,3]` or `[K,9]` NPY can replace NPZ; omit `--condition-key`. Coordinates must already use the trained scale. For the explicit legacy profiles only: total N112; supported K2/4/8/12, with K4/8/12 used in training. Invalid/degenerate known triangles are rejected.

Inference reads only learned weights, C and noise. It receives no GT, source IDs or losses and performs no guidance, projection, welding or repair. `raw.npz` stores the output, all51 states and original Gaussian; `run.json` records identity, runtime, C preservation and call counts.

## Checks and code boundaries

~~~bash
python -B -m unittest discover -s native_t1/tests -v
~~~

`train`/`train-edge5 --check-only` with a new output directory performs one microbatch forward/backward and zero updates. It is an explicitly requested GPU check, not an import side effect. Help/imports start no training.

`chair_model.py`, `chair_checkpoint.py` and `chair_sampling.py` define the default START route. `model.py`, `context_geometry.py` and `sampling.py` retain shared operations and legacy N112 behavior. `training.py` retains FM; `objectives.py` and `losses/` hold optional training objectives. `geometry_checkpoint.py` validates recipe identity. Portable execution imports no historical `experiments/` package.

Known-coordinate preservation does not guarantee seam connectivity or watertightness. Full historical collision/topology audits are separate from portable lightweight metrics; unexecuted audits remain `NOT_RUN`. See [dependencies and checkpoint boundaries](DEPENDENCIES.md). Failed-candidate source is archived locally with its evidence, not part of the main training path.
