# Native In-Context T1

Two maintained paths extend the official MeshFlow backbone: **pure T1** and **T_geo**, an experimental per-face geometry encoder for the supplied context. This package publishes code, tests and a small task recipe. Checkpoints, datasets, generated meshes, experiment output arrays and caches are not bundled.

The supported task is the registered **two-chair, N=112, mixed K=4/8/12** sandbox. Known faces use time 1; free faces use time t. All parameters train with the original free-only flow-matching loss. Both paths use the same 50-step clamped Euler sampler, preserving known faces in all 51 states. There is no FC/FF auxiliary loss, guidance, welding or mesh repair.

T_geo is a research candidate. In the local pilot it improved free-surface RMS on 8/8 new-patch outputs from already seen objects, but only 3/8 original-patch outputs, with interface-quality tradeoffs. See [pilot evidence and limits](docs/tgeo_pilot.md). This is not a new-object generalization or watertightness claim. T_graph was evaluated but is not a maintained public workflow.

## Environment

Run from the repository root on Linux or WSL, with Python 3.10 and a CUDA GPU supporting BF16. The tested runtime is PyTorch 2.7.1+cu128. Use this extension's requirements rather than the upstream demo environment:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r native_t1/requirements.txt
```

Do not install FlashAttention in this environment: Native execution uses PyTorch **math SDPA**. The upstream warning about missing FlashAttention is expected. Strict deterministic algorithms are enabled, TF32 is off, the backbone uses BF16 autocast, and parameters/integration coordinates stay FP32. T_geo computes its geometry and encoder in FP32. CUBLAS workspace configuration is set before CUDA initialization; unsupported deterministic operations stop execution.

Same-input repeatability does not imply exact invariance to a different face order. The BF16 backbone showed order-sensitive rounding in the local pilot. Its original numerical failure is disclosed in the pilot note; production precision has not been changed to hide it. Public commands neither load private regression evidence nor apply its historical exception automatically.

## External official assets

Download the official assets separately; these commands do not download them:

- [Official chair EMA checkpoint, last.pt](https://huggingface.co/datasets/qsun2001/meshflow/resolve/main/v1/120m-ot-v-chair/checkpoints/last.pt).
- [Official ShapeNet main archive, shapenet.tar.gz](https://huggingface.co/datasets/qsun2001/meshflow/resolve/main/obj_data/shapenet.tar.gz).

The chair checkpoint SHA256 must be:

```text
bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d
```

Only its `ema` mapping is used. Native known/free role embeddings start at zero. Missing or different assets fail explicitly; random backbone weights are never substituted.

Extract the data outside Git. Preparation accepts the directory containing `objaverse_occ_v5_ids/`, or that NPZ directory itself. The [recipe](recipes/chair_n112.json) contains two object IDs, source hashes, fixed face indices and preprocessing metadata:

```text
objaverse_occ_v5_ids/
  2af09bd8df40506c9e646678ef50aa3d.npz
  5b0e833cf2ea465e42bd82e95587d8af.npz
```

```bash
python -B -m native_t1 prepare \
  --data-root /datasets/shapenet \
  --output native_t1/data/chair_n112.npz
```

Preparation preserves the original coordinate transform, real 112 faces, known/free complements and pre-OT targets. It does not pad or truncate triangles and needs no historical `experiments/` tree. K2 is archived for historical sampling; training uses K4/8/12. The output NPZ is a local artifact ignored by Git.

## Check the path without updates

```bash
python -B -m unittest discover -s native_t1/tests -v

python -B -m native_t1 train \
  --inputs native_t1/data/chair_n112.npz \
  --official-checkpoint /weights/last.pt \
  --context-encoder none --check-only \
  --out native_t1/runs/preflight_pure
```

The CUDA `--check-only` path loads real weights, prepares one effective batch (8 CPU OT calls on success), and runs one microbatch forward/backward with **zero optimizer updates**. It checks that model parameters remain unchanged and saves no checkpoint or generation. This is not a full optimizer-step test or the historical permutation/regression suite. CPU test commands use fixtures and mocks; requesting help or importing starts no training. Run the checks in your own environment; publication does not imply unexecuted remote GPU validation.

Every output directory must be new. Numerical failures are recorded and propagated without retries or precision fallback.

## Path 1: pure T1

Start from the external official chair EMA:

```bash
python -B -m native_t1 train \
  --inputs native_t1/data/chair_n112.npz \
  --official-checkpoint /weights/last.pt \
  --context-encoder none --seed 10 --updates 500 \
  --out native_t1/runs/t1_500
```

`--context-encoder none` is the default. Training uses correct context only, effective batch 8/microbatch 1, logit-normal time, fresh Gaussian noise, free-only nested OT and the original face/corner permutations. The effective-batch denominator counts all free coordinates. All model parameters are trainable; modules stay in eval mode to suppress dropout while autograd remains enabled.

AdamW is fresh for every invocation: lr=1e-5, betas=(0.9,0.95), weight_decay=0, grad clip=1. There is no scheduler, new EMA or auxiliary loss. An explicit update budget is required. Checkpoints default to every 250 updates and the endpoint; `--save-every` changes checkpoint spacing, not the budget. Training does not automatically launch sampling.

Build a pure checkpoint with the 500 → 1000 → 1500 schedule:

```bash
# New AdamW and a fresh seed10 input stream.
python -B -m native_t1 train \
  --inputs native_t1/data/chair_n112.npz \
  --init-t1 native_t1/runs/t1_500/additional_step500_cumulative_step500.pt \
  --context-encoder none --seed 10 --updates 500 \
  --out native_t1/runs/a_1000

# New AdamW, but continue the previous checkpoint's input RNG.
python -B -m native_t1 train \
  --inputs native_t1/data/chair_n112.npz \
  --init-t1 native_t1/runs/a_1000/additional_step500_cumulative_step1000.pt \
  --context-encoder none --continue-stream --updates 500 \
  --out native_t1/runs/a_1500
```

`--continue-stream` requires the same prepared archive and cannot be combined with `--seed`. It restores input RNG only, not optimizer momentum. These commands create **your own** pure1500 checkpoint. They do not give it the identity or measured scores of the historical local A1500 checkpoint. Retraining is not promised bitwise-identical across GPU/software environments.

## Path 2: T_geo candidate

T_geo adds 149,888 trainable parameters. Each known triangle supplies 13 features: centroid, sorted edge lengths, area and six entries of the unsigned unit-normal outer product. A FP32 encoder injects a zero-initialized exit into known-face tokens after coordinate/role embedding. Its context operator is **P=I**: it passes no messages between different known faces. Features use only supplied C, never free GT or source IDs. They are recomputed during training without detached or cross-update condition caches.

For a matched comparison, start both branches from the same pure1500 checkpoint with a fresh seed1010 stream and fresh AdamW. These are two explicit training commands; run only the budgets you intend:

```bash
# Matched pure-FM baseline: +500 updates.
python -B -m native_t1 train \
  --inputs native_t1/data/chair_n112.npz \
  --init-t1 native_t1/runs/a_1500/additional_step500_cumulative_step1500.pt \
  --context-encoder none --seed 1010 --updates 500 \
  --out native_t1/runs/pure_2000

# Candidate: same parent model and data stream, +500 updates.
python -B -m native_t1 train \
  --inputs native_t1/data/chair_n112.npz \
  --init-t1 native_t1/runs/a_1500/additional_step500_cumulative_step1500.pt \
  --context-encoder geo --encoder-seed 1010 --seed 1010 --updates 500 \
  --out native_t1/runs/geo_2000
```

For a real geo forward/backward check without updates, replace `--updates 500` with `--check-only` and choose a distinct `--out`. The geo exit starts at zero. Later nonzero gradients and output changes alone do not establish useful conditioning; all model and encoder parameters still train with the same free-only FM loss.

To preserve an already trained geo branch, use `--context-encoder geo --init-geo /weights/your_geo.pt` instead of `--init-t1`. Do not pass `--encoder-seed`: the learned encoder is loaded, not reinitialized. AdamW is still fresh. Choose either `--seed` for a new input stream or `--continue-stream` for its saved RNG with the same prepared archive. Specify the additional `--updates` budget and a new output directory. Continuation is supported functionality, not an automatic next experiment or evidence from the fixed pilot.

## Sample a trained checkpoint

Pure model:

```bash
python -B -m native_t1 sample --profile trained \
  --checkpoint native_t1/runs/pure_2000/additional_step500_cumulative_step2000.pt \
  --condition native_t1/data/chair_n112.npz \
  --condition-key parent0_K12_constraints \
  --seed 9601 --out native_t1/runs/sample_pure
```

Geo model with the same condition and seed:

```bash
python -B -m native_t1 sample --profile trained-geo \
  --checkpoint native_t1/runs/geo_2000/additional_step500_cumulative_step2000.pt \
  --condition native_t1/data/chair_n112.npz \
  --condition-key parent0_K12_constraints \
  --seed 9601 --out native_t1/runs/sample_geo
```

Inference needs only the model and C, not GT. A standalone FP32 `[K,3,3]` or `[K,9]` NPY can replace the NPZ; omit `--condition-key`. Coordinates must already be in the trained model scale. K is restricted to 2/4/8/12 and total N to 112. Geo also rejects invalid or degenerate known triangles.

`raw.npz` contains the output, all 51 states and initial Gaussian. `run.json` records checkpoint identity, runtime assertions, C preservation and actual call counts. Neither loader silently substitutes a different model; neither sampler projects or repairs the mesh.

## Code and compatibility boundaries

| Component | Responsibility |
| --- | --- |
| `prepare.py`, `recipes/` | Rebuild the fixed task from external official data |
| Pure and geo checkpoint loaders | Strict official, pure and geo checkpoint identities |
| `train_cli.py` | Explicit single-branch budgets, streams, checkpoints and preflight |
| `model.py`, `context_geometry.py` | Native known/free time and optional C-only geo encoder |
| `runtime.py`, `sampling.py` | Stable execution and shared clamped Euler sampler |
| `data.py`, `training.py` | Input stream, nested OT and masked FM update |
| `metrics.py`, `metric_geometry.py` | Lightweight offline surface/boundary diagnostics |

The portable schemas are `native_t1_training_v1` for pure T1 and `native_t1_geo_training_v1` for geo. Geo loading also explicitly supports the audited legacy **geo-only** schema; it never silently treats a graph checkpoint as geo. See [DEPENDENCIES.md](DEPENDENCIES.md).

Historical `a-continue`, `original-t1` and `verify-historical` profiles require separately held historical assets, which are not published. The local sampling default `a-continue` is retained for compatibility; fresh clones should explicitly use `--profile trained` or `--profile trained-geo` and their checkpoint. Public commands above need no historical results or private paths.

Lightweight evaluation does not perform full CF/FF intersection or exact topology audits: those checks remain `NOT_RUN` unless separately executed. This code publication performs no new training or generation and redistributes no pilot weights, data, raw outputs or figures.