# Native In-Context T1

A standalone, pure flow-matching completion extension to the official MeshFlow backbone. This directory contains code, tests and the small task recipe only. **Checkpoints, datasets, cached files and experiment outputs are not distributed.**

The current scope is the registered **two-chair, N=112, mixed K=4/8/12** task. Known faces use time 1; free faces use time t. All backbone and role parameters are fine-tuned. Sampling uses 50 Euler steps and preserves the supplied known faces in all 51 states. There are no FC/FF auxiliary losses, extra interface layers, guidance, welding or geometric repair. This is a reproducible mechanism sandbox, not a claim of generalization to arbitrary meshes.

## 1. Environment

Run from the repository root on Linux or WSL with Python 3.10 and a CUDA GPU supporting BF16. The tested environment is PyTorch 2.7.1+cu128. Create a separate environment; the upstream demo/training dependencies are not required for this extension.

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r native_t1/requirements.txt
```

Do not install FlashAttention in this environment: this path requires PyTorch **math SDPA**. The upstream import warning about missing FlashAttention is expected. Strict determinism is enforced; unsupported deterministic operations stop the run instead of silently changing precision or kernels. TF32 is disabled, forwards use BF16, parameters and integration states stay FP32. The package sets CUBLAS workspace configuration before CUDA initialization.

## 2. External official assets

Obtain these official releases separately; downloading is not performed by our code:

- [Official chair EMA checkpoint, last.pt](https://huggingface.co/datasets/qsun2001/meshflow/resolve/main/v1/120m-ot-v-chair/checkpoints/last.pt).
- [Official ShapeNet main archive, shapenet.tar.gz](https://huggingface.co/datasets/qsun2001/meshflow/resolve/main/obj_data/shapenet.tar.gz).

The released chair checkpoint is pinned to SHA256:

```text
bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d
```

Only its `ema` dictionary is used. Missing or different assets fail explicitly; there is no random-backbone substitute. The Native known/free role embedding starts at zero.

Extract the dataset outside Git. The preparation command accepts the directory containing `objaverse_occ_v5_ids/`, or that NPZ directory itself. The recipe in [recipes/chair_n112.json](recipes/chair_n112.json) identifies the two source objects, source hashes and fixed nested patches:

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

This reconstructs the original coordinates, fixed patches and pre-OT targets using the official preprocessing. It neither pads/truncates real faces nor imports old experiment scripts. K2 is also archived for historical sampling; training uses K4/8/12. The generated NPZ is a local artifact and is ignored by Git.

## 3. Verify the training path without updates

```bash
python -B -m unittest discover -s native_t1/tests -v

python -B -m native_t1 train \
  --inputs native_t1/data/chair_n112.npz \
  --official-checkpoint /weights/last.pt \
  --check-only \
  --out native_t1/runs/preflight
```

The CPU tests use small fixtures and mocks, not the real model or dataset. The explicit CUDA preflight loads the real EMA, prepares one effective batch (8 CPU OT calls), and runs **one microbatch forward/backward with zero optimizer updates**. It checks that the parameter hash remains unchanged. It does not claim to exercise a full eight-microbatch optimizer step, and it saves no checkpoint or generation.

Each output directory must be new. Importing this package or requesting help does not start training.

## 4. Train pure T1

A complete first-stage command, using only the external official assets and the prepared NPZ:

```bash
python -B -m native_t1 train \
  --inputs native_t1/data/chair_n112.npz \
  --official-checkpoint /weights/last.pt \
  --seed 10 --updates 500 \
  --out native_t1/runs/t1_500
```

The trainer uses all model parameters, correct context only, effective batch 8/microbatch 1, logit-normal time, new Gaussian noise, free-only nested OT, and original face/corner permutations. The masked free-coordinate FM velocity MSE and its effective-batch denominator are unchanged.

AdamW is freshly created with lr=1e-5, betas=(0.9,0.95), weight_decay=0, grad clip=1. There is no scheduler, new EMA, context dropout or auxiliary loss. Modules remain in eval mode to suppress dropout while normal autograd is enabled.

The explicit update budget is mandatory. There is no automatic continuation or generation at the end. Default checkpoints are written every 250 updates and at the final step; `--save-every` controls checkpoint spacing, not the update budget. Outputs include `run.json`, input-stream hashes and local checkpoints. Numerical failures are recorded and propagated without retry or precision fallback.

To follow the pure-FM 500 → 1000 → 1500 route:

```bash
# Branch A: same model weights, new AdamW, fresh seed10 input stream.
python -B -m native_t1 train \
  --inputs native_t1/data/chair_n112.npz \
  --init-t1 native_t1/runs/t1_500/additional_step500_cumulative_step500.pt \
  --seed 10 --updates 500 \
  --out native_t1/runs/a_1000

# A_continue: new AdamW; continue A's saved input RNG state.
python -B -m native_t1 train \
  --inputs native_t1/data/chair_n112.npz \
  --init-t1 native_t1/runs/a_1000/additional_step500_cumulative_step1000.pt \
  --continue-stream --updates 500 \
  --out native_t1/runs/a_continue_1500
```

`--init-t1` accepts the explicit `native_t1_training_v1` schema from this trainer. It always resets optimizer momentum. `--continue-stream` restores only the saved input RNG and requires the same prepared archive; it cannot be combined with `--seed`. This is branch fine-tuning, not an exact optimizer resume. The code does not promise bitwise-identical retraining across different GPU/software environments.

## 5. Sample your trained model

```bash
python -B -m native_t1 sample --profile trained \
  --checkpoint native_t1/runs/a_continue_1500/additional_step500_cumulative_step1500.pt \
  --condition native_t1/data/chair_n112.npz \
  --condition-key parent0_K12_constraints \
  --seed 9601 \
  --out native_t1/runs/sample_p0
```

Inference needs only the model and known faces C, not GT. A standalone FP32 `[K,3,3]` or `[K,9]` NPY can replace the input NPZ; omit `--condition-key` in that case. Coordinates must already be in the trained model scale. K is restricted to 2/4/8/12 and total N to 112.

`raw.npz` contains the output, all 51 states and the initial Gaussian. `run.json` records checkpoint identity, runtime assertions, C preservation and actual call counts. No projection or mesh repair is applied.

## Code map

| File | Purpose |
| --- | --- |
| `prepare.py`, `recipes/` | Rebuild the registered task from external official data |
| `portable_checkpoint.py` | Official EMA initialization and portable trained-checkpoint loading |
| `train_cli.py` | Explicit update budget, stream handling, checkpointing, zero-update preflight |
| `model.py` | Native per-face time, known/free roles and official Transformer operations |
| `runtime.py`, `sampling.py` | Strict execution policy and original clamped Euler |
| `data.py`, `training.py` | Original input stream and masked pure-FM update |
| `metrics.py`, `metric_geometry.py` | Lightweight offline surface/boundary/coverage diagnostics |
| `tests/` | CPU fixtures, import isolation, identity and CLI contracts |

Full CF/FF intersection and exact topology audits are not included in the lightweight evaluator; these fields remain `NOT_RUN`. See [DEPENDENCIES.md](DEPENDENCIES.md) for boundaries and pinned assets.

For compatibility with local historical work, `baseline.load_baseline()` and the default `sample --profile a-continue` still pin the previously obtained A_continue cumulative1500 file. `original-t1`, `checkpoint.load_t1()`, and `verify-historical` (`verify` alias) retain the original step500 meaning. **Those profiles require separately supplied historical assets, which are not in Git.** A fresh clone should use the prepare/train commands above and `sample --profile trained`; none of those commands needs the historical `experiments/` tree.
