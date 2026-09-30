# Dependencies and asset boundaries

Native T1 extends the official MeshFlow source based on commit `8cc74da013d3d2feec1853cccbb1f9ac37814b95`. Its production paths import official `models/`, `datasets/mesh_dataset.py` and `utils/ot_utils.py`, with no Python imports from `experiments/`. Maintained public training paths are pure T1 and T_geo; the old multi-arm research controllers are not required.

## Runtime

Use Python 3.10, PyTorch 2.7.1+cu128 and [requirements.txt](requirements.txt). Dependencies include NumPy, SciPy, einops, PyYAML, trimesh and tqdm. Native execution uses strict deterministic algorithms, TF32 off, BF16 backbone autocast, FP32 parameters/integration coordinates and math SDPA. The geo branch computes features, MLP and exit in FP32, then casts the residual to the hidden dtype for addition. There is no all-FP32 backbone production fallback.

FlashAttention, POT, the custom Chamfer extension, cloud SDKs and upstream demo dependencies are not required. Preparation uses official dataset preprocessing. Training uses CPU SciPy Hungarian matching for nested OT; sampling uses no OT. Pure and geo share the same loss, optimizer definition, input stream and clamped Euler sampler.

Strict deterministic execution covers a fixed operation order. It does not guarantee bitwise output equality after face permutation. The local BF16 full-backbone permutation failure remains documented in [the pilot note](docs/tgeo_pilot.md). Public preflight is explicitly a zero-update forward/backward check, not that historical experiment's complete gate suite. No private evidence is implicitly loaded to waive a numerical comparison.

## Architecture and official initialization

The official config is `configs/snet/base-120m-ot-v-chair.yaml`: v3, width 768, 12 layers, 12 heads, coordinate embedding pe_freq20, RMSNorm/QK normalization. Its architecture max length is 800; this task uses real N=112. Upstream YAML CFG and optimizer defaults do not override the Native contract.

The loader checks the UTF-8/LF-normalized config SHA256, equivalent across Windows CRLF and Linux LF:

```text
fb46e1728884d32d2e0119a8d0ab449cf59dbeeabe5180dcfc3ac9a3c61a570f
```

The external official chair `last.pt` file SHA256 is:

```text
bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d
```

Its `ema` mapping is loaded strictly, and the Native 2x768 role table starts at zero. Meta-device construction avoids random backbone initialization. Official coordinate-embedding closure constants are reconstructed on CPU. No official source asset or weight file is modified.

## Checkpoint identities

| Identity | Meaning |
| --- | --- |
| Official chair EMA | External initialization before Native fine-tuning |
| `native_t1_training_v1` | User-trained pure Native model, completed updates and stream state |
| `native_t1_geo_training_v1` | User-trained Native model with explicit geo metadata and parent identity |
| Legacy geo-only checkpoint | Explicit compatibility for the audited T_geo endpoint; not a graph-to-geo conversion |
| Historical A1500 | A separately pinned local checkpoint, not included or inferred from an update count |

Portable checkpoints retain full model state, architecture/config/input hashes, model-state hash, completed/cumulative updates, input RNG and optimizer state. Loading is strict on schema and tensor keys/shapes. A model with the same update count is not thereby the historical model. The new geo schema is distinct from the legacy fixed-pilot schema.

`--init-t1` with `--context-encoder geo` creates a fresh zero-exit branch on a pure checkpoint. `--init-geo` preserves an existing trained geo branch. A fresh branch uses the explicit encoder seed (1010 in the pilot); the data stream seed is separate. Source vertex IDs and free GT never enter the encoder. Geometry is reconstructed from C on every forward, without detached cross-update condition caches.

Loaders default to frozen FP32 parameters/eval mode. Training explicitly enables every parameter, while retaining eval mode to suppress dropout. Each invocation creates fresh AdamW; saved optimizer momentum is not implicitly restored. For `--init-t1` or `--init-geo`, `--continue-stream` restores input RNG only, requires the same prepared archive, and excludes `--seed`. It is not an exact optimizer resume. `--encoder-seed` applies only when attaching a fresh geo branch, not when loading a trained one.

## Task recipe

[recipes/chair_n112.json](recipes/chair_n112.json) contains only object IDs, binary asset/array hashes, fixed source face indices and preprocessing parameters. It contains no vertices, model tensors, generated geometry or evaluation arrays.

Preparation reconstructs targets, C, free complements and pre-OT arrays from separately obtained source NPZ files. It validates shapes, coordinate transforms, source identities and array hashes. Training remains limited to the two registered parents and mixed K4/8/12; the recipe is not a general ShapeNet training split. No historical run directory is needed.

## Evaluation and compatibility

Lightweight metrics cover surface/boundary distance, coverage, triangle shape and nonfinite values. They do not certify nonintersection, watertightness or manifoldness. Complete CF/FF and exact topology are `NOT_RUN` unless separately executed. Preserving C by clamping does not establish seam closure.

`baseline.py` retains historical pure A_continue cumulative 1500; `checkpoint.py` and `verify.py` retain original step 500. These optional profiles reference private historical assets and are not prerequisites for public commands. Fresh-clone use is `prepare` → `train --official-checkpoint` → pure continuation or geo training → `sample` with an explicit `trained`/`trained-geo` profile and external checkpoint.

## Published and excluded files

Published material is source code, small CPU tests, documentation and the metadata-only recipe. Weights, source/prepared datasets, training/generated outputs, raw audit arrays, figures, local validation logs, Python caches and temporary checkpoints are excluded. The [pilot note](docs/tgeo_pilot.md) states existing evidence and limits; it is not a redistributed experiment archive or a new experiment.