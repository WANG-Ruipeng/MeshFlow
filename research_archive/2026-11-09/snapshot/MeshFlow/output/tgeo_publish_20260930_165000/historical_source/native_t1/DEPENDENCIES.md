# Dependencies and asset boundaries

Native T1 uses the official MeshFlow code at base commit `8cc74da013d3d2feec1853cccbb1f9ac37814b95`. The new package imports official `models/`, `datasets/mesh_dataset.py` and `utils/ot_utils.py`; it does not import Python code from `experiments/`.

## Runtime

Python 3.10, PyTorch 2.7.1+cu128, and the versions pinned in [requirements.txt](requirements.txt) are the tested environment. Native execution uses strict deterministic algorithms, TF32 off, BF16 autocast, FP32 model parameters/integration, and math SDPA only. No FlashAttention, POT, custom Chamfer extension, cloud SDK or upstream demo dependencies are required.

Official data preparation uses NumPy, trimesh and the upstream dataset class. CPU SciPy Hungarian matching is used only when producing training samples. It is not a GPU generation operation. Sampling requires a CUDA GPU and does not invoke OT.

## Fixed architecture and official initialization

The official config is `configs/snet/base-120m-ot-v-chair.yaml`: v3, width768, 12 layers, 12 heads, coordinate embedding pe_freq20, RMSNorm/QK normalization. The architecture max length is800; this registered task still uses actual N=112. The generic upstream CFG and optimizer settings in the YAML do not override the Native sampler or training contract.

The portable loader validates the config's UTF-8/LF-normalized SHA256, so Windows CRLF and Linux LF checkouts are equivalent:

```text
fb46e1728884d32d2e0119a8d0ab449cf59dbeeabe5180dcfc3ac9a3c61a570f
```

Official chair `last.pt` file SHA256:

```text
bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d
```

Its unique `ema` mapping is loaded strictly; the 2×768 role table is initialized to zero. Meta-device construction avoids random backbone initialization. The official coordinate-embedding closure constants are reconstructed on CPU; they are not left as meta tensors. No official source or weight file is modified.

## Portable checkpoints

The trainer writes `native_t1_training_v1` with explicit Native architecture metadata, model state hash, normalized config hash, input archive hash, progress, input RNG state and optimizer state. Loading verifies architecture, pure-FM loss identity, finished progress and strict tensor keys/shapes. Model state is checked after loading.

Inference is frozen FP32/eval. Training explicitly enables all model parameters but keeps modules in eval state to disable dropout. The trainer always creates a fresh AdamW when starting from an existing endpoint; saved optimizer state is not implicitly restored. There are no synthetic or random-weight fallbacks.

## Dataset recipe

[recipes/chair_n112.json](recipes/chair_n112.json) contains only reproducibility metadata: two official object IDs, binary asset/array hashes, fixed source face indices and preprocessing parameters. It contains no vertices, model tensors, generated geometry or evaluation results.

The preparation command builds targets, C, free complements and original pre-OT arrays directly from separately obtained source NPZ files. Shapes, coordinates, source identities and array hashes are validated. No historical run directory is needed.

## Legacy compatibility

`checkpoint.py` and `verify.py` retain the original step500 local replay interface. `baseline.py` pins the later pure-FM A_continue cumulative1500 endpoint. These optional profiles reference historical assets by path and hash; those assets are not published. The default local profile is intentionally not replaced by a random or official-only model when its checkpoint is absent.

Public from-scratch use is `prepare` → `train --official-checkpoint` → `sample --profile trained`. The independent import tests prohibit experiment module imports and implicit checkpoint/CUDA work during import.

## Excluded from Git

Weights, source/prepared datasets, training/generated outputs, local validation records, small audit snapshots, Python caches and temporary checkpoints are ignored. The optional lightweight evaluator reports unimplemented topology/intersection work as `NOT_RUN`. Historical research output is not redistributed as part of this code package.
