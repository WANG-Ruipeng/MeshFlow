# Dependencies and asset boundaries

Native T1 extends the official MeshFlow source based on commit `8cc74da013d3d2feec1853cccbb1f9ac37814b95`. The maintained baseline is Native T1 + the C-only geo encoder + the original FM objective. Edge/surface supervision remains an explicit optional recipe. Portable execution uses official `models/`, `datasets/mesh_dataset.py` and `utils/ot_utils.py`, without Python imports from historical `experiments/` or private result directories.

## Runtime

Use Python 3.10, PyTorch 2.7.1+cu128 and [requirements.txt](requirements.txt): NumPy, SciPy, einops, PyYAML, trimesh and tqdm. Execution uses strict deterministic algorithms, TF32 off, BF16 backbone autocast, FP32 parameters/integration coordinates and math SDPA. The geo encoder and auxiliary geometry objectives run in FP32. No all-FP32 backbone fallback is enabled.

Training retains CPU SciPy Hungarian matching for nested noise-target OT. Model forward/backward and geometry losses run on the GPU; inference uses no OT or CPU geometry correction. FlashAttention, POT, a custom Chamfer extension, cloud SDKs and the upstream demo environment are unnecessary. The fixed surface objective is evaluated directly in PyTorch; it does not require GeomLoss or an additional transport solver.

Strict repeatability refers to the same operation order. BF16 face reordering can change accumulation rounding; the original observation remains in the [T_geo pilot note](docs/tgeo_pilot.md). Public preflight is a zero-update execution check, not the historical full permutation gate suite.

## Model and assets

The official config is `configs/snet/base-120m-ot-v-chair.yaml`: v3, width 768, 12 layers, 12 heads, coordinate embedding pe_freq20, RMSNorm/QK normalization. Its architecture max length is 800; START uses real N128..256; the legacy sandbox uses N112. Native command settings, not upstream optimizer defaults, define training.

The loader checks the UTF-8/LF-normalized config SHA256, preserving identity across Windows CRLF and Linux LF:

~~~text
fb46e1728884d32d2e0119a8d0ab449cf59dbeeabe5180dcfc3ac9a3c61a570f
~~~

The external official chair EMA file SHA256 is:

~~~text
bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d
~~~

Only `ema` is loaded, strictly. The native 2x768 role table starts at zero. Meta-device construction avoids random backbone initialization; official embedding closure constants are reconstructed on CPU. No official asset is modified.

The geo encoder adds 149,888 parameters. Its features use only C: centroid, sorted edge lengths, area and six components of the unsigned normal outer product. Its context operator is P=I, with no messages between distinct known triangles. Features are recomputed per forward. Public graph mode is unsupported.

## Checkpoints and training state

| Schema or asset | Meaning |
| --- | --- |
| `chair_hybrid_coupling_checkpoint_v1` | Exact retained START; variable N128..256, original full recovery state; inference-only public loader |
| Official chair EMA | External initialization before native fine-tuning |
| `native_t1_training_v1` | Pure native FM model |
| `native_t1_geo_training_v1` | Native model with the geo encoder and FM training |
| `native_t1_geo_objective_training_v1` | Geo model with an explicit portable training-objective recipe |
| Explicit legacy geo schema | Strict compatibility for an audited geo model, never graph conversion |
| Local exported JEdge5 | Model and saved input stream; optimizer payload stripped |

Checkpoint identity includes tensor-state, config and input hashes, encoder metadata, completed/cumulative updates and stream progress. Objective checkpoints also record their loss recipe and fixed coefficients. The exported checkpoint can have a different file hash from its original training file while preserving the same generator tensor hash and stream. Loaders validate the actual schema rather than infer identity from a filename or update count.

The local G0 starting copy is `native_t1/checkpoints/tgeo_g0_cumulative2500.pt`, file SHA256 `3a66659e86586f54d2ec30ed3d4a18275ab69ba0f8cba2866ef67b340c481c0e`. Its generator state SHA256 is `6fa39d1c8c1deade17467075160281dcf01948c45e90ca4a9ad111a7dd97c392`. These are local assets, not included in a clone. Commands to construct your own model are in [fm_baseline.md](docs/fm_baseline.md#build-from-official-assets).

`--init-t1 --context-encoder geo` attaches a new zero-exit geo branch. `--init-geo` restores the learned branch without reinitialization. `--encoder-seed` is only for creating a new branch. All model parameters train, with modules in eval mode; loaders otherwise return frozen FP32 models.

Every invocation of the legacy N112 trainer constructs fresh AdamW. `--continue-stream` restores only saved input RNG, requires the same prepared archive and excludes `--seed`. Saved optimizer momentum is never implicitly resumed. An exported checkpoint that strips optimizer state can therefore still support this continuation.


## Local compact checkpoint index

These retained local exports are outside Git. The export audit verified identical generator tensors and saved input RNG against their source endpoints, followed by strict CPU loading; it did not run new training or generation. Each omits optimizer momentum and remains usable with fresh-AdamW continuation.

| Relative path | Bytes | File SHA256 |
| --- | ---: | --- |
| `native_t1/checkpoints/jedge5_cumulative3000.pt` | 521691011 | `cc7a248fa9a8570059f225cd1c313f21f08585925407a79418df6d63c9cae0e9` |
| `native_t1/checkpoints/ablations/fm_cumulative3000.pt` | 521689571 | `0f22ce498ca2c27dc74b43890fb8aad20f0f3e7153aa74ee6d3c14ea445a2660` |
| `native_t1/checkpoints/ablations/surface_cumulative3000.pt` | 521691211 | `93546de419582302e70cf140dae3aa44aad9ab913bd11b394482a8c347203999` |

The FM baseline uses the pure-FM geo schema; surface and JEdge5 use the explicit objective schema. The edge5 sample profile requires an edge5 objective checkpoint and rejects the FM or surface ablations. The explicit legacy `fm-geo` profile requires pure FM and rejects auxiliary-objective checkpoints. Generic explicit ablations can use `sample --profile trained-geo --checkpoint PATH`.

## Supervision and inference boundary

The prepared [two-object recipe](recipes/chair_n112.json) contains source identities and hashes, fixed face indices and preprocessing metadata, not model tensors or geometry arrays. Preparation reconstructs data from separately obtained official source files. The legacy N112 public trainer is restricted to these two parents and mixed K4/8/12; this is not the START training lineage.

Training-only source identities identify true GT known/free interface edges and corresponding free corners after input permutations. Free targets and fixed GT bbox scales define supervision. These labels do not enter the generator or sampling API. Geometry losses supervise the endpoint estimate from the existing velocity forward; they do not run a second model or project generated coordinates.

Inference needs model weights, FP32 C and a noise seed; START additionally requires explicit total N128..256. The legacy N112/50-step definition is identical for the `fm`, `surface` and `edge5` recipes. Auxiliary objectives add no inference modules.

## Evaluation, compatibility and excluded files

Portable metrics cover surface/boundary distances, coverage, triangle shape and nonfinite values. Full CF/FF collision and exact topology audits remain `NOT_RUN` unless independently executed. Fixed C coordinates do not imply shared seam indices, connected topology or watertightness.

The local sampling default is `chair-hybrid`, pinned to the retained START file and model state; see [START](docs/chair_start.md). `fm-geo` and `edge5` remain explicit legacy N112 profiles. Fresh clones provide their own `--checkpoint`; no local weight or old result directory is downloaded automatically. Historical pure profiles and the earlier T_geo pilot note are compatibility/provenance references, not prerequisites for the main workflow.

Published material consists of source, small CPU tests, documentation and the metadata-only recipe. Weights, datasets, prepared arrays, generated outputs, figures, caches and local validation records are excluded. Local cleanup may remove temporary checkpoints; its maintenance index records retained identities and removals separately. That index is optional local provenance and is not an input to training, loading or sampling.

The compact historical Edge-only endpoint is `native_t1/checkpoints/ablations/edge_only_cumulative3000.pt` (521710363 bytes, file SHA256 `61d09ce272d2b677ed0ae045e81e8a0c6229cbf77b08de0a612e4a76ff10015f`). Its experimental schema is intentionally not accepted by the public loader. Recover its exact historical loader from the local maintenance source ZIP if needed; it is not required for FM or JEdge5 execution.
