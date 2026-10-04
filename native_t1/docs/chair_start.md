# Chair HYBRID START: maintained sampling baseline

The default `native_t1 sample` profile is **`chair-hybrid`**: the retained `OT_HYBRID / pilot1000 / conditional_total5000` endpoint, called START in the retired MOMENT experiment. It is Native T1 + C-only Geo with the original FM loss. The training coupling used lambda=0.25; it adds no inference layer, extra loss, noise blending or guidance at sampling time.

This profile supports the actual registered total face-count range **128..256**, with separate valid/known masks. It is not the historical N112 `fm-geo` checkpoint. The implementation preserves the original START model-forward operations and Euler update order; it imports no local experiment package.

## Load and sample

The local immutable checkpoint is `native_t1/checkpoints/chair_hybrid_start_total5000.pt`. It is a hard link to the retained original full checkpoint, so both local paths keep the same bytes without a second large file allocation. The source path and exact identities are in [the metadata-only manifest](../recipes/chair_hybrid_start.json).

```bash
python -B -m native_t1 sample \
  --condition /path/to/C.npy --num-faces 163 \
  --seed 2452526625 --out native_t1/runs/chair_start_sample
```

`--profile chair-hybrid` is optional because it is the default. Total N must be explicit; it is not inferred from C, filenames, targets or source IDs. C must be finite FP32 `[K,3,3]` or `[K,9]`, with `0<K<N`, in the original trained coordinate scale. For NPZ, specify `--condition-key C`; only that condition array is read. No automatic coordinate normalization is applied.

The Gaussian is the historical local `numpy.random.RandomState(seed).randn(N-K,3,3)` draw, cast from FP64 to FP32, inserted into free slots after the first K known slots. Seeds are integers in `[0,2**32)`. It is not the legacy N112 full-array Torch RNG draw. Sampling uses 50 Euler steps, BF16 model output, FP32 parameters/integration, deterministic math SDPA, and exact C preservation in all 51 saved states. `raw.npz` contains the result, path and initial Gaussian; `run.json` records checkpoint identity and actual call counts.

For a relocated copy, pass `--checkpoint PATH`. This named profile accepts only the exact retained START file SHA256 and model state. Missing assets fail explicitly; no download, random initialization, substitution or fallback is attempted. Git contains source, tests and identity metadata, not the 1.565 GB weight. A fresh clone must receive this retained checkpoint separately.

```python
from native_t1.chair_checkpoint import load_chair_start
model, audit = load_chair_start(device="cpu")
```

CPU loading validates the original schema, file/config/state identities, parameter shapes and frozen inference state. It performs no forward pass and does not initialize CUDA. The full original AdamW, RNG and stream state remain in the checkpoint, but inference does not restore or reset them.

## Training and compatibility boundaries

The existing `train` / `train-edge5` commands remain the explicit-budget N112 sandbox. They are not a multi-parent HYBRID continuation trainer and do not accept this checkpoint through `--init-geo`. Starting another training experiment requires an explicit data/coupling/budget contract; choosing the sampling default starts no training.

Historical N112 sampling remains explicit: `--profile fm-geo`, `edge5`, `a-continue`, `original-t1`, `trained`, or `trained-geo`. Those paths retain their original loaders, noise definition, K2/4/8/12 restrictions and N112 sampler. `--num-faces` may be omitted for them or set to112, never another N. JEdge5 remains optional, and the historical `WORKING_CHECKPOINT` compatibility symbol retains its old Edge5 meaning; current routing uses `DEFAULT_WORKING_CHECKPOINT`.

## MOMENT retirement and evidence limits

The MOMENT V2 experiment and its finite-surface precursors are retired from active code. The V2 MOMENT and H_CONT continuation checkpoints are removed after source/evidence archival; neither is an alternative default. START, original data, raw outputs, reports, per-case tables and the decision-material ZIP remain available locally. Archived source snapshots may retain historical imports and deleted-weight paths; they are evidence, not live runtime dependencies or runnable resume promises.

The fixed V2 comparison did not support promoting MOMENT: primary MSE increased2.96% versus equal-budget H_CONT, only3/8 validation parents improved, and validation correct-target columns remained9/24. This is one limited-budget training stream, not a universal impossibility result. The chosen START has its own geometry/control limitations; exact known-coordinate preservation does not imply correct target selection, seam connectivity or watertightness.

Retirement validation consists of strict CPU loading of the retained real checkpoint, exact model-forward AST comparison with its original implementation, frozen-bank Gaussian identity checks, pure-tensor Euler expression checks, CLI/identity rejection tests and the public CPU suite. No new real-model forward, training or generation is part of the rollback. New GPU repeatability claims require a separately authorized run.
