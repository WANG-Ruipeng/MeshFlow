# Optional C-preserving postprocessing

`native_t1 postprocess` applies the fixed **GEO_C** geometric cleanup to an existing RAW mesh on CPU. It preserves the first K known triangles exactly and writes separate processed files. Sampling still produces RAW by default. This operation needs Python 3.10+ and NumPy (tested with 1.26.4), with no checkpoint, PyTorch import, GPU, GT, scoring assets or local experiment package.

## Run on an existing RAW

From the repository root:

```bash
python -B -m native_t1 postprocess \
  --input native_t1/runs/chair_start_sample/raw.npz \
  --known-faces 32 --out native_t1/runs/chair_start_post
```

Replace `32` with the actual number of known faces in this RAW. **The first K faces must be C**; K is supplied explicitly, never guessed from geometry or a filename. Input must be finite FP32 `[N,3,3]` or `[N,9]` with `0<K<N`, in the original MeshFlow training coordinates. NPY is also accepted. NPZ defaults to key `output`; for an archive using key `raw`, pass `--input-key raw`. Archives are loaded with `allow_pickle=False`.

Use a new output directory. Existing outputs are never overwritten. Inputs are read-only and the command does not normalize, rerun the sampler, draw noise or fetch missing assets. The package retains its deterministic environment guard; a conflicting `CUBLAS_WORKSPACE_CONFIG` remains an explicit error.

A standalone module entrypoint is also available:

```bash
python -B -m native_t1.postprocess_cli --input raw.npy \
  --known-faces 32 --out processed
```

## Fixed geometry contract

The implementation ports `CHAIR_START_POSTPROCESS_COMPARISON_V1` without changing its algorithm. The radius is fixed at `0.015 / (0.95 * 0.3762)`, approximately **0.0419709561 model-coordinate units**. It is an engineering convention inspired by the paper, not a verified reproduction of the authors' full processing chain or runtime scale. There is no radius sweep or output-bounding-box normalization.

1. Preserve every known triangle's FP32 coordinate bits, order, multiplicity and winding, including signed zeros. Do not delete even a degenerate or duplicate C face.
2. Snap a free corner only when exactly one distinct numeric C anchor lies within the radius. Multiple candidate anchors leave that corner fixed.
3. Consolidate the other free corners onto lexicographically ordered original-point representatives. Representatives stay fixed; no transitive chaining or averaging enlarges the displacement bound.
4. Delete only free exact repeated-vertex/collinear triangles and unordered exact duplicates. C wins over a duplicate free face; among free duplicates, keep the first source face.
5. Build indexed vertices by exact FP32 coordinate bits and preserve source-face/corner mappings. Verify `vertices[faces]` reproduces the processed triangle soup byte for byte.

Original shared-vertex IDs are not present in a triangle soup. Coordinate-based indexing does **not** prove preservation of an arbitrary original vertex-ID graph. Some free faces may be deleted, including all of them on degenerate inputs; final N is reported separately from input N. There is no guarantee of watertightness, manifoldness, seam completion, consistent orientation or absence of intersections.

This is geometric consolidation, with no learned denoiser, smoothing, hole filling, clipping, quantization or normal/winding repair. The existing `gradio_demo.py` processing path is a separate implementation without this C-protection contract.

## Outputs and Python API

| File | Content |
| --- | --- |
| `post.npz` | Processed `output`, indexed `vertices`/`faces`, C, source mappings, all corner displacements/statuses and deletion records as non-object arrays |
| `post.ply` | Binary little-endian indexed mesh with FP32 vertices and triangle indices |
| `postprocess.json` | Input/output hashes, fixed configuration, RAW-to-GEO_C stage labels, geometry stats, per-face deletion reasons and readback checks |

A successful receipt is written only after NPZ/PLY roundtrip and C-preservation checks. A failure after starting leaves `STOPPED` evidence in its new output directory; it is not silently retried. These are geometry and IO checks, not a re-execution of collision/quality metrics. Generated arrays and meshes belong outside Git.

```python
import numpy as np
from native_t1.postprocessing import process

with np.load("raw.npz", allow_pickle=False) as archive:
    raw = archive["output"].reshape(-1, 3, 3)
result = process(raw, K=32)
processed = result["output"]
vertices, faces = result["vertices"], result["faces"]
```

`process` does not modify its caller's array. The function accepts the frozen radius only; its result includes the original face/corner provenance even for deleted free faces. The convenience file API is `native_t1.postprocess_cli.run(input_path, known_faces, out, input_key="output")`.

## Existing evidence and validation

The original 32 saved START outputs covered eight already-observed parents and four noises each. All 32 retained C exactly. The FF-illegal affected-free-face fraction fell from 91.8064% to 84.6360%, with all eight parent averages improving. This is the original FF metric, not a whole-mesh or paper-comparable self-intersection score.

Seam coverage changed from 13.1179% to 13.5828%; only three parent averages improved and the benefit was concentrated in one parent. C conflict coverage rose from 3.4271% to 3.5553%; nine outputs lost 22 free faces. The result supports optional geometric cleanup on that panel, not a claim of general shape repair. The learned denoising stage was not run because verified assets were unavailable. Original scorers, RAW, reports and results stay unchanged locally.

Portable synthetic tests require no weights or private mesh assets:

```bash
python -B -m unittest native_t1.tests.test_postprocessing \
  native_t1.tests.test_postprocess_cli -v
```

Source provenance is recorded in [the recipe manifest](../recipes/postprocess_geo_c.json). The local migration additionally compares all saved arrays and audit fields with the original 32 GEO_C outputs; those local meshes are not needed by the published module or test suite.

Migration validation on 2026-10-08 passed: 32/32 saved cases, all 416 historical arrays byte-identical, face records and all stats equal except elapsed time; 97 historical inputs/results unchanged. All 130 public CPU tests passed, including 21 new geometry/IO tests. The CLI was exercised with Torch/model/experiment imports explicitly blocked. No retained model was loaded and no new generator sample or scoring run was performed.
