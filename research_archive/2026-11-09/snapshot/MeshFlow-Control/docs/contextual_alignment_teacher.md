# Frozen SigLIP training targets

This optional teacher belongs to `CHAIR_CONTEXTUAL_ALIGNMENT_SIE_PILOT_V1`. It is not loaded by default model construction, training without this recipe, or inference. The original MeshFlow workspace is read-only.

## Exact public asset

- Repository: `google/siglip-base-patch16-256`.
- Revision: `b078df89e446d623010d890864d4207fe6399f61`.
- `model.safetensors`: 812,856,640 bytes; SHA256 `f0cee7c815135c44a515eff72ab3040499744920442bc25567cd04efc93f8f65`.
- Model card license: Apache-2.0. Upstream: https://huggingface.co/google/siglip-base-patch16-256/tree/b078df89e446d623010d890864d4207fe6399f61
- The loader uses the official `SiglipVisionModel` and official `SiglipImageProcessor`. All `vision_model.*` tensors from the exact public safetensors are loaded with `strict=True`; no remote code, text tower, tokenizer, or private upload is used.
- Targets use `pooler_output`, dimension 768. The teacher is eval, has no trainable parameters, and encodes in FP32 with autocast disabled and eager attention. CPU strict loading verified 208 tensors and 92,930,304 vision parameters, without a forward or CUDA initialization.
- Transformer API reference: https://huggingface.co/docs/transformers/model_doc/siglip

The isolated teacher environment is under the experiment's `software/teacher_venv`, with system site packages inherited. The existing training environment is unchanged. Fixed additional versions are Transformers 4.57.6, huggingface-hub 0.36.0, and tokenizers 0.22.1. The actual complete version receipt is `teacher/teacher_loading_cpu.json`; upstream metadata and per-file checks are under `teacher/assets/`.

## Training geometry and render definition

The frozen 1000 x 8 plan contains four exact alpha=1 records per step. Only those 4000 records are eligible for teacher targets. They resolve to 32 unique clean TRAINING geometries and 128 unique encoded images. Each geometry is the actual FP32 full target, with the C/free partition independently reconstructed through original face identities. Cache identity uses dtype, shape, and exact geometry bytes; it is not just a UID. Augmented-width records, DEV inputs, and generated model outputs are excluded.

`siglip_render.py` implements a deterministic CPU software z-buffer, rendering the unmodified input triangle soup. Four azimuths are 45, 135, 225, and 315 degrees, with elevation 20 degrees, Y up, and 256 x 256 pixels. The common orthographic camera is fitted to all eligible TRAINING geometries and all four views. Its half-extent is multiplied by 1.10. There is no per-object rescaling. Material is neutral gray, background light gray, lighting fixed, and faces are double-sided. Encoded images have no wireframe, condition color, labels, or UID. Degenerate projected triangles contribute no pixels; geometry arrays are retained unchanged.

The fixed contact sheet shows the first eight training geometries by UID, all four views. Its labels are outside the encoded images. Actual AI visual review is recorded in `teacher/render_review.json`, bound to both the contact and render manifest hashes; it is not a human usability assessment.

An initial CPU raster assignment error occurred before any encoded image was saved. Its source, error, and one-line correction are preserved under `teacher/engineering/render_index_assignment/`. No teacher forward occurred during that attempt, and the geometry/camera definition did not change.

## Target formula and lookup

For each geometry, each of the four official pooler vectors is independently L2 normalized. Their equal FP32 mean is then L2 normalized again. Zero or nonfinite vectors are errors. `TargetStore(runroot).for_sample(sample)` returns a CPU `numpy.float32[768]` copy. It only accepts registered alpha=1 samples. At lookup it reconstructs the original geometry from the actual sample context, clean free x1, source-face IDs, and corner permutations, then verifies the exact frozen geometry hash. The target is supervision only and is never passed to the model or alignment head as input.

The canonical completion file is `runroot/teacher_manifest.json`. Per-geometry NPZ files contain raw poolers, normalized view vectors, and the final target. The manifest binds the public model revision/weight hash, processor config, source hashes, camera/render definition, actual training plan and data manifest, encoded PNG hashes, feature file hashes, and target array hashes.

## Execution and accounting

CPU preparation uses `python -m meshflow_control.teachers.siglip_targets --stage prepare --data DATA --plan PLAN --out RUNROOT`. Actual encoding uses the isolated teacher interpreter with `PYTHONPATH` pointing to this repo's `src`, and `--stage encode --out RUNROOT --ledger SHARED_LEDGER --batch-size 16`.

The encode CLI acquires `runtime.gpu_worker` itself; its parent must not acquire a second copy of that lock. At the current frozen manifest, encoding requires 8 actual teacher forwards and 128 separately accounted teacher images, below the 4096 image cap. Each forward is reserved before execution. Per-batch CUDA synchronization and peak allocator resets occur inside the real budgeted call, without additional forwards. Batch receipts contain synchronized forward seconds and allocated/reserved peaks. Separate resource snapshots record the loaded teacher, completed encoding before unload, and completed unload. PyTorch allocator peaks are not total process VRAM; nvidia-smi device values also include unrelated processes.

Completed receipts are hash-checked and reused. An incomplete attempted teacher batch is not silently replayed. A complete target manifest is verified and returned without re-encoding or rewriting target files. Root owns any explicitly authorized recovery and all GPU dispatch decisions.

## Diagnostics and limits

`feature_statistics.json` reports pairwise target cosine and both equal-geometry and actual-training-frequency normalized constant-mean baselines. These diagnose target distinguishability and a trivial alignment-loss baseline. They do not establish geometric attachment or control quality. Global image semantics can overlook thin structures, local intersections, or C/free interface defects. Any claimed control benefit still requires the experiment's actual generated outputs and frozen S/I/E and shape evaluation.

The independent synthetic CPU suite `tests/test_siglip_targets.py` has 12 passing cases covering fixed framing, double-sided rendering, exact geometry identity, normalization order, invalid poolers, shuffled face/corner target reconstruction, alpha/split identity rejection, cache integrity, and the constant-mean diagnostic. Tests need no public weights, old experiment directory, network, or GPU.