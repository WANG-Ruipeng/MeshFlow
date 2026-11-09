# Coarse layout data contract

This is an **ORACLE_PRIVILEGED_SOURCE_LAYOUT** diagnostic. Source geometry supplies two possible coarse region bounds; it is not C-only deployment, a complete part segmentation, a mathematical upper bound, or source-style reconstruction scoring.

The original 30 training parents, 30 C arrays and natural target meshes are unchanged. The original six-parent / 12-request evaluation manifest is copied without changing any request dictionary, seeds, N or C identity. No new independent parent, geometry repair, remeshing, OT, model call or scientific noise was used in data preparation.

## Representation

There are two slots in the fixed order `BACK_REGION`, `SUPPORT_REGION`. Each `float32[2,8]` token contains the region box centre relative to C's area centroid divided by the fixed model L; the **full** x/y/z extent divided by L; and the two-dimensional BACK/SUPPORT one-hot. Validity is a separate `bool[2]` mask. An unavailable region is UNKNOWN, never certified absent. Entire-layout dropout sets all values to zero and both validity flags false.

Each box bounds the union of reviewed non-C source faces in that category. A support region may contain a lower seat shell, end frame or pedestal assembly. It is not required to denote one leg, one connected component, or the complete support system. Source face IDs, input geometry, fine mesh coordinates and part counts are not supplied to the model. Source IDs remain in the audit NPZ/JSON to verify the selected bounds.

Training X_WIDTH multiplies the absolute box x bounds by the same positive alpha as C/Y; C's area centre is recomputed from the actual unchanged known triangles in the training sample. Width audit at alpha 0.9, 1.0 and 1.1 exactly matches independent box recomputation from transformed source faces for all 33 unique train/evaluation conditions.

## Provenance and actual review

Twenty-six unique conditions inherit region semantics from existing accepted source-region records, after exact equality of the entire target triangle array has been checked. Version-mismatched historical cases are recorded in `region_inventory.json` and not transferred. Region faces belonging to the current C are removed before a box is constructed.

Seven training parents (27eb79 and the six HIGH parents) lacked directly reusable regions. A lower-structure geometric proposal was rendered against the actual source target in three views. The data agent actually inspected all seven images and accepted them as coarse support regions, with the observations and image hashes preserved in `manual_source_review.json`. This was source curation, nonblind AI_PROXY, not an output-quality judgment or independent human review. Their back slots stay UNKNOWN even where a visible back exists.

The 33 source-image pages are in `region_gallery/`. Only the seven newly proposed pages are claimed as new actual visual review; older semantics are explicitly inherited.

## Runtime entry points

`r2.pilots.bench_layout_oracle_v1.data.training_layout(case, row, sample)` returns `(tokens, valid)`. `row` requires `condition_id`, `alpha`, and optionally `layout_drop` or `layout_npz`. It reads C from `sample['xt'][sample['known_mask']]`; no noisy free geometry is used to construct oracle P.

`eval_layout(q)` returns stored tokens and validity. Evaluation records are in **`layout_eval_manifest.json`**, keyed by the original `request_id`. The ordinary and NULL paths must not acquire source P.

`verification.json` records the identity, non-C region, width-transform and NULL checks. `protected_sources.json` contains the unchanged source hashes. Data build is complete; do not rerun construction after experiment registration merely to tune regions to generated outcomes.

## Actual inference-input identity cross-check

The independent read-only script `../audits/check_eval_layout_identity.py` called `evaluation.engine().source_assets(q)` for all 12 registered requests. Its actual `C_vertices[C_faces_model]` has exactly the same float32 triangle-coordinate multiset as the source case C used to construct P. The C area centroid and independently recomputed layout tokens agree within explicitly recorded float summation/storage tolerances. `../audits/eval_layout_identity.json` records PASS for all 12 requests. This check loaded no model, imported no Torch, invoked no CUDA, and did not modify the data or its construction code.
