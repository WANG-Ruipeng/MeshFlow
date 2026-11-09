# Independent student inference
The standalone entry is `r2/pilots/single_step_denoiser_distillation_v1/infer_one.py`. No arguments or `--help` displays help; supplying inputs without `--execute` only validates on CPU. This experiment does not run additional standalone model calls. Its fixed 48 real + 18 synthetic evaluations use `inference.py`.

Example after the final step4000 export exists:
```sh
python -B -m r2.pilots.single_step_denoiser_distillation_v1.infer_one --input /path/raw.npz --input-key raw --known-faces 6 --out /mnt/e/MeshFlow/experiments/r2_single_step_denoiser_distillation_v1/user_inference/example --geo
```
This is a dry validation. Adding `--execute` requests one new standalone student call, separately recorded under that output. It is not automatically included in or launched by this pilot. Existing or interrupted output directories cannot be silently reused.

Inputs must be finite FP32 triangle arrays [N,3,3] or [N,9]. N means all faces, 1 < N <= 800. A boolean `known_mask` may mark C at arbitrary face positions; alternatively, explicit `--known-faces K` declares the first K faces. Any supplied C array must exactly match those RAW face bytes. The default checkpoint is `checkpoints/student_inference.pt`, strictly requiring the fixed step4000 student endpoint.

The entry makes a stable C-first partition, calls the student once, adds its residual once in FP32, inverse-scales, restores original C bytes, and maps preGEO arrays back to the original face order. It does not detach known features inside the network. The student reads full input N and geometry, without a known-role embedding. Face reordering has ordinary finite-precision attention reduction effects; no bitwise permutation-invariance claim is made.

`pregeo.npz` preserves the original input face order. `pregeo_Cprefix.npz` provides the exact C-prefix representation used by maintained GEO_C. Optional `--geo` calls the unchanged NumPy-only `native_t1.postprocess_cli.run`, with the fixed radius and deletion rules. It does not import the CPU scoring module into the GPU process.

POST has fewer faces when GEO deletes free triangles and uses real indexed vertices/faces. POST stays in retained C-prefix/free order. `post_original_index_maps.npz` maps each POST face and representative vertex back to the original input face/corner IDs, and maps original corners to actual POST vertex IDs (-1 when absent). The source maps explicitly handle deleted faces; output N is not assumed to equal input N.

Runtime uses the small student state, verified db86bc32423c316c00bcc2f645a26243e8010f20 decoder source snapshot and small provenance manifests. It does not open the teacher checkpoint or construct a teacher. Original source licensing remains in repository `LICENSE` and `r2/LICENSE`; source provenance and hashes are preserved rather than replacing the old files.

Validation in this work: CLI help, existing-input CPU validation, arbitrary-mask partition/inverse byte checks, and AST validation. No additional GPU forward or GEO job was run for this standalone entry. Actual strict export reload/inference are accounted within the authorized preflight and final evaluation.
