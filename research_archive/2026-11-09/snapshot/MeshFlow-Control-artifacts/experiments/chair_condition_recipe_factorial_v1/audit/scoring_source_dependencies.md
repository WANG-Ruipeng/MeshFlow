# Scoring, reference and rendering source dependency audit

Status: PASS. No Python source in the old E:/MeshFlow workspace is required by the audited current scoring/reference/render call chains. The migrated actual calculation code is present in the frozen source_snapshot, not just wrappers.

Verified 24 local implementation files against both frozen registration and source_snapshot. Frozen registration SHA256: 7c87a8d0e4fb48a56e2dd62e42331cc56222a29294aad572760b661496057089.

## Actual call chains

- main geometry / whole mesh / CF and FF intersection: tools/score_recipe_factorial_conflict.py:score_one -> evaluation.geometry.scan_geometry -> evaluation.pair_audit.pair_scan (local lazy import of intersections) -> evaluation.intersections.verified_pair / classify_triangle_pair_primary -> evaluation.seams.classify_triangle_pair -> evaluation.shape.shape_quality
- main I / finite surface and boundary distances: tools/score_recipe_factorial_conflict.py:score_one -> evaluation.control.trusted_cf / conflict_metrics -> evaluation.seams.nearest_segments -> evaluation.surface.surface_distance_squared -> trimesh.triangles.closest_point
- A20 historical reference metrics: tools/score_recipe_factorial_conflict.py:auxiliary_reference_score -> evaluation.evaluate.score_request (panel P1) -> evaluation.legacy_quality.prepare_reference / score_free -> evaluation.finite_edges -> evaluation.control.score_control -> evaluation.quality.pr_metrics / interface_metrics -> evaluation.surface
- diversity: tools/score_recipe_factorial_conflict.py:diversity -> evaluation.surface.surface_distance_squared -> trimesh.triangles.closest_point
- gallery render: tools/evaluate_recipe_factorial.py:render -> tools/evaluate_schedule40.py:render_page -> matplotlib Agg / matplotlib.pyplot -> mpl_toolkits.mplot3d.art3d.Poly3DCollection / Line3DCollection
- RAW/input binding validation: tools/evaluate_recipe_factorial.py:verify_output -> evaluation.io -> mesh_io.read_ply

## Historical source provenance copies

These copies preserve source lineage; they are not runtime dependencies. Existing files were read only.

| Original path | SHA256 | Status |
|---|---|---|
| /mnt/e/MeshFlow/experiments/mf_h1_local0/evaluate.py | 26db58e02ae13214ff559168362b0b213a92fb8bcf3c5e98aa5adacc0065e7ee | PASS |
| /mnt/e/MeshFlow/experiments/mf_h1_local0/native_inpainting_metrics.py | 31bb8386aa0c5abfce484f308bd2675627596e2a300e48ac897a815bb9642735 | PASS |
| /mnt/e/MeshFlow/experiments/mf_h1_local0/native_patch_seams.py | e9493e36288e5dc7fe36161421e5feceb42a9c0fb66c62491e0f62a745bb5020 | PASS |
| /mnt/e/MeshFlow/experiments/mf_h1_local0/guided_native_geometry_geometry.py | 514049667fb70971f253f9c6a67239b14af1799274b3f673bc9a4b4ec5f6507f | PASS |
| /mnt/e/MeshFlow/experiments/mf_h1_local0/guided_native_geometry_audit.py | 70dc76c233bec41f0ed8f98704283de2e1ee23f1332071072cdabe2d5433f36c | PASS |
| /mnt/e/MeshFlow/experiments/mf_h1_local0/gpu_guidance_evaluation.py | 8086ecb459b43fc11db210f3b0f75b1cf90351a447596f9440fca69028bdbf1e | PASS |
| /mnt/e/MeshFlow/native_t1/runs/edge_factorial_20261001_123000/source_snapshot/experiments/mf_h1_local0/edge_factorial_pilot/edge_metric.py | 3889d84895b91648bab4103341b313a94e0153afdb87836561fbfd6bac957f98 | PASS |
| /mnt/e/MeshFlow/experiments/mf_h1_local0/control_mvp48/eval_scoring.py | 88f5b759094e2b2f1df97063f4c1bb21be0597e84e6a2caa695f351d4dd7e208 | PASS |
| /mnt/e/MeshFlow/experiments/mf_h1_local0/chair_boundary_attachment_v1/control_metrics.py | a6036ccd2f6063a8a863591d4aa79e5358de7ab617449a4c19b133f095ecf9ab | PASS |
| /mnt/e/MeshFlow/experiments/mf_h1_local0/chair_control_usability_quick_v1/scoring.py | c287c97f0e76bf59ec99fe05f7fe494f27d69799d8ae0a938844d3a3663be01c | PASS |

## Installed library metadata

| Distribution | Version |
|---|---|
| numpy | 1.26.4 |
| scipy | 1.15.3 |
| trimesh | 4.10.0 |
| matplotlib | 3.10.9 |
| Pillow | 12.3.0 |
| contourpy | 1.3.2 |
| cycler | 0.12.1 |
| fonttools | 4.65.0 |
| kiwisolver | 1.5.1 |
| pyparsing | 3.3.3 |
| python-dateutil | 2.9.0.post0 |
| packaging | 26.3 |
| networkx | 3.4.2 |
| rtree | 1.4.1 |
| numba | 0.67.0 |

## Limits and execution record

- Static dependency audit, not a replay of scoring or rendering.
- The full local evaluation package is recorded as a conservative implementation closure; individual stages use the listed narrower chains.
- evaluate_recipe_factorial.sample_label contains lazy torch/training imports for sampling only, outside this scoring/render audit; no such import or call was executed.
- Historical source copies document provenance and are not runtime-loaded. An old original may itself contain unused dynamic loaders/model imports; this does not make them dependencies of the migrated scorer.
- Third-party copies cover relevant Trimesh geometry implementation, not all dependency source or compiled libraries. Versions are reported for installed numerical/render distributions.

torch imported: False. Model loads/forwards, scoring, rendering: all zero. Additional source snapshot bytes: 258289.

Full paths, SHA256 values, import line numbers/scopes, snapshots and source-before/after checks are in scoring_source_dependencies.json.
