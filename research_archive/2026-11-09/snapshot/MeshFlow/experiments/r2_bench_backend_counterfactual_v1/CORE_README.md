# CPU backend counterfactual core

Executed 8 `GEO(RAW)` tasks and 2 unique source-reference `GEO(source)` tasks with the unchanged production `geometry_from_proposal` function. All tasks completed; the core took 3.75 seconds with one CPU worker and one thread. No model, D0, training, OT or random-number call was made.

The fixed radius is `0.04197095609837992` model units. No generated bounding-box normalization, threshold sweep, alternate clustering or source-index shortcut was used. Input/source-code hashes were frozen before processing and verified again afterward.

Resume/verify completed work from the repository root:

```sh
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_backend_counterfactual_v1.core run
```

Completed receipts are checked and reused. An unfinished task directory stops execution instead of silently repeating its geometry call. The resource dispatch receipt must already exist under `resources/`.

`manifest.json` lists the exact 8 generated requests plus 2 source targets, all original stages, the 16 reused RAW/BASE score files and new output paths. `core_complete.json` records the actual scope and receipt hashes. Each `outputs/<job_id>/` includes the original-compatible post/pre-GEO NPZ schema, source restoration and anchor reports, indexed PLY, per-face stage CSV and geometry deletion reasons.

`per_face_stages.csv` carries the original input slot through RAW, complete unconstrained D0 proposal, C-restored proposal, merge-before-cleanup and retained POST stages. The merged soup is reconstructed exactly from the saved `corner_target_input_ids`, including faces subsequently deleted. POST stages contain retained slots only; no face is silently assigned zero area because it was deleted. For the historical RAW/D0/RESTORED/BASE branch, `eventual_GEO_*` refers to the original D0→GEO result; for the RAW_GEO branch it refers to the new direct RAW→GEO result. The two branches share the same within-request slot identities.

`stage_summary.csv` records total/free face counts and geometric free area. These are geometric measurements, not a certification of intact supports, useful surface coverage or low intersections. A large retained area can include invalid overlaps. Quality scoring and actual visual review are separate.

Source targets are packed as the exact canonical `case.C` followed by `case.Y` in unchanged `known_ids/free_ids` order. Full source face/corner/vertex identities are retained separately; incoming free corners receive independent indices. `source_target_face_ids` maps packed slots to source faces, while the production `post.source_face_ids` continues to mean packed input slots. The additional retained source ID arrays compose the production corner permutation. The source rows are two reference meshes, not eight independent samples and not open-generation success evidence.

One preparation attempt failed before registration at Python import resolution because the legacy installed package lacked `geo_c`. Explicitly selecting this repository's `r2/src` fixed the import. That attempt executed zero GEO/model calls, wrote no task output and is not a geometry retry. The finalized script hash is recorded in `manifest.json`.
