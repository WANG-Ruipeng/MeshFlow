# Bench source-resolution USER tasks — pending adapter

This directory contains actual C/Y partitions of unchanged, authenticated clean source meshes. It is a staging asset, **not a runnable training dataset**. All source targets exceed the maintained total-face contract of N=128..256. The current production C-only SEAM layout is also not certified for these conditions. No model, sampling, training, OT or new simplification was run.

`manifest.json` and `per_condition.csv` retain every proposed condition, including rejected regions. `accepted_manifest.json` contains only source-task proposals passing both the source-C topology gate and actual AI_PROXY image review; `held_manifest.json` preserves failed proposals and reasons. Admission here does not authorize training or change the model face contract. All production training/evaluation lists remain empty.

Each condition has an `input_C_only.npz` with C, N and reversible C topology metadata; `paired_C_Y.npz` includes the exact original-face complement and mapping. `supervision_source_only.npz` contains source interface ownership for supervision/diagnostics only. These source-owner labels do not establish a deployable C-only layout. Every C/Y union reproduces the unchanged full target bitwise; no new surfaces, caps, connectors or missing-target substitutions were introduced. C uses original face IDs and source winding. Y is one reference completion, not the uniquely required generated style.

The parent/family split was frozen before exact face selection. R1 is the proposed seating surface/assembly; R2 is a distinct support, frame, leg or rear-frame region. Two whole visual families remain adaptation-unseen. Existing 26 parents and 91 previously trained conditions were not relabeled as new holdouts. If a parent's R1 fails, its R2 is not counted as an immediately usable same-parent holdout. `evaluation_staging/` contains the resulting three source-task groups, with no registered models or noise and `evaluation_executable=false`.

`gallery/index.html` presents all 30 actual two-view M | C | Y | C+Y pages. Each page uses its parent's shared source bounding frame. C is blue; Y is gray. `reviews/` preserves the three agents' actual AI_PROXY observations. These are developmental proxy reviews, not independent human certification. Whole coherent assemblies may contain multiple intentional boards or closed parts; isolated point samples and incomplete cut regions are held. A visually clear part can still fail the independent source-C topology check.

Whole-source finite-pair legality is inherited from the previous complete clean-source receipts, with source and receipt hashes verified. C/Y are exact subsets of the same geometry, so no new all-pairs intersection scan was needed. The new checks concern partition identity, C local topology, original corner-permutation restoration and the actual semantic input region. Cached source legality does not certify model generation, uniqueness of completion or universal semantic correctness.

Reproducible CPU stages from the repository root:

```text
python -B -m r2.data_tools.bench_deployment_tasks_v1.source_tasks inspect
python -B -m r2.data_tools.bench_deployment_tasks_v1.source_tasks freeze
python -B -m r2.data_tools.bench_deployment_tasks_v1.source_tasks materialize
python -B -m r2.data_tools.bench_deployment_tasks_v1.source_tasks render
python -B -m r2.data_tools.bench_deployment_tasks_v1.source_tasks close
```

The close stage requires the saved actual review records; it does not manufacture new reviews. Existing files are verified rather than silently overwritten. `execution_receipt.json` records the one pre-output hash-literal correction and actual successful stage wall times. `summary.json` and `COMPLETE.json` give the final counts and completion hashes.
