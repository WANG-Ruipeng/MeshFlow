# Evidence and recovery index

This directory verifies completed Step2 outputs. It does not select a new default, rerun jobs, publish, or alter historical evidence.

- Full live optimizer checkpoint: `../../training/checkpoint_step1500.pt` (local1500/global15500); generator/head exports: `../../inference/`.
- Training curves and CPU payload audit: `../../training_summary/`.
- Original 44 fixed requests and noise identities: `../../eval_manifest.json`.
- RAW and single-D0 BASE: `../../evaluation/MULTIPART_S1500/`; original 88 scores: `../../evaluation/scores/`.
- Unified GUARD_GEO/FINAL and per-cell ledger: `../../unified/cells/`.
- All 396 old/new metric rows and paired tables: `../../comparison/`.
- Numerical interpretation and layout diagnostics: `../../analysis/`, `../../layout_diagnostic/`.
- All 44 galleries and three actual AI_PROXY review groups: `../../gallery/`, `../../visual_review/`.
- Source snapshots, SHA256, HEAD-relative code.diff, actual calls and phase timing are local to this archive.

Read-only status: `python -B -m r2.pilots.chair_multipart_seam_v1.resume`.
Read-only complete-stage check: `python -B -m r2.pilots.chair_multipart_seam_v1.closeout --check`.
New immutable closeout: `python -B -m r2.pilots.chair_multipart_seam_v1.closeout --finalize`.

No stage is automatically run on missing evidence. Explicit resume requires the existing frozen entrypoint and its ownership/failure checks; unresolved neural attempts must never be silently repeated. Do not retrain completed step1500. Gallery review is nonblind AI_PROXY; exact topology is UNKNOWN from images. New recipe combines training pool, head and additional updates, so matched output comparisons do not isolate one change. Source owner labels are diagnostic only.
