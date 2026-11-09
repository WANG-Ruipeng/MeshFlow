# Full-seat learning-curve evidence package

This archive preserves actual code, unchanged score definitions, the complete per-output/parent/paired CSVs, learning-curve records, the independent boundary normal/tangential component summaries, 192 actual AI_PROXY judgments, and the final report. Main checkpoints are the preregistered 0/1000/3000/6000 points; SMALL_C200 and CREF_C200 remain separate frozen context references.

`source_hashes.csv` maps source bytes to original paths. Text members use ZIP-LZMA; declared_path_hashes.csv consolidates identical path/hash declarations while preserving every receipt in receipts_json. `code.diff` contains new untracked source file contents as well as tracked changes under this experiment directory. `artifact_inventory.csv` indexes local weights, raw meshes, OT caches and full logs, including recorded file hashes; its checksum_method explicitly distinguishes new hashing from an existing save/audit receipt. OT and plan caches are indexed from existing audit receipts without another filesystem scan or rehash. Weights, all RAW arrays, large OT caches and full per-boundary-point traces stay local and are not in this small ZIP. Missing historical paths are not fabricated. Deleted weight identities and receipts remain in cleanup CSVs.

The images in `preview/` are uniformly resized JPEG previews of all 24 requests for both saved-mesh views and finite sections, without result-dependent selection. Original reviewed images and their hashes remain in the review CSV and gallery manifests. Preview compression adds no new visual judgments.

Local workspace: E:/MeshFlow. Read `REPORT.md` and the frozen registration before using the preserved `source/repository/r2/pilots/full_seat_learning_curve_v1/resume.ps1`, `run.py`, or `orchestrate.py`. These are the actual recovery entrypoints, not authorization for new training, generation, changed budgets, a default-model switch, commit or push. Final CPU summaries can be refreshed with these commands, using only completed saved logs, geometry-query arrays and actual reviews:

```bash
python -B -m r2.pilots.full_seat_learning_curve_v1.analysis
python -B -m r2.pilots.full_seat_learning_curve_v1.comparison
python -B -m r2.pilots.full_seat_learning_curve_v1.boundary_components
python -B -m r2.pilots.full_seat_learning_curve_v1.review_summary
python -B -m r2.pilots.full_seat_learning_curve_v1.artifact_archive --status
python -B -m r2.pilots.full_seat_learning_curve_v1.artifact_archive --build
```

Only the `boundary_components_v1/latest.json` snapshot is packaged; earlier partial snapshots and its per-point NPZ arrays remain local and indexed. Its final manifest must cover all 192 registered main model/request/stage cells. Derived normal/tangential components retain their independent definition and NA values, without changing the original boundary scorer.  Package completion is an evidence-completeness statement, not a favorable scientific outcome.


Small-package scope: LOCAL_ONLY_REDUNDANT_TRACES.json names three redundant raw-log / prior-audit ledgers kept on disk with exact SHA256. Every CSV, scoring source, actual review row and all 48 preview images remain included. The complete per_update.csv provides all 6000 updates. Original full traces remain local. This package was finalized from the prepared entries with `python -B -m r2.pilots.full_seat_learning_curve_v1.small_archive_finalize`; no OT rescan or scientific rerun was made.
