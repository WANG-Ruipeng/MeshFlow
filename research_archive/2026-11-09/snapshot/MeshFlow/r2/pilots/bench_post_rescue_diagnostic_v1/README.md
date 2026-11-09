# Bench saved-output CPU checks

Completed task: frozen `SEAM_S15000` and `SEAM_S21000`, 48 requests each.
Results: `experiments/r2_bench_post_rescue_diagnostic_v1/REPORT.md`.

The existing D0/GEO/source-orientation artifacts are read-only. The retained
HINGE decoder runs only at 1.5%L with opposite winding required; original
Bench scorers and frozen queries are reused. No neural model or GPU compute.

## Check completed deliverables

From the existing WSL environment and repository directory:

```bash
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -m r2.pilots.bench_post_rescue_diagnostic_v1.closeout
```

This verifies cached report/table/image/code hashes only. It never samples,
scores, trains, or reruns geometry. `--seal` is the one-time closeout operation.

## Implementation and authoritative evidence

- `register.py`: source identity, scope, and resource registration.
- `hinge.py`: resumable per-cell CPU HINGE and original scoring; completed
  receipts are verified before reuse. All 96 cells have completed.
- `summary_v2.py`: authoritative 135-field tables. The original mesh metric
  `boundary_edges` is separate from input metadata `C_boundary_edges`.
- `deletion.py`: exact saved clustering/deletion audit, with RAW/D0/PRE
  dimensions and per-face identities. No radius sweep or GEO replay needed.
- `point_contact.py`: explicitly separate diagnostic allowing one indexed
  seam-tip/bulk point jump, then legal bulk edges. This is not attachment success.
- `render_diagnostic.py`, `record_visual.py`: three frozen RAW/PRE/POST pages
  and actual nonblind AI_PROXY observations; not a full new HINGE visual panel.

Use `hinge/tables_v2/`, not the initial summary's colliding `boundary_edges`
field. The independent connectivity ledger in `hinge/tables/` is unaffected.
Original results, partial engineering-error records, and all corrections are
preserved. Do not rerun exclusive registration/aggregation entries over their
completed outputs or turn this completed diagnostic into an automatic new run.
