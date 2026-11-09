# HYBRID RAW/D0 selection: CPU-only recovery and evidence

This branch consumes the frozen 46 `M` requests from `r2_bench_group_weight_match_v1`. It does not load a generator or denoiser, draw noise, train, or change the original assets. See `PROPOSAL.md` and `registration.json` for the scientific contract.

The selector uses area ratio `<0.35` or shortest-edge ratio `<0.30` on nondegenerate RAW free faces, followed by exactly one original RAW-GEO vertex-cluster ring. Selected complete faces use RAW; remaining free faces use the existing single-D0 proposal. C stays byte-identical. The output follows the original GEO_C/source orientation/HINGE_OPP015/ALPHA/HOLE chain.

## Entry points

Use the existing WSL environment from `/mnt/e/MeshFlow`:

```text
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_four_routes_v1.hybrid.run run
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_four_routes_v1.hybrid.render_stream
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_four_routes_v1.hybrid.summarize
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_four_routes_v1.hybrid.review_record --input experiments/r2_bench_four_routes_v1/hybrid/visual_review/manual/backend_data.json --shard backend_data
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_four_routes_v1.hybrid.visual_summary
```

Do not dispatch a second writer while a scoring/rendering entry point is active. Complete receipts are reused only after hash verification. A partial failed request is deliberately not retried or overwritten automatically; inspect its failure record and obtain an explicit engineering recovery decision. No new scientific settings or additional requests are implied by this runbook.

## Evidence locations

- `outputs/<request>/selection.npz`, `per_face_selection.csv`, `mixed_boundary.json`: exact slot/corner selection, original mappings and outer-boundary coordinate disagreement.
- `outputs/<request>/ledger_BASE_POST.json` and `ledger_CLEAN_FINAL_POST.json`: original RAW-slot legal-edge and illegal-pair fates. Synthetic bridge IDs are not matched across pipelines.
- `cpu/HYBRID/<request>/`: unchanged complete CPU backend and BASE/FINAL scores.
- `results/`: all 177 original metric fields, paired values, parent means, area/deletion and selection ledgers; no metric was dropped for an unfavorable result.
- `gallery/`: 46 main RAW/M_FINAL/HYBRID_FINAL pages plus 46 BASE supplemental pages, fixed original cameras and sections.
- `visual_review/`: V2 definitions, fresh nonblind AI_PROXY observations and source-image hashes. The three-axis intersection is not certification of no fragments or self-intersection.
- `audits/dispatch_accounting_note.json`: actual conservative one-worker scoring dispatch and the resource-field compatibility limitation; one renderer overlaps completed jobs.

The RAW-GEO call exists only to define selector clusters. A separately scored complete GEO_RAW control was not added to this experiment. RAW-C coordinate disagreement may already exist before selection and must not be labeled as a new RAW/D0 tear; use the matched legal-edge ledger for such claims.

## One-time in-memory scheduling recovery

`preserved_worker_recovery.py` is a historical controller for this execution, **not a reusable restart command**. Its PID identities and five request IDs belong only to the 2026-10-27 running queue. Do not invoke it again. Its registration rejects an existing recovery directory and verifies process ownership before any signal.

The controller temporarily suspends the original slow geometry worker and this run's renderer, preserves their memory, and calls the unchanged `one(task, regsha)` only for five previously untouched registered requests. At most two new geometry workers are active, started incrementally after the first baseline scan and resource measurement. Every result and nested CPU artifact is hash-verified before the original processes resume. The original pool subsequently reads those same five completed receipts; this is cache reuse, not five repeated scientific calculations.

The records are under `dispatch/preserved_worker_recovery_v1/`. While that controller is active, do not start another scorer, renderer, or controller. A failed recovery stops further dispatch and leaves the original processes suspended for explicit handling; it does not silently retry or resume into partial output directories.

Original request/stage wall times include explicit suspension. `results/phase_runtime.csv` preserves those values and records the suspension interval separately. Wall time with suspension removed is not CPU time or a standalone throughput benchmark. The initial one-worker resource fallback, the zero-call overly strict OS-thread preflight, and their corrections remain separately documented under `audits/`.
