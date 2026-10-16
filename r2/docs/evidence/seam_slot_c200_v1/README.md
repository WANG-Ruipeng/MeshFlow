# SEAM small evidence archive

Snapshot date: 2026-10-17. Read [the maintained results note](../../seam_slot_results.md) first.

This directory preserves **byte-identical copies of selected historical evidence**, not a new evaluation. It contains 29 small source files plus all 16 compressed paired figures from the existing review archive: complete per-output CSV tables (including bad tails), the actual recorded AI_PROXY reviews, reports, a normal-constraint erratum, and model identity metadata. Large weights, raw meshes, point queries, full-resolution local figures, trajectories, and the full training cache remain local and are not included in Git.

[Browse all 16 paired figures](c200/evaluation/gallery/README.md). No rerendering or good-example selection was performed. Total copied evidence is 6,855,881 bytes.

| Directory | Original local experiment | Scope |
|---|---|---|
| `c200/` | `experiments/r2_seam_slot_c200_v1/` | Two 200-step arms, 64 RAW/POST rows, 64 actual nonblind AI_PROXY rows |
| `geometry_guard/` | `experiments/r2_seam_geometry_guard_v1/` | CPU attribution and capacity protection; includes the formal 60-degree formula erratum |
| `occupancy_rollout/` | `experiments/r2_seam_occupancy_rollout_v1/` | Frozen PREDICTED / ALL_ON / SOURCE_ORACLE, 192 stage rows, 144 actual AI_PROXY rows |

`source_hashes.csv` maps every copied file to its original repository-relative path, byte size and SHA256. A `zip!member` source denotes exact member bytes extracted from `experiments/r2_seam_slot_c200_v1/SEAM_SLOT_C200_REVIEW.zip` (archive SHA256 `f68d1b07633f4ca34f8c0a5d6a8dd894631bf4d8ad08e2ee92413948493c5c7f`). The archived reports are intentionally unchanged: their historical statements such as “no commit/push” refer to the experiment at that time, and their relative links/absolute local paths may require the original local experiment directories. Use the maintained results note for working links in this small archive.

`c200/evaluation/per_output.csv` has 64 rows: 2 arms × 16 requests × RAW/POST. `occupancy_rollout/analysis/per_output.csv` has 192 rows: 3 layouts × 16 requests × RAW/PRE_GEO/POST/CAPACITY_POST. `geometry_guard/capacity/per_output.csv` has 32 rows: BASELINE/CAPACITY_GUARD × 16 requests. These are overlapping evidence, not independent repeated experiments; PREDICTED reuses the original candidate.

Fractions are stored on the 0–1 scale in the source per-output tables; report percentages multiply by 100. Macro averages do not pool face counts or surface areas. NA fields must not be replaced with zero, particularly FULL's nonexistent seam roles. Preserve metric statuses and denominators when recalculating. All C inputs here are TRAIN conditions; four noise replicates are not four independent parents.

For a standard-library-only checksum and scalar aggregation check, run from the repository root:

```text
python r2/docs/evidence/seam_slot_c200_v1/verify_evidence.py
```

This reads only copied files, does not load any model or original mesh, and prints the observed main-table values. It cannot certify mesh legality, visual judgments or generalization independently. Source weight hashes are historical metadata; this checker does not access or load weight files.
