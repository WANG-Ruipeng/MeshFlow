# Bench multi-part data delivery

- [Report](REPORT.md)
- [Filterable gallery](gallery/index.html)
- [316 condition rows](data/conditions.csv)
- [TRAIN277 / DEV39 manifests](data/manifest.json)
- [Per-parent part coverage](data/parent_coverage.csv)
- [342 actual AI_PROXY reviews](data/semantic_review.csv)
- [26 exclusions](data/excluded.csv)
- [255 current SEAM-compatible conditions](data/current_seam_compatible.json)
- [61 valid conditions needing entry/layout extension](data/native_requires_entry_extension.json)
- [Independent exact-array verification](independent_export_verification.json)
- [Execution ledger](execution_ledger.json)
- [CPU resume/verify entry](../../r2/data_tools/bench_parts_v1/run.py)
- [Source snapshot](code_snapshot/README.md)

`data/inputs/` contains C-only NPZ + OBJ. `data/paired/` contains C/Y supervision.
The complete source mesh and source owner/alias labels are never in the inference pack.
72 parents, unchanged TRAIN64 / DEV8; additional conditions are not additional parents.

Local artifact size at closeout: 48,644,604 bytes (46.39 MiB), excluding this small index/hash inventory.
No weights or original complete source meshes are copied into this delivery.
No training, generator/denoiser calls, new noise, commit or push.
