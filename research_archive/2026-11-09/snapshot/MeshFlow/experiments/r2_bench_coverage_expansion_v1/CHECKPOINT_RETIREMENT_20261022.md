# Checkpoint lifecycle update — 2026-10-22

User-authorized storage cleanup retired the completed branch's old local1250 rolling.pt; COVERAGE and USER v2 also retired checkpoint_step500.pt. The exact paths and SHA256 are in E:/MeshFlow/maintenance/storage_cleanup_20261022/prepared_manifest.json. Final step1500 full/inference weights and initial step0 remain, as do all original reports, JSON sidecars, data, RAW/POST and scores.

Historical intermediate-weight replay is retired. Do not restore or recompute the deleted intermediates automatically. Old inventory receipts describe historical availability; use the cleanup receipt for present availability. No scientific results were changed.
