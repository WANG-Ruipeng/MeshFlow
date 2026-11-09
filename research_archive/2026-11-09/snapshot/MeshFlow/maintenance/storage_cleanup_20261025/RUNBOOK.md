# Cleanup and archive recovery

The authoritative deletion scope is `prepared_manifest.json`, sealed by `prepared_manifest.sha256`. Each deletion is recorded in `deletion_ledger.jsonl` with its original hash, size and reason.

This is a storage maintenance operation, not an experiment or model-selection rerun. No model is loaded. The original source datasets, RAW/POST outputs, noise/query arrays, metrics, visual reviews and reports stay available. Removed weight tensors are not archived elsewhere by this operation; do not automatically restore or retrain them.

Stages are implemented by `cleanup.ps1`:

1. `Prepare`: resolve and validate exact files, reject reparse paths, hash candidates and protected assets. This stage refuses to replace an existing plan.
2. `Workspace`: remove only listed files under `E:\MeshFlow`, checking each hash immediately before deletion.
3. `External`: remove only five listed retired historical checkpoint files under `E:\MeshFlow-Control-artifacts\experiments`. This requires filesystem authorization for that exact external scope.
4. `Verify`: verify all deletions and protected hashes, then write `cleanup_receipt.json`.
5. `verify_active_inputs.py`: check the 4,301 selected current nonweight references and 159 archived current-round originals, without loading a model.
6. `closeout_docs.py`: after verified completion, write the report, retained-model index and separate retirement notices. It never changes an original experiment report or completion receipt.

A recorded deletion can be skipped on recovery only if the path remains absent and its ledger hash matches the sealed plan. Unrecorded missing or modified files cause a hard stop. Never infer that a partial stage succeeded merely because some files are missing.

The current-round compact ZIP is `deliverables/R2_BENCH_RAW_LOCAL_REPAIR_V1_REVIEW_SMALL.zip` (3,508,889 bytes). `local_repair_archive_receipt.json` records its member hashes. All full original local-repair experiment files remain under `experiments/r2_bench_raw_local_repair_v1`.

Only redundant expanded members of the old distillation `review_package` are removed. Its original 21,964,835-byte ZIP and `PACKAGE_MANIFEST.csv` remain and can reconstruct those members byte-for-byte. This archival reconstruction is different from restoring retired neural weights.

No default model change, training, generation, rescoring, commit or push is part of this maintenance task.

## Final supplement

`cleanup_legacy_small.ps1` completed the separately audited 237-file old-worktree retirement. The first-pass receipt is preserved as a historical stage, and its generic KEEP list is superseded for those exact files. Do not rerun the first-pass full verifier expecting these retired files to exist. The authoritative final state is `FINAL_CLEANUP_RECEIPT.json` and `effective_retained_assets.json`; `finalize_cleanup.py` verified the effective retained hashes and both deletion ledgers. Final documentation supersedes the `_FIRST_PASS.md` snapshots.
