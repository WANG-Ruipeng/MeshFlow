# MOMENT retirement read-only inventory

No deletions, model loads, forwards or GPU calls were performed by this audit. Root owns final hashes, complete source archiving and deletion.

| Candidate group | Files | Bytes |
|---|---:|---:|
| chair_moment_channel_v2_active_package | 20 | 208665 |
| chair_surface_attention_v1_active_package | 19 | 223152 |
| chair_surface_attention_v1r1_active_package | 12 | 106184 |
| MOMENT_XY_V2_checkpoint_weights | 5 | 7831804343 |
| H_CONT_V2_checkpoint_weights | 5 | 7825414599 |
| V2_OT_mapping_NPZ | 8000 | 19319007 |

External textual references found: **4**; actual imports of retiring packages: **0**. The four references are new START identity rejection checks and their tests in native_t1/chair_checkpoint.py and native_t1/tests/test_chair_start.py, verified by direct source reading. The complete static text scan is saved in dependency_scan.jsonl; archived source and run evidence do contain historical references.

Mandatory retained START SHA256: d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80. Existing decision ZIP remains at native_t1/reviews/CHAIR_MOMENT_CHANNEL_V2_DECISION_MATERIALS.zip.

OT attempt records retained: 8000 files, 8248102 bytes. The small synthetic CPU fixture checkpoint is retained separately from formal weights.

- Inventory only: no candidate was modified or deleted; no model or GPU calls.
- Root explicitly confirmed the scope as V2 MOMENT and H_CONT 10 checkpoints, V2 OT mapping NPZ, and the three active packages after full source ZIP preservation.
- All original data, raw, reports, CSV, logs, protocols, audits, source_code snapshots, checkpoint JSON sidecars, OT attempt JSON and existing decision ZIP remain protected.
- Existing historical/source_code references are evidence. Archived scripts will require restoring archived source dependencies to run after retirement.
- OT NPZ contain derived face/corner maps and identity metadata, not original inputs or generated raw. Only the retired V2 stream consumes this cache.
- V1/r1 contain shared MOMENT and SURFACE implementation; they are included only because root explicitly confirmed retiring their active copies with complete source preservation.
- Large checkpoint hashes here are retained sidecar claims, not fresh hashes. Parent owns definitive hashing, ZIP archiving, deletion and protection verification.
- Initial temporary WSL inventory stopped at unavailable rg without final artifacts; Windows rg supplied the saved dependency scan. No scientific action was involved.

Exact resolved Windows/WSL paths and per-file byte sizes are in inventory.json. No recursive deletion of run directories is proposed.

Candidate total: 8061 files, 15677075950 bytes. The three active packages contain 51 source files and no bytecode files.
