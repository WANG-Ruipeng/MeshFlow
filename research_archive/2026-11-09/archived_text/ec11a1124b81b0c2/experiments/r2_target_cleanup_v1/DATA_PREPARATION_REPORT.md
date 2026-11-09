# Conservative target deduplication preparation

The paired pool contains the **16 existing TRAIN47 complete-seat parents on which exact free-face deduplication changes the target**. The same 16 parents and the same original C are supplied to both arms. The other 31 unchanged parents remain in the audit inventory; they are not main training examples. This preparation does not train or evaluate a model.

| Quantity | ORIGINAL | DEDUP |
|---|---:|---:|
| Main parents / conditions | 16 / 16 | 16 / 16 |
| Total faces, summed once per parent | 3158 | 2828 |
| Free faces, summed once per parent | 3047 | 2717 |
| N range | 140–256 | 128–248 |
| Saved confirmed illegal Y/Y face pairs | 756 | 353 |
| Original legal C/Y interfaces retained | 119 | 119 |
| Parents with neither saved illegal nor unknown Y/Y pair after cleanup | — | 7 |

Exactly 330 redundant free faces are removed from 330 exact-coordinate duplicate groups. All 330 groups also share the same original vertex-ID set. Within each group, all C faces and all verified legal C/Y interface-owner faces are protected; otherwise the lowest source row is retained. All retained original corner orders, source vertex IDs and coordinates remain unchanged. No group has multiple protected copies left. The union of finite geometric triangles is identical before and after deletion; this does not mean face multiplicity or global topology is identical.

**Versioned interpretation note (2026-10-16):** The independent saved-identity winding audit confirms that all330 groups have opposite original winding, and their canonical vertex-ID rows are identical. These are exact coincident two-sided source copies, not evidence of330 randomly corrupted triangles. The original author may have intended double-sided thin surfaces; that intention is UNKNOWN. Canonical training nevertheless presents each group as two identical triangle targets. This intervention removes duplicate target multiplicity while also changing original two-sided source semantics. It preserves the unoriented surface union, not all orientation or multiplicity semantics. See `source_duplicate_winding_audit.csv` and `.json`; no frozen NPZ or manifest was changed by this clarification.

The all47 inventory changes from 1138 to 735 saved confirmed illegal Y/Y pairs. All 366 original legal C/Y interfaces retain their original two-face incidence. The count of parents with neither saved illegal nor unknown Y/Y pairs rises from 16 to 23. This is only a feasibility inventory, not a declaration that all 23 meshes are clean or a different training-pool selection.

Risk counts filter the already-complete original pair ledger by surviving source face identities. No detector is rerun. Surviving coordinates are unchanged, so existing proper intersections and overlaps between surviving faces remain. No arbitrary intersecting face is removed. There is no D0 processing, welding, hole filling, coordinate change or retriangulation.

This experiment tests **removing exact duplicate free triangles**, not all data cleaning. Target N, free-coordinate loss denominators, OT dimensions and per-parent surface multiplicity change. The paired training must record these differences; it must not insert dummy faces to force the original N. Holding parents fixed does not isolate face multiplicity from the associated N change.

The manifests preserve the original parent/task descriptive fields for reader compatibility. Those inherited quality/lineage fields, including `exact_duplicate_triangles` and original `CC_CF_FF`, describe the original source only. Use `paired_parent_summary.csv`, `inherited_FF_pair_survival.csv` and `data_manifest.json` for the post-deletion counts. NPZ `cleanup_input_source_rows` maps each compact training row back to the original source row; old original face and vertex IDs remain saved.

## Files and recovery

- `datasets/ORIGINAL/train_manifest.json`, `datasets/DEDUP/train_manifest.json`: 16 parents/tasks per arm, schema `r2_target_cleanup_training_data_v1`, pool `COMMON_AFFECTED_16`.
- `data_manifest.json`: frozen provenance, source/output hashes, selection and summary.
- `paired_parent_summary.csv`, `zero_FF_feasibility.csv`: all47 outcomes, including unchanged controls.
- `exact_duplicate_groups.csv`, `source_face_mapping.csv`: every group, original/canonical corner identity, removed face and reversible surviving-face mapping.
- `legal_interface_preservation.csv`: all366 original verified legal interfaces.
- `inherited_FF_pair_survival.csv`: every inherited illegal/unknown Y/Y pair and deletion survival.
- `audits/training_reader_roundtrip.json`: original dataset reader compatibility, including width augmentation identity checks.

Run `python -B -m r2.tools.target_cleanup_v1.prepare --stage prepare` to prepare once, or verify all frozen hashes on subsequent invocation. Run `python -B -m r2.tools.target_cleanup_v1.verify` for the CPU dataset-reader checks. Preparation writes only this new experiment directory. Old assets remain read-only. No training, model forward, noise draw, OT solve or new geometric pair scan is performed by these entry points.
