# CONTROL feedback autonomous review

This is a compact review package, not a complete experiment backup. It contains the final report, complete registered CSV tables, comparison gates, accounting audit, protocols/manifests, final and actually executed new source, original scoring source, and source hashes.

RAW outputs with trajectories, checkpoints including optimizer state, full logs, and the full galleries remain locally available. Their paths and hashes are listed in LOCAL_MATERIALS_MANIFEST.json and CHECKPOINT_MANIFEST.json. No weight or raw-geometry payload is in this ZIP.

code.diff contains every final new experiment Python file as an addition against /dev/null. Executed worker snapshots preserve versions used before engineering fixes. The final current source is not presented as the source of all earlier calls.

Exact byte-identical duplicate source/proof/metadata copies share one physical member. ALIAS_AND_OMISSION_INDEX.json records every logical path, local source path, SHA-256 and canonical member. Large complete rendering manifests are losslessly encoded as .json.xz with both compressed and original SHA-256 in the same index. Actual review/selection JSON, all scientific CSVs and current/original scoring source remain directly readable. Run python RESTORE_ALIASES.py in the extracted folder to restore these exact original files if needed. Every unique selected review byte sequence and every complete evaluation/parent/noise CSV is retained; no unique selected evidence is dropped.
