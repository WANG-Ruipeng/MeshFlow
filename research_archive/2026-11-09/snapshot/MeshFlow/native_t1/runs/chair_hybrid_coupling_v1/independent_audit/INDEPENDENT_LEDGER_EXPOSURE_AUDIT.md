# Independent ledger and exposure audit

Status: **PASS**

Actual optimizer returns: 2000; training forwards: 16000; training backwards: 16000.
Failed actual calls: 0; unresolved calls: 0.

Checks reconstruct CALL_BEGIN/RETURN/FAILED counts from durable ledgers, compare accounting.json, require 1000 UPDATE_DONE and 8000 exposure rows per arm, compare paired sample input hashes, enforce 32 parents × 250 exposures and four alpha=1 slots per batch, and compare the full batch free-coordinate denominator with logs and per-sample FM contributions.

Recognized non-CALL CPU scoring metadata are listed in non_call_metadata_rows.json and counted in non_call_metadata_summary.json. They do not count as model forwards; unknown or malformed CALL/event records remain audit errors. Endpoint identity checks use actual PT bytes and their metadata sidecars: pilot_step=1000, conditional_total_step=5000, original Geo4000 source, full AdamW/RNG/stream identity, and the frozen protocol. PT files are never deserialized.

Missing, failed, duplicated or inconsistent evidence is BLOCKED. This audit does not modify training, model weights, protocol, frozen code or data, and does not repair missing rows.

Artifacts: audit_result.json; call_count_audit.csv; paired_input_audit.csv; parent_exposure_audit.csv; batch_normalization_audit.csv; endpoint_identity_audit.json; audited_input_sha256.json.
