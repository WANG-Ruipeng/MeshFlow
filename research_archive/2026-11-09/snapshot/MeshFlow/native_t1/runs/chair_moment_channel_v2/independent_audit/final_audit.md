# Independent V2 final audit

Status: **PASS**. CPU read-only inspection; no model or GPU execution.

- ledger: PASS
- training: PASS
- checkpoints: PASS
- generation: PASS
- identities: PASS

Failures: 0. Missing/incomplete items: 0. Detailed checks are in `final_audit.json`.

## Verified counts and identities

- Every ledger CALL_BEGIN has exactly one matching CALL_RETURN; no CALL_FAIL, pending calls, orphan returns, duplicate call IDs, or unregistered I/P calls were found. Actual model call failures: **0**.
- There were 30,020 forward calls: 120 precheck, 16,000 training, and 13,900 Euler sampling calls. The ledger also contains 44 precheck backward calls, 16,000 training backward calls, and 2,000 optimizer updates.
- H_CONT and MOMENT_XY each completed 1,000 updates and 8,000 sample exposures. All 8,000 paired samples and all 1,000 shared batch hashes agree with the frozen plan and durable actual-call records.
- Both endpoint checkpoints are at moment step 1,000 and total step 6,000. File hashes, complete model tensors, full AdamW state and parameter ownership, stream state, RNG state, and frozen protocol identity were verified. MOMENT_XY has exactly 104,496 new parameters; H_CONT has none.
- The fixed evaluation contains 432 requests: 414 ready and 18 NOT_APPLICABLE. START contributes 138 cached outputs; each trained arm contributes 138 new outputs. Two additional initial zero-channel replays make 278 actual samplings and 13,900 NFE.
- All 416 raw-output records, including the two replays, passed file and array hash, finite FP32 geometry, face-count, input-bank, noise, mask, and final known-coordinate byte checks. The original noise identity was checked across all 48 cross-alpha groups. Both zero-channel replays equal their historical outputs byte for byte.
- All 2,464 protected files and all 20 frozen V2 implementation files retained their registered hashes. No added I/P result files were present.

## Evidence limits

The audit independently reconstructs training sample and batch hashes from durable exposure and call records; every training tensor was not persisted. Final known-coordinate bytes are checked directly from raw outputs. Persistence at all 51 Euler states is supported by the stored actual sampler checks because V2 did not request full trajectories. This audit establishes execution and artifact integrity; quality scores are evaluated separately.

## Preserved audit engineering failure

The first CPU audit stopped with one auditor engineering error: START cached rows use `metadata` for cache markers and `historical_metadata` for the original solver evidence. The audit reader initially chose the former and encountered a missing `nfe` key. The original script and failed JSON/Markdown remain under `final_audit.metadata_interface_failed.*`; the exact repair is in `final_audit_metadata_interface_fix.diff` and its record is in `final_audit_metadata_interface_fix.json`.

After correcting only that metadata routing, the complete CPU audit passed. **Auditor engineering failures: 1; actual model call failures: 0.** No scientific source, raw output, model state, or experiment operation was changed or replayed. The audit used two CPU threads with CUDA hidden and instantiated no model.
