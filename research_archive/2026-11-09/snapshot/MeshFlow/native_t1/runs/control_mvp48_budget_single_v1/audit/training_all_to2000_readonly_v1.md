# Three-arm completed 1000 to 2000 audit

Status: PASS

| Arm | New updates | Forward / backward microbatches | Unique update attempts | Resume |
|---|---:|---:|---:|---|
| T1 | 1000 | 8000 / 8000 | 1000 | RESTORE_ALL; no reset |
| Geo | 1000 | 8000 / 8000 | 1000 | RESTORE_ALL; no reset |
| Full | 1000 | 8000 / 8000 | 1000 | RESTORE_ALL; no reset |

The 1000 batch hashes match exactly across T1, Geo, Full and the frozen shared registry. CSV rows and UPDATE_BEGIN/UPDATE_VALIDATED events are ordered, complete, and contain no duplicate update attempts.

Source step1000 and fixed step2000 checkpoint file hashes match their sidecars and frozen identities. All arms changed role/backbone parameters; Geo and Full also changed Geo parameters.

Full retained the original ramp: additional_step = logical_step - 500 = 501..1500. The 996 nonzero auxiliary rows reconstruct ramp 1 with maximum absolute error 8.6e-08; 4 zero-auxiliary rows remain zero. There is no standalone saved ramp column, so this conclusion combines existing scalar arithmetic with frozen code/protocol identity.

This audit added zero training updates, model forwards, OT calls, and GPU workers. It did not deserialize checkpoint tensors. It verifies the completed training segment, not generated quality or subsequent stages.

Evidence and checkpoint hashes: [JSON](training_all_to2000_readonly_v1.json)
