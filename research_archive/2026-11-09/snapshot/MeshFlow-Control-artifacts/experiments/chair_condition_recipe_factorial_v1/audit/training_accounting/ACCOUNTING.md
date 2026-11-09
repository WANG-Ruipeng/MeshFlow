# Training accounting

Status: PASS

Only R2/R3/R4 are charged to this new training budget. R1 is 6000 historical updates and zero new updates.

| Recipe | Completion | Unique updates | Physical updates | Replayed completed updates |
|---|---|---:|---:|---:|
| R2_MIX_H_ALL | COMPLETE | 6000 | 6000 | 0 |
| R3_40_H_LATE | COMPLETE | 6000 | 6000 | 0 |
| R4_MIX_H_LATE | COMPLETE | 6000 | 6000 | 0 |

Concurrent elapsed wall times include contention. Summed worker wall/update times are not isolated GPU benchmarks or total phase elapsed time.

PyTorch peaks cover reset-per-effective-update forward/backward/optimizer, exclude context and desktop. Device-used samples include all GPU processes; neither is a per-process OS VRAM measurement.

## Evidence limits

- No model/optimizer tensor deserialization. Internal state hashes are receipt claims bound to actual immutable file SHA.
- Optimizer continuity evidence combines frozen worker validation code, per-update generator_adamw_step, successful optimizer calls and checkpoint/resume events; it is not a direct tensor comparison at every step.
- Exposure rows are written before computation; loaded rows alone do not establish a forward/backward or successful update.
- For repeated updates final logical rows select the latest successful call timestamp; every physical successful call and failed/unfinished attempt remains separately counted.
- Incomplete live snapshots are not final accounting and are labelled PENDING.

## Failed or pending checks

