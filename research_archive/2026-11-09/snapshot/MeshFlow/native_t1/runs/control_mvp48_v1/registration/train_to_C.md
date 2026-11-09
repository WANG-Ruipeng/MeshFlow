# Control MVP48 Session B completion / Session C handoff

repair_v2 bound; T1, Geo and Full each reached logical step1000. Exactly 2500 optimizer updates and attempts, zero failed attempts. Worker exited with code 0.
TRAIN_READY.json SHA256: `ede603fea6917a5d092518e85087671ddd263a45434225c213d0ba7cff1e55c6`. The marker, checkpoint index, report, log, diff and source files remain frozen.

| Phase | Actual updates | Load s | OT s | Train s | Save + verify s | Phase wall s |
|---|---:|---:|---:|---:|---:|---:|
| T1 | 1000 | 9.260 | 24.608 | 1109.338 | 42.395 | 1210.591 |
| Geo_shared | 500 | 8.941 | 12.231 | 571.740 | 20.793 | 627.750 |
| Geo | 500 | 14.835 | 12.062 | 575.074 | 21.770 | 637.427 |
| Full | 500 | 15.731 | 12.018 | 618.743 | 20.116 | 680.951 |

Total GPU-run wall time 3214.375 s, including real precheck and final verification; official asset restoration occurred earlier. Lock wait 0.000032 s. Real precheck 21.797 s, zero optimizer updates.
Training executed 20000 free-target OT calls; real precheck executed 8. Cache hits 0. CPU implementation-test OT is separate fixture work.
Observed global GPU used peak: 6788 MiB, including unrelated processes. WSL process memory was N/A; no per-worker process peak is claimed. Phase-scoped allocated/reserved peaks are in train_completion.json; allocated is part of reserved.

Use this sole model/loader implementation:

```python
from experiments.mf_h1_local0.control_mvp48.model_mvp import (
    load_mvp_checkpoint, velocity_mvp, integrate_mvp_step)
model, actual_metadata, hashes = load_mvp_checkpoint(path, device)
# velocity_mvp(model, x, t, N, valid_mask, known_mask)
# integrate_mvp_step(model, x, t, dt, N, valid_mask, known_mask, context)
```

Final paths: `checkpoints/T1_step1000.pt`, `checkpoints/Geo_step1000.pt`, `checkpoints/Full_step1000.pt`. Five retained checkpoints and full metadata/hashes: `checkpoints/checkpoints.json`.
Shared fork SHA256: `54dd846afcc54ccab78200d8df3817a5801cabbe8bd3f643f36e388302739899`.
Before requesting CUDA, acquire `/tmp/wrp_meshflow_jedge5_diag_gpu0.lock`. The loader returns a frozen eval model and actual metadata/hashes. Eval disables dropout; velocity helper leaves autograd mode to its caller.
Inputs: FP32 x/context [B,padded_N,9], FP32 free time t[B], int64 total real N[B] in 128..256, separate bool valid/known masks [B,padded_N]. Known must be valid. Padding is excluded from N; free = valid & ~known. Known time is 1 through the final head. Raw known velocity may be nonzero; the shared FP32 Euler helper preserves C exactly.
Coordinates already use the original model scale. Do not multiply by 2 or normalize again. No feature statistics were fitted. Source/GT labels are training supervision, never model inputs. No EMA, new model implementation, loss changes, welding or sampling extension is authorized by this receipt.
Historical BF16 permutation boundary remains FAIL (atol .015625, rtol .03125; 10/1008 coordinates; max_abs .064453125); no threshold relaxation or repair claim.
C retains its own frozen evaluation/sampling budget. B performed no evaluation-based selection/retraining and did not change the default model.
Supplementary completion facts and loss/gradient ranges: `registration/train_completion.json`. Authoritative frozen report: `training/TRAIN_REPORT.md`; ledger: `training/training_log.csv`; actual source diff: `training/code.diff`.
