# Confirmation sampling accounting

Status: PASS

Actual ledger totals: 128 new rollouts, 128 completed rollouts, 6400 successful sampling forwards; 0 reused outputs contribute zero new forwards.

All 128 registered outputs were checked against actual C/Gaussian input bytes, FP32 RAW and trajectory, PLY, 51 states, nfe50 and BF16/FP32 runtime proof.

The expected6400 is recorded separately from actual ledger counting. Failure, unfinished call and retry records are retained in accounting.json and exceptions.csv.

Dispatch failure/retry markers are preserved separately from per-cell attempts. A continued label may start untouched cells, but no cell may have a second rollout.
No model, scoring, rendering, quality summary or anonymous method mapping was used.
