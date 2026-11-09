# Main sampling accounting

Status: PASS

Actual ledger totals: 312 new rollouts, 312 completed rollouts, 15600 successful sampling forwards; 72 reused outputs contribute zero new forwards.

All 384 registered outputs were checked against actual C/Gaussian input bytes, FP32 RAW and trajectory, PLY, 51 states, nfe50 and BF16/FP32 runtime proof.

The expected15600 is recorded separately from actual ledger counting. Failure, unfinished call and retry records are retained in accounting.json and exceptions.csv.

No model, scoring, rendering, quality summary or anonymous method mapping was used.
