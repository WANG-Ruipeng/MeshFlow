# Frozen official Bench prior: motif guidance pilot

This isolated pilot implements the registered experiment in
`maintenance/motif_guidance_design_20261029/PROTOCOL.md`.
It does not change the maintained sampling defaults.

The generator is the official Bench EMA, loaded strictly into the original
DiT. Parameters remain frozen. G0 overwrites the shadow condition at export;
G1 clamps it after every update; G2 applies endpoint-error guidance to shadow
condition slots; G3 applies the same guidance to all slots. G2 and G3 compute
the full input VJP before masking. Count CFG and motif guidance are separate.

All methods use 50 Euler updates, FP32 state integration, BF16 forward, strict
determinism, and the same saved free-noise arrays. Source C coordinates and
corner winding are restored at export. Closed C has an empty boundary record;
it does not acquire an artificial interface requirement.

The finite budget is 200 new RAW outputs (including 16 auxiliary calibration
outputs), 184 single official D0 calls, and 4,860 input VJPs. There are no
optimizer updates. EXPOSURE reference outputs and scores are reused by hash.
All RAW outputs must be saved before the D0 stage starts.

## Entrypoints

Run from the repository root with the existing WSL Python environment:

```text
python -B -m r2.pilots.bench_full_prior_motif_guidance_v1.run status
python -B -m r2.pilots.bench_full_prior_motif_guidance_v1.run resources
```

Explicit registered job lists are dispatched with `run sample --jobs-json`.
The controller records resources and launches workers incrementally. Completed
outputs are authenticated and skipped. Uncommitted scientific calls cannot be
silently replayed. `generate --recover-export CELL_ID` can finish an export on
CPU only when all 50 updates have already been safely committed.

`run cpu` performs existing RAW or POST scoring. `run evaluate -- ...` wraps
the D0, collection, rendering and calibration-lock stages with an execution
ledger. `analyze.py` reads saved evidence; it does not infer visual labels or
run models. Actual visual reviews are stored separately with image hashes.

Execution-specific job lists, code revisions, engineering incidents, receipts,
results and recovery instructions are under
`experiments/r2_bench_full_prior_motif_guidance_v1/`.
