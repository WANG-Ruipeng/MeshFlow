# Bench D0 loader and scale evidence

The official Bench D0 strict CPU load passed with 296 FP32 EMA state keys and
455,592,963 parameters. No D0 forward or CUDA initialization occurred in the
preflight. The actual checkpoint is synset `02828884`, SHA256
`67765c728c503b61c21624c05997ea026c9de486d1529052fafa134be3336b14`.

The first pre-forward configuration guard expected the Chair identity-encoder
`fixed_std=0.02`; the actual official Bench checkpoint declares `fixed_std=0.05`.
Only the independent Bench loader expectation was corrected. The official
checkpoint, architecture, scientific sampling definition, frozen scorer,
training code, and sample code were not changed. The retained official proposal
API calls `pe_nerf` and residual `decode`, without drawing an encoder posterior.

The before snapshot is explicitly reconstructed from the recorded single patch;
its hash is calculated now, not claimed to have been captured before the failed
preflight. The after hash matches the loader hash saved by the successful strict
CPU load. The failure and success receipts remain unchanged and are copied here.

## Normalization contract

Let `v` be the source coordinates after the official shared mesh normalization to
the interval [-0.5, 0.5]. The shared official generator `train.py` explicitly
constructs the dataset with `do_dataset_normalize=True`; that dataset sets
`std=0.3762` and emits `2*v/0.3762`. The Bench config uses this same dataset and
synset 02828884. The new Bench data adapter performs the same transformation.

The locally archived official D0 `train_vae.py` explicitly sets `vae=True` for
training and validation. Its dataset divides by its separate dataset standard
deviation only when `not self.vae`, so that branch is skipped for D0 even when
`do_dataset_normalize=True`. Its emitted D0 coordinates are `0.95*v/0.5 = 1.9*v`.
Consequently the generator-to-D0 multiplier is `(1.9)/(2/0.3762) = 0.35739`,
exactly `0.95*0.3762` in the retained wrapper. The inverse returns the proposal
to generator coordinates. This is supported by the shared training code and
actual Bench configuration, not an assumption that a Chair-only scale applies.
The archived official inference `post_mesh.py` consumes coordinates at the
[-0.95, 0.95] scale without a category-specific replacement.

## One-time CLI/dependency review

- Run in the repository root with `r2/src` on PYTHONPATH; the existing WSL venv
  already imported the required scoring and D0 dependencies successfully.
- Postprocessing refuses to start before both arms have all 48 registered RAWs,
  matching hashes and 50 generator forwards. It uses one D0 call per output,
  then exact C/seam-anchor restoration, original GEO_C and source winding.
- Root dispatch must enforce GPU resource headroom before each D0 worker;
  postprocess records resources but does not independently choose concurrency.
- Unresolved D0/geometry attempts stop instead of silently replaying. A saved
  proposal can finish geometry without another forward when no geometry attempt
  has started. Complete outputs are hash-verified; scoring verifies all POST
  artifact files again.
- Old assets, scoring definitions and prior results are read-only. New files
  belong to this independent Bench experiment. No GT reconstruction score is
  introduced; ordinary C/F metrics remain applicable to PLAIN, while structured
  seam-only metrics are NA for PLAIN, not zero.
- Conditional aggregation risk: the existing aggregator discovers numerical
  fields by whether any row has a finite value. A legitimately all-NA field in
  this category could trigger its 124-column guard. No scorer change was made.
  If this occurs, reuse the already recorded 124 names in
  evaluation/input_audit/receipt.json and preserve the NA values in aggregation;
  no additional geometry scans are required.

This archive adds no forwards, samples, optimizer updates, noise draws, queries,
geometry scans, cleanup or source-code changes. It does not claim the future
GPU postprocessing has already completed.
