# Confirmation sampling preparation and audit

These tools prepare the already-selected C20_40_MIX / R2_MIX_H_ALL pair only. They do not start a sampler, model, scoring job, render or dispatcher. Tests use tiny metadata/ledger fixtures.

## Prepare after both immutable training endpoints exist

Run in the existing WSL environment:

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B \
    /mnt/e/MeshFlow-Control/tools/prepare_recipe_confirmation_sampling.py \
    --out /mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1 \
    --authorization /mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1/registrations/confirmation_training.json \
    --stage prepare-job

Before both COMPLETE receipts plus two final checkpoints/two generator exports and the existing confirm phase registration are available, the result is PENDING and no sampling job is written. Once ready, actual file SHA, completed source/input/initialization identity and endpoint receipts are checked; both checkpoints are bound to the frozen confirm phase, then registrations/confirmation_sampling_preparation.json and confirmation_sampling_jobs.json are frozen.

Use --stage verify-job after preparation to check the frozen result again. Checkpoint tensors are not deserialized. These receipt checks do not replace a separate CPU tensor-state audit.

The future wave is confirmation_sampling_input_repeat1. Its jobs call the frozen evaluate_recipe_confirmation.py with --phase confirm and --controller-lock-owned. The dispatcher still uses the main RUN and shared audit/dispatch_state.json, retaining the global OOM downgrade state. It receives two jobs, each64 fixed requests, n8/9/10/11 and50 Euler evaluations; total128 new cells and6400 expected sampling forwards. Root alone decides when to dispatch after measuring resources.

## Read-only audit after all128 output receipts exist

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B \
    /mnt/e/MeshFlow-Control/tools/audit_recipe_confirmation_sampling.py \
    --run /mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1

Output is audit/confirmation_sampling_accounting, with per_output.csv, per_label.csv, exceptions.csv, source_hashes.csv, accounting.json and ACCOUNTING.md. Existing audit output is protected from overwrite. Missing completed sampling receipts return PENDING.

The auditor reuses the original main ledger parser at audit/sampling_accounting/audit_sampling.py, pinned by SHA c3e973ed1c42fa7820ad76d34e8b58dabfc66a87ff73983d3ff1f128736ce7e7 before import. It counts actual RESERVE/RETURN/ERROR/ROLLOUT_COMPLETE events, checks Euler0..49, retains incomplete/orphan/retry records, and calls the unchanged verify_output for actual C/Gaussian/FP32/PLY/51-state/runtime proof. RETURN records do not need context: the real sampler does not put it there.

Every receipt must identify its registered input and its own output directory, match the sampling worker's completed row, and describe one NEW_EULER50 rollout. Reused RAW cannot be presented as a new confirm output. Checkpoint binding and the saved generator identity are read as small JSON; no weights are loaded by this auditor. A lawful second label dispatch onto untouched cells is recorded separately from a forbidden second rollout of the same cell.

20 CPU synthetic tests passed, including PENDING/no-publication, final SHA/initialization mismatches, wrong phase and Euler order, failed/truncated ledger, and external/historical RAW identity rejection. No actual confirmation preparation/audit, output quality read, sampling or new scientific computation was executed while preparing these tools.
