# Run and resume

Entry point (WSL Ubuntu, installed project environment):

    /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B /mnt/e/MeshFlow-Control/tools/run_recipe_factorial.py --stage main

Individual stages: prepare, train, sample, evaluate. Main performs only the mandatory factorial stage. Curriculum and confirmation are gated by a later written decision after numerical and visual evaluation.

The controller owns /tmp/wrp_meshflow_jedge5_diag_gpu0.lock and at most two measured-resource workers. It refuses an unclosed owner registry or ambiguous prior attempt. Never start another controller while this one is active.

Preparation verifies the old R1 lineage and hashes, the immutable 26-parent stream and paired inputs, the 64 evaluation inputs and shared query banks. A completed precompute receipt is reused. An outstanding CPU precompute is not dispatched again.

Training saves model, AdamW, RNG and stream cursor every100 updates. Only the registered bounded parallel-OOM fallback may resume: one downgrade to a single worker, maximum99 completed updates replay, all physical calls preserved. Code/numerical failures and solo OOM stop new dispatch; they do not authorize silent retries.

Sampling registers64 cells per model and records every rollout before execution. Reused RAW requires checkpoint/input/trajectory/precision/export proof. A failed cell cannot be resampled. Score/render require all phase RAW receipts.

All files in registrations/source_hashes.json are frozen. Historical files and RAW are read-only. RUN_STATE.json is a convenience status; durable worker calls, checkpoint receipts and phase ledgers are the accounting authority.

Optional stage limits: two curricula6000 each, then at most one independent candidate/control repeat6000 each if adoption is proposed. No extra B/readout/loss or default-model changes. Total new unique update ceiling42000; mandatory18000; reused R1 6000 reported separately.