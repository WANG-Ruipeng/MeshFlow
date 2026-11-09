# R2_C_REFERENCE_FRAME_V1

This independent experiment is authorized on 2026-10-13. Maintained model code, old weights, old metrics and old results are read-only. No commit/push or default promotion is authorized.

Scientific contract: one 200-update continuation from EXP67 step6000, restoring the same generator, AdamW and RNG as SMALL_C200; reuse all1600 ordered exposure records. Four familiar C, batch8/micro1, lr1e-5, original free-only FM, HYBRID0.25 and BF16-forward/FP32-state runtime. No new parameters, teacher, loss, or condition labels. Whole-pair width augmentation precedes C-area-center calculation. Backbone inputs are local to that center; Geo reads original augmented world C in the same slots. Recompute target OT in preOT units (subtract center/2); never translate cached xt or noise. Do not demean Gaussian.

Evaluation: frozen48 requests (familiar32, DEV16), exact C/N/K/Gaussian/query/source identities from R2_SEAT_NOISE_TRANSFER_V1. Two new states CREF_C0/CREF_C200 (96rollouts/4800generator forwards), two read-only cached controls EXP67_FROZEN/SMALL_C200. All96new RAW must exist before any POST. World free is local free+center; external C is copied from original bytes in all51states. POST is one official D0, GEO_C, source C restoration and fixed-C free orientation. 96newD0 calls,192newCPUstage scores plus192cached scores. No RBF, no additional noise/checkpoints or sampling sweep.

Main question: does local200 show repeated clear RAW C adoption improvements relative to SMALL_C200 across multiple conditions/noises, without comparable losses or missing-body/conflict costs? local0 separates immediate wrapper effects; improvement over a damaged local0 alone is insufficient. Familiar/DEV, RAW/POST, YES/NO/UNCERTAIN, legal local edges, AVMR/PMR, CF/FF/I and whole-body completeness stay separate. These are development conditions, not an untouched generalization test. AI_PROXY visual review is not human evaluation; original source style is not a unique desired completion.

## Recover

Run from WSL:

```sh
cd /mnt/e/MeshFlow
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.c_reference_frame_v1.run --stage all
```

Stage choices: prepare, train, generate, post, score, render. Preparation, source/code identities and both frozen registrations must pass before GPU dispatch. Completed cells are hash-verified and reused. Do not replay a pending or failed model call. A training INTERRUPTED_SAFE checkpoint can resume only from its exact ledger cursor. Resource stops preserve unrelated user programs. Actual visual review and report/final audit follow rendering; the pipeline never marks synthetic/default judgments as reviewed.

GPU scheduling starts with one worker under the user's approximate6GiB process allowance, at least2GiB device headroom, and current measured load. Training and sampling have separate resource checks. CPU OT/scoring use at most2workers x2threads. WDDM per-PID VRAM may be unavailable; PyTorch allocated/reserved do not prove a whole-process bound.

## Completed analysis and delivery

All model/scoring stages are complete. Do not repeat GPU work to read this experiment.
The saved actual review CSVs must exist before `analyse` opens the anonymous key.

```sh
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.c_reference_frame_v1.analyse
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.c_reference_frame_v1.report
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B experiments/r2_c_reference_frame_v1/independent_audit/final_audit.py --final
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.c_reference_frame_v1.finalize
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.c_reference_frame_v1.package
```

These are offline finalization commands, not authorization to alter frozen results. Prefer reading REPORT.md and COMPLETE.json for a sealed run. The initial registration remains unchanged; engineering/001_import_path records the explicit import-path amendment. engineering/002_offline_table_format_attempt.json records a reverted, guarded formatting attempt with no scientific calls. Missing transition categories in historical_review_sensitivity.csv are zero counts, not missing evaluations.
