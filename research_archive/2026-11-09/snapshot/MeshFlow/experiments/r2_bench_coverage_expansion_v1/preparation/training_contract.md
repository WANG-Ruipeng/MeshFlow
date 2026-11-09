# Bench coverage expansion: training-entry audit and proposed contract

Date: 2026-10-21. Status: READ_ONLY_AUDIT_COMPLETE / FINAL_DATA_FREEZE_PENDING.

Execution decision subsequently confirmed by the parent/user: **one arm, 1500 updates and fixed 48-request evaluation; N remains 128..256**. Restore legitimate closed/no-owner task coverage and add qualified held-out-from-conditional-training TRAIN parents where available. Preserve original target geometry; no deduplication. The wide-N analysis in section 4 is a future capability boundary only and is explicitly not implemented or authorized by this run.

This document records source inspection and file/metadata verification only. No generator construction, model forward, CUDA initialization, training, generation, or checkpoint mutation was performed by this audit. It does not freeze an as-yet-unmeasured new data pool or authorize extra arms, probes, retries, or longer training.

## 1. Recommended same-start reference

Use the retained A initial snapshot, not A's trained endpoint and not COMMON's historical AdamW:

- `experiments/r2_bench_adaptation_strength_v1/training/A_COMMON_LR1E5/checkpoint_step0.pt`
- Exists, 521,699,511 bytes. SHA256 actually recomputed during this audit: `75d54c15f3ff2ee1328dddca1269352db48f5ecbca4d459eaff8c603c2297398`.
- Adjacent JSON records arm `A_COMMON_LR1E5`, local step 0, global step 12000.
- Generator state hash: `7f96d80f8ed4cd6f803a4d5b94fb91560497cb1ea134de69eebf5916de57e5ba`.
- Fresh optimizer hash: `6e0db9a3226cd9a460d9f87f8b3ac89b217958a590e2b3704fd3656752278ffe`.
- RNG hash: `5ab3c61d3c7e9e9578bfa19a293c7f33ba4f5cdc81ae59a014027adb8e5a30b4`.
- This audit did not deserialize tensor payloads. Internal state verification remains the actual authorized worker's strict startup check.

`r2/pilots/bench_state_schedule_v1/training.py::import_start` already verifies this exact snapshot, loads the fresh AdamW and RNG, and verifies identities. Reuse its logic in an independent experiment module. Do not call it during CPU preparation: it moves the model to CUDA. Do not call an old experiment's `checked()` or `train()` with a forged root; those deliberately require the old frozen data counts and paths.

The old A endpoint also still exists (`checkpoint_step1500.pt`, 1,565,065,195 bytes); reuse its saved outputs as the control. It is not the new arm's initialization.

## 2. Scientific contract to freeze before training

- One ordinary synchronous SEAM data-coverage arm, 1500 optimizer updates, 12000 per-instance forward/backward calls, local 0..1500/global 12000..13500.
- Exact A0 generator / fresh optimizer / runtime RNG; original Native/T1 + role + Geo + owner-only SEAM, zero new parameters. No PERT, ASYNC, couple, new loss, head retraining or source oracle at inference.
- AdamW: both backbone and condition groups lr 1e-5, betas (.9,.95), eps 1e-8, weight decay 0, clip norm 1. Effective batch 8, microbatch 1, FP32 parameters and integration, BF16 forward, eval mode to disable dropout.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic algorithms with strict errors, TF32 off, SDPA math, numeric threads 2 per training process.
- Role-restricted OT before HYBRID lambda .25, original half-scale OT identity, free scalar-coordinate batch denominator, no anchored or padded coordinates in FM.
- Parent-balanced sampling with four distinct parents per update. Retain A's two instances per selected condition: identity scale and per-axis U[.92,1.08]. Scale complete geometry C and Y together before OT; do not scale Gaussian arrays. Part and variant round-robin within each parent.
- Freeze realized parent/condition roster, source versions, split identities, geometries, metadata, schedule, seeds, augmentation ranges, and code hashes. Changed parent roster necessarily changes examples and their OT; do not claim the old exact data stream is reused. Keep scalar randomness policy and update budget matched, and report actual parent/part/condition exposure.
- Freeze N128..256 unchanged. Increasing parents and condition coverage together tests an expanded data policy, not an isolated effect of parent count or target cleaning. Do not alter native adapters or face-count conditioning.
- Preset final step1500 is the main result. Intermediate checkpoints may provide safe resume, never image-based checkpoint selection. No extra neural fixed probes needed for comparability with the most recent state-schedule experiment; log existing training forward losses by tip/bulk and N/part.

## 3. Closed C and no-owner C are already representable

`r2/pilots/bench_parts_convergence_v1/data.py` supports closed C through `_closed_sample`: no seam slots, ordinary native completion, bulk-only OT and FM. Open C with all-false owner labels uses `seam_abc_v1/slot_data_extension.py::supervised_sample`, producing ordinary bulk slots. The SEAM adapter delegates no-seam samples to NativeModel.

The old CLEAN pool excluded 44 closed TRAIN C and 38 open/no-source-owner C as a data selection policy; this was not proof that the generator cannot train them. No need to cut artificial openings into legitimate closed legs or manufacture owners. Boundary-related metrics remain NA for closed C. Source owner labels are training supervision only and must not enter inference metadata used by the generator.

All cases still require nonempty C and nonempty free variables, consistent nonmanifold-free C index topology, `B_C <= N-K` under the current one-candidate-per-open-edge layout, and unique owner assignment per source free face. Alias faces remain instantiated once, with other boundary slots returned to bulk; a third source vertex already in C remains a free tip. Keep current semantics.

## 4. N beyond 256: explicit local engineering extension is required

The backbone already has `max_length=800`, face_bin 20, no absolute sequence position embeddings, and dynamically shaped arrays. However the maintained conditional interfaces impose explicit 128..256 gates:

- `r2/src/meshflow_control/models/native.py::NativeModel.forward`, lines 228–229.
- `r2/src/meshflow_control/seam/model_adapter.py::seam_forward`, lines 73–74.
- `r2/src/meshflow_control/seam/slot_data.py::edge_features`.
- `r2/pilots/bench_parts_convergence_v1/data.py::edge_features`, also delegating to the legacy gate for open C.
- Historical case/preparation checks and CLEAN `make_schedule` freeze old counts and are not generic new-pool entrypoints.

Implement any wider-range adapter only under the new experiment package, without changing maintained model, original scorer or historical packages. It must handle both mixed-SEAM and ordinary no-seam forward branches; merely relaxing the seam guard still fails when it delegates closed/no-owner examples to NativeModel. Preserve original parameter names/state loading and exactly delegate N128..256 paths whenever possible. Test data packing, masks, N conservation, source face/corner round-trip, anchored target equality and no-source-information inference on CPU, without unbudgeted model calls.

The count embedding has 40 ordinary classes (`800//20`) plus a CFG null row. With `y//20`, **N=800 reaches the null row**. The official data loader uses N<800. Therefore a new allowed upper bound must be <=799 (or a smaller frozen cap such as 512); do not reinterpret N=800 as a normal trained count class.

The C-only occupancy features contain K/256 and N/256. For the fixed old evaluation panel these inputs remain unchanged and cached layouts are reused. They do not affect generator training if no head is trained/called; the training code need not calculate unused `head_features`. If future new-N evaluation calls the head, its extrapolation is a separate contract requiring disclosure, not a silent renormalization to N/800.

No actual wide-N GPU memory measurement exists in the reviewed recent runs. Attention, Geo and OT workloads increase with N; do not infer safe concurrency from the low-N peak or model-file size. Do not change microbatch/precision just to fit more workers.

## 5. Fixed evaluation and unchanged postprocessing

Reuse `experiments/r2_bench_adaptation_strength_v1/eval_manifest.json` exactly: 48 requests, 12 parents, two C per parent, two fixed noise arrays. TRAIN4/16 outputs, DEV8/32 outputs; open-interface denominators TRAIN14 and DEV26. Seven DEV parents are official test; one is official train but conditional-training held out. Keep each DEV parent excluded from the expanded training pool, including alternative simplifications and exact duplicate identities; disclose near-family uncertainty if not checked.

Reuse `OLD_S15000` C-only layout caches, original C/N/sidecars/noise, camera and cross-section positions. Read `bench_state_schedule_v1/eval_plan.py::cached_layout` as the checked example. One final48 RAW job set, barrier before postprocessing, then exactly48 single official Bench D0 calls; unchanged anchor restoration, GEO_C/source orientation, HINGE_OPP015, ALPHA and HOLE. Use old A saved scores/images as read-only reference.

Report RAW, BASE_POST and FINAL separately, full per-output/per-parent metrics and actual AI_PROXY review including the established position sections. Retain face/area survival and invalid-pair counts beside CF/FF/I ratios, condition use/replacement, visible broken supports/layers, valid C/F and seam–bulk edges, reachable bulk area, closed-C NA reasons. The source mesh is a training example/reference completion, not the unique allowed generated style.

## 6. Resource and runtime evidence, not a promised runtime

Closest ordinary training reference A1500: 2819.92 s with concurrent workers. Latest PERT/ASYNC1500: 2713.1/2982.3 s with concurrent contention. These are not isolated benchmarks and do not predict wide-N runtime.

- RTX5080 16303 MiB; recent low-N training allocator peak allocated 2809.3 MiB, reserved 3206.0 MiB (per real update including backward/AdamW).
- Recent state-schedule ASYNC reserved peak3224 MiB. Full-device observed peak10233 MiB included two workers/background processes.
- WDDM per-process VRAM was NA; prior dispatch planning used4500MiB per training worker as a conservative estimate, not an observed process peak.
- Historical final A48 RAW257.33 s, D0+GEO47.56 s, CPU RAW scoring44.15 s and CPU POST chain/scoring278.65 s, with overlap/competition.

Before dispatch inspect actual GPU/free/compute/processes plus CPU/RAM. With a wider N pool, start the already-authorized single arm, observe its real early updates including high-N examples and optimizer allocation, and keep >=max(2GiB,10%totalVRAM) extra free headroom. No synthetic GPU probes. CPU data checks can overlap only with bounded CPU threads. Numerical/OOM failure or a saved cursor trailing completed work must be preserved and handled under an explicit recovery contract; no silent replay.

## Source files inspected

`AGENTS.md`, `WORKSPACE.md`, `native_t1/docs/control_metrics.md`, `r2/docs/research_route_memory.md`; recent Bench reports and execution summaries; adaptation-strength training/common; state-schedule training/eval_plan/registration; clean-multipart data/head; multipart data; owner-only slot extension; Native/SEAM model adapters; backbone count embedding and MODEL_CONFIG.
