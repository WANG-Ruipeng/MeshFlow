# R2_OWNER_LAYOUT_V1 frozen design

2026-10-17, before any head forward/update. User authorizes independent C-only owner-layout learning and conditional frozen generation, not automatic generator continuation.

## Scientific contract

Closest routes: R31/SEAM occupancy, ABC B_POOL47, and r2_seam_followup_v1. Last experiment showed a causal contribution of inference layout within frozen B, not sole blame on head capacity, proof of global generalization, or full geometry convergence. >=30% reached bulk area is descriptive, not complete task success. Historical R07 and deterministic companion slots are outside this test.

Source TRAIN47 has 400 C open edges: 360 independent owner, 34 true Null, 6 supported alias. Train binary owner-active vs inactive (Null or alias). Alias audit does not mean prediction of partner IDs or alias decoding. No geometry/source repair. Existing boundary order and owner convention retained. C geometry alone need not uniquely determine a source completion's roles; source accuracy is a supervised-layout diagnostic, not universal completion correctness.

## Head budget and comparison

Two CPU-only models, independent fresh AdamW and fixed seeds. MLP is diagnostic comparison; CONTEXT is the preselected generation candidate, no post-result fallback to MLP.

- Input: original C-only 24 edge features plus C boundary-order rank/(E-1) and E/32. Both models receive identical features. N is already an original input. No parent IDs, Y, source face IDs, or labels in forward. Rank depends on local input vertex-ID ordering; no claim of vertex-renumbering invariance. Features standardized from TRAIN47 only.
- MLP: 26->32 SiLU->1 (897 parameters). CONTEXT: projection26->32, two residual attention blocks (4 heads, FF width64, LayerNorm, no dropout), output1. No separate global source features.
- Each performs exactly 2000 full-batch TRAIN47 optimizer updates, 47 parent exposures/update, 94000 total parent exposures. Loss: unweighted BCE per edge, first mean within each parent then mean across47. This is equal parent weighting, not artificially enforcing a 90% output prior. LR0.001, weight_decay0.0001, fresh AdamW. Fixed init2026101701. Output bias initialized0. No temperature/threshold/hyperparameter scan, no early stopping/best checkpoint.
- CPU FP32 parameters/optimizer, BF16 autocast forward, FP32 BCE; deterministic algorithms, one CPU thread/process. No CUDA calls. Two independent head processes can overlap after resource check. Final step2000 only; losses logged from existing training forward, no extra train probes.
- One terminal inference per model on packed TRAIN47 and one on packed DEV3. Save every edge logit/probability, predicted owner, target role, parent confusion, source-role error and calibration diagnostics. DEV labels never enter optimizer, normalization, candidate selection or threshold. These are already observed DEV parents, not fresh test data.

## Predefined gate (CONTEXT only)

TRAIN: owner micro recall>=90%; true Null specificity>=80%; alias specificity>=50%; each familiar parent73/3f/85/8c owner recall>=90%; all4 familiar true Null edges inactive. TRAIN accuracy alone is fitting evidence.

DEV final veto: each of2d/610/608 owner recall>=90%, its single true Null inactive, at least2/3 alias inactive. With small denominators this means all20 DEV owners recalled. DEV is evaluated once and can stop dispatch, never trigger refitting or threshold tuning. Preserve failed gates, no retry/extra updates.

Calibration evidence: unweighted NLL, Brier, fixed10-bin ECE for source owner labels, plus per-role confusion. No learned temperature or claim of calibrated probabilities from in-sample fitting.

## Conditional frozen generator budget

Only if all gates pass: CONTEXT-predicted7 layouts saved once from C-only terminal predictions, identical for frozen LONG and B_POOL47. Original ABC28 requests/arm (TRAIN16,DEV12), same C/N/noise/query. 56 RAW Euler50=2800 generator forwards, then56 single officialD0, original GEO_C/source winding. No generation until all56 RAW saved. No new head calls in generation; no source labels consulted. Preserve original POST and separately apply1.5%L actual-cluster midpoint binding (56 CPU cells), using previous definition with strengthened current-root C-anchor guard; no2% sweep. No additional generator training, A loss, oracle/all-on arm or generator checkpoint selection.

Compare cached LONG/B and B_LONG_LAYOUT, separate TRAIN/DEV and RAW/originalPOST/boundPOST; all124 original numeric metrics, exact legality/tearing/deletion ledgers and actual nonblindAI_PROXY review. If gate fails, mark generation/post/binding NOT_RUN_GATE_FAILED and deliver head weights/results/report/recovery entry without interpreting nonexistent geometry.

## Resources, failures, stop

Initial measured RTX5080 16303MiB total/13000MiB free, GPU5%, no listed compute process;24CPU,65.88GB RAM available. Head stages use2 CPU processes x1thread, noGPU. Check again before any GPU wave; keep>=2GiB headroom, launch incrementally and observe representative authorized sampling. Prior comparable generator allocator peak0.60GB/D0~1.9GB are not processVRAM estimates; use conservative process allowance and live total occupancy. No unrelated process termination.

Preserve original sources, scorers, weights, raw, reports. Separate output/code directory. Failures logged; no silent retry or changed scientific settings. Once gate-failure report or prescribed paired generation completes, stop. No default change, commit, push or cleanup.
