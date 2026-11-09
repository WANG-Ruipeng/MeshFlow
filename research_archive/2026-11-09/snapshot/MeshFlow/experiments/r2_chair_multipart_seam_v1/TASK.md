# CHAIR_MULTIPART_SEAM_V1 — frozen scope

User authorized both Step 1 and Step 2 of attachment
`61baf64c-3eb5-49a4-b2a8-9b6ebe21bece/Pasted text.txt` on 2026-10-19.

This folder owns Step 2. Step 1 is in `../r2_unified_rescue_post_v1/`.

## Fixed training

- One candidate, 1,500 additional optimizer updates. No best-checkpoint selection,
  adaptive extension, extra arm, loss sweep, reward optimization, or teacher.
- Restore complete `PLAIN3000_step1000.pt` (the retained PLAIN_S1000), live global
  14,000, matched generator AdamW and RNG. Endpoint global 15,500. Source SHA256:
  `51a0a60a498c7c1f4f747c0c7c993c9cf2f32cf69558bd90c4ab4383dda8e85c`.
- Existing legal TRAIN SEAT_FULL and BACK_FULL, expected 47 + 31 conditions;
  actual verified manifest governs. No DEV parent enters training. Preserve full
  original targets, face counts, owner/alias rules and source geometry.
- Batch 8/microbatch 1. Each step: two uniformly scheduled seat parents and two
  back parents; each contributes clean and original width-augmented instances,
  giving four seat and four back instances. Exact 1,500-step stream is frozen.
- Original Native/T1+role+Geo, SEAM, role-restricted OT then HYBRID0.25, free-only
  FM and inherited optimizer settings. No coupling or new generator loss.
- New C-only head: original 24 edge features plus C-relative midpoint xyz and
  world height in the actual y-up convention; two width-32/four-head attention
  layers, dropout 0.1, unweighted BCE, fresh AdamW lr .001/betas(.9,.95)/wd .01,
  fixed logit>=0. Train only alongside the registered 1,500 updates. No independent
  head pretraining or DEV threshold fitting. Source owner is a training label,
  never an inference input. Head loss has no gradient into generator.
- Do not infer required interfaces from semantic names or a top/bottom rule.
  No added "bulk-centroid boundary guard"; that heuristic was not sufficiently
  defined and could prohibit valid completion. The named Step 1 geometric guards
  are fully specified separately and retain all measured costs.

## Fixed evaluation

- Original 28 SEAT + 16 BACK requests, 15 distinct C, exact old Gaussian arrays,
  N/K, source identity and 2,048 C query points. No new noise/query draws.
- One final live endpoint: 44 Euler50 outputs (2,200 generator forwards), 15
  C-only head inference calls, 44 single official Chair D0 calls. All RAW before
  D0. FP32 parameters/integration, BF16 forward, strict deterministic math SDPA.
- Save RAW, original D0/GEO/source-orientation POST, and the fixed Step 1 guard
  plus HINGE/ALPHA/HOLE chain. No method selected from results. Compare original
  and unified post stages separately on old and new models where available.
- Full original metrics and denominators, per-output/condition/parent rows,
  actual AI_PROXY surfaces and fixed-location sections. Report SEAT/BACK and
  TRAIN/FAMILIAR/DEV separately. BACK FAMILIAR becomes a seen training condition;
  it is not new-condition generalization after this adaptation.
- Historical S2000 is global15,000 with a different head and training history;
  this global15,500 candidate is a recipe test, not an equal-budget attribution
  to any one of data mixture, head architecture, or training duration.

## Prior evidence and interpretation

Nearest route: R31 shared seam slots and its owner-layout/multi-parent follow-ups;
  keep their bounded positives and contradictory FF/I costs. The new change is
  including verified BACK conditions in SEAM learning while preserving SEAT
  coverage, with a jointly trained geometric-context head. Chair BACK14/16 visual
  role judgments and final2/16 bulk>=30 do not prove catastrophic forgetting or
  that every top edge must be Null. Bench post diagnostics show a D0/GEO thin-face
  mechanism on selected outputs, not that all deleted faces are valid supports.

## Execution

One independent GPU training arm; CPU Step 1 can overlap when measured CPU/RAM
  load permits. Do not create a second training worker or change batch to fill
  VRAM. Inspect nvidia-smi/resources before dispatch, keep >=2GiB additional
  headroom and record observed first-update peaks. Small CPU head contract checks
  are separately counted, not training. Scientific failures preserve attempts;
  do not silently replay a started GPU update/rollout.

Completed checkpoints/caches are reusable through identity-checked recovery.
  Finish fixed task, preserve weights/evidence, stop. No cleanup, default switch,
  commit or push.

## Before-execution engineering notes

1. Input registration initially missed `r2/src` on Python search path; import of
   `source_winding` failed before output/model calls. Local source path was added;
   the same 44 inputs then registered successfully without resampling.
2. Original BACK scoring registration required a redundant `source_npz` field
   absent in older SEAT manifests. The independent adapter now uses the original
   C identity metadata and hashes available files. No geometry, original scorer,
   frozen input, or query was changed; failure preceded scoring/model calls.
3. Static review caught that spawned scoring children would lose the parent's
   experiment-label overrides. The independent `score_dispatch.py` configures
   each child before using the unchanged scorer. An actual CPU spawn-path check
   passed, with zero geometry scans or model calls.
4. Before inference execution, the worker gained a frozen-registration identity
   check and recorded the generator tensor hash. Before any comparison table
   was computed, the summary reader gained checks of old RAW mesh, detector
   output and producer hashes. Its prior code/registration are preserved under
   `engineering/comparison_preexecution_00/`; metrics and requests did not change.

Step 1's CPU same-face-cluster repair and any repeated geometry/scoring work are
accounted separately in its own engineering-attempt directory. They do not
restart this training or repeat any D0/model calls.
