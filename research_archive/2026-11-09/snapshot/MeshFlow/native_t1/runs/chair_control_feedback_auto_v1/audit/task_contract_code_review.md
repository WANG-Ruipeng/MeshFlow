# Independent contract/source review

REVIEW_COMPLETE_ACTIONS_REQUIRED_AT_STAGE_BOUNDARIES

No current scientific violation was observed. The following concrete stage/resume guards need attention before their affected actions; the running exploration FM stage does not need to be discarded.

## GPU_TIME_GUARD

GPU occupancy is checked only before the worker. Training checks wall deadline but sampling has neither deadline. A worker starting below8hours could cross8hours before the next startup check. The current4hour wall window makes this unlikely for this run, but the8hour hard requirement is not enforced locally.

Action: Check accumulated occupancy plus current lease time before each update and rollout; stop at a complete update, save complete state, and leave enough reserved time for the next atomic operation. Keep lock/CPU wait out of GPU-time accounting.

Evidence: /mnt/e/MeshFlow/experiments/mf_h1_local0/chair_control_feedback_auto_v1/worker.py:61, /mnt/e/MeshFlow/experiments/mf_h1_local0/chair_control_feedback_auto_v1/training.py:104

## REPEAT_CONFIRM_RELEASE

The repeat phase is selectable without a positive DEV+quality decision. CONFIRM sampling requires only release.status=FROZEN; it does not bind the requested checkpoint to the five permitted model identities or require both independent repeat endpoints. Evaluation additionally requires a candidate hash, but that does not establish all release identities.

Action: Before repeat require frozen DEV decision, selected recipe/step and matching control. Before CONFIRM require frozen release listing exact hashes for START, explore candidate/control and repeat candidate/control; require completed independent repeat state and newseed before allowing any CONFIRM samples/scores.

Evidence: /mnt/e/MeshFlow/experiments/mf_h1_local0/chair_control_feedback_auto_v1/training.py:86, /mnt/e/MeshFlow/experiments/mf_h1_local0/chair_control_feedback_auto_v1/worker.py:15

## UNTIL500_THEN1000

A successful until500 call creates complete.json with steps500. A later until1000 sees complete exists, skips worker launch, then asserts steps>=until. It does not use the valid500pointer to continue.

Action: Treat completion as satisfying only the recorded step: if completed step<until, resume from the validated pointer rather than assert. Preserve prior completion record as a stage receipt.

Evidence: /mnt/e/MeshFlow/experiments/mf_h1_local0/chair_control_feedback_auto_v1/run.py:25, /mnt/e/MeshFlow/experiments/mf_h1_local0/chair_control_feedback_auto_v1/run.py:31

## MANIFEST_RESUME_IDENTITY

Worker validates START hash in protocol but does not compare current pool manifest bytes with protocol.manifest_sha256 before loading/sampling. Cached result reuse checks checkpoint/raw hashes but does not validate current request bank hash or result request/condition identity. A changed same-ID request could therefore reuse unrelated cached geometry.

Action: Before model load verify frozen protocol and selected manifest hash. For cached results validate request_id/split/checkpoint/bank/condition identities as well as raw and export hashes; current requests can be reconstructed from immutable manifests without any new generation.

Evidence: /mnt/e/MeshFlow/experiments/mf_h1_local0/chair_control_feedback_auto_v1/worker.py:58, /mnt/e/MeshFlow/experiments/mf_h1_local0/chair_control_feedback_auto_v1/worker.py:22

## Verified current learning path

- Correct retained OT_HYBRID pilot1000/total5000 START, strict original model/AdamW state and optimizer step5000 restoration; new current steps are5000+feedback_step.
- Primary8microbatch stream retains free-only original OT, FP32 mixed HYBRID target and lambda.25, 8distinct parents, 4unit+4uniform widths,32train parents. Original FM denominator sums all8 valid free scalar counts.
- Auxiliary replay uses genuine16START Euler50 paths on TRAIN_FEEDBACK only; all16 payload mappings recover original GT exactly and all51C states match. No DEV/CONFIRM score is used to prepare replay.
- Auxiliary CPU scores only select a supervised example. training.update differentiates .1*masked_fm_loss(model_velocity, detached_GT_based_target) with real model backward; model inputs remain xt/t/N/masks. No score constant is presented as a generator gradient.
- Auxiliary x1 retains the true matched GT free coordinates and exact C; collate_aux overrides Xt/u exactly. Primary collation is unchanged. Both auxiliary arms share parent/noise/time and one additional forward/backward.
- Optimizer updates and all forward/backward/sampling attempts are reserved before calls. Global6000update,512rollout,128feedback,25600NFE,64precheckF/32precheckB and160000sample-equivalent forward caps exist. All model batches are1.
- 500/1000DEV sampling is invoked by the training worker, with CPU scoring and galleries driven independently. Shape quality and interval-conservative three-metric gates are separate; severe I/E parent regressions are separately retained.
- Four mesh views plus wireframe, same per-parent frames, C-blue/free-orange including GT, and three metric overlays exist in visualization code. AI_VISUAL_REVIEW remains pending until images are actually reviewed.
- No default model change, no extra generator parameters, no MOMENT/SURFACE/UCorr/endpoint snapping inserted. The current pool is intentionally frozen START and can become off-policy; this limitation is stated in its specification.

Real replay audit:137checks plus576exact auxiliary subchecks PASS. No model/optimizer/sampling call was made by this review.
