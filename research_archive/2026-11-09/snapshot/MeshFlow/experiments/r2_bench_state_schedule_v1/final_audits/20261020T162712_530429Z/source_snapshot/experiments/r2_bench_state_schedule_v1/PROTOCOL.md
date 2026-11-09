# R2_BENCH_STATE_SCHEDULE_V1 — frozen endpoint comparison

User authorized Idea 1 and Idea 2 separately, each 1500 updates, with the completed Bench A as control. No combined arm, Idea 3, longer training, checkpoint selection, default change, commit or push.

## Shared contract

Import the exact retained A_COMMON_LR1E5 step0 snapshot: COMMON12000 live generator, fresh AdamW and runtime RNG. Both arms replay A's 1500×8 row stream, original 58 TRAIN parents/195 conditions, original owner-only supervision, OT then HYBRID .25, independent model/optimizer instances. Both parameter groups use lr1e-5; betas .9/.95, eps1e-8, decay0, clip1. Batch8/micro1. Strict deterministic BF16 forward, FP32 parameters/moments/integration, math SDPA, TF32 off. Checkpoints 0/final1500 plus replaceable rolling safe checkpoint every250; no intermediate generation or new neural probe.

The completed A final48 is reused as synchronous baseline, with original hashes and evidence. TRAIN and DEV are reported separately; DEV is already observed development data, not a new test set.

## Idea 1 — PERTURBED_STATE

For existing t in [.3,.9], independently corrupt with probability .25. Stable row-hash-derived draws do not advance any original RNG. Draw alpha in [.3,.8]; d=alpha(mu_C-mu_Yfree), with C mean over original C face corners and Yfree mean over only trainable target corners. Preserve all anchors. Set Xtilde=X+t*d on free corners; corrective target Ucorr=(Y-Xtilde)/(1-t). Use the same free-scalar-denominator squared regression; no additional loss or parameter. Non-corrupted rows retain byte-identical arrays. Inference remains original synchronous Euler50.

This is endpoint corrective regression on deliberately perturbed states. It is **not** the true derivative of the constructed corrupted interpolation, and is not claimed to inherit a conditional-flow-matching marginal theorem. Its remaining-time denominator may amplify targets near .9; log actual target ratios and gradient norms. Centroid translation does not certify a semantic duplicate-seat negative. The original HYBRID distribution is not proven to lack these states.

## Idea 2 — ASYNC_LEAD_SEAM

s(t)=min(t/.7,1), bulk time=t. Preserve original x0, x1, OT identities and random stream. Only active seam tip interpolation changes to (1-s)x0+s*x1. Network predicts local-clock velocity U=x1-x0. Tip direct loss applies only for t<.7; bulk loss always applies. Denominator remains original free scalar count. Report active counts; t is logistic-normal, so active exposure is not assumed to be70%.

Attention times: original C=1, seam=s, bulk=t. Corner modulation: fixed anchors=1, free seam tip=s, bulk=t. Original role identities and Geo's original C mask remain unchanged. No generated seam becomes user geometry.

Euler50 advances bulk by dt*v and tip by delta_s*v. The first35 updates integrate seam to its generated endpoint, then freeze it for15 steps. No target coordinates enter rollout. This uses fewer seam integration steps than baseline50 and can freeze a wrong tip. Training later sees true tips, rollout sees generated tips: this mismatch is a reported limitation, not a guarantee of connection. Endpoint diagnostic is X+(1-s)v for active tips, X+(1-t)v for bulk; late teacher-forced zero tip error is excluded.

## Evaluation and stopping

Final1500 only: 48 fixed requests per arm, exactly the baseline C/N/layout/noise/source-winding/query/camera/cross-section identities. 96 new RAW (4800 generator forwards) then, only after all RAW saved, 96 single official Bench D0 calls. Same GEO_C/source winding and CLEAN_FINAL_POST HINGE_OPP015/ALPHA/HOLE chain. Preserve RAW/BASE_POST/FINAL; 288 new score rows and144 baseline rows.

Full per-output/condition/parent metrics retained: use/whole shape visual axes, legal CF/seam-bulk edges and bulk reach, CF/FF/I, small angles, islands, deletion and surface distances. I is C-domain conflict coverage, not whole-mesh Ri. Closed C has interface NA. Unique source completion is not the success criterion. Review actual same-position section images as AI_PROXY, not human review. No favorable-only metric reporting.

CPU tests check corruption identity/masks/RNG, async path calculus/local versus global units, fixed denominator, no-seam/tau1 fallback and post35 tip freezing. First real authorized training update verifies backward and resource peak; no synthetic GPU probe. Dispatch second arm only after measured headroom and compute/CPU capacity allow. Log resources and actual concurrency; no unrelated process termination.

Preserve errors and incomplete budgets. Safe exact-cursor pauses may resume with no duplicated scientific calls. Failed/pending model calls, OOM, lost updates or numerical changes cannot be silently retried. Fixed task ends after final results and report.

## History boundary

Closest evidence: R31 shared seam endpoints; LOCAL_NOISE changed tip initial distribution; COUPLE changed endpoint one-ring loss; Idea4A CPU checked endpoint-preserving bridges. Neither candidate is claimed to be a universal first. This test changes off-path supervised state coverage or per-role clocks, respectively, while retaining the same geometry representation. Positive endpoint pairing would support continuing that particular mechanism; unchanged/negative geometry does not disprove every corrective or asynchronous model.
