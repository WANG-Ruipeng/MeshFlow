# R2_SEAM_SLOT_C200_V1 — frozen paired prototype

Scope: FULL4_C200 and SEAM_SLOT_C200, each exactly 200 updates, from the same FULL_S6000 live generator, matching AdamW and restored RNG (global12000→12200). Four seen TRAIN complete seat conditions:73b369/3f7417/85b169/8c9b20. No DEV, graft test, new loss on generator, added rollout arm, or automatic scale-up.

Batch8/micro1: every father once clean and once width-augmented U(.9,1.1), 400 exposures/father/arm. Paired source/time/augmentation/random arrays; restricted OT and Xt are necessarily different. Original OT then HYBRID.25, standard Gaussian inference, Euler50. Source geometry is a supervised example, not the unique correct completion.

Candidate: single-edge owner seam slots, two original C vertex references plus one generated point. Null remains a full bulk slot; requested N is fixed. Mixed face attention uses free t; original C attention uses1. Role/MLP/final are per corner. Generator FM excludes fixed coordinates with actual scalar denominator9(N-K)-6S. Fresh independent C-only24→32→1 occupancy head, 833parameters; weighted BCE, negative8.25/positive66/124, mean over all132edges in each batch; separate AdamW lr.001, beta(.9,.95), wd0, clip1. No FM/head gradient cross-coupling. Generator hyperparameters and optimizer moments inherited unchanged.

The C-only threshold logit>=0 is fixed once per C; four eval noises share the predicted layout. Source occupied labels are used only in supervised training and evaluation, never rollout. All-occupied baseline:62/66accuracy,62/62occupiedrecall,0/4Nullrecall,2/4exactlayout. Source disagreement is not automatically an invalid completion.

Evaluation: four C×four pre-frozen independent noise arrays×two arms=32RAW, then32single officialD0/POST. Save allRAW before D0. Candidate restores C plus seam anchors; ordinary arm restores C. Original GEO_C/source winding are reused; post-retention corner/vertex identity is verified, deleted seams are not revived. This is a whole-system comparison, not a loss-only ablation. No candidate step0 output is generated or replaced by historical FULL output.

Shared metrics: RAW adoption/body review; legalCF edges/length, overall free reachability, originalCF/FF/I, angles/degeneracy/AVMR/PMR. Candidate diagnostic: retained seams with legalCF as seeds, grow through only legalfree adjacency, reachedbulkarea/retainedbulkarea; excludeCfrompaths and seamfromareas. Report absolute areas, counts and seam-bulkedge length. FULL has no seam role:NA. Emptybulk:NA. A30% marker is not mesh certification.

Yhat diagnostics reuse normal training forwards only, time-binned. Source winding restored for normals; degeneracy invalid. These are target-containing training-state fits, not rollout success. Nearestbulkvertex distance is not sharededge evidence.

Nearest prior route:R10 partner closure reached teacher-state success without rollout closure. Here shared C identities are imposed from the initial state and occupied/Null is predicted from C. The remaining hypothesis is whether seam and bulk jointly organize into a useful body. Success on four trained C does not establish unseen-parent generalization, multi-edge owner support or third-point-C references.

Runtime: strict deterministic/mathSDPA/BF16 forward/FP32 parameters+integration. CPU OT2workers. Start one authorized training worker, observe backward and optimizer memory, then remeasure before a second; retain>=2GiBheadroom. No unrelated user process termination. Callcounts, failures and safe checkpoints are recorded. No silent retry, changed settings, bestcheckpoint selection, cleanup/defaultchange/commit/push.
