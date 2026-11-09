# Bench control transfer pilot

User authorized retiring redundant negative checkpoints and training other local
MeshFlow object categories, with approximately four hours available on
2026-10-18. First execution: 09:12:32 UTC; target closeout: 13:12:32 UTC.

This is category-specific adaptation of the same control method to Bench. It is
not zero-shot transfer of a Chair checkpoint and does not measure whether the
official pretrained prior has seen the held-out Bench parents.

- Use existing official Bench generator EMA; identical Native/T1 + role + Geo
  initialization for PLAIN and SEAM; fresh AdamW in both arms.
- Target 64 TRAIN and 8 DEV independent parents, one existing N128..256 mesh
  each, one coherent regular input surface with K>=2. Freeze actual eligible
  counts after the deterministic source audit and input gallery review.
- Each arm: 3000 updates, batch8/microbatch1, original free-coordinate FM,
  role-restricted OT for SEAM, HYBRID0.25, lr1e-5, BF16 forward/FP32 optimizer.
  No additional loss, RL, teacher, RBF, coordinate change or dedup target.
- Balanced parent exposure, shared parent/augmentation/time/noise seed stream;
  different representations necessarily have independent valid OT assignments.
- SEAM occupancy uses only C geometry and requested N at inference; source
  owner/alias labels are supervision and audit information, never DEV layout.
- Fixed endpoint at step3000. Intermediate checkpoint is recovery evidence,
  not a best-checkpoint selection pool. No adaptive metric-based extension.
- Fixed panel: 4 TRAIN + 8 DEV parents, one C each, two registered Gaussian
  noise arrays. Both models see identical input arrays. Total48 RAW outputs.
- Euler50, then one official Bench D0 per output, restore C/seam anchors,
  unchanged GEO_C and source-winding export. All RAW precede first D0.
- If official Bench D0 is unavailable, report that stage BLOCKED and retain
  RAW; never substitute Chair D0 without declaring a different experiment.
- Preserve full original objective quality/connection fields and per-output
  results. No unique-source reconstruction ranking. Actual AI_PROXY review
  records are separate; render shared-scale views and condition-position cuts.
- At least max(2GiB,10%total VRAM) free headroom; start one authorized worker,
  measure representative training process peak/throughput before considering
  another worker. Preserve unrelated user workloads.
- Log failures and actual work; no silent replay, no science changes to fit
  concurrency, no automatic commit/push or maintained-default switch.

Closest retained route is R31 SEAM plus the PLAIN_S2000 sufficient-training
follow-up. The substantive change is object category and fresh official
initialization, with paired ordinary conditional control. The observation of
interest is whether SEAM increases legitimate C/free and free-body attachment
on held-out Bench parents at acceptable geometry/visual cost. This limited pilot
cannot certify arbitrary-category or arbitrary-user-input generalization.
