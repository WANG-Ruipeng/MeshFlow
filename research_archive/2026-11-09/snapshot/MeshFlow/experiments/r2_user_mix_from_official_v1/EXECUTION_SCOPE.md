# R2 USER-majority from official: fixed single-run scope

Authorized by the user on 2026-10-10. Experiment: R2_USER_MIX_FROM_OFFICIAL_V1.

One model only: same official Chair EMA, Native/T1 + role + Geo, fresh AdamW, conditional update 0 to 6000. No pretrained unconditional training is repeated. Model and optimizer initialization are audited, without claiming conditioned-output equivalence to the unconditional model.

Data: the existing 25 common parents; 117 active USER conditions (56 seat / 61 back) plus 395 OLD conditions. Each batch contains 8 distinct parents: 3 USER clean, 3 USER width-augmented, 1 OLD clean, 1 OLD width-augmented. OLD has one C20 and one C40 request per batch; their augmentation assignment alternates by block. Every parent has 1920 exposures (1440 USER / 480 OLD). No HOLD/DEV enters training. 48000 is exposure count, not new meshes. Known source geometry defects remain disclosed.

The sample-slot ratio is USER75% / OLD25%. It is a pilot choice, not an optimized proportion. The inherited whole-batch free scalar normalization remains unchanged. Frozen plan counts 61961499 USER and 15072480 OLD free coordinates (80.434% / 19.566%); these are coordinate counts, not measured gradient contributions.

FM only, HYBRID lambda0.25 throughout; original free-only OT before independent epsilon mixing. Batch8/micro1, lr1e-5, AdamW betas0.9/0.95, weight decay0, clip1. BF16 forward, FP32 parameters and integration, strict deterministic operations, TF32 off, math SDPA. No B, teacher, readout, loss changes, additional arms, or intermediate checkpoint selection.

Evaluation: 6 existing USER DEV conditions, two fixed Gaussian inputs each. Only 12 new Euler50 RAWs (600 sampling forward calls). Reuse 36 strictly verified historical RAWs/scores, for 48 total records across four models. No new OLDREG generations, postprocessing, unique-source GT ranking, or old-task promotion gate. 24 paired-color/neutral gallery PNGs, common cameras. Review whole-shape plausibility and C-use independently, marking uncertain cases and AI_PROXY rather than human judgments.

Interpretation: only 2 seat and 4 back DEV conditions, previously observed. Historical R2 trained on26 parents for6000 steps, continuations total7000 steps on25 common parents. These are practical existing references, not a matched causal ablation of the 75/25 mixture or a new independent test set.

Checkpoints: one rotating recovery checkpoint every500 updates, one immutable full final6000 checkpoint and one generator export. Safe interruption can resume only at the exact completed update/stream/RNG/AdamW cursor. No replay after failed or ambiguous calls; no automatic reduced-precision/batch retries. Fixed budget ends after scoring, gallery and report; no default switch or git commit/push.

Runtime: two CPU preparation workers; only one authorized training arm. GPU resources are measured before dispatch and observed during real updates. CPU scoring may use two independent workers after RAW completion. No extra probe training or sampling jobs.

Engineering setup: the first controller prepare-data command failed before child dispatch because a Path object was not JSON-serializable in the event log. Fixed before any OT/model call from that command; source frozen afterwards. This is not a training or generation retry.