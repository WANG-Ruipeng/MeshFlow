# Complete stored-trajectory readout

108 existing Euler50 trajectories were analyzed on one CPU thread in 4.44 seconds. No model call, new Gaussian, OT, or geometric scoring was used. Definitions, masks, time partitions and the four-request risk gate were frozen in `definition.json` before computation. Actual PNG verification found both figures readable. Full per-output, role/time partition, per-parent and paired tables are adjacent.

S combines deviation from the net direction with nonuniform speed. The path/chord ratio distinguishes monotone speed changes along a straight line from a longer path; it is a ratio in the complete selected free-coordinate state space, not an average of individual 3D corner tracks. Reported orthogonal and parallel terms decompose S. A larger value is not by itself a geometric failure or integration error.

## Current Native → Coverage paired result

Same C, N, fixed noise array, slot identities and free mask for every pair. Means average the two noises before giving equal weight to parents.

| Group | Parents / requests | Native S | Coverage S | Native path/chord | Coverage path/chord | Native orthogonal | Coverage orthogonal |
|---|---:|---:|---:|---:|---:|---:|---:|
| OLD_TRAIN | 8 / 16 | .000878247 | .000948484 | 1.019181 | 1.020459 | .000664650 | .000721228 |
| DEV | 4 / 8 | .001113629 | .001122428 | 1.022898 | 1.023051 | .000809153 | .000826627 |
| NEW_TRAIN | 2 / 4 | .000959484 | .000763876 | 1.021289 | 1.015019 | .000789108 | .000566337 |
| NEW_PARENT_HOLDOUT | 1 / 2 | .000482067 | .000477216 | 1.016949 | 1.017186 | .000420277 | .000423285 |

There is no uniform increase in path deviation after the latest adaptation. The two new training parents have **less** S and orthogonal motion, despite the prior experiment's severe postprocessing surface/face losses. This directly limits the claim that those losses all originate in a newly more tortuous rollout.

The old 6b bad tail is different: parent-equal S .001657598 → .002293796 and path/chord 1.031682 → 1.044895, with orthogonal energy .001147869 → .001617080. It is a specific association worth retaining, not evidence that curvature caused its FF/I change. Other parents offset or differ from this behavior; do not summarize only that tail.

Within NEW_TRAIN, a8 S .001149215 → .000927803 and path/chord 1.025773 → 1.017676; 27 S .000769753 → .000599950 and path/chord 1.016805 → 1.012362. Both parents—not just one noise—lower their parent means. Early/mid/late S contributions for NEW_TRAIN change from .000524036/.000151975/.000283472 to .000520383/.000097029/.000146464. Most of this S reduction is in the mid/late contributions, which does not establish a causal failure time.

## Same old panel, COMMON → Native and Native → SEAM

COMMON and Native use exactly the same layout, C and noise on 24 requests. COMMON → Native increases DEV S .000824767 → .001113629 and path/chord 1.017336 → 1.022898. OLD_TRAIN S increases only .000833690 → .000878247 while path/chord falls 1.020843 → 1.019181. Thus S and path-length ratio can move in different directions; S cannot simply be relabeled “curvature.”

Native → SEAM has different fixed/free corner masks. The following are **per-model normalized descriptive statistics**, not the same-coordinate tensor comparison or a single-variable causal effect:

| Group | Native all-free S | SEAM all-free S | Native all-free path/chord | SEAM all-free path/chord |
|---|---:|---:|---:|---:|
| OLD_TRAIN, 8 parents | .000878247 | .000778848 | 1.019181 | 1.016387 |
| DEV, 4 parents | .001113629 | .001163762 | 1.022898 | 1.023901 |

SEAM role separation is material. OLD_TRAIN bulk S=.000773266, tip S=.001462893; DEV bulk S=.001142408, tip S=.004797411. Corresponding DEV path/chord is 1.023446 for bulk versus 1.080038 for tips. All 4 DEV and 8 OLD_TRAIN parents have valid tip-role denominators. These role scales describe their separate dynamics; they do not certify that tip motion or topology is wrong, and must not be compared to Native's nonexistent tip role (NA).

## Fixed integration-risk gate

| Frozen model/request | B |
|---|---:|
| Native 6b/n0 | .00266989045 |
| Native a8/n0 | .00271951861 |
| Coverage 6b/n0 | .00344235523 |
| Coverage a8/n0 | .00220845714 |

All are below the predeclared .005 threshold. `risk_gate.json` therefore says **NOT_TRIGGERED**; no Euler100 dispatch is warranted under this experiment's fixed rule. This screening proxy is not an integration-error upper bound. Failing to trigger it does **not** exclude integration error, validate Euler50 accuracy, or prove OT is correct. The source-supervised coupling/shared-error probes provide a separate evidence stream.

The old historical CC/UU full trajectories were inventoried and remain available, but are not substituted for current-model evidence or pooled into these current 108 statistics. CU/UC intentional model switches are excluded.
