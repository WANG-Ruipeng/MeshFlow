# T_geo pilot: evidence and limits

This note summarizes a completed local experiment. Publishing its implementation and this note performs **zero new optimizer updates or generations**. The maintained candidate is the per-face **T_geo (P=I)** branch; T_graph was evaluated but is not retained as a maintained public workflow.

## What was compared

All three arms started from the same historical pure-FM A_continue checkpoint at cumulative 1500 updates, then received 500 fresh updates each. T_base had no new encoder. T_geo added a 149,888-parameter C-only encoder, with a zero-initialized exit, per-face 13-dimensional features and P=I. T_graph had matched parameters/initialization but used C-internal adjacency. All arms trained all parameters with the same free-only FM loss, fresh AdamW and identical seed 1010 input streams. No guidance, new loss or mesh repair was used.

The historical parent checkpoint SHA256 was:

```text
d8366d8c172407ec97ee4288143863825fae583d2c879d200f3862315747e400
```

That file is not distributed. Rebuilding a pure1500 model from the official EMA with public commands creates a new identified checkpoint; it does not reproduce this file identity by declaration. The measurements below apply to the historical experiment, not automatically to newly trained weights.

The task used two already seen chair parents with 112 faces. Evaluation used each parent's original K12 patch with 4 fixed independent noises (9701–9704), plus 2 offline-selected new K12 patches per parent with 2 noises each (9701–9702). New patches were selected from GT topology before generation, without output-based selection. They were not used in this fine-tuning round, but their parents and full targets were already seen. Four model states, including unchanged T0, produced 64 formal outputs; 4 additional full samples were fixed replays. The experiment performed 1500 new optimizer updates total.

## Surface accuracy

Errors are free-surface symmetric RMS divided by the full-target bbox diagonal L, expressed as percent. Differences below are newer minus reference; negative is better. The median is the median of paired differences, not a subtraction of output medians.

| Comparison | Original patches: mean / median difference | Improved | New patches: mean / median difference | Improved |
| --- | --- | --- | --- | --- |
| Pure T_base − starting T0 | −0.1352 / −0.0323 percentage points | 5/8 | −0.1095 / −0.0494 | 5/8 |
| T_geo − matched T_base | +0.1627 / +0.0155 | 3/8 | −0.4066 / −0.3197 | 8/8 |
| T_graph − T_geo | −0.1181 / +0.0018 | 4/8 | +0.4433 / +0.5618 | 3/8 |

The new-patch T_geo signal was not caused by a single output: after removing any one sample within a parent, its mean paired difference remained negative (P0 range −0.6380 to −0.2647; P1 −0.4064 to −0.2480 percentage points). This is a small-sample sensitivity check, not significance or unseen-object generalization.

Original-patch behavior was less favorable. P0 seed 9703 worsened from T_base 0.1560% L to T_geo 1.1420% L. That one output explains 99.51% of the P0 original-patch total error reduction from T_geo to T_graph; without it, the remaining mean difference is only −0.0016 percentage points. All outputs remain in the reported statistics. The original-patch P1 T_geo error worsened on 4/4 noises relative to T_base.

All four model states already showed positive two-direction binding in the fixed same-noise cross-context comparisons. These observations do not establish that adding the encoder created conditional binding from scratch.

## Interface and mesh-quality tradeoffs

Better free-surface RMS did not consistently improve interfaces. On new patches, T_geo versus the matched pure baseline showed:

| Parent | CF affected-free-face fraction | FF illegal-pair count | Boundary Gamma RMS | Boundary coverage |
| --- | --- | --- | --- | --- |
| P0 | +2.25 percentage points; worse 3/4 | Mean −8.75; fewer 4/4 | Mean −0.0528 percentage points; better 1/4 | −14.90 percentage points; worse 3/4 |
| P1 | −1.25 percentage points; worse 0/4 | Mean −4.25; fewer 2/4 | Mean −0.3979 percentage points; better 3/4 | +29.24 percentage points; worse 0/4 |

CF means verified proper intersections between C and free faces. FF counts verified illegal intersections among free faces. A separate CPU audit detected illegal intersections in **64/64 formal outputs**. No output had an exact C/free shared vertex or edge under exact-coordinate indexing. Nonzero conditioning responses, clamped C, or zero new free-related vertex-link anomalies do not establish watertight seams. Exact-predicate certification and continuous collision safety were not run.

T_geo is retained as a **positive candidate for further validation**, with clear interface tradeoffs. It is not consistently better on original and new contexts, not a proven new-object generalizer, and not an industrial mesh-validity solution. Adjacency did not show a reliable incremental advantage in this budget. No automatic larger training or repair step follows from publication.

## Numerical boundary

Production used BF16 backbone forwards, FP32 parameters/integration, deterministic algorithms, TF32 off and math SDPA. On a fixed diagnostic state, same-input repetition was bitwise identical. Combining face and corner permutations nevertheless exceeded the registered BF16 tolerance: 10/1008 velocity coordinates failed `atol=0.015625, rtol=0.03125`, maximum absolute difference 0.064453125 and RMS 0.0088499709.

Zero-initialized new branches reproduced the corresponding T0 outputs, including this inherited error. The authorized experiment kept this numerical FAIL visible and used independent hard checks for baseline regression, zero-branch identity, feature permutation, gradients, hashes and all 51 clamped states. It did not relabel the BF16 comparison as PASS.

Separate FP32 diagnostics reduced the combined-permutation maximum difference to about 1.1563e-5 and RMS to 1.4645e-6, passing their joint `atol=rtol=1e-5` criterion. The first observed differing BF16 first-block site was the SDPA output before projection; this does not identify a unique internal SDPA operation. Production did not switch to FP32.

Public training/preflight does not import old private regression evidence or grant another checkpoint that experiment's exception. Its documented zero-update preflight is not a promise of exact permutation invariance. New environments and independently retrained models require their own validation.