# Frozen occupancy audit

Completed without head/generator/D0 forward, optimizer update or GPU compute.

The executed loss is the all-valid-edge mean over each 132-edge update:

`L = mean[-w1*y*log(sigmoid(z)) - w0*(1-y)*log(1-sigmoid(z))]`, with `w0=8.25` and `w1=0.532258064516129`. Each microbatch edge mean is multiplied by its edge count /132. The implementation agrees with the frozen final protocol. Prototype DESIGN.md's older per-condition-average description is not the executed reduction.

At the population optimum for original probability p, weighted sigmoid q satisfies `q=w1*p/(w1*p+w0*(1-p))`. Thus the analytically deweighted logit is `z-log(w1/w0)` and p>=0.5 corresponds to raw z>=-2.740840023925. This identity does not certify empirical calibration after 200 updates. The old z>=0 is a legitimate cost-sensitive rule, corresponding at the ideal optimum to original p>=0.939393939394; it is not an unweighted 50% rule.

| Policy | TP | TN | FP | FN | Positive recall | Null recall | Exact C layouts |
|---|---:|---:|---:|---:|---:|---:|---:|
| ORIGINAL_ZERO | 36 | 4 | 0 | 26 | 0.580645 | 1.000000 | 1/4 |
| ANALYTIC_DEWEIGHT | 62 | 0 | 4 | 0 | 1.000000 | 0.000000 | 2/4 |
| ALL_OCCUPIED_REFERENCE | 62 | 0 | 4 | 0 | 1.000000 | 0.000000 | 2/4 |

The fixed analytic threshold activates all 66 edges: positive recall rises from36/62 to62/62, but all4 source Null edges also activate. It exactly equals the all-occupied reference on this panel. It does not demonstrate a better source-layout classifier, eliminate mismatch, or establish reduced FF/I. A rollout would test a real inference-layout intervention, with the original C-only inputs and no source-label oracle. Four C, not16 independent noise trials, are the occupancy denominator.

All16 saved RAW prediction arrays exactly agree with the per-edge table, and all source labels agree with saved NPZ labels. Existing checkpoint bytes are hashed but never loaded. The executable exposes `calibrated_occupied(logits, edge_mask, negative_weight=..., positive_weight=...)`, which accepts no Y/labels. Original model/head/threshold/defaults remain unchanged.
