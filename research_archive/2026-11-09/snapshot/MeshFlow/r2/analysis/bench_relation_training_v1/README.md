# Bench relation-training offline analysis

This independent module reads the completed `results/per_output.csv` and its
authenticated collector receipt for `R2_BENCH_RELATION_TRAINING_V1`. It imports
no pilot, neural network, geometry scorer, or GPU library, and changes no frozen
training files. It will wait without writing an incomplete result if collection
is unavailable. Old adaptation `results/` is superseded; the new collector
already uses the corrected V2 schema, retaining whole-mesh `boundary_edges` and
separate input `C_boundary_edges`.

After all 432 score rows have been collected:

```text
python -B -m r2.analysis.bench_relation_training_v1.run --check-inputs
python -B -m r2.analysis.bench_relation_training_v1.run
```

Outputs are isolated in the experiment's `analysis_relation_v1/`:

- `all_metrics_summary.csv`: every registered original numeric field, ALL,
  historical TRAIN and DEV, also split by part. Primary averaging is noise →
  condition → parent; request mean, median, extremes and sum are separately named.
- `key_metrics.csv`: wider summary for original CF/FF/I, absolute illegal pairs,
  actual and deleted faces, surface area, thin/degenerate triangles, islands,
  topology and seam/bulk reach. Original fractions stay in 0..1 units.
- `I_support_summary.csv`: original values with all recorded support, PASS-only,
  LOW_SUPPORT-only and other-status strata; no missing values replaced by zero.
- `paired_per_output.csv`, `paired_metric_summary.csv` and
  `paired_per_parent_all_metrics.csv`: both RELATION−USER and RELATION−A,
  paired on the identical request and stage. I additionally reports the
  intersection where both sides are PASS. Increase/decrease are numeric signs,
  never automatically labelled quality wins.
- `per_parent_all_metrics.csv` and `per_output_with_membership.csv`: all original
  numeric values and current pool membership without relabelling historical TRAIN.
- `training_windows.csv`, `training_summary.json` and exposure tables: fixed
  100-update windows, original scalar SSE/denominator aggregation, actual parent
  and condition exposures. One-step endpoint errors aggregate the recorded
  `endpoint_sse`, `bulk_endpoint_sse`, and `seam_tip_endpoint_sse` directly,
  divided by their corresponding recorded effective scalar counts. They are
  not reconstructed from velocity MSE and an average flow time. These curves
  do not declare convergence.
- `definitions.json`, `completion.json`: accounting rules, input/output hashes,
  zero model/scoring calls, and exact original field list.

No unique source shape is used as a generation success target. No visual adoption
labels are invented from metrics. The existing 30% bulk threshold remains
descriptive, not a new acceptance gate. Original CSV, scores and reports remain
unchanged. A second run verifies completed output identities; changes require a
new version rather than overwriting the first result.

CPU accounting checks:

```text
python -B -m unittest r2.analysis.bench_relation_training_v1.test_aggregation
```
