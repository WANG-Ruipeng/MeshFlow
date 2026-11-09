# Eight-label comparison preparation

Prepared only: 12 new synthetic tests plus 27 existing analysis regression tests passed (39 total, 1.11s, return code0). All105 frozen main/course sources remain unchanged. No actual course score/log/checkpoint data were read; no model, scoring, sampling or training calls ran.

After main and course scoring are complete, from E:/MeshFlow-Control in WSL:

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B tools/compare_recipe_all_baselines.py \
      --out /mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1

Input is main/evaluation and course/evaluation per_output.json plus their phase registration and scoring_receipt. Both actual request-file hashes are verified. All8 labels are bound to the same64 request IDs and actual C/noise/query/denominator identities. No checkpoints, RAW meshes, sourceGT or private model mapping are read.

Output is analysis/all_baselines/recipe_all_baselines_table_comparison_v1_<provenancehash>/:
- all_metrics.json and per_output.csv preserve512 original primary metric rows.
- metric_catalog.csv preserves all directions, statuses, all-NA and nonnumeric fields.
- per_parent_metrics.csv and model_summary.csv cover8labels x2ratios x ALL/original6/extended2 for every metric.
- paired_noise_metrics.csv, paired_parent_metrics.csv and paired_parent_summary.csv cover only the two curricula minus START5000/R1_40_H_ALL/HIST_FM6000.
- provenance.json, README.md, immutable analysis_receipt.json and analysis/all_baselines/latest.json.

Four noise values form each parent mean; parents have equal weights. Missing values do not shrink the primary mean denominator. At20%, original6/extended2 means parent membership from the frozen40% registration, not a new condition cohort. Improvements follow the existing individual metric direction with exact signs; unknown/descriptive directions have NA improvement counts. Uncertainty directions are not quality certification. There is no scalar composite or automatic selection. ReferenceGT/S/E and diversity remain in the earlier separate analyzers and are not redefined here.

Missing input files return PENDING/exit2 without creating analysis output. Completed output resumes only after its artifact hashes match. Any duplicate, failed contract, altered request, C/noise/query mismatch or malformed complete panel raises an error. No source inputs are modified.

Source hashes and snapshots, actual test stdout and the105 frozen-file checks are adjacent.
