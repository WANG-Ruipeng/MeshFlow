# Registered binding V2: preparation and change note

No real course quality/score table or private method mapping was read before this source freeze. This change was tested only on temporary synthetic tables. No model/scoring/rendering/generation operation was executed.

The frozen V1 tool remains byte-identical. Its horizontal comparisons required all models to agree on input/query hashes, but did not bind unanimous values back to the frozen request/query registry. A common wrong hash could therefore pass that limited check. V2 rejects that case before any aggregation.

V2 adds:
- Every row's input hash, inner request ID and label must match the registered request.
- Inner and outer query hashes must equal the query bank registered for that condition.
- Query C hash, parent, ratio, count, shared-noise rule, weight denominator and L0 must match registration.
- All16 query bank file hashes match their frozen registration; no arrays are loaded or recomputed.
- Input/source hashes are stable before/after the unchanged V1 aggregation.
- An independent immutable binding receipt binds the V1 output receipt to these checks.

Numerical aggregation, metric definitions/directions,512 original rows,6 practical-baseline pairs, fixed denominators and no-automatic-selection policy remain unchanged. Old registration/source/results remain untouched.

Tests: 12 new synthetic tests plus39 earlier regression tests,51 PASS in1.43s, return code0. Counterexamples include unanimous wrong input/query hash across all8 models, inner/outer mismatch, wrong C/denominator/request registry, missing cells and altered query file bytes.

After all actual scoring is complete, use V2 instead of invoking V1 directly:

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B tools/compare_recipe_all_baselines_v2.py \
      --out /mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1

Original numerical tables remain under analysis/all_baselines/recipe_all_baselines_table_comparison_v1_<identity>/.
The new binding receipt is under analysis/all_baselines/recipe_all_baselines_registered_binding_v2_<identity>/binding_receipt.json.
analysis/all_baselines/latest_v2.json links both. Missing input files return PENDING without analysis output.

Separate freeze: registrations/all_baselines_analysis_v2_sources.json.
