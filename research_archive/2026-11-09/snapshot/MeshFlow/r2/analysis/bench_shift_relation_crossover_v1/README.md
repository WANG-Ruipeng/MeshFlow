# Fixed t=.5 crossover — independent CPU evidence analysis

Prepared before candidate sampling/evaluation. This directory imports the
established pure-CPU aggregation functions read-only instead of copying or
modifying previous experiment sources. It contains no generator/scorer calls.
`blind_prepare.py` is independently owned and is not used by these modules.

- `run.py`: authenticate the completed new collector (432 rows, 177 numeric
  fields), preserve all fields/statuses, aggregate ALL/history-TRAIN/DEV/parts
  and each parent, and pair against both frozen references. Original request
  fields must equal the inherited manifest; new prefix metadata is allowed.
- `fixed_sets.json`: exact nine prior DEV adoption gains and eight prior shared
  positives, frozen from the completed previous fresh review. Membership does
  not change if the candidate succeeds or fails. These are evidence subsets,
  not new metrics or newly defined promotion rules.
- `closeout.py`: authenticate 48 actual cached states at index25, all 1200 new
  suffix forwards, 48 D0 calls, actual full-page new review and protected files.
  It requires the new nonblind three-model review and a result-grounded
  `analysis_notes/decision_notes.md`. Before these exist it returns WAITING and
  creates neither REPORT nor COMPLETE. The old two-column anonymous review is
  separate evidence and must not be passed off as a review of the new candidate.
- `test_analysis.py`: denominator/parent weighting, I support, pairing identity,
  immutable output, missing-input behavior and fixed-set membership checks.

CPU commands, with `CUDA_VISIBLE_DEVICES=` and BLAS/OMP threads=1:

```text
python -B -m unittest r2.analysis.bench_shift_relation_crossover_v1.test_analysis
python -B -m r2.analysis.bench_shift_relation_crossover_v1.run --check-inputs
python -B -m r2.analysis.bench_shift_relation_crossover_v1.run
python -B -m r2.analysis.bench_shift_relation_crossover_v1.closeout --check-only
python -B -m r2.analysis.bench_shift_relation_crossover_v1.closeout
```

Output: `experiments/r2_bench_shift_relation_crossover_v1/analysis_crossover_v1`.
PASS_ONLY I is a parent macro; request means remain separately labelled.
Closed-C seam values stay NA. Absolute CF/FF counts, face deletion/retention,
area, all five visual axes and uncertainty remain visible. No checkpoint or
switch-time selection, extra generation, automatic model promotion or push.
