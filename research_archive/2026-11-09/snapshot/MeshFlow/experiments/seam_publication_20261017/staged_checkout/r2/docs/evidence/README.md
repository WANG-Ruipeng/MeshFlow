# Compact research evidence

This directory accompanies [research_results.md](../research_results.md). It contains **all rows** of each selected source CSV, including adverse outputs. It is not the complete local metric archive and does not claim a new experiment.

- `r2_recipe_*`: 640 outputs across main, course and confirmation phases; the derived summary keeps phase/model identity and averages noise within parent before equal-parent aggregation.
- `user_finetune_*`, `user_mix_*`: 84 continuation outputs and the entire 48-output USER comparison, plus existing nonblind AI_PROXY text observations. Images are not uploaded and were not newly reviewed.
- `official_post_*`: 96 RAW/POST quality rows, all 72 paired summaries, and all 768 rows from eight boundary-matching thresholds.
- `student_*`, `teacher_benchmark.csv`, `official_multistep_*`: full selected fidelity, POST, timing and multi-pass panels, including failures to preserve teacher quality.
- `bounded_restore_*`, `rbf_feedback_paired.csv`: all D0/RBF/BLEND rows and the RBF/feedback comparison.
- `source_C_conditions.csv`, `source_winding_*`: all six conditions, 36 comparisons and 108 stage topology rows. Direction gains do not create new geometric connections.
- `denoise_migration_*`: the completed 60-case CPU integration regression and original completion receipt; zero model/scoring calls. The independent real-D0 and wheel validations also passed; their separate budgets and limits are in [denoise_validation.md](../denoise_validation.md).
- `denoise_cpu_tests.json`, `denoise_real_D0.json`, `denoise_wheel_validation.json`, `denoise_wheel_arrays.csv`: compact verified publication receipts, retaining all 128 wheel array comparisons. One actual D0 model call belongs only to the separate GPU validation; the evidence exporter performs zero scientific calls.
- `user_panel_requests.csv`: all 12 shared requests with parent/condition IDs, actual noise seeds and original C/Gaussian/checkpoint hashes.

CSV values are original strings, including NA/status fields; **fractions are not percentages** unless a field explicitly ends in `_pp`. CF proper and CF illegal are distinct: proper counts only transverse penetration; illegal unions those affected free faces with positive-area coplanar overlap. Both columns remain explicit where available. The bounded tables alias CF to proper and CF_illegal to the union; the source-winding inherited quality summary uses the explicit illegal column. Equal values on this panel do not equate definitions. See [CF_metric_contracts.csv](CF_metric_contracts.csv). FF illegal, ConflictCov_C, boundary matching and graph reach have different domains/denominators. POST free denominators can change. Request IDs plus method/label/stage/phase identify rows; the same request across experiments is not a new independent sample.

[source_manifest.json](source_manifest.json) records source logical locations, full source SHA256, exported hashes, retained/omitted columns and row counts. `repository/` means the repository root; `artifact/` means the separate local experiment archive. `legacy/` means the original standalone scorer source archive; its actual files are checked against the frozen scorer registration. These are logical provenance keys, not public download URLs. Machine paths are removed; source reports and raw assets remain local. The untracked local metric-policy file is recorded only as LOCAL_POLICY_HASH_ONLY_NOT_PUBLISHED; public definitions are self-contained in research_results.md and CF_metric_contracts.csv, with no link requiring that local file. Omitted columns are recorded explicitly, chiefly redundant paths, identities or scalar diagnostic fields. The original full tables remain protected.

[build_evidence.py](build_evidence.py) rebuilds these tables using the original local archives. It reads only and writes only this directory. It does not import a model or run a scorer:

```bash
python r2/docs/evidence/build_evidence.py --artifact-root /path/to/local/experiment-archive
```

The recipe summary and CF contract audit are newly computed: arithmetic aggregation and field/equality checks of saved values, marked DERIVED. Neither performs scoring. The report rounds values for readability; CSVs preserve precision. Definition/source/report hashes identify the original studies even when their complete local archives are not published. Weights and models are not included.
