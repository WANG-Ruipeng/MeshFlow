# Sampling timing presentation correction

The frozen evaluator saved each actual rollout duration in `generated_rows.json -> rows[].metadata.seconds`.
The frozen reporting adapter searched `wall_seconds` / `elapsed_seconds`, so its automatic sampling timing count was zero despite saved timing measurements.

This standalone, post-controller CPU script reads the existing actual `metadata.seconds` first, preserves explicit legacy time fields only as a fallback, and leaves missing times as NA. Cached START rows never enter model timing statistics. Counts and summary mean/median use only actual SAMPLED rows with finite nonnegative measured durations, with the measured count and missing count disclosed.

Original report, both cost tables (CSV/JSON), and the original paired artifact manifest are retained byte-for-byte under `originals/`. Corrected tables remain independently available here and replace the corresponding presentation tables under `paired_results/`. The unchanged frozen `report.render` regenerates the main report using those corrected tables. This does not modify raw arrays, generated-record metadata, scoring, grouping, primary endpoints, quality decisions, training math, or frozen source.

`sampling_timing_changes.csv` records every row's original and corrected time and source field. `report.diff` is the actual main-report display change; `cost_summary.diff` is the summary-table display change; `code.diff` records only this newly added independent script, not a patch to frozen experiment code. `review_receipt.json` contains before/after hashes and preservation checks. Global delivery hashes should be refreshed after this script and figure rendering complete.

Sampling durations retain their original `sample_case` measurement scope; they are not an isolated forward benchmark, a per-step NFE estimate, or historical cache latency. No model or scoring calls are made by this correction.
