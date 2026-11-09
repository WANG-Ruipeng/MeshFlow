# Final saved-results reporting and actual visual review

The helper is external to the frozen repository. It imports only Python standard-library modules, reads CSV/JSON/PNG bytes, and writes only `audit/report_ready/`. It never imports scoring, model or geometry implementations and never reads `gallery/method_mapping.json`.

## Commands

```text
python -B audit/report_ready.py --stage schema
python -B audit/report_ready.py --stage tables
python -B audit/report_ready.py --stage gallery
python -B audit/report_ready.py --stage all
```

Use the registered WSL Python and absolute script path. `schema` can run before completion and records readiness only. `tables` requires saved `analysis/results.json` COMPLETE, 192 PASS outputs (32 verified START caches + 160 new scores), diversity 288/288, six upstream table hashes, exact fixed parent/noise denominators, and algebraic consistency of saved means. The scorer is never called. `gallery` requires the final 48-page/192-output manifest and verifies all 48 blind + 32 reference image hashes. It creates templates, **not reviews**. `all` performs tables and gallery sequentially.

## Output navigation

- P1_OLD/NEW/ALL absolute and comparison CSVs: existing point values and conservative bounds, plus explicitly named display columns; S/I/E fractions become percent/percentage points only in the display columns.
- P2_OPEN tables: four hand-designed control conditions. Missing GT/interface/exposure-dependent quantities remain NA, not zero or failures.
- `comparison_facts.csv`: C-A, C-B, B-A and A/B/C-START; parent means/median/sign counts/opposing-noise counts, and clearly named ratio-of-group-means relative changes. It does not substitute per-parent relative changes.
- `parent_patterns.csv`, `noise_patterns.csv`: all fixed parents/noises for the selected metrics; no parent is dropped.
- `screen_bounds.csv`/`screens_unchanged.json`: the original numerical screen and its original PASS/FAIL/INCOMPLETE states. No new threshold or promotion rule is introduced.
- `control_cost_tails.csv`: original severe-parent and output-bound tails; `single_output_extrema.csv` also retains descriptive quality/distance extremes without adding thresholds.
- `leave_one_parent_out.csv` and `leave_one_ranges.csv`: descriptive leave-one sensitivity only. Formal full-eight-parent result and gates remain unchanged.
- `diversity_condition_rows.csv`/`diversity_summary.csv`: original saved variation, not a larger-is-better score or original TMD.
- `summary.json`/`TABLES.md`: report-ready facts, denominator and interpretation caveats. `source_hashes.json` binds input bytes and the helper version; no final training/accounting claim is made.

## Gallery source schema and safe review sequence

The original `gallery/gallery_manifest.json` has top-level `status=FINAL_COMPLETE`, `identity`, `pages`, `output_count=192`, `blind_pages=48`, `reference_pages=32`, `AI_VISUAL_REVIEW`, `human_usability`, `mesh_modified`, timing and worker fields. **Do not read its `labels` or identity output keys before recording anonymous observations**, because they reveal the real method ordering. The helper reads them only to omit them; it does not load the mapping.

Each original page contains request_id, parent_id, condition_id, panel, candidate, cohort, blind_page/blind_sha256, anonymous_order, labels, view_scope and review statuses. P1 additionally has reference_page/reference_sha256. The safe `gallery_pages_blinded.json` copies only request metadata, aliases, image paths/hashes and view names, never actual labels or identity keys.

Actual layout: methods are ROWS. Blind pages are 5 rows (Input C, Method 1, Method 2, Method 3, Method 4) by 5 columns (front_3q, back_3q, front, side, wire). P1 reference pages add a sixth Source reference row. The renderer uses the same condition camera across all four methods and four noises; no mesh repair, recentering or per-method crop is applied. Blue=C, orange=free. The reference is one source completion, not the unique valid answer.

1. Open `BLIND_INDEX.html` and inspect every assigned page at readable scale, including all four views and wire. Record observations in a separate CSV copied from `blind_review_template.csv`; do not mark image inspection merely from a file/receipt existing.
2. Freeze/save blind observations before opening `REFERENCE_INDEX.html`. Record reference-specific observations separately using `reference_review_template.csv`, retaining a link to the original blind row. P2 has no reference page.
3. Only after observations are recorded, reveal the independent method mapping and add resolved labels in a derived joined review. Preserve original anonymous observations.

## Actual review CSV schema

One row = one request/noise and one anonymous model. Blind template has 192 rows; reference template 128 rows, all explicitly NOT_REVIEWED_TEMPLATE with empty observations. They are not completion evidence.

Identification: request_id, parent_id, condition_id, panel, cohort, candidate, anonymous_method, stage, page, page_sha256. Provenance: review_kind=AI_VISUAL_REVIEW, reviewer, observed_utc, view_scope (must list actually inspected front_3q/back_3q/front/side/wire), review_status. Observations: C_compatibility, overall_coherence, missing_body, duplicate_overlap, long_thin_sheet, style_vs_fragmentation, certainty, render_readability, notes. Reference-only fields: source_specific_difference, alternative_valid_completion, blind_observation_row_id.

Use specific observations and uncertainty (not_apparent/possible/evident/unable_to_judge). Different style or leg count is not automatically fragmentation or failure. C coordinate preservation is an audited geometric contract; image impressions of compatibility describe nearby completion relationships and must not override the byte checks. Human usability stays NOT_RUN; do not invent a usable-rate percentage. Seeing all images can coexist with uncertain scientific conclusions.
