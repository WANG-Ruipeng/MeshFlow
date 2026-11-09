# Contextual alignment S/I/E evaluation

This entry point implements only the fixed CHAIR_CONTEXTUAL_ALIGNMENT_SIE_PILOT_V1 evaluation. It does not change the prior readout study, default sampling, or any scientific scoring function.

Run from the installed research repository (or use the absolute tool path):

```bash
python -B tools/contextual_alignment_evaluation.py --stage prepare --out /external/experiments/chair_contextual_alignment_sie_v1
python -B tools/contextual_alignment_evaluation.py --stage score --out /external/experiments/chair_contextual_alignment_sie_v1 --workers 2
python -B tools/contextual_alignment_evaluation.py --stage aggregate --out /external/experiments/chair_contextual_alignment_sie_v1
python -B tools/contextual_alignment_evaluation.py --stage render --out /external/experiments/chair_contextual_alignment_sie_v1
```

The study has 48 immutable requests and four labels (START, A_FM_1000, B_FREE_ALIGN_1000, C_KNOWN_ALIGN_1000), hence 192 output cells. P1 has original eight DEV parents, OLD candidates 0/1 and this study's NEW candidates 6/7. The integer candidate labels distinguish them from prior n4/n5; the actual new seed derives from the exact UTF-8 string `CHAIR_CONTEXTUAL_ALIGNMENT_SIE_PILOT_V1/eval/{uid}/{0,1}`, first four SHA256 bytes as little-endian uint32, then the original MT19937 Gaussian algorithm. Prior registered DEV/OLD/FRESH/quick banks including n0..n5 are checked by actual C and Gaussian arrays. CONFIRM is not read.

P2 reuses the original seat2, seat32, back2 and seat_back4 inputs, all four original noises and N192. It has no source GT, required Gamma or reference visibility: source S/E/distances remain NA. The two seat tessellations are not independent shape evidence.

All conditions, input banks and scoring references are frozen under `data/evaluation/`. Input NPZ includes only C, z, N/K and masks. Evaluator-only GT/Gamma/visibility/query arrays never enter model arguments. Model completion manifests use `outputs/manifest.json` with rows containing the full request plus label, raw_path/raw_sha256 and result_path. Paths may refer to the exact 32 verified START caches; new output arrays must be saved before scoring. Cached historical run receipts omit top-level status, so trust comes from saved score PASS, source identity and actual raw/51-state/C/z/export contracts, with explicit failed statuses still refused.

The exact current scorer is reused without mutating its globals. P2's existing I reference is loaded by a function-local wrapper to prevent rewriting frozen references. All 19 scorer source files and hashes are retained. Cached scalars are not re-scored; 160 new outputs require scientific scoring. New Q_F2048 follows the original candidate-specific random rule remapped onto each new surface. No GT/C query is redrawn, and no old predicted Q_F coordinate set is assigned to a new surface.

Incremental scoring may use `--label`. Missing/failed output cells stay in the fixed 192-row table, and missing noises/parents never reduce a denominator. The full controller must inspect `analysis/results.json`, not infer scientific completeness from a successful subprocess exit.

Principal outputs:
- `per_output_metrics.csv`: all 192 cells, contracts, statuses, original metrics.
- `paired_noise_metrics.csv`, `paired_parent_metrics.csv`: six registered comparisons with original intervals.
- `sie_summary.csv`: OLD/NEW/ALL P1, separate OPEN P2.
- `analysis/comparison_summary.csv`, `leave_one_parent_out.csv`, `control_tails.csv`.
- `analysis/sie_screen.json`: frozen development screen versus A, no automatic promotion.
- `analysis/diversity_pairs.csv` and `diversity_per_condition.csv`: all six candidate pairs per condition and label (288 total), old symmetric fixed-point-to-finite-surface definition. Final aggregate computes missing pairs; cached pair identities are checked. Variation is not larger-is-better.
- `OPEN_GALLERY.html`: all 48 anonymous source-hidden pages followed by 32 P1 reference pages, representing all 192 outputs.

A parent mean is computed over its 2 OLD, 2 NEW or 4 ALL noises before equal weighting over 8 P1 parents. NEW means new noise on observed DEV parents, not an unseen-parent test. Detection intervals are `[candidate_lower-reference_upper, candidate_upper-reference_lower]`; E uses the original global bounds including uncertain reference eligibility. PARTIAL alone is neither automatic FAIL nor PASS.

The registered screen requires ALL S gain >=.01, parent median >0, at least5/8 positive parent S changes, NEW S gain >0, ALL I change <=.01 and E change >=-.01. Intervals crossing a threshold give INCOMPLETE; wholly adverse intervals can FAIL. No source-MSE veto is introduced. Parent I increase/E decrease over.10 is separately disclosed, including possible versus confirmed bound crossings, and per-output point and bound extrema are preserved. Leave-one-parent-out is descriptive and never replaces the formal denominator.

Anonymous method order is deterministically fixed from the study namespace before outcomes; its separate mapping is `gallery/method_mapping.json`. Each condition shares one camera across all four labels and all four noises, including the full extent of abnormal geometry. Four views plus wire display original C blue/free orange without postprocessing. Images contain only anonymous method labels. Rendering COMPLETE is not AI visual review: actual source-hidden inspection must precede source-reference inspection; human usability remains NOT_RUN.
