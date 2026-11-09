# Independent final-report factual review

Status: PASS. No substantive numerical mismatch or broken numerical table selector found. The two interpretation clarifications and final-accounting refresh requested during review have been implemented and independently read back.

Scope: saved CSV/JSON and report/diagnostic source only. Zero model forwards, checkpoint loads, geometry queries, scoring calls or training. No frozen source/results were edited. P2 anonymous observations were frozen before numerical review. Real method mapping and P1 references were not read.

## Verified numerical and interpretive claims

- B-A S: ALL +1.1004588601 pp, median +0.2190491938 pp, 5/8 positive parents; OLD +2.2270180143 pp; NEW -0.0261002941 pp. The report correctly describes a small old-noise incremental benefit relative to matched A, and does not call B entirely ineffective.
- C-A ALL S +0.1951404562 pp, median -0.3463945757 pp, 3/8 positive; NEW -1.0751065443 pp. C-B S is -0.7616305575/-1.0490062502/-0.9053184038 pp for OLD/NEW/ALL, with ALL counts 2 positive/5 negative/1 zero. C-B ALL I +0.5233764648 pp and E +0.7611481817 pp. The tradeoff supports lack of evidence for known-side superiority, not uniform inferiority on every metric.
- Relative to START, ALL S gains A/B/C are +1.1597043352/+2.2601631953/+1.3548447915 pp; only 3/3/2 parents improve and all parent medians are negative. Large positive 866e/86ec contributions and negative f596 changes are preserved.
- Across all six contrasts, no ALL parent I/E cost exceeds 10 pp, including conservative cost bounds. NEW f596 C-A I is +11.3037109375 pp. At f596 n7, A/B/C I are 30.029296875/43.798828125/52.05078125%; B-A/C-A/C-B are +13.76953125/+22.021484375/+8.251953125 pp. Report rounding is correct.
- Worst E changes B-A/C-A: 866e n0 -7.2265625/-7.080078125 pp. C-B: bdd29 n1 -1.5625 pp. Point and conservative extrema agree here.
- B-A f596 S +6.4339860290 pp accounts for 73.08299136% of summed parent change. Leave-one ranges B-A +0.3385264074 to +1.3760736374 pp; C-A -0.3560898523 to +0.5773362467 pp; C-B -1.3013627289 to -0.6946162597 pp. C-A flips negative excluding 866e or f596. Formal eight-parent denominators remain unchanged.
- P1: S/I 128 PASS each; E 32 PASS plus 96 PARTIAL. P2: I 64 PASS and S/E 64 NA_NOT_APPLICABLE. P1 S/I bounds coincide. E bounds can coincide despite PARTIAL eligibility evidence; the report retains PARTIAL and global bounds, not confidence intervals.
- Both frozen screens explicitly FAIL. B fails NEW S; C fails the four S criteria. I/E criteria pass conservatively. No shape-MSE veto, significance interpretation or automatic promotion is introduced.
- Conflict components are correctly interpreted as crossing/near-duplicate union with intersection removed. Component changes are not asserted to prove a unique mechanism.
- Every output has zero exact duplicate excess, zero zero-area fraction, zero near-degenerate fraction and zero exact-coordinate C/free shared edges. P1 FF affected-free means START/A/B/C are 91.364/90.702/90.520/90.644%. The report does not call these whole-mesh literature Ri or welded attachment.
- Diversity: 288 unordered pairs/576 directed distance evaluations; C/A P1 ALL relative change about -4.50%, B/A about -0.07%. Saved surface variation is distinguished from literature TMD, with no larger-is-better or mode-collapse inference.
- P2 I: C-A +0.09765625 pp, C-B +0.3723144531 pp with four conditions positive, B-A -0.2746582031 pp; A-START -3.01513671875 pp. Conditions are not represented as four independent source parents.
- All 48 absolute-metric selectors used by report tables resolve uniquely. Explicit UTF-8 reading confirms valid Chinese source; the initial PowerShell default-encoding display artifact was not a source corruption.
- Each actual training log has 1000 rows. Independently reaggregated FM/raw-align/gradient means match training_interpretation_complete.json. Losses, final-window means, teacher constant baseline, six sparse gradient observations and clip counts match. Online training fitting is not presented as generated control success.
- 32,012 generator forwards and 24,022 backward-path calls agree with the independently completed closure audit. The latter includes 12 autograd.grad observations; nested head combined backward is not added twice. Device-level resource scope and nested timing limitations are disclosed.

## Resolved clarifications

1. The report now states: compute p95 per output, average noises within parent, then equally average eight parents. This is not a percentile of pooled distances.
2. The sparse auxiliary norms now explicitly include 0.1/4, concern the first alpha=1 microbatch at each registered step and seven fixed early parameter tensors, and are not whole-network/whole-batch norms.
3. final_accounting.json was refreshed at time_unix 1791440963.3665028; its controller is COMPUTE_COMPLETE and all ten execution-binding hashes remain true. The earlier RUNNING/score snapshot is no longer the current report evidence.
4. The final visual-review receipt/summary remains the report builder's independent prerequisite. This factual review does not replace actual P1 reference-image inspection and does not authorize bypassing those assertions.

## Saved-vector p95 audit

The report-only helper reads saved squared triangle-surface distances, checks shape (2048,), finite/nonnegative entries and original MSE normalization, then computes quantile(sqrt(d2)/L0, 0.95, method=linear) separately by direction. Aggregation is output -> noise mean within parent -> equal parents. The receipt records 128 P1 outputs, 64 P2 NA, 256 reproduced directional MSE scalars, zero new queries/distance calls/model calls. Existing CSV and helper hashes match its receipt. Legacy fixed-512-query p95 remains distinct, and original S/I/E and screens are unchanged.

## Reviewed source hashes

Review time: 2026-10-08T06:33:18.253922+00:00

- audit/build_final_report.py: a6d3bec39ba137b665cfc6d84acd6d5d7ed534d937b2648a5652f271b2f9be11
- analysis/comparison_summary.csv: 16bb953e393b4f372987d6dbc44b6073b360b291dfde1c88076fa292ef2453a4
- paired_parent_metrics.csv: f28ba8470760ec15f7306655a45fbbfeedea213a137e52e818404f4661f438ee
- paired_noise_metrics.csv: 9742657538441dd3045c7aa976d90e9e0ca3ebff76c37446511246b443b6725a
- per_output_metrics.csv: c6f3c3aaada41ccb9a9b9b0bbc4f0afc0dc192a80556c58b75eb000b90367bb9
- audit/report_ready/P1_ALL_absolute.csv: b787701c8c1e04a765b4741ade733cef74152feeca9151d6cdd1da3fb505a26f
- audit/report_ready/screens_unchanged.json: 088328feebdf25045a87c8f00799431b8ccec21cf9430f893aefa7da376f865c
- audit/report_ready/control_cost_tails.csv: acbaf80614496b7ece87d09e749a6d05ea7be06d11a4fccfd26321727ad540c2
- audit/report_ready/single_output_extrema.csv: 6815c381469c2ad4281d9491001fb588e2c3ddd492c4d0983797a21af339b78b
- analysis/leave_one_parent_out.csv: 7b01742ab0dbdf138c4795c0205e2af50b9c59b7d519a950e0ffc320304b1a28
- audit/report_ready/diversity_summary.csv: 53fc7cafdf0a88126180de848d36eeec13c132d704cefaebe910603937c9c9c0
- analysis/distance_tails/receipt.json: 99837d957cd3fb649b1a515d88a0cf6071b1cd904edb8b31f245c54235158636
- analysis/distance_tails/summary_distance_p95.csv: be815cd008ab5af9bbd43ebee61cd6ca8b329316941e475ea7bd98736a698160
- audit/summarize_saved_distance_tails.py: 4b9f0e2c043fe28dfa3096c4e1221fe8fbe8e25c5202ffe49cc0daf642d5fc62
- audit/training_interpretation_complete.json: 95f461a27fc701a7e5af007c567377c735a49226ba7d587a679ba2ba42828cb4
- audit/final_accounting.json: 0dc7028f40560b90c944adab11fa2a6049d113838f745ea5d7645745414a5a13
