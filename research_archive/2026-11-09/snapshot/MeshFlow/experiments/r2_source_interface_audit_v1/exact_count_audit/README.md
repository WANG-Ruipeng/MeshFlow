# Exact count audit: NEW32 POST EXP minus SMALL

This is a DERIVED arithmetic audit of existing matching counts, not geometry re-scoring. Original MINI and old experiment files were read-only. No model, noise, render, or visual-review calls occurred.

For A-VMR use Fraction(matched_count, boundary_vertex_count). For PMR use the recorded, count-verified perfect indicator. Subtract EXP minus SMALL per request; average two noises within C, then average two C within each parent. Parent signs use exact rational numerators. No epsilon is selected.

The original float path is reproduced from original VMR/perfect CSV strings using request subtraction then each pairwise mean. Both stages have two values, matching the original float64 mean arithmetic. Absolute parent means and their difference are retained separately; the latter must not be substituted for the original paired-delta arithmetic.

| Metric | Setting | Exact improved / tied / worse | Original strict float improved / tied / worse |
|---|---|---:|---:|
| A_VMR | raw_tau_1e-6 | 7/17/8 | 7/17/8 |
| PMR | raw_tau_1e-6 | 0/32/0 | 0/32/0 |
| A_VMR | L_tau_1e-06 | 7/17/8 | 7/17/8 |
| PMR | L_tau_1e-06 | 0/32/0 | 0/32/0 |
| A_VMR | L_tau_0.0001 | 7/17/8 | 7/17/8 |
| PMR | L_tau_0.0001 | 0/32/0 | 0/32/0 |
| A_VMR | L_tau_0.001 | 7/17/8 | 7/17/8 |
| PMR | L_tau_0.001 | 0/32/0 | 0/32/0 |
| A_VMR | L_tau_0.002 | 7/17/8 | 7/17/8 |
| PMR | L_tau_0.002 | 0/32/0 | 0/32/0 |
| A_VMR | L_tau_0.005 | 5/15/12 | 5/15/12 |
| PMR | L_tau_0.005 | 0/32/0 | 0/32/0 |
| A_VMR | L_tau_0.01 | 10/11/11 | 11/10/11 |
| PMR | L_tau_0.01 | 1/29/2 | 1/29/2 |
| A_VMR | L_tau_0.02 | 16/8/8 | 17/7/8 |
| PMR | L_tau_0.02 | 2/27/3 | 2/27/3 |

Exactly two parent signs differ. Each is an exact tie with a positive floating cancellation residual:

- c7c7079b7555bf075e994da5ba753c34 / L_tau_0.02: SMALL = EXP = 13/50; exact delta 0; original float delta 1.9081958235744878e-17 (0x1.6000000000000p-56).
- ef463d298bb401b3ce1493daf6a835b0 / L_tau_0.01: SMALL = EXP = 19/52; exact delta 0; original float delta 2.7755575615628914e-17 (0x1.0000000000000p-55).

The attachment claims 10/11/11 at 1%L and 16/8/8 at 2%L are both confirmed. Other A-VMR settings and all eight PMR settings have unchanged parent sign counts. This refines numerical improvement counts, not the matching geometry or substantive mean effect. Matching is not a certification of topological attachment.

Files: per_request.csv (2048 metric pairs), per_condition.csv (1024), per_parent.csv (512), summary.csv (16), float_sign_disagreements.csv (2), summary.json, source_hashes.json, and the standalone stdlib audit_exact_counts.py. Original statuses and numeric strings remain explicit. All selected source statuses are PASS; the script preserves NA for non-PASS or empty boundaries.

All ratios and differences are in fraction units 0..1, not percent. Per-row tau, fixed scale, threshold_model, and boundary denominator are retained. The raw model-unit 1e-6 and fixed-L fraction 1e-6 are different settings.
