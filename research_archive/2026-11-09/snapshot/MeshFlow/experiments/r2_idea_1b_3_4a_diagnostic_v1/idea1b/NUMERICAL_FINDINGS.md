# Additional numerical findings (no rescoring)

All 24 request consistency checks PASS. Total objective calls 4635, status counts {'0': 6, 'NO_FEASIBLE_DOF': 6, '1': 12}, feasibility rejections {'AFFINE_DISPLACEMENT_CAP': 638}.
Objective improved 16/24; 6/24 are fully constrained to identity by existing C/F shared anchors.
All eight 608/610 optimizations reach the fixed 300-call budget. They all improve the distance objective but none yields a legal C/F edge; budget exhaustion does not prove optimality.
608 n3 creates one in-budget cluster candidate (1.942282%L), rejected by ANGLE_NEW_OR_WORSE on its affected neighborhood. Other seven have zero in-budget unoccupied cluster candidates.

Relative to the original B2 output, the sole new screening joint request is DEV 2d n1. Its one new interface is not a zero-cost success: one new illegal CF pair appears during affine motion (offset by another disappearance in CF affected-face fraction), and B2 adds one illegal FF pair. FF affected-free fraction rises 1.612903 pp vs original POST; I drops .097656 pp. New POST semantic review remains pending.

Global affine alone preserves all 24 original illegal FF source-pair sets exactly under the actual scorer; no tolerance-driven FF classification change was observed. B2 has separate incremental ledgers and remains zero-new-CF relative to the affine input, not necessarily relative to original POST.

Displacement maxima across requests: {'affine_POST_max_pctL': 3.499999937736488, 'affine_pre_geo_max_pctL': 3.4999999793245147, 'B2_bound_max_from_transformed_pre_pctL': 1.7589597214407142, 'B2_bound_total_from_original_pre_pctL': 3.287728660040315, 'final_retained_max_from_original_pre_pctL': 3.7886387712016667}. Bound-corner total <=5.5%L follows the two separate caps; final retained coordinates relative to pre_geo also contain prior GEO consolidation and must not be mislabeled as B2-only displacement.

Fixed selection gallery has 16 pages, each original POST/AFFINE/AFFINE+B2 with common camera and scale. No automatic new adoption/body labels were produced.
