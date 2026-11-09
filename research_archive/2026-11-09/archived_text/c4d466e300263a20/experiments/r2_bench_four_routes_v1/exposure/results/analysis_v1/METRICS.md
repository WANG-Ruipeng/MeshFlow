# Bench M additional exposure: complete numerical results

M is the frozen starting point. EXPOSURE continues its live generator, AdamW and RNG for3000 additional updates. Primary contrast EXPOSURE-M; additional compute and OLD/FOCUS exposure are disclosed.
所有分数沿用原定义，父等权；I主展示只用PASS支持域，完整177字段保留原值与NA。
Fractions use0-1, not percent. OLD/FOCUS/DEV/HOLDOUT/HIGH_TRAIN/HIGH_HOLDOUT/NEW_WIDE_TRAIN remain separate; no automatic promotion.
本文件不产生视觉标签。实际图审及支撑完整性必须另见 visual_review。

## OLD / RAW

| metric | M start | EXPOSURE +3000 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.104158 (8/8父) | 0.101289 (8/8父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.875126 (8/8父) | 0.866803 (8/8父) | fraction_0_to_1 |
| ConflictCov_C | 0.0888672 (8/8父) | 0.0454102 (8/8父) | fraction_0_to_1 |
| CF_illegal_pair_count | 34.9375 (8/8父) | 31.3125 (8/8父) | count |
| FF_illegal_pair_count | 233.938 (8/8父) | 225.125 (8/8父) | count |
| free_minimum_angle_lt5_fraction | 0.376524 (8/8父) | 0.38059 (8/8父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.303675 (8/8父) | 0.306224 (8/8父) | fraction_0_to_1 |
| deleted_free_faces | 0 (8/8父) | 0 (8/8父) | count |
| actual_free_faces | 196.625 (8/8父) | 196.625 (8/8父) | count |
| free_total_surface_area | 33.9641 (8/8父) | 32.1009 (8/8父) | model_length_squared |
| new_bridge_faces | 0 (8/8父) | 0 (8/8父) | count |
| bridge_area | 0 (8/8父) | 0 (8/8父) | model_length_squared |
| deleted_original_POST_faces | 0 (8/8父) | 0 (8/8父) | count |
| hinge_diag_isolated_1face_edge_components | 196.625 (8/8父) | 196.625 (8/8父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 196.625 (8/8父) | 196.625 (8/8父) | count |
| C_F_valid_edge_count | 0 (8/8父) | 0 (8/8父) | count |
| C_F_valid_edge_total_length | 0 (8/8父) | 0 (8/8父) | model_length |
| C_F_has_valid_edge | 0 (8/8父) | 0 (8/8父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (8/8父) | 0 (8/8父) | count |
| nonmanifold_edges | 0 (8/8父) | 0 (8/8父) | count |
| C_F_nonmanifold_edge_count | 0 (8/8父) | 0 (8/8父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (8/8父) | 0 (8/8父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (8/8父) | 1 (8/8父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (8/8父) | 0 (8/8父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/8父) | NA (0/8父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/8父) | NA (0/8父) | model_length_squared |

## OLD / BASE_POST

| metric | M start | EXPOSURE +3000 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.067376 (8/8父) | 0.0791395 (8/8父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.174657 (8/8父) | 0.114885 (8/8父) | fraction_0_to_1 |
| ConflictCov_C | 0.0570679 (8/8父) | 0.0151978 (8/8父) | fraction_0_to_1 |
| CF_illegal_pair_count | 6.5625 (8/8父) | 7.9375 (8/8父) | count |
| FF_illegal_pair_count | 13.6875 (8/8父) | 7.9375 (8/8父) | count |
| free_minimum_angle_lt5_fraction | 0.354922 (8/8父) | 0.370751 (8/8父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.300275 (8/8父) | 0.297 (8/8父) | fraction_0_to_1 |
| deleted_free_faces | 115.375 (8/8父) | 111.75 (8/8父) | count |
| actual_free_faces | 81.25 (8/8父) | 84.875 (8/8父) | count |
| free_total_surface_area | 26.8353 (8/8父) | 24.8969 (8/8父) | model_length_squared |
| new_bridge_faces | 0 (8/8父) | 0 (8/8父) | count |
| bridge_area | 0 (8/8父) | 0 (8/8父) | model_length_squared |
| deleted_original_POST_faces | 0 (8/8父) | 0 (8/8父) | count |
| hinge_diag_isolated_1face_edge_components | 39.8125 (8/8父) | 39.75 (8/8父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 39.375 (8/8父) | 39.25 (8/8父) | count |
| C_F_valid_edge_count | 1.125 (8/8父) | 1.375 (8/8父) | count |
| C_F_valid_edge_total_length | 1.63123 (8/8父) | 2.21507 (8/8父) | model_length |
| C_F_has_valid_edge | 0.4375 (8/8父) | 0.5625 (8/8父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 1.125 (8/8父) | 1.375 (8/8父) | count |
| nonmanifold_edges | 0.1875 (8/8父) | 0.3125 (8/8父) | count |
| C_F_nonmanifold_edge_count | 0 (8/8父) | 0.0625 (8/8父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.0578198 (8/8父) | 0.130489 (8/8父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.601705 (8/8父) | 0.471477 (8/8父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.340475 (8/8父) | 0.398034 (8/8父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/8父) | NA (0/8父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/8父) | NA (0/8父) | model_length_squared |

## OLD / CLEAN_FINAL_POST

| metric | M start | EXPOSURE +3000 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0592767 (8/8父) | 0.0748898 (8/8父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.128789 (8/8父) | 0.0735641 (8/8父) | fraction_0_to_1 |
| ConflictCov_C | 0.0566406 (8/8父) | 0.0150452 (8/8父) | fraction_0_to_1 |
| CF_illegal_pair_count | 5.75 (8/8父) | 7.0625 (8/8父) | count |
| FF_illegal_pair_count | 9.8125 (8/8父) | 5.25 (8/8父) | count |
| free_minimum_angle_lt5_fraction | 0.345963 (8/8父) | 0.350165 (8/8父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.291306 (8/8父) | 0.288868 (8/8父) | fraction_0_to_1 |
| deleted_free_faces | 115.375 (8/8父) | 111.75 (8/8父) | count |
| actual_free_faces | 81.25 (8/8父) | 84.875 (8/8父) | count |
| free_total_surface_area | 27.0716 (8/8父) | 25.1277 (8/8父) | model_length_squared |
| new_bridge_faces | 0 (8/8父) | 0 (8/8父) | count |
| bridge_area | 0 (8/8父) | 0 (8/8父) | model_length_squared |
| deleted_original_POST_faces | 0 (8/8父) | 0 (8/8父) | count |
| hinge_diag_isolated_1face_edge_components | 24.1875 (8/8父) | 22.75 (8/8父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 23.8125 (8/8父) | 22.5 (8/8父) | count |
| C_F_valid_edge_count | 1.3125 (8/8父) | 1.5625 (8/8父) | count |
| C_F_valid_edge_total_length | 1.74952 (8/8父) | 2.44857 (8/8父) | model_length |
| C_F_has_valid_edge | 0.4375 (8/8父) | 0.5625 (8/8父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 1.3125 (8/8父) | 1.5625 (8/8父) | count |
| nonmanifold_edges | 0.1875 (8/8父) | 0.3125 (8/8父) | count |
| C_F_nonmanifold_edge_count | 0 (8/8父) | 0.0625 (8/8父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.10282 (8/8父) | 0.166118 (8/8父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.602337 (8/8父) | 0.471708 (8/8父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.294843 (8/8父) | 0.362174 (8/8父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/8父) | NA (0/8父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/8父) | NA (0/8父) | model_length_squared |

## FOCUS / RAW

| metric | M start | EXPOSURE +3000 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.123687 (2/2父) | 0.180324 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.876578 (2/2父) | 0.874882 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | 0.0839844 (2/2父) | 0.150024 (2/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 45.25 (2/2父) | 61.25 (2/2父) | count |
| FF_illegal_pair_count | 195.5 (2/2父) | 199.25 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.49444 (2/2父) | 0.507758 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.389189 (2/2父) | 0.394749 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 0 (2/2父) | 0 (2/2父) | count |
| actual_free_faces | 188 (2/2父) | 188 (2/2父) | count |
| free_total_surface_area | 15.8932 (2/2父) | 16.5423 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 188 (2/2父) | 188 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 188 (2/2父) | 188 (2/2父) | count |
| C_F_valid_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| C_F_valid_edge_total_length | 0 (2/2父) | 0 (2/2父) | model_length |
| C_F_has_valid_edge | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| nonmanifold_edges | 0 (2/2父) | 0 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (2/2父) | 1 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | NA (0/2父) | model_length_squared |

## FOCUS / BASE_POST

| metric | M start | EXPOSURE +3000 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.00757576 (2/2父) | 0.0108696 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.0627706 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | 0.000244141 (2/2父) | 0.000244141 (2/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 0.25 (2/2父) | 0.25 (2/2父) | count |
| FF_illegal_pair_count | 1 (2/2父) | 0 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.749459 (2/2父) | 0.694344 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.719426 (2/2父) | 0.662928 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 154 (2/2父) | 156 (2/2父) | count |
| actual_free_faces | 34 (2/2父) | 32 (2/2父) | count |
| free_total_surface_area | 3.59751 (2/2父) | 3.62307 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 20.75 (2/2父) | 21 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 20.75 (2/2父) | 21 (2/2父) | count |
| C_F_valid_edge_count | 0.75 (2/2父) | 0.5 (2/2父) | count |
| C_F_valid_edge_total_length | 2.34906 (2/2父) | 2.32975 (2/2父) | model_length |
| C_F_has_valid_edge | 0.5 (2/2父) | 0.5 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0.75 (2/2父) | 0.5 (2/2父) | count |
| nonmanifold_edges | 0 (2/2父) | 0 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.372382 (2/2父) | 0.380031 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.579498 (2/2父) | 0.440706 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.0481202 (2/2父) | 0.179263 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | NA (0/2父) | model_length_squared |

## FOCUS / CLEAN_FINAL_POST

| metric | M start | EXPOSURE +3000 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.00757576 (2/2父) | 0.0108696 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.0508658 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | 0.000244141 (2/2父) | 0.000244141 (2/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 0.25 (2/2父) | 0.25 (2/2父) | count |
| FF_illegal_pair_count | 0.75 (2/2父) | 0 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.743506 (2/2父) | 0.694344 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.705898 (2/2父) | 0.662928 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 154 (2/2父) | 156 (2/2父) | count |
| actual_free_faces | 34 (2/2父) | 32 (2/2父) | count |
| free_total_surface_area | 3.67545 (2/2父) | 3.73535 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 16.25 (2/2父) | 14.25 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 16.25 (2/2父) | 14.25 (2/2父) | count |
| C_F_valid_edge_count | 0.75 (2/2父) | 0.5 (2/2父) | count |
| C_F_valid_edge_total_length | 2.34906 (2/2父) | 2.32975 (2/2父) | model_length |
| C_F_has_valid_edge | 0.5 (2/2父) | 0.5 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0.75 (2/2父) | 0.5 (2/2父) | count |
| nonmanifold_edges | 0 (2/2父) | 0 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.370956 (2/2父) | 0.379392 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.582187 (2/2父) | 0.43635 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.0468566 (2/2父) | 0.184258 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | NA (0/2父) | model_length_squared |

## DEV / RAW

| metric | M start | EXPOSURE +3000 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0594395 (4/4父) | 0.0719943 (4/4父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.890257 (4/4父) | 0.895526 (4/4父) | fraction_0_to_1 |
| ConflictCov_C | 0.128174 (4/4父) | 0.137024 (4/4父) | fraction_0_to_1 |
| CF_illegal_pair_count | 13 (4/4父) | 16.25 (4/4父) | count |
| FF_illegal_pair_count | 262.25 (4/4父) | 288.25 (4/4父) | count |
| free_minimum_angle_lt5_fraction | 0.459958 (4/4父) | 0.397532 (4/4父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.334156 (4/4父) | 0.25696 (4/4父) | fraction_0_to_1 |
| deleted_free_faces | 0 (4/4父) | 0 (4/4父) | count |
| actual_free_faces | 174 (4/4父) | 174 (4/4父) | count |
| free_total_surface_area | 36.5787 (4/4父) | 36.9293 (4/4父) | model_length_squared |
| new_bridge_faces | 0 (4/4父) | 0 (4/4父) | count |
| bridge_area | 0 (4/4父) | 0 (4/4父) | model_length_squared |
| deleted_original_POST_faces | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_isolated_1face_edge_components | 174 (4/4父) | 174 (4/4父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 174 (4/4父) | 174 (4/4父) | count |
| C_F_valid_edge_count | 0 (4/4父) | 0 (4/4父) | count |
| C_F_valid_edge_total_length | 0 (4/4父) | 0 (4/4父) | model_length |
| C_F_has_valid_edge | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (4/4父) | 0 (4/4父) | count |
| nonmanifold_edges | 0 (4/4父) | 0 (4/4父) | count |
| C_F_nonmanifold_edge_count | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (4/4父) | 1 (4/4父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/4父) | NA (0/4父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/4父) | NA (0/4父) | model_length_squared |

## DEV / BASE_POST

| metric | M start | EXPOSURE +3000 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0569387 (4/4父) | 0.081493 (4/4父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.254107 (4/4父) | 0.20938 (4/4父) | fraction_0_to_1 |
| ConflictCov_C | 0.112854 (4/4父) | 0.128357 (4/4父) | fraction_0_to_1 |
| CF_illegal_pair_count | 9.25 (4/4父) | 11 (4/4父) | count |
| FF_illegal_pair_count | 19 (4/4父) | 21.625 (4/4父) | count |
| free_minimum_angle_lt5_fraction | 0.640632 (4/4父) | 0.550733 (4/4父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.542315 (4/4父) | 0.385895 (4/4父) | fraction_0_to_1 |
| deleted_free_faces | 104.375 (4/4父) | 90.125 (4/4父) | count |
| actual_free_faces | 69.625 (4/4父) | 83.875 (4/4父) | count |
| free_total_surface_area | 22.9396 (4/4父) | 25.9585 (4/4父) | model_length_squared |
| new_bridge_faces | 0 (4/4父) | 0 (4/4父) | count |
| bridge_area | 0 (4/4父) | 0 (4/4父) | model_length_squared |
| deleted_original_POST_faces | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_isolated_1face_edge_components | 23.875 (4/4父) | 32.625 (4/4父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 20.625 (4/4父) | 28.625 (4/4父) | count |
| C_F_valid_edge_count | 0 (4/4父) | 0 (4/4父) | count |
| C_F_valid_edge_total_length | 0 (4/4父) | 0 (4/4父) | model_length |
| C_F_has_valid_edge | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (4/4父) | 0 (4/4父) | count |
| nonmanifold_edges | 2.25 (4/4父) | 2.625 (4/4父) | count |
| C_F_nonmanifold_edge_count | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (4/4父) | 0.935355 (4/4父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (4/4父) | 0.0646448 (4/4父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/4父) | NA (0/4父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/4父) | NA (0/4父) | model_length_squared |

## DEV / CLEAN_FINAL_POST

| metric | M start | EXPOSURE +3000 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0569387 (4/4父) | 0.0801343 (4/4父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.205879 (4/4父) | 0.182306 (4/4父) | fraction_0_to_1 |
| ConflictCov_C | 0.112366 (4/4父) | 0.127441 (4/4父) | fraction_0_to_1 |
| CF_illegal_pair_count | 9.25 (4/4父) | 10.875 (4/4父) | count |
| FF_illegal_pair_count | 17.625 (4/4父) | 19.375 (4/4父) | count |
| free_minimum_angle_lt5_fraction | 0.62965 (4/4父) | 0.531529 (4/4父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.526745 (4/4父) | 0.359 (4/4父) | fraction_0_to_1 |
| deleted_free_faces | 104.375 (4/4父) | 90.125 (4/4父) | count |
| actual_free_faces | 69.625 (4/4父) | 83.875 (4/4父) | count |
| free_total_surface_area | 23.1124 (4/4父) | 26.28 (4/4父) | model_length_squared |
| new_bridge_faces | 0 (4/4父) | 0 (4/4父) | count |
| bridge_area | 0 (4/4父) | 0 (4/4父) | model_length_squared |
| deleted_original_POST_faces | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_isolated_1face_edge_components | 16.375 (4/4父) | 20.125 (4/4父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 14.125 (4/4父) | 16.875 (4/4父) | count |
| C_F_valid_edge_count | 0 (4/4父) | 0 (4/4父) | count |
| C_F_valid_edge_total_length | 0 (4/4父) | 0 (4/4父) | model_length |
| C_F_has_valid_edge | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (4/4父) | 0 (4/4父) | count |
| nonmanifold_edges | 2.25 (4/4父) | 2.625 (4/4父) | count |
| C_F_nonmanifold_edge_count | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (4/4父) | 0.935198 (4/4父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (4/4父) | 0.0648023 (4/4父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/4父) | NA (0/4父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/4父) | NA (0/4父) | model_length_squared |

## HOLDOUT / RAW

| metric | M start | EXPOSURE +3000 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.253456 (1/1父) | 0.262673 (1/1父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.898618 (1/1父) | 0.898618 (1/1父) | fraction_0_to_1 |
| ConflictCov_C | 0.097168 (1/1父) | 0.103516 (1/1父) | fraction_0_to_1 |
| CF_illegal_pair_count | 81 (1/1父) | 83.5 (1/1父) | count |
| FF_illegal_pair_count | 264.5 (1/1父) | 265 (1/1父) | count |
| free_minimum_angle_lt5_fraction | 0.172811 (1/1父) | 0.170507 (1/1父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.105991 (1/1父) | 0.0990783 (1/1父) | fraction_0_to_1 |
| deleted_free_faces | 0 (1/1父) | 0 (1/1父) | count |
| actual_free_faces | 217 (1/1父) | 217 (1/1父) | count |
| free_total_surface_area | 21.6825 (1/1父) | 21.6118 (1/1父) | model_length_squared |
| new_bridge_faces | 0 (1/1父) | 0 (1/1父) | count |
| bridge_area | 0 (1/1父) | 0 (1/1父) | model_length_squared |
| deleted_original_POST_faces | 0 (1/1父) | 0 (1/1父) | count |
| hinge_diag_isolated_1face_edge_components | 217 (1/1父) | 217 (1/1父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 217 (1/1父) | 217 (1/1父) | count |
| C_F_valid_edge_count | 0 (1/1父) | 0 (1/1父) | count |
| C_F_valid_edge_total_length | 0 (1/1父) | 0 (1/1父) | model_length |
| C_F_has_valid_edge | 0 (1/1父) | 0 (1/1父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (1/1父) | 0 (1/1父) | count |
| nonmanifold_edges | 0 (1/1父) | 0 (1/1父) | count |
| C_F_nonmanifold_edge_count | 0 (1/1父) | 0 (1/1父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (1/1父) | 0 (1/1父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (1/1父) | 1 (1/1父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (1/1父) | 0 (1/1父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/1父) | NA (0/1父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/1父) | NA (0/1父) | model_length_squared |

## HOLDOUT / BASE_POST

| metric | M start | EXPOSURE +3000 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.199302 (1/1父) | 0.216535 (1/1父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.147365 (1/1父) | 0.165354 (1/1父) | fraction_0_to_1 |
| ConflictCov_C | 0.123291 (1/1父) | 0.127197 (1/1父) | fraction_0_to_1 |
| CF_illegal_pair_count | 33.5 (1/1父) | 35 (1/1父) | count |
| FF_illegal_pair_count | 10.5 (1/1父) | 11 (1/1父) | count |
| free_minimum_angle_lt5_fraction | 0.187143 (1/1父) | 0.188976 (1/1父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.143302 (1/1父) | 0.165354 (1/1父) | fraction_0_to_1 |
| deleted_free_faces | 91.5 (1/1父) | 90 (1/1父) | count |
| actual_free_faces | 125.5 (1/1父) | 127 (1/1父) | count |
| free_total_surface_area | 10.2298 (1/1父) | 10.1921 (1/1父) | model_length_squared |
| new_bridge_faces | 0 (1/1父) | 0 (1/1父) | count |
| bridge_area | 0 (1/1父) | 0 (1/1父) | model_length_squared |
| deleted_original_POST_faces | 0 (1/1父) | 0 (1/1父) | count |
| hinge_diag_isolated_1face_edge_components | 95 (1/1父) | 100 (1/1父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 95 (1/1父) | 100 (1/1父) | count |
| C_F_valid_edge_count | 0 (1/1父) | 0 (1/1父) | count |
| C_F_valid_edge_total_length | 0 (1/1父) | 0 (1/1父) | model_length |
| C_F_has_valid_edge | 0 (1/1父) | 0 (1/1父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (1/1父) | 0 (1/1父) | count |
| nonmanifold_edges | 0 (1/1父) | 0 (1/1父) | count |
| C_F_nonmanifold_edge_count | 0 (1/1父) | 0 (1/1父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (1/1父) | 0 (1/1父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (1/1父) | 1 (1/1父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (1/1父) | 0 (1/1父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/1父) | NA (0/1父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/1父) | NA (0/1父) | model_length_squared |

## HOLDOUT / CLEAN_FINAL_POST

| metric | M start | EXPOSURE +3000 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.199302 (1/1父) | 0.212598 (1/1父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.131365 (1/1父) | 0.133858 (1/1父) | fraction_0_to_1 |
| ConflictCov_C | 0.123047 (1/1父) | 0.127686 (1/1父) | fraction_0_to_1 |
| CF_illegal_pair_count | 32 (1/1父) | 34 (1/1父) | count |
| FF_illegal_pair_count | 9 (1/1父) | 9 (1/1父) | count |
| free_minimum_angle_lt5_fraction | 0.187143 (1/1父) | 0.188976 (1/1父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.127365 (1/1父) | 0.153543 (1/1父) | fraction_0_to_1 |
| deleted_free_faces | 91.5 (1/1父) | 90 (1/1父) | count |
| actual_free_faces | 125.5 (1/1父) | 127 (1/1父) | count |
| free_total_surface_area | 10.2645 (1/1父) | 10.219 (1/1父) | model_length_squared |
| new_bridge_faces | 0 (1/1父) | 0 (1/1父) | count |
| bridge_area | 0 (1/1父) | 0 (1/1父) | model_length_squared |
| deleted_original_POST_faces | 0 (1/1父) | 0 (1/1父) | count |
| hinge_diag_isolated_1face_edge_components | 65 (1/1父) | 77 (1/1父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 65 (1/1父) | 77 (1/1父) | count |
| C_F_valid_edge_count | 0 (1/1父) | 0 (1/1父) | count |
| C_F_valid_edge_total_length | 0 (1/1父) | 0 (1/1父) | model_length |
| C_F_has_valid_edge | 0 (1/1父) | 0 (1/1父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (1/1父) | 0 (1/1父) | count |
| nonmanifold_edges | 0 (1/1父) | 0 (1/1父) | count |
| C_F_nonmanifold_edge_count | 0 (1/1父) | 0 (1/1父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (1/1父) | 0 (1/1父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (1/1父) | 1 (1/1父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (1/1父) | 0 (1/1父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/1父) | NA (0/1父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/1父) | NA (0/1父) | model_length_squared |

## HIGH_TRAIN / RAW

| metric | M start | EXPOSURE +3000 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0686832 (2/2父) | 0.0700201 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.757297 (2/2父) | 0.76025 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 32.25 (2/2父) | 31.75 (2/2父) | count |
| FF_illegal_pair_count | 303.25 (2/2父) | 295.75 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.505793 (2/2父) | 0.542892 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.446134 (2/2父) | 0.487634 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 0 (2/2父) | 0 (2/2父) | count |
| actual_free_faces | 319 (2/2父) | 319 (2/2父) | count |
| free_total_surface_area | 34.6542 (2/2父) | 35.9504 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 319 (2/2父) | 319 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 319 (2/2父) | 319 (2/2父) | count |
| C_F_valid_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| C_F_valid_edge_total_length | 0 (2/2父) | 0 (2/2父) | model_length |
| C_F_has_valid_edge | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| nonmanifold_edges | 0 (2/2父) | 0 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (2/2父) | 1 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | NA (0/2父) | model_length_squared |

## HIGH_TRAIN / BASE_POST

| metric | M start | EXPOSURE +3000 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0131579 (2/2父) | 0.037037 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.0625602 (2/2父) | 0.0509259 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 3.25 (2/2父) | 11.5 (2/2父) | count |
| FF_illegal_pair_count | 2.5 (2/2父) | 1.75 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.365674 (2/2父) | 0.305633 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.329478 (2/2父) | 0.284028 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 269 (2/2父) | 273.25 (2/2父) | count |
| actual_free_faces | 50 (2/2父) | 45.75 (2/2父) | count |
| free_total_surface_area | 15.5698 (2/2父) | 16.4056 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 28.25 (2/2父) | 23.5 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 28 (2/2父) | 22.5 (2/2父) | count |
| C_F_valid_edge_count | 0.75 (2/2父) | 1 (2/2父) | count |
| C_F_valid_edge_total_length | 3.98724 (2/2父) | 5.31632 (2/2父) | model_length |
| C_F_has_valid_edge | 0.5 (2/2父) | 0.5 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0.75 (2/2父) | 1 (2/2父) | count |
| nonmanifold_edges | 0.25 (2/2父) | 0.5 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.0183549 (2/2父) | 0.0276615 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.923782 (2/2父) | 0.826334 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.0578629 (2/2父) | 0.146004 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | NA (0/2父) | model_length_squared |

## HIGH_TRAIN / CLEAN_FINAL_POST

| metric | M start | EXPOSURE +3000 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0131579 (2/2父) | 0.037037 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.0625602 (2/2父) | 0.0509259 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 3.25 (2/2父) | 11.5 (2/2父) | count |
| FF_illegal_pair_count | 2.5 (2/2父) | 1.75 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.361288 (2/2父) | 0.305633 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.31632 (2/2父) | 0.280942 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 269 (2/2父) | 273.25 (2/2父) | count |
| actual_free_faces | 50 (2/2父) | 45.75 (2/2父) | count |
| free_total_surface_area | 15.6766 (2/2父) | 16.4849 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 24.75 (2/2父) | 19.75 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 24.5 (2/2父) | 18.75 (2/2父) | count |
| C_F_valid_edge_count | 0.75 (2/2父) | 1 (2/2父) | count |
| C_F_valid_edge_total_length | 3.98724 (2/2父) | 5.31632 (2/2父) | model_length |
| C_F_has_valid_edge | 0.5 (2/2父) | 0.5 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0.75 (2/2父) | 1 (2/2父) | count |
| nonmanifold_edges | 0.25 (2/2父) | 0.5 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.0187464 (2/2父) | 0.0273089 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.924216 (2/2父) | 0.826838 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.057038 (2/2父) | 0.145853 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | NA (0/2父) | model_length_squared |

## HIGH_HOLDOUT / RAW

| metric | M start | EXPOSURE +3000 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.109901 (2/2父) | 0.107162 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.944309 (2/2父) | 0.952513 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | 0.172852 (2/2父) | 0.137695 (2/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 110.5 (2/2父) | 106 (2/2父) | count |
| FF_illegal_pair_count | 1002.25 (2/2父) | 1034 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.414761 (2/2父) | 0.390386 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.380444 (2/2父) | 0.355802 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 0 (2/2父) | 0 (2/2父) | count |
| actual_free_faces | 522.5 (2/2父) | 522.5 (2/2父) | count |
| free_total_surface_area | 53.8223 (2/2父) | 48.9734 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 522.5 (2/2父) | 522.5 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 522.5 (2/2父) | 522.5 (2/2父) | count |
| C_F_valid_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| C_F_valid_edge_total_length | 0 (2/2父) | 0 (2/2父) | model_length |
| C_F_has_valid_edge | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| nonmanifold_edges | 0 (2/2父) | 0 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (2/2父) | 1 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | NA (0/2父) | model_length_squared |

## HIGH_HOLDOUT / BASE_POST

| metric | M start | EXPOSURE +3000 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.211942 (2/2父) | 0.231247 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.208073 (2/2父) | 0.109895 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | 0.0960693 (2/2父) | 0.0532227 (2/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 32.5 (2/2父) | 35.5 (2/2父) | count |
| FF_illegal_pair_count | 8 (2/2父) | 3 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.455952 (2/2父) | 0.45967 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.350223 (2/2父) | 0.351079 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 471 (2/2父) | 474.75 (2/2父) | count |
| actual_free_faces | 51.5 (2/2父) | 47.75 (2/2父) | count |
| free_total_surface_area | 30.2494 (2/2父) | 30.2899 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 20.5 (2/2父) | 17.5 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 17.25 (2/2父) | 14.75 (2/2父) | count |
| C_F_valid_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| C_F_valid_edge_total_length | 0 (2/2父) | 0 (2/2父) | model_length |
| C_F_has_valid_edge | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| nonmanifold_edges | 2.5 (2/2父) | 2.25 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0.25 (2/2父) | 0 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.759375 (2/2父) | 1 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.240625 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | NA (0/2父) | model_length_squared |

## HIGH_HOLDOUT / CLEAN_FINAL_POST

| metric | M start | EXPOSURE +3000 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.211942 (2/2父) | 0.231247 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.17061 (2/2父) | 0.0973951 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | 0.085083 (2/2父) | 0.0532227 (2/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 32.25 (2/2父) | 35 (2/2父) | count |
| FF_illegal_pair_count | 6.25 (2/2父) | 2.75 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.433631 (2/2父) | 0.45967 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.343973 (2/2父) | 0.338579 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 471 (2/2父) | 474.75 (2/2父) | count |
| actual_free_faces | 51.5 (2/2父) | 47.75 (2/2父) | count |
| free_total_surface_area | 30.5081 (2/2父) | 30.4697 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 13.5 (2/2父) | 11.5 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 10.75 (2/2父) | 9 (2/2父) | count |
| C_F_valid_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| C_F_valid_edge_total_length | 0 (2/2父) | 0 (2/2父) | model_length |
| C_F_has_valid_edge | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| nonmanifold_edges | 2.5 (2/2父) | 2.25 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0.25 (2/2父) | 0 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.759377 (2/2父) | 1 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.240623 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | NA (0/2父) | model_length_squared |

## NEW_WIDE_TRAIN / RAW

| metric | M start | EXPOSURE +3000 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0667595 (4/4父) | 0.043448 (4/4父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.912793 (4/4父) | 0.880365 (4/4父) | fraction_0_to_1 |
| ConflictCov_C | 0.111938 (2/4父) | 0.0457764 (2/4父) | fraction_0_to_1 |
| CF_illegal_pair_count | 80.25 (4/4父) | 51.5 (4/4父) | count |
| FF_illegal_pair_count | 1025.12 (4/4父) | 653 (4/4父) | count |
| free_minimum_angle_lt5_fraction | 0.384182 (4/4父) | 0.471786 (4/4父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.301434 (4/4父) | 0.373731 (4/4父) | fraction_0_to_1 |
| deleted_free_faces | 0 (4/4父) | 0 (4/4父) | count |
| actual_free_faces | 553 (4/4父) | 553 (4/4父) | count |
| free_total_surface_area | 27.4613 (4/4父) | 19.8244 (4/4父) | model_length_squared |
| new_bridge_faces | 0 (4/4父) | 0 (4/4父) | count |
| bridge_area | 0 (4/4父) | 0 (4/4父) | model_length_squared |
| deleted_original_POST_faces | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_isolated_1face_edge_components | 553 (4/4父) | 553 (4/4父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 553 (4/4父) | 553 (4/4父) | count |
| C_F_valid_edge_count | 0 (4/4父) | 0 (4/4父) | count |
| C_F_valid_edge_total_length | 0 (4/4父) | 0 (4/4父) | model_length |
| C_F_has_valid_edge | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (4/4父) | 0 (4/4父) | count |
| nonmanifold_edges | 0 (4/4父) | 0 (4/4父) | count |
| C_F_nonmanifold_edge_count | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (4/4父) | 1 (4/4父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/4父) | NA (0/4父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/4父) | NA (0/4父) | model_length_squared |

## NEW_WIDE_TRAIN / BASE_POST

| metric | M start | EXPOSURE +3000 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0961712 (4/4父) | 0.000730994 (4/4父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.196002 (4/4父) | 0.0926834 (4/4父) | fraction_0_to_1 |
| ConflictCov_C | 0.0458984 (2/4父) | 0.0379639 (2/4父) | fraction_0_to_1 |
| CF_illegal_pair_count | 8.25 (4/4父) | 0.125 (4/4父) | count |
| FF_illegal_pair_count | 24 (4/4父) | 8.5 (4/4父) | count |
| free_minimum_angle_lt5_fraction | 0.58373 (4/4父) | 0.59247 (4/4父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.485376 (4/4父) | 0.461232 (4/4父) | fraction_0_to_1 |
| deleted_free_faces | 458.75 (4/4父) | 455.625 (4/4父) | count |
| actual_free_faces | 94.25 (4/4父) | 97.375 (4/4父) | count |
| free_total_surface_area | 10.5462 (4/4父) | 6.80498 (4/4父) | model_length_squared |
| new_bridge_faces | 0 (4/4父) | 0 (4/4父) | count |
| bridge_area | 0 (4/4父) | 0 (4/4父) | model_length_squared |
| deleted_original_POST_faces | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_isolated_1face_edge_components | 52.125 (4/4父) | 62.5 (4/4父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 49.125 (4/4父) | 60.625 (4/4父) | count |
| C_F_valid_edge_count | 1 (4/4父) | 2.25 (4/4父) | count |
| C_F_valid_edge_total_length | 4.3543 (4/4父) | 7.9647 (4/4父) | model_length |
| C_F_has_valid_edge | 0.375 (4/4父) | 0.75 (4/4父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 1 (4/4父) | 2.25 (4/4父) | count |
| nonmanifold_edges | 1.625 (4/4父) | 0.875 (4/4父) | count |
| C_F_nonmanifold_edge_count | 0 (4/4父) | 0.125 (4/4父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.0524677 (4/4父) | 0.0889314 (4/4父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.806518 (4/4父) | 0.660698 (4/4父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.141014 (4/4父) | 0.250371 (4/4父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/4父) | NA (0/4父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/4父) | NA (0/4父) | model_length_squared |

## NEW_WIDE_TRAIN / CLEAN_FINAL_POST

| metric | M start | EXPOSURE +3000 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0954445 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.181629 (4/4父) | 0.0829673 (4/4父) | fraction_0_to_1 |
| ConflictCov_C | 0.0435791 (2/4父) | 0.0262451 (2/4父) | fraction_0_to_1 |
| CF_illegal_pair_count | 8.125 (4/4父) | 0 (4/4父) | count |
| FF_illegal_pair_count | 22.5 (4/4父) | 7.625 (4/4父) | count |
| free_minimum_angle_lt5_fraction | 0.567118 (4/4父) | 0.56165 (4/4父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.466578 (4/4父) | 0.43201 (4/4父) | fraction_0_to_1 |
| deleted_free_faces | 458.75 (4/4父) | 455.625 (4/4父) | count |
| actual_free_faces | 94.25 (4/4父) | 97.375 (4/4父) | count |
| free_total_surface_area | 10.7877 (4/4父) | 6.96885 (4/4父) | model_length_squared |
| new_bridge_faces | 0 (4/4父) | 0 (4/4父) | count |
| bridge_area | 0 (4/4父) | 0 (4/4父) | model_length_squared |
| deleted_original_POST_faces | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_isolated_1face_edge_components | 38.625 (4/4父) | 47.625 (4/4父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 35.75 (4/4父) | 45.875 (4/4父) | count |
| C_F_valid_edge_count | 1 (4/4父) | 2.25 (4/4父) | count |
| C_F_valid_edge_total_length | 4.3543 (4/4父) | 7.9647 (4/4父) | model_length |
| C_F_has_valid_edge | 0.375 (4/4父) | 0.75 (4/4父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 1 (4/4父) | 2.25 (4/4父) | count |
| nonmanifold_edges | 1.625 (4/4父) | 0.875 (4/4父) | count |
| C_F_nonmanifold_edge_count | 0 (4/4父) | 0.125 (4/4父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.0576929 (4/4父) | 0.100283 (4/4父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.805389 (4/4父) | 0.657198 (4/4父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.136918 (4/4父) | 0.24252 (4/4父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/4父) | NA (0/4父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/4父) | NA (0/4父) | model_length_squared |
