# Bench wider HIGH pool: complete numerical results

B is the completed narrow HIGH reference (38 cached +8 new outputs). W is the one new1500-step wide HIGH arm from the identical P0. Contrast: W-B.
所有分数沿用原定义，父等权；I主展示只用PASS支持域，完整177字段保留原值与NA。
Fractions use0-1, not percent. OLD/FOCUS/DEV/HOLDOUT/HIGH_TRAIN/HIGH_HOLDOUT/NEW_WIDE_TRAIN remain separate; no automatic promotion.
本文件不产生视觉标签。实际图审及支撑完整性必须另见 visual_review。

## OLD / RAW

| metric | B narrow | W wide | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0922111 (8/8父) | 0.09069 (8/8父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.850119 (8/8父) | 0.85466 (8/8父) | fraction_0_to_1 |
| ConflictCov_C | 0.0376282 (8/8父) | 0.0292358 (8/8父) | fraction_0_to_1 |
| CF_illegal_pair_count | 33.125 (8/8父) | 30.875 (8/8父) | count |
| FF_illegal_pair_count | 205.438 (8/8父) | 215.75 (8/8父) | count |
| free_minimum_angle_lt5_fraction | 0.389759 (8/8父) | 0.379613 (8/8父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.316694 (8/8父) | 0.310358 (8/8父) | fraction_0_to_1 |
| deleted_free_faces | 0 (8/8父) | 0 (8/8父) | count |
| actual_free_faces | 196.625 (8/8父) | 196.625 (8/8父) | count |
| free_total_surface_area | 31.3943 (8/8父) | 34.1868 (8/8父) | model_length_squared |
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

| metric | B narrow | W wide | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0653743 (8/8父) | 0.070915 (8/8父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.120557 (8/8父) | 0.165707 (8/8父) | fraction_0_to_1 |
| ConflictCov_C | 0.0150146 (8/8父) | 0.0160217 (8/8父) | fraction_0_to_1 |
| CF_illegal_pair_count | 6.3125 (8/8父) | 6.75 (8/8父) | count |
| FF_illegal_pair_count | 8.375 (8/8父) | 14.9375 (8/8父) | count |
| free_minimum_angle_lt5_fraction | 0.35971 (8/8父) | 0.351154 (8/8父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.286391 (8/8父) | 0.293952 (8/8父) | fraction_0_to_1 |
| deleted_free_faces | 110.312 (8/8父) | 107.938 (8/8父) | count |
| actual_free_faces | 86.3125 (8/8父) | 88.6875 (8/8父) | count |
| free_total_surface_area | 24.6144 (8/8父) | 27.3013 (8/8父) | model_length_squared |
| new_bridge_faces | 0 (8/8父) | 0 (8/8父) | count |
| bridge_area | 0 (8/8父) | 0 (8/8父) | model_length_squared |
| deleted_original_POST_faces | 0 (8/8父) | 0 (8/8父) | count |
| hinge_diag_isolated_1face_edge_components | 41.0625 (8/8父) | 42.9375 (8/8父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 40.1875 (8/8父) | 42.3125 (8/8父) | count |
| C_F_valid_edge_count | 1 (8/8父) | 1.1875 (8/8父) | count |
| C_F_valid_edge_total_length | 2.06804 (8/8父) | 1.57386 (8/8父) | model_length |
| C_F_has_valid_edge | 0.5625 (8/8父) | 0.5 (8/8父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 1 (8/8父) | 1.1875 (8/8父) | count |
| nonmanifold_edges | 0.4375 (8/8父) | 0.3125 (8/8父) | count |
| C_F_nonmanifold_edge_count | 0.0625 (8/8父) | 0 (8/8父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.0599955 (8/8父) | 0.0512624 (8/8父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.607836 (8/8父) | 0.564312 (8/8父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.332168 (8/8父) | 0.384426 (8/8父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/8父) | NA (0/8父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/8父) | NA (0/8父) | model_length_squared |

## OLD / CLEAN_FINAL_POST

| metric | B narrow | W wide | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.06276 (8/8父) | 0.0672275 (8/8父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.091824 (8/8父) | 0.138896 (8/8父) | fraction_0_to_1 |
| ConflictCov_C | 0.0148621 (8/8父) | 0.0155334 (8/8父) | fraction_0_to_1 |
| CF_illegal_pair_count | 6.0625 (8/8父) | 6.3125 (8/8父) | count |
| FF_illegal_pair_count | 6.3125 (8/8父) | 12.5625 (8/8父) | count |
| free_minimum_angle_lt5_fraction | 0.344514 (8/8父) | 0.337897 (8/8父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.27314 (8/8父) | 0.28579 (8/8父) | fraction_0_to_1 |
| deleted_free_faces | 110.312 (8/8父) | 107.938 (8/8父) | count |
| actual_free_faces | 86.3125 (8/8父) | 88.6875 (8/8父) | count |
| free_total_surface_area | 24.8545 (8/8父) | 27.458 (8/8父) | model_length_squared |
| new_bridge_faces | 0 (8/8父) | 0 (8/8父) | count |
| bridge_area | 0 (8/8父) | 0 (8/8父) | model_length_squared |
| deleted_original_POST_faces | 0 (8/8父) | 0 (8/8父) | count |
| hinge_diag_isolated_1face_edge_components | 25.6875 (8/8父) | 27 (8/8父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 24.875 (8/8父) | 26.5625 (8/8父) | count |
| C_F_valid_edge_count | 1 (8/8父) | 1.375 (8/8父) | count |
| C_F_valid_edge_total_length | 2.06804 (8/8父) | 1.80736 (8/8父) | model_length |
| C_F_has_valid_edge | 0.5625 (8/8父) | 0.5 (8/8父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 1 (8/8父) | 1.375 (8/8父) | count |
| nonmanifold_edges | 0.4375 (8/8父) | 0.3125 (8/8父) | count |
| C_F_nonmanifold_edge_count | 0.0625 (8/8父) | 0 (8/8父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.0902084 (8/8父) | 0.114553 (8/8父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.607965 (8/8父) | 0.565791 (8/8父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.301827 (8/8父) | 0.319656 (8/8父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/8父) | NA (0/8父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/8父) | NA (0/8父) | model_length_squared |

## FOCUS / RAW

| metric | B narrow | W wide | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.122537 (2/2父) | 0.127021 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.857227 (2/2父) | 0.868835 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | 0.0714111 (2/2父) | 0.0823975 (2/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 40.25 (2/2父) | 44.25 (2/2父) | count |
| FF_illegal_pair_count | 190 (2/2父) | 204.75 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.499971 (2/2父) | 0.48556 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.402463 (2/2父) | 0.39528 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 0 (2/2父) | 0 (2/2父) | count |
| actual_free_faces | 188 (2/2父) | 188 (2/2父) | count |
| free_total_surface_area | 15.7702 (2/2父) | 16.2175 (2/2父) | model_length_squared |
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

| metric | B narrow | W wide | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.034897 (2/2父) | 0.0142857 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 0 (2/2父) | 0 (2/2父) | count |
| FF_illegal_pair_count | 0.5 (2/2父) | 0.25 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.659809 (2/2父) | 0.681905 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.624138 (2/2父) | 0.659122 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 156.5 (2/2父) | 157.75 (2/2父) | count |
| actual_free_faces | 31.5 (2/2父) | 30.25 (2/2父) | count |
| free_total_surface_area | 3.45842 (2/2父) | 3.49007 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 19.75 (2/2父) | 19 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 19.25 (2/2父) | 19 (2/2父) | count |
| C_F_valid_edge_count | 1 (2/2父) | 1 (2/2父) | count |
| C_F_valid_edge_total_length | 3.62273 (2/2父) | 2.36837 (2/2父) | model_length |
| C_F_has_valid_edge | 0.75 (2/2父) | 0.5 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 1 (2/2父) | 1 (2/2父) | count |
| nonmanifold_edges | 0.25 (2/2父) | 0 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.396729 (2/2父) | 0.38458 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.602164 (2/2父) | 0.601226 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.00110741 (2/2父) | 0.0141932 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | NA (0/2父) | model_length_squared |

## FOCUS / CLEAN_FINAL_POST

| metric | B narrow | W wide | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.0217391 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 0 (2/2父) | 0 (2/2父) | count |
| FF_illegal_pair_count | 0.25 (2/2父) | 0 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.65323 (2/2父) | 0.674763 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.617559 (2/2父) | 0.636828 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 156.5 (2/2父) | 157.75 (2/2父) | count |
| actual_free_faces | 31.5 (2/2父) | 30.25 (2/2父) | count |
| free_total_surface_area | 3.59336 (2/2父) | 3.53564 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 17.25 (2/2父) | 15.25 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 16.75 (2/2父) | 15.25 (2/2父) | count |
| C_F_valid_edge_count | 1 (2/2父) | 1 (2/2父) | count |
| C_F_valid_edge_total_length | 3.62273 (2/2父) | 2.36837 (2/2父) | model_length |
| C_F_has_valid_edge | 0.75 (2/2父) | 0.5 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 1 (2/2父) | 1 (2/2父) | count |
| nonmanifold_edges | 0.25 (2/2父) | 0 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.395545 (2/2父) | 0.3831 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.603347 (2/2父) | 0.602764 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.00110728 (2/2父) | 0.0141354 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | NA (0/2父) | model_length_squared |

## DEV / RAW

| metric | B narrow | W wide | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0595824 (4/4父) | 0.0589639 (4/4父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.874498 (4/4父) | 0.88908 (4/4父) | fraction_0_to_1 |
| ConflictCov_C | 0.125 (4/4父) | 0.127869 (4/4父) | fraction_0_to_1 |
| CF_illegal_pair_count | 13 (4/4父) | 12.875 (4/4父) | count |
| FF_illegal_pair_count | 233.25 (4/4父) | 266.375 (4/4父) | count |
| free_minimum_angle_lt5_fraction | 0.444582 (4/4父) | 0.459485 (4/4父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.309994 (4/4父) | 0.333389 (4/4父) | fraction_0_to_1 |
| deleted_free_faces | 0 (4/4父) | 0 (4/4父) | count |
| actual_free_faces | 174 (4/4父) | 174 (4/4父) | count |
| free_total_surface_area | 34.869 (4/4父) | 36.3304 (4/4父) | model_length_squared |
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

| metric | B narrow | W wide | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0617901 (4/4父) | 0.0637091 (4/4父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.164067 (4/4父) | 0.240975 (4/4父) | fraction_0_to_1 |
| ConflictCov_C | 0.115479 (4/4父) | 0.11969 (4/4父) | fraction_0_to_1 |
| CF_illegal_pair_count | 9 (4/4父) | 10 (4/4父) | count |
| FF_illegal_pair_count | 12.5 (4/4父) | 21.125 (4/4父) | count |
| free_minimum_angle_lt5_fraction | 0.658842 (4/4父) | 0.651295 (4/4父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.540728 (4/4父) | 0.54588 (4/4父) | fraction_0_to_1 |
| deleted_free_faces | 106.125 (4/4父) | 105.125 (4/4父) | count |
| actual_free_faces | 67.875 (4/4父) | 68.875 (4/4父) | count |
| free_total_surface_area | 20.8231 (4/4父) | 22.6203 (4/4父) | model_length_squared |
| new_bridge_faces | 0 (4/4父) | 0 (4/4父) | count |
| bridge_area | 0 (4/4父) | 0 (4/4父) | model_length_squared |
| deleted_original_POST_faces | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_isolated_1face_edge_components | 25.25 (4/4父) | 26.625 (4/4父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 22.625 (4/4父) | 23.875 (4/4父) | count |
| C_F_valid_edge_count | 0 (4/4父) | 0 (4/4父) | count |
| C_F_valid_edge_total_length | 0 (4/4父) | 0 (4/4父) | model_length |
| C_F_has_valid_edge | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (4/4父) | 0 (4/4父) | count |
| nonmanifold_edges | 2 (4/4父) | 2.25 (4/4父) | count |
| C_F_nonmanifold_edge_count | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (4/4父) | 1 (4/4父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/4父) | NA (0/4父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/4父) | NA (0/4父) | model_length_squared |

## DEV / CLEAN_FINAL_POST

| metric | B narrow | W wide | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0603856 (4/4父) | 0.0637091 (4/4父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.11494 (4/4父) | 0.201394 (4/4父) | fraction_0_to_1 |
| ConflictCov_C | 0.117188 (4/4父) | 0.11676 (4/4父) | fraction_0_to_1 |
| CF_illegal_pair_count | 8.625 (4/4父) | 10 (4/4父) | count |
| FF_illegal_pair_count | 10.25 (4/4父) | 19.375 (4/4父) | count |
| free_minimum_angle_lt5_fraction | 0.64658 (4/4父) | 0.643542 (4/4父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.528287 (4/4父) | 0.537927 (4/4父) | fraction_0_to_1 |
| deleted_free_faces | 106.125 (4/4父) | 105.125 (4/4父) | count |
| actual_free_faces | 67.875 (4/4父) | 68.875 (4/4父) | count |
| free_total_surface_area | 21.0498 (4/4父) | 22.8271 (4/4父) | model_length_squared |
| new_bridge_faces | 0 (4/4父) | 0 (4/4父) | count |
| bridge_area | 0 (4/4父) | 0 (4/4父) | model_length_squared |
| deleted_original_POST_faces | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_isolated_1face_edge_components | 18 (4/4父) | 19.5 (4/4父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 15.625 (4/4父) | 17.375 (4/4父) | count |
| C_F_valid_edge_count | 0 (4/4父) | 0 (4/4父) | count |
| C_F_valid_edge_total_length | 0 (4/4父) | 0 (4/4父) | model_length |
| C_F_has_valid_edge | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (4/4父) | 0 (4/4父) | count |
| nonmanifold_edges | 2 (4/4父) | 2.25 (4/4父) | count |
| C_F_nonmanifold_edge_count | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (4/4父) | 1 (4/4父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/4父) | NA (0/4父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/4父) | NA (0/4父) | model_length_squared |

## HOLDOUT / RAW

| metric | B narrow | W wide | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.25576 (1/1父) | 0.251152 (1/1父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.857143 (1/1父) | 0.857143 (1/1父) | fraction_0_to_1 |
| ConflictCov_C | 0.0983887 (1/1父) | 0.103027 (1/1父) | fraction_0_to_1 |
| CF_illegal_pair_count | 85 (1/1父) | 82 (1/1父) | count |
| FF_illegal_pair_count | 259 (1/1父) | 255 (1/1父) | count |
| free_minimum_angle_lt5_fraction | 0.175115 (1/1父) | 0.172811 (1/1父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.110599 (1/1父) | 0.108295 (1/1父) | fraction_0_to_1 |
| deleted_free_faces | 0 (1/1父) | 0 (1/1父) | count |
| actual_free_faces | 217 (1/1父) | 217 (1/1父) | count |
| free_total_surface_area | 21.8687 (1/1父) | 21.972 (1/1父) | model_length_squared |
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

| metric | B narrow | W wide | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.201094 (1/1父) | 0.209691 (1/1父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.110313 (1/1父) | 0.0972441 (1/1父) | fraction_0_to_1 |
| ConflictCov_C | 0.101318 (1/1父) | 0.122314 (1/1父) | fraction_0_to_1 |
| CF_illegal_pair_count | 32 (1/1父) | 36.5 (1/1父) | count |
| FF_illegal_pair_count | 8.5 (1/1父) | 7 (1/1父) | count |
| free_minimum_angle_lt5_fraction | 0.178031 (1/1父) | 0.187068 (1/1父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.118875 (1/1父) | 0.15209 (1/1父) | fraction_0_to_1 |
| deleted_free_faces | 90.5 (1/1父) | 88.5 (1/1父) | count |
| actual_free_faces | 126.5 (1/1父) | 128.5 (1/1父) | count |
| free_total_surface_area | 10.3283 (1/1父) | 10.3283 (1/1父) | model_length_squared |
| new_bridge_faces | 0 (1/1父) | 0 (1/1父) | count |
| bridge_area | 0 (1/1父) | 0 (1/1父) | model_length_squared |
| deleted_original_POST_faces | 0 (1/1父) | 0 (1/1父) | count |
| hinge_diag_isolated_1face_edge_components | 99.5 (1/1父) | 97.5 (1/1父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 99.5 (1/1父) | 97.5 (1/1父) | count |
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

| metric | B narrow | W wide | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.197188 (1/1父) | 0.205754 (1/1父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.0907813 (1/1父) | 0.0738038 (1/1父) | fraction_0_to_1 |
| ConflictCov_C | 0.101074 (1/1父) | 0.123779 (1/1父) | fraction_0_to_1 |
| CF_illegal_pair_count | 31 (1/1父) | 33.5 (1/1父) | count |
| FF_illegal_pair_count | 6.5 (1/1父) | 5.5 (1/1父) | count |
| free_minimum_angle_lt5_fraction | 0.170219 (1/1父) | 0.183131 (1/1父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.114969 (1/1父) | 0.128831 (1/1父) | fraction_0_to_1 |
| deleted_free_faces | 90.5 (1/1父) | 88.5 (1/1父) | count |
| actual_free_faces | 126.5 (1/1父) | 128.5 (1/1父) | count |
| free_total_surface_area | 10.3399 (1/1父) | 10.4533 (1/1父) | model_length_squared |
| new_bridge_faces | 0 (1/1父) | 0 (1/1父) | count |
| bridge_area | 0 (1/1父) | 0 (1/1父) | model_length_squared |
| deleted_original_POST_faces | 0 (1/1父) | 0 (1/1父) | count |
| hinge_diag_isolated_1face_edge_components | 66.5 (1/1父) | 61 (1/1父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 66.5 (1/1父) | 61 (1/1父) | count |
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

| metric | B narrow | W wide | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.10801 (2/2父) | 0.0702986 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.685105 (2/2父) | 0.759693 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 36.5 (2/2父) | 33 (2/2父) | count |
| FF_illegal_pair_count | 196.25 (2/2父) | 292.25 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.534537 (2/2父) | 0.512923 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.472204 (2/2父) | 0.461954 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 0 (2/2父) | 0 (2/2父) | count |
| actual_free_faces | 319 (2/2父) | 319 (2/2父) | count |
| free_total_surface_area | 23.5768 (2/2父) | 34.7958 (2/2父) | model_length_squared |
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

| metric | B narrow | W wide | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0 (2/2父) | 0.0204918 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.0158378 (2/2父) | 0.103663 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 0 (2/2父) | 3.5 (2/2父) | count |
| FF_illegal_pair_count | 1 (2/2父) | 3.5 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.314143 (2/2父) | 0.350612 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.314143 (2/2父) | 0.326022 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 268 (2/2父) | 269.5 (2/2父) | count |
| actual_free_faces | 51 (2/2父) | 49.5 (2/2父) | count |
| free_total_surface_area | 8.38694 (2/2父) | 15.5351 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 32.75 (2/2父) | 26.5 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 32.75 (2/2父) | 26.25 (2/2父) | count |
| C_F_valid_edge_count | 1 (2/2父) | 1 (2/2父) | count |
| C_F_valid_edge_total_length | 5.31632 (2/2父) | 5.31632 (2/2父) | model_length |
| C_F_has_valid_edge | 0.5 (2/2父) | 0.5 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 1 (2/2父) | 1 (2/2父) | count |
| nonmanifold_edges | 0 (2/2父) | 0.25 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.0253524 (2/2父) | 0.016982 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.950562 (2/2父) | 0.925291 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.0240854 (2/2父) | 0.0577272 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | NA (0/2父) | model_length_squared |

## HIGH_TRAIN / CLEAN_FINAL_POST

| metric | B narrow | W wide | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0 (2/2父) | 0.0204918 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.0158378 (2/2父) | 0.091368 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 0 (2/2父) | 3.5 (2/2父) | count |
| FF_illegal_pair_count | 1 (2/2父) | 3 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.314143 (2/2父) | 0.346514 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.314143 (2/2父) | 0.317826 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 268 (2/2父) | 269.5 (2/2父) | count |
| actual_free_faces | 51 (2/2父) | 49.5 (2/2父) | count |
| free_total_surface_area | 8.4122 (2/2父) | 15.6312 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 30.5 (2/2父) | 22.5 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 30.5 (2/2父) | 22.25 (2/2父) | count |
| C_F_valid_edge_count | 1 (2/2父) | 1 (2/2父) | count |
| C_F_valid_edge_total_length | 5.31632 (2/2父) | 5.31632 (2/2父) | model_length |
| C_F_has_valid_edge | 0.5 (2/2父) | 0.5 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 1 (2/2父) | 1 (2/2父) | count |
| nonmanifold_edges | 0 (2/2父) | 0.25 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.025211 (2/2父) | 0.0173604 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.950747 (2/2父) | 0.925621 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.0240424 (2/2父) | 0.0570189 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | NA (0/2父) | model_length_squared |

## HIGH_HOLDOUT / RAW

| metric | B narrow | W wide | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.109216 (2/2父) | 0.108849 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.943372 (2/2父) | 0.94309 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | 0.0866699 (2/2父) | 0.18811 (2/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 108.25 (2/2父) | 112.25 (2/2父) | count |
| FF_illegal_pair_count | 976.25 (2/2父) | 1011 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.427594 (2/2父) | 0.413558 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.393377 (2/2父) | 0.373393 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 0 (2/2父) | 0 (2/2父) | count |
| actual_free_faces | 522.5 (2/2父) | 522.5 (2/2父) | count |
| free_total_surface_area | 51.6978 (2/2父) | 53.5974 (2/2父) | model_length_squared |
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

| metric | B narrow | W wide | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.207484 (2/2父) | 0.201852 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.162757 (2/2父) | 0.179984 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | 0.0449219 (2/2父) | 0.0489502 (2/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 31.25 (2/2父) | 29.5 (2/2父) | count |
| FF_illegal_pair_count | 5 (2/2父) | 6.25 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.454865 (2/2父) | 0.501031 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.338406 (2/2父) | 0.404323 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 473.5 (2/2父) | 471.5 (2/2父) | count |
| actual_free_faces | 49 (2/2父) | 51 (2/2父) | count |
| free_total_surface_area | 29.9687 (2/2父) | 30.0612 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 19.5 (2/2父) | 20.5 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 16.25 (2/2父) | 16 (2/2父) | count |
| C_F_valid_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| C_F_valid_edge_total_length | 0 (2/2父) | 0 (2/2父) | model_length |
| C_F_has_valid_edge | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| nonmanifold_edges | 2.5 (2/2父) | 2.25 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0.25 (2/2父) | 0.25 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.811128 (2/2父) | 0.754162 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.188872 (2/2父) | 0.245838 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | NA (0/2父) | model_length_squared |

## HIGH_HOLDOUT / CLEAN_FINAL_POST

| metric | B narrow | W wide | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.207484 (2/2父) | 0.201852 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.124807 (2/2父) | 0.152317 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | 0.0455322 (2/2父) | 0.0482178 (2/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 31 (2/2父) | 29 (2/2父) | count |
| FF_illegal_pair_count | 3.75 (2/2父) | 5.25 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.434773 (2/2父) | 0.494781 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.311557 (2/2父) | 0.404323 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 473.5 (2/2父) | 471.5 (2/2父) | count |
| actual_free_faces | 49 (2/2父) | 51 (2/2父) | count |
| free_total_surface_area | 30.085 (2/2父) | 30.1532 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 13.25 (2/2父) | 15 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 10.75 (2/2父) | 11.5 (2/2父) | count |
| C_F_valid_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| C_F_valid_edge_total_length | 0 (2/2父) | 0 (2/2父) | model_length |
| C_F_has_valid_edge | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| nonmanifold_edges | 2.5 (2/2父) | 2.25 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0.25 (2/2父) | 0.25 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.811829 (2/2父) | 0.754152 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.188171 (2/2父) | 0.245848 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | NA (0/2父) | model_length_squared |

## NEW_WIDE_TRAIN / RAW

| metric | B narrow | W wide | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0853239 (4/4父) | 0.0672439 (4/4父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.926407 (4/4父) | 0.903743 (4/4父) | fraction_0_to_1 |
| ConflictCov_C | 0.168213 (2/4父) | 0.106201 (2/4父) | fraction_0_to_1 |
| CF_illegal_pair_count | 90.5 (4/4父) | 82.125 (4/4父) | count |
| FF_illegal_pair_count | 1257 (4/4父) | 1014.38 (4/4父) | count |
| free_minimum_angle_lt5_fraction | 0.472585 (4/4父) | 0.385643 (4/4父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.363383 (4/4父) | 0.302908 (4/4父) | fraction_0_to_1 |
| deleted_free_faces | 0 (4/4父) | 0 (4/4父) | count |
| actual_free_faces | 553 (4/4父) | 553 (4/4父) | count |
| free_total_surface_area | 54.2272 (4/4父) | 26.7443 (4/4父) | model_length_squared |
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

| metric | B narrow | W wide | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0662325 (4/4父) | 0.101289 (4/4父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.363772 (4/4父) | 0.238405 (4/4父) | fraction_0_to_1 |
| ConflictCov_C | 0.0137939 (2/4父) | 0.0439453 (2/4父) | fraction_0_to_1 |
| CF_illegal_pair_count | 12.125 (4/4父) | 10.625 (4/4父) | count |
| FF_illegal_pair_count | 68 (4/4父) | 26.625 (4/4父) | count |
| free_minimum_angle_lt5_fraction | 0.538157 (4/4父) | 0.580688 (4/4父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.454722 (4/4父) | 0.497017 (4/4父) | fraction_0_to_1 |
| deleted_free_faces | 446 (4/4父) | 458.875 (4/4父) | count |
| actual_free_faces | 107 (4/4父) | 94.125 (4/4父) | count |
| free_total_surface_area | 28.0413 (4/4父) | 10.0658 (4/4父) | model_length_squared |
| new_bridge_faces | 0 (4/4父) | 0 (4/4父) | count |
| bridge_area | 0 (4/4父) | 0 (4/4父) | model_length_squared |
| deleted_original_POST_faces | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_isolated_1face_edge_components | 56.125 (4/4父) | 54.75 (4/4父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 48.75 (4/4父) | 50.25 (4/4父) | count |
| C_F_valid_edge_count | 1.25 (4/4父) | 0.875 (4/4父) | count |
| C_F_valid_edge_total_length | 5.30164 (4/4父) | 3.71099 (4/4父) | model_length |
| C_F_has_valid_edge | 0.5 (4/4父) | 0.375 (4/4父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 1.25 (4/4父) | 0.875 (4/4父) | count |
| nonmanifold_edges | 4.25 (4/4父) | 2.625 (4/4父) | count |
| C_F_nonmanifold_edge_count | 0.125 (4/4父) | 0 (4/4父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.0435804 (4/4父) | 0.0408581 (4/4父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.697812 (4/4父) | 0.868958 (4/4父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.258607 (4/4父) | 0.090184 (4/4父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/4父) | NA (0/4父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/4父) | NA (0/4父) | model_length_squared |

## NEW_WIDE_TRAIN / CLEAN_FINAL_POST

| metric | B narrow | W wide | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0662325 (4/4父) | 0.100472 (4/4父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.348225 (4/4父) | 0.216116 (4/4父) | fraction_0_to_1 |
| ConflictCov_C | 0.0135498 (2/4父) | 0.0328369 (2/4父) | fraction_0_to_1 |
| CF_illegal_pair_count | 12.125 (4/4父) | 10.5 (4/4父) | count |
| FF_illegal_pair_count | 63.75 (4/4父) | 24.125 (4/4父) | count |
| free_minimum_angle_lt5_fraction | 0.531599 (4/4父) | 0.57075 (4/4父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.44663 (4/4父) | 0.483564 (4/4父) | fraction_0_to_1 |
| deleted_free_faces | 446 (4/4父) | 458.875 (4/4父) | count |
| actual_free_faces | 107 (4/4父) | 94.125 (4/4父) | count |
| free_total_surface_area | 28.3116 (4/4父) | 10.1886 (4/4父) | model_length_squared |
| new_bridge_faces | 0 (4/4父) | 0 (4/4父) | count |
| bridge_area | 0 (4/4父) | 0 (4/4父) | model_length_squared |
| deleted_original_POST_faces | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_isolated_1face_edge_components | 44.5 (4/4父) | 41.125 (4/4父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 38.125 (4/4父) | 37.125 (4/4父) | count |
| C_F_valid_edge_count | 1.25 (4/4父) | 0.875 (4/4父) | count |
| C_F_valid_edge_total_length | 5.30164 (4/4父) | 3.71099 (4/4父) | model_length |
| C_F_has_valid_edge | 0.5 (4/4父) | 0.375 (4/4父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 1.25 (4/4父) | 0.875 (4/4父) | count |
| nonmanifold_edges | 4.25 (4/4父) | 2.625 (4/4父) | count |
| C_F_nonmanifold_edge_count | 0.125 (4/4父) | 0 (4/4父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.0787174 (4/4父) | 0.0476398 (4/4父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.697861 (4/4父) | 0.869146 (4/4父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.223422 (4/4父) | 0.0832137 (4/4父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/4父) | NA (0/4父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/4父) | NA (0/4父) | model_length_squared |
