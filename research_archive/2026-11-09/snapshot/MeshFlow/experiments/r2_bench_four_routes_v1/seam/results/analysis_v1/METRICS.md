# Bench original SEAM continuation: complete numerical results

B is the frozen narrow natural-N Native source. S continues its live generator and AdamW for1500 original owner-only SEAM updates; the historical C-only head is frozen. NEW_WIDE_TRAIN parents are unseen for B and S. This compares the whole SEAM continuation, not an equal-budget Native contrast.
所有分数沿用原定义，父等权；I主展示只用PASS支持域，完整177字段保留原值与NA。
Fractions use0-1, not percent. OLD/FOCUS/DEV/HOLDOUT/HIGH_TRAIN/HIGH_HOLDOUT/NEW_WIDE_TRAIN remain separate; no automatic promotion.
本文件不产生视觉标签。实际图审及支撑完整性必须另见 visual_review。

## OLD / RAW

| metric | B narrow Native | S SEAM1500 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0922111 (8/8父) | 0.0651444 (8/8父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.850119 (8/8父) | 0.854348 (8/8父) | fraction_0_to_1 |
| ConflictCov_C | 0.0376282 (8/8父) | 0.0174866 (8/8父) | fraction_0_to_1 |
| CF_illegal_pair_count | 33.125 (8/8父) | 15.75 (8/8父) | count |
| FF_illegal_pair_count | 205.438 (8/8父) | 251.125 (8/8父) | count |
| free_minimum_angle_lt5_fraction | 0.389759 (8/8父) | 0.371979 (8/8父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.316694 (8/8父) | 0.299663 (8/8父) | fraction_0_to_1 |
| deleted_free_faces | 0 (8/8父) | 0 (8/8父) | count |
| actual_free_faces | 196.625 (8/8父) | 196.625 (8/8父) | count |
| free_total_surface_area | 31.3943 (8/8父) | 34.7487 (8/8父) | model_length_squared |
| new_bridge_faces | 0 (8/8父) | 0 (8/8父) | count |
| bridge_area | 0 (8/8父) | 0 (8/8父) | model_length_squared |
| deleted_original_POST_faces | 0 (8/8父) | 0 (8/8父) | count |
| hinge_diag_isolated_1face_edge_components | 196.625 (8/8父) | 179.5 (8/8父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 196.625 (8/8父) | 179.5 (8/8父) | count |
| C_F_valid_edge_count | 0 (8/8父) | 17.125 (8/8父) | count |
| C_F_valid_edge_total_length | 0 (8/8父) | 17.1865 (8/8父) | model_length |
| C_F_has_valid_edge | 0 (8/8父) | 1 (8/8父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (8/8父) | 17.125 (8/8父) | count |
| nonmanifold_edges | 0 (8/8父) | 0 (8/8父) | count |
| C_F_nonmanifold_edge_count | 0 (8/8父) | 0 (8/8父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (8/8父) | 0.117379 (8/8父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (8/8父) | 0.882621 (8/8父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (8/8父) | 0 (8/8父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/8父) | 0 (8/8父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/8父) | 31.4003 (8/8父) | model_length_squared |

## OLD / BASE_POST

| metric | B narrow Native | S SEAM1500 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0653743 (8/8父) | 0.0177253 (8/8父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.120557 (8/8父) | 0.174951 (8/8父) | fraction_0_to_1 |
| ConflictCov_C | 0.0150146 (8/8父) | 0.0365601 (8/8父) | fraction_0_to_1 |
| CF_illegal_pair_count | 6.3125 (8/8父) | 1.375 (8/8父) | count |
| FF_illegal_pair_count | 8.375 (8/8父) | 15.4375 (8/8父) | count |
| free_minimum_angle_lt5_fraction | 0.35971 (8/8父) | 0.365575 (8/8父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.286391 (8/8父) | 0.298723 (8/8父) | fraction_0_to_1 |
| deleted_free_faces | 110.312 (8/8父) | 105.375 (8/8父) | count |
| actual_free_faces | 86.3125 (8/8父) | 91.25 (8/8父) | count |
| free_total_surface_area | 24.6144 (8/8父) | 27.9604 (8/8父) | model_length_squared |
| new_bridge_faces | 0 (8/8父) | 0 (8/8父) | count |
| bridge_area | 0 (8/8父) | 0 (8/8父) | model_length_squared |
| deleted_original_POST_faces | 0 (8/8父) | 0 (8/8父) | count |
| hinge_diag_isolated_1face_edge_components | 41.0625 (8/8父) | 35.5625 (8/8父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 40.1875 (8/8父) | 35.3125 (8/8父) | count |
| C_F_valid_edge_count | 1 (8/8父) | 17 (8/8父) | count |
| C_F_valid_edge_total_length | 2.06804 (8/8父) | 16.4541 (8/8父) | model_length |
| C_F_has_valid_edge | 0.5625 (8/8父) | 1 (8/8父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 1 (8/8父) | 17 (8/8父) | count |
| nonmanifold_edges | 0.4375 (8/8父) | 0.375 (8/8父) | count |
| C_F_nonmanifold_edge_count | 0.0625 (8/8父) | 0.3125 (8/8父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.0599955 (8/8父) | 0.18758 (8/8父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.607836 (8/8父) | 0.583933 (8/8父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.332168 (8/8父) | 0.228488 (8/8父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/8父) | 0.0796799 (8/8父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/8父) | 25.1141 (8/8父) | model_length_squared |

## OLD / CLEAN_FINAL_POST

| metric | B narrow Native | S SEAM1500 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.06276 (8/8父) | 0.0148829 (8/8父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.091824 (8/8父) | 0.124493 (8/8父) | fraction_0_to_1 |
| ConflictCov_C | 0.0148621 (8/8父) | 0.0363464 (8/8父) | fraction_0_to_1 |
| CF_illegal_pair_count | 6.0625 (8/8父) | 1 (8/8父) | count |
| FF_illegal_pair_count | 6.3125 (8/8父) | 11.625 (8/8父) | count |
| free_minimum_angle_lt5_fraction | 0.344514 (8/8父) | 0.351341 (8/8父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.27314 (8/8父) | 0.289113 (8/8父) | fraction_0_to_1 |
| deleted_free_faces | 110.312 (8/8父) | 105.375 (8/8父) | count |
| actual_free_faces | 86.3125 (8/8父) | 91.5 (8/8父) | count |
| free_total_surface_area | 24.8545 (8/8父) | 28.2105 (8/8父) | model_length_squared |
| new_bridge_faces | 0 (8/8父) | 0.25 (8/8父) | count |
| bridge_area | 0 (8/8父) | 0.0192059 (8/8父) | model_length_squared |
| deleted_original_POST_faces | 0 (8/8父) | 0 (8/8父) | count |
| hinge_diag_isolated_1face_edge_components | 25.6875 (8/8父) | 22.4375 (8/8父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 24.875 (8/8父) | 22.25 (8/8父) | count |
| C_F_valid_edge_count | 1 (8/8父) | 17.0625 (8/8父) | count |
| C_F_valid_edge_total_length | 2.06804 (8/8父) | 16.4575 (8/8父) | model_length |
| C_F_has_valid_edge | 0.5625 (8/8父) | 1 (8/8父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 1 (8/8父) | 17.0625 (8/8父) | count |
| nonmanifold_edges | 0.4375 (8/8父) | 0.375 (8/8父) | count |
| C_F_nonmanifold_edge_count | 0.0625 (8/8父) | 0.3125 (8/8父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.0902084 (8/8父) | 0.257355 (8/8父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.607965 (8/8父) | 0.580861 (8/8父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.301827 (8/8父) | 0.161784 (8/8父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/8父) | 0.179077 (8/8父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/8父) | 25.3239 (8/8父) | model_length_squared |

## FOCUS / RAW

| metric | B narrow Native | S SEAM1500 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.122537 (2/2父) | 0.0793363 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.857227 (2/2父) | 0.890914 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | 0.0714111 (2/2父) | 0.010498 (2/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 40.25 (2/2父) | 19.75 (2/2父) | count |
| FF_illegal_pair_count | 190 (2/2父) | 236.25 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.499971 (2/2父) | 0.478363 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.402463 (2/2父) | 0.379764 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 0 (2/2父) | 0 (2/2父) | count |
| actual_free_faces | 188 (2/2父) | 188 (2/2父) | count |
| free_total_surface_area | 15.7702 (2/2父) | 15.9657 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 188 (2/2父) | 164 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 188 (2/2父) | 164 (2/2父) | count |
| C_F_valid_edge_count | 0 (2/2父) | 24 (2/2父) | count |
| C_F_valid_edge_total_length | 0 (2/2父) | 54.2455 (2/2父) | model_length |
| C_F_has_valid_edge | 0 (2/2父) | 1 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (2/2父) | 24 (2/2父) | count |
| nonmanifold_edges | 0 (2/2父) | 0 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (2/2父) | 0.223817 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (2/2父) | 0.776183 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | 0 (2/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | 12.0735 (2/2父) | model_length_squared |

## FOCUS / BASE_POST

| metric | B narrow Native | S SEAM1500 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.034897 (2/2父) | 0.0699665 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 0 (2/2父) | 0 (2/2父) | count |
| FF_illegal_pair_count | 0.5 (2/2父) | 2.25 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.659809 (2/2父) | 0.636866 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.624138 (2/2父) | 0.559819 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 156.5 (2/2父) | 137.25 (2/2父) | count |
| actual_free_faces | 31.5 (2/2父) | 50.75 (2/2父) | count |
| free_total_surface_area | 3.45842 (2/2父) | 5.33761 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 19.75 (2/2父) | 21 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 19.25 (2/2父) | 20.5 (2/2父) | count |
| C_F_valid_edge_count | 1 (2/2父) | 23.75 (2/2父) | count |
| C_F_valid_edge_total_length | 3.62273 (2/2父) | 54.2132 (2/2父) | model_length |
| C_F_has_valid_edge | 0.75 (2/2父) | 1 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 1 (2/2父) | 23.75 (2/2父) | count |
| nonmanifold_edges | 0.25 (2/2父) | 1 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0 (2/2父) | 0.75 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.396729 (2/2父) | 0.693014 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.602164 (2/2父) | 0.115923 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.00110741 (2/2父) | 0.191063 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | 0.290271 (2/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | 2.52494 (2/2父) | model_length_squared |

## FOCUS / CLEAN_FINAL_POST

| metric | B narrow Native | S SEAM1500 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.0217391 (2/2父) | 0.0542444 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 0 (2/2父) | 0 (2/2父) | count |
| FF_illegal_pair_count | 0.25 (2/2父) | 1.5 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.65323 (2/2父) | 0.623953 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.617559 (2/2父) | 0.535713 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 156.5 (2/2父) | 137.25 (2/2父) | count |
| actual_free_faces | 31.5 (2/2父) | 51.25 (2/2父) | count |
| free_total_surface_area | 3.59336 (2/2父) | 5.69097 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0.5 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0.000725976 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 17.25 (2/2父) | 17.25 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 16.75 (2/2父) | 16.75 (2/2父) | count |
| C_F_valid_edge_count | 1 (2/2父) | 23.75 (2/2父) | count |
| C_F_valid_edge_total_length | 3.62273 (2/2父) | 54.2132 (2/2父) | model_length |
| C_F_has_valid_edge | 0.75 (2/2父) | 1 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 1 (2/2父) | 23.75 (2/2父) | count |
| nonmanifold_edges | 0.25 (2/2父) | 1 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0 (2/2父) | 0.75 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.395545 (2/2父) | 0.71043 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.603347 (2/2父) | 0.116145 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.00110728 (2/2父) | 0.173425 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | 0.367963 (2/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | 2.72681 (2/2父) | model_length_squared |

## DEV / RAW

| metric | B narrow Native | S SEAM1500 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0595824 (4/4父) | 0.0665653 (4/4父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.874498 (4/4父) | 0.900323 (4/4父) | fraction_0_to_1 |
| ConflictCov_C | 0.125 (4/4父) | 0.232849 (4/4父) | fraction_0_to_1 |
| CF_illegal_pair_count | 13 (4/4父) | 14.5 (4/4父) | count |
| FF_illegal_pair_count | 233.25 (4/4父) | 282.375 (4/4父) | count |
| free_minimum_angle_lt5_fraction | 0.444582 (4/4父) | 0.386407 (4/4父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.309994 (4/4父) | 0.260636 (4/4父) | fraction_0_to_1 |
| deleted_free_faces | 0 (4/4父) | 0 (4/4父) | count |
| actual_free_faces | 174 (4/4父) | 174 (4/4父) | count |
| free_total_surface_area | 34.869 (4/4父) | 36.5954 (4/4父) | model_length_squared |
| new_bridge_faces | 0 (4/4父) | 0 (4/4父) | count |
| bridge_area | 0 (4/4父) | 0 (4/4父) | model_length_squared |
| deleted_original_POST_faces | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_isolated_1face_edge_components | 174 (4/4父) | 171.5 (4/4父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 174 (4/4父) | 171.5 (4/4父) | count |
| C_F_valid_edge_count | 0 (4/4父) | 2.5 (4/4父) | count |
| C_F_valid_edge_total_length | 0 (4/4父) | 10.136 (4/4父) | model_length |
| C_F_has_valid_edge | 0 (4/4父) | 1 (4/4父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (4/4父) | 2.5 (4/4父) | count |
| nonmanifold_edges | 0 (4/4父) | 0 (4/4父) | count |
| C_F_nonmanifold_edge_count | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (4/4父) | 0.0478814 (4/4父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (4/4父) | 0.952119 (4/4父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/4父) | 0 (4/4父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/4父) | 34.8023 (4/4父) | model_length_squared |

## DEV / BASE_POST

| metric | B narrow Native | S SEAM1500 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0617901 (4/4父) | 0.0726779 (4/4父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.164067 (4/4父) | 0.282141 (4/4父) | fraction_0_to_1 |
| ConflictCov_C | 0.115479 (4/4父) | 0.107483 (4/4父) | fraction_0_to_1 |
| CF_illegal_pair_count | 9 (4/4父) | 9.375 (4/4父) | count |
| FF_illegal_pair_count | 12.5 (4/4父) | 22.75 (4/4父) | count |
| free_minimum_angle_lt5_fraction | 0.658842 (4/4父) | 0.563398 (4/4父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.540728 (4/4父) | 0.396805 (4/4父) | fraction_0_to_1 |
| deleted_free_faces | 106.125 (4/4父) | 92.75 (4/4父) | count |
| actual_free_faces | 67.875 (4/4父) | 81.25 (4/4父) | count |
| free_total_surface_area | 20.8231 (4/4父) | 25.8888 (4/4父) | model_length_squared |
| new_bridge_faces | 0 (4/4父) | 0 (4/4父) | count |
| bridge_area | 0 (4/4父) | 0 (4/4父) | model_length_squared |
| deleted_original_POST_faces | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_isolated_1face_edge_components | 25.25 (4/4父) | 32.125 (4/4父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 22.625 (4/4父) | 30.875 (4/4父) | count |
| C_F_valid_edge_count | 0 (4/4父) | 2.5 (4/4父) | count |
| C_F_valid_edge_total_length | 0 (4/4父) | 10.136 (4/4父) | model_length |
| C_F_has_valid_edge | 0 (4/4父) | 1 (4/4父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (4/4父) | 2.5 (4/4父) | count |
| nonmanifold_edges | 2 (4/4父) | 1 (4/4父) | count |
| C_F_nonmanifold_edge_count | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (4/4父) | 0.0598115 (4/4父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (4/4父) | 0.378238 (4/4父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (4/4父) | 0.561951 (4/4父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/4父) | 0 (4/4父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/4父) | 24.2472 (4/4父) | model_length_squared |

## DEV / CLEAN_FINAL_POST

| metric | B narrow Native | S SEAM1500 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0603856 (4/4父) | 0.0726779 (4/4父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.11494 (4/4父) | 0.237997 (4/4父) | fraction_0_to_1 |
| ConflictCov_C | 0.117188 (4/4父) | 0.111267 (4/4父) | fraction_0_to_1 |
| CF_illegal_pair_count | 8.625 (4/4父) | 9.375 (4/4父) | count |
| FF_illegal_pair_count | 10.25 (4/4父) | 20.125 (4/4父) | count |
| free_minimum_angle_lt5_fraction | 0.64658 (4/4父) | 0.540213 (4/4父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.528287 (4/4父) | 0.382429 (4/4父) | fraction_0_to_1 |
| deleted_free_faces | 106.125 (4/4父) | 92.75 (4/4父) | count |
| actual_free_faces | 67.875 (4/4父) | 81.25 (4/4父) | count |
| free_total_surface_area | 21.0498 (4/4父) | 26.1206 (4/4父) | model_length_squared |
| new_bridge_faces | 0 (4/4父) | 0 (4/4父) | count |
| bridge_area | 0 (4/4父) | 0 (4/4父) | model_length_squared |
| deleted_original_POST_faces | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_isolated_1face_edge_components | 18 (4/4父) | 21 (4/4父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 15.625 (4/4父) | 20 (4/4父) | count |
| C_F_valid_edge_count | 0 (4/4父) | 2.5 (4/4父) | count |
| C_F_valid_edge_total_length | 0 (4/4父) | 10.136 (4/4父) | model_length |
| C_F_has_valid_edge | 0 (4/4父) | 1 (4/4父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (4/4父) | 2.5 (4/4父) | count |
| nonmanifold_edges | 2 (4/4父) | 1 (4/4父) | count |
| C_F_nonmanifold_edge_count | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (4/4父) | 0.0592254 (4/4父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (4/4父) | 0.378185 (4/4父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (4/4父) | 0.562589 (4/4父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/4父) | 0 (4/4父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/4父) | 24.479 (4/4父) | model_length_squared |

## HOLDOUT / RAW

| metric | B narrow Native | S SEAM1500 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.25576 (1/1父) | 0.195853 (1/1父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.857143 (1/1父) | 0.887097 (1/1父) | fraction_0_to_1 |
| ConflictCov_C | 0.0983887 (1/1父) | 0.212646 (1/1父) | fraction_0_to_1 |
| CF_illegal_pair_count | 85 (1/1父) | 60.5 (1/1父) | count |
| FF_illegal_pair_count | 259 (1/1父) | 293.5 (1/1父) | count |
| free_minimum_angle_lt5_fraction | 0.175115 (1/1父) | 0.14977 (1/1父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.110599 (1/1父) | 0.0852535 (1/1父) | fraction_0_to_1 |
| deleted_free_faces | 0 (1/1父) | 0 (1/1父) | count |
| actual_free_faces | 217 (1/1父) | 217 (1/1父) | count |
| free_total_surface_area | 21.8687 (1/1父) | 24.8415 (1/1父) | model_length_squared |
| new_bridge_faces | 0 (1/1父) | 0 (1/1父) | count |
| bridge_area | 0 (1/1父) | 0 (1/1父) | model_length_squared |
| deleted_original_POST_faces | 0 (1/1父) | 0 (1/1父) | count |
| hinge_diag_isolated_1face_edge_components | 217 (1/1父) | 196 (1/1父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 217 (1/1父) | 196 (1/1父) | count |
| C_F_valid_edge_count | 0 (1/1父) | 21 (1/1父) | count |
| C_F_valid_edge_total_length | 0 (1/1父) | 14.5728 (1/1父) | model_length |
| C_F_has_valid_edge | 0 (1/1父) | 1 (1/1父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (1/1父) | 21 (1/1父) | count |
| nonmanifold_edges | 0 (1/1父) | 0 (1/1父) | count |
| C_F_nonmanifold_edge_count | 0 (1/1父) | 0 (1/1父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (1/1父) | 0.228911 (1/1父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (1/1父) | 0.771089 (1/1父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (1/1父) | 0 (1/1父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/1父) | 0 (1/1父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/1父) | 19.1613 (1/1父) | model_length_squared |

## HOLDOUT / BASE_POST

| metric | B narrow Native | S SEAM1500 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.201094 (1/1父) | 0.0958874 (1/1父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.110313 (1/1父) | 0.310823 (1/1父) | fraction_0_to_1 |
| ConflictCov_C | 0.101318 (1/1父) | 0.381836 (1/1父) | fraction_0_to_1 |
| CF_illegal_pair_count | 32 (1/1父) | 16 (1/1父) | count |
| FF_illegal_pair_count | 8.5 (1/1父) | 26 (1/1父) | count |
| free_minimum_angle_lt5_fraction | 0.178031 (1/1父) | 0.168182 (1/1父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.118875 (1/1父) | 0.138745 (1/1父) | fraction_0_to_1 |
| deleted_free_faces | 90.5 (1/1父) | 81 (1/1父) | count |
| actual_free_faces | 126.5 (1/1父) | 136 (1/1父) | count |
| free_total_surface_area | 10.3283 (1/1父) | 14.9349 (1/1父) | model_length_squared |
| new_bridge_faces | 0 (1/1父) | 0 (1/1父) | count |
| bridge_area | 0 (1/1父) | 0 (1/1父) | model_length_squared |
| deleted_original_POST_faces | 0 (1/1父) | 0 (1/1父) | count |
| hinge_diag_isolated_1face_edge_components | 99.5 (1/1父) | 89 (1/1父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 99.5 (1/1父) | 89 (1/1父) | count |
| C_F_valid_edge_count | 0 (1/1父) | 21 (1/1父) | count |
| C_F_valid_edge_total_length | 0 (1/1父) | 14.5728 (1/1父) | model_length |
| C_F_has_valid_edge | 0 (1/1父) | 1 (1/1父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (1/1父) | 21 (1/1父) | count |
| nonmanifold_edges | 0 (1/1父) | 0 (1/1父) | count |
| C_F_nonmanifold_edge_count | 0 (1/1父) | 0 (1/1父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (1/1父) | 0.414535 (1/1父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (1/1父) | 0.472043 (1/1父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (1/1父) | 0.113421 (1/1父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/1父) | 0 (1/1父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/1父) | 8.73719 (1/1父) | model_length_squared |

## HOLDOUT / CLEAN_FINAL_POST

| metric | B narrow Native | S SEAM1500 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.197188 (1/1父) | 0.0920996 (1/1父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.0907813 (1/1父) | 0.269589 (1/1父) | fraction_0_to_1 |
| ConflictCov_C | 0.101074 (1/1父) | 0.381592 (1/1父) | fraction_0_to_1 |
| CF_illegal_pair_count | 31 (1/1父) | 15.5 (1/1父) | count |
| FF_illegal_pair_count | 6.5 (1/1父) | 22 (1/1父) | count |
| free_minimum_angle_lt5_fraction | 0.170219 (1/1父) | 0.168182 (1/1父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.114969 (1/1父) | 0.135173 (1/1父) | fraction_0_to_1 |
| deleted_free_faces | 90.5 (1/1父) | 81 (1/1父) | count |
| actual_free_faces | 126.5 (1/1父) | 136 (1/1父) | count |
| free_total_surface_area | 10.3399 (1/1父) | 14.9461 (1/1父) | model_length_squared |
| new_bridge_faces | 0 (1/1父) | 0 (1/1父) | count |
| bridge_area | 0 (1/1父) | 0 (1/1父) | model_length_squared |
| deleted_original_POST_faces | 0 (1/1父) | 0 (1/1父) | count |
| hinge_diag_isolated_1face_edge_components | 66.5 (1/1父) | 64 (1/1父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 66.5 (1/1父) | 64 (1/1父) | count |
| C_F_valid_edge_count | 0 (1/1父) | 21 (1/1父) | count |
| C_F_valid_edge_total_length | 0 (1/1父) | 14.5728 (1/1父) | model_length |
| C_F_has_valid_edge | 0 (1/1父) | 1 (1/1父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (1/1父) | 21 (1/1父) | count |
| nonmanifold_edges | 0 (1/1父) | 0 (1/1父) | count |
| C_F_nonmanifold_edge_count | 0 (1/1父) | 0 (1/1父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (1/1父) | 0.414347 (1/1父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (1/1父) | 0.468162 (1/1父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (1/1父) | 0.117491 (1/1父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/1父) | 0 (1/1父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/1父) | 8.74839 (1/1父) | model_length_squared |

## HIGH_TRAIN / RAW

| metric | B narrow Native | S SEAM1500 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.10801 (2/2父) | 0.0442291 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.685105 (2/2父) | 0.75234 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 36.5 (2/2父) | 15.75 (2/2父) | count |
| FF_illegal_pair_count | 196.25 (2/2父) | 230 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.534537 (2/2父) | 0.523396 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.472204 (2/2父) | 0.466856 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 0 (2/2父) | 0 (2/2父) | count |
| actual_free_faces | 319 (2/2父) | 319 (2/2父) | count |
| free_total_surface_area | 23.5768 (2/2父) | 25.9958 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 319 (2/2父) | 275 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 319 (2/2父) | 275 (2/2父) | count |
| C_F_valid_edge_count | 0 (2/2父) | 44 (2/2父) | count |
| C_F_valid_edge_total_length | 0 (2/2父) | 144.484 (2/2父) | model_length |
| C_F_has_valid_edge | 0 (2/2父) | 1 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (2/2父) | 44 (2/2父) | count |
| nonmanifold_edges | 0 (2/2父) | 0 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (2/2父) | 0.149266 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (2/2父) | 0.850734 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | 0 (2/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | 22.3439 (2/2父) | model_length_squared |

## HIGH_TRAIN / BASE_POST

| metric | B narrow Native | S SEAM1500 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0 (2/2父) | 0.00696811 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.0158378 (2/2父) | 0.0632372 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 0 (2/2父) | 1.5 (2/2父) | count |
| FF_illegal_pair_count | 1 (2/2父) | 3.75 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.314143 (2/2父) | 0.477917 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.314143 (2/2父) | 0.468655 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 268 (2/2父) | 249.5 (2/2父) | count |
| actual_free_faces | 51 (2/2父) | 69.5 (2/2父) | count |
| free_total_surface_area | 8.38694 (2/2父) | 10.9352 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 32.75 (2/2父) | 24 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 32.75 (2/2父) | 23.25 (2/2父) | count |
| C_F_valid_edge_count | 1 (2/2父) | 26.25 (2/2父) | count |
| C_F_valid_edge_total_length | 5.31632 (2/2父) | 103.919 (2/2父) | model_length |
| C_F_has_valid_edge | 0.5 (2/2父) | 1 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 1 (2/2父) | 26.25 (2/2父) | count |
| nonmanifold_edges | 0 (2/2父) | 0 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.0253524 (2/2父) | 0.289663 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.950562 (2/2父) | 0.689175 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.0240854 (2/2父) | 0.0211624 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | 0.00562082 (2/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | 8.06929 (2/2父) | model_length_squared |

## HIGH_TRAIN / CLEAN_FINAL_POST

| metric | B narrow Native | S SEAM1500 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0 (2/2父) | 0.00696811 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.0158378 (2/2父) | 0.0399519 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 0 (2/2父) | 1.5 (2/2父) | count |
| FF_illegal_pair_count | 1 (2/2父) | 2.25 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.314143 (2/2父) | 0.477917 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.314143 (2/2父) | 0.468655 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 268 (2/2父) | 249.5 (2/2父) | count |
| actual_free_faces | 51 (2/2父) | 69.5 (2/2父) | count |
| free_total_surface_area | 8.4122 (2/2父) | 11.1085 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 30.5 (2/2父) | 23.5 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 30.5 (2/2父) | 22.75 (2/2父) | count |
| C_F_valid_edge_count | 1 (2/2父) | 26.25 (2/2父) | count |
| C_F_valid_edge_total_length | 5.31632 (2/2父) | 103.919 (2/2父) | model_length |
| C_F_has_valid_edge | 0.5 (2/2父) | 1 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 1 (2/2父) | 26.25 (2/2父) | count |
| nonmanifold_edges | 0 (2/2父) | 0 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.025211 (2/2父) | 0.302204 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.950747 (2/2父) | 0.680824 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.0240424 (2/2父) | 0.0169729 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | 0.0192347 (2/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | 8.17421 (2/2父) | model_length_squared |

## HIGH_HOLDOUT / RAW

| metric | B narrow Native | S SEAM1500 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.109216 (2/2父) | 0.114948 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.943372 (2/2父) | 0.959529 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | 0.0866699 (2/2父) | 0.239868 (2/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 108.25 (2/2父) | 116.25 (2/2父) | count |
| FF_illegal_pair_count | 976.25 (2/2父) | 1083.25 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.427594 (2/2父) | 0.401012 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.393377 (2/2父) | 0.367748 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 0 (2/2父) | 0 (2/2父) | count |
| actual_free_faces | 522.5 (2/2父) | 522.5 (2/2父) | count |
| free_total_surface_area | 51.6978 (2/2父) | 55.1689 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 522.5 (2/2父) | 521.5 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 522.5 (2/2父) | 521.5 (2/2父) | count |
| C_F_valid_edge_count | 0 (2/2父) | 1 (2/2父) | count |
| C_F_valid_edge_total_length | 0 (2/2父) | 5.31632 (2/2父) | model_length |
| C_F_has_valid_edge | 0 (2/2父) | 0.5 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (2/2父) | 1 (2/2父) | count |
| nonmanifold_edges | 0 (2/2父) | 0 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (2/2父) | 0.0615954 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (2/2父) | 0.938405 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | 0 (1/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | 19.4151 (1/2父) | model_length_squared |

## HIGH_HOLDOUT / BASE_POST

| metric | B narrow Native | S SEAM1500 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.207484 (2/2父) | 0.207889 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.162757 (2/2父) | 0.225965 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | 0.0449219 (2/2父) | 0.140015 (2/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 31.25 (2/2父) | 30 (2/2父) | count |
| FF_illegal_pair_count | 5 (2/2父) | 10 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.454865 (2/2父) | 0.477869 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.338406 (2/2父) | 0.377615 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 473.5 (2/2父) | 469.75 (2/2父) | count |
| actual_free_faces | 49 (2/2父) | 52.75 (2/2父) | count |
| free_total_surface_area | 29.9687 (2/2父) | 34.0395 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 19.5 (2/2父) | 22.25 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 16.25 (2/2父) | 17.75 (2/2父) | count |
| C_F_valid_edge_count | 0 (2/2父) | 1 (2/2父) | count |
| C_F_valid_edge_total_length | 0 (2/2父) | 5.31632 (2/2父) | model_length |
| C_F_has_valid_edge | 0 (2/2父) | 0.5 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (2/2父) | 1 (2/2父) | count |
| nonmanifold_edges | 2.5 (2/2父) | 4.5 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0.25 (2/2父) | 0.25 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (2/2父) | 0.0874036 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.811128 (2/2父) | 0.291066 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.188872 (2/2父) | 0.621531 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | 0 (1/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | 11.8808 (1/2父) | model_length_squared |

## HIGH_HOLDOUT / CLEAN_FINAL_POST

| metric | B narrow Native | S SEAM1500 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.207484 (2/2父) | 0.207889 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.124807 (2/2父) | 0.20124 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | 0.0455322 (2/2父) | 0.140137 (2/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 31 (2/2父) | 30 (2/2父) | count |
| FF_illegal_pair_count | 3.75 (2/2父) | 9 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.434773 (2/2父) | 0.461565 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.311557 (2/2父) | 0.36131 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 473.5 (2/2父) | 469.75 (2/2父) | count |
| actual_free_faces | 49 (2/2父) | 52.75 (2/2父) | count |
| free_total_surface_area | 30.085 (2/2父) | 34.3082 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 13.25 (2/2父) | 16.25 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 10.75 (2/2父) | 12.25 (2/2父) | count |
| C_F_valid_edge_count | 0 (2/2父) | 1 (2/2父) | count |
| C_F_valid_edge_total_length | 0 (2/2父) | 5.31632 (2/2父) | model_length |
| C_F_has_valid_edge | 0 (2/2父) | 0.5 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (2/2父) | 1 (2/2父) | count |
| nonmanifold_edges | 2.5 (2/2父) | 4.5 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0.25 (2/2父) | 0.25 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (2/2父) | 0.0876283 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.811829 (2/2父) | 0.292169 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.188171 (2/2父) | 0.620202 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | 0 (1/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | 12.0283 (1/2父) | model_length_squared |

## NEW_WIDE_TRAIN / RAW

| metric | B narrow Native | S SEAM1500 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0853239 (4/4父) | 0.07523 (4/4父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.926407 (4/4父) | 0.943016 (4/4父) | fraction_0_to_1 |
| ConflictCov_C | 0.168213 (2/4父) | 0.151245 (2/4父) | fraction_0_to_1 |
| CF_illegal_pair_count | 90.5 (4/4父) | 87 (4/4父) | count |
| FF_illegal_pair_count | 1257 (4/4父) | 1303.75 (4/4父) | count |
| free_minimum_angle_lt5_fraction | 0.472585 (4/4父) | 0.465928 (4/4父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.363383 (4/4父) | 0.360655 (4/4父) | fraction_0_to_1 |
| deleted_free_faces | 0 (4/4父) | 0 (4/4父) | count |
| actual_free_faces | 553 (4/4父) | 553 (4/4父) | count |
| free_total_surface_area | 54.2272 (4/4父) | 59.2772 (4/4父) | model_length_squared |
| new_bridge_faces | 0 (4/4父) | 0 (4/4父) | count |
| bridge_area | 0 (4/4父) | 0 (4/4父) | model_length_squared |
| deleted_original_POST_faces | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_isolated_1face_edge_components | 553 (4/4父) | 545.75 (4/4父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 553 (4/4父) | 545.75 (4/4父) | count |
| C_F_valid_edge_count | 0 (4/4父) | 7.25 (4/4父) | count |
| C_F_valid_edge_total_length | 0 (4/4父) | 36.7074 (4/4父) | model_length |
| C_F_has_valid_edge | 0 (4/4父) | 0.75 (4/4父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (4/4父) | 7.25 (4/4父) | count |
| nonmanifold_edges | 0 (4/4父) | 0 (4/4父) | count |
| C_F_nonmanifold_edge_count | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (4/4父) | 0.101974 (4/4父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (4/4父) | 0.898026 (4/4父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/4父) | 0 (3/4父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/4父) | 35.5453 (3/4父) | model_length_squared |

## NEW_WIDE_TRAIN / BASE_POST

| metric | B narrow Native | S SEAM1500 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0662325 (4/4父) | 0.062073 (4/4父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.363772 (4/4父) | 0.404785 (4/4父) | fraction_0_to_1 |
| ConflictCov_C | 0.0137939 (2/4父) | 0.0216064 (2/4父) | fraction_0_to_1 |
| CF_illegal_pair_count | 12.125 (4/4父) | 18 (4/4父) | count |
| FF_illegal_pair_count | 68 (4/4父) | 108.875 (4/4父) | count |
| free_minimum_angle_lt5_fraction | 0.538157 (4/4父) | 0.569993 (4/4父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.454722 (4/4父) | 0.488984 (4/4父) | fraction_0_to_1 |
| deleted_free_faces | 446 (4/4父) | 440 (4/4父) | count |
| actual_free_faces | 107 (4/4父) | 113 (4/4父) | count |
| free_total_surface_area | 28.0413 (4/4父) | 32.5221 (4/4父) | model_length_squared |
| new_bridge_faces | 0 (4/4父) | 0 (4/4父) | count |
| bridge_area | 0 (4/4父) | 0 (4/4父) | model_length_squared |
| deleted_original_POST_faces | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_isolated_1face_edge_components | 56.125 (4/4父) | 60.875 (4/4父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 48.75 (4/4父) | 49.25 (4/4父) | count |
| C_F_valid_edge_count | 1.25 (4/4父) | 6.5 (4/4父) | count |
| C_F_valid_edge_total_length | 5.30164 (4/4父) | 32.1633 (4/4父) | model_length |
| C_F_has_valid_edge | 0.5 (4/4父) | 0.75 (4/4父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 1.25 (4/4父) | 6.5 (4/4父) | count |
| nonmanifold_edges | 4.25 (4/4父) | 5.375 (4/4父) | count |
| C_F_nonmanifold_edge_count | 0.125 (4/4父) | 0.625 (4/4父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.0435804 (4/4父) | 0.159715 (4/4父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.697812 (4/4父) | 0.524778 (4/4父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.258607 (4/4父) | 0.315508 (4/4父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/4父) | 0.0137476 (3/4父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/4父) | 20.8554 (3/4父) | model_length_squared |

## NEW_WIDE_TRAIN / CLEAN_FINAL_POST

| metric | B narrow Native | S SEAM1500 | unit |
| --- | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0662325 (4/4父) | 0.062073 (4/4父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.348225 (4/4父) | 0.390059 (4/4父) | fraction_0_to_1 |
| ConflictCov_C | 0.0135498 (2/4父) | 0.0219727 (2/4父) | fraction_0_to_1 |
| CF_illegal_pair_count | 12.125 (4/4父) | 18 (4/4父) | count |
| FF_illegal_pair_count | 63.75 (4/4父) | 104.125 (4/4父) | count |
| free_minimum_angle_lt5_fraction | 0.531599 (4/4父) | 0.562951 (4/4父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.44663 (4/4父) | 0.481115 (4/4父) | fraction_0_to_1 |
| deleted_free_faces | 446 (4/4父) | 440 (4/4父) | count |
| actual_free_faces | 107 (4/4父) | 113 (4/4父) | count |
| free_total_surface_area | 28.3116 (4/4父) | 32.8039 (4/4父) | model_length_squared |
| new_bridge_faces | 0 (4/4父) | 0 (4/4父) | count |
| bridge_area | 0 (4/4父) | 0 (4/4父) | model_length_squared |
| deleted_original_POST_faces | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_isolated_1face_edge_components | 44.5 (4/4父) | 48.375 (4/4父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 38.125 (4/4父) | 38.125 (4/4父) | count |
| C_F_valid_edge_count | 1.25 (4/4父) | 6.5 (4/4父) | count |
| C_F_valid_edge_total_length | 5.30164 (4/4父) | 32.1633 (4/4父) | model_length |
| C_F_has_valid_edge | 0.5 (4/4父) | 0.75 (4/4父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 1.25 (4/4父) | 6.5 (4/4父) | count |
| nonmanifold_edges | 4.25 (4/4父) | 5.375 (4/4父) | count |
| C_F_nonmanifold_edge_count | 0.125 (4/4父) | 0.625 (4/4父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.0787174 (4/4父) | 0.158733 (4/4父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.697861 (4/4父) | 0.524651 (4/4父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.223422 (4/4父) | 0.316616 (4/4父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/4父) | 0.0137176 (3/4父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/4父) | 21.0571 (3/4父) | model_length_squared |
