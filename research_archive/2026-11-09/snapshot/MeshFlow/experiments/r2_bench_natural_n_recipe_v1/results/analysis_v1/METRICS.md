# Bench natural-N recipe: complete numerical results

P0 is frozen exposure B: old30 outputs are reused; high-N outputs are newly generated. A/B each add1500 updates. Contrasts: A-P0, B-P0 and B-A.
所有分数沿用原定义，父等权；I主展示只用PASS支持域，完整177字段保留原值与NA。
Fractions use0-1, not percent. OLD/FOCUS/DEV/HOLDOUT/HIGH_TRAIN/HIGH_HOLDOUT remain separate; no automatic promotion.
本文件不产生视觉标签。实际图审及支撑完整性必须另见 visual_review。

## OLD / RAW

| metric | P0 | A | B | unit |
| --- | ---: | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0930587 (8/8父) | 0.110044 (8/8父) | 0.0922111 (8/8父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.829363 (8/8父) | 0.845745 (8/8父) | 0.850119 (8/8父) | fraction_0_to_1 |
| ConflictCov_C | 0.0335693 (8/8父) | 0.0342407 (8/8父) | 0.0376282 (8/8父) | fraction_0_to_1 |
| CF_illegal_pair_count | 31.5625 (8/8父) | 37.125 (8/8父) | 33.125 (8/8父) | count |
| FF_illegal_pair_count | 204.688 (8/8父) | 211.562 (8/8父) | 205.438 (8/8父) | count |
| free_minimum_angle_lt5_fraction | 0.398598 (8/8父) | 0.390888 (8/8父) | 0.389759 (8/8父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.328359 (8/8父) | 0.315409 (8/8父) | 0.316694 (8/8父) | fraction_0_to_1 |
| deleted_free_faces | 0 (8/8父) | 0 (8/8父) | 0 (8/8父) | count |
| actual_free_faces | 196.625 (8/8父) | 196.625 (8/8父) | 196.625 (8/8父) | count |
| free_total_surface_area | 34.4695 (8/8父) | 31.6579 (8/8父) | 31.3943 (8/8父) | model_length_squared |
| new_bridge_faces | 0 (8/8父) | 0 (8/8父) | 0 (8/8父) | count |
| bridge_area | 0 (8/8父) | 0 (8/8父) | 0 (8/8父) | model_length_squared |
| deleted_original_POST_faces | 0 (8/8父) | 0 (8/8父) | 0 (8/8父) | count |
| hinge_diag_isolated_1face_edge_components | 196.625 (8/8父) | 196.625 (8/8父) | 196.625 (8/8父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 196.625 (8/8父) | 196.625 (8/8父) | 196.625 (8/8父) | count |
| C_F_valid_edge_count | 0 (8/8父) | 0 (8/8父) | 0 (8/8父) | count |
| C_F_valid_edge_total_length | 0 (8/8父) | 0 (8/8父) | 0 (8/8父) | model_length |
| C_F_has_valid_edge | 0 (8/8父) | 0 (8/8父) | 0 (8/8父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (8/8父) | 0 (8/8父) | 0 (8/8父) | count |
| nonmanifold_edges | 0 (8/8父) | 0 (8/8父) | 0 (8/8父) | count |
| C_F_nonmanifold_edge_count | 0 (8/8父) | 0 (8/8父) | 0 (8/8父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (8/8父) | 0 (8/8父) | 0 (8/8父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (8/8父) | 1 (8/8父) | 1 (8/8父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (8/8父) | 0 (8/8父) | 0 (8/8父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/8父) | NA (0/8父) | NA (0/8父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/8父) | NA (0/8父) | NA (0/8父) | model_length_squared |

## OLD / BASE_POST

| metric | P0 | A | B | unit |
| --- | ---: | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0579988 (8/8父) | 0.0708015 (8/8父) | 0.0653743 (8/8父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.198035 (8/8父) | 0.138945 (8/8父) | 0.120557 (8/8父) | fraction_0_to_1 |
| ConflictCov_C | 0.0138245 (8/8父) | 0.0166016 (8/8父) | 0.0150146 (8/8父) | fraction_0_to_1 |
| CF_illegal_pair_count | 5.6875 (8/8父) | 6.5625 (8/8父) | 6.3125 (8/8父) | count |
| FF_illegal_pair_count | 14.5 (8/8父) | 9.0625 (8/8父) | 8.375 (8/8父) | count |
| free_minimum_angle_lt5_fraction | 0.350783 (8/8父) | 0.351826 (8/8父) | 0.35971 (8/8父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.298863 (8/8父) | 0.290668 (8/8父) | 0.286391 (8/8父) | fraction_0_to_1 |
| deleted_free_faces | 114.75 (8/8父) | 111.438 (8/8父) | 110.312 (8/8父) | count |
| actual_free_faces | 81.875 (8/8父) | 85.1875 (8/8父) | 86.3125 (8/8父) | count |
| free_total_surface_area | 28.1494 (8/8父) | 24.9582 (8/8父) | 24.6144 (8/8父) | model_length_squared |
| new_bridge_faces | 0 (8/8父) | 0 (8/8父) | 0 (8/8父) | count |
| bridge_area | 0 (8/8父) | 0 (8/8父) | 0 (8/8父) | model_length_squared |
| deleted_original_POST_faces | 0 (8/8父) | 0 (8/8父) | 0 (8/8父) | count |
| hinge_diag_isolated_1face_edge_components | 40.3125 (8/8父) | 40.4375 (8/8父) | 41.0625 (8/8父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 39.625 (8/8父) | 39.375 (8/8父) | 40.1875 (8/8父) | count |
| C_F_valid_edge_count | 0.8125 (8/8父) | 1.375 (8/8父) | 1 (8/8父) | count |
| C_F_valid_edge_total_length | 1.31461 (8/8父) | 2.27608 (8/8父) | 2.06804 (8/8父) | model_length |
| C_F_has_valid_edge | 0.4375 (8/8父) | 0.5625 (8/8父) | 0.5625 (8/8父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0.8125 (8/8父) | 1.375 (8/8父) | 1 (8/8父) | count |
| nonmanifold_edges | 0.3125 (8/8父) | 0.625 (8/8父) | 0.4375 (8/8父) | count |
| C_F_nonmanifold_edge_count | 0 (8/8父) | 0.0625 (8/8父) | 0.0625 (8/8父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.034626 (8/8父) | 0.0909027 (8/8父) | 0.0599955 (8/8父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.549954 (8/8父) | 0.544033 (8/8父) | 0.607836 (8/8父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.41542 (8/8父) | 0.365064 (8/8父) | 0.332168 (8/8父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/8父) | NA (0/8父) | NA (0/8父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/8父) | NA (0/8父) | NA (0/8父) | model_length_squared |

## OLD / CLEAN_FINAL_POST

| metric | P0 | A | B | unit |
| --- | ---: | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0550467 (8/8父) | 0.0636614 (8/8父) | 0.06276 (8/8父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.150574 (8/8父) | 0.103848 (8/8父) | 0.091824 (8/8父) | fraction_0_to_1 |
| ConflictCov_C | 0.0134888 (8/8父) | 0.0163574 (8/8父) | 0.0148621 (8/8父) | fraction_0_to_1 |
| CF_illegal_pair_count | 5 (8/8父) | 6.0625 (8/8父) | 6.0625 (8/8父) | count |
| FF_illegal_pair_count | 11.0625 (8/8父) | 6.625 (8/8父) | 6.3125 (8/8父) | count |
| free_minimum_angle_lt5_fraction | 0.336776 (8/8父) | 0.335363 (8/8父) | 0.344514 (8/8父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.288581 (8/8父) | 0.282336 (8/8父) | 0.27314 (8/8父) | fraction_0_to_1 |
| deleted_free_faces | 114.75 (8/8父) | 111.438 (8/8父) | 110.312 (8/8父) | count |
| actual_free_faces | 81.875 (8/8父) | 85.1875 (8/8父) | 86.3125 (8/8父) | count |
| free_total_surface_area | 28.3136 (8/8父) | 25.1987 (8/8父) | 24.8545 (8/8父) | model_length_squared |
| new_bridge_faces | 0 (8/8父) | 0 (8/8父) | 0 (8/8父) | count |
| bridge_area | 0 (8/8父) | 0 (8/8父) | 0 (8/8父) | model_length_squared |
| deleted_original_POST_faces | 0 (8/8父) | 0 (8/8父) | 0 (8/8父) | count |
| hinge_diag_isolated_1face_edge_components | 24.375 (8/8父) | 24.0625 (8/8父) | 25.6875 (8/8父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 23.75 (8/8父) | 23.25 (8/8父) | 24.875 (8/8父) | count |
| C_F_valid_edge_count | 0.9375 (8/8父) | 1.5 (8/8父) | 1 (8/8父) | count |
| C_F_valid_edge_total_length | 1.52647 (8/8父) | 2.33739 (8/8父) | 2.06804 (8/8父) | model_length |
| C_F_has_valid_edge | 0.4375 (8/8父) | 0.5625 (8/8父) | 0.5625 (8/8父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0.9375 (8/8父) | 1.5 (8/8父) | 1 (8/8父) | count |
| nonmanifold_edges | 0.3125 (8/8父) | 0.625 (8/8父) | 0.4375 (8/8父) | count |
| C_F_nonmanifold_edge_count | 0 (8/8父) | 0.0625 (8/8父) | 0.0625 (8/8父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.12755 (8/8父) | 0.110851 (8/8父) | 0.0902084 (8/8父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.551254 (8/8父) | 0.544458 (8/8父) | 0.607965 (8/8父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.321196 (8/8父) | 0.344691 (8/8父) | 0.301827 (8/8父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/8父) | NA (0/8父) | NA (0/8父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/8父) | NA (0/8父) | NA (0/8父) | model_length_squared |

## FOCUS / RAW

| metric | P0 | A | B | unit |
| --- | ---: | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.175295 (2/2父) | 0.144263 (2/2父) | 0.122537 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.831711 (2/2父) | 0.879912 (2/2父) | 0.857227 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | 0.120483 (2/2父) | 0.122925 (2/2父) | 0.0714111 (2/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 56.75 (2/2父) | 49.25 (2/2父) | 40.25 (2/2父) | count |
| FF_illegal_pair_count | 183.25 (2/2父) | 204.5 (2/2父) | 190 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.501047 (2/2父) | 0.504381 (2/2父) | 0.499971 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.39469 (2/2父) | 0.411888 (2/2父) | 0.402463 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| actual_free_faces | 188 (2/2父) | 188 (2/2父) | 188 (2/2父) | count |
| free_total_surface_area | 15.559 (2/2父) | 16.2744 (2/2父) | 15.7702 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 188 (2/2父) | 188 (2/2父) | 188 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 188 (2/2父) | 188 (2/2父) | 188 (2/2父) | count |
| C_F_valid_edge_count | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| C_F_valid_edge_total_length | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | model_length |
| C_F_has_valid_edge | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| nonmanifold_edges | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (2/2父) | 1 (2/2父) | 1 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | NA (0/2父) | NA (0/2父) | model_length_squared |

## FOCUS / BASE_POST

| metric | P0 | A | B | unit |
| --- | ---: | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0113636 (2/2父) | 0.01 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.0285714 (2/2父) | 0.02 (2/2父) | 0.034897 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 0.25 (2/2父) | 0.25 (2/2父) | 0 (2/2父) | count |
| FF_illegal_pair_count | 0.5 (2/2父) | 0.25 (2/2父) | 0.5 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.689587 (2/2父) | 0.688527 (2/2父) | 0.659809 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.624315 (2/2父) | 0.645942 (2/2父) | 0.624138 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 155.5 (2/2父) | 156.75 (2/2父) | 156.5 (2/2父) | count |
| actual_free_faces | 32.5 (2/2父) | 31.25 (2/2父) | 31.5 (2/2父) | count |
| free_total_surface_area | 3.52929 (2/2父) | 3.43422 (2/2父) | 3.45842 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 19.75 (2/2父) | 22.25 (2/2父) | 19.75 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 19 (2/2父) | 22.25 (2/2父) | 19.25 (2/2父) | count |
| C_F_valid_edge_count | 0.75 (2/2父) | 0.5 (2/2父) | 1 (2/2父) | count |
| C_F_valid_edge_total_length | 2.34906 (2/2父) | 2.32975 (2/2父) | 3.62273 (2/2父) | model_length |
| C_F_has_valid_edge | 0.5 (2/2父) | 0.5 (2/2父) | 0.75 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0.75 (2/2父) | 0.5 (2/2父) | 1 (2/2父) | count |
| nonmanifold_edges | 0.25 (2/2父) | 0 (2/2父) | 0.25 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.396211 (2/2父) | 0.375313 (2/2父) | 0.396729 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.590191 (2/2父) | 0.447714 (2/2父) | 0.602164 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.0135976 (2/2父) | 0.176973 (2/2父) | 0.00110741 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | NA (0/2父) | NA (0/2父) | model_length_squared |

## FOCUS / CLEAN_FINAL_POST

| metric | P0 | A | B | unit |
| --- | ---: | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0113636 (2/2父) | 0.01 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.0285714 (2/2父) | 0.02 (2/2父) | 0.0217391 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 0.25 (2/2父) | 0.25 (2/2父) | 0 (2/2父) | count |
| FF_illegal_pair_count | 0.5 (2/2父) | 0.25 (2/2父) | 0.25 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.689587 (2/2父) | 0.688527 (2/2父) | 0.65323 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.616962 (2/2父) | 0.638367 (2/2父) | 0.617559 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 155.5 (2/2父) | 156.75 (2/2父) | 156.5 (2/2父) | count |
| actual_free_faces | 32.5 (2/2父) | 31.25 (2/2父) | 31.5 (2/2父) | count |
| free_total_surface_area | 3.56663 (2/2父) | 3.50866 (2/2父) | 3.59336 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 15.25 (2/2父) | 17 (2/2父) | 17.25 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 14.5 (2/2父) | 17 (2/2父) | 16.75 (2/2父) | count |
| C_F_valid_edge_count | 0.75 (2/2父) | 0.5 (2/2父) | 1 (2/2父) | count |
| C_F_valid_edge_total_length | 2.34906 (2/2父) | 2.32975 (2/2父) | 3.62273 (2/2父) | model_length |
| C_F_has_valid_edge | 0.5 (2/2父) | 0.5 (2/2父) | 0.75 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0.75 (2/2父) | 0.5 (2/2父) | 1 (2/2父) | count |
| nonmanifold_edges | 0.25 (2/2父) | 0 (2/2父) | 0.25 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.392842 (2/2父) | 0.385352 (2/2父) | 0.395545 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.593871 (2/2父) | 0.448484 (2/2父) | 0.603347 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.0132865 (2/2父) | 0.166164 (2/2父) | 0.00110728 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | NA (0/2父) | NA (0/2父) | model_length_squared |

## DEV / RAW

| metric | P0 | A | B | unit |
| --- | ---: | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0480981 (4/4父) | 0.0605702 (4/4父) | 0.0595824 (4/4父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.834257 (4/4父) | 0.902436 (4/4父) | 0.874498 (4/4父) | fraction_0_to_1 |
| ConflictCov_C | 0.119995 (4/4父) | 0.119446 (4/4父) | 0.125 (4/4父) | fraction_0_to_1 |
| CF_illegal_pair_count | 10.125 (4/4父) | 12.875 (4/4父) | 13 (4/4父) | count |
| FF_illegal_pair_count | 204.875 (4/4父) | 251.625 (4/4父) | 233.25 (4/4父) | count |
| free_minimum_angle_lt5_fraction | 0.457509 (4/4父) | 0.470117 (4/4父) | 0.444582 (4/4父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.362409 (4/4父) | 0.345582 (4/4父) | 0.309994 (4/4父) | fraction_0_to_1 |
| deleted_free_faces | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | count |
| actual_free_faces | 174 (4/4父) | 174 (4/4父) | 174 (4/4父) | count |
| free_total_surface_area | 31.724 (4/4父) | 36.9762 (4/4父) | 34.869 (4/4父) | model_length_squared |
| new_bridge_faces | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | count |
| bridge_area | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | model_length_squared |
| deleted_original_POST_faces | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_isolated_1face_edge_components | 174 (4/4父) | 174 (4/4父) | 174 (4/4父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 174 (4/4父) | 174 (4/4父) | 174 (4/4父) | count |
| C_F_valid_edge_count | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | count |
| C_F_valid_edge_total_length | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | model_length |
| C_F_has_valid_edge | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | count |
| nonmanifold_edges | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | count |
| C_F_nonmanifold_edge_count | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (4/4父) | 1 (4/4父) | 1 (4/4父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/4父) | NA (0/4父) | NA (0/4父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/4父) | NA (0/4父) | NA (0/4父) | model_length_squared |

## DEV / BASE_POST

| metric | P0 | A | B | unit |
| --- | ---: | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0306744 (4/4父) | 0.0600909 (4/4父) | 0.0617901 (4/4父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.151432 (4/4父) | 0.247033 (4/4父) | 0.164067 (4/4父) | fraction_0_to_1 |
| ConflictCov_C | 0.010437 (4/4父) | 0.10907 (4/4父) | 0.115479 (4/4父) | fraction_0_to_1 |
| CF_illegal_pair_count | 2.5 (4/4父) | 8.875 (4/4父) | 9 (4/4父) | count |
| FF_illegal_pair_count | 11 (4/4父) | 18.875 (4/4父) | 12.5 (4/4父) | count |
| free_minimum_angle_lt5_fraction | 0.657237 (4/4父) | 0.654886 (4/4父) | 0.658842 (4/4父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.573252 (4/4父) | 0.559729 (4/4父) | 0.540728 (4/4父) | fraction_0_to_1 |
| deleted_free_faces | 115.875 (4/4父) | 108.125 (4/4父) | 106.125 (4/4父) | count |
| actual_free_faces | 58.125 (4/4父) | 65.875 (4/4父) | 67.875 (4/4父) | count |
| free_total_surface_area | 17.9482 (4/4父) | 21.8877 (4/4父) | 20.8231 (4/4父) | model_length_squared |
| new_bridge_faces | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | count |
| bridge_area | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | model_length_squared |
| deleted_original_POST_faces | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_isolated_1face_edge_components | 19.625 (4/4父) | 26.625 (4/4父) | 25.25 (4/4父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 18.5 (4/4父) | 22.75 (4/4父) | 22.625 (4/4父) | count |
| C_F_valid_edge_count | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | count |
| C_F_valid_edge_total_length | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | model_length |
| C_F_has_valid_edge | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | count |
| nonmanifold_edges | 0.875 (4/4父) | 2.125 (4/4父) | 2 (4/4父) | count |
| C_F_nonmanifold_edge_count | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (4/4父) | 1 (4/4父) | 1 (4/4父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/4父) | NA (0/4父) | NA (0/4父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/4父) | NA (0/4父) | NA (0/4父) | model_length_squared |

## DEV / CLEAN_FINAL_POST

| metric | P0 | A | B | unit |
| --- | ---: | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0293858 (4/4父) | 0.0587173 (4/4父) | 0.0603856 (4/4父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.117834 (4/4父) | 0.211369 (4/4父) | 0.11494 (4/4父) | fraction_0_to_1 |
| ConflictCov_C | 0.010437 (4/4父) | 0.10376 (4/4父) | 0.117188 (4/4父) | fraction_0_to_1 |
| CF_illegal_pair_count | 2.375 (4/4父) | 8.375 (4/4父) | 8.625 (4/4父) | count |
| FF_illegal_pair_count | 9.75 (4/4父) | 17.125 (4/4父) | 10.25 (4/4父) | count |
| free_minimum_angle_lt5_fraction | 0.646914 (4/4父) | 0.638443 (4/4父) | 0.64658 (4/4父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.565553 (4/4父) | 0.529402 (4/4父) | 0.528287 (4/4父) | fraction_0_to_1 |
| deleted_free_faces | 115.875 (4/4父) | 108.125 (4/4父) | 106.125 (4/4父) | count |
| actual_free_faces | 58.125 (4/4父) | 65.875 (4/4父) | 67.875 (4/4父) | count |
| free_total_surface_area | 18.1038 (4/4父) | 22.2155 (4/4父) | 21.0498 (4/4父) | model_length_squared |
| new_bridge_faces | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | count |
| bridge_area | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | model_length_squared |
| deleted_original_POST_faces | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_isolated_1face_edge_components | 13.5 (4/4父) | 19.375 (4/4父) | 18 (4/4父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 12.375 (4/4父) | 16 (4/4父) | 15.625 (4/4父) | count |
| C_F_valid_edge_count | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | count |
| C_F_valid_edge_total_length | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | model_length |
| C_F_has_valid_edge | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | count |
| nonmanifold_edges | 0.875 (4/4父) | 2.125 (4/4父) | 2 (4/4父) | count |
| C_F_nonmanifold_edge_count | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (4/4父) | 1 (4/4父) | 1 (4/4父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/4父) | NA (0/4父) | NA (0/4父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/4父) | NA (0/4父) | NA (0/4父) | model_length_squared |

## HOLDOUT / RAW

| metric | P0 | A | B | unit |
| --- | ---: | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.258065 (1/1父) | 0.25576 (1/1父) | 0.25576 (1/1父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.910138 (1/1父) | 0.912442 (1/1父) | 0.857143 (1/1父) | fraction_0_to_1 |
| ConflictCov_C | 0.0893555 (1/1父) | 0.0949707 (1/1父) | 0.0983887 (1/1父) | fraction_0_to_1 |
| CF_illegal_pair_count | 82.5 (1/1父) | 84 (1/1父) | 85 (1/1父) | count |
| FF_illegal_pair_count | 279 (1/1父) | 272.5 (1/1父) | 259 (1/1父) | count |
| free_minimum_angle_lt5_fraction | 0.179724 (1/1父) | 0.179724 (1/1父) | 0.175115 (1/1父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.0967742 (1/1父) | 0.108295 (1/1父) | 0.110599 (1/1父) | fraction_0_to_1 |
| deleted_free_faces | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | count |
| actual_free_faces | 217 (1/1父) | 217 (1/1父) | 217 (1/1父) | count |
| free_total_surface_area | 20.8604 (1/1父) | 21.7003 (1/1父) | 21.8687 (1/1父) | model_length_squared |
| new_bridge_faces | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | count |
| bridge_area | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | model_length_squared |
| deleted_original_POST_faces | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | count |
| hinge_diag_isolated_1face_edge_components | 217 (1/1父) | 217 (1/1父) | 217 (1/1父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 217 (1/1父) | 217 (1/1父) | 217 (1/1父) | count |
| C_F_valid_edge_count | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | count |
| C_F_valid_edge_total_length | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | model_length |
| C_F_has_valid_edge | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | count |
| nonmanifold_edges | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | count |
| C_F_nonmanifold_edge_count | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (1/1父) | 1 (1/1父) | 1 (1/1父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/1父) | NA (0/1父) | NA (0/1父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/1父) | NA (0/1父) | NA (0/1父) | model_length_squared |

## HOLDOUT / BASE_POST

| metric | P0 | A | B | unit |
| --- | ---: | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.187013 (1/1父) | 0.19317 (1/1父) | 0.201094 (1/1父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.208009 (1/1父) | 0.123465 (1/1父) | 0.110313 (1/1父) | fraction_0_to_1 |
| ConflictCov_C | 0.0930176 (1/1父) | 0.119141 (1/1父) | 0.101318 (1/1父) | fraction_0_to_1 |
| CF_illegal_pair_count | 31 (1/1父) | 32 (1/1父) | 32 (1/1父) | count |
| FF_illegal_pair_count | 15 (1/1父) | 9 (1/1父) | 8.5 (1/1父) | count |
| free_minimum_angle_lt5_fraction | 0.157397 (1/1父) | 0.184837 (1/1父) | 0.178031 (1/1父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.128337 (1/1父) | 0.152726 (1/1父) | 0.118875 (1/1父) | fraction_0_to_1 |
| deleted_free_faces | 97 (1/1父) | 90.5 (1/1父) | 90.5 (1/1父) | count |
| actual_free_faces | 120 (1/1父) | 126.5 (1/1父) | 126.5 (1/1父) | count |
| free_total_surface_area | 9.40243 (1/1父) | 10.4248 (1/1父) | 10.3283 (1/1父) | model_length_squared |
| new_bridge_faces | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | count |
| bridge_area | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | model_length_squared |
| deleted_original_POST_faces | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | count |
| hinge_diag_isolated_1face_edge_components | 93.5 (1/1父) | 104.5 (1/1父) | 99.5 (1/1父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 93.5 (1/1父) | 104.5 (1/1父) | 99.5 (1/1父) | count |
| C_F_valid_edge_count | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | count |
| C_F_valid_edge_total_length | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | model_length |
| C_F_has_valid_edge | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | count |
| nonmanifold_edges | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | count |
| C_F_nonmanifold_edge_count | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (1/1父) | 1 (1/1父) | 1 (1/1父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/1父) | NA (0/1父) | NA (0/1父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/1父) | NA (0/1父) | NA (0/1父) | model_length_squared |

## HOLDOUT / CLEAN_FINAL_POST

| metric | P0 | A | B | unit |
| --- | ---: | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.187013 (1/1父) | 0.189004 (1/1父) | 0.197188 (1/1父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.166852 (1/1父) | 0.100094 (1/1父) | 0.0907813 (1/1父) | fraction_0_to_1 |
| ConflictCov_C | 0.0930176 (1/1父) | 0.114014 (1/1父) | 0.101074 (1/1父) | fraction_0_to_1 |
| CF_illegal_pair_count | 31 (1/1父) | 31 (1/1父) | 31 (1/1父) | count |
| FF_illegal_pair_count | 11 (1/1父) | 7 (1/1父) | 6.5 (1/1父) | count |
| free_minimum_angle_lt5_fraction | 0.157397 (1/1父) | 0.184837 (1/1父) | 0.170219 (1/1父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.128337 (1/1父) | 0.148966 (1/1父) | 0.114969 (1/1父) | fraction_0_to_1 |
| deleted_free_faces | 97 (1/1父) | 90.5 (1/1父) | 90.5 (1/1父) | count |
| actual_free_faces | 120 (1/1父) | 126.5 (1/1父) | 126.5 (1/1父) | count |
| free_total_surface_area | 9.4731 (1/1父) | 10.4676 (1/1父) | 10.3399 (1/1父) | model_length_squared |
| new_bridge_faces | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | count |
| bridge_area | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | model_length_squared |
| deleted_original_POST_faces | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | count |
| hinge_diag_isolated_1face_edge_components | 71 (1/1父) | 70.5 (1/1父) | 66.5 (1/1父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 71 (1/1父) | 70.5 (1/1父) | 66.5 (1/1父) | count |
| C_F_valid_edge_count | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | count |
| C_F_valid_edge_total_length | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | model_length |
| C_F_has_valid_edge | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | count |
| nonmanifold_edges | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | count |
| C_F_nonmanifold_edge_count | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (1/1父) | 1 (1/1父) | 1 (1/1父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/1父) | NA (0/1父) | NA (0/1父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/1父) | NA (0/1父) | NA (0/1父) | model_length_squared |

## HIGH_TRAIN / RAW

| metric | P0 | A | B | unit |
| --- | ---: | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0551471 (2/2父) | 0.0240085 (2/2父) | 0.10801 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.925691 (2/2父) | 0.920956 (2/2父) | 0.685105 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | NA (0/2父) | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 39.5 (2/2父) | 25.25 (2/2父) | 36.5 (2/2父) | count |
| FF_illegal_pair_count | 518.25 (2/2父) | 554.75 (2/2父) | 196.25 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.555147 (2/2父) | 0.522894 (2/2父) | 0.534537 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.4624 (2/2父) | 0.44424 (2/2父) | 0.472204 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| actual_free_faces | 319 (2/2父) | 319 (2/2父) | 319 (2/2父) | count |
| free_total_surface_area | 53.4667 (2/2父) | 46.0195 (2/2父) | 23.5768 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 319 (2/2父) | 319 (2/2父) | 319 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 319 (2/2父) | 319 (2/2父) | 319 (2/2父) | count |
| C_F_valid_edge_count | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| C_F_valid_edge_total_length | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | model_length |
| C_F_has_valid_edge | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| nonmanifold_edges | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (2/2父) | 1 (2/2父) | 1 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | NA (0/2父) | NA (0/2父) | model_length_squared |

## HIGH_TRAIN / BASE_POST

| metric | P0 | A | B | unit |
| --- | ---: | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0866847 (2/2父) | 0.056145 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.142592 (2/2父) | 0.170467 (2/2父) | 0.0158378 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | NA (0/2父) | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 18.5 (2/2父) | 11.5 (2/2父) | 0 (2/2父) | count |
| FF_illegal_pair_count | 8.25 (2/2父) | 13.5 (2/2父) | 1 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.514165 (2/2父) | 0.542884 (2/2父) | 0.314143 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.479524 (2/2父) | 0.488982 (2/2父) | 0.314143 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 261.25 (2/2父) | 266.5 (2/2父) | 268 (2/2父) | count |
| actual_free_faces | 57.75 (2/2父) | 52.5 (2/2父) | 51 (2/2父) | count |
| free_total_surface_area | 27.7571 (2/2父) | 22.1131 (2/2父) | 8.38694 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 26.5 (2/2父) | 25.75 (2/2父) | 32.75 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 23.5 (2/2父) | 20.25 (2/2父) | 32.75 (2/2父) | count |
| C_F_valid_edge_count | 0 (2/2父) | 0 (2/2父) | 1 (2/2父) | count |
| C_F_valid_edge_total_length | 0 (2/2父) | 0 (2/2父) | 5.31632 (2/2父) | model_length |
| C_F_has_valid_edge | 0 (2/2父) | 0 (2/2父) | 0.5 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (2/2父) | 0 (2/2父) | 1 (2/2父) | count |
| nonmanifold_edges | 1.5 (2/2父) | 3 (2/2父) | 0 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (2/2父) | 0 (2/2父) | 0.0253524 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.635685 (2/2父) | 0.813027 (2/2父) | 0.950562 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.364315 (2/2父) | 0.186973 (2/2父) | 0.0240854 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | NA (0/2父) | NA (0/2父) | model_length_squared |

## HIGH_TRAIN / CLEAN_FINAL_POST

| metric | P0 | A | B | unit |
| --- | ---: | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0866847 (2/2父) | 0.056145 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.137642 (2/2父) | 0.160263 (2/2父) | 0.0158378 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | NA (0/2父) | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 18.5 (2/2父) | 11.5 (2/2父) | 0 (2/2父) | count |
| FF_illegal_pair_count | 7.5 (2/2父) | 13.25 (2/2父) | 1 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.485093 (2/2父) | 0.524813 (2/2父) | 0.314143 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.455169 (2/2父) | 0.480716 (2/2父) | 0.314143 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 261.25 (2/2父) | 266.5 (2/2父) | 268 (2/2父) | count |
| actual_free_faces | 57.75 (2/2父) | 52.5 (2/2父) | 51 (2/2父) | count |
| free_total_surface_area | 28.1543 (2/2父) | 22.3313 (2/2父) | 8.4122 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 19.25 (2/2父) | 18.75 (2/2父) | 30.5 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 16.25 (2/2父) | 14.25 (2/2父) | 30.5 (2/2父) | count |
| C_F_valid_edge_count | 0 (2/2父) | 0 (2/2父) | 1 (2/2父) | count |
| C_F_valid_edge_total_length | 0 (2/2父) | 0 (2/2父) | 5.31632 (2/2父) | model_length |
| C_F_has_valid_edge | 0 (2/2父) | 0 (2/2父) | 0.5 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (2/2父) | 0 (2/2父) | 1 (2/2父) | count |
| nonmanifold_edges | 1.5 (2/2父) | 3 (2/2父) | 0 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (2/2父) | 0 (2/2父) | 0.025211 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.635112 (2/2父) | 0.813856 (2/2父) | 0.950747 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.364888 (2/2父) | 0.186144 (2/2父) | 0.0240424 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | NA (0/2父) | NA (0/2父) | model_length_squared |

## HIGH_HOLDOUT / RAW

| metric | P0 | A | B | unit |
| --- | ---: | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.119677 (2/2父) | 0.11326 (2/2父) | 0.109216 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.941217 (2/2父) | 0.950257 (2/2父) | 0.943372 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | 0.103638 (2/2父) | 0.131104 (2/2父) | 0.0866699 (2/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 115.75 (2/2父) | 119 (2/2父) | 108.25 (2/2父) | count |
| FF_illegal_pair_count | 955.5 (2/2父) | 986 (2/2父) | 976.25 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.445452 (2/2父) | 0.443015 (2/2父) | 0.427594 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.405172 (2/2父) | 0.400796 (2/2父) | 0.393377 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| actual_free_faces | 522.5 (2/2父) | 522.5 (2/2父) | 522.5 (2/2父) | count |
| free_total_surface_area | 56.6675 (2/2父) | 57.1641 (2/2父) | 51.6978 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 522.5 (2/2父) | 522.5 (2/2父) | 522.5 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 522.5 (2/2父) | 522.5 (2/2父) | 522.5 (2/2父) | count |
| C_F_valid_edge_count | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| C_F_valid_edge_total_length | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | model_length |
| C_F_has_valid_edge | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| nonmanifold_edges | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 1 (2/2父) | 1 (2/2父) | 1 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | NA (0/2父) | NA (0/2父) | model_length_squared |

## HIGH_HOLDOUT / BASE_POST

| metric | P0 | A | B | unit |
| --- | ---: | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.238154 (2/2父) | 0.205177 (2/2父) | 0.207484 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.20958 (2/2父) | 0.223494 (2/2父) | 0.162757 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | 0.0565186 (2/2父) | 0.0479736 (2/2父) | 0.0449219 (2/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 38.25 (2/2父) | 38 (2/2父) | 31.25 (2/2父) | count |
| FF_illegal_pair_count | 8.25 (2/2父) | 10 (2/2父) | 5 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.499219 (2/2父) | 0.511876 (2/2父) | 0.454865 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.397027 (2/2父) | 0.426351 (2/2父) | 0.338406 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 470.25 (2/2父) | 465.75 (2/2父) | 473.5 (2/2父) | count |
| actual_free_faces | 52.25 (2/2父) | 56.75 (2/2父) | 49 (2/2父) | count |
| free_total_surface_area | 36.2384 (2/2父) | 33.925 (2/2父) | 29.9687 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 20.75 (2/2父) | 22.5 (2/2父) | 19.5 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 16 (2/2父) | 17 (2/2父) | 16.25 (2/2父) | count |
| C_F_valid_edge_count | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| C_F_valid_edge_total_length | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | model_length |
| C_F_has_valid_edge | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| nonmanifold_edges | 4 (2/2父) | 3.75 (2/2父) | 2.5 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0.25 (2/2父) | 0.25 (2/2父) | 0.25 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.758794 (2/2父) | 0.760537 (2/2父) | 0.811128 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.241206 (2/2父) | 0.239463 (2/2父) | 0.188872 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | NA (0/2父) | NA (0/2父) | model_length_squared |

## HIGH_HOLDOUT / CLEAN_FINAL_POST

| metric | P0 | A | B | unit |
| --- | ---: | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.238154 (2/2父) | 0.205177 (2/2父) | 0.207484 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.202227 (2/2父) | 0.211809 (2/2父) | 0.124807 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | 0.0565186 (2/2父) | 0.0479736 (2/2父) | 0.0455322 (2/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 38.25 (2/2父) | 38 (2/2父) | 31 (2/2父) | count |
| FF_illegal_pair_count | 8 (2/2父) | 9.5 (2/2父) | 3.75 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.493267 (2/2父) | 0.505626 (2/2父) | 0.434773 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.378254 (2/2父) | 0.414666 (2/2父) | 0.311557 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 470.25 (2/2父) | 465.75 (2/2父) | 473.5 (2/2父) | count |
| actual_free_faces | 52.25 (2/2父) | 56.75 (2/2父) | 49 (2/2父) | count |
| free_total_surface_area | 36.4417 (2/2父) | 34.0281 (2/2父) | 30.085 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 14.25 (2/2父) | 18.5 (2/2父) | 13.25 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 10.5 (2/2父) | 13.25 (2/2父) | 10.75 (2/2父) | count |
| C_F_valid_edge_count | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| C_F_valid_edge_total_length | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | model_length |
| C_F_has_valid_edge | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| nonmanifold_edges | 4 (2/2父) | 3.75 (2/2父) | 2.5 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0.25 (2/2父) | 0.25 (2/2父) | 0.25 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.758793 (2/2父) | 0.760548 (2/2父) | 0.811829 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.241207 (2/2父) | 0.239452 (2/2父) | 0.188171 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | NA (0/2父) | NA (0/2父) | model_length_squared |
