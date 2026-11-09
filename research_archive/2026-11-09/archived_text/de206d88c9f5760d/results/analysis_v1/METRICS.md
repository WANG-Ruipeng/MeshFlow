# Bench 曝光分配：完整数值结果

P0 为只读 Coverage 终点；A/B 各新增1500步。三个对比为 A−P0、B−P0、B−A。
所有分数沿用原定义，父等权；I主展示只用PASS支持域，完整177字段保留原值与NA。
fraction以0–1表示，不是百分数。OLD/FOCUS/DEV/HOLDOUT分开；不据此自动晋级。
本文件不产生视觉标签。实际图审及支撑完整性必须另见 visual_review。

## OLD / RAW

| metric | P0 | A | B | unit |
| --- | ---: | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0977699 (8/8父) | 0.0726876 (8/8父) | 0.0930587 (8/8父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.872977 (8/8父) | 0.842905 (8/8父) | 0.829363 (8/8父) | fraction_0_to_1 |
| ConflictCov_C | 0.0401611 (8/8父) | 0.0244141 (8/8父) | 0.0335693 (8/8父) | fraction_0_to_1 |
| CF_illegal_pair_count | 33.4375 (8/8父) | 24.625 (8/8父) | 31.5625 (8/8父) | count |
| FF_illegal_pair_count | 239.312 (8/8父) | 216.875 (8/8父) | 204.688 (8/8父) | count |
| free_minimum_angle_lt5_fraction | 0.365663 (8/8父) | 0.371276 (8/8父) | 0.398598 (8/8父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.297735 (8/8父) | 0.306799 (8/8父) | 0.328359 (8/8父) | fraction_0_to_1 |
| deleted_free_faces | 0 (8/8父) | 0 (8/8父) | 0 (8/8父) | count |
| actual_free_faces | 196.625 (8/8父) | 196.625 (8/8父) | 196.625 (8/8父) | count |
| free_total_surface_area | 37.9647 (8/8父) | 37.7339 (8/8父) | 34.4695 (8/8父) | model_length_squared |
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
| CF_illegal_affected_free_fraction | 0.0890836 (8/8父) | 0.0681453 (8/8父) | 0.0579988 (8/8父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.19754 (8/8父) | 0.181529 (8/8父) | 0.198035 (8/8父) | fraction_0_to_1 |
| ConflictCov_C | 0.0514526 (8/8父) | 0.00619507 (8/8父) | 0.0138245 (8/8父) | fraction_0_to_1 |
| CF_illegal_pair_count | 8.9375 (8/8父) | 6.625 (8/8父) | 5.6875 (8/8父) | count |
| FF_illegal_pair_count | 17.4375 (8/8父) | 18.25 (8/8父) | 14.5 (8/8父) | count |
| free_minimum_angle_lt5_fraction | 0.342638 (8/8父) | 0.376969 (8/8父) | 0.350783 (8/8父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.286377 (8/8父) | 0.322716 (8/8父) | 0.298863 (8/8父) | fraction_0_to_1 |
| deleted_free_faces | 109.125 (8/8父) | 106.688 (8/8父) | 114.75 (8/8父) | count |
| actual_free_faces | 87.5 (8/8父) | 89.9375 (8/8父) | 81.875 (8/8父) | count |
| free_total_surface_area | 31.2789 (8/8父) | 30.6909 (8/8父) | 28.1494 (8/8父) | model_length_squared |
| new_bridge_faces | 0 (8/8父) | 0 (8/8父) | 0 (8/8父) | count |
| bridge_area | 0 (8/8父) | 0 (8/8父) | 0 (8/8父) | model_length_squared |
| deleted_original_POST_faces | 0 (8/8父) | 0 (8/8父) | 0 (8/8父) | count |
| hinge_diag_isolated_1face_edge_components | 45.25 (8/8父) | 38.9375 (8/8父) | 40.3125 (8/8父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 44.25 (8/8父) | 38.75 (8/8父) | 39.625 (8/8父) | count |
| C_F_valid_edge_count | 0.9375 (8/8父) | 1.0625 (8/8父) | 0.8125 (8/8父) | count |
| C_F_valid_edge_total_length | 0.809522 (8/8父) | 1.35329 (8/8父) | 1.31461 (8/8父) | model_length |
| C_F_has_valid_edge | 0.5 (8/8父) | 0.5 (8/8父) | 0.4375 (8/8父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0.9375 (8/8父) | 1.0625 (8/8父) | 0.8125 (8/8父) | count |
| nonmanifold_edges | 0.375 (8/8父) | 0.0625 (8/8父) | 0.3125 (8/8父) | count |
| C_F_nonmanifold_edge_count | 0 (8/8父) | 0 (8/8父) | 0 (8/8父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.0235481 (8/8父) | 0.0889662 (8/8父) | 0.034626 (8/8父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.617773 (8/8父) | 0.552391 (8/8父) | 0.549954 (8/8父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.358679 (8/8父) | 0.358643 (8/8父) | 0.41542 (8/8父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/8父) | NA (0/8父) | NA (0/8父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/8父) | NA (0/8父) | NA (0/8父) | model_length_squared |

## OLD / CLEAN_FINAL_POST

| metric | P0 | A | B | unit |
| --- | ---: | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0869609 (8/8父) | 0.0664289 (8/8父) | 0.0550467 (8/8父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.158006 (8/8父) | 0.141579 (8/8父) | 0.150574 (8/8父) | fraction_0_to_1 |
| ConflictCov_C | 0.0506897 (8/8父) | 0.00610352 (8/8父) | 0.0134888 (8/8父) | fraction_0_to_1 |
| CF_illegal_pair_count | 8.125 (8/8父) | 5.9375 (8/8父) | 5 (8/8父) | count |
| FF_illegal_pair_count | 13.9375 (8/8父) | 14.3125 (8/8父) | 11.0625 (8/8父) | count |
| free_minimum_angle_lt5_fraction | 0.332746 (8/8父) | 0.359779 (8/8父) | 0.336776 (8/8父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.274445 (8/8父) | 0.307601 (8/8父) | 0.288581 (8/8父) | fraction_0_to_1 |
| deleted_free_faces | 109.125 (8/8父) | 106.688 (8/8父) | 114.75 (8/8父) | count |
| actual_free_faces | 87.5 (8/8父) | 89.9375 (8/8父) | 81.875 (8/8父) | count |
| free_total_surface_area | 31.4695 (8/8父) | 30.8516 (8/8父) | 28.3136 (8/8父) | model_length_squared |
| new_bridge_faces | 0 (8/8父) | 0 (8/8父) | 0 (8/8父) | count |
| bridge_area | 0 (8/8父) | 0 (8/8父) | 0 (8/8父) | model_length_squared |
| deleted_original_POST_faces | 0 (8/8父) | 0 (8/8父) | 0 (8/8父) | count |
| hinge_diag_isolated_1face_edge_components | 29.6875 (8/8父) | 23.625 (8/8父) | 24.375 (8/8父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 28.875 (8/8父) | 23.5 (8/8父) | 23.75 (8/8父) | count |
| C_F_valid_edge_count | 1 (8/8父) | 1.25 (8/8父) | 0.9375 (8/8父) | count |
| C_F_valid_edge_total_length | 0.831152 (8/8父) | 1.41236 (8/8父) | 1.52647 (8/8父) | model_length |
| C_F_has_valid_edge | 0.5 (8/8父) | 0.5 (8/8父) | 0.4375 (8/8父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 1 (8/8父) | 1.25 (8/8父) | 0.9375 (8/8父) | count |
| nonmanifold_edges | 0.375 (8/8父) | 0.0625 (8/8父) | 0.3125 (8/8父) | count |
| C_F_nonmanifold_edge_count | 0 (8/8父) | 0 (8/8父) | 0 (8/8父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.0709316 (8/8父) | 0.15795 (8/8父) | 0.12755 (8/8父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.618364 (8/8父) | 0.553055 (8/8父) | 0.551254 (8/8父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.310704 (8/8父) | 0.288994 (8/8父) | 0.321196 (8/8父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/8父) | NA (0/8父) | NA (0/8父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/8父) | NA (0/8父) | NA (0/8父) | model_length_squared |

## FOCUS / RAW

| metric | P0 | A | B | unit |
| --- | ---: | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.150369 (2/2父) | 0.174764 (2/2父) | 0.175295 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.861106 (2/2父) | 0.873245 (2/2父) | 0.831711 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | 0.10022 (2/2父) | 0.111694 (2/2父) | 0.120483 (2/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 48.25 (2/2父) | 55.25 (2/2父) | 56.75 (2/2父) | count |
| FF_illegal_pair_count | 206.25 (2/2父) | 206.5 (2/2父) | 183.25 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.498319 (2/2父) | 0.487758 (2/2父) | 0.501047 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.403584 (2/2父) | 0.398024 (2/2父) | 0.39469 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| actual_free_faces | 188 (2/2父) | 188 (2/2父) | 188 (2/2父) | count |
| free_total_surface_area | 16.381 (2/2父) | 15.9821 (2/2父) | 15.559 (2/2父) | model_length_squared |
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
| CF_illegal_affected_free_fraction | 0 (2/2父) | 0 (2/2父) | 0.0113636 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.0178571 (2/2父) | 0.0125 (2/2父) | 0.0285714 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 0 (2/2父) | 0 (2/2父) | 0.25 (2/2父) | count |
| FF_illegal_pair_count | 0.25 (2/2父) | 0.25 (2/2父) | 0.5 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.711418 (2/2父) | 0.701809 (2/2父) | 0.689587 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.687784 (2/2父) | 0.655947 (2/2父) | 0.624315 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 157.25 (2/2父) | 154.75 (2/2父) | 155.5 (2/2父) | count |
| actual_free_faces | 30.75 (2/2父) | 33.25 (2/2父) | 32.5 (2/2父) | count |
| free_total_surface_area | 3.66537 (2/2父) | 3.68079 (2/2父) | 3.52929 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 19.5 (2/2父) | 20.75 (2/2父) | 19.75 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 19.5 (2/2父) | 20.75 (2/2父) | 19 (2/2父) | count |
| C_F_valid_edge_count | 0.75 (2/2父) | 0.5 (2/2父) | 0.75 (2/2父) | count |
| C_F_valid_edge_total_length | 3.60344 (2/2父) | 2.32975 (2/2父) | 2.34906 (2/2父) | model_length |
| C_F_has_valid_edge | 0.75 (2/2父) | 0.5 (2/2父) | 0.5 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0.75 (2/2父) | 0.5 (2/2父) | 0.75 (2/2父) | count |
| nonmanifold_edges | 0 (2/2父) | 0 (2/2父) | 0.25 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.377104 (2/2父) | 0.38021 (2/2父) | 0.396211 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.387433 (2/2父) | 0.42333 (2/2父) | 0.590191 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.235463 (2/2父) | 0.19646 (2/2父) | 0.0135976 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | NA (0/2父) | NA (0/2父) | model_length_squared |

## FOCUS / CLEAN_FINAL_POST

| metric | P0 | A | B | unit |
| --- | ---: | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0 (2/2父) | 0 (2/2父) | 0.0113636 (2/2父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.0178571 (2/2父) | 0 (2/2父) | 0.0285714 (2/2父) | fraction_0_to_1 |
| ConflictCov_C | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | fraction_0_to_1 |
| CF_illegal_pair_count | 0 (2/2父) | 0 (2/2父) | 0.25 (2/2父) | count |
| FF_illegal_pair_count | 0.25 (2/2父) | 0 (2/2父) | 0.5 (2/2父) | count |
| free_minimum_angle_lt5_fraction | 0.711418 (2/2父) | 0.701809 (2/2父) | 0.689587 (2/2父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.673078 (2/2父) | 0.649537 (2/2父) | 0.616962 (2/2父) | fraction_0_to_1 |
| deleted_free_faces | 157.25 (2/2父) | 154.75 (2/2父) | 155.5 (2/2父) | count |
| actual_free_faces | 30.75 (2/2父) | 33.25 (2/2父) | 32.5 (2/2父) | count |
| free_total_surface_area | 3.7954 (2/2父) | 3.77666 (2/2父) | 3.56663 (2/2父) | model_length_squared |
| new_bridge_faces | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| bridge_area | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | model_length_squared |
| deleted_original_POST_faces | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_isolated_1face_edge_components | 13.75 (2/2父) | 14.25 (2/2父) | 15.25 (2/2父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 13.75 (2/2父) | 14.25 (2/2父) | 14.5 (2/2父) | count |
| C_F_valid_edge_count | 0.75 (2/2父) | 0.5 (2/2父) | 0.75 (2/2父) | count |
| C_F_valid_edge_total_length | 3.60344 (2/2父) | 2.32975 (2/2父) | 2.34906 (2/2父) | model_length |
| C_F_has_valid_edge | 0.75 (2/2父) | 0.5 (2/2父) | 0.5 (2/2父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0.75 (2/2父) | 0.5 (2/2父) | 0.75 (2/2父) | count |
| nonmanifold_edges | 0 (2/2父) | 0 (2/2父) | 0.25 (2/2父) | count |
| C_F_nonmanifold_edge_count | 0 (2/2父) | 0 (2/2父) | 0 (2/2父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0.379016 (2/2父) | 0.37387 (2/2父) | 0.392842 (2/2父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.373147 (2/2父) | 0.437106 (2/2父) | 0.593871 (2/2父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.247837 (2/2父) | 0.189024 (2/2父) | 0.0132865 (2/2父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/2父) | NA (0/2父) | NA (0/2父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/2父) | NA (0/2父) | NA (0/2父) | model_length_squared |

## DEV / RAW

| metric | P0 | A | B | unit |
| --- | ---: | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0583163 (4/4父) | 0.0476033 (4/4父) | 0.0480981 (4/4父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.889799 (4/4父) | 0.883254 (4/4父) | 0.834257 (4/4父) | fraction_0_to_1 |
| ConflictCov_C | 0.121399 (4/4父) | 0.121094 (4/4父) | 0.119995 (4/4父) | fraction_0_to_1 |
| CF_illegal_pair_count | 12.75 (4/4父) | 9.875 (4/4父) | 10.125 (4/4父) | count |
| FF_illegal_pair_count | 261.125 (4/4父) | 258.625 (4/4父) | 204.875 (4/4父) | count |
| free_minimum_angle_lt5_fraction | 0.461395 (4/4父) | 0.432332 (4/4父) | 0.457509 (4/4父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.333622 (4/4父) | 0.337704 (4/4父) | 0.362409 (4/4父) | fraction_0_to_1 |
| deleted_free_faces | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | count |
| actual_free_faces | 174 (4/4父) | 174 (4/4父) | 174 (4/4父) | count |
| free_total_surface_area | 36.1386 (4/4父) | 31.6968 (4/4父) | 31.724 (4/4父) | model_length_squared |
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
| CF_illegal_affected_free_fraction | 0.0647565 (4/4父) | 0.0273009 (4/4父) | 0.0306744 (4/4父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.252706 (4/4父) | 0.236533 (4/4父) | 0.151432 (4/4父) | fraction_0_to_1 |
| ConflictCov_C | 0.115112 (4/4父) | 0.0130005 (4/4父) | 0.010437 (4/4父) | fraction_0_to_1 |
| CF_illegal_pair_count | 9.75 (4/4父) | 3 (4/4父) | 2.5 (4/4父) | count |
| FF_illegal_pair_count | 22.875 (4/4父) | 17.75 (4/4父) | 11 (4/4父) | count |
| free_minimum_angle_lt5_fraction | 0.66929 (4/4父) | 0.667101 (4/4父) | 0.657237 (4/4父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.554389 (4/4父) | 0.551942 (4/4父) | 0.573252 (4/4父) | fraction_0_to_1 |
| deleted_free_faces | 104 (4/4父) | 119.75 (4/4父) | 115.875 (4/4父) | count |
| actual_free_faces | 70 (4/4父) | 54.25 (4/4父) | 58.125 (4/4父) | count |
| free_total_surface_area | 21.9699 (4/4父) | 16.5685 (4/4父) | 17.9482 (4/4父) | model_length_squared |
| new_bridge_faces | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | count |
| bridge_area | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | model_length_squared |
| deleted_original_POST_faces | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_isolated_1face_edge_components | 26.75 (4/4父) | 20.125 (4/4父) | 19.625 (4/4父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 23 (4/4父) | 18 (4/4父) | 18.5 (4/4父) | count |
| C_F_valid_edge_count | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | count |
| C_F_valid_edge_total_length | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | model_length |
| C_F_has_valid_edge | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | count |
| nonmanifold_edges | 2.25 (4/4父) | 1.875 (4/4父) | 0.875 (4/4父) | count |
| C_F_nonmanifold_edge_count | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.99336 (4/4父) | 1 (4/4父) | 1 (4/4父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.00663957 (4/4父) | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/4父) | NA (0/4父) | NA (0/4父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/4父) | NA (0/4父) | NA (0/4父) | model_length_squared |

## DEV / CLEAN_FINAL_POST

| metric | P0 | A | B | unit |
| --- | ---: | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.0647565 (4/4父) | 0.0273009 (4/4父) | 0.0293858 (4/4父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.22259 (4/4父) | 0.216833 (4/4父) | 0.117834 (4/4父) | fraction_0_to_1 |
| ConflictCov_C | 0.114197 (4/4父) | 0.0130615 (4/4父) | 0.010437 (4/4父) | fraction_0_to_1 |
| CF_illegal_pair_count | 9.75 (4/4父) | 3 (4/4父) | 2.375 (4/4父) | count |
| FF_illegal_pair_count | 20.625 (4/4父) | 16.5 (4/4父) | 9.75 (4/4父) | count |
| free_minimum_angle_lt5_fraction | 0.653217 (4/4父) | 0.659595 (4/4父) | 0.646914 (4/4父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.540216 (4/4父) | 0.54528 (4/4父) | 0.565553 (4/4父) | fraction_0_to_1 |
| deleted_free_faces | 104 (4/4父) | 119.75 (4/4父) | 115.875 (4/4父) | count |
| actual_free_faces | 70 (4/4父) | 54.25 (4/4父) | 58.125 (4/4父) | count |
| free_total_surface_area | 22.1959 (4/4父) | 16.7602 (4/4父) | 18.1038 (4/4父) | model_length_squared |
| new_bridge_faces | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | count |
| bridge_area | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | model_length_squared |
| deleted_original_POST_faces | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_isolated_1face_edge_components | 19.5 (4/4父) | 14 (4/4父) | 13.5 (4/4父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 15.875 (4/4父) | 12.25 (4/4父) | 12.375 (4/4父) | count |
| C_F_valid_edge_count | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | count |
| C_F_valid_edge_total_length | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | model_length |
| C_F_has_valid_edge | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| C_F_valid_opposite_winding_edge_count | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | count |
| nonmanifold_edges | 2.25 (4/4父) | 1.875 (4/4父) | 0.875 (4/4父) | count |
| C_F_nonmanifold_edge_count | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | count |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | 0 (4/4父) | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| hinge_diag_fully_C_disconnected_free_area_fraction | 0.99336 (4/4父) | 1 (4/4父) | 1 (4/4父) | fraction_0_to_1 |
| hinge_diag_vertex_attached_only_free_area_fraction | 0.00663957 (4/4父) | 0 (4/4父) | 0 (4/4父) | fraction_0_to_1 |
| free_only_reached_bulk_area_fraction | NA (0/4父) | NA (0/4父) | NA (0/4父) | fraction_0_to_1 |
| retained_bulk_area | NA (0/4父) | NA (0/4父) | NA (0/4父) | model_length_squared |

## HOLDOUT / RAW

| metric | P0 | A | B | unit |
| --- | ---: | ---: | ---: | --- |
| CF_illegal_affected_free_fraction | 0.258065 (1/1父) | 0.260369 (1/1父) | 0.258065 (1/1父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.923963 (1/1父) | 0.919355 (1/1父) | 0.910138 (1/1父) | fraction_0_to_1 |
| ConflictCov_C | 0.0900879 (1/1父) | 0.0861816 (1/1父) | 0.0893555 (1/1父) | fraction_0_to_1 |
| CF_illegal_pair_count | 81 (1/1父) | 80 (1/1父) | 82.5 (1/1父) | count |
| FF_illegal_pair_count | 282.5 (1/1父) | 275 (1/1父) | 279 (1/1父) | count |
| free_minimum_angle_lt5_fraction | 0.172811 (1/1父) | 0.177419 (1/1父) | 0.179724 (1/1父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.0990783 (1/1父) | 0.101382 (1/1父) | 0.0967742 (1/1父) | fraction_0_to_1 |
| deleted_free_faces | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | count |
| actual_free_faces | 217 (1/1父) | 217 (1/1父) | 217 (1/1父) | count |
| free_total_surface_area | 21.6152 (1/1父) | 20.9344 (1/1父) | 20.8604 (1/1父) | model_length_squared |
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
| CF_illegal_affected_free_fraction | 0.245708 (1/1父) | 0.192059 (1/1父) | 0.187013 (1/1父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.204449 (1/1父) | 0.17182 (1/1父) | 0.208009 (1/1父) | fraction_0_to_1 |
| ConflictCov_C | 0.113037 (1/1父) | 0.0737305 (1/1父) | 0.0930176 (1/1父) | fraction_0_to_1 |
| CF_illegal_pair_count | 38.5 (1/1父) | 28 (1/1父) | 31 (1/1父) | count |
| FF_illegal_pair_count | 14 (1/1父) | 13 (1/1父) | 15 (1/1父) | count |
| free_minimum_angle_lt5_fraction | 0.18067 (1/1父) | 0.189011 (1/1父) | 0.157397 (1/1父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.153133 (1/1父) | 0.151966 (1/1父) | 0.128337 (1/1父) | fraction_0_to_1 |
| deleted_free_faces | 90.5 (1/1父) | 92 (1/1父) | 97 (1/1父) | count |
| actual_free_faces | 126.5 (1/1父) | 125 (1/1父) | 120 (1/1父) | count |
| free_total_surface_area | 9.94527 (1/1父) | 9.58605 (1/1父) | 9.40243 (1/1父) | model_length_squared |
| new_bridge_faces | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | count |
| bridge_area | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | model_length_squared |
| deleted_original_POST_faces | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | count |
| hinge_diag_isolated_1face_edge_components | 102 (1/1父) | 100 (1/1父) | 93.5 (1/1父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 102 (1/1父) | 100 (1/1父) | 93.5 (1/1父) | count |
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
| CF_illegal_affected_free_fraction | 0.241949 (1/1父) | 0.188242 (1/1父) | 0.187013 (1/1父) | fraction_0_to_1 |
| FF_illegal_affected_free_fraction | 0.169392 (1/1父) | 0.156553 (1/1父) | 0.166852 (1/1父) | fraction_0_to_1 |
| ConflictCov_C | 0.113037 (1/1父) | 0.0715332 (1/1父) | 0.0930176 (1/1父) | fraction_0_to_1 |
| CF_illegal_pair_count | 36.5 (1/1父) | 27.5 (1/1父) | 31 (1/1父) | count |
| FF_illegal_pair_count | 11.5 (1/1父) | 11.5 (1/1父) | 11 (1/1父) | count |
| free_minimum_angle_lt5_fraction | 0.176911 (1/1父) | 0.18481 (1/1父) | 0.157397 (1/1父) | fraction_0_to_1 |
| free_q_gt10_fraction | 0.148966 (1/1父) | 0.135929 (1/1父) | 0.128337 (1/1父) | fraction_0_to_1 |
| deleted_free_faces | 90.5 (1/1父) | 92 (1/1父) | 97 (1/1父) | count |
| actual_free_faces | 126.5 (1/1父) | 125 (1/1父) | 120 (1/1父) | count |
| free_total_surface_area | 9.91153 (1/1父) | 9.58744 (1/1父) | 9.4731 (1/1父) | model_length_squared |
| new_bridge_faces | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | count |
| bridge_area | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | model_length_squared |
| deleted_original_POST_faces | 0 (1/1父) | 0 (1/1父) | 0 (1/1父) | count |
| hinge_diag_isolated_1face_edge_components | 75 (1/1父) | 66.5 (1/1父) | 71 (1/1父) | count |
| hinge_diag_isolated_1face_unfiltered_edge_components | 75 (1/1父) | 66.5 (1/1父) | 71 (1/1父) | count |
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
