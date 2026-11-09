# 核心几何量：父等权对照（I仅PASS）

原评分单位；fraction 为0–1，不是百分数。单元为父等权均值〔有效请求/有效父〕。
OLD_TRAIN 是原 CURRENT_TRAIN 的报告别名。SEAM 未在新增6请求评测，写 NA；Native 的 SEAM 专属量仍 NA。
保留所有支持分母与排除原因；不自动晋级。

## OLD_TRAIN / RAW

| 字段 | 单位 | Native6000 | Candidate1500 | 旧SEAM3000 |
|---|---|---:|---:|---:|
| CF_illegal_affected_free_fraction | fraction_0_to_1 | 0.094139426〔16/8〕 | 0.097769852〔16/8〕 | 0.044905506〔16/8〕 |
| FF_illegal_affected_free_fraction | fraction_0_to_1 | 0.88144288〔16/8〕 | 0.87297681〔16/8〕 | 0.85043553〔16/8〕 |
| ConflictCov_C | fraction_0_to_1 | 0.041046143〔16/8〕 | 0.040161133〔16/8〕 | 0.013458252〔16/8〕 |
| CF_illegal_pair_count | count | 28.8125〔16/8〕 | 33.4375〔16/8〕 | 11.9375〔16/8〕 |
| FF_illegal_pair_count | count | 250.5〔16/8〕 | 239.3125〔16/8〕 | 238.1875〔16/8〕 |
| free_minimum_angle_lt5_fraction | fraction_0_to_1 | 0.37227126〔16/8〕 | 0.36566338〔16/8〕 | 0.37841086〔16/8〕 |
| free_q_gt10_fraction | fraction_0_to_1 | 0.29595367〔16/8〕 | 0.2977353〔16/8〕 | 0.30803167〔16/8〕 |
| deleted_free_faces | count | 0〔16/8〕 | 0〔16/8〕 | 0〔16/8〕 |
| actual_free_faces | count | 196.625〔16/8〕 | 196.625〔16/8〕 | 196.625〔16/8〕 |
| free_total_surface_area | model_length_squared | 35.969731〔16/8〕 | 37.964698〔16/8〕 | 32.828503〔16/8〕 |
| new_bridge_faces | count | 0〔16/8〕 | 0〔16/8〕 | 0〔16/8〕 |
| bridge_area | model_length_squared | 0〔16/8〕 | 0〔16/8〕 | 0〔16/8〕 |
| deleted_original_POST_faces | count | 0〔16/8〕 | 0〔16/8〕 | 0〔16/8〕 |
| hinge_diag_isolated_1face_edge_components | count | 196.625〔16/8〕 | 196.625〔16/8〕 | 179.5〔16/8〕 |
| hinge_diag_isolated_1face_unfiltered_edge_components | count | 196.625〔16/8〕 | 196.625〔16/8〕 | 179.5〔16/8〕 |
| C_F_valid_edge_count | count | 0〔16/8〕 | 0〔16/8〕 | 17.125〔16/8〕 |
| C_F_valid_edge_total_length | model_length | 0〔16/8〕 | 0〔16/8〕 | 17.186526〔16/8〕 |
| C_F_has_valid_edge | fraction_0_to_1 | 0〔16/8〕 | 0〔16/8〕 | 1〔16/8〕 |
| C_F_valid_opposite_winding_edge_count | count | 0〔16/8〕 | 0〔16/8〕 | 17.125〔16/8〕 |
| nonmanifold_edges | count | 0〔16/8〕 | 0〔16/8〕 | 0〔16/8〕 |
| C_F_nonmanifold_edge_count | count | 0〔16/8〕 | 0〔16/8〕 | 0〔16/8〕 |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | fraction_0_to_1 | 0〔16/8〕 | 0〔16/8〕 | 0.11275995〔16/8〕 |
| hinge_diag_fully_C_disconnected_free_area_fraction | fraction_0_to_1 | 1〔16/8〕 | 1〔16/8〕 | 0.88724005〔16/8〕 |
| hinge_diag_vertex_attached_only_free_area_fraction | fraction_0_to_1 | 0〔16/8〕 | 0〔16/8〕 | 0〔16/8〕 |
| free_only_reached_bulk_area_fraction | fraction_0_to_1 | NA〔0/0〕 | NA〔0/0〕 | 0〔16/8〕 |
| retained_bulk_area | model_length_squared | NA〔0/0〕 | NA〔0/0〕 | 29.426252〔16/8〕 |

## OLD_TRAIN / BASE_POST

| 字段 | 单位 | Native6000 | Candidate1500 | 旧SEAM3000 |
|---|---|---:|---:|---:|
| CF_illegal_affected_free_fraction | fraction_0_to_1 | 0.096230441〔16/8〕 | 0.089083582〔16/8〕 | 0.024897799〔16/8〕 |
| FF_illegal_affected_free_fraction | fraction_0_to_1 | 0.17294859〔16/8〕 | 0.19754016〔16/8〕 | 0.13991892〔16/8〕 |
| ConflictCov_C | fraction_0_to_1 | 0.019042969〔16/8〕 | 0.051452637〔16/8〕 | 0.033813477〔16/8〕 |
| CF_illegal_pair_count | count | 8.875〔16/8〕 | 8.9375〔16/8〕 | 1.3125〔16/8〕 |
| FF_illegal_pair_count | count | 14.3125〔16/8〕 | 17.4375〔16/8〕 | 10.0625〔16/8〕 |
| free_minimum_angle_lt5_fraction | fraction_0_to_1 | 0.35071176〔16/8〕 | 0.34263838〔16/8〕 | 0.36322817〔16/8〕 |
| free_q_gt10_fraction | fraction_0_to_1 | 0.29259737〔16/8〕 | 0.28637702〔16/8〕 | 0.29780367〔16/8〕 |
| deleted_free_faces | count | 106.875〔16/8〕 | 109.125〔16/8〕 | 105.875〔16/8〕 |
| actual_free_faces | count | 89.75〔16/8〕 | 87.5〔16/8〕 | 90.75〔16/8〕 |
| free_total_surface_area | model_length_squared | 28.969228〔16/8〕 | 31.278892〔16/8〕 | 26.434182〔16/8〕 |
| new_bridge_faces | count | 0〔16/8〕 | 0〔16/8〕 | 0〔16/8〕 |
| bridge_area | model_length_squared | 0〔16/8〕 | 0〔16/8〕 | 0〔16/8〕 |
| deleted_original_POST_faces | count | 0〔16/8〕 | 0〔16/8〕 | 0〔16/8〕 |
| hinge_diag_isolated_1face_edge_components | count | 40.9375〔16/8〕 | 45.25〔16/8〕 | 35.625〔16/8〕 |
| hinge_diag_isolated_1face_unfiltered_edge_components | count | 40.1875〔16/8〕 | 44.25〔16/8〕 | 34.6875〔16/8〕 |
| C_F_valid_edge_count | count | 1.125〔16/8〕 | 0.9375〔16/8〕 | 16.875〔16/8〕 |
| C_F_valid_edge_total_length | model_length | 1.6119423〔16/8〕 | 0.80952185〔16/8〕 | 16.69937〔16/8〕 |
| C_F_has_valid_edge | fraction_0_to_1 | 0.5〔16/8〕 | 0.5〔16/8〕 | 1〔16/8〕 |
| C_F_valid_opposite_winding_edge_count | count | 1.125〔16/8〕 | 0.9375〔16/8〕 | 16.875〔16/8〕 |
| nonmanifold_edges | count | 0.4375〔16/8〕 | 0.375〔16/8〕 | 0.75〔16/8〕 |
| C_F_nonmanifold_edge_count | count | 0〔16/8〕 | 0〔16/8〕 | 0.3125〔16/8〕 |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | fraction_0_to_1 | 0.055787826〔16/8〕 | 0.023548113〔16/8〕 | 0.18339562〔16/8〕 |
| hinge_diag_fully_C_disconnected_free_area_fraction | fraction_0_to_1 | 0.59644136〔16/8〕 | 0.61777326〔16/8〕 | 0.47610669〔16/8〕 |
| hinge_diag_vertex_attached_only_free_area_fraction | fraction_0_to_1 | 0.34777081〔16/8〕 | 0.35867863〔16/8〕 | 0.34049769〔16/8〕 |
| free_only_reached_bulk_area_fraction | fraction_0_to_1 | NA〔0/0〕 | NA〔0/0〕 | 0.088465318〔16/8〕 |
| retained_bulk_area | model_length_squared | NA〔0/0〕 | NA〔0/0〕 | 23.481055〔16/8〕 |

## OLD_TRAIN / CLEAN_FINAL_POST

| 字段 | 单位 | Native6000 | Candidate1500 | 旧SEAM3000 |
|---|---|---:|---:|---:|
| CF_illegal_affected_free_fraction | fraction_0_to_1 | 0.093345825〔16/8〕 | 0.086960928〔16/8〕 | 0.020860277〔16/8〕 |
| FF_illegal_affected_free_fraction | fraction_0_to_1 | 0.14569332〔16/8〕 | 0.15800587〔16/8〕 | 0.10250319〔16/8〕 |
| ConflictCov_C | fraction_0_to_1 | 0.018676758〔16/8〕 | 0.050689697〔16/8〕 | 0.033660889〔16/8〕 |
| CF_illegal_pair_count | count | 8.6875〔16/8〕 | 8.125〔16/8〕 | 0.75〔16/8〕 |
| FF_illegal_pair_count | count | 11.875〔16/8〕 | 13.9375〔16/8〕 | 7.4375〔16/8〕 |
| free_minimum_angle_lt5_fraction | fraction_0_to_1 | 0.33972952〔16/8〕 | 0.33274647〔16/8〕 | 0.34743518〔16/8〕 |
| free_q_gt10_fraction | fraction_0_to_1 | 0.2785021〔16/8〕 | 0.27444471〔16/8〕 | 0.28283066〔16/8〕 |
| deleted_free_faces | count | 106.875〔16/8〕 | 109.125〔16/8〕 | 105.875〔16/8〕 |
| actual_free_faces | count | 89.75〔16/8〕 | 87.5〔16/8〕 | 90.875〔16/8〕 |
| free_total_surface_area | model_length_squared | 29.23315〔16/8〕 | 31.46954〔16/8〕 | 26.71058〔16/8〕 |
| new_bridge_faces | count | 0〔16/8〕 | 0〔16/8〕 | 0.125〔16/8〕 |
| bridge_area | model_length_squared | 0〔16/8〕 | 0〔16/8〕 | 0.0024392093〔16/8〕 |
| deleted_original_POST_faces | count | 0〔16/8〕 | 0〔16/8〕 | 0〔16/8〕 |
| hinge_diag_isolated_1face_edge_components | count | 25〔16/8〕 | 29.6875〔16/8〕 | 21.875〔16/8〕 |
| hinge_diag_isolated_1face_unfiltered_edge_components | count | 24.25〔16/8〕 | 28.875〔16/8〕 | 20.9375〔16/8〕 |
| C_F_valid_edge_count | count | 1.625〔16/8〕 | 1〔16/8〕 | 16.875〔16/8〕 |
| C_F_valid_edge_total_length | model_length | 1.8015172〔16/8〕 | 0.83115234〔16/8〕 | 16.69937〔16/8〕 |
| C_F_has_valid_edge | fraction_0_to_1 | 0.5625〔16/8〕 | 0.5〔16/8〕 | 1〔16/8〕 |
| C_F_valid_opposite_winding_edge_count | count | 1.625〔16/8〕 | 1〔16/8〕 | 16.875〔16/8〕 |
| nonmanifold_edges | count | 0.4375〔16/8〕 | 0.375〔16/8〕 | 0.75〔16/8〕 |
| C_F_nonmanifold_edge_count | count | 0〔16/8〕 | 0〔16/8〕 | 0.3125〔16/8〕 |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | fraction_0_to_1 | 0.087072058〔16/8〕 | 0.070931606〔16/8〕 | 0.24193577〔16/8〕 |
| hinge_diag_fully_C_disconnected_free_area_fraction | fraction_0_to_1 | 0.59821805〔16/8〕 | 0.61836436〔16/8〕 | 0.47411632〔16/8〕 |
| hinge_diag_vertex_attached_only_free_area_fraction | fraction_0_to_1 | 0.31470989〔16/8〕 | 0.31070404〔16/8〕 | 0.28394792〔16/8〕 |
| free_only_reached_bulk_area_fraction | fraction_0_to_1 | NA〔0/0〕 | NA〔0/0〕 | 0.17303879〔16/8〕 |
| retained_bulk_area | model_length_squared | NA〔0/0〕 | NA〔0/0〕 | 23.740152〔16/8〕 |

## NEW_TRAIN / RAW

| 字段 | 单位 | Native6000 | Candidate1500 | 旧SEAM3000 |
|---|---|---:|---:|---:|
| CF_illegal_affected_free_fraction | fraction_0_to_1 | 0.07159292〔4/2〕 | 0.15036873〔4/2〕 | NA〔0/0〕 |
| FF_illegal_affected_free_fraction | fraction_0_to_1 | 0.91284661〔4/2〕 | 0.86110619〔4/2〕 | NA〔0/0〕 |
| ConflictCov_C | fraction_0_to_1 | 0.068481445〔4/2〕 | 0.10021973〔4/2〕 | NA〔0/0〕 |
| CF_illegal_pair_count | count | 19.5〔4/2〕 | 48.25〔4/2〕 | NA〔0/0〕 |
| FF_illegal_pair_count | count | 430.25〔4/2〕 | 206.25〔4/2〕 | NA〔0/0〕 |
| free_minimum_angle_lt5_fraction | fraction_0_to_1 | 0.34218289〔4/2〕 | 0.49831858〔4/2〕 | NA〔0/0〕 |
| free_q_gt10_fraction | fraction_0_to_1 | 0.26125369〔4/2〕 | 0.40358407〔4/2〕 | NA〔0/0〕 |
| deleted_free_faces | count | 0〔4/2〕 | 0〔4/2〕 | NA〔0/0〕 |
| actual_free_faces | count | 188〔4/2〕 | 188〔4/2〕 | NA〔0/0〕 |
| free_total_surface_area | model_length_squared | 28.918792〔4/2〕 | 16.381033〔4/2〕 | NA〔0/0〕 |
| new_bridge_faces | count | 0〔4/2〕 | 0〔4/2〕 | NA〔0/0〕 |
| bridge_area | model_length_squared | 0〔4/2〕 | 0〔4/2〕 | NA〔0/0〕 |
| deleted_original_POST_faces | count | 0〔4/2〕 | 0〔4/2〕 | NA〔0/0〕 |
| hinge_diag_isolated_1face_edge_components | count | 188〔4/2〕 | 188〔4/2〕 | NA〔0/0〕 |
| hinge_diag_isolated_1face_unfiltered_edge_components | count | 188〔4/2〕 | 188〔4/2〕 | NA〔0/0〕 |
| C_F_valid_edge_count | count | 0〔4/2〕 | 0〔4/2〕 | NA〔0/0〕 |
| C_F_valid_edge_total_length | model_length | 0〔4/2〕 | 0〔4/2〕 | NA〔0/0〕 |
| C_F_has_valid_edge | fraction_0_to_1 | 0〔4/2〕 | 0〔4/2〕 | NA〔0/0〕 |
| C_F_valid_opposite_winding_edge_count | count | 0〔4/2〕 | 0〔4/2〕 | NA〔0/0〕 |
| nonmanifold_edges | count | 0〔4/2〕 | 0〔4/2〕 | NA〔0/0〕 |
| C_F_nonmanifold_edge_count | count | 0〔4/2〕 | 0〔4/2〕 | NA〔0/0〕 |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | fraction_0_to_1 | 0〔4/2〕 | 0〔4/2〕 | NA〔0/0〕 |
| hinge_diag_fully_C_disconnected_free_area_fraction | fraction_0_to_1 | 1〔4/2〕 | 1〔4/2〕 | NA〔0/0〕 |
| hinge_diag_vertex_attached_only_free_area_fraction | fraction_0_to_1 | 0〔4/2〕 | 0〔4/2〕 | NA〔0/0〕 |
| free_only_reached_bulk_area_fraction | fraction_0_to_1 | NA〔0/0〕 | NA〔0/0〕 | NA〔0/0〕 |
| retained_bulk_area | model_length_squared | NA〔0/0〕 | NA〔0/0〕 | NA〔0/0〕 |

## NEW_TRAIN / BASE_POST

| 字段 | 单位 | Native6000 | Candidate1500 | 旧SEAM3000 |
|---|---|---:|---:|---:|
| CF_illegal_affected_free_fraction | fraction_0_to_1 | 0.052958656〔4/2〕 | 0〔4/2〕 | NA〔0/0〕 |
| FF_illegal_affected_free_fraction | fraction_0_to_1 | 0.25423034〔4/2〕 | 0.017857143〔4/2〕 | NA〔0/0〕 |
| ConflictCov_C | fraction_0_to_1 | 0.031005859〔4/2〕 | 0〔4/2〕 | NA〔0/0〕 |
| CF_illegal_pair_count | count | 3.25〔4/2〕 | 0〔4/2〕 | NA〔0/0〕 |
| FF_illegal_pair_count | count | 23.25〔4/2〕 | 0.25〔4/2〕 | NA〔0/0〕 |
| free_minimum_angle_lt5_fraction | fraction_0_to_1 | 0.60729513〔4/2〕 | 0.71141835〔4/2〕 | NA〔0/0〕 |
| free_q_gt10_fraction | fraction_0_to_1 | 0.49917343〔4/2〕 | 0.6877839〔4/2〕 | NA〔0/0〕 |
| deleted_free_faces | count | 130.25〔4/2〕 | 157.25〔4/2〕 | NA〔0/0〕 |
| actual_free_faces | count | 57.75〔4/2〕 | 30.75〔4/2〕 | NA〔0/0〕 |
| free_total_surface_area | model_length_squared | 20.233125〔4/2〕 | 3.6653746〔4/2〕 | NA〔0/0〕 |
| new_bridge_faces | count | 0〔4/2〕 | 0〔4/2〕 | NA〔0/0〕 |
| bridge_area | model_length_squared | 0〔4/2〕 | 0〔4/2〕 | NA〔0/0〕 |
| deleted_original_POST_faces | count | 0〔4/2〕 | 0〔4/2〕 | NA〔0/0〕 |
| hinge_diag_isolated_1face_edge_components | count | 21.25〔4/2〕 | 19.5〔4/2〕 | NA〔0/0〕 |
| hinge_diag_isolated_1face_unfiltered_edge_components | count | 19.75〔4/2〕 | 19.5〔4/2〕 | NA〔0/0〕 |
| C_F_valid_edge_count | count | 0.5〔4/2〕 | 0.75〔4/2〕 | NA〔0/0〕 |
| C_F_valid_edge_total_length | model_length | 1.2933429〔4/2〕 | 3.6034393〔4/2〕 | NA〔0/0〕 |
| C_F_has_valid_edge | fraction_0_to_1 | 0.5〔4/2〕 | 0.75〔4/2〕 | NA〔0/0〕 |
| C_F_valid_opposite_winding_edge_count | count | 0.5〔4/2〕 | 0.75〔4/2〕 | NA〔0/0〕 |
| nonmanifold_edges | count | 1.5〔4/2〕 | 0〔4/2〕 | NA〔0/0〕 |
| C_F_nonmanifold_edge_count | count | 0〔4/2〕 | 0〔4/2〕 | NA〔0/0〕 |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | fraction_0_to_1 | 0.002107844〔4/2〕 | 0.37710396〔4/2〕 | NA〔0/0〕 |
| hinge_diag_fully_C_disconnected_free_area_fraction | fraction_0_to_1 | 0.9953715〔4/2〕 | 0.38743287〔4/2〕 | NA〔0/0〕 |
| hinge_diag_vertex_attached_only_free_area_fraction | fraction_0_to_1 | 0.0025206548〔4/2〕 | 0.23546317〔4/2〕 | NA〔0/0〕 |
| free_only_reached_bulk_area_fraction | fraction_0_to_1 | NA〔0/0〕 | NA〔0/0〕 | NA〔0/0〕 |
| retained_bulk_area | model_length_squared | NA〔0/0〕 | NA〔0/0〕 | NA〔0/0〕 |

## NEW_TRAIN / CLEAN_FINAL_POST

| 字段 | 单位 | Native6000 | Candidate1500 | 旧SEAM3000 |
|---|---|---:|---:|---:|
| CF_illegal_affected_free_fraction | fraction_0_to_1 | 0.047144703〔4/2〕 | 0〔4/2〕 | NA〔0/0〕 |
| FF_illegal_affected_free_fraction | fraction_0_to_1 | 0.20867386〔4/2〕 | 0.017857143〔4/2〕 | NA〔0/0〕 |
| ConflictCov_C | fraction_0_to_1 | 0.02734375〔4/2〕 | 0〔4/2〕 | NA〔0/0〕 |
| CF_illegal_pair_count | count | 3〔4/2〕 | 0〔4/2〕 | NA〔0/0〕 |
| FF_illegal_pair_count | count | 19.25〔4/2〕 | 0.25〔4/2〕 | NA〔0/0〕 |
| free_minimum_angle_lt5_fraction | fraction_0_to_1 | 0.57278455〔4/2〕 | 0.71141835〔4/2〕 | NA〔0/0〕 |
| free_q_gt10_fraction | fraction_0_to_1 | 0.48226867〔4/2〕 | 0.67307801〔4/2〕 | NA〔0/0〕 |
| deleted_free_faces | count | 130.25〔4/2〕 | 157.25〔4/2〕 | NA〔0/0〕 |
| actual_free_faces | count | 57.75〔4/2〕 | 30.75〔4/2〕 | NA〔0/0〕 |
| free_total_surface_area | model_length_squared | 20.516126〔4/2〕 | 3.7953953〔4/2〕 | NA〔0/0〕 |
| new_bridge_faces | count | 0〔4/2〕 | 0〔4/2〕 | NA〔0/0〕 |
| bridge_area | model_length_squared | 0〔4/2〕 | 0〔4/2〕 | NA〔0/0〕 |
| deleted_original_POST_faces | count | 0〔4/2〕 | 0〔4/2〕 | NA〔0/0〕 |
| hinge_diag_isolated_1face_edge_components | count | 13.75〔4/2〕 | 13.75〔4/2〕 | NA〔0/0〕 |
| hinge_diag_isolated_1face_unfiltered_edge_components | count | 12.75〔4/2〕 | 13.75〔4/2〕 | NA〔0/0〕 |
| C_F_valid_edge_count | count | 0.5〔4/2〕 | 0.75〔4/2〕 | NA〔0/0〕 |
| C_F_valid_edge_total_length | model_length | 1.2933429〔4/2〕 | 3.6034393〔4/2〕 | NA〔0/0〕 |
| C_F_has_valid_edge | fraction_0_to_1 | 0.5〔4/2〕 | 0.75〔4/2〕 | NA〔0/0〕 |
| C_F_valid_opposite_winding_edge_count | count | 0.5〔4/2〕 | 0.75〔4/2〕 | NA〔0/0〕 |
| nonmanifold_edges | count | 1.5〔4/2〕 | 0〔4/2〕 | NA〔0/0〕 |
| C_F_nonmanifold_edge_count | count | 0〔4/2〕 | 0〔4/2〕 | NA〔0/0〕 |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | fraction_0_to_1 | 0.0020621334〔4/2〕 | 0.37901637〔4/2〕 | NA〔0/0〕 |
| hinge_diag_fully_C_disconnected_free_area_fraction | fraction_0_to_1 | 0.99547168〔4/2〕 | 0.37314675〔4/2〕 | NA〔0/0〕 |
| hinge_diag_vertex_attached_only_free_area_fraction | fraction_0_to_1 | 0.0024661827〔4/2〕 | 0.24783688〔4/2〕 | NA〔0/0〕 |
| free_only_reached_bulk_area_fraction | fraction_0_to_1 | NA〔0/0〕 | NA〔0/0〕 | NA〔0/0〕 |
| retained_bulk_area | model_length_squared | NA〔0/0〕 | NA〔0/0〕 | NA〔0/0〕 |

## NEW_PARENT_HOLDOUT / RAW

| 字段 | 单位 | Native6000 | Candidate1500 | 旧SEAM3000 |
|---|---|---:|---:|---:|
| CF_illegal_affected_free_fraction | fraction_0_to_1 | 0.25576037〔2/1〕 | 0.25806452〔2/1〕 | NA〔0/0〕 |
| FF_illegal_affected_free_fraction | fraction_0_to_1 | 0.91474654〔2/1〕 | 0.92396313〔2/1〕 | NA〔0/0〕 |
| ConflictCov_C | fraction_0_to_1 | 0.097167969〔2/1〕 | 0.090087891〔2/1〕 | NA〔0/0〕 |
| CF_illegal_pair_count | count | 82〔2/1〕 | 81〔2/1〕 | NA〔0/0〕 |
| FF_illegal_pair_count | count | 288〔2/1〕 | 282.5〔2/1〕 | NA〔0/0〕 |
| free_minimum_angle_lt5_fraction | fraction_0_to_1 | 0.14976959〔2/1〕 | 0.17281106〔2/1〕 | NA〔0/0〕 |
| free_q_gt10_fraction | fraction_0_to_1 | 0.092165899〔2/1〕 | 0.099078341〔2/1〕 | NA〔0/0〕 |
| deleted_free_faces | count | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| actual_free_faces | count | 217〔2/1〕 | 217〔2/1〕 | NA〔0/0〕 |
| free_total_surface_area | model_length_squared | 20.702181〔2/1〕 | 21.61522〔2/1〕 | NA〔0/0〕 |
| new_bridge_faces | count | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| bridge_area | model_length_squared | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| deleted_original_POST_faces | count | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| hinge_diag_isolated_1face_edge_components | count | 217〔2/1〕 | 217〔2/1〕 | NA〔0/0〕 |
| hinge_diag_isolated_1face_unfiltered_edge_components | count | 217〔2/1〕 | 217〔2/1〕 | NA〔0/0〕 |
| C_F_valid_edge_count | count | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| C_F_valid_edge_total_length | model_length | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| C_F_has_valid_edge | fraction_0_to_1 | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| C_F_valid_opposite_winding_edge_count | count | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| nonmanifold_edges | count | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| C_F_nonmanifold_edge_count | count | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | fraction_0_to_1 | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| hinge_diag_fully_C_disconnected_free_area_fraction | fraction_0_to_1 | 1〔2/1〕 | 1〔2/1〕 | NA〔0/0〕 |
| hinge_diag_vertex_attached_only_free_area_fraction | fraction_0_to_1 | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| free_only_reached_bulk_area_fraction | fraction_0_to_1 | NA〔0/0〕 | NA〔0/0〕 | NA〔0/0〕 |
| retained_bulk_area | model_length_squared | NA〔0/0〕 | NA〔0/0〕 | NA〔0/0〕 |

## NEW_PARENT_HOLDOUT / BASE_POST

| 字段 | 单位 | Native6000 | Candidate1500 | 旧SEAM3000 |
|---|---|---:|---:|---:|
| CF_illegal_affected_free_fraction | fraction_0_to_1 | 0.20720635〔2/1〕 | 0.24570802〔2/1〕 | NA〔0/0〕 |
| FF_illegal_affected_free_fraction | fraction_0_to_1 | 0.19504762〔2/1〕 | 0.20444862〔2/1〕 | NA〔0/0〕 |
| ConflictCov_C | fraction_0_to_1 | 0.095703125〔2/1〕 | 0.11303711〔2/1〕 | NA〔0/0〕 |
| CF_illegal_pair_count | count | 33〔2/1〕 | 38.5〔2/1〕 | NA〔0/0〕 |
| FF_illegal_pair_count | count | 15.5〔2/1〕 | 14〔2/1〕 | NA〔0/0〕 |
| free_minimum_angle_lt5_fraction | fraction_0_to_1 | 0.19136508〔2/1〕 | 0.18067043〔2/1〕 | NA〔0/0〕 |
| free_q_gt10_fraction | fraction_0_to_1 | 0.14749206〔2/1〕 | 0.15313283〔2/1〕 | NA〔0/0〕 |
| deleted_free_faces | count | 91.5〔2/1〕 | 90.5〔2/1〕 | NA〔0/0〕 |
| actual_free_faces | count | 125.5〔2/1〕 | 126.5〔2/1〕 | NA〔0/0〕 |
| free_total_surface_area | model_length_squared | 9.5201196〔2/1〕 | 9.9452721〔2/1〕 | NA〔0/0〕 |
| new_bridge_faces | count | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| bridge_area | model_length_squared | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| deleted_original_POST_faces | count | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| hinge_diag_isolated_1face_edge_components | count | 97.5〔2/1〕 | 102〔2/1〕 | NA〔0/0〕 |
| hinge_diag_isolated_1face_unfiltered_edge_components | count | 97.5〔2/1〕 | 102〔2/1〕 | NA〔0/0〕 |
| C_F_valid_edge_count | count | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| C_F_valid_edge_total_length | model_length | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| C_F_has_valid_edge | fraction_0_to_1 | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| C_F_valid_opposite_winding_edge_count | count | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| nonmanifold_edges | count | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| C_F_nonmanifold_edge_count | count | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | fraction_0_to_1 | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| hinge_diag_fully_C_disconnected_free_area_fraction | fraction_0_to_1 | 1〔2/1〕 | 1〔2/1〕 | NA〔0/0〕 |
| hinge_diag_vertex_attached_only_free_area_fraction | fraction_0_to_1 | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| free_only_reached_bulk_area_fraction | fraction_0_to_1 | NA〔0/0〕 | NA〔0/0〕 | NA〔0/0〕 |
| retained_bulk_area | model_length_squared | NA〔0/0〕 | NA〔0/0〕 | NA〔0/0〕 |

## NEW_PARENT_HOLDOUT / CLEAN_FINAL_POST

| 字段 | 单位 | Native6000 | Candidate1500 | 旧SEAM3000 |
|---|---|---:|---:|---:|
| CF_illegal_affected_free_fraction | fraction_0_to_1 | 0.2032381〔2/1〕 | 0.24194862〔2/1〕 | NA〔0/0〕 |
| FF_illegal_affected_free_fraction | fraction_0_to_1 | 0.1591746〔2/1〕 | 0.16939223〔2/1〕 | NA〔0/0〕 |
| ConflictCov_C | fraction_0_to_1 | 0.092529297〔2/1〕 | 0.11303711〔2/1〕 | NA〔0/0〕 |
| CF_illegal_pair_count | count | 32〔2/1〕 | 36.5〔2/1〕 | NA〔0/0〕 |
| FF_illegal_pair_count | count | 12.5〔2/1〕 | 11.5〔2/1〕 | NA〔0/0〕 |
| free_minimum_angle_lt5_fraction | fraction_0_to_1 | 0.19136508〔2/1〕 | 0.17691103〔2/1〕 | NA〔0/0〕 |
| free_q_gt10_fraction | fraction_0_to_1 | 0.13955556〔2/1〕 | 0.14896617〔2/1〕 | NA〔0/0〕 |
| deleted_free_faces | count | 91.5〔2/1〕 | 90.5〔2/1〕 | NA〔0/0〕 |
| actual_free_faces | count | 125.5〔2/1〕 | 126.5〔2/1〕 | NA〔0/0〕 |
| free_total_surface_area | model_length_squared | 9.542638〔2/1〕 | 9.9115311〔2/1〕 | NA〔0/0〕 |
| new_bridge_faces | count | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| bridge_area | model_length_squared | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| deleted_original_POST_faces | count | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| hinge_diag_isolated_1face_edge_components | count | 66〔2/1〕 | 75〔2/1〕 | NA〔0/0〕 |
| hinge_diag_isolated_1face_unfiltered_edge_components | count | 66〔2/1〕 | 75〔2/1〕 | NA〔0/0〕 |
| C_F_valid_edge_count | count | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| C_F_valid_edge_total_length | model_length | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| C_F_has_valid_edge | fraction_0_to_1 | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| C_F_valid_opposite_winding_edge_count | count | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| nonmanifold_edges | count | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| C_F_nonmanifold_edge_count | count | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | fraction_0_to_1 | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| hinge_diag_fully_C_disconnected_free_area_fraction | fraction_0_to_1 | 1〔2/1〕 | 1〔2/1〕 | NA〔0/0〕 |
| hinge_diag_vertex_attached_only_free_area_fraction | fraction_0_to_1 | 0〔2/1〕 | 0〔2/1〕 | NA〔0/0〕 |
| free_only_reached_bulk_area_fraction | fraction_0_to_1 | NA〔0/0〕 | NA〔0/0〕 | NA〔0/0〕 |
| retained_bulk_area | model_length_squared | NA〔0/0〕 | NA〔0/0〕 | NA〔0/0〕 |

## DEV / RAW

| 字段 | 单位 | Native6000 | Candidate1500 | 旧SEAM3000 |
|---|---|---:|---:|---:|
| CF_illegal_affected_free_fraction | fraction_0_to_1 | 0.062456681〔8/4〕 | 0.058316271〔8/4〕 | 0.061803043〔8/4〕 |
| FF_illegal_affected_free_fraction | fraction_0_to_1 | 0.90488784〔8/4〕 | 0.88979872〔8/4〕 | 0.89212531〔8/4〕 |
| ConflictCov_C | fraction_0_to_1 | 0.14050293〔8/4〕 | 0.12139893〔8/4〕 | 0.13964844〔8/4〕 |
| CF_illegal_pair_count | count | 14.125〔8/4〕 | 12.75〔8/4〕 | 13.5〔8/4〕 |
| FF_illegal_pair_count | count | 274.75〔8/4〕 | 261.125〔8/4〕 | 257.625〔8/4〕 |
| free_minimum_angle_lt5_fraction | fraction_0_to_1 | 0.45985662〔8/4〕 | 0.46139509〔8/4〕 | 0.44734778〔8/4〕 |
| free_q_gt10_fraction | fraction_0_to_1 | 0.3411613〔8/4〕 | 0.33362244〔8/4〕 | 0.33398182〔8/4〕 |
| deleted_free_faces | count | 0〔8/4〕 | 0〔8/4〕 | 0〔8/4〕 |
| actual_free_faces | count | 174〔8/4〕 | 174〔8/4〕 | 174〔8/4〕 |
| free_total_surface_area | model_length_squared | 43.912336〔8/4〕 | 36.138589〔8/4〕 | 40.467518〔8/4〕 |
| new_bridge_faces | count | 0〔8/4〕 | 0〔8/4〕 | 0〔8/4〕 |
| bridge_area | model_length_squared | 0〔8/4〕 | 0〔8/4〕 | 0〔8/4〕 |
| deleted_original_POST_faces | count | 0〔8/4〕 | 0〔8/4〕 | 0〔8/4〕 |
| hinge_diag_isolated_1face_edge_components | count | 174〔8/4〕 | 174〔8/4〕 | 171.5〔8/4〕 |
| hinge_diag_isolated_1face_unfiltered_edge_components | count | 174〔8/4〕 | 174〔8/4〕 | 171.5〔8/4〕 |
| C_F_valid_edge_count | count | 0〔8/4〕 | 0〔8/4〕 | 2.5〔8/4〕 |
| C_F_valid_edge_total_length | model_length | 0〔8/4〕 | 0〔8/4〕 | 10.135987〔8/4〕 |
| C_F_has_valid_edge | fraction_0_to_1 | 0〔8/4〕 | 0〔8/4〕 | 1〔8/4〕 |
| C_F_valid_opposite_winding_edge_count | count | 0〔8/4〕 | 0〔8/4〕 | 2.5〔8/4〕 |
| nonmanifold_edges | count | 0〔8/4〕 | 0〔8/4〕 | 0〔8/4〕 |
| C_F_nonmanifold_edge_count | count | 0〔8/4〕 | 0〔8/4〕 | 0〔8/4〕 |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | fraction_0_to_1 | 0〔8/4〕 | 0〔8/4〕 | 0.046132245〔8/4〕 |
| hinge_diag_fully_C_disconnected_free_area_fraction | fraction_0_to_1 | 1〔8/4〕 | 1〔8/4〕 | 0.95386776〔8/4〕 |
| hinge_diag_vertex_attached_only_free_area_fraction | fraction_0_to_1 | 0〔8/4〕 | 0〔8/4〕 | 0〔8/4〕 |
| free_only_reached_bulk_area_fraction | fraction_0_to_1 | NA〔0/0〕 | NA〔0/0〕 | 0〔8/4〕 |
| retained_bulk_area | model_length_squared | NA〔0/0〕 | NA〔0/0〕 | 38.54153〔8/4〕 |

## DEV / BASE_POST

| 字段 | 单位 | Native6000 | Candidate1500 | 旧SEAM3000 |
|---|---|---:|---:|---:|
| CF_illegal_affected_free_fraction | fraction_0_to_1 | 0.075485898〔8/4〕 | 0.064756491〔8/4〕 | 0.081341579〔8/4〕 |
| FF_illegal_affected_free_fraction | fraction_0_to_1 | 0.27644124〔8/4〕 | 0.2527057〔8/4〕 | 0.30001404〔8/4〕 |
| ConflictCov_C | fraction_0_to_1 | 0.128479〔8/4〕 | 0.1151123〔8/4〕 | 0.13195801〔8/4〕 |
| CF_illegal_pair_count | count | 9.375〔8/4〕 | 9.75〔8/4〕 | 10〔8/4〕 |
| FF_illegal_pair_count | count | 22.125〔8/4〕 | 22.875〔8/4〕 | 30.375〔8/4〕 |
| free_minimum_angle_lt5_fraction | fraction_0_to_1 | 0.59017929〔8/4〕 | 0.66928982〔8/4〕 | 0.56851548〔8/4〕 |
| free_q_gt10_fraction | fraction_0_to_1 | 0.47095243〔8/4〕 | 0.55438943〔8/4〕 | 0.44815502〔8/4〕 |
| deleted_free_faces | count | 105.25〔8/4〕 | 104〔8/4〕 | 105.125〔8/4〕 |
| actual_free_faces | count | 68.75〔8/4〕 | 70〔8/4〕 | 68.875〔8/4〕 |
| free_total_surface_area | model_length_squared | 27.664603〔8/4〕 | 21.9699〔8/4〕 | 27.588789〔8/4〕 |
| new_bridge_faces | count | 0〔8/4〕 | 0〔8/4〕 | 0〔8/4〕 |
| bridge_area | model_length_squared | 0〔8/4〕 | 0〔8/4〕 | 0〔8/4〕 |
| deleted_original_POST_faces | count | 0〔8/4〕 | 0〔8/4〕 | 0〔8/4〕 |
| hinge_diag_isolated_1face_edge_components | count | 20.625〔8/4〕 | 26.75〔8/4〕 | 22.375〔8/4〕 |
| hinge_diag_isolated_1face_unfiltered_edge_components | count | 18.625〔8/4〕 | 23〔8/4〕 | 20.875〔8/4〕 |
| C_F_valid_edge_count | count | 0〔8/4〕 | 0〔8/4〕 | 2.5〔8/4〕 |
| C_F_valid_edge_total_length | model_length | 0〔8/4〕 | 0〔8/4〕 | 10.135987〔8/4〕 |
| C_F_has_valid_edge | fraction_0_to_1 | 0〔8/4〕 | 0〔8/4〕 | 1〔8/4〕 |
| C_F_valid_opposite_winding_edge_count | count | 0〔8/4〕 | 0〔8/4〕 | 2.5〔8/4〕 |
| nonmanifold_edges | count | 1.25〔8/4〕 | 2.25〔8/4〕 | 1.375〔8/4〕 |
| C_F_nonmanifold_edge_count | count | 0〔8/4〕 | 0〔8/4〕 | 0〔8/4〕 |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | fraction_0_to_1 | 0〔8/4〕 | 0〔8/4〕 | 0.070184342〔8/4〕 |
| hinge_diag_fully_C_disconnected_free_area_fraction | fraction_0_to_1 | 1〔8/4〕 | 0.99336043〔8/4〕 | 0.52809264〔8/4〕 |
| hinge_diag_vertex_attached_only_free_area_fraction | fraction_0_to_1 | 0〔8/4〕 | 0.0066395703〔8/4〕 | 0.40172302〔8/4〕 |
| free_only_reached_bulk_area_fraction | fraction_0_to_1 | NA〔0/0〕 | NA〔0/0〕 | 0〔8/4〕 |
| retained_bulk_area | model_length_squared | NA〔0/0〕 | NA〔0/0〕 | 25.776967〔8/4〕 |

## DEV / CLEAN_FINAL_POST

| 字段 | 单位 | Native6000 | Candidate1500 | 旧SEAM3000 |
|---|---|---:|---:|---:|
| CF_illegal_affected_free_fraction | fraction_0_to_1 | 0.072881731〔8/4〕 | 0.064756491〔8/4〕 | 0.079026764〔8/4〕 |
| FF_illegal_affected_free_fraction | fraction_0_to_1 | 0.24051743〔8/4〕 | 0.22259009〔8/4〕 | 0.27209148〔8/4〕 |
| ConflictCov_C | fraction_0_to_1 | 0.12854004〔8/4〕 | 0.11419678〔8/4〕 | 0.13146973〔8/4〕 |
| CF_illegal_pair_count | count | 9.25〔8/4〕 | 9.75〔8/4〕 | 9.875〔8/4〕 |
| FF_illegal_pair_count | count | 20.25〔8/4〕 | 20.625〔8/4〕 | 28.25〔8/4〕 |
| free_minimum_angle_lt5_fraction | fraction_0_to_1 | 0.58012763〔8/4〕 | 0.65321662〔8/4〕 | 0.55882345〔8/4〕 |
| free_q_gt10_fraction | fraction_0_to_1 | 0.46517749〔8/4〕 | 0.54021619〔8/4〕 | 0.44101144〔8/4〕 |
| deleted_free_faces | count | 105.25〔8/4〕 | 104〔8/4〕 | 105.125〔8/4〕 |
| actual_free_faces | count | 68.75〔8/4〕 | 70〔8/4〕 | 68.875〔8/4〕 |
| free_total_surface_area | model_length_squared | 27.799473〔8/4〕 | 22.195948〔8/4〕 | 27.699267〔8/4〕 |
| new_bridge_faces | count | 0〔8/4〕 | 0〔8/4〕 | 0〔8/4〕 |
| bridge_area | model_length_squared | 0〔8/4〕 | 0〔8/4〕 | 0〔8/4〕 |
| deleted_original_POST_faces | count | 0〔8/4〕 | 0〔8/4〕 | 0〔8/4〕 |
| hinge_diag_isolated_1face_edge_components | count | 14.5〔8/4〕 | 19.5〔8/4〕 | 16.375〔8/4〕 |
| hinge_diag_isolated_1face_unfiltered_edge_components | count | 12.625〔8/4〕 | 15.875〔8/4〕 | 14.875〔8/4〕 |
| C_F_valid_edge_count | count | 0〔8/4〕 | 0〔8/4〕 | 2.5〔8/4〕 |
| C_F_valid_edge_total_length | model_length | 0〔8/4〕 | 0〔8/4〕 | 10.135987〔8/4〕 |
| C_F_has_valid_edge | fraction_0_to_1 | 0〔8/4〕 | 0〔8/4〕 | 1〔8/4〕 |
| C_F_valid_opposite_winding_edge_count | count | 0〔8/4〕 | 0〔8/4〕 | 2.5〔8/4〕 |
| nonmanifold_edges | count | 1.25〔8/4〕 | 2.25〔8/4〕 | 1.375〔8/4〕 |
| C_F_nonmanifold_edge_count | count | 0〔8/4〕 | 0〔8/4〕 | 0〔8/4〕 |
| hinge_diag_legal_opposed_edge_C_reached_free_area_fraction | fraction_0_to_1 | 0〔8/4〕 | 0〔8/4〕 | 0.069940846〔8/4〕 |
| hinge_diag_fully_C_disconnected_free_area_fraction | fraction_0_to_1 | 1〔8/4〕 | 0.99336043〔8/4〕 | 0.52830398〔8/4〕 |
| hinge_diag_vertex_attached_only_free_area_fraction | fraction_0_to_1 | 0〔8/4〕 | 0.0066395703〔8/4〕 | 0.40175518〔8/4〕 |
| free_only_reached_bulk_area_fraction | fraction_0_to_1 | NA〔0/0〕 | NA〔0/0〕 | 0〔8/4〕 |
| retained_bulk_area | model_length_squared | NA〔0/0〕 | NA〔0/0〕 | 25.887446〔8/4〕 |

