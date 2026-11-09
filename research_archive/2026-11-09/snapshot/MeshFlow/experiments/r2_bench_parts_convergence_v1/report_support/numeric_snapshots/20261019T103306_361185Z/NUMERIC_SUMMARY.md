# Bench 已有数值摘要（报告支持，PRELIMINARY_PENDING_FINAL）

只读完成表，0模型调用、0重评分。保留全部124字段；以下是便于阅读的质量、接续与代价字段，不是事后挑选的论文主指标。
原始比例保持0–1，未换算成百分比；长度为模型坐标单位，面积为模型坐标平方单位。均值为原表父等权值；括号内为有效输出/登记输出。

闭合C不要求接续。原一般接口字段对闭合C可能为0，不能把全组接口有无直接称为任务成功率；seam特有字段对Native及闭合C保留NA。图中薄片/孔洞/主体缺失不能由小角或连通分量数直接认证，仍需独立AI_PROXY图审。

## 完成状态

- COMMON12000：`NATIVE_S12000_full`，COMPLETE
- PAIRED15000：`NATIVE_S15000__SEAM_S15000_full`，COMPLETE
- FINAL：`NATIVE_S21000__SEAM_S21000_full`，PENDING

## RAW / TRAIN：父等权

| 原字段 | NATIVE_S12000 | NATIVE_S15000 | SEAM_S15000 |
|---|---:|---:|---:|
| `C_preserved_exact` | 1 (16/16) | 1 (16/16) | 1 (16/16) |
| `CF_illegal_affected_free_fraction` | 0.06948301 (16/16) | 0.08297231 (16/16) | 0.06431083 (16/16) |
| `FF_illegal_affected_free_fraction` | 0.9300946 (16/16) | 0.9424532 (16/16) | 0.921403 (16/16) |
| `ConflictCov_C` | 0.1747437 (16/16) | 0.1832275 (16/16) | 0.1671448 (16/16) |
| `ConflictCov_C_upper` | 0.1747437 (16/16) | 0.1832275 (16/16) | 0.1671448 (16/16) |
| `ConflictCov_C_uncertain_fraction` | 0 (16/16) | 0 (16/16) | 0 (16/16) |
| `CF_illegal_pair_count` | 21 (16/16) | 24 (16/16) | 19.625 (16/16) |
| `FF_illegal_pair_count` | 440.4375 (16/16) | 455.25 (16/16) | 448.875 (16/16) |
| `C_F_has_valid_edge` | 0 (16/16) | 0 (16/16) | 0.875 (16/16) |
| `C_F_valid_edge_count` | 0 (16/16) | 0 (16/16) | 6.625 (16/16) |
| `C_F_valid_edge_total_length` | 0 (16/16) | 0 (16/16) | 15.85679 (16/16) |
| `C_F_valid_opposite_winding_edge_count` | 0 (16/16) | 0 (16/16) | 6.625 (16/16) |
| `opposite_winding_reached_free_area_fraction` | 0 (16/16) | 0 (16/16) | 0.08108308 (16/16) |
| `free_only_reached_bulk_area_fraction` | NA (0/16) | NA (0/16) | 0 (14/16) |
| `bulk_reach_ge_0_30_descriptive` | NA (0/16) | NA (0/16) | 0 (14/16) |
| `legal_seam_bulk_edge_count` | NA (0/16) | NA (0/16) | 0 (14/16) |
| `seed_connected_legal_seam_bulk_edge_count` | NA (0/16) | NA (0/16) | 0 (14/16) |
| `A_VMR__L_tau_0.005` | 0.2 (14/16) | 0.3125 (14/16) | 0.9375 (14/16) |
| `PMR__L_tau_0.005` | 0.1875 (14/16) | 0.3125 (14/16) | 0.875 (14/16) |
| `boundary_distance_mean_model` | 0.3357992 (14/16) | 0.2802992 (14/16) | 0.01883712 (14/16) |
| `free_minimum_angle_lt5_fraction` | 0.3355149 (16/16) | 0.3622753 (16/16) | 0.3512643 (16/16) |
| `free_q_gt10_fraction` | 0.2644446 (16/16) | 0.2838126 (16/16) | 0.2823881 (16/16) |
| `free_angle_p05_degrees` | 1.612893 (16/16) | 1.380601 (16/16) | 1.474741 (16/16) |
| `free_near_degenerate_count` | 0 (16/16) | 0 (16/16) | 0 (16/16) |
| `free_exact_zero_area_count` | 0 (16/16) | 0 (16/16) | 0 (16/16) |
| `actual_free_faces` | 202.25 (16/16) | 202.25 (16/16) | 202.25 (16/16) |
| `deleted_free_faces` | 0 (16/16) | 0 (16/16) | 0 (16/16) |
| `deleted_seam_faces` | NA (0/16) | NA (0/16) | 0 (14/16) |
| `nonmanifold_edges` | 0 (16/16) | 0 (16/16) | 0 (16/16) |
| `C_F_nonmanifold_edge_count` | 0 (16/16) | 0 (16/16) | 0 (16/16) |
| `EXACT_COORD_face_components` | 203.625 (16/16) | 203.625 (16/16) | 197 (16/16) |
| `EXACT_COORD_same_direction_two_face_edges` | 0 (16/16) | 0 (16/16) | 0 (16/16) |
| `free_total_surface_area` | 68.21491 (16/16) | 52.06935 (16/16) | 57.16758 (16/16) |
| `retained_seam_faces` | NA (0/16) | NA (0/16) | 6.875 (14/16) |
| `retained_bulk_faces` | NA (0/16) | NA (0/16) | 196.625 (14/16) |

## RAW / DEV：父等权

| 原字段 | NATIVE_S12000 | NATIVE_S15000 | SEAM_S15000 |
|---|---:|---:|---:|
| `C_preserved_exact` | 1 (32/32) | 1 (32/32) | 1 (32/32) |
| `CF_illegal_affected_free_fraction` | 0.07070358 (32/32) | 0.06864134 (32/32) | 0.06984275 (32/32) |
| `FF_illegal_affected_free_fraction` | 0.952246 (32/32) | 0.9481697 (32/32) | 0.9369084 (32/32) |
| `ConflictCov_C` | 0.09933472 (32/32) | 0.08007812 (32/32) | 0.1158142 (32/32) |
| `ConflictCov_C_upper` | 0.09933472 (32/32) | 0.08007812 (32/32) | 0.1158142 (32/32) |
| `ConflictCov_C_uncertain_fraction` | 0 (32/32) | 0 (32/32) | 0 (32/32) |
| `CF_illegal_pair_count` | 20.28125 (32/32) | 19 (32/32) | 20.53125 (32/32) |
| `FF_illegal_pair_count` | 399.8125 (32/32) | 378.5938 (32/32) | 395.75 (32/32) |
| `C_F_has_valid_edge` | 0 (32/32) | 0 (32/32) | 0.75 (32/32) |
| `C_F_valid_edge_count` | 0 (32/32) | 0 (32/32) | 3.25 (32/32) |
| `C_F_valid_edge_total_length` | 0 (32/32) | 0 (32/32) | 9.019893 (32/32) |
| `C_F_valid_opposite_winding_edge_count` | 0 (32/32) | 0 (32/32) | 3.25 (32/32) |
| `opposite_winding_reached_free_area_fraction` | 0 (32/32) | 0 (32/32) | 0.07425149 (32/32) |
| `free_only_reached_bulk_area_fraction` | NA (0/32) | NA (0/32) | 0 (26/32) |
| `bulk_reach_ge_0_30_descriptive` | NA (0/32) | NA (0/32) | 0 (26/32) |
| `legal_seam_bulk_edge_count` | NA (0/32) | NA (0/32) | 0 (26/32) |
| `seed_connected_legal_seam_bulk_edge_count` | NA (0/32) | NA (0/32) | 0 (26/32) |
| `A_VMR__L_tau_0.005` | 0 (26/32) | 0.01388889 (26/32) | 0.8354167 (26/32) |
| `PMR__L_tau_0.005` | 0 (26/32) | 0 (26/32) | 0.625 (26/32) |
| `boundary_distance_mean_model` | 0.5280744 (26/32) | 0.4857189 (26/32) | 0.05576233 (26/32) |
| `free_minimum_angle_lt5_fraction` | 0.3509833 (32/32) | 0.3695004 (32/32) | 0.3496175 (32/32) |
| `free_q_gt10_fraction` | 0.2735734 (32/32) | 0.2949384 (32/32) | 0.2714935 (32/32) |
| `free_angle_p05_degrees` | 2.271382 (32/32) | 2.12816 (32/32) | 2.339686 (32/32) |
| `free_near_degenerate_count` | 0 (32/32) | 0 (32/32) | 0 (32/32) |
| `free_exact_zero_area_count` | 0 (32/32) | 0 (32/32) | 0 (32/32) |
| `actual_free_faces` | 169.25 (32/32) | 169.25 (32/32) | 169.25 (32/32) |
| `deleted_free_faces` | 0 (32/32) | 0 (32/32) | 0 (32/32) |
| `deleted_seam_faces` | NA (0/32) | NA (0/32) | 0 (26/32) |
| `nonmanifold_edges` | 0 (32/32) | 0 (32/32) | 0 (32/32) |
| `C_F_nonmanifold_edge_count` | 0 (32/32) | 0 (32/32) | 0 (32/32) |
| `EXACT_COORD_face_components` | 170.4375 (32/32) | 170.4375 (32/32) | 167.1875 (32/32) |
| `EXACT_COORD_same_direction_two_face_edges` | 0 (32/32) | 0 (32/32) | 0 (32/32) |
| `free_total_surface_area` | 56.8586 (32/32) | 41.34938 (32/32) | 46.88305 (32/32) |
| `retained_seam_faces` | NA (0/32) | NA (0/32) | 3.9375 (26/32) |
| `retained_bulk_faces` | NA (0/32) | NA (0/32) | 172.1875 (26/32) |

## POST / TRAIN：父等权

| 原字段 | NATIVE_S12000 | NATIVE_S15000 | SEAM_S15000 |
|---|---:|---:|---:|
| `C_preserved_exact` | 1 (16/16) | 1 (16/16) | 1 (16/16) |
| `CF_illegal_affected_free_fraction` | 0.06998662 (16/16) | 0.06354856 (16/16) | 0.07183436 (16/16) |
| `FF_illegal_affected_free_fraction` | 0.2690818 (16/16) | 0.2907221 (16/16) | 0.3533668 (16/16) |
| `ConflictCov_C` | 0.112915 (16/16) | 0.07873535 (16/16) | 0.1560364 (16/16) |
| `ConflictCov_C_upper` | 0.112915 (16/16) | 0.07873535 (16/16) | 0.1560364 (16/16) |
| `ConflictCov_C_uncertain_fraction` | 0 (16/16) | 0 (16/16) | 0 (16/16) |
| `CF_illegal_pair_count` | 9.75 (16/16) | 8.6875 (16/16) | 10.75 (16/16) |
| `FF_illegal_pair_count` | 28.625 (16/16) | 38.875 (16/16) | 43.8125 (16/16) |
| `C_F_has_valid_edge` | 0.1875 (16/16) | 0.25 (16/16) | 0.875 (16/16) |
| `C_F_valid_edge_count` | 1.25 (16/16) | 1.9375 (16/16) | 5.3125 (16/16) |
| `C_F_valid_edge_total_length` | 2.405681 (16/16) | 3.554254 (16/16) | 14.13738 (16/16) |
| `C_F_valid_opposite_winding_edge_count` | 1.25 (16/16) | 1.9375 (16/16) | 5.3125 (16/16) |
| `opposite_winding_reached_free_area_fraction` | 0.01599351 (16/16) | 0.02241655 (16/16) | 0.1022916 (16/16) |
| `free_only_reached_bulk_area_fraction` | NA (0/16) | NA (0/16) | 0.001262939 (14/16) |
| `bulk_reach_ge_0_30_descriptive` | NA (0/16) | NA (0/16) | 0 (14/16) |
| `legal_seam_bulk_edge_count` | NA (0/16) | NA (0/16) | 0.5 (14/16) |
| `seed_connected_legal_seam_bulk_edge_count` | NA (0/16) | NA (0/16) | 0.5 (14/16) |
| `A_VMR__L_tau_0.005` | 0.1611111 (14/16) | 0.2440972 (14/16) | 0.9305556 (14/16) |
| `PMR__L_tau_0.005` | 0.0625 (14/16) | 0.125 (14/16) | 0.75 (14/16) |
| `boundary_distance_mean_model` | 0.3390476 (14/16) | 0.3040193 (14/16) | 0.0200665 (14/16) |
| `free_minimum_angle_lt5_fraction` | 0.4190459 (16/16) | 0.4623099 (16/16) | 0.421474 (16/16) |
| `free_q_gt10_fraction` | 0.318289 (16/16) | 0.3359301 (16/16) | 0.3208555 (16/16) |
| `free_angle_p05_degrees` | 1.711908 (16/16) | 1.535116 (16/16) | 1.683163 (16/16) |
| `free_near_degenerate_count` | 0 (16/16) | 0 (16/16) | 0 (16/16) |
| `free_exact_zero_area_count` | 0 (16/16) | 0 (16/16) | 0 (16/16) |
| `actual_free_faces` | 94.0625 (16/16) | 97.875 (16/16) | 100.75 (16/16) |
| `deleted_free_faces` | 108.1875 (16/16) | 104.375 (16/16) | 101.5 (16/16) |
| `deleted_seam_faces` | NA (0/16) | NA (0/16) | 1.125 (14/16) |
| `nonmanifold_edges` | 2.75 (16/16) | 3.125 (16/16) | 2.9375 (16/16) |
| `C_F_nonmanifold_edge_count` | 0.0625 (16/16) | 0.375 (16/16) | 0.625 (16/16) |
| `EXACT_COORD_face_components` | 41.75 (16/16) | 42.25 (16/16) | 41.6875 (16/16) |
| `EXACT_COORD_same_direction_two_face_edges` | 0 (16/16) | 0 (16/16) | 0 (16/16) |
| `free_total_surface_area` | 47.45973 (16/16) | 35.51678 (16/16) | 40.93991 (16/16) |
| `retained_seam_faces` | NA (0/16) | NA (0/16) | 5.75 (14/16) |
| `retained_bulk_faces` | NA (0/16) | NA (0/16) | 92.6875 (14/16) |

## POST / DEV：父等权

| 原字段 | NATIVE_S12000 | NATIVE_S15000 | SEAM_S15000 |
|---|---:|---:|---:|
| `C_preserved_exact` | 1 (32/32) | 1 (32/32) | 1 (32/32) |
| `CF_illegal_affected_free_fraction` | 0.07247688 (32/32) | 0.0744191 (32/32) | 0.07760421 (32/32) |
| `FF_illegal_affected_free_fraction` | 0.2807092 (32/32) | 0.2759906 (32/32) | 0.3113649 (32/32) |
| `ConflictCov_C` | 0.06008911 (32/32) | 0.05905151 (32/32) | 0.07775879 (32/32) |
| `ConflictCov_C_upper` | 0.06008911 (32/32) | 0.05905151 (32/32) | 0.07775879 (32/32) |
| `ConflictCov_C_uncertain_fraction` | 0 (32/32) | 0 (32/32) | 0 (32/32) |
| `CF_illegal_pair_count` | 10.40625 (32/32) | 10 (32/32) | 11.15625 (32/32) |
| `FF_illegal_pair_count` | 32.1875 (32/32) | 26.09375 (32/32) | 33.6875 (32/32) |
| `C_F_has_valid_edge` | 0 (32/32) | 0 (32/32) | 0.75 (32/32) |
| `C_F_valid_edge_count` | 0 (32/32) | 0 (32/32) | 3.21875 (32/32) |
| `C_F_valid_edge_total_length` | 0 (32/32) | 0 (32/32) | 9.004528 (32/32) |
| `C_F_valid_opposite_winding_edge_count` | 0 (32/32) | 0 (32/32) | 3.21875 (32/32) |
| `opposite_winding_reached_free_area_fraction` | 0 (32/32) | 0 (32/32) | 0.09404865 (32/32) |
| `free_only_reached_bulk_area_fraction` | NA (0/32) | NA (0/32) | 0.0001035839 (26/32) |
| `bulk_reach_ge_0_30_descriptive` | NA (0/32) | NA (0/32) | 0 (26/32) |
| `legal_seam_bulk_edge_count` | NA (0/32) | NA (0/32) | 0.03125 (26/32) |
| `seed_connected_legal_seam_bulk_edge_count` | NA (0/32) | NA (0/32) | 0.03125 (26/32) |
| `A_VMR__L_tau_0.005` | 0 (26/32) | 0.02777778 (26/32) | 0.8354167 (26/32) |
| `PMR__L_tau_0.005` | 0 (26/32) | 0 (26/32) | 0.625 (26/32) |
| `boundary_distance_mean_model` | 0.5525472 (26/32) | 0.4968313 (26/32) | 0.05644919 (26/32) |
| `free_minimum_angle_lt5_fraction` | 0.445366 (32/32) | 0.4861715 (32/32) | 0.4563787 (32/32) |
| `free_q_gt10_fraction` | 0.3678867 (32/32) | 0.4039286 (32/32) | 0.3713016 (32/32) |
| `free_angle_p05_degrees` | 1.883183 (32/32) | 1.739833 (32/32) | 1.99395 (32/32) |
| `free_near_degenerate_count` | 0 (32/32) | 0 (32/32) | 0 (32/32) |
| `free_exact_zero_area_count` | 0 (32/32) | 0 (32/32) | 0 (32/32) |
| `actual_free_faces` | 83.375 (32/32) | 80.8125 (32/32) | 84.09375 (32/32) |
| `deleted_free_faces` | 85.875 (32/32) | 88.4375 (32/32) | 85.15625 (32/32) |
| `deleted_seam_faces` | NA (0/32) | NA (0/32) | 0.03125 (26/32) |
| `nonmanifold_edges` | 3.34375 (32/32) | 2.53125 (32/32) | 2.6875 (32/32) |
| `C_F_nonmanifold_edge_count` | 0 (32/32) | 0 (32/32) | 0 (32/32) |
| `EXACT_COORD_face_components` | 36.8125 (32/32) | 40 (32/32) | 37.875 (32/32) |
| `EXACT_COORD_same_direction_two_face_edges` | 0 (32/32) | 0 (32/32) | 0 (32/32) |
| `free_total_surface_area` | 38.37144 (32/32) | 27.49873 (32/32) | 32.02184 (32/32) |
| `retained_seam_faces` | NA (0/32) | NA (0/32) | 3.90625 (26/32) |
| `retained_bulk_faces` | NA (0/32) | NA (0/32) | 79.59375 (26/32) |

## 使用边界

- `parent_equal_all124.csv`、`part_strata_all124.csv`、`per_parent_all124.csv`保留原值及分母；原表哈希见manifest。
- `paired_summary_deltas.csv`、`paired_parent_deltas.csv`分别是同checkpoint的SEAM−Native，不混用不同训练预算。
- `denominators.json`披露闭合C、seam适用状态、C精确保持、CF/FF面对分母；其请求计数只是描述性计数。
- `deletion_reason_counts.json`来自原删面账本，不根据面数差猜删面原因；删面降低不自动代表完整主体保住。
- 本入口不按DEV挑checkpoint、不晋级默认、不将来源唯一款式作为成功标准。
