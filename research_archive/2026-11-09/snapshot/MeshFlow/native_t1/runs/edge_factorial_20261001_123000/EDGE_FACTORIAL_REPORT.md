# Edge-only fixed-coefficient factorial report

This run adds one Edge-only +500-update arm from G0 cumulative2500 and reuses the L0/L5/JEdge5 cumulative3000 endpoints. All registered outputs in this noise batch are freshly generated. The default model is unchanged.

**1. Does Edge-only provide an independent complete-generation gain?**
有局部均值增量，但没有稳定的全面收益。旧片上 Edge-only 相对 L0 的剩余 RMS 从 1.5161%L 降到 1.4313%L（相对下降 5.59%），Gamma p95 从 3.7488%L 降到 3.6077%L；但 RMS 仅 11/24 配对改善，配对差中位数反而 +0.01233 个百分点，FF 平均增加 5.583 对。新片上 RMS 从 2.3092%L 升到 2.5648%L，Gamma p95 从 4.5834%L 升到 4.7587%L，FF 增加 5.8125 对。Teacher D_edge 比 L0 下降 30.73%，只能证明固定训练状态上的目标下降，不能替代完整生成收益。

**2. Is L5 still necessary on the Edge background?**
L5 在 Edge 背景上有独立的 FF 质量价值，但没有证明它是普遍必要的组成。JEdge5 相对 Edge-only：旧片 FF 平均减少 7.375 对（21/24 改善），新片减少 7.125 对（9/16 改善），四次留一种子分析的均值都下降。代价是新片剩余 RMS 从 2.5648%L 升到 2.5827%L，Gamma p95 从 4.7587%L 升到 5.0893%L；旧片 CF 平均增加 0.6667 对。两者体现取舍，不能以 FF 一列宣称 L5 全面必要，也不能据此自动切换默认模型。

**3. Do joint gains extend to previously unobserved context sets?**
没有在新上下文集合上复现综合正收益。新片 JEdge5 对 L0 的剩余 RMS 为 2.5827%L 对 2.3092%L，Gamma p95 为 5.0893%L 对 4.5834%L，分别相对恶化约 11.85% 和 11.04%；移除任一种子后，两项均值仍更差。旧片的对应均值则从 1.5161%L 降到 1.3881%L、从 3.7488%L 降到 3.2443%L。新 patch3 与历史上下文重合 11/12，几何新颖性有限；较低重合的 patch4 单独看，JEdge5 对 L0 的剩余 RMS 差 +0.05663 个百分点、仅 2/8 改善，Gamma p95 差 -0.50447 个百分点、5/8 改善。P1 patch4 有局部收益，P0 patch4 补全仍差。所有结论只针对两个见过的父对象。

**4. Do surface proximity and free-edge proximity agree?**
并不完全一致。相同查询点下的边距离不小于面距离检查全部通过，但严格覆盖可出现反向变化：旧片 Edge-only 对 L0 的 Gamma 表面 coverage(<0.1%L) 从 0.28622 降到 0.26048，边缘 coverage 从 0.13443 升到 0.16439。新片 JEdge5 对 Edge-only 的两种严格覆盖都提高（表面 0.15427→0.19133，边缘 0.07127→0.10617），两种 p95 却同时变差（表面 4.7587→5.0893%L，边缘 4.7609→5.0906%L）。说明部分区段更近、尾部更远，不能用覆盖比例代替整条接口的精度；160 条输出的精确 C/free 共边仍为 0。

**5. Do P0/patch2 failure and seed concentration persist?**
P0/patch2 失败仍存在。JEdge5 对 Edge-only 的剩余 RMS 平均恶化 0.60632 个百分点、Gamma p95 恶化 0.64727 个百分点，两项均 4/4 更差；同时 FF 减少 5 对但 CF 增加 3.5 对，也均为 4/4，属于明确的质量转移。种子集中性仍影响旧片的补全增量：去掉 10202 后，JEdge5 相对 Edge-only 的旧片 RMS 均值差由 -0.04329 变为 +0.04781，JEdge5 相对 L5 的差由 -0.15473 变为 +0.04023。反过来，新片 JEdge5 对 L0 的整体退化在四次留一种子分析中均存在，不能归因于单一坏 seed。

**6. What bias does five-point quadrature show, and is a separate follow-up warranted?**
五点求积对短程共线端点错位有已验证的放大偏差：连续值为 delta^3/(3*ell*L^2)，五点值为 delta^2/(8*L^2)。当 delta/ell=0.001 时，loss 为连续值的 375 倍，方向导数为 250 倍。垂直平移保持 h^2/L^2；中点对称塌缩仍可出现正 loss、零梯度。19 个构型中有 4 个密集参考未达到预定加密收敛门槛，包括小位移情形，不能把 1025 点自动称为真值。值得下一轮单独研究求积的优化偏置，但本轮没有证据证明更密求积会改善生成，也没有更改生产 loss、epsilon 或系数。

The evidence concerns two previously trained parents and one training seed. New context sets were not used in fine-tuning or earlier recorded evaluation, but their objects and some constituent triangles were observed as training targets. No new-shape or training-seed generalization is established.

## Frozen matrix and assets

| Item | Value |
| --- | --- |
| Models | L0, Edge_only, L5, JEdge5 |
| Generation seeds | [10201, 10202, 10203, 10204] |
| Registered units | 10 |
| Formal outputs | 160 |
| Old/new context units | 6/4 |
| N/K/free and Euler | 112 / 12 / 100; 50 Euler steps |
| All-C states | 51 per rollout; exact preservation checked from raw |
| Git HEAD | 5ef35b421ad00ccd3a97867ce0e33e5f77f65b1c |
| Seed registration | PASS |
| Historical context audit scope | All discoverable input-named NPZ registrations in native_t1/runs, experiments/mf_h1_local0/outputs and output, plus pinned training archive and old evaluation bank. Only exact full-target/source/C identities are accepted. Snapshot duplicates and generated outputs are not selection inputs. |

| Identity | Path | Bytes | File SHA256 |
| --- | --- | --- | --- |
| initial_checkpoint | /mnt/e/MeshFlow/native_t1/checkpoints/tgeo_g0_cumulative2500.pt | 521693171 | 3a66659e86586f54d2ec30ed3d4a18275ab69ba0f8cba2866ef67b340c481c0e |
| inputs | /mnt/e/MeshFlow/output/t1_publish_20260930/chair_n112.npz | 32023 | 7059671d55f2f3be8777d13e02a62255d8a0c0c7c1c1c5b4f3b24357091ba6fc |
| evaluation_inputs | /mnt/e/MeshFlow/native_t1/runs/edge_factorial_20261001_123000/registration/evaluation_inputs.npz | 323579 | 2154881d77b946a2a18959503930fd1d405f88f0d1f56d64a8f0d86710ed01b4 |
| teacher_samples | /mnt/e/MeshFlow/native_t1/runs/l3_l5_interface_pilot_20261001_104600/teacher_samples.npz | 256594 | f7bb3e8d91c2d8be8b62a08e29ae09a355864252673cbee0d9ea7e2a6fe88836 |
| endpoint L0 | /mnt/e/MeshFlow/native_t1/checkpoints/ablations/fm_cumulative3000.pt | 521689571 | 0f22ce498ca2c27dc74b43890fb8aad20f0f3e7153aa74ee6d3c14ea445a2660 |
| endpoint L5 | /mnt/e/MeshFlow/native_t1/checkpoints/ablations/surface_cumulative3000.pt | 521691211 | 93546de419582302e70cf140dae3aa44aad9ab913bd11b394482a8c347203999 |
| endpoint JEdge5 | /mnt/e/MeshFlow/native_t1/checkpoints/jedge5_cumulative3000.pt | 521691011 | cc7a248fa9a8570059f225cd1c313f21f08585925407a79418df6d63c9cae0e9 |
| endpoint Edge_only | /mnt/e/MeshFlow/native_t1/runs/edge_factorial_20261001_123000/Edge_only/additional_step500_cumulative_step3000.pt | 1565098955 | fcef320b370311c5893f5a0f6e4ea30d81b68e19ceda305116e9c2deb6e357ae |

Context novelty was registered by geometry and raw source-ID BFS before outputs. New patch3 in each parent differs from a historical set by only one face (11/12 overlap); it is a new face set with weak geometric novelty. Patch4 has substantially lower historical overlap. No patch was selected by generated quality.

| Unit | Group | Area/full area | Interface edges | Interface length | Connected components | Faces different from training C12 | Max historical overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0_patch0 | old | 0.3216469 | 12 | 18.43315 | 1 | 0 | 1 |
| P0_patch1 | old | 0.0827602 | 8 | 20.98786 | 1 | 10 | 1 |
| P0_patch2 | old | 0.09193655 | 6 | 8.044255 | 1 | 12 | 1 |
| P0_patch3 | new | 0.09098985 | 8 | 11.50447 | 1 | 12 | 0.9166667 |
| P0_patch4 | new | 0.07985154 | 6 | 13.68178 | 1 | 11 | 0.08333333 |
| P1_patch0 | old | 0.1948989 | 14 | 14.11988 | 1 | 0 | 1 |
| P1_patch1 | old | 0.100509 | 8 | 14.87041 | 1 | 11 | 1 |
| P1_patch2 | old | 0.1664539 | 6 | 8.657981 | 1 | 12 | 1 |
| P1_patch3 | new | 0.1647393 | 8 | 11.92567 | 1 | 12 | 0.9166667 |
| P1_patch4 | new | 0.1079586 | 6 | 6.593616 | 1 | 10 | 0.1666667 |

| Model | Generator state SHA256 | Cumulative updates |
| --- | --- | --- |
| L0 | f006c49c8c70efa4bf2df6ca96431c844a25a1bdd38e2db4756115a37a23545b | 3000 |
| L5 | 3c596db36fd7d5836427906fedecd7485b7be3e62cc6f605dbcb25b62c0c6641 | 3000 |
| JEdge5 | b9c1df5326ad4815ec3c32ed20e08dc34ff6248ff05be2c412e051fcb728aae3 | 3000 |
| Edge_only | b9ed4bfd5916d9731daec5afd9732b8bdfff5dbd81749dfda4c7ec3e12f74c7a | 3000 |

Removed historical training-file paths were resolved through verified compact exports with identical tensors and input RNG. No approximate endpoint substitution or old-output reuse is implied.

## Complete generation: old and new contexts separately

Errors, p95 and collision counts are lower-is-better; coverage is higher-is-better. Area and exact-connectivity counts are descriptive. Each macro is the noise mean within a parent/patch cell, then the equal mean of those cells. Four noisy outputs are not four independent parent objects.

### OLD contexts

| Metric | L0 / M00 | Edge-only / M10 | L5 / M01 | JEdge5 / M11 |
| --- | --- | --- | --- | --- |
| Free symmetric RMS (%L) | 1.516135 | 1.431346 | 1.542791 | 1.388058 |
| Free to GT RMS (%L) | 1.560747 | 1.363915 | 1.489049 | 1.387466 |
| GT to free RMS (%L) | 1.302869 | 1.350321 | 1.418138 | 1.219677 |
| Fixed-GT coverage <1% L | 0.751709 | 0.7779134 | 0.7686361 | 0.7873535 |
| Gamma to surface RMS (%L) | 1.97591 | 1.934635 | 2.136296 | 1.768891 |
| Gamma to surface p95 (%L) | 3.748847 | 3.607692 | 3.923858 | 3.244336 |
| Gamma to surface coverage <0.1% L | 0.2862196 | 0.260478 | 0.2241074 | 0.2697791 |
| Gamma to free edges RMS (%L) | 2.019772 | 1.968046 | 2.173401 | 1.802332 |
| Gamma to free edges p95 (%L) | 3.752766 | 3.611818 | 3.940339 | 3.256585 |
| Gamma to free edges coverage <0.1% L | 0.1344286 | 0.164392 | 0.114274 | 0.1462602 |
| CF affected free-face fraction | 0.06833333 | 0.04916667 | 0.04708333 | 0.05541667 |
| CF proper pairs | 8.791667 | 6.875 | 6.458333 | 7.541667 |
| FF illegal pairs | 77.08333 | 82.66667 | 77.66667 | 75.29167 |
| Free min-angle <5deg fraction | 0.1808333 | 0.1758333 | 0.1720833 | 0.1691667 |
| Free q>10 fraction | 0.05208333 | 0.04791667 | 0.0475 | 0.04791667 |
| Free mean q | 3.721221 | 3.669355 | 4.421844 | 3.87579 |
| Free total area (model units squared) | 44.72615 | 44.08083 | 44.38695 | 44.122 |

Model means, medians and worst observed values for the two primary new-context outcomes (old cells retained as costs):

| Model | Metric | Mean | Median | Worst |
| --- | --- | --- | --- | --- |
| L0 | Free symmetric RMS (%L) | 1.516135 | 0.9005537 | 5.038459 @ P0_patch2/10202 |
| L0 | Gamma to surface p95 (%L) | 3.748847 | 2.226741 | 10.07965 @ P0_patch2/10202 |
| L0 | Gamma to free edges p95 (%L) | 3.752766 | 2.226746 | 10.07965 @ P0_patch2/10202 |
| Edge_only | Free symmetric RMS (%L) | 1.431346 | 0.9897972 | 5.368653 @ P0_patch2/10202 |
| Edge_only | Gamma to surface p95 (%L) | 3.607692 | 2.024534 | 10.59222 @ P0_patch2/10202 |
| Edge_only | Gamma to free edges p95 (%L) | 3.611818 | 2.025161 | 10.59222 @ P0_patch2/10202 |
| L5 | Free symmetric RMS (%L) | 1.542791 | 0.99614 | 6.317787 @ P0_patch1/10202 |
| L5 | Gamma to surface p95 (%L) | 3.923858 | 2.135203 | 13.05532 @ P0_patch1/10202 |
| L5 | Gamma to free edges p95 (%L) | 3.940339 | 2.135203 | 13.05532 @ P0_patch1/10202 |
| JEdge5 | Free symmetric RMS (%L) | 1.388058 | 0.7888068 | 6.45461 @ P0_patch2/10202 |
| JEdge5 | Gamma to surface p95 (%L) | 3.244336 | 1.379118 | 12.73242 @ P0_patch2/10202 |
| JEdge5 | Gamma to free edges p95 (%L) | 3.256585 | 1.379118 | 12.73242 @ P0_patch2/10202 |

All metrics, including topology inheritance, degeneracy, NaN/Inf, coplanar/unresolved pairs, and all five fixed coverages:

<details><summary>Complete scalar macro table</summary>

| Metric | L0 | Edge_only | L5 | JEdge5 |
| --- | --- | --- | --- | --- |
| CF_coplanar_pair_count | 0 | 0 | 0 | 0 |
| CF_proper_affected_free_fraction | 0.06833333 | 0.04916667 | 0.04708333 | 0.05541667 |
| CF_proper_pair_count | 8.791667 | 6.875 | 6.458333 | 7.541667 |
| CF_unresolved_or_degenerate_pair_count | 0 | 0 | 0 | 0 |
| C_current_winding_conflict_count | 7.833333 | 7.833333 | 7.833333 | 7.833333 |
| C_inherited_nonmanifold_vertex_count | 0.3333333 | 0.3333333 | 0.3333333 | 0.3333333 |
| C_surface_area_fraction | 0.1597009 | 0.1597009 | 0.1597009 | 0.1597009 |
| FF_coplanar_pair_count | 0 | 0 | 0 | 0 |
| FF_illegal_pair_count | 77.08333 | 82.66667 | 77.66667 | 75.29167 |
| FF_proper_pair_count | 77.08333 | 82.66667 | 77.66667 | 75.29167 |
| FF_unresolved_or_degenerate_pair_count | 0 | 0 | 0 | 0 |
| Gamma_surface_coverage_0.0005_fraction | 0.1584221 | 0.1504738 | 0.08638282 | 0.1217436 |
| Gamma_surface_coverage_0.001_fraction | 0.2862196 | 0.260478 | 0.2241074 | 0.2697791 |
| Gamma_surface_coverage_0.002_fraction | 0.4642376 | 0.4354733 | 0.4245164 | 0.4609787 |
| Gamma_surface_coverage_0.005_fraction | 0.6459096 | 0.6814409 | 0.6689064 | 0.6863608 |
| Gamma_surface_coverage_0.01_fraction | 0.7250637 | 0.7395961 | 0.7448236 | 0.7466997 |
| Gamma_to_free_edges_RMS_bbox_pct | 2.019772 | 1.968046 | 2.173401 | 1.802332 |
| Gamma_to_free_edges_coverage_0.0005_fraction | 0.05232072 | 0.06140803 | 0.03389112 | 0.03347724 |
| Gamma_to_free_edges_coverage_0.001_fraction | 0.1344286 | 0.164392 | 0.114274 | 0.1462602 |
| Gamma_to_free_edges_coverage_0.002_fraction | 0.3231054 | 0.3302678 | 0.3297179 | 0.3651742 |
| Gamma_to_free_edges_coverage_0.005_fraction | 0.5878393 | 0.6327485 | 0.6200364 | 0.6392482 |
| Gamma_to_free_edges_coverage_0.01_fraction | 0.6997389 | 0.7242633 | 0.7294315 | 0.7408023 |
| Gamma_to_free_edges_p95_bbox_pct | 3.752766 | 3.611818 | 3.940339 | 3.256585 |
| Gamma_to_free_edges_zero_length_edge_count | 0 | 0 | 0 | 0 |
| R12_to_free_RMS_bbox_pct | 1.302869 | 1.350321 | 1.418138 | 1.219677 |
| boundary_RMS_bbox_pct | 1.97591 | 1.934635 | 2.136296 | 1.768891 |
| boundary_coverage_fraction | 0.2862196 | 0.260478 | 0.2241074 | 0.2697791 |
| boundary_p95_bbox_pct | 3.748847 | 3.607692 | 3.923858 | 3.244336 |
| exact_C_free_shared_edge_count | 0 | 0 | 0 | 0 |
| exact_C_free_shared_vertex_count | 0 | 0 | 0 | 0 |
| exact_index_boundary_edge_count | 309 | 309 | 309 | 309 |
| exact_index_overincident_edge_count | 0 | 0 | 0 | 0 |
| fixed_R12_coverage_RMS_bbox_pct | 1.276307 | 1.300242 | 1.407789 | 1.209166 |
| fixed_R12_coverage_within_1pct_fraction | 0.751709 | 0.7779134 | 0.7686361 | 0.7873535 |
| free_angle_p05_degrees | 4.456541 | 4.551042 | 4.68184 | 4.716756 |
| free_exact_zero_area_count | 0 | 0 | 0 | 0 |
| free_minimum_angle_lt5_fraction | 0.1808333 | 0.1758333 | 0.1720833 | 0.1691667 |
| free_near_degenerate_count | 0 | 0 | 0 | 0 |
| free_q_gt10_fraction | 0.05208333 | 0.04791667 | 0.0475 | 0.04791667 |
| free_q_mean | 3.721221 | 3.669355 | 4.421844 | 3.87579 |
| free_related_current_winding_conflict_count | 0 | 0 | 0 | 0 |
| free_to_R12_RMS_bbox_pct | 1.560747 | 1.363915 | 1.489049 | 1.387466 |
| free_total_surface_area | 44.72615 | 44.08083 | 44.38695 | 44.122 |
| inf_coordinate_count | 0 | 0 | 0 | 0 |
| nan_coordinate_count | 0 | 0 | 0 | 0 |
| near_C_area_fraction | 0.007486979 | 0.005533854 | 0.005859375 | 0.003173828 |
| near_C_boundary_band_area_fraction | 0.001139323 | 0.0008951823 | 0.0008138021 | 0.0006510417 |
| near_C_outside_boundary_band_area_fraction | 0.006347656 | 0.004638672 | 0.005045573 | 0.002522786 |
| new_free_related_nonmanifold_vertex_count | 0 | 0 | 0 | 0 |
| remainder_symmetric_RMS_bbox_pct | 1.516135 | 1.431346 | 1.542791 | 1.388058 |
| shared_edge_component_count | 101 | 101 | 101 | 101 |
| whole_GT_to_generated_RMS_bbox_pct | 1.130621 | 1.152058 | 1.208435 | 1.061617 |
| whole_generated_to_GT_RMS_bbox_pct | 0.9002332 | 0.8122899 | 0.8862993 | 0.7758779 |
| whole_symmetric_RMS_bbox_pct | 1.029673 | 1.009359 | 1.07102 | 0.9409977 |

</details>

### NEW contexts

| Metric | L0 / M00 | Edge-only / M10 | L5 / M01 | JEdge5 / M11 |
| --- | --- | --- | --- | --- |
| Free symmetric RMS (%L) | 2.309158 | 2.564774 | 2.63505 | 2.582696 |
| Free to GT RMS (%L) | 2.272841 | 2.508705 | 2.611833 | 2.47208 |
| GT to free RMS (%L) | 2.137251 | 2.367435 | 2.413885 | 2.460467 |
| Fixed-GT coverage <1% L | 0.6435547 | 0.6248779 | 0.6104736 | 0.6289062 |
| Gamma to surface RMS (%L) | 2.717423 | 2.88937 | 2.958896 | 3.024283 |
| Gamma to surface p95 (%L) | 4.583399 | 4.758668 | 4.771748 | 5.089309 |
| Gamma to surface coverage <0.1% L | 0.1082412 | 0.1542738 | 0.1205865 | 0.1913297 |
| Gamma to free edges RMS (%L) | 2.776685 | 2.937288 | 3.01505 | 3.077538 |
| Gamma to free edges p95 (%L) | 4.584544 | 4.76092 | 4.784565 | 5.090571 |
| Gamma to free edges coverage <0.1% L | 0.0240556 | 0.07127371 | 0.04409245 | 0.1061715 |
| CF affected free-face fraction | 0.038125 | 0.034375 | 0.031875 | 0.0325 |
| CF proper pairs | 6.875 | 6.25 | 5.625 | 5.6875 |
| FF illegal pairs | 87.0625 | 92.875 | 92.4375 | 85.75 |
| Free min-angle <5deg fraction | 0.188125 | 0.181875 | 0.18125 | 0.17 |
| Free q>10 fraction | 0.060625 | 0.06125 | 0.068125 | 0.058125 |
| Free mean q | 4.021416 | 5.588012 | 3.874574 | 6.311632 |
| Free total area (model units squared) | 45.1978 | 42.92912 | 43.2689 | 42.56675 |

Model means, medians and worst observed values for the two primary new-context outcomes (old cells retained as costs):

| Model | Metric | Mean | Median | Worst |
| --- | --- | --- | --- | --- |
| L0 | Free symmetric RMS (%L) | 2.309158 | 1.610471 | 6.560958 @ P0_patch4/10203 |
| L0 | Gamma to surface p95 (%L) | 4.583399 | 2.629045 | 12.33722 @ P0_patch4/10204 |
| L0 | Gamma to free edges p95 (%L) | 4.584544 | 2.629045 | 12.33722 @ P0_patch4/10204 |
| Edge_only | Free symmetric RMS (%L) | 2.564774 | 1.726518 | 6.638825 @ P0_patch4/10203 |
| Edge_only | Gamma to surface p95 (%L) | 4.758668 | 2.584004 | 12.22282 @ P0_patch4/10203 |
| Edge_only | Gamma to free edges p95 (%L) | 4.76092 | 2.585018 | 12.22282 @ P0_patch4/10203 |
| L5 | Free symmetric RMS (%L) | 2.63505 | 1.798692 | 6.500386 @ P0_patch4/10203 |
| L5 | Gamma to surface p95 (%L) | 4.771748 | 2.553593 | 12.0185 @ P0_patch4/10203 |
| L5 | Gamma to free edges p95 (%L) | 4.784565 | 2.553593 | 12.0185 @ P0_patch4/10203 |
| JEdge5 | Free symmetric RMS (%L) | 2.582696 | 1.722942 | 6.713297 @ P0_patch4/10202 |
| JEdge5 | Gamma to surface p95 (%L) | 5.089309 | 4.1062 | 12.1357 @ P0_patch4/10202 |
| JEdge5 | Gamma to free edges p95 (%L) | 5.090571 | 4.106661 | 12.1357 @ P0_patch4/10202 |

All metrics, including topology inheritance, degeneracy, NaN/Inf, coplanar/unresolved pairs, and all five fixed coverages:

<details><summary>Complete scalar macro table</summary>

| Metric | L0 | Edge_only | L5 | JEdge5 |
| --- | --- | --- | --- | --- |
| CF_coplanar_pair_count | 0 | 0 | 0 | 0 |
| CF_proper_affected_free_fraction | 0.038125 | 0.034375 | 0.031875 | 0.0325 |
| CF_proper_pair_count | 6.875 | 6.25 | 5.625 | 5.6875 |
| CF_unresolved_or_degenerate_pair_count | 0 | 0 | 0 | 0 |
| C_current_winding_conflict_count | 9.25 | 9.25 | 9.25 | 9.25 |
| C_inherited_nonmanifold_vertex_count | 0 | 0 | 0 | 0 |
| C_surface_area_fraction | 0.1108848 | 0.1108848 | 0.1108848 | 0.1108848 |
| FF_coplanar_pair_count | 0 | 0 | 0 | 0 |
| FF_illegal_pair_count | 87.0625 | 92.875 | 92.4375 | 85.75 |
| FF_proper_pair_count | 87.0625 | 92.875 | 92.4375 | 85.75 |
| FF_unresolved_or_degenerate_pair_count | 0 | 0 | 0 | 0 |
| Gamma_surface_coverage_0.0005_fraction | 0.05919689 | 0.0690104 | 0.05884588 | 0.06775548 |
| Gamma_surface_coverage_0.001_fraction | 0.1082412 | 0.1542738 | 0.1205865 | 0.1913297 |
| Gamma_surface_coverage_0.002_fraction | 0.251571 | 0.3109888 | 0.2252896 | 0.310522 |
| Gamma_surface_coverage_0.005_fraction | 0.5141954 | 0.5703125 | 0.5239152 | 0.5330796 |
| Gamma_surface_coverage_0.01_fraction | 0.6376309 | 0.651509 | 0.6325993 | 0.6619478 |
| Gamma_to_free_edges_RMS_bbox_pct | 2.776685 | 2.937288 | 3.01505 | 3.077538 |
| Gamma_to_free_edges_coverage_0.0005_fraction | 0.005889604 | 0.0162409 | 0.01405226 | 0.01502263 |
| Gamma_to_free_edges_coverage_0.001_fraction | 0.0240556 | 0.07127371 | 0.04409245 | 0.1061715 |
| Gamma_to_free_edges_coverage_0.002_fraction | 0.1391772 | 0.2273519 | 0.1387749 | 0.2218902 |
| Gamma_to_free_edges_coverage_0.005_fraction | 0.4314782 | 0.4751973 | 0.423304 | 0.4540016 |
| Gamma_to_free_edges_coverage_0.01_fraction | 0.5858602 | 0.5970307 | 0.5727177 | 0.6020535 |
| Gamma_to_free_edges_p95_bbox_pct | 4.584544 | 4.76092 | 4.784565 | 5.090571 |
| Gamma_to_free_edges_zero_length_edge_count | 0 | 0 | 0 | 0 |
| R12_to_free_RMS_bbox_pct | 2.137251 | 2.367435 | 2.413885 | 2.460467 |
| boundary_RMS_bbox_pct | 2.717423 | 2.88937 | 2.958896 | 3.024283 |
| boundary_coverage_fraction | 0.1082412 | 0.1542738 | 0.1205865 | 0.1913297 |
| boundary_p95_bbox_pct | 4.583399 | 4.758668 | 4.771748 | 5.089309 |
| exact_C_free_shared_edge_count | 0 | 0 | 0 | 0 |
| exact_C_free_shared_vertex_count | 0 | 0 | 0 | 0 |
| exact_index_boundary_edge_count | 307 | 307 | 307 | 307 |
| exact_index_overincident_edge_count | 0 | 0 | 0 | 0 |
| fixed_R12_coverage_RMS_bbox_pct | 2.110255 | 2.334289 | 2.353624 | 2.410736 |
| fixed_R12_coverage_within_1pct_fraction | 0.6435547 | 0.6248779 | 0.6104736 | 0.6289062 |
| free_angle_p05_degrees | 4.448012 | 4.466422 | 4.428554 | 4.61573 |
| free_exact_zero_area_count | 0 | 0 | 0 | 0 |
| free_minimum_angle_lt5_fraction | 0.188125 | 0.181875 | 0.18125 | 0.17 |
| free_near_degenerate_count | 0 | 0 | 0 | 0 |
| free_q_gt10_fraction | 0.060625 | 0.06125 | 0.068125 | 0.058125 |
| free_q_mean | 4.021416 | 5.588012 | 3.874574 | 6.311632 |
| free_related_current_winding_conflict_count | 0 | 0 | 0 | 0 |
| free_to_R12_RMS_bbox_pct | 2.272841 | 2.508705 | 2.611833 | 2.47208 |
| free_total_surface_area | 45.1978 | 42.92912 | 43.2689 | 42.56675 |
| inf_coordinate_count | 0 | 0 | 0 | 0 |
| nan_coordinate_count | 0 | 0 | 0 | 0 |
| near_C_area_fraction | 0.008300781 | 0.009399414 | 0.009765625 | 0.009521484 |
| near_C_boundary_band_area_fraction | 0.0002441406 | 0 | 0.0001220703 | 0.0002441406 |
| near_C_outside_boundary_band_area_fraction | 0.008056641 | 0.009399414 | 0.009643555 | 0.009277344 |
| new_free_related_nonmanifold_vertex_count | 0 | 0 | 0 | 0 |
| remainder_symmetric_RMS_bbox_pct | 2.309158 | 2.564774 | 2.63505 | 2.582696 |
| shared_edge_component_count | 101 | 101 | 101 | 101 |
| whole_GT_to_generated_RMS_bbox_pct | 1.944577 | 2.141535 | 2.183286 | 2.204626 |
| whole_generated_to_GT_RMS_bbox_pct | 1.45403 | 1.564156 | 1.702275 | 1.655034 |
| whole_symmetric_RMS_bbox_pct | 1.722904 | 1.885187 | 1.964473 | 1.958169 |

</details>

## Matched four-cell effects

Edge_off=M10-M00; L5_off=M01-M00; Edge_on=M11-M01; L5_on=M11-M10; I=M11-M10-M01+M00. Each difference is formed for the same parent/patch/seed before means or medians. A paired median is not a difference of model medians. I is fixed-coefficient non-additivity; unequal auxiliary exposure prevents a pure-mechanism synergy claim.

### OLD paired effects

**Edge_off**

| Metric | Mean delta | Paired median | Better/outputs | Worst delta |
| --- | --- | --- | --- | --- |
| Free symmetric RMS (%L) | -0.08478993 | 0.01232925 | 11/24 | 0.6967054 @ P0_patch1/10201 |
| Free to GT RMS (%L) | -0.1968317 | -0.01107043 | 13/24 | 0.308892 @ P0_patch2/10202 |
| GT to free RMS (%L) | 0.04745205 | 0.01265046 | 11/24 | 1.166625 @ P0_patch1/10201 |
| Fixed-GT coverage <1% L | 0.02620443 | 0.001953125 | 13/24 | -0.06445312 @ P0_patch1/10201 |
| Gamma to surface RMS (%L) | -0.04127508 | 0.008793791 | 10/24 | 0.8775137 @ P1_patch0/10204 |
| Gamma to surface p95 (%L) | -0.1411553 | 0.01074077 | 8/24 | 1.930173 @ P0_patch1/10201 |
| Gamma to surface coverage <0.1% L | -0.02574157 | -0.0009641551 | 10/24 | -0.3240348 @ P0_patch2/10201 |
| Gamma to free edges RMS (%L) | -0.05172587 | -0.006658407 | 14/24 | 0.8669207 @ P1_patch0/10204 |
| Gamma to free edges p95 (%L) | -0.1409477 | 0.01074077 | 8/24 | 1.930173 @ P0_patch1/10201 |
| Gamma to free edges coverage <0.1% L | 0.02996343 | 4.104791e-05 | 12/24 | -0.07307447 @ P0_patch1/10204 |
| CF affected free-face fraction | -0.01916667 | -0.01 | 13/24 | 0.05 @ P1_patch2/10203 |
| CF proper pairs | -1.916667 | -1 | 14/24 | 9 @ P0_patch1/10204 |
| FF illegal pairs | 5.583333 | 4 | 8/24 | 33 @ P1_patch0/10204 |
| Free min-angle <5deg fraction | -0.005 | 0 | 9/24 | 0.05 @ P1_patch1/10204 |
| Free q>10 fraction | -0.004166667 | 0 | 6/24 | 0.04 @ P0_patch1/10203 |
| Free mean q | -0.05186597 | 0.005269746 | 12/24 | 0.8376376 @ P0_patch2/10204 |
| Free total area (model units squared) | -0.6453209 | -0.2537124 | DESCRIPTIVE | DESCRIPTIVE |

**L5_off**

| Metric | Mean delta | Paired median | Better/outputs | Worst delta |
| --- | --- | --- | --- | --- |
| Free symmetric RMS (%L) | 0.02665596 | 0.005204157 | 11/24 | 1.70354 @ P0_patch1/10202 |
| Free to GT RMS (%L) | -0.07169825 | 0.02049107 | 9/24 | 1.290106 @ P0_patch1/10202 |
| GT to free RMS (%L) | 0.115269 | 0.007403461 | 11/24 | 2.036251 @ P0_patch1/10202 |
| Fixed-GT coverage <1% L | 0.01692708 | -0.0009765625 | 11/24 | -0.1542969 @ P0_patch1/10202 |
| Gamma to surface RMS (%L) | 0.1603861 | 0.01489188 | 9/24 | 2.999366 @ P0_patch0/10202 |
| Gamma to surface p95 (%L) | 0.1750114 | 0.04569007 | 10/24 | 8.762633 @ P0_patch0/10202 |
| Gamma to surface coverage <0.1% L | -0.06211213 | -0.001922287 | 10/24 | -0.5227114 @ P0_patch0/10202 |
| Gamma to free edges RMS (%L) | 0.1536293 | 0.01007338 | 11/24 | 2.960516 @ P0_patch0/10202 |
| Gamma to free edges p95 (%L) | 0.1875728 | 0.04568523 | 10/24 | 8.683634 @ P0_patch0/10202 |
| Gamma to free edges coverage <0.1% L | -0.02015464 | -0.0009618424 | 9/24 | -0.4476848 @ P0_patch0/10202 |
| CF affected free-face fraction | -0.02125 | -0.01 | 15/24 | 0.02 @ P1_patch2/10203 |
| CF proper pairs | -2.333333 | -1.5 | 16/24 | 4 @ P1_patch1/10202 |
| FF illegal pairs | 0.5833333 | -1 | 12/24 | 31 @ P1_patch0/10203 |
| Free min-angle <5deg fraction | -0.00875 | 0 | 8/24 | 0.03 @ P0_patch0/10202 |
| Free q>10 fraction | -0.004583333 | 0 | 8/24 | 0.03 @ P0_patch2/10202 |
| Free mean q | 0.7006229 | -0.006076252 | 13/24 | 18.13729 @ P0_patch2/10204 |
| Free total area (model units squared) | -0.3391981 | -0.2454286 | DESCRIPTIVE | DESCRIPTIVE |

**Edge_on**

| Metric | Mean delta | Paired median | Better/outputs | Worst delta |
| --- | --- | --- | --- | --- |
| Free symmetric RMS (%L) | -0.1547333 | 0.002565554 | 12/24 | 1.017439 @ P0_patch2/10202 |
| Free to GT RMS (%L) | -0.1015829 | -0.003535328 | 13/24 | 0.7973992 @ P0_patch2/10201 |
| GT to free RMS (%L) | -0.198461 | -0.02380185 | 14/24 | 1.198292 @ P0_patch2/10202 |
| Fixed-GT coverage <1% L | 0.01871745 | 0.00390625 | 13/24 | -0.09179688 @ P0_patch2/10202 |
| Gamma to surface RMS (%L) | -0.3674052 | -0.0267716 | 16/24 | 1.567535 @ P0_patch2/10202 |
| Gamma to surface p95 (%L) | -0.6795222 | -0.02781213 | 18/24 | 1.818954 @ P0_patch2/10202 |
| Gamma to surface coverage <0.1% L | 0.04567168 | 0.00861912 | 13/24 | -0.3919967 @ P1_patch2/10203 |
| Gamma to free edges RMS (%L) | -0.371069 | -0.04423276 | 16/24 | 1.567497 @ P0_patch2/10202 |
| Gamma to free edges p95 (%L) | -0.6837539 | -0.03004392 | 18/24 | 1.818954 @ P0_patch2/10202 |
| Gamma to free edges coverage <0.1% L | 0.03198624 | 0 | 10/24 | -0.4670848 @ P1_patch0/10204 |
| CF affected free-face fraction | 0.008333333 | 0 | 8/24 | 0.1 @ P0_patch0/10204 |
| CF proper pairs | 1.083333 | 0 | 9/24 | 9 @ P0_patch0/10204 |
| FF illegal pairs | -2.375 | -4 | 14/24 | 18 @ P0_patch1/10201 |
| Free min-angle <5deg fraction | -0.002916667 | 0 | 10/24 | 0.05 @ P1_patch2/10204 |
| Free q>10 fraction | 0.0004166667 | 0 | 6/24 | 0.05 @ P0_patch1/10203 |
| Free mean q | -0.5460538 | 0.0263496 | 11/24 | 4.47364 @ P1_patch0/10201 |
| Free total area (model units squared) | -0.264951 | -0.3346451 | DESCRIPTIVE | DESCRIPTIVE |

**L5_on**

| Metric | Mean delta | Paired median | Better/outputs | Worst delta |
| --- | --- | --- | --- | --- |
| Free symmetric RMS (%L) | -0.04328741 | 0.004982145 | 11/24 | 1.085957 @ P0_patch2/10202 |
| Free to GT RMS (%L) | 0.02355052 | 0.0114278 | 10/24 | 1.569889 @ P0_patch2/10201 |
| GT to free RMS (%L) | -0.1306441 | -0.01556244 | 15/24 | 1.399907 @ P0_patch2/10202 |
| Fixed-GT coverage <1% L | 0.009440104 | 0.001953125 | 14/24 | -0.1015625 @ P0_patch2/10202 |
| Gamma to surface RMS (%L) | -0.165744 | -0.009126996 | 13/24 | 1.900347 @ P0_patch2/10202 |
| Gamma to surface p95 (%L) | -0.3633556 | -0.04205929 | 13/24 | 2.140197 @ P0_patch2/10202 |
| Gamma to surface coverage <0.1% L | 0.009301122 | 0 | 11/24 | -0.4735125 @ P0_patch0/10201 |
| Gamma to free edges RMS (%L) | -0.1657138 | -0.007481953 | 14/24 | 1.900347 @ P0_patch2/10202 |
| Gamma to free edges p95 (%L) | -0.3552334 | -0.03548106 | 13/24 | 2.140197 @ P0_patch2/10202 |
| Gamma to free edges coverage <0.1% L | -0.01813182 | -0.001193064 | 7/24 | -0.3436284 @ P0_patch0/10201 |
| CF affected free-face fraction | 0.00625 | 0 | 11/24 | 0.17 @ P0_patch0/10204 |
| CF proper pairs | 0.6666667 | 0 | 9/24 | 17 @ P0_patch0/10204 |
| FF illegal pairs | -7.375 | -8 | 21/24 | 13 @ P0_patch0/10202 |
| Free min-angle <5deg fraction | -0.006666667 | 0 | 9/24 | 0.06 @ P1_patch1/10202 |
| Free q>10 fraction | -8.673617e-19 | 0 | 8/24 | 0.04 @ P1_patch1/10202 |
| Free mean q | 0.206435 | 0.001804146 | 12/24 | 4.33672 @ P1_patch0/10201 |
| Free total area (model units squared) | 0.04117186 | -0.1429732 | DESCRIPTIVE | DESCRIPTIVE |

**I**

| Metric | Mean delta | Paired median | Better/outputs | Worst delta |
| --- | --- | --- | --- | --- |
| Free symmetric RMS (%L) | -0.06994337 | 0.0143137 | 10/24 | 1.46504 @ P0_patch2/10204 |
| Free to GT RMS (%L) | 0.09524877 | 0.0102494 | 11/24 | 1.341317 @ P0_patch2/10204 |
| GT to free RMS (%L) | -0.245913 | -0.002179667 | 12/24 | 0.8458505 @ P0_patch2/10202 |
| Fixed-GT coverage <1% L | -0.007486979 | 0.001953125 | 13/24 | -0.3457031 @ P1_patch2/10202 |
| Gamma to surface RMS (%L) | -0.3261301 | 0.005460374 | 11/24 | 1.384237 @ P1_patch2/10202 |
| Gamma to surface p95 (%L) | -0.5383669 | -0.02998519 | 14/24 | 2.54273 @ P1_patch2/10202 |
| Gamma to surface coverage <0.1% L | 0.07141325 | 0.008639452 | 13/24 | -0.6721062 @ P1_patch2/10203 |
| Gamma to free edges RMS (%L) | -0.3193431 | 0.004613972 | 12/24 | 1.444776 @ P1_patch2/10202 |
| Gamma to free edges p95 (%L) | -0.5428062 | -0.05680866 | 14/24 | 2.33661 @ P1_patch2/10202 |
| Gamma to free edges coverage <0.1% L | 0.002022814 | 0.0006964206 | 12/24 | -0.7178044 @ P1_patch0/10204 |
| CF affected free-face fraction | 0.0275 | 0.005 | 9/24 | 0.26 @ P0_patch0/10204 |
| CF proper pairs | 3 | 0.5 | 10/24 | 28 @ P0_patch0/10204 |
| FF illegal pairs | -7.958333 | -4 | 15/24 | 20 @ P0_patch0/10202 |
| Free min-angle <5deg fraction | 0.002083333 | -0.01 | 14/24 | 0.15 @ P1_patch1/10202 |
| Free q>10 fraction | 0.004583333 | 0 | 7/24 | 0.12 @ P1_patch1/10202 |
| Free mean q | -0.4941879 | 0.04457105 | 11/24 | 4.229985 @ P1_patch0/10201 |
| Free total area (model units squared) | 0.3803699 | 0.08908342 | DESCRIPTIVE | DESCRIPTIVE |

Parent means remain separate within each old/new context group:

| Effect | Parent | Free RMS | Surface p95 | Edge p95 | Strict surface coverage | Strict edge coverage | FF pairs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Edge_off | 0 | 0.01257749 | 0.04979683 | 0.04232107 | -0.09120421 | 0.02125225 | -1 |
| Edge_off | 1 | -0.1821574 | -0.3321073 | -0.3242165 | 0.03972106 | 0.03867461 | 12.16667 |
| L5_off | 0 | 0.2167027 | 0.9067235 | 0.9013019 | -0.1460957 | -0.06745654 | -4.666667 |
| L5_off | 1 | -0.1633908 | -0.5567008 | -0.5261563 | 0.02187142 | 0.02714727 | 5.833333 |
| Edge_on | 0 | -0.2799641 | -1.321431 | -1.308128 | 0.1247629 | 0.1064692 | 0.5833333 |
| Edge_on | 1 | -0.0295025 | -0.03761315 | -0.05938 | -0.03341958 | -0.04249675 | -5.333333 |
| L5_on | 0 | -0.07583885 | -0.4645045 | -0.4491469 | 0.06987146 | 0.01776044 | -3.083333 |
| L5_on | 1 | -0.01073596 | -0.2622066 | -0.2613198 | -0.05126922 | -0.05402409 | -11.66667 |
| I | 0 | -0.2925416 | -1.371228 | -1.350449 | 0.2159671 | 0.08521699 | 1.583333 |
| I | 1 | 0.1526549 | 0.2944942 | 0.2648365 | -0.07314063 | -0.08117136 | -17.5 |

Cell-level paired means/medians and improvement counts (including P0/patch2):

| Effect | Cell | RMS mean/median/wins | Surface p95 mean/median/wins | Edge p95 mean/median/wins | FF mean/wins | Strict edge coverage delta |
| --- | --- | --- | --- | --- | --- | --- |
| Edge_off | P0_patch0 | 0.01321484 / 0.01240014 / 2/4 | 0.02994264 / 0.03901955 / 1/4 | 0.01062986 / 0.01904004 / 1/4 | -3.75 / 3/4 | 0.07576963 |
| Edge_off | P0_patch1 | 0.2984514 / 0.2119036 / 0/4 | 0.6791453 / 0.3610002 / 0/4 | 0.6757216 / 0.3536547 / 0/4 | -3 / 2/4 | -0.02018251 |
| Edge_off | P0_patch2 | -0.2739338 / -0.05293313 / 2/4 | -0.5596975 / -0.2292836 / 2/4 | -0.5593882 / -0.2286651 / 2/4 | 3.75 / 1/4 | 0.008169615 |
| Edge_off | P1_patch0 | -0.02038875 / 0.01232925 / 1/4 | 0.3066501 / 0.1905942 / 0/4 | 0.3067779 / 0.1905942 / 0/4 | 19.25 / 0/4 | 0.07561909 |
| Edge_off | P1_patch1 | -0.2058818 / -0.0596542 / 2/4 | -0.668035 / -0.3848738 / 4/4 | -0.6444879 / -0.3377795 / 4/4 | 14.25 / 0/4 | 0.004376586 |
| Edge_off | P1_patch2 | -0.3202015 / -0.1980745 / 4/4 | -0.6349371 / 0.00557091 / 1/4 | -0.6349395 / 0.005566076 / 1/4 | 3 / 2/4 | 0.03602815 |
| L5_off | P0_patch0 | 0.2387088 / -0.01333266 / 3/4 | 2.186181 / 0.01132298 / 2/4 | 2.167395 / 0.01324994 / 2/4 | -6.75 / 4/4 | -0.2209982 |
| L5_off | P0_patch1 | 0.4923845 / 0.1158459 / 0/4 | 0.8391943 / 0.1379327 / 0/4 | 0.8372269 / 0.1341263 / 0/4 | -7.25 / 3/4 | 0.01392098 |
| L5_off | P0_patch2 | -0.08098504 / 0.04978606 / 2/4 | -0.3052046 / -0.133645 / 3/4 | -0.3007159 / -0.1246676 / 3/4 | 0 / 1/4 | 0.004707558 |
| L5_off | P1_patch0 | -0.07270747 / 0.008431732 / 1/4 | 0.08274995 / 0.06636136 / 0/4 | 0.08327906 / 0.06636136 / 0/4 | 13.75 / 1/4 | 0.1389973 |
| L5_off | P1_patch1 | -0.1684633 / -0.003887126 / 2/4 | -0.4046416 / -0.2480404 / 3/4 | -0.4046416 / -0.2480404 / 3/4 | 7.5 / 1/4 | -0.06726769 |
| L5_off | P1_patch2 | -0.2490017 / -0.07706203 / 3/4 | -1.348211 / -0.006260621 / 2/4 | -1.257106 / 0.02021761 / 2/4 | -3.75 / 2/4 | 0.009712155 |
| Edge_on | P0_patch0 | -0.2168511 / 0.02435105 / 1/4 | -2.225175 / -0.05888196 / 4/4 | -2.20231 / -0.0551553 / 4/4 | 5 / 1/4 | 0.2378797 |
| Edge_on | P0_patch1 | -1.036415 / -0.02796409 / 3/4 | -2.131893 / -0.08698853 / 3/4 | -2.123338 / -0.06987747 / 3/4 | -2 / 2/4 | 0.01719722 |
| Edge_on | P0_patch2 | 0.413374 / 0.3600632 / 1/4 | 0.3927749 / 0.1174098 / 1/4 | 0.401265 / 0.1343899 / 1/4 | -1.25 / 2/4 | 0.0643308 |
| Edge_on | P1_patch0 | -0.003384625 / -0.005023576 / 2/4 | -0.04245587 / -0.04152362 / 4/4 | -0.04298498 / -0.04152362 / 4/4 | -5.5 / 2/4 | -0.1893574 |
| Edge_on | P1_patch1 | -0.09560858 / -0.05130641 / 3/4 | -0.1637733 / -0.07579394 / 3/4 | -0.1637733 / -0.07579394 / 3/4 | -3.25 / 3/4 | 0.06725731 |
| Edge_on | P1_patch2 | 0.01048572 / 0.01995816 / 2/4 | 0.09338969 / -0.02781213 / 3/4 | 0.02861826 / -0.1308719 / 3/4 | -7.25 / 4/4 | -0.005390206 |
| L5_on | P0_patch0 | 0.008642846 / 0.007057461 / 1/4 | -0.06893682 / -0.08657853 / 3/4 | -0.04554535 / -0.04648178 / 3/4 | 2 / 2/4 | -0.05888812 |
| L5_on | P0_patch1 | -0.8424822 / -0.3305667 / 4/4 | -1.971844 / -1.044571 / 3/4 | -1.961833 / -1.024547 / 3/4 | -6.25 / 4/4 | 0.05130071 |
| L5_on | P0_patch2 | 0.6063228 / 0.6562814 / 0/4 | 0.6472677 / 0.2130485 / 0/4 | 0.6599372 / 0.2383875 / 0/4 | -5 / 4/4 | 0.06086875 |
| L5_on | P1_patch0 | -0.05570334 / -0.06330784 / 3/4 | -0.266356 / -0.1628198 / 3/4 | -0.2664839 / -0.1628198 / 3/4 | -11 / 3/4 | -0.1259791 |
| L5_on | P1_patch1 | -0.05819008 / -0.07654656 / 2/4 | 0.09962009 / -0.006683379 / 2/4 | 0.07607294 / -0.006683379 / 2/4 | -10 / 4/4 | -0.004386964 |
| L5_on | P1_patch2 | 0.08168553 / 0.1001921 / 1/4 | -0.619884 / -0.1511701 / 2/4 | -0.5935485 / -0.1511701 / 2/4 | -14 / 4/4 | -0.0317062 |
| I | P0_patch0 | -0.2300659 / 0.002856544 / 2/4 | -2.255118 / -0.09790151 / 3/4 | -2.21294 / -0.09504835 / 3/4 | 8.75 / 0/4 | 0.16211 |
| I | P0_patch1 | -1.334867 / -0.4129634 / 4/4 | -2.811039 / -1.092526 / 4/4 | -2.79906 / -1.06807 / 4/4 | 1 / 2/4 | 0.03737973 |
| I | P0_patch2 | 0.6873078 / 0.5065781 / 0/4 | 0.9524724 / 0.9616343 / 0/4 | 0.9606531 / 0.9610158 / 0/4 | -5 / 3/4 | 0.05616119 |
| I | P1_patch0 | 0.01700412 / -0.01407492 / 2/4 | -0.3491059 / -0.2291812 / 4/4 | -0.3497629 / -0.2291812 / 4/4 | -24.75 / 4/4 | -0.2649764 |
| I | P1_patch1 | 0.1102732 / 0.04590785 / 1/4 | 0.5042617 / 0.1789161 / 1/4 | 0.4807146 / 0.1789161 / 1/4 | -17.5 / 3/4 | 0.06288073 |
| I | P1_patch2 | 0.3306872 / 0.2890729 / 1/4 | 0.7283267 / 0.3199717 / 2/4 | 0.6635577 / 0.3199765 / 2/4 | -10.25 / 3/4 | -0.04141836 |

### NEW paired effects

**Edge_off**

| Metric | Mean delta | Paired median | Better/outputs | Worst delta |
| --- | --- | --- | --- | --- |
| Free symmetric RMS (%L) | 0.2556165 | -0.01245367 | 9/16 | 2.48475 @ P0_patch3/10203 |
| Free to GT RMS (%L) | 0.2358641 | 0.04641013 | 5/16 | 2.119231 @ P0_patch3/10203 |
| GT to free RMS (%L) | 0.2301839 | 0.008701204 | 7/16 | 2.821419 @ P0_patch3/10203 |
| Fixed-GT coverage <1% L | -0.01867676 | -0.00390625 | 7/16 | -0.7617188 @ P0_patch3/10203 |
| Gamma to surface RMS (%L) | 0.1719465 | 0.003102823 | 7/16 | 4.367165 @ P0_patch3/10203 |
| Gamma to surface p95 (%L) | 0.1752696 | 0.02193817 | 7/16 | 6.827599 @ P0_patch3/10203 |
| Gamma to surface coverage <0.1% L | 0.04603259 | 0.00877445 | 10/16 | -0.1304534 @ P1_patch3/10204 |
| Gamma to free edges RMS (%L) | 0.1606026 | 0.008335381 | 6/16 | 4.361639 @ P0_patch3/10203 |
| Gamma to free edges p95 (%L) | 0.1763755 | 0.03312172 | 7/16 | 6.834009 @ P0_patch3/10203 |
| Gamma to free edges coverage <0.1% L | 0.04721811 | 0.00389895 | 11/16 | -0.013561 @ P1_patch3/10204 |
| CF affected free-face fraction | -0.00375 | 0 | 6/16 | 0.02 @ P1_patch3/10203 |
| CF proper pairs | -0.625 | 0 | 5/16 | 5 @ P1_patch4/10204 |
| FF illegal pairs | 5.8125 | -1.5 | 9/16 | 87 @ P0_patch3/10203 |
| Free min-angle <5deg fraction | -0.00625 | 0 | 7/16 | 0.03 @ P1_patch4/10202 |
| Free q>10 fraction | 0.000625 | 0 | 6/16 | 0.1 @ P0_patch3/10203 |
| Free mean q | 1.566596 | -0.1061909 | 10/16 | 29.98184 @ P0_patch3/10204 |
| Free total area (model units squared) | -2.268679 | -1.024309 | DESCRIPTIVE | DESCRIPTIVE |

**L5_off**

| Metric | Mean delta | Paired median | Better/outputs | Worst delta |
| --- | --- | --- | --- | --- |
| Free symmetric RMS (%L) | 0.3258923 | 0.1043558 | 5/16 | 2.469617 @ P0_patch3/10203 |
| Free to GT RMS (%L) | 0.3389928 | 0.2001138 | 4/16 | 2.148674 @ P0_patch3/10203 |
| GT to free RMS (%L) | 0.2766341 | -0.02240047 | 10/16 | 2.769251 @ P0_patch3/10203 |
| Fixed-GT coverage <1% L | -0.03308105 | 0.002929688 | 9/16 | -0.7773438 @ P0_patch3/10203 |
| Gamma to surface RMS (%L) | 0.2414732 | -0.02655719 | 11/16 | 4.385439 @ P0_patch3/10203 |
| Gamma to surface p95 (%L) | 0.1883493 | -0.1412265 | 11/16 | 6.896504 @ P0_patch3/10203 |
| Gamma to surface coverage <0.1% L | 0.0123453 | 0 | 7/16 | -0.270503 @ P1_patch3/10204 |
| Gamma to free edges RMS (%L) | 0.2383651 | -0.01935326 | 11/16 | 4.38391 @ P0_patch3/10203 |
| Gamma to free edges p95 (%L) | 0.2000205 | -0.09001244 | 11/16 | 6.913839 @ P0_patch3/10203 |
| Gamma to free edges coverage <0.1% L | 0.02003685 | 0 | 6/16 | -0.03696568 @ P0_patch3/10204 |
| CF affected free-face fraction | -0.00625 | 0 | 5/16 | 0.02 @ P1_patch3/10203 |
| CF proper pairs | -1.25 | -0.5 | 8/16 | 4 @ P1_patch3/10203 |
| FF illegal pairs | 5.375 | 1 | 7/16 | 89 @ P0_patch3/10203 |
| Free min-angle <5deg fraction | -0.006875 | 0 | 4/16 | 0.04 @ P1_patch3/10202 |
| Free q>10 fraction | 0.0075 | 0.005 | 5/16 | 0.09 @ P0_patch3/10203 |
| Free mean q | -0.1468425 | -0.009871406 | 8/16 | 1.210553 @ P1_patch4/10202 |
| Free total area (model units squared) | -1.928903 | -0.2119249 | DESCRIPTIVE | DESCRIPTIVE |

**Edge_on**

| Metric | Mean delta | Paired median | Better/outputs | Worst delta |
| --- | --- | --- | --- | --- |
| Free symmetric RMS (%L) | -0.0523538 | 0.03156556 | 7/16 | 0.3917436 @ P0_patch3/10201 |
| Free to GT RMS (%L) | -0.1397535 | -0.0007299566 | 9/16 | 0.4198393 @ P0_patch4/10204 |
| GT to free RMS (%L) | 0.04658146 | 0.06586409 | 6/16 | 0.6493689 @ P0_patch3/10201 |
| Fixed-GT coverage <1% L | 0.01843262 | -0.0009765625 | 6/16 | -0.02734375 @ P0_patch3/10201 |
| Gamma to surface RMS (%L) | 0.06538628 | -0.003073391 | 9/16 | 2.106617 @ P1_patch3/10202 |
| Gamma to surface p95 (%L) | 0.3175606 | 0.04714292 | 6/16 | 6.786119 @ P1_patch3/10202 |
| Gamma to surface coverage <0.1% L | 0.07074315 | 0.005803787 | 9/16 | -0.2510272 @ P1_patch4/10203 |
| Gamma to free edges RMS (%L) | 0.06248754 | -0.007525258 | 9/16 | 2.016007 @ P1_patch3/10202 |
| Gamma to free edges p95 (%L) | 0.3060063 | 0.027012 | 7/16 | 6.786119 @ P1_patch3/10202 |
| Gamma to free edges coverage <0.1% L | 0.06207902 | 0.0009771928 | 9/16 | -0.2114661 @ P1_patch4/10203 |
| CF affected free-face fraction | 0.000625 | 0 | 5/16 | 0.06 @ P1_patch4/10202 |
| CF proper pairs | 0.0625 | 0 | 6/16 | 11 @ P1_patch4/10202 |
| FF illegal pairs | -6.6875 | -4.5 | 11/16 | 22 @ P0_patch4/10203 |
| Free min-angle <5deg fraction | -0.01125 | -0.005 | 8/16 | 0.09 @ P1_patch4/10204 |
| Free q>10 fraction | -0.01 | 0 | 6/16 | 0.03 @ P0_patch4/10203 |
| Free mean q | 2.437058 | -0.01513637 | 9/16 | 30.80119 @ P0_patch3/10204 |
| Free total area (model units squared) | -0.7021441 | -0.3057426 | DESCRIPTIVE | DESCRIPTIVE |

**L5_on**

| Metric | Mean delta | Paired median | Better/outputs | Worst delta |
| --- | --- | --- | --- | --- |
| Free symmetric RMS (%L) | 0.01792202 | 0.02197161 | 7/16 | 0.7542056 @ P1_patch4/10204 |
| Free to GT RMS (%L) | -0.03662486 | 0.0009867115 | 8/16 | 1.07464 @ P1_patch4/10204 |
| GT to free RMS (%L) | 0.09303158 | 0.0171313 | 7/16 | 0.9884051 @ P1_patch3/10202 |
| Fixed-GT coverage <1% L | 0.00402832 | 0.00390625 | 11/16 | -0.06835938 @ P1_patch3/10202 |
| Gamma to surface RMS (%L) | 0.134913 | 0.002888994 | 8/16 | 2.340324 @ P1_patch3/10202 |
| Gamma to surface p95 (%L) | 0.3306403 | 0.01822917 | 7/16 | 6.68618 @ P1_patch3/10202 |
| Gamma to surface coverage <0.1% L | 0.03705586 | 0.0009751 | 8/16 | -0.2510272 @ P1_patch4/10203 |
| Gamma to free edges RMS (%L) | 0.1402501 | 0.001134227 | 8/16 | 2.336008 @ P1_patch3/10202 |
| Gamma to free edges p95 (%L) | 0.3296512 | 0.007122696 | 8/16 | 6.68618 @ P1_patch3/10202 |
| Gamma to free edges coverage <0.1% L | 0.03489777 | 0 | 6/16 | -0.06997014 @ P1_patch3/10202 |
| CF affected free-face fraction | -0.001875 | 0 | 6/16 | 0.02 @ P1_patch4/10202 |
| CF proper pairs | -0.5625 | -0.5 | 8/16 | 7 @ P1_patch3/10202 |
| FF illegal pairs | -7.125 | -2.5 | 9/16 | 9 @ P0_patch4/10201 |
| Free min-angle <5deg fraction | -0.011875 | 0 | 7/16 | 0.04 @ P0_patch4/10203 |
| Free q>10 fraction | -0.003125 | 0 | 7/16 | 0.04 @ P1_patch3/10202 |
| Free mean q | 0.7236191 | 0.0009642839 | 7/16 | 8.309862 @ P1_patch3/10201 |
| Free total area (model units squared) | -0.362368 | -0.08540843 | DESCRIPTIVE | DESCRIPTIVE |

**I**

| Metric | Mean delta | Paired median | Better/outputs | Worst delta |
| --- | --- | --- | --- | --- |
| Free symmetric RMS (%L) | -0.3079703 | -0.01612102 | 9/16 | 0.5814825 @ P1_patch4/10204 |
| Free to GT RMS (%L) | -0.3756176 | -0.0777809 | 9/16 | 0.7815464 @ P1_patch4/10204 |
| GT to free RMS (%L) | -0.1836025 | 0.01740467 | 8/16 | 0.7357779 @ P1_patch3/10202 |
| Fixed-GT coverage <1% L | 0.03710938 | 0.002929688 | 10/16 | -0.1660156 @ P1_patch3/10202 |
| Gamma to surface RMS (%L) | -0.1065602 | 0.007691886 | 8/16 | 3.350168 @ P1_patch3/10202 |
| Gamma to surface p95 (%L) | 0.142291 | 0.01613242 | 8/16 | 9.514455 @ P1_patch3/10202 |
| Gamma to surface coverage <0.1% L | 0.02471056 | 0.01435898 | 10/16 | -0.2800619 @ P1_patch4/10203 |
| Gamma to free edges RMS (%L) | -0.09811507 | 0.02374859 | 7/16 | 3.299508 @ P1_patch3/10202 |
| Gamma to free edges p95 (%L) | 0.1296307 | 0.00180423 | 8/16 | 9.514455 @ P1_patch3/10202 |
| Gamma to free edges coverage <0.1% L | 0.01486092 | -0.0001374207 | 7/16 | -0.2291494 @ P1_patch4/10203 |
| CF affected free-face fraction | 0.004375 | 0 | 7/16 | 0.05 @ P0_patch3/10203 |
| CF proper pairs | 0.6875 | 0 | 7/16 | 10 @ P1_patch3/10202 |
| FF illegal pairs | -12.5 | 1 | 7/16 | 19 @ P1_patch4/10201 |
| Free min-angle <5deg fraction | -0.005 | -0.015 | 11/16 | 0.1 @ P1_patch4/10204 |
| Free q>10 fraction | -0.010625 | 0 | 7/16 | 0.03 @ P0_patch4/10202 |
| Free mean q | 0.8704616 | 0.225015 | 7/16 | 8.367439 @ P1_patch3/10201 |
| Free total area (model units squared) | 1.566535 | 0.3076719 | DESCRIPTIVE | DESCRIPTIVE |

Parent means remain separate within each old/new context group:

| Effect | Parent | Free RMS | Surface p95 | Edge p95 | Strict surface coverage | Strict edge coverage | FF pairs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Edge_off | 0 | 0.5542735 | 0.8403046 | 0.8411331 | 0.01987573 | 0.03420049 | 8.875 |
| Edge_off | 1 | -0.04304043 | -0.4897654 | -0.4883821 | 0.07218945 | 0.06023572 | 2.75 |
| L5_off | 0 | 0.4918333 | 0.90628 | 0.9125439 | -0.01724104 | -0.006813772 | 9 |
| L5_off | 1 | 0.1599514 | -0.5295814 | -0.512503 | 0.04193165 | 0.04688747 | 1.75 |
| Edge_on | 0 | 0.104896 | -0.006324668 | -0.01145162 | 0.0578394 | 0.06073723 | -1.875 |
| Edge_on | 1 | -0.2096036 | 0.6414459 | 0.6234641 | 0.08364691 | 0.06342081 | -11.5 |
| L5_on | 0 | 0.04245585 | 0.05965067 | 0.05995924 | 0.02072262 | 0.01972297 | -1.75 |
| L5_on | 1 | -0.006611806 | 0.6016299 | 0.5993432 | 0.05338911 | 0.05007256 | -12.5 |
| I | 0 | -0.4493775 | -0.8466293 | -0.8525847 | 0.03796367 | 0.02653675 | -10.75 |
| I | 1 | -0.1665632 | 1.131211 | 1.111846 | 0.01145746 | 0.003185091 | -14.25 |

Cell-level paired means/medians and improvement counts (including P0/patch2):

| Effect | Cell | RMS mean/median/wins | Surface p95 mean/median/wins | Edge p95 mean/median/wins | FF mean/wins | Strict edge coverage delta |
| --- | --- | --- | --- | --- | --- | --- |
| Edge_off | P0_patch3 | 1.067431 / 0.9315015 / 1/4 | 1.967175 / 0.9439918 / 2/4 | 1.968777 / 0.9439918 / 2/4 | 14.75 / 3/4 | 0.06693905 |
| Edge_off | P0_patch4 | 0.04111556 / 0.0350683 / 2/4 | -0.2865657 / 0.04807005 / 1/4 | -0.2865112 / 0.04817911 / 1/4 | 3 / 2/4 | 0.001461925 |
| Edge_off | P1_patch3 | -0.1150334 / -0.02253716 / 3/4 | -0.6201663 / -0.03536221 / 2/4 | -0.6230262 / -0.04108217 / 2/4 | -1.5 / 2/4 | 0.06117939 |
| Edge_off | P1_patch4 | 0.02895252 / -0.06043443 / 3/4 | -0.3593645 / -0.06932074 / 2/4 | -0.3537379 / -0.05809047 / 2/4 | 7 / 2/4 | 0.05929205 |
| L5_off | P0_patch3 | 1.004128 / 0.8702935 / 2/4 | 2.179041 / 1.117782 / 2/4 | 2.191048 / 1.117782 / 2/4 | 21.75 / 2/4 | -0.0111901 |
| L5_off | P0_patch4 | -0.02046126 / -0.03010577 / 2/4 | -0.3664811 / -0.06730293 / 2/4 | -0.3659601 / -0.06626095 / 2/4 | -3.75 / 2/4 | -0.002437442 |
| L5_off | P1_patch3 | 0.123884 / 0.1109292 / 0/4 | -0.7965449 / -0.1574581 / 4/4 | -0.7635814 / -0.106244 / 4/4 | -5.5 / 2/4 | 0.02586084 |
| L5_off | P1_patch4 | 0.1960187 / 0.2202211 / 1/4 | -0.2626179 / -0.09443031 / 3/4 | -0.2614246 / -0.0912086 / 3/4 | 9 / 1/4 | 0.0679141 |
| Edge_on | P0_patch3 | 0.04808308 / 0.03156556 / 1/4 | -0.1461892 / -0.2042074 / 2/4 | -0.1559145 / -0.2083113 / 2/4 | -11 / 3/4 | 0.1161904 |
| Edge_on | P0_patch4 | 0.1617089 / 0.1728358 / 1/4 | 0.1335399 / 0.1205541 / 2/4 | 0.1330113 / 0.1194969 / 2/4 | 7.25 / 2/4 | 0.0052841 |
| Edge_on | P1_patch3 | -0.1952021 / -0.1245203 / 3/4 | 1.796265 / 0.1756967 / 0/4 | 1.761698 / 0.1753165 / 1/4 | -14.75 / 4/4 | 0.1298041 |
| Edge_on | P1_patch4 | -0.2240051 / -0.1406765 / 2/4 | -0.5133733 / -0.2113871 / 2/4 | -0.5147694 / -0.2109255 / 2/4 | -8.25 / 2/4 | -0.002962524 |
| L5_on | P0_patch3 | -0.01522039 / -0.01048951 / 2/4 | 0.06567684 / 0.0631489 / 0/4 | 0.06635611 / 0.06355003 / 0/4 | -4 / 2/4 | 0.03806121 |
| L5_on | P0_patch4 | 0.1001321 / 0.01606834 / 2/4 | 0.05362451 / -0.04211135 / 3/4 | 0.05356238 / -0.04223559 / 3/4 | 0.5 / 2/4 | 0.001384733 |
| L5_on | P1_patch3 | 0.04371533 / 0.04950948 / 2/4 | 1.619887 / -0.01404551 / 2/4 | 1.621142 / -0.01404551 / 2/4 | -18.75 / 4/4 | 0.0944856 |
| L5_on | P1_patch4 | -0.05693894 / 0.2335372 / 1/4 | -0.4166267 / -0.006980518 / 2/4 | -0.4224561 / -0.01848358 / 3/4 | -6.25 / 1/4 | 0.005659523 |
| I | P0_patch3 | -1.019348 / -0.8669009 / 3/4 | -2.113364 / -1.054634 / 2/4 | -2.124692 / -1.054232 / 2/4 | -25.75 / 1/4 | 0.04925132 |
| I | P0_patch4 | 0.1205934 / 0.1502051 / 2/4 | 0.4201056 / 0.103758 / 2/4 | 0.4195225 / 0.1025918 / 2/4 | 4.25 / 1/4 | 0.003822175 |
| I | P1_patch3 | -0.08016868 / 0.03062839 / 2/4 | 2.416431 / 0.09219853 / 2/4 | 2.384724 / 0.09219853 / 2/4 | -13.25 / 3/4 | 0.06862476 |
| I | P1_patch4 | -0.2529577 / -0.08024205 / 2/4 | -0.1540088 / -0.009137159 / 2/4 | -0.1610315 / -0.02386147 / 2/4 | -15.25 / 2/4 | -0.06225458 |

## Every seed and leave-one-seed-out sensitivity

All four seeds remain in the primary results. Each row below omits exactly one registered seed and recomputes the equal-cell macro; no row selects a new winner. Full per-cell/per-parent omitted-seed values are retained in results.json.

### OLD seed sensitivity

| Effect | Scope | Seed | Free symmetric RMS (%L) | Gamma to surface p95 (%L) | Gamma to free edges p95 (%L) | Gamma to surface coverage <0.1% L | Gamma to free edges coverage <0.1% L | FF illegal pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Edge_off | seed-only | 10201 | 0.1280906 | 0.2444765 | 0.2601746 | -0.1304446 | 0.02768882 | 6.666667 |
| Edge_off | seed-only | 10202 | -0.1667457 | -0.4972627 | -0.5104291 | 0.07520201 | 0.00674105 | 2.5 |
| Edge_off | seed-only | 10203 | -0.06213794 | -0.3536517 | -0.3561018 | 0.07132031 | 0.03568527 | 4.666667 |
| Edge_off | seed-only | 10204 | -0.2383667 | 0.04181692 | 0.04256552 | -0.119044 | 0.04973857 | 8.5 |
| Edge_off | omit-seed | 10201 | -0.1557501 | -0.2696992 | -0.2746551 | 0.009159426 | 0.03072163 | 5.222222 |
| Edge_off | omit-seed | 10202 | -0.05747136 | -0.02245276 | -0.01778724 | -0.05938943 | 0.03770422 | 6.611111 |
| Edge_off | omit-seed | 10203 | -0.0923406 | -0.0703231 | -0.06922965 | -0.05809554 | 0.02805615 | 5.888889 |
| Edge_off | omit-seed | 10204 | -0.03359768 | -0.202146 | -0.2021188 | 0.00535925 | 0.02337171 | 4.611111 |
| L5_off | seed-only | 10201 | 0.09137862 | 0.1054208 | 0.1189342 | -0.1296944 | -0.03039352 | -1.5 |
| L5_off | seed-only | 10202 | 0.2563064 | 0.9419031 | 0.9806471 | -0.05736043 | -0.0818681 | -1 |
| L5_off | seed-only | 10203 | -0.02773688 | -0.3003039 | -0.302754 | 0.01299017 | -0.01922463 | -1 |
| L5_off | seed-only | 10204 | -0.2133243 | -0.04697466 | -0.04653591 | -0.07438382 | 0.0508677 | 5.833333 |
| L5_off | omit-seed | 10201 | 0.00508174 | 0.1982082 | 0.2104524 | -0.03958469 | -0.01674168 | 1.277778 |
| L5_off | omit-seed | 10202 | -0.0498942 | -0.08061924 | -0.07678524 | -0.06369603 | 0.0004165167 | 1.111111 |
| L5_off | omit-seed | 10203 | 0.04478691 | 0.3334498 | 0.3510151 | -0.08714623 | -0.02046464 | 1.111111 |
| L5_off | omit-seed | 10204 | 0.1066494 | 0.2490067 | 0.2656091 | -0.05802156 | -0.04382875 | -1.166667 |
| Edge_on | seed-only | 10201 | 0.05727826 | -0.1554944 | -0.1516859 | 0.06583069 | -0.01021815 | 5.833333 |
| Edge_on | seed-only | 10202 | -0.7396238 | -2.607501 | -2.627853 | 0.1191415 | 0.0495354 | -5.166667 |
| Edge_on | seed-only | 10203 | 0.004562803 | 0.07264504 | 0.07264504 | -0.04370534 | 0.0218517 | -1.833333 |
| Edge_on | seed-only | 10204 | 0.05884952 | -0.0277382 | -0.02812118 | 0.04141991 | 0.06677602 | -8.333333 |
| Edge_on | omit-seed | 10201 | -0.2254038 | -0.8541981 | -0.8611099 | 0.03895201 | 0.04605437 | -5.111111 |
| Edge_on | omit-seed | 10202 | 0.0402302 | -0.03686253 | -0.03572069 | 0.02118175 | 0.02613652 | -1.444444 |
| Edge_on | omit-seed | 10203 | -0.207832 | -0.9302446 | -0.9358869 | 0.07546402 | 0.03536442 | -2.555556 |
| Edge_on | omit-seed | 10204 | -0.2259276 | -0.8967835 | -0.9022981 | 0.04708893 | 0.02038965 | -0.3888889 |
| L5_on | seed-only | 10201 | 0.02056633 | -0.2945501 | -0.2929263 | 0.06658082 | -0.06830049 | -2.333333 |
| L5_on | seed-only | 10202 | -0.3165717 | -1.168335 | -1.136777 | -0.01342099 | -0.03907376 | -8.666667 |
| L5_on | seed-only | 10203 | 0.03896386 | 0.1259929 | 0.1259929 | -0.1020355 | -0.0330582 | -7.5 |
| L5_on | seed-only | 10204 | 0.08389189 | -0.1165298 | -0.1172226 | 0.08608013 | 0.06790515 | -11 |
| L5_on | omit-seed | 10201 | -0.06457199 | -0.3862907 | -0.3760024 | -0.009792112 | -0.001408935 | -9.055556 |
| L5_on | omit-seed | 10202 | 0.04780736 | -0.095029 | -0.09471869 | 0.01687516 | -0.01115118 | -6.944444 |
| L5_on | omit-seed | 10203 | -0.0707045 | -0.5264717 | -0.5156421 | 0.04641332 | -0.01315637 | -7.333333 |
| L5_on | omit-seed | 10204 | -0.08568051 | -0.4456308 | -0.4345703 | -0.01629188 | -0.04681082 | -6.166667 |
| I | seed-only | 10201 | -0.07081229 | -0.3999709 | -0.4118605 | 0.1962753 | -0.03790697 | -0.8333333 |
| I | seed-only | 10202 | -0.5728781 | -2.110238 | -2.117424 | 0.04393944 | 0.04279435 | -7.666667 |
| I | seed-only | 10203 | 0.06670075 | 0.4262968 | 0.4287469 | -0.1150257 | -0.01383357 | -6.5 |
| I | seed-only | 10204 | 0.2972162 | -0.06955512 | -0.07068669 | 0.160464 | 0.01703745 | -16.83333 |
| I | omit-seed | 10201 | -0.06965373 | -0.5844989 | -0.5864547 | 0.02979258 | 0.01533274 | -10.33333 |
| I | omit-seed | 10202 | 0.09770156 | -0.01440976 | -0.01793345 | 0.08057119 | -0.0115677 | -8.055556 |
| I | omit-seed | 10203 | -0.1154914 | -0.8599215 | -0.8666572 | 0.1335596 | 0.007308275 | -8.444444 |
| I | omit-seed | 10204 | -0.1923299 | -0.6946375 | -0.7001794 | 0.04172968 | -0.002982064 | -5 |

### NEW seed sensitivity

| Effect | Scope | Seed | Free symmetric RMS (%L) | Gamma to surface p95 (%L) | Gamma to free edges p95 (%L) | Gamma to surface coverage <0.1% L | Gamma to free edges coverage <0.1% L | FF illegal pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Edge_off | seed-only | 10201 | 0.1114815 | 0.1028709 | 0.1033543 | -0.005048049 | 0.03652891 | -2.75 |
| Edge_off | seed-only | 10202 | 0.1600208 | -0.2188232 | -0.2187453 | 0.07287978 | 0.03984569 | -13 |
| Edge_off | seed-only | 10203 | 0.8916071 | 1.747835 | 1.752115 | 0.06704094 | 0.03453267 | 36.5 |
| Edge_off | seed-only | 10204 | -0.1406433 | -0.9308047 | -0.9312221 | 0.04925769 | 0.07796515 | 2.5 |
| Edge_off | omit-seed | 10201 | 0.3036615 | 0.1994025 | 0.2007159 | 0.06305947 | 0.05078117 | 8.666667 |
| Edge_off | omit-seed | 10202 | 0.2874818 | 0.3066339 | 0.3080824 | 0.03708353 | 0.04967558 | 12.08333 |
| Edge_off | omit-seed | 10203 | 0.04361967 | -0.348919 | -0.348871 | 0.03902981 | 0.05144658 | -4.416667 |
| Edge_off | omit-seed | 10204 | 0.3877031 | 0.5439611 | 0.5455747 | 0.04495756 | 0.03696909 | 6.916667 |
| L5_off | seed-only | 10201 | 0.09733096 | -0.1027062 | -0.09451205 | -0.01060355 | 0.003954513 | -8 |
| L5_off | seed-only | 10202 | 0.6246232 | -0.06807711 | -0.06807711 | 0.04833487 | 0.01501509 | -3.25 |
| L5_off | seed-only | 10203 | 0.644677 | 1.643182 | 1.68209 | 0.08599418 | 0.07430484 | 23 |
| L5_off | seed-only | 10204 | -0.06306177 | -0.7190013 | -0.7194186 | -0.07434429 | -0.01312704 | 9.75 |
| L5_off | omit-seed | 10201 | 0.4020795 | 0.2853678 | 0.298198 | 0.01999492 | 0.02539763 | 9.833333 |
| L5_off | omit-seed | 10202 | 0.2263154 | 0.2738247 | 0.2893863 | 0.0003487802 | 0.02171077 | 8.25 |
| L5_off | omit-seed | 10203 | 0.2196308 | -0.2965949 | -0.2940026 | -0.01220432 | 0.001947519 | -0.5 |
| L5_off | omit-seed | 10204 | 0.4555437 | 0.4907995 | 0.5065001 | 0.04124183 | 0.03109148 | 3.916667 |
| Edge_on | seed-only | 10201 | -0.005576259 | 0.1877601 | 0.179789 | 0.0182815 | 0.03646906 | 4.75 |
| Edge_on | seed-only | 10202 | -0.1209621 | 1.248499 | 1.248699 | 0.1001719 | 0.01839276 | -10.75 |
| Edge_on | seed-only | 10203 | -0.2254595 | 0.02358764 | -0.01485921 | 0.03262676 | 0.06576501 | 1 |
| Edge_on | seed-only | 10204 | 0.1425826 | -0.1896039 | -0.1896039 | 0.1318925 | 0.1276893 | -21.75 |
| Edge_on | omit-seed | 10201 | -0.06794632 | 0.3608275 | 0.3480787 | 0.08823037 | 0.07061568 | -10.5 |
| Edge_on | omit-seed | 10202 | -0.02948438 | 0.007247958 | -0.008224707 | 0.06093359 | 0.07664111 | -5.333333 |
| Edge_on | omit-seed | 10203 | 0.005348097 | 0.4155516 | 0.4129614 | 0.08344862 | 0.06085036 | -9.25 |
| Edge_on | omit-seed | 10204 | -0.1173326 | 0.4866154 | 0.4712096 | 0.05036004 | 0.04020894 | -1.666667 |
| L5_on | seed-only | 10201 | -0.01972677 | -0.01781695 | -0.01807737 | 0.01272601 | 0.003894657 | -0.5 |
| L5_on | seed-only | 10202 | 0.3436403 | 1.399245 | 1.399367 | 0.07562695 | -0.006437843 | -1 |
| L5_on | seed-only | 10203 | -0.4723896 | -0.08106613 | -0.08488474 | 0.05158 | 0.1055372 | -12.5 |
| L5_on | seed-only | 10204 | 0.2201641 | 0.0221996 | 0.0221996 | 0.008290509 | 0.03659708 | -14.5 |
| L5_on | omit-seed | 10201 | 0.03047162 | 0.4467927 | 0.4455607 | 0.04516582 | 0.04523214 | -9.333333 |
| L5_on | omit-seed | 10202 | -0.09065073 | -0.02556116 | -0.02692084 | 0.02419884 | 0.0486763 | -9.166667 |
| L5_on | omit-seed | 10203 | 0.1813592 | 0.4678758 | 0.4678299 | 0.03221449 | 0.0113513 | -5.333333 |
| L5_on | omit-seed | 10204 | -0.04949202 | 0.4334539 | 0.4321351 | 0.04664432 | 0.03433133 | -4.666667 |
| I | seed-only | 10201 | -0.1170577 | 0.08488921 | 0.07643468 | 0.02332955 | -5.985523e-05 | 7.5 |
| I | seed-only | 10202 | -0.2809829 | 1.467322 | 1.467444 | 0.02729208 | -0.02145293 | 2.25 |
| I | seed-only | 10203 | -1.117067 | -1.724248 | -1.766974 | -0.03441418 | 0.03123234 | -35.5 |
| I | seed-only | 10204 | 0.2832259 | 0.7412009 | 0.7416182 | 0.0826348 | 0.04972412 | -24.25 |
| I | omit-seed | 10201 | -0.3716079 | 0.1614249 | 0.1473628 | 0.0251709 | 0.01983451 | -19.16667 |
| I | omit-seed | 10202 | -0.3169661 | -0.2993859 | -0.3163072 | 0.02385006 | 0.02696553 | -17.41667 |
| I | omit-seed | 10203 | -0.03827157 | 0.7644706 | 0.7618324 | 0.04441881 | 0.009403778 | -4.833333 |
| I | omit-seed | 10204 | -0.5050357 | -0.05734561 | -0.07436507 | 0.005402484 | 0.003239851 | -8.583333 |

## Condition binding and geometric interpretation

The four-grid binding reuses the same raw free sets and fixed target samples. Positive binding is already established in prior baselines; it is not a newly invented success gate. Low boundary distance is neither shared-index connectivity nor watertightness.

| Patch | Model | Delta0 mean | Delta1 mean | B mean | Both positive /4 |
| --- | --- | --- | --- | --- | --- |
| 0 | L0 | 7.758829 | 9.120035 | 8.439432 | 4 |
| 0 | Edge_only | 7.66878 | 9.110815 | 8.389798 | 4 |
| 0 | L5 | 7.464317 | 9.154028 | 8.309172 | 4 |
| 0 | JEdge5 | 7.635391 | 9.197227 | 8.416309 | 4 |
| 1 | L0 | 4.81386 | 4.84604 | 4.82995 | 4 |
| 1 | Edge_only | 4.450239 | 4.982 | 4.71612 | 4 |
| 1 | L5 | 4.300547 | 4.742414 | 4.521481 | 4 |
| 1 | JEdge5 | 5.314481 | 5.695488 | 5.504985 | 4 |
| 2 | L0 | 4.214644 | 4.150273 | 4.182459 | 4 |
| 2 | Edge_only | 4.516796 | 4.327792 | 4.422294 | 4 |
| 2 | L5 | 4.335435 | 4.223365 | 4.2794 | 4 |
| 2 | JEdge5 | 3.913755 | 4.092743 | 4.003249 | 4 |
| 3 | L0 | 5.630598 | 6.323166 | 5.976882 | 4 |
| 3 | Edge_only | 4.74238 | 5.246529 | 4.994454 | 4 |
| 3 | L5 | 4.747803 | 4.951688 | 4.849745 | 4 |
| 3 | JEdge5 | 4.783712 | 5.135598 | 4.959655 | 4 |
| 4 | L0 | 2.007691 | 1.693324 | 1.850508 | 3 |
| 4 | Edge_only | 1.927202 | 1.439251 | 1.683227 | 2 |
| 4 | L5 | 2.115638 | 1.227305 | 1.671472 | 2 |
| 4 | JEdge5 | 1.859626 | 1.402054 | 1.63084 | 3 |

| Patch | Effect | B mean | B paired median | Higher B /4 |
| --- | --- | --- | --- | --- |
| 0 | Edge_off | -0.04963418 | -0.0409406 | 1/4 |
| 0 | L5_off | -0.1302594 | -0.01038369 | 2/4 |
| 0 | Edge_on | 0.1071366 | -0.0356228 | 1/4 |
| 0 | L5_on | 0.02651131 | 0.02704022 | 3/4 |
| 0 | I | 0.1567708 | 0.005317802 | 2/4 |
| 1 | Edge_off | -0.1138305 | -0.1734156 | 1/4 |
| 1 | L5_off | -0.3084694 | -0.0921055 | 0/4 |
| 1 | Edge_on | 0.9835039 | -0.04116153 | 1/4 |
| 1 | L5_on | 0.788865 | 0.1170806 | 3/4 |
| 1 | I | 1.097334 | 0.190845 | 4/4 |
| 2 | Edge_off | 0.2398355 | 0.2615139 | 3/4 |
| 2 | L5_off | 0.09694155 | 0.03634426 | 2/4 |
| 2 | Edge_on | -0.2761512 | -0.2677852 | 1/4 |
| 2 | L5_on | -0.4190451 | -0.3284035 | 0/4 |
| 2 | I | -0.5159867 | -0.5095911 | 0/4 |
| 3 | Edge_off | -0.9824277 | -0.5577699 | 1/4 |
| 3 | L5_off | -1.127137 | -0.9099624 | 2/4 |
| 3 | Edge_on | 0.1099095 | -0.007135037 | 2/4 |
| 3 | L5_on | -0.03479964 | -0.02659684 | 2/4 |
| 3 | I | 1.092337 | 0.7847239 | 2/4 |
| 4 | Edge_off | -0.1672809 | -0.1591855 | 0/4 |
| 4 | L5_off | -0.1790357 | -0.1287851 | 0/4 |
| 4 | Edge_on | -0.04063205 | -0.156209 | 2/4 |
| 4 | L5_on | -0.05238681 | -0.1786241 | 2/4 |
| 4 | I | 0.1266489 | -0.1049856 | 2/4 |

Gamma-to-free-edges uses the exact same registered boundary points/weights, every finite free triangle edge, no GT correspondence, welding, duplicate-edge removal or coordinate changes. Zero-length edges are points. Point-to-edge distance must be at least point-to-surface distance within 1e-10 L. Both diagnostics report strict coverages at 0.05%, 0.1%, 0.2%, 0.5%, 1.0% L; 0.1% remains the historical reference.

## Teacher, training and quadrature evidence

| Model | Status | States | Reused | FM | L3 | L5 | D_edge |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L0 | PASS | 16 | True | 0.0002439098 | 1.285465e-06 | 1.683963e-05 | 1.219859e-06 |
| Edge_only | PASS | 16 | False | 0.0002338906 | 9.125402e-07 | 1.475611e-05 | 8.450108e-07 |
| L5 | PASS | 16 | True | 0.0002298109 | 9.762009e-07 | 1.477512e-05 | 9.240298e-07 |
| JEdge5 | PASS | 16 | True | 0.0002693629 | 9.775541e-07 | 1.65472e-05 | 9.325684e-07 |

Only Edge-only adds the registered 16 teacher forwards. Old models are read-only reuse after endpoint/state/metric identity checks; NOT_COMPARABLE remains explicit. These are eight training pairs at t=.5/.75, not 16 independent objects or a training-distribution expectation. Teacher ranks do not determine generation quality.

| Step | FM | Gated Edge mean | Weighted geometry | Total | Active/8 | Clip-before norm |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.0007183109 | 2.529912e-07 | 2.060049e-08 | 0.0007183315 | 3 | 0.1477722 |
| 50 | 0.0001372571 | 3.022478e-07 | 1.230567e-06 | 0.0001384877 | 3 | 0.08834139 |
| 100 | 0.0007447762 | 5.788087e-07 | 2.356554e-06 | 0.0007471327 | 5 | 0.0996953 |
| 250 | 0.0001428876 | 3.832403e-07 | 1.560319e-06 | 0.0001444479 | 7 | 0.04444271 |
| 500 | 0.0007530802 | 2.963994e-07 | 1.206756e-06 | 0.0007542869 | 4 | 0.07353328 |

| Step | Parameter group | Gradient RMS before clip | Update RMS | Gradient finite/nonzero | Update finite/nonzero |
| --- | --- | --- | --- | --- | --- |
| 1 | Geo | 7.029356e-07 | 7.573844e-06 | True/149498 | True/149475 |
| 1 | role | 1.10583e-05 | 9.542512e-06 | True/1536 | True/1536 |
| 1 | backbone | 1.294797e-05 | 8.382163e-06 | True/130216607 | True/129841457 |
| 50 | Geo | 4.588338e-07 | 2.217323e-06 | True/149500 | True/149450 |
| 50 | role | 4.189512e-06 | 2.378092e-06 | True/1536 | True/1536 |
| 50 | backbone | 7.740592e-06 | 2.055169e-06 | True/130212676 | True/129678879 |
| 100 | Geo | 7.82251e-07 | 1.757229e-06 | True/149501 | True/149425 |
| 100 | role | 2.534391e-05 | 2.213248e-06 | True/1536 | True/1536 |
| 100 | backbone | 8.734996e-06 | 2.337148e-06 | True/130215804 | True/129716761 |
| 250 | Geo | 2.877278e-07 | 3.215508e-06 | True/149497 | True/149494 |
| 250 | role | 3.984529e-06 | 3.430373e-06 | True/1536 | True/1536 |
| 250 | backbone | 3.894108e-06 | 2.688628e-06 | True/130213454 | True/129774461 |
| 500 | Geo | 8.805853e-07 | 1.620521e-06 | True/149500 | True/149397 |
| 500 | role | 1.115743e-05 | 1.874625e-06 | True/1536 | True/1536 |
| 500 | backbone | 6.442926e-06 | 1.850135e-06 | True/130215721 | True/129646193 |

Training log forward values precede their recorded optimizer update: the step500 forward uses cumulative2999 weights, whereas complete generation and the new teacher diagnostic use the saved cumulative3000 endpoint. Different logged batches/times do not establish convergence. Edge-only never adds L5 to its training backward.

| Step | Microbatch | t | FM gradient norm | Edge gradient norm | Weighted Edge/FM | Cosine | Angle degrees |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 2 | 0.529789 | 0.0001166481 | 4.112242e-06 | 0.1435301 | 0.1180926 | 83.21797 |
| 500 | 0 | 0.8865871 | 0.0002114772 | 4.016602e-07 | 0.007732813 | 0.1336978 | 82.31667 |

The independent FP64 quadrature diagnostic is delivered as quadrature_diagnostic.json and its loss/derivative figure. Five-point production loss and dense references are separate; diagnostic findings do not alter this arm, coefficients or sampler.

| Quadrature item | Value |
| --- | --- |
| Status | PASS_ANALYTIC_WITNESSES |
| Dense cases not converged | ['collinear_delta_0.001', 'collinear_delta_0.01', 'collinear_delta_0.05', 'centered_length_ratio_1.2'] |
| Cases | 19 |

| Case | Five-point | 1025-point | Analytic continuous | Five derivative | Analytic derivative | Dense convergence |
| --- | --- | --- | --- | --- | --- | --- |
| collinear_delta_0.001 | 1.25e-07 | 4.888177e-10 | 3.333333e-10 | 0.00025 | 1e-06 | NOT_CONVERGED_AT_FIXED_GRID |
| collinear_delta_0.01 | 1.25e-05 | 3.349081e-07 | 3.333333e-07 | 0.0025 | 0.0001 | NOT_CONVERGED_AT_FIXED_GRID |
| collinear_delta_0.05 | 0.0003125 | 4.16746e-05 | 4.166667e-05 | 0.0125 | 0.0025 | NOT_CONVERGED_AT_FIXED_GRID |
| collinear_delta_0.1 | 0.00125 | 0.0003333492 | 0.0003333333 | 0.025 | 0.01 | CONVERGED_AT_FIXED_TOLERANCE |
| collinear_delta_0.2 | 0.005 | 0.002666698 | 0.002666667 | 0.05 | 0.04 | CONVERGED_AT_FIXED_TOLERANCE |
| perpendicular_h_0.001 | 1e-06 | 1e-06 | 1e-06 | 0.002 | 0.002 | CONVERGED_AT_FIXED_TOLERANCE |
| perpendicular_h_0.01 | 0.0001 | 0.0001 | 0.0001 | 0.02 | 0.02 | CONVERGED_AT_FIXED_TOLERANCE |
| perpendicular_h_0.05 | 0.0025 | 0.0025 | 0.0025 | 0.1 | 0.1 | CONVERGED_AT_FIXED_TOLERANCE |
| perpendicular_h_0.1 | 0.01 | 0.01 | 0.01 | 0.2 | 0.2 | CONVERGED_AT_FIXED_TOLERANCE |
| perpendicular_h_0.2 | 0.04 | 0.04 | 0.04 | 0.4 | 0.4 | CONVERGED_AT_FIXED_TOLERANCE |
| centered_length_ratio_0.6 | 0.005 | 0.002666698 | 0.002666667 | -0.025 | -0.02 | CONVERGED_AT_FIXED_TOLERANCE |
| centered_length_ratio_0.8 | 0.00125 | 0.0003333492 | 0.0003333333 | -0.0125 | -0.005 | CONVERGED_AT_FIXED_TOLERANCE |
| centered_length_ratio_1.2 | 0.00125 | 0.0002777968 | 0.0002777778 | 0.0125 | 0.003935185 | NOT_CONVERGED_AT_FIXED_GRID |
| centered_length_ratio_1.4 | 0.005 | 0.001904806 | 0.001904762 | 0.025 | 0.01292517 | CONVERGED_AT_FIXED_TOLERANCE |
| centered_rotation_0.1 | 0.0009343792 | 0.0008305608 | 0.0008305593 | 0.01862525 | 0.01655578 | CONVERGED_AT_FIXED_TOLERANCE |
| centered_rotation_0.3 | 0.008187393 | 0.007277697 | 0.007277683 | 0.05293523 | 0.04705354 | CONVERGED_AT_FIXED_TOLERANCE |
| centered_rotation_0.7 | 0.03890779 | 0.03458477 | 0.0345847 | 0.09238591 | 0.08212081 | CONVERGED_AT_FIXED_TOLERANCE |
| rotation_endpoint_exchange | 0.008187393 | 0.007277697 | 0.007277683 | 0.05293523 | 0.04705354 | CONVERGED_AT_FIXED_TOLERANCE |
| symmetric_midpoint_collapse | 0.046875 | 0.04166675 | 0.04166667 | 0 | 0 | CONVERGED_AT_FIXED_TOLERANCE |

![Independent fixed quadrature loss and directional derivative diagnostic](quadrature_loss_derivative.png)

For a collinear shift 0<delta<ell/4, the production five-point value is delta^2/(8 L^2), while the continuous symmetric mean is delta^3/(3 ell L^2). The ratio is 3 ell/(8 delta): small shifts can be over-weighted by sparse endpoint quadrature. A symmetric midpoint collapse can retain positive loss with zero gradient. Neither observation proves that denser integration improves generation.

## Runtime, costs, checks and endpoint

Actual total accounting is copied from the final ledger; stage-specific counters below are components and must not be added to these totals again.

| Actual total ledger field | Recorded value |
| --- | --- |
| optimizer_updates | 500 |
| formal_full_generations | 160 |
| complete_reload_replays | 1 |
| total_complete_rollouts | 161 |
| model_forward_calls_total | 12071 |
| training_model_forwards | 4000 |
| wiring_and_reload_model_forwards | 5 |
| new_teacher_model_forwards | 16 |
| sampling_model_forwards_including_replay | 8050 |
| model_parameter_backwards_total | 4001 |
| training_backwards | 4000 |
| zero_update_wiring_backwards | 1 |
| CPU_nested_OT_calls | 4000 |
| CPU_quadrature_and_fixture_loss_evaluations | 173 |
| CPU_quadrature_parameter_autograd_calls | 58 |
| CPU_parity_edge_loss_evaluations | 40 |
| CPU_parity_velocity_autograd_calls | 32 |
| training_velocity_diagnostic_autograd_calls | 4 |
| small_autograd_calls_total_excluding_model_backward | 94 |
| CPU_finite_difference_loss_evaluations_included_in_quadrature_total | 114 |
| additional_formal_generation_retries | 0 |

| Runtime environment / GPU | Recorded value |
| --- | --- |
| python | 3.10.21 \| packaged by conda-forge \| (main, Aug 21 2026, 22:42:31) [GCC 14.4.0] |
| platform | Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 |
| torch | 2.7.1+cu128 |
| torch_cuda | 12.8 |
| nvidia_smi | name, memory.total [MiB], memory.used [MiB], driver_version NVIDIA GeForce RTX 5080, 16303 MiB, 6769 MiB, 591.86  |
| CUDA_initialised_by_environment_probe | False |

Read-only environment probe during authorized training; no model or tensor CUDA operation.

Final timing buckets are separately recorded; metadata labels identify seconds and any inclusion/exclusion rules. Overlapping diagnostic and stage timers are not summed here.

| Final timing field | Recorded value |
| --- | --- |
| training_wall_seconds | 623.6644 |
| training_model_loss_backward_update_seconds | 576.9291 |
| training_CPU_OT_stream_seconds | 5.024378 |
| training_supervision_prepare_seconds | 10.20281 |
| training_checkpoint_save_seconds | 18.3584 |
| sampling_stage_including_loads_and_replay_seconds | 422.1351 |
| native_sampler_including_synchronization_state_transfer_seconds | 368.4852 |
| trajectory_file_save_seconds | 4.123394 |
| offline_geometry_audit_seconds | 149.2911 |
| teacher_stage_seconds | 13.52307 |
| read_only_final_asset_audit_seconds | 17.96082 |
| note | Sub-buckets overlap with stage totals: do not add them. Native sampling includes synchronization and state copies, not pure network throughput. CPU OT and offline audits are not GPU generation. |

| Actual runtime field | Value |
| --- | --- |
| cublas_workspace_config | :4096:8 |
| deterministic_algorithms | True |
| deterministic_warn_only | False |
| tf32_matmul | False |
| tf32_cudnn | False |
| cudnn_benchmark | False |
| external_flash_available | False |
| sdpa_math_enabled | True |
| sdpa_flash_enabled | False |
| sdpa_mem_efficient_enabled | False |
| sdpa_cudnn_enabled | False |
| cuda_autocast_enabled | True |
| cuda_autocast_dtype | torch.bfloat16 |
| gradient_enabled | True |
| parameter_dtype | torch.float32 |
| parameter_elements_checked | 130401155 |
| coordinate_tensors_checked | 1 |
| active_context_depth | 0 |
| native_active_context_depth | 1 |
| native_precision | bf16 |
| all_modules_eval | True |
| gradient_checkpointing | False |
| trainable_parameters | 130401155 |
| all_backbone_parameters_trainable | True |
| role_parameters | 1536 |
| frozen_backbone_guard_used | False |

| Counter | Actual |
| --- | --- |
| training.OT_calls | 4000 |
| training.OT_returns | 4000 |
| training.forward_attempts | 4000 |
| training.forward_returns | 4000 |
| training.backward_attempts | 4000 |
| training.backward_returns | 4000 |
| training.optimizer_step_attempts | 500 |
| training.optimizer_updates | 500 |
| training.velocity_autograd_attempts | 4 |
| training.velocity_autograd_returns | 4 |
| wiring.forward_attempts | 5 |
| wiring.forward_returns | 5 |
| wiring.backward_attempts | 1 |
| wiring.backward_returns | 1 |
| teacher.forward_attempts | 16 |
| teacher.forward_returns | 16 |
| teacher.L3_loss_calls | 16 |
| teacher.L5_loss_calls | 16 |
| teacher.D_edge_loss_calls | 16 |
| sampling.noise_generation_calls | 4 |
| sampling.formal_output_attempts | 160 |
| sampling.formal_outputs_saved | 160 |
| sampling.model_forward_attempts | 8050 |
| sampling.model_forward_returns | 8050 |
| sampling.euler_updates | 8050 |
| sampling.complete_rollouts | 161 |
| sampling.rollout_attempts | 161 |
| sampling.model_backward_calls | 0 |
| sampling.optimizer_updates | 0 |
| sampling.geometry_gradient_calls | 0 |
| sampling.postprocessing_repairs | 0 |
| sampling.trajectory_save_seconds | 4.123394 |
| sampling.native_sampling_seconds | 368.4852 |
| sampling.generation_wall_seconds | 373.1465 |
| sampling.replay_output_attempts | 1 |
| sampling.replay_outputs_verified | 1 |
| CPU_parity.edge_loss_calls | 40 |
| CPU_parity.velocity_autograd_attempts | 32 |
| CPU_parity.velocity_autograd_returns | 32 |

| Training/diagnostic loss work | Actual |
| --- | --- |
| wiring.edge_attempts | 1 |
| wiring.edge_returns | 1 |
| wiring.L5_calls | 0 |
| training.edge_attempts | 1997 |
| training.edge_returns | 1997 |
| training.L5_calls | 0 |
| training_velocity_probe.edge_attempts | 2 |
| training_velocity_probe.edge_returns | 2 |
| training_velocity_probe.L5_calls | 0 |

| Saved test record | Status | Checks | CPU/autograd scope |
| --- | --- | --- | --- |
| cpu_math_tests.json | PASS | 6 | {'loss_attempts': 2, 'loss_returns': 2, 'production_five_loss_calls': 2, 'dense_loss_calls': 0, 'parameter_autograd_attempts': 1, 'parameter_autograd_returns': 1, 'finite_difference_evaluations': 0, 'finite_difference_comparisons': 0, 'finite_difference_skipped_at_switch': 0, 'projection_regime_checks': 0, 'target_preparations': 2, 'analytic_reference_evaluations': 1, 'model_forwards': 0, 'OT_calls': 0, 'optimizer_updates': 0, 'gamma_to_free_edges_calls_attempted': 7, 'gamma_to_free_edges_point_segment_pairs': 90, 'gamma_to_free_edges_zero_length_occurrences': 7, 'gamma_to_free_edges_calls_returned': 6} |

| Saved test record | Status | Checks | CPU/autograd scope |
| --- | --- | --- | --- |
| evaluation_test_record.json | PASS | 15 | Scalar matrix, pairing-before-median, five factorial formulas, equal cells, coverage sign, LOSO, incomplete new-unit handling |

| Saved test record | Status | Checks | CPU/autograd scope |
| --- | --- | --- | --- |
| report_test_record.json | PASS | 8 | Report text from pure synthetic scalar matrix; no real scores, files or figures |

| Offline audit counter | Actual |
| --- | --- |
| model_calls | 0 |
| OT_calls | 0 |
| geometry_autograd_calls | 0 |
| optimizer_updates | 0 |
| repairs | 0 |
| input_parent_patch_audits | 10 |
| input_indexed_mesh_audits | 30 |
| input_exact_index_calls | 10 |
| verified_pair_tests | 984660 |
| primary_pair_tests | 1176660 |
| independent_pair_tests | 984660 |
| independent_AABB_disjoint | 912820 |
| static_context_audits | 10 |
| historical_seam_evaluations | 160 |
| endpoint_metric_evaluations | 160 |
| boundary_distance_evaluations | 160 |
| static_context_audit_reuses | 160 |
| near_C_fixed_area_sample_diagnostics | 160 |
| indexed_mesh_audits | 320 |
| exact_coordinate_indexings | 320 |
| geometry_repairs | 0 |
| generated_topology_attributions | 160 |
| t1_native_confirmation_endpoint_scores | 160 |
| additional_fixed_Gamma_surface_distance_calls | 160 |
| gamma_to_free_edges_calls_attempted | 160 |
| gamma_to_free_edges_point_segment_pairs | 24758400 |
| gamma_to_free_edges_zero_length_occurrences | 0 |
| gamma_to_free_edges_calls_returned | 160 |

| Cost bucket | Seconds |
| --- | --- |
| training.CPU_OT_and_stream_seconds | 5.024378 |
| training.supervision_prepare_seconds | 10.20281 |
| training.model_update_seconds | 576.9291 |
| training.checkpoint_seconds | 18.3584 |
| training.wall_seconds | 623.6644 |
| stage.preflight | 7.416853 |
| stage.train | 624.5249 |
| stage.teacher | 13.52307 |
| stage.sample | 422.1351 |
| stage.evaluate | 150.0027 |
| stage.report | 3.686691 |
| offline audit | 149.2911 |

Model-update timing includes geometry loss, backward, synchronization and scheduled diagnostics. CPU OT/stream preparation, supervision setup and checkpoint saving are separately recorded. Sampling includes the recorded transfer/trajectory-storage work; these times are not pure network throughput. The inference definition remains 50 generator calls per completed rollout.

| Check | Status |
| --- | --- |
| Wiring | PASS |
| Strict saved endpoint reload | PASS |
| Training | PASS |
| Evaluation | PASS |
| Final protection | PASS |
| Default model changed | False |

The inherited BF16 face-permutation FAIL remains an inherited numerical limitation at its original tolerance; this auxiliary-loss experiment does not certify that it was fixed. Independent numerical triangle tests are not exact-predicate or continuous-collision proofs. Unexecuted audits remain NOT_RUN.

Strict Edge-only inference load (one process; no training or sampling):

~~~bash
python -B -c "from experiments.mf_h1_local0.edge_factorial_pilot.checkpoint import load; m,a=load('/mnt/e/MeshFlow/native_t1/runs/edge_factorial_20261001_123000/Edge_only/additional_step500_cumulative_step3000.pt'); print(a)"
~~~

## Delivery files

| Artifact | Link |
| --- | --- |
| Complete result and metadata | [results.json](results.json) |
| Per-output metrics | [compact_per_output.csv](compact_per_output.csv) |
| Matched per-seed factorial effects | [compact_per_seed_effects.csv](compact_per_seed_effects.csv) |
| Code diff | [code.diff](code.diff) |
| Frozen input / patch / noise registration | [registration/registration.json](registration/registration.json) |
| CPU mathematical fixture record | [cpu_math_tests.json](cpu_math_tests.json) |
| Independent raw-output / asset protection audit | [final_raw_asset_audit.json](final_raw_asset_audit.json) |
| Independent mathematics / wiring review | [math_and_wiring_review.json](math_and_wiring_review.json) |
| Independent scalar summary review | [summary_review.json](summary_review.json) |
| Environment versions and GPU | [environment.json](environment.json) |
| Independent quadrature diagnostic | [quadrature_diagnostic.json](quadrature_diagnostic.json) |

## Fixed first-seed raw views

The first registered seed was fixed before generation. Every image uses unchanged raw coordinates, the same Y-up camera (elevation18, azimuth-60), and one common set of bounds across models/parents/patches. It is a fixed illustration, not an outcome-selected representative.

![Patch 0; full/free, both parents, four models](fixed_seed10201_patch0_v2.png)

![Patch 1; full/free, both parents, four models](fixed_seed10201_patch1_v2.png)

![Patch 2; full/free, both parents, four models](fixed_seed10201_patch2_v2.png)

![Patch 3; full/free, both parents, four models](fixed_seed10201_patch3_v2.png)

![Patch 4; full/free, both parents, four models](fixed_seed10201_patch4_v2.png)

Next step: 停止本轮。优先定位新上下文的补全覆盖与 Gamma 尾部退化；后续若另行授权，再把求积近似作为单因素实验。保留 Edge-only 作单项对照、JEdge5 作 FF/局部接口取舍版本。默认配方保持不变，不表示本轮证实其综合优势；不追加训练、组合、强度扫描或推理修补。

No quadrature change, new coefficient, source anchoring, third loss, extra training or automatic default switch is performed.
