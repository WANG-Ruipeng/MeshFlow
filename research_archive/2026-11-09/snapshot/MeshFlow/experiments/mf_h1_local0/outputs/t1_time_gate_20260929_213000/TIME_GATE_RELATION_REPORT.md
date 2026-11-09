# C_gate 时间门控关系损失：固定预算验证

本轮只有一个新训练臂 C_gate；旧 A/C 的32份终点评分与原始输出只读复用。新增完整生成固定为两个父对象×9501–9508、K12，每份 N112/free100。评价沿用完整独立 CF/FF/CC、边界、形状与拓扑审计，不更改阈值或分母。

9501–9508 是前轮已观察过的固定配对噪声；这轮属于已知 seed 上的探索性跟进，不包装成独立新噪声确认。

门控同时降低辅助损失总暴露并改变时间分配。没有全时段降低权重的对照，不能把结果唯一归因于时间定位；本轮也不是泛化实验。

## 六个问题的回答

1. 未在两个父对象上恢复 A 的精度。P0 剩余 RMS 均值 A/C/C_gate 为 1.251969%/1.351986%/1.380038%，门控未达到预期的约1.25%；P1 为 0.684499%/0.870473%/0.688787%，数值接近 A，但未预设非劣界限，不能宣布等价。两个对象相对 A 都只有4/8个seed的剩余RMS改善。

2. 完整生成中的 FF 收益大体保留：相对 A，P0 平均减少21.75对（8/8降低），P1减少19.75对（7/8降低）。相对 C，P0为103.25对 vs103.375对，平均基本持平且仅4/8更少；P1为92.25对 vs105.5对，7/8更少。它不是每个样本都优于 C，也不是无自交保证。

3. 接口仍有代价。P0 Gamma RMS/p95由 A 的1.184559%/2.008669%升至1.525262%/3.067488%，尽管coverage由0.257282升至0.303491。P1 Gamma RMS/p95略低于 A，但coverage由0.198284降至0.145238，CF pair由16.625升至17.25。相对 C，两对象CF pair及窄角均值都更高。不能由FF减少推断接口或整体拓扑全面改善。

4. FF减少是多数噪声上的完整生成现象，不是仅有训练loss下降。RMS变化更依赖困难seed：P0相对A的9502增加0.977442个百分点，放大均值代价；P0 Gamma尾部恶化主要来自9506。P1相对C的9502下降1.279604个百分点，贡献了大部分RMS均值收益。全部seed保留在主结果，未删除异常值。配对中位数与各模型边际中位数分别报告。

5. 不支持全指标 Pareto 改善，也不能据此宣称论文原创性或高噪声梯度就是伪影。仅49.3%的训练样本激活辅助项且不补偿总量；本实验同时改变时间支持范围和总辅助监督曝光，没有等曝光的全时段对照。9501–9508已经被观察过，本轮是两个训练对象上的探索性配对跟进，不是独立新噪声确认或泛化证明。

6. 保留 C_gate 作为降低FF相交的候选和机制记录，暂不替代纯T1继续训练臂 A。若继续，应先冻结现有模型做独立未见噪声的确认，重点检验P0精度和接口尾部；不应直接组合B+C或扩大训练。当前预算已结束，未执行下一阶段。

## 两个假设与判断边界

假设1检验 C_gate 是否相对 A 保持或恢复几何精度；假设2检验其是否相对 C 保留相交方面的收益。两种参照分开报告，不因优于其中一个模型就判定另一假设成立。

**H1**：NOT_ESTABLISHED：P0没有恢复A；P1均值接近A，尚未证明非劣或普遍改善。

**H2**：SUPPORTED_WITH_LIMITS：FF相对A的减少进入了多数噪声的完整生成；P0接近原C平均水平，P1进一步减少；仍有CF/覆盖/窄角代价和大量残余相交。

仅凭剩余RMS与FF相交两个均值不能称为全指标 Pareto 改善。双向误差、Gamma RMS/p95/coverage、CF、FF、窄角、退化、绑定以及逐seed结果均需检查；均值取舍与困难噪声不能从报告移除。

## 稳定入口、资产与实际训练

运行状态：PASS_EXECUTION；共同运行配置：{"CUBLAS_WORKSPACE_CONFIG":":4096:8","deterministic":true,"warn_only":false,"TF32":false,"SDPA":"math","parameters":"FP32","forward":"BF16","integration":"FP32"}

初始化与固定方案：{"path":"/mnt/e/MeshFlow/experiments/mf_h1_local0/outputs/native_patch_20260929_010000/Native_correct/endpoint_step500.pt","file_sha256":"7928d77507bf1fe610876f61b5b11e614263323b5fc0f595cc6740ffd24af634","state_sha256":"37e31fe5925b8f870b9a4d3bbbbc84edc7bf30177946a6f67f4529a7f8e97172","old_updates":500}；训练配置：{"seed":10,"additional_updates_each":500,"total_update_cap":500,"effective_batch":8,"microbatch":1,"K":[4,8,12],"parents":2,"stream":"Unchanged NativeTrainingStream Native_correct; independent RandomState streams reset identically","OT":"Original free-only optimal_sum_numpy reorders noise Z only; target Y ordering and labels follow source_face_ids/corner_permutations","time":"original logit-normal","optimizer":"fresh AdamW each arm","lr":1e-05,"betas":[0.9,0.95],"weight_decay":0,"clip":1,"scheduler":false,"EMA":false,"dropout":false,"eval_modules":true,"all_parameters_trainable":true,"FC_weight":0.1,"FF_weight":0.1,"aux_scale":"original coordinates and per-coordinate MSE; no bbox normalization","auxiliary_accumulation":"Sum group losses across all8 microbatches / total valid groups across those8; FC/FF separately","checkpoint_additional_steps":[250,500],"checkpoint_cumulative_steps":[750,1000],"diagnostic_steps":[1,100,250,500],"save_optimizer":true,"continue_old_optimizer":false,"no_training_cache":true,"gate":"stored FP32 sampled t >=0.5","arms":["C_gate"]}

门控阈值：0.5000；完整有效组分母保持：True；实际公式：FM + I[t>=.5]*(.1 FC + .1 FF); original full effective8 group denominators; no active-fraction compensation

旧 A/C 结果来源：/mnt/e/MeshFlow/experiments/mf_h1_local0/outputs/t1_interface_3arm_20260929_181500。旧32份评分仅复用，不重算。

| 旧模型 | 已记录step500 SHA256 |
| --- | --- |
| A | 43e56e107b0c6d28766aebd8cf8a0a764d8d52ef62ba2ca6e756d881634f876b |
| C | 8a40651bcf54aebb2c48cd19a2433fb4e0679370c0da6e79e916cc727f067168 |

源报告/结果 SHA 与实际大checkpoint字节复核保留在 results.assets；本轮新 checkpoint 的本地路径和身份如下。

| 实际更新数 | 参数数 | 可训练参数 | 训练秒 | 峰值显存字节 | 同流审计数 |
| --- | --- | --- | --- | --- | --- |
| 500 | 130251267 | 130251267 | 575.4130 | 2709430272 | 500 |

| 检查点 | 累计更新 | 本地路径 | 实际SHA256 | 优化器已存 |
| --- | --- | --- | --- | --- |
| 250 | 750 | /mnt/e/MeshFlow/experiments/mf_h1_local0/outputs/t1_time_gate_20260929_213000/C_gate/additional_step250_cumulative_step750.pt | 232595f1c39e772c4333d30e55fe17248f1409425eae98736b453f39276b7fe5 | True |
| 500 | 1000 | /mnt/e/MeshFlow/experiments/mf_h1_local0/outputs/t1_time_gate_20260929_213000/C_gate/additional_step500_cumulative_step1000.pt | 4e473de1ea35617c1e6a2b938e0a85b4ab006a3a266d6a8ad79360c0c349a3b5 | True |

| 更新 | FM | 门控后FC | 门控后FF | 实际加权辅助 | 总loss | clip前grad范数 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.0002 | 1.412e-05 | 1.016e-05 | 2.429e-06 | 0.0002 | 0.0475 |
| 100 | 0.0018 | 1.069e-05 | 1.237e-05 | 2.305e-06 | 0.0018 | 0.0374 |
| 250 | 0.0003 | 5.498e-06 | 5.196e-06 | 1.069e-06 | 0.0003 | 0.0604 |
| 500 | 0.0010 | 8.058e-06 | 5.154e-06 | 1.321e-06 | 0.0010 | 0.1342 |

少数训练日志对应不同目标、噪声和t，不能据此宣称loss单调下降、收敛或完整生成改善。每条梯度/参数更新审计与门控暴露原始值保留在JSON。

FC/FF日志是门控后实际施加的项；没有额外计算 inactive 样本的未门控关系损失，因此不把它们标为原始FC/FF。

| 更新 | 记录参数 | 梯度RMS | 实际参数更新RMS | 梯度有限 | 更新有限 |
| --- | --- | --- | --- | --- | --- |
| 1 | role_embedding | 4.069e-06 | 9.288e-06 | True | True |
| 1 | backbone.x_embedder.mlp.0.weight | 4.348e-06 | 9.773e-06 | True | True |
| 1 | backbone.t_embedder.mlp.0.weight | 1.728e-05 | 7.739e-06 | True | True |
| 1 | backbone.layers.0.attn.qkv_proj.weight | 4.382e-07 | 6.376e-06 | True | True |
| 1 | backbone.final_layer.linear.weight | 0.0009 | 9.300e-06 | True | True |
| 100 | role_embedding | 5.477e-06 | 2.768e-06 | True | True |
| 100 | backbone.x_embedder.mlp.0.weight | 7.285e-06 | 2.896e-06 | True | True |
| 100 | backbone.t_embedder.mlp.0.weight | 6.691e-06 | 1.570e-06 | True | True |
| 100 | backbone.layers.0.attn.qkv_proj.weight | 5.289e-07 | 2.449e-06 | True | True |
| 100 | backbone.final_layer.linear.weight | 0.0007 | 3.301e-06 | True | True |
| 250 | role_embedding | 2.521e-06 | 1.540e-06 | True | True |
| 250 | backbone.x_embedder.mlp.0.weight | 4.059e-06 | 1.567e-06 | True | True |
| 250 | backbone.t_embedder.mlp.0.weight | 3.912e-05 | 8.673e-07 | True | True |
| 250 | backbone.layers.0.attn.qkv_proj.weight | 4.405e-07 | 1.325e-06 | True | True |
| 250 | backbone.final_layer.linear.weight | 0.0012 | 1.972e-06 | True | True |
| 500 | role_embedding | 9.259e-06 | 2.210e-06 | True | True |
| 500 | backbone.x_embedder.mlp.0.weight | 1.394e-05 | 2.248e-06 | True | True |
| 500 | backbone.t_embedder.mlp.0.weight | 7.211e-05 | 1.361e-06 | True | True |
| 500 | backbone.layers.0.attn.qkv_proj.weight | 1.862e-06 | 2.046e-06 | True | True |
| 500 | backbone.final_layer.linear.weight | 0.0025 | 4.247e-06 | True | True |

门控暴露（仍用完整effective8组分母，不按active比例补偿）：

| 记录 | 样本 | active样本 | FC active/total | FF active/total | 全inactive更新 |
| --- | --- | --- | --- | --- | --- |
| 预登记 | 4000 | 1972 | 17682/36000 | 114376/232000 | [28,330,358] |
| 实际 | 4000 | 1972 | 17682/36000 | 114376/232000 | [28,330,358] |

实现夹具中断记录：1次，发生在模型前向和优化器更新均为0时。合成配对均值为 -1.9999999999999998，相对期望 -2 的 FP64 误差2.22e-16；仅将该测试的精确相等断言改为 rtol=0、atol=1e-12。实际评分、损失和超参数没有改变。

评价合成夹具累计调用3次：1次失败、2次通过（包含独立复查）；原CPU数学12次梯度门禁复用，没有重跑。完整失败堆栈保留在JSON。修正记录：Synthetic scalar expected -2 differed by FP64 roundoff; evaluation arithmetic assertion only. No model, optimizer, noise, hyperparameter, or geometry change; passed12 math-gradient checks reused, not repeated.

GPU范围为模型前向/反向、FM及关系损失与采样积分。沿用的CPU nested OT训练预处理与源索引标签构造不冒称GPU；离线几何审计不参与生成，不修补输出。

实际调用账本：

| 项目 | 实际次数/分账 |
| --- | --- |
| model_forward_attempts | 4800 |
| model_forward_returns | 4800 |
| backward_attempts | 4000 |
| backward_completed | 4000 |
| optimizer_step_attempts | 500 |
| optimizer_updates | 500 |
| training_OT_calls | 4000 |
| training_OT_attempts | 4000 |
| rollout_attempts | 16 |
| complete_rollouts | 16 |
| euler_updates | 800 |
| geometry_guidance_calls | 0 |
| geometry_repairs | 0 |
| formula_autograd_calls | 12 |
| formula_autograd_returns | 12 |
| evaluation_synthetic_fixture_attempts | 3 |
| evaluation_synthetic_fixture_passes | 2 |
| training_active_relation_evaluations | 1972 |
| teacher_forwards | 0 |

## 三模型逐对象完整生成

每个单元为均值 / 中位数 / [Q25,Q75] / 最差值@seed。RMS和p95单位为% bbox，比例使用原评价分母。所有8个seed保留，两个父对象分别报告。

| 模型/对象 | 剩余对称RMS | free→R12 | R12→free | 完整表面对称RMS |
| --- | --- | --- | --- | --- |
| A/P0 | 1.2520 / 0.5759 / [0.2968, 1.6035] / 3.7855@9503 | 1.1751 / 0.7441 / [0.2314, 1.5407] / 3.3166@9503 | 1.2755 / 0.3999 / [0.2470, 1.6598] / 4.2024@9503 | 0.9241 / 0.2867 / [0.1989, 1.3354] / 2.9054@9503 |
| A/P1 | 0.6845 / 0.6590 / [0.4240, 0.9194] / 1.3166@9501 | 0.6459 / 0.6305 / [0.2455, 0.8754] / 1.5644@9501 | 0.6942 / 0.7221 / [0.5118, 0.9082] / 1.2216@9506 | 0.5310 / 0.5967 / [0.2343, 0.7659] / 0.9827@9506 |
| C/P0 | 1.3520 / 0.5631 / [0.2557, 2.2519] / 3.6362@9503 | 1.3144 / 0.7076 / [0.1895, 2.4864] / 3.2708@9508 | 1.3000 / 0.3273 / [0.2557, 1.8291] / 4.0961@9503 | 0.9499 / 0.2210 / [0.1772, 1.4730] / 2.8677@9503 |
| C/P1 | 0.8705 / 0.9697 / [0.4656, 1.1213] / 1.4817@9502 | 0.8249 / 0.9192 / [0.2900, 1.1886] / 1.5315@9502 | 0.8805 / 0.9588 / [0.6238, 1.0711] / 1.4302@9502 | 0.6967 / 0.8398 / [0.2710, 0.8846] / 1.4006@9502 |
| C_gate/P0 | 1.3800 / 0.7372 / [0.2246, 2.3194] / 3.7060@9503 | 1.3511 / 0.8821 / [0.1388, 2.6819] / 3.1797@9503 | 1.3075 / 0.5201 / [0.2566, 1.7347] / 4.1664@9503 | 0.9319 / 0.2412 / [0.1705, 1.4930] / 2.8835@9503 |
| C_gate/P1 | 0.6888 / 0.6532 / [0.3561, 0.9596] / 1.3074@9501 | 0.6548 / 0.5863 / [0.2774, 0.9191] / 1.5576@9501 | 0.6907 / 0.7559 / [0.3788, 0.9535] / 1.1725@9506 | 0.5442 / 0.5429 / [0.2592, 0.8424] / 0.9201@9506 |

| 模型/对象 | Gamma RMS | Gamma p95 | Gamma coverage |
| --- | --- | --- | --- |
| A/P0 | 1.1846 / 0.3611 / [0.2452, 1.7985] / 3.6779@9503 | 2.0087 / 0.3143 / [0.2299, 3.5523] / 6.4770@9503 | 0.2573 / 0.1375 / [0.0058, 0.4702] / 0.0038@9507 |
| A/P1 | 1.3204 / 1.1787 / [0.7312, 1.8999] / 2.7485@9507 | 2.7534 / 2.2725 / [1.0902, 3.6720] / 7.6880@9507 | 0.1983 / 0.0881 / [0.0512, 0.2704] / 0.0019@9508 |
| C/P0 | 1.3246 / 0.5106 / [0.2513, 2.0071] / 4.0952@9503 | 2.2105 / 0.4325 / [0.3433, 4.0036] / 6.8324@9503 | 0.2314 / 0.1135 / [0.0058, 0.3669] / 0.0039@9508 |
| C/P1 | 1.7385 / 1.8631 / [1.2520, 2.2403] / 2.7747@9507 | 3.9793 / 3.7206 / [2.6402, 4.7645] / 7.8015@9507 | 0.0921 / 0.0576 / [0.0072, 0.0696] / 0.0019@9503 |
| C_gate/P0 | 1.5253 / 0.8935 / [0.1370, 2.8764] / 4.1239@9503 | 3.0675 / 1.8937 / [0.2058, 5.7429] / 7.9746@9506 | 0.3035 / 0.2360 / [0.0087, 0.5394] / 0@9502 |
| C_gate/P1 | 1.3109 / 1.1184 / [0.5547, 2.0242] / 2.7707@9507 | 2.6368 / 2.0598 / [0.7750, 3.5349] / 7.7666@9507 | 0.1452 / 0.0800 / [0.0536, 0.1526] / 0.0019@9508 |

| 模型/对象 | CF_proper_affected_free_fraction | CF_proper_pair_count | FF_illegal_pair_count | CF_coplanar_pair_count | FF_coplanar_pair_count | CF_unresolved_or_degenerate_pair_count | FF_unresolved_or_degenerate_pair_count | free_minimum_angle_lt5_fraction | free_q_gt10_fraction | free_near_degenerate_count | free_exact_zero_area_count | nan_coordinate_count | inf_coordinate_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A/P0 | 0.1950 / 0.1850 / [0.1450, 0.2550] / 0.3300@9505 | 28.1250 / 28.5000 / [23.7500, 34.5000] / 47.0000@9505 | 125.0000 / 118.0000 / [104.7500, 141.5000] / 176.0000@9502 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0.1275 / 0.1200 / [0.1000, 0.1500] / 0.2200@9506 | 0.0187 / 0 / [0, 0.0425] / 0.0600@9503 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 |
| A/P1 | 0.1087 / 0.1100 / [0.0875, 0.1350] / 0.1700@9504 | 16.6250 / 16.0000 / [11.0000, 21.5000] / 26.0000@9502 | 112.0000 / 108.5000 / [98.5000, 117.7500] / 145.0000@9505 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0.1962 / 0.1800 / [0.1675, 0.2375] / 0.2600@9501 | 0.0788 / 0.0750 / [0.0600, 0.0925] / 0.1200@9506 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 |
| C/P0 | 0.1725 / 0.1650 / [0.1425, 0.2250] / 0.2900@9504 | 22.2500 / 26.0000 / [19.0000, 28.0000] / 31.0000@9504 | 103.3750 / 87.0000 / [77.7500, 138.5000] / 167.0000@9502 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0.1138 / 0.1150 / [0.0775, 0.1475] / 0.1900@9504 | 0.0150 / 0.0050 / [0, 0.0300] / 0.0500@9508 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 |
| C/P1 | 0.1200 / 0.1100 / [0.0975, 0.1300] / 0.2500@9502 | 16.1250 / 15.5000 / [12.0000, 17.7500] / 31.0000@9502 | 105.5000 / 103.0000 / [79.0000, 135.7500] / 156.0000@9505 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0.1775 / 0.1700 / [0.1400, 0.2350] / 0.2500@9501 | 0.0650 / 0.0600 / [0.0400, 0.0825] / 0.1200@9506 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 |
| C_gate/P0 | 0.1688 / 0.1700 / [0.1300, 0.2050] / 0.3200@9504 | 23.6250 / 27.5000 / [18.0000, 28.7500] / 40.0000@9504 | 103.2500 / 85.5000 / [80.5000, 131.0000] / 155.0000@9502 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0.1200 / 0.1200 / [0.0775, 0.1500] / 0.1900@9506 | 0.0163 / 0.0050 / [0, 0.0325] / 0.0500@9503 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 |
| C_gate/P1 | 0.1200 / 0.1000 / [0.0900, 0.1400] / 0.2500@9502 | 17.2500 / 15.5000 / [12.7500, 18.0000] / 37.0000@9502 | 92.2500 / 89.5000 / [66.2500, 105.0000] / 155.0000@9505 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0.1888 / 0.1700 / [0.1450, 0.2525] / 0.2600@9501 | 0.0688 / 0.0600 / [0.0375, 0.0775] / 0.1600@9506 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 |

## 同噪声配对差与质量代价

差值统一为 C_gate−参考：误差/非法pair负值为降低，coverage正值为增加。改善次数完全沿用原精确差值规则；未用新阈值筛选seed。

| 配对 | 指标 | 差值分布 | 改善 | 平 | 变差 |
| --- | --- | --- | --- | --- | --- |
| C_gate−A/P0 | remainder_symmetric_RMS_bbox_pct | 0.1281 / -0.0131 / [-0.0714, 0.1260] / 0.9774@9502 | 4 | 0 | 4 |
| C_gate−A/P0 | free_to_R12_RMS_bbox_pct | 0.1760 / -0.0329 / [-0.1116, 0.1337] / 1.5594@9502 | 5 | 0 | 3 |
| C_gate−A/P0 | R12_to_free_RMS_bbox_pct | 0.0321 / -0.0189 / [-0.0395, 0.0581] / 0.3293@9506 | 5 | 0 | 3 |
| C_gate−A/P0 | boundary_RMS_bbox_pct | 0.3407 / -0.0099 / [-0.0169, 0.2651] / 2.4188@9506 | 5 | 0 | 3 |
| C_gate−A/P0 | boundary_p95_bbox_pct | 1.0588 / -0.0012 / [-0.0407, 0.4239] / 7.7353@9506 | 4 | 0 | 4 |
| C_gate−A/P0 | boundary_coverage_fraction | 0.0462 / 0.0304 / [-0.0004, 0.0692] / -0.0058@9502 | 6 | 0 | 2 |
| C_gate−A/P0 | CF_proper_affected_free_fraction | -0.0262 / -0.0150 / [-0.0375, 0.0050] / 0.0200@9503 | 5 | 1 | 2 |
| C_gate−A/P0 | FF_illegal_pair_count | -21.7500 / -23.5000 / [-29.2500, -12.0000] / -6.0000@9508 | 8 | 0 | 0 |
| C_gate−A/P0 | free_minimum_angle_lt5_fraction | -0.0075 / 0 / [-0.0150, 0] / 0.0100@9501 | 3 | 4 | 1 |
| C_gate−A/P0 | free_q_gt10_fraction | -0.0025 / 0 / [-0.0025, 0] / 0.0100@9506 | 2 | 5 | 1 |
| C_gate−A/P1 | remainder_symmetric_RMS_bbox_pct | 0.0043 / -0.0010 / [-0.0215, 0.0771] / 0.1298@9502 | 4 | 0 | 4 |
| C_gate−A/P1 | free_to_R12_RMS_bbox_pct | 0.0089 / -0.0013 / [-0.0349, 0.0891] / 0.1736@9502 | 4 | 0 | 4 |
| C_gate−A/P1 | R12_to_free_RMS_bbox_pct | -0.0035 / -0.0007 / [-0.0224, 0.0661] / 0.0791@9502 | 4 | 0 | 4 |
| C_gate−A/P1 | boundary_RMS_bbox_pct | -0.0095 / -0.0221 / [-0.0574, 0.0478] / 0.1832@9505 | 4 | 0 | 4 |
| C_gate−A/P1 | boundary_p95_bbox_pct | -0.1166 / -0.1397 / [-0.2078, 0.0198] / 0.2024@9502 | 5 | 0 | 3 |
| C_gate−A/P1 | boundary_coverage_fraction | -0.0530 / -0.0069 / [-0.0450, 0.0024] / -0.2874@9502 | 3 | 1 | 4 |
| C_gate−A/P1 | CF_proper_affected_free_fraction | 0.0112 / 0 / [0, 0.0075] / 0.1000@9502 | 1 | 5 | 2 |
| C_gate−A/P1 | FF_illegal_pair_count | -19.7500 / -14.5000 / [-42.0000, -6.0000] / 10.0000@9505 | 7 | 0 | 1 |
| C_gate−A/P1 | free_minimum_angle_lt5_fraction | -0.0075 / 0 / [-0.0150, 0] / 0.0200@9506 | 3 | 4 | 1 |
| C_gate−A/P1 | free_q_gt10_fraction | -0.0100 / -0.0200 / [-0.0300, 0.0050] / 0.0400@9506 | 5 | 1 | 2 |
| C_gate−C/P0 | remainder_symmetric_RMS_bbox_pct | 0.0281 / -0.0185 / [-0.0636, 0.1031] / 0.4871@9506 | 5 | 0 | 3 |
| C_gate−C/P0 | free_to_R12_RMS_bbox_pct | 0.0367 / -0.0223 / [-0.0902, 0.1336] / 0.5165@9506 | 5 | 0 | 3 |
| C_gate−C/P0 | R12_to_free_RMS_bbox_pct | 0.0075 / -0.0257 / [-0.0447, 0.0210] / 0.4639@9506 | 5 | 0 | 3 |
| C_gate−C/P0 | boundary_RMS_bbox_pct | 0.2007 / -0.0189 / [-0.0912, 0.0096] / 2.2860@9506 | 5 | 0 | 3 |
| C_gate−C/P0 | boundary_p95_bbox_pct | 0.8570 / -0.0079 / [-0.0940, 0.0148] / 7.5120@9506 | 5 | 0 | 3 |
| C_gate−C/P0 | boundary_coverage_fraction | 0.0721 / 0.0261 / [-0.0015, 0.1292] / -0.0058@9502 | 5 | 1 | 2 |
| C_gate−C/P0 | CF_proper_affected_free_fraction | -0.0037 / 0.0100 / [0.0075, 0.0150] / 0.0300@9504 | 1 | 1 | 6 |
| C_gate−C/P0 | FF_illegal_pair_count | -0.1250 / 0.5000 / [-9.0000, 9.0000] / 17.0000@9507 | 4 | 0 | 4 |
| C_gate−C/P0 | free_minimum_angle_lt5_fraction | 0.0062 / 0.0100 / [-0.0100, 0.0200] / 0.0300@9503 | 3 | 0 | 5 |
| C_gate−C/P0 | free_q_gt10_fraction | 0.0013 / 0 / [-0.0025, 0.0025] / 0.0200@9503 | 2 | 4 | 2 |
| C_gate−C/P1 | remainder_symmetric_RMS_bbox_pct | -0.1817 / -0.0052 / [-0.1634, 0.0663] / 0.0720@9506 | 4 | 0 | 4 |
| C_gate−C/P1 | free_to_R12_RMS_bbox_pct | -0.1701 / 0.0029 / [-0.0825, 0.0561] / 0.1038@9501 | 3 | 0 | 5 |
| C_gate−C/P1 | R12_to_free_RMS_bbox_pct | -0.1898 / -0.0012 / [-0.1979, 0.0344] / 0.0815@9506 | 4 | 0 | 4 |
| C_gate−C/P1 | boundary_RMS_bbox_pct | -0.4276 / -0.0423 / [-0.4361, -0.0172] / 0.1345@9504 | 7 | 0 | 1 |
| C_gate−C/P1 | boundary_p95_bbox_pct | -1.3425 / -0.2438 / [-1.1721, -0.0925] / 0.0662@9504 | 7 | 0 | 1 |
| C_gate−C/P1 | boundary_coverage_fraction | 0.0532 / 0.0171 / [0.0012, 0.0335] / -0.0494@9508 | 6 | 0 | 2 |
| C_gate−C/P1 | CF_proper_affected_free_fraction | -8.674e-19 / 0 / [-0.0100, 0] / 0.0400@9504 | 3 | 4 | 1 |
| C_gate−C/P1 | FF_illegal_pair_count | -13.2500 / -10.5000 / [-16.0000, -2.5000] / 1.0000@9504 | 7 | 0 | 1 |
| C_gate−C/P1 | free_minimum_angle_lt5_fraction | 0.0112 / 0.0100 / [0.0050, 0.0125] / 0.0600@9503 | 2 | 0 | 6 |
| C_gate−C/P1 | free_q_gt10_fraction | 0.0038 / 0 / [-0.0025, 0.0125] / 0.0400@9506 | 2 | 3 | 3 |

| 配对 | seed | Δ剩余RMS | Δfree→R12 | ΔR12→free | ΔGammaRMS | ΔGammap95 | Δcoverage | ΔCFfraction | ΔFFillegal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C_gate−A/P0 | 9501 | 0.0908 | 0.1294 | -0.0053 | -0.0097 | -0.0486 | 0.0085 | -0.0300 | -12 |
| C_gate−A/P0 | 9502 | 0.9774 | 1.5594 | 0.1107 | 0.2048 | 0.4578 | -0.0058 | -0.0100 | -21 |
| C_gate−A/P0 | 9503 | -0.0794 | -0.1369 | -0.0359 | 0.4460 | 0.4125 | 5.112e-05 | 0.0200 | -12 |
| C_gate−A/P0 | 9504 | 0.0144 | -0.0372 | 0.0405 | -0.0166 | -0.0445 | 0.1523 | 0.0200 | -33 |
| C_gate−A/P0 | 9505 | -0.0407 | -0.0286 | -0.0502 | -0.2899 | -0.0395 | 0.1080 | -0.1300 | -36 |
| C_gate−A/P0 | 9506 | 0.2318 | 0.1465 | 0.3293 | 2.4188 | 7.7353 | 0.0562 | -0.0600 | -26 |
| C_gate−A/P0 | 9507 | -0.1011 | -0.1112 | -0.1000 | -0.0101 | 0.0258 | 0.0523 | -0.0200 | -28 |
| C_gate−A/P0 | 9508 | -0.0687 | -0.1131 | -0.0325 | -0.0177 | -0.0282 | -0.0019 | 0 | -6 |
| C_gate−A/P1 | 9501 | -0.0092 | -0.0069 | -0.0134 | -0.0524 | -0.1884 | 0.0057 | 0 | -6 |
| C_gate−A/P1 | 9502 | 0.1298 | 0.1736 | 0.0791 | 0.1244 | 0.2024 | -0.2874 | 0.1000 | -8 |
| C_gate−A/P1 | 9503 | 0.0896 | 0.1117 | 0.0696 | -0.0480 | -0.1595 | -0.1119 | 0.0300 | -6 |
| C_gate−A/P1 | 9504 | -0.1884 | -0.2002 | -0.1775 | -0.2365 | -0.4803 | 0.0019 | 0 | -42 |
| C_gate−A/P1 | 9505 | 0.0730 | 0.0816 | 0.0649 | 0.1832 | -0.1200 | 0.0039 | 0 | 10 |
| C_gate−A/P1 | 9506 | -0.0584 | -0.0697 | -0.0491 | -0.0726 | -0.2659 | -0.0138 | 0 | -21 |
| C_gate−A/P1 | 9507 | -0.0042 | -0.0233 | -0.0019 | 0.0222 | 0.0785 | -0.0228 | -0.0400 | -43 |
| C_gate−A/P1 | 9508 | 0.0021 | 0.0043 | 0.0004 | 0.0037 | 0.0002 | 0 | 0 | -42 |
| C_gate−C/P0 | 9501 | -0.1131 | -0.1554 | -0.0523 | -0.0317 | -0.0701 | 0.1060 | 0.0100 | 12 |
| C_gate−C/P0 | 9502 | 0.2027 | 0.3226 | -0.0149 | 0.0033 | 0.0007 | -0.0058 | 0 | -12 |
| C_gate−C/P0 | 9503 | 0.0699 | 0.0706 | 0.0704 | 0.0287 | 0.0571 | 0 | 0.0100 | -19 |
| C_gate−C/P0 | 9504 | -0.0365 | -0.0332 | -0.0421 | -0.3879 | -0.1657 | 0.1985 | 0.0300 | 8 |
| C_gate−C/P0 | 9505 | -0.0006 | -0.0114 | 0.0046 | -0.0061 | -0.0031 | -0.0058 | 0.0300 | -6 |
| C_gate−C/P0 | 9506 | 0.4871 | 0.5165 | 0.4639 | 2.2860 | 7.5120 | 0.2319 | -0.1300 | -8 |
| C_gate−C/P0 | 9507 | -0.0471 | -0.0685 | -0.0366 | -0.0390 | -0.0127 | 0.0466 | 0.0100 | 17 |
| C_gate−C/P0 | 9508 | -0.3380 | -0.3477 | -0.3328 | -0.2476 | -0.4624 | 0.0057 | 0.0100 | 7 |
| C_gate−C/P1 | 9501 | 0.0694 | 0.1038 | 0.0207 | -0.0315 | -0.2048 | 0.0172 | -0.0100 | -3 |
| C_gate−C/P1 | 9502 | -1.2796 | -1.2940 | -1.2712 | -1.4673 | -3.3927 | 0.3435 | 0 | -50 |
| C_gate−C/P1 | 9503 | -0.2285 | -0.2730 | -0.1846 | -0.0532 | -0.2828 | 0.0772 | 0 | -11 |
| C_gate−C/P1 | 9504 | 0.0652 | 0.0543 | 0.0753 | 0.1345 | 0.0662 | -0.0012 | 0.0400 | 1 |
| C_gate−C/P1 | 9505 | -0.0107 | -0.0190 | -0.0027 | -0.0216 | -0.1117 | 0.0019 | 0 | -1 |
| C_gate−C/P1 | 9506 | 0.0720 | 0.0613 | 0.0815 | -0.0924 | -0.4320 | 0.0171 | -0.0100 | -16 |
| C_gate−C/P1 | 9507 | 0.0004 | 0.0014 | 0.0003 | -0.0040 | -0.0349 | 0.0190 | -0.0200 | -16 |
| C_gate−C/P1 | 9508 | -0.1417 | 0.0043 | -0.2377 | -1.8851 | -6.3474 | -0.0494 | 0 | -10 |

质量均值上升项单列；比较两个保存的边际均值以免把舍入零误写成趋势，原配对计数保持不变。

| 配对 | 父对象 | 质量项 | 参考均值 | 门控均值 | 差 |
| --- | --- | --- | --- | --- | --- |
| C_gate−A | 1 | CF_proper_affected_free_fraction | 0.1087 | 0.1200 | 0.0112 |
| C_gate−A | 1 | CF_proper_pair_count | 16.6250 | 17.2500 | 0.6250 |
| C_gate−C | 0 | CF_proper_pair_count | 22.2500 | 23.6250 | 1.3750 |
| C_gate−C | 0 | free_minimum_angle_lt5_fraction | 0.1138 | 0.1200 | 0.0062 |
| C_gate−C | 0 | free_q_gt10_fraction | 0.0150 | 0.0163 | 0.0013 |
| C_gate−C | 1 | CF_proper_pair_count | 16.1250 | 17.2500 | 1.1250 |
| C_gate−C | 1 | free_minimum_angle_lt5_fraction | 0.1775 | 0.1888 | 0.0113 |
| C_gate−C | 1 | free_q_gt10_fraction | 0.0650 | 0.0688 | 0.0038 |

| 配对 | seed | Δ剩余RMS | 增加的质量代价项 |
| --- | --- | --- | --- |
| C_gate−A/P0 | 9501 | 0.0908 | ["free_minimum_angle_lt5_fraction"] |
| C_gate−A/P0 | 9503 | -0.0794 | ["CF_proper_affected_free_fraction","CF_proper_pair_count"] |
| C_gate−A/P0 | 9504 | 0.0144 | ["CF_proper_affected_free_fraction"] |
| C_gate−A/P0 | 9506 | 0.2318 | ["free_q_gt10_fraction"] |
| C_gate−A/P1 | 9501 | -0.0092 | ["free_q_gt10_fraction"] |
| C_gate−A/P1 | 9502 | 0.1298 | ["CF_proper_affected_free_fraction","CF_proper_pair_count"] |
| C_gate−A/P1 | 9503 | 0.0896 | ["CF_proper_affected_free_fraction","CF_proper_pair_count"] |
| C_gate−A/P1 | 9504 | -0.1884 | ["CF_proper_pair_count"] |
| C_gate−A/P1 | 9505 | 0.0730 | ["FF_illegal_pair_count"] |
| C_gate−A/P1 | 9506 | -0.0584 | ["free_minimum_angle_lt5_fraction","free_q_gt10_fraction"] |
| C_gate−A/P1 | 9508 | 0.0021 | ["CF_proper_pair_count"] |
| C_gate−C/P0 | 9501 | -0.1131 | ["CF_proper_affected_free_fraction","FF_illegal_pair_count","free_minimum_angle_lt5_fraction"] |
| C_gate−C/P0 | 9502 | 0.2027 | ["free_minimum_angle_lt5_fraction"] |
| C_gate−C/P0 | 9503 | 0.0699 | ["CF_proper_affected_free_fraction","CF_proper_pair_count","free_minimum_angle_lt5_fraction","free_q_gt10_fraction"] |
| C_gate−C/P0 | 9504 | -0.0365 | ["CF_proper_affected_free_fraction","CF_proper_pair_count","FF_illegal_pair_count"] |
| C_gate−C/P0 | 9505 | -0.0006 | ["CF_proper_affected_free_fraction"] |
| C_gate−C/P0 | 9506 | 0.4871 | ["free_minimum_angle_lt5_fraction","free_q_gt10_fraction"] |
| C_gate−C/P0 | 9507 | -0.0471 | ["CF_proper_affected_free_fraction","CF_proper_pair_count","FF_illegal_pair_count","free_minimum_angle_lt5_fraction"] |
| C_gate−C/P0 | 9508 | -0.3380 | ["CF_proper_affected_free_fraction","CF_proper_pair_count","FF_illegal_pair_count"] |
| C_gate−C/P1 | 9501 | 0.0694 | ["free_minimum_angle_lt5_fraction","free_q_gt10_fraction"] |
| C_gate−C/P1 | 9502 | -1.2796 | ["CF_proper_pair_count"] |
| C_gate−C/P1 | 9503 | -0.2285 | ["CF_proper_pair_count","free_minimum_angle_lt5_fraction"] |
| C_gate−C/P1 | 9504 | 0.0652 | ["CF_proper_affected_free_fraction","CF_proper_pair_count","FF_illegal_pair_count","free_minimum_angle_lt5_fraction"] |
| C_gate−C/P1 | 9505 | -0.0107 | ["free_minimum_angle_lt5_fraction"] |
| C_gate−C/P1 | 9506 | 0.0720 | ["CF_proper_pair_count","free_minimum_angle_lt5_fraction","free_q_gt10_fraction"] |
| C_gate−C/P1 | 9507 | 0.0004 | ["free_minimum_angle_lt5_fraction","free_q_gt10_fraction"] |

## 同噪声条件绑定四格

复用每个seed两个已生成上下文的完整交叉评分，不额外生成错配样本。既有绑定已成立，8/8正值仅为描述，不能作为新成功证据；考察相对 A/C 的增量。

| 模型 | Delta0 | Delta1 | B | 双向正次数 | 集中度/留一 |
| --- | --- | --- | --- | --- | --- |
| A | 6.7125 / 7.2233 / [6.4260, 7.6853] / 4.0975@9503 | 7.5386 / 8.2200 / [7.1298, 8.5859] / 3.9073@9503 | 7.1255 / 7.8412 / [6.8635, 8.1820] / 4.0024@9503 | 8 | {"net_sum":57.00418950672743,"mean":7.1255236883409285,"maximum_absolute_seed":9502,"maximum_absolute_value":8.446297329222547,"leave_one_out_means":[{"omitted_seed":9501,"mean":7.05975162392198},{"omitted_seed":9502,"mean":6.936841739643555},{"omitted_seed":9503,"mean":7.571679452305418},{"omitted_seed":9504,"mean":6.945062058080766},{"omitted_seed":9505,"mean":6.986822011947871},{"omitted_seed":9506,"mean":7.066107323676634},{"omitted_seed":9507,"mean":6.984435017791843},{"omitted_seed":9508,"mean":7.453490279359362}],"sum_positive":57.00418950672743,"sum_negative":0,"sum_absolute":57.00418950672743,"largest_positive_seed":9502,"largest_positive_value":8.446297329222547,"net_without_largest_positive_seed":48.55789217750488,"net_becomes_nonpositive_without_largest_positive":false,"largest_positive_exceeds_half_net":false,"largest_seed_net_contribution_percent":14.816976440346979,"contribution_percent_status":"DEFINED_POSITIVE_NET_WITH_LIMITED_CANCELLATION","policy":{"relative_net_floor":0.1,"absolute_net_floor_bbox_percentage_points":1e-08,"rule":"Single-seed/net percentage displayed only for positive net >0.1*sum(abs(B)) and >1e-8; otherwise NOT_DEFINED","always_report":"All seed B, positive/negative sums, net, largest positive B, and leave-largest-positive-out net"},"values_by_seed":[{"seed":9501,"value":7.585928139273572},{"seed":9502,"value":8.446297329222547},{"seed":9503,"value":4.002433340589505},{"seed":9504,"value":8.388755100162065},{"seed":9505,"value":8.096435423092327},{"seed":9506,"value":7.54143824099099},{"seed":9507,"value":8.113144382184526},{"seed":9508,"value":4.8297575512119}]} |
| C | 6.7621 / 7.2761 / [6.2775, 7.7703] / 4.5766@9508 | 7.2944 / 8.1643 / [6.9500, 8.5068] / 3.6237@9503 | 7.0283 / 7.6490 / [6.7142, 8.1682] / 4.1257@9503 | 8 | {"net_sum":56.22605567854141,"mean":7.028256959817676,"maximum_absolute_seed":9504,"maximum_absolute_value":8.504865196379184,"leave_one_out_means":[{"omitted_seed":9501,"mean":6.96776991238349},{"omitted_seed":9502,"mean":6.956707016296077},{"omitted_seed":9503,"mean":7.442910478381822},{"omitted_seed":9504,"mean":6.817312926023175},{"omitted_seed":9505,"mean":6.865946739090487},{"omitted_seed":9506,"mean":6.922462725734815},{"omitted_seed":9507,"mean":6.863791056110547},{"omitted_seed":9508,"mean":7.389154824520994}],"sum_positive":56.22605567854141,"sum_negative":0,"sum_absolute":56.22605567854141,"largest_positive_seed":9504,"largest_positive_value":8.504865196379184,"net_without_largest_positive_seed":47.72119048216222,"net_becomes_nonpositive_without_largest_positive":false,"largest_positive_exceeds_half_net":false,"largest_seed_net_contribution_percent":15.12619922159159,"contribution_percent_status":"DEFINED_POSITIVE_NET_WITH_LIMITED_CANCELLATION","policy":{"relative_net_floor":0.1,"absolute_net_floor_bbox_percentage_points":1e-08,"rule":"Single-seed/net percentage displayed only for positive net >0.1*sum(abs(B)) and >1e-8; otherwise NOT_DEFINED","always_report":"All seed B, positive/negative sums, net, largest positive B, and leave-largest-positive-out net"},"values_by_seed":[{"seed":9501,"value":7.451666291856977},{"seed":9502,"value":7.529106564468871},{"seed":9503,"value":4.12568232986865},{"seed":9504,"value":8.504865196379184},{"seed":9505,"value":8.164428504908003},{"seed":9506,"value":7.768816598397706},{"seed":9507,"value":8.179518285767575},{"seed":9508,"value":4.501971906894447}]} |
| C_gate | 6.6337 / 7.0739 / [5.8370, 7.7795] / 4.3359@9503 | 7.5515 / 8.1710 / [7.1061, 8.8155] / 3.9869@9503 | 7.0926 / 7.6846 / [6.7599, 8.1403] / 4.1614@9503 | 8 | {"net_sum":56.74094826596926,"mean":7.092618533246157,"maximum_absolute_seed":9504,"maximum_absolute_value":8.58963581157227,"leave_one_out_means":[{"omitted_seed":9501,"mean":7.026114332606148},{"omitted_seed":9502,"mean":6.989975669952314},{"omitted_seed":9503,"mean":7.511370408191577},{"omitted_seed":9504,"mean":6.878758922056712},{"omitted_seed":9505,"mean":6.947861003381435},{"omitted_seed":9506,"mean":7.050960206321437},{"omitted_seed":9507,"mean":6.928200631553021},{"omitted_seed":9508,"mean":7.407707091906613}],"sum_positive":56.74094826596926,"sum_negative":0,"sum_absolute":56.74094826596926,"largest_positive_seed":9504,"largest_positive_value":8.58963581157227,"net_without_largest_positive_seed":48.15131245439699,"net_becomes_nonpositive_without_largest_positive":false,"largest_positive_exceeds_half_net":false,"largest_seed_net_contribution_percent":15.13833672872182,"contribution_percent_status":"DEFINED_POSITIVE_NET_WITH_LIMITED_CANCELLATION","policy":{"relative_net_floor":0.1,"absolute_net_floor_bbox_percentage_points":1e-08,"rule":"Single-seed/net percentage displayed only for positive net >0.1*sum(abs(B)) and >1e-8; otherwise NOT_DEFINED","always_report":"All seed B, positive/negative sums, net, largest positive B, and leave-largest-positive-out net"},"values_by_seed":[{"seed":9501,"value":7.558147937726229},{"seed":9502,"value":7.811118576303061},{"seed":9503,"value":4.161355408628214},{"seed":9504,"value":8.58963581157227},{"seed":9505,"value":8.105921242299212},{"seed":9506,"value":7.384226821719196},{"seed":9507,"value":8.24354384509811},{"seed":9508,"value":4.886998622622967}]} |

| 配对 | ΔDelta0 | ΔDelta1 | ΔB | B增加次数 | 双向增量均正次数 |
| --- | --- | --- | --- | --- | --- |
| C_gate−A | -0.0787 / 0.0246 / [-0.1161, 0.0934] / -0.8061@9502 | 0.0129 / 0.0395 / [-0.0814, 0.1016] / -0.4642@9502 | -0.0329 / 0.0334 / [-0.0601, 0.1375] / -0.6352@9502 | 5 | 3 |
| C_gate−C | -0.1283 / -0.0172 / [-0.3443, 0.0975] / -0.6583@9502 | 0.2571 / 0.0976 / [0.0451, 0.3989] / -0.2674@9506 | 0.0644 / 0.0744 / [0.0121, 0.1504] / -0.3846@9506 | 6 | 4 |

| 模型 | seed | E00 | E01 | E11 | E10 | Delta0 | Delta1 | B | 双向正 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 9501 | 0.6499 | 7.9424 | 1.3166 | 9.1959 | 7.2926 | 7.8793 | 7.5859 | True |
| A | 9502 | 1.0246 | 7.9999 | 0.0723 | 9.9897 | 6.9753 | 9.9173 | 8.4463 | True |
| A | 9503 | 3.7855 | 7.8830 | 0.7643 | 4.6717 | 4.0975 | 3.9073 | 4.0024 | True |
| A | 9504 | 0.2078 | 8.2321 | 0.5537 | 9.3070 | 8.0242 | 8.7533 | 8.3888 | True |
| A | 9505 | 0.1794 | 7.8611 | 0.8458 | 9.3571 | 7.6816 | 8.5112 | 8.0964 | True |
| A | 9506 | 0.5019 | 7.6559 | 1.1402 | 9.0690 | 7.1540 | 7.9288 | 7.5414 | True |
| A | 9507 | 0.3265 | 8.0227 | 0.4566 | 8.9867 | 7.6962 | 8.5301 | 8.1131 | True |
| A | 9508 | 3.3401 | 8.1183 | 0.3264 | 5.2077 | 4.7782 | 4.8813 | 4.8298 | True |
| C | 9501 | 0.8537 | 7.9661 | 1.2380 | 9.0289 | 7.1124 | 7.7909 | 7.4517 | True |
| C | 9502 | 1.7994 | 8.6268 | 1.4817 | 9.7125 | 6.8274 | 8.2308 | 7.5291 | True |
| C | 9503 | 3.6362 | 8.2639 | 1.0825 | 4.7061 | 4.6277 | 3.6237 | 4.1257 | True |
| C | 9504 | 0.2587 | 8.1715 | 0.3002 | 9.3970 | 7.9128 | 9.0969 | 8.5049 | True |
| C | 9505 | 0.1394 | 7.9986 | 0.9295 | 9.3991 | 7.8592 | 8.4696 | 8.1644 | True |
| C | 9506 | 0.2466 | 7.6864 | 1.0098 | 9.1076 | 7.4398 | 8.0978 | 7.7688 | True |
| C | 9507 | 0.2725 | 8.0131 | 0.4520 | 9.0704 | 7.7407 | 8.6184 | 8.1795 | True |
| C | 9508 | 3.6095 | 8.1860 | 0.4701 | 4.8975 | 4.5766 | 4.4274 | 4.5020 | True |
| C_gate | 9501 | 0.7406 | 7.9504 | 1.3074 | 9.2139 | 7.2098 | 7.9065 | 7.5581 | True |
| C_gate | 9502 | 2.0021 | 8.1712 | 0.2021 | 9.6552 | 6.1691 | 9.4531 | 7.8111 | True |
| C_gate | 9503 | 3.7060 | 8.0419 | 0.8539 | 4.8408 | 4.3359 | 3.9869 | 4.1614 | True |
| C_gate | 9504 | 0.2222 | 8.2331 | 0.3654 | 9.5338 | 8.0109 | 9.1684 | 8.5896 | True |
| C_gate | 9505 | 0.1388 | 7.9151 | 0.9188 | 9.3544 | 7.7763 | 8.4355 | 8.1059 | True |
| C_gate | 9506 | 0.7337 | 7.6717 | 1.0818 | 8.9123 | 6.9380 | 7.8305 | 7.3842 | True |
| C_gate | 9507 | 0.2254 | 8.0145 | 0.4524 | 9.1503 | 7.7892 | 8.6979 | 8.2435 | True |
| C_gate | 9508 | 3.2715 | 8.1123 | 0.3285 | 5.2617 | 4.8408 | 4.9332 | 4.8870 | True |

## 接口拓扑与原始输出保护

| 模型/对象 | C全保持 | N112/free100全保持 | exact_C_free_shared_edge_count | exact_C_free_shared_vertex_count | C_inherited_nonmanifold_vertex_count | new_free_related_nonmanifold_vertex_count | free_related_current_winding_conflict_count | shared_edge_component_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A/P0 | True | True | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 101.0000 / 101.0000 / [101.0000, 101.0000] / 101.0000@9501 |
| A/P1 | True | True | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 2.0000 / 2.0000 / [2.0000, 2.0000] / 2.0000@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 101.0000 / 101.0000 / [101.0000, 101.0000] / 101.0000@9501 |
| C/P0 | True | True | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 101.0000 / 101.0000 / [101.0000, 101.0000] / 101.0000@9501 |
| C/P1 | True | True | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 2.0000 / 2.0000 / [2.0000, 2.0000] / 2.0000@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 101.0000 / 101.0000 / [101.0000, 101.0000] / 101.0000@9501 |
| C_gate/P0 | True | True | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 101.0000 / 101.0000 / [101.0000, 101.0000] / 101.0000@9501 |
| C_gate/P1 | True | True | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 2.0000 / 2.0000 / [2.0000, 2.0000] / 2.0000@9501 | 0 / 0 / [0, 0] / 0@9501 | 0 / 0 / [0, 0] / 0@9501 | 101.0000 / 101.0000 / [101.0000, 101.0000] / 101.0000@9501 |

输入继承的C异常与新增free相关异常分开。孤立三角形的合法link不能证明有效连接曲面，精确共边缺失也不是肉眼裂缝大小。CF/FF/CC完整静态数值检测不等于精确谓词或连续碰撞认证。未执行项目保持NOT_RUN，不用轻量scorer替代完整审计。

## 固定 seed9501 图

P0/P1各显示 A、C、C_gate 的full112/free100，共12面板；相机与完整坐标范围相同，蓝色已知C，灰色自由面。图不根据得分选择。

![固定9501完整与自由面](fixed_9501_A_C_C_gate_full_and_free.png)

## 成本、保护与停止

A/C为旧16+16输出复用，不能计入本轮新增生成。实际新增生成数按 C_gate 记录与调用账本，门禁/失败/中断分别保留。

| 耗时项目 | 秒数 |
| --- | --- |
| assets_seconds | 23.3391 |
| sampling_seconds | 76.9431 |
| offline_audit_seconds | 38.7547 |
| training_seconds | 575.4130 |
| checkpoint_save_and_hash_seconds | 20.2914 |

| 最终只读/保护审计 | 状态 | 检查数量 |
| --- | --- | --- |
| status | PASS | NOT_RUN / UNKNOWN |
| training | NOT_RUN / UNKNOWN | NOT_RUN / UNKNOWN |
| raw | NOT_RUN / UNKNOWN | NOT_RUN / UNKNOWN |
| protected_metadata_independent | NOT_RUN / UNKNOWN | 4286 |
| method | Independent readers used saved JSON/NPZ only; no extra model, OT, geometry scoring, autograd, or optimizer calls. | NOT_RUN / UNKNOWN |
| after | PASS | 4286 |
| critical_files_after | PASS | 62 |
| execution_after | PASS | 4 |

执行门禁状态：{"math":"PASS","evaluation_arithmetic":"PASS","stream_identity":"PASS","gate_exposure":"PASS"}

后续建议：先保留固定模型与本轮结果，纯A仍作为主要精度参照；下一轮若授权，优先独立新噪声确认，不组合新模块、不自动加步数。若要归因时间分配本身，还需独立区分降低总辅助曝光的作用。。本轮没有执行。

训练固定预算结束后停止；不按结果追加训练、更换损失、重抽噪声或添加组合臂。
