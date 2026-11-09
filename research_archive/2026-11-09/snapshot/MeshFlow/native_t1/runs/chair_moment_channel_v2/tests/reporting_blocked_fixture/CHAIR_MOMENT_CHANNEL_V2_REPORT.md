# Chair Moment Channel Learning V2

**QUALITY_RESULT：NA_NOT_EVALUATED**。IMPLEMENTED：IMPLEMENTATION_BLOCKED；TRAINED：NOT_RUN；EVALUATED：NOT_RUN。

共同起点为 Hybrid OT_HYBRID / pilot1000 / conditional_total5000 / λ=.25。H_CONT 与 MOMENT_XY 各自最多继续 1000 更新；正式终点必须是 moment_step1000 / conditional_total6000。旧起点的 total5000 不是新终点。两臂继承同一旧 AdamW 状态，MOMENT 的新增参数和优化器状态单列；保持 Geo、FM、λ=.25 及默认模型不变。

本轮检验可学习质心矩关系通道能否超过等预算续训，并保留共同起点质量。V1/r1 的有限表面积分失败不充当 V2 门禁，也不能预设本轮模型优劣。

## 实际执行与预算

下列数字来自本轮 accounting.json 的实际账本摘要。初始化两次完整 Euler50 回放单列；138 个 START 历史缓存不计新生成。额外 I/P 前向为 0。未消耗余量不授权追加科学任务。

```json
{}
```

|标签|新采样|历史缓存|原 NA|采样计时样本|采样秒均值|采样秒中位数|
|---|---|---|---|---|---|---|
|START|0|0|0|0|NA|NA|
|H_CONT_END|0|0|0|0|NA|NA|
|MOMENT_XY_END|0|0|0|0|NA|NA|

训练总耗时、每更新耗时与显存作用域保留在 paired_results/cost_summary.json 的实际 training_summary 和 stage 字段；初始化回放不混入正式 endpoint 行。采样缓存耗时不作为模型吞吐。

NA：尚无对应实测行。

初始化回放结果：NOT_RUN；完成数：NA；采样 NFE：NA。

## 数值合同与成本门禁

本轮总门禁：**IMPLEMENTATION_BLOCKED**；真实预检：NOT_RUN。门禁未通过或固定终点未齐全时，正式方法质量结论为 NA。

FP32 新接受合同为 e_inf=max|Δ|/max(1,max|v_ref|)≤5e-5，e_rms=RMS(Δ)/max(1,RMS(v_ref))≤1e-5。旧逐元素 atol=1e-6、rtol=1e-5 的 flags 原样并列保留；新归一化接受不改写旧 FAIL。BF16 非零置换及长度变化只作数值披露，不套 FP32 门槛。两精度零初始化输出及同形状 padding 内容改变必须逐字节相等，无效 padding 输入梯度必须为零。

NA：尚无对应实测行。

NA：尚无对应实测行。

NA：尚无对应实测行。

完整逐层定位、首次差异与梯度接线以 precheck/result.json、保存数组及 CPU 测试原表为准。不能只因旧 H 有置换误差就把 MOMENT 的全部误差归给旧 backbone；同形状 padding 不变性和零输入梯度是直接执行合同。

NA：尚无对应实测行。

成本要求每个原任务完整 forward 和 forward+backward 均不超过 H_CONT 的 3 倍，并保留至少 15% 显存。正式 H/M 使用预注册重复；MOMENT_V1_REFERENCE 只是一轮等价优化的旧实现成本参照，单次测量不足以给出稳健排名。计时保留重复值、median/min/max；不能把并发墙钟当成孤立算子性能。门禁成本状态：NOT_RUN。

## 固定主终点与起点保真

fixed_query_MSE = mean([d(q,F)/L0]^2)，q 是原冻结 512 个 GT free coverage 查询点。主面板仅 A20 / validation / alpha=1，此时 L=L0。先在每父内平均两份原噪声，再对 8 父等权。配对差中位数从逐父差计算；缺失噪声不缩小分母。两份采样噪声不代表两个独立训练随机流。

|比较（后−前）|前值|后值|父均差|配对差中位数|改善/8|最差父|最差差|有效/8|开发性几何候选|
|---|---|---|---|---|---|---|---|---|---|
|H_CONT_END-START|NA|NA|NA|NA|NA/8|NA|NA|0/8|NA|
|MOMENT_XY_END-H_CONT_END|NA|NA|NA|NA|NA/8|NA|NA|0/8|NA|
|MOMENT_XY_END-START|NA|NA|NA|NA|NA/8|NA|NA|0/8|NA|

几何候选必须同时满足父均差<0、逐父差中位数<0、改善≥6/8，且 8 父两噪声齐全。这是开发性筛选，不是显著性检验。MOMENT−H_CONT 测新增通道效应；MOMENT−START 直接检查保真；只击败退化对照不等于改善起点。

|比较|噪声|前值|后值|父均差|配对差中位数|改善父|有效/8|
|---|---|---|---|---|---|---|---|
|H_CONT_END-START|0|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|1|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|0|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|1|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|0|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|1|NA|NA|NA|NA|NA|0/8|

## 几何方向与完整质量代价

以下为真实原方向值：GT→F = R12_to_free_RMS_bbox_pct/100；F→GT = free_to_R12_RMS_bbox_pct/100。原双向 RMS 使用另一组冻结 GT 点和预测表面积采样，不能与主终点 MSE、coverage 查询点或 RMS 的平方混用。各方向先逐输出计算，再噪声均值、父等权。

|A20 validation 标签|主 MSE|GT→F RMS/L|F→GT RMS/L|原双向 RMS/L|预测小角率|同 GT 小角率|小角率绝对差|
|---|---|---|---|---|---|---|---|
|START|NA|NA|NA|NA|NA|NA|NA|
|H_CONT_END|NA|NA|NA|NA|NA|NA|NA|
|MOMENT_XY_END|NA|NA|NA|NA|NA|NA|NA|

小角参照来自每个完全相同的 GT free 补集及原评分公式。小角率升降、或更接近 GT 分布，均不能独立证明完整网格质量；角度分位数、退化计数和 GT 角度数组另行保留。

|比较|面板/划分/比例|质量项|前值|后值|配对差|改善父|退化父|有效/登记|
|---|---|---|---|---|---|---|---|---|
|H_CONT_END-START|A/train/0.2|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/train/0.2|Gamma_to_free_edges_coverage_0.001_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/train/0.2|CF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/train/0.2|FF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/train/0.2|CF_proper_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/train/0.2|FF_illegal_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/train/0.2|free_minimum_angle_lt5_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/train/0.2|free_near_degenerate_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/train/0.2|fixed_query_p95_L|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/train/0.2|coverage_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/train/0.2|GT_to_F_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/train/0.2|F_to_GT_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/train/0.2|bidirectional_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/train/0.4|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/train/0.4|Gamma_to_free_edges_coverage_0.001_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/train/0.4|CF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/train/0.4|FF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/train/0.4|CF_proper_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/train/0.4|FF_illegal_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/train/0.4|free_minimum_angle_lt5_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/train/0.4|free_near_degenerate_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/train/0.4|fixed_query_p95_L|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/train/0.4|coverage_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/train/0.4|GT_to_F_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/train/0.4|F_to_GT_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/train/0.4|bidirectional_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/validation/0.2|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/validation/0.2|Gamma_to_free_edges_coverage_0.001_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/validation/0.2|CF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/validation/0.2|FF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/validation/0.2|CF_proper_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/validation/0.2|FF_illegal_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/validation/0.2|free_minimum_angle_lt5_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/validation/0.2|free_near_degenerate_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/validation/0.2|fixed_query_p95_L|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/validation/0.2|coverage_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/validation/0.2|GT_to_F_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/validation/0.2|F_to_GT_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/validation/0.2|bidirectional_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/validation/0.4|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/validation/0.4|Gamma_to_free_edges_coverage_0.001_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/validation/0.4|CF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/validation/0.4|FF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/validation/0.4|CF_proper_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/validation/0.4|FF_illegal_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/validation/0.4|free_minimum_angle_lt5_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/validation/0.4|free_near_degenerate_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/validation/0.4|fixed_query_p95_L|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/validation/0.4|coverage_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/validation/0.4|GT_to_F_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/validation/0.4|F_to_GT_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|A/validation/0.4|bidirectional_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|C/holdout/0.2|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|C/holdout/0.2|Gamma_to_free_edges_coverage_0.001_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|C/holdout/0.2|CF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|C/holdout/0.2|FF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|C/holdout/0.2|CF_proper_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|C/holdout/0.2|FF_illegal_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|C/holdout/0.2|free_minimum_angle_lt5_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|C/holdout/0.2|free_near_degenerate_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|C/holdout/0.2|fixed_query_p95_L|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|C/holdout/0.2|coverage_fraction|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|C/holdout/0.2|GT_to_F_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|C/holdout/0.2|F_to_GT_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|H_CONT_END-START|C/holdout/0.2|bidirectional_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.2|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.2|Gamma_to_free_edges_coverage_0.001_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.2|CF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.2|FF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.2|CF_proper_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.2|FF_illegal_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.2|free_minimum_angle_lt5_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.2|free_near_degenerate_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.2|fixed_query_p95_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.2|coverage_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.2|GT_to_F_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.2|F_to_GT_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.2|bidirectional_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.4|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.4|Gamma_to_free_edges_coverage_0.001_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.4|CF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.4|FF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.4|CF_proper_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.4|FF_illegal_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.4|free_minimum_angle_lt5_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.4|free_near_degenerate_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.4|fixed_query_p95_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.4|coverage_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.4|GT_to_F_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.4|F_to_GT_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.4|bidirectional_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.2|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.2|Gamma_to_free_edges_coverage_0.001_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.2|CF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.2|FF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.2|CF_proper_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.2|FF_illegal_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.2|free_minimum_angle_lt5_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.2|free_near_degenerate_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.2|fixed_query_p95_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.2|coverage_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.2|GT_to_F_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.2|F_to_GT_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.2|bidirectional_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.4|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.4|Gamma_to_free_edges_coverage_0.001_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.4|CF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.4|FF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.4|CF_proper_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.4|FF_illegal_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.4|free_minimum_angle_lt5_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.4|free_near_degenerate_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.4|fixed_query_p95_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.4|coverage_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.4|GT_to_F_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.4|F_to_GT_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.4|bidirectional_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|C/holdout/0.2|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|C/holdout/0.2|Gamma_to_free_edges_coverage_0.001_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|C/holdout/0.2|CF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|C/holdout/0.2|FF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|C/holdout/0.2|CF_proper_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|C/holdout/0.2|FF_illegal_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|C/holdout/0.2|free_minimum_angle_lt5_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|C/holdout/0.2|free_near_degenerate_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|C/holdout/0.2|fixed_query_p95_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|C/holdout/0.2|coverage_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|C/holdout/0.2|GT_to_F_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|C/holdout/0.2|F_to_GT_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-H_CONT_END|C/holdout/0.2|bidirectional_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/train/0.2|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/train/0.2|Gamma_to_free_edges_coverage_0.001_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/train/0.2|CF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/train/0.2|FF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/train/0.2|CF_proper_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/train/0.2|FF_illegal_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/train/0.2|free_minimum_angle_lt5_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/train/0.2|free_near_degenerate_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/train/0.2|fixed_query_p95_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/train/0.2|coverage_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/train/0.2|GT_to_F_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/train/0.2|F_to_GT_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/train/0.2|bidirectional_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/train/0.4|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/train/0.4|Gamma_to_free_edges_coverage_0.001_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/train/0.4|CF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/train/0.4|FF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/train/0.4|CF_proper_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/train/0.4|FF_illegal_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/train/0.4|free_minimum_angle_lt5_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/train/0.4|free_near_degenerate_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/train/0.4|fixed_query_p95_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/train/0.4|coverage_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/train/0.4|GT_to_F_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/train/0.4|F_to_GT_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/train/0.4|bidirectional_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/validation/0.2|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/validation/0.2|Gamma_to_free_edges_coverage_0.001_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/validation/0.2|CF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/validation/0.2|FF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/validation/0.2|CF_proper_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/validation/0.2|FF_illegal_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/validation/0.2|free_minimum_angle_lt5_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/validation/0.2|free_near_degenerate_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/validation/0.2|fixed_query_p95_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/validation/0.2|coverage_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/validation/0.2|GT_to_F_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/validation/0.2|F_to_GT_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/validation/0.2|bidirectional_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/validation/0.4|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/validation/0.4|Gamma_to_free_edges_coverage_0.001_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/validation/0.4|CF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/validation/0.4|FF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/validation/0.4|CF_proper_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/validation/0.4|FF_illegal_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/validation/0.4|free_minimum_angle_lt5_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/validation/0.4|free_near_degenerate_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/validation/0.4|fixed_query_p95_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/validation/0.4|coverage_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/validation/0.4|GT_to_F_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/validation/0.4|F_to_GT_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|A/validation/0.4|bidirectional_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|C/holdout/0.2|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|C/holdout/0.2|Gamma_to_free_edges_coverage_0.001_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|C/holdout/0.2|CF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|C/holdout/0.2|FF_unresolved_or_degenerate_pair_count|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|C/holdout/0.2|CF_proper_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|C/holdout/0.2|FF_illegal_affected_free_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|C/holdout/0.2|free_minimum_angle_lt5_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|C/holdout/0.2|free_near_degenerate_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|C/holdout/0.2|fixed_query_p95_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|C/holdout/0.2|coverage_fraction|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|C/holdout/0.2|GT_to_F_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|C/holdout/0.2|F_to_GT_original_RMS_L|NA|NA|NA|NA|NA|0/8|
|MOMENT_XY_END-START|C/holdout/0.2|bidirectional_original_RMS_L|NA|NA|NA|NA|NA|0/8|

A train/validation 的 20%/40% 独立列出；原不可构造 40% 项保留 NA。C 是此前已观察的留出回归面板，不是全新独立测试，也不参与选择。C/N/finite、X/Y/Z span 与 median、Gamma、FF/CF unresolved 和退化角度的所有字段见逐输出与配对 CSV；严格全局 watertight 审计为 NOT_RUN。

coverage_partition.csv 使用相同冻结查询逐点拆成 both_in / enter / leave / both_out，分区贡献为平方距离差总和除原完整 Q，四区精确闭合总 MSE 差。新增覆盖必须同距离尾部、接口和相交代价一起评估。

## 宽度响应与固定目标列

W 为 free 三角角点 X 的 q95−q05。r_base=W(F1)/W(Y1)；b_self=[W(F1.1)−W(F.9)]/[.2W(F1)]；b_GT 的分母换为 .2W(Y1)。每条父/噪声曲线先算 |b−1|，再父内噪声平均、父等权，不能先平均有正负偏差的斜率。

|比较|划分|候选|指标|前值|后值|父均差|配对差中位数|有效/登记|
|---|---|---|---|---|---|---|---|---|
|H_CONT_END-START|train|MODEL|b_self_abs_error|NA|NA|NA|NA|0/4|
|H_CONT_END-START|train|MODEL|b_GT_abs_error|NA|NA|NA|NA|0/4|
|H_CONT_END-START|train|MODEL|baseline_abs_ratio_error|NA|NA|NA|NA|0/4|
|H_CONT_END-START|train|MODEL|mean_A_column|NA|NA|NA|NA|0/4|
|H_CONT_END-START|train|MODEL|all_three_strict_win|NA|NA|NA|NA|0/4|
|H_CONT_END-START|validation|MODEL|b_self_abs_error|NA|NA|NA|NA|0/4|
|H_CONT_END-START|validation|MODEL|b_GT_abs_error|NA|NA|NA|NA|0/4|
|H_CONT_END-START|validation|MODEL|baseline_abs_ratio_error|NA|NA|NA|NA|0/4|
|H_CONT_END-START|validation|MODEL|mean_A_column|NA|NA|NA|NA|0/4|
|H_CONT_END-START|validation|MODEL|all_three_strict_win|NA|NA|NA|NA|0/4|
|H_CONT_END-START|train|No-response|b_self_abs_error|NA|NA|NA|NA|0/4|
|H_CONT_END-START|train|No-response|b_GT_abs_error|NA|NA|NA|NA|0/4|
|H_CONT_END-START|train|No-response|baseline_abs_ratio_error|NA|NA|NA|NA|0/4|
|H_CONT_END-START|train|No-response|mean_A_column|NA|NA|NA|NA|0/4|
|H_CONT_END-START|train|No-response|all_three_strict_win|NA|NA|NA|NA|0/4|
|H_CONT_END-START|validation|No-response|b_self_abs_error|NA|NA|NA|NA|0/4|
|H_CONT_END-START|validation|No-response|b_GT_abs_error|NA|NA|NA|NA|0/4|
|H_CONT_END-START|validation|No-response|baseline_abs_ratio_error|NA|NA|NA|NA|0/4|
|H_CONT_END-START|validation|No-response|mean_A_column|NA|NA|NA|NA|0/4|
|H_CONT_END-START|validation|No-response|all_three_strict_win|NA|NA|NA|NA|0/4|
|H_CONT_END-START|train|Post-scale|b_self_abs_error|NA|NA|NA|NA|0/4|
|H_CONT_END-START|train|Post-scale|b_GT_abs_error|NA|NA|NA|NA|0/4|
|H_CONT_END-START|train|Post-scale|baseline_abs_ratio_error|NA|NA|NA|NA|0/4|
|H_CONT_END-START|train|Post-scale|mean_A_column|NA|NA|NA|NA|0/4|
|H_CONT_END-START|train|Post-scale|all_three_strict_win|NA|NA|NA|NA|0/4|
|H_CONT_END-START|validation|Post-scale|b_self_abs_error|NA|NA|NA|NA|0/4|
|H_CONT_END-START|validation|Post-scale|b_GT_abs_error|NA|NA|NA|NA|0/4|
|H_CONT_END-START|validation|Post-scale|baseline_abs_ratio_error|NA|NA|NA|NA|0/4|
|H_CONT_END-START|validation|Post-scale|mean_A_column|NA|NA|NA|NA|0/4|
|H_CONT_END-START|validation|Post-scale|all_three_strict_win|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|train|MODEL|b_self_abs_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|train|MODEL|b_GT_abs_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|train|MODEL|baseline_abs_ratio_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|train|MODEL|mean_A_column|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|train|MODEL|all_three_strict_win|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|validation|MODEL|b_self_abs_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|validation|MODEL|b_GT_abs_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|validation|MODEL|baseline_abs_ratio_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|validation|MODEL|mean_A_column|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|validation|MODEL|all_three_strict_win|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|train|No-response|b_self_abs_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|train|No-response|b_GT_abs_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|train|No-response|baseline_abs_ratio_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|train|No-response|mean_A_column|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|train|No-response|all_three_strict_win|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|validation|No-response|b_self_abs_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|validation|No-response|b_GT_abs_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|validation|No-response|baseline_abs_ratio_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|validation|No-response|mean_A_column|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|validation|No-response|all_three_strict_win|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|train|Post-scale|b_self_abs_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|train|Post-scale|b_GT_abs_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|train|Post-scale|baseline_abs_ratio_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|train|Post-scale|mean_A_column|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|train|Post-scale|all_three_strict_win|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|validation|Post-scale|b_self_abs_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|validation|Post-scale|b_GT_abs_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|validation|Post-scale|baseline_abs_ratio_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|validation|Post-scale|mean_A_column|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-H_CONT_END|validation|Post-scale|all_three_strict_win|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|train|MODEL|b_self_abs_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|train|MODEL|b_GT_abs_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|train|MODEL|baseline_abs_ratio_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|train|MODEL|mean_A_column|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|train|MODEL|all_three_strict_win|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|validation|MODEL|b_self_abs_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|validation|MODEL|b_GT_abs_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|validation|MODEL|baseline_abs_ratio_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|validation|MODEL|mean_A_column|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|validation|MODEL|all_three_strict_win|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|train|No-response|b_self_abs_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|train|No-response|b_GT_abs_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|train|No-response|baseline_abs_ratio_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|train|No-response|mean_A_column|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|train|No-response|all_three_strict_win|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|validation|No-response|b_self_abs_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|validation|No-response|b_GT_abs_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|validation|No-response|baseline_abs_ratio_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|validation|No-response|mean_A_column|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|validation|No-response|all_three_strict_win|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|train|Post-scale|b_self_abs_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|train|Post-scale|b_GT_abs_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|train|Post-scale|baseline_abs_ratio_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|train|Post-scale|mean_A_column|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|train|Post-scale|all_three_strict_win|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|validation|Post-scale|b_self_abs_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|validation|Post-scale|b_GT_abs_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|validation|Post-scale|baseline_abs_ratio_error|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|validation|Post-scale|mean_A_column|NA|NA|NA|NA|0/4|
|MOMENT_XY_END-START|validation|Post-scale|all_three_strict_win|NA|NA|NA|NA|0/4|

E[alpha,beta] 行是生成输入，列是评分目标。固定 beta 比较正确输入与另外两输入：A_beta=mean(E[alpha≠beta,beta])−E[beta,beta]。严格胜要求正确输入同时小于两错误输入；平局不算胜。验证共 4 父×2 噪声×3 目标=24 列，每噪声 12 列；全部三列正确/同一输入通吃以 8 条曲线为分母，每噪声 4 条。列不能冒充 24 个独立父。

|标签|划分|候选|噪声|严格列胜/登记|三列全胜/登记|同输入通吃/登记|有效列|平均 A|
|---|---|---|---|---|---|---|---|---|
|START|train|MODEL|all|NA/24|NA/8|NA/8|0|NA|
|START|train|MODEL|0|NA/12|NA/4|NA/4|0|NA|
|START|train|MODEL|1|NA/12|NA/4|NA/4|0|NA|
|START|train|No-response|all|NA/24|NA/8|NA/8|0|NA|
|START|train|No-response|0|NA/12|NA/4|NA/4|0|NA|
|START|train|No-response|1|NA/12|NA/4|NA/4|0|NA|
|START|train|Post-scale|all|NA/24|NA/8|NA/8|0|NA|
|START|train|Post-scale|0|NA/12|NA/4|NA/4|0|NA|
|START|train|Post-scale|1|NA/12|NA/4|NA/4|0|NA|
|START|validation|MODEL|all|NA/24|NA/8|NA/8|0|NA|
|START|validation|MODEL|0|NA/12|NA/4|NA/4|0|NA|
|START|validation|MODEL|1|NA/12|NA/4|NA/4|0|NA|
|START|validation|No-response|all|NA/24|NA/8|NA/8|0|NA|
|START|validation|No-response|0|NA/12|NA/4|NA/4|0|NA|
|START|validation|No-response|1|NA/12|NA/4|NA/4|0|NA|
|START|validation|Post-scale|all|NA/24|NA/8|NA/8|0|NA|
|START|validation|Post-scale|0|NA/12|NA/4|NA/4|0|NA|
|START|validation|Post-scale|1|NA/12|NA/4|NA/4|0|NA|
|H_CONT_END|train|MODEL|all|NA/24|NA/8|NA/8|0|NA|
|H_CONT_END|train|MODEL|0|NA/12|NA/4|NA/4|0|NA|
|H_CONT_END|train|MODEL|1|NA/12|NA/4|NA/4|0|NA|
|H_CONT_END|train|No-response|all|NA/24|NA/8|NA/8|0|NA|
|H_CONT_END|train|No-response|0|NA/12|NA/4|NA/4|0|NA|
|H_CONT_END|train|No-response|1|NA/12|NA/4|NA/4|0|NA|
|H_CONT_END|train|Post-scale|all|NA/24|NA/8|NA/8|0|NA|
|H_CONT_END|train|Post-scale|0|NA/12|NA/4|NA/4|0|NA|
|H_CONT_END|train|Post-scale|1|NA/12|NA/4|NA/4|0|NA|
|H_CONT_END|validation|MODEL|all|NA/24|NA/8|NA/8|0|NA|
|H_CONT_END|validation|MODEL|0|NA/12|NA/4|NA/4|0|NA|
|H_CONT_END|validation|MODEL|1|NA/12|NA/4|NA/4|0|NA|
|H_CONT_END|validation|No-response|all|NA/24|NA/8|NA/8|0|NA|
|H_CONT_END|validation|No-response|0|NA/12|NA/4|NA/4|0|NA|
|H_CONT_END|validation|No-response|1|NA/12|NA/4|NA/4|0|NA|
|H_CONT_END|validation|Post-scale|all|NA/24|NA/8|NA/8|0|NA|
|H_CONT_END|validation|Post-scale|0|NA/12|NA/4|NA/4|0|NA|
|H_CONT_END|validation|Post-scale|1|NA/12|NA/4|NA/4|0|NA|
|MOMENT_XY_END|train|MODEL|all|NA/24|NA/8|NA/8|0|NA|
|MOMENT_XY_END|train|MODEL|0|NA/12|NA/4|NA/4|0|NA|
|MOMENT_XY_END|train|MODEL|1|NA/12|NA/4|NA/4|0|NA|
|MOMENT_XY_END|train|No-response|all|NA/24|NA/8|NA/8|0|NA|
|MOMENT_XY_END|train|No-response|0|NA/12|NA/4|NA/4|0|NA|
|MOMENT_XY_END|train|No-response|1|NA/12|NA/4|NA/4|0|NA|
|MOMENT_XY_END|train|Post-scale|all|NA/24|NA/8|NA/8|0|NA|
|MOMENT_XY_END|train|Post-scale|0|NA/12|NA/4|NA/4|0|NA|
|MOMENT_XY_END|train|Post-scale|1|NA/12|NA/4|NA/4|0|NA|
|MOMENT_XY_END|validation|MODEL|all|NA/24|NA/8|NA/8|0|NA|
|MOMENT_XY_END|validation|MODEL|0|NA/12|NA/4|NA/4|0|NA|
|MOMENT_XY_END|validation|MODEL|1|NA/12|NA/4|NA/4|0|NA|
|MOMENT_XY_END|validation|No-response|all|NA/24|NA/8|NA/8|0|NA|
|MOMENT_XY_END|validation|No-response|0|NA/12|NA/4|NA/4|0|NA|
|MOMENT_XY_END|validation|No-response|1|NA/12|NA/4|NA/4|0|NA|
|MOMENT_XY_END|validation|Post-scale|all|NA/24|NA/8|NA/8|0|NA|
|MOMENT_XY_END|validation|Post-scale|0|NA/12|NA/4|NA/4|0|NA|
|MOMENT_XY_END|validation|Post-scale|1|NA/12|NA/4|NA/4|0|NA|

完整 3×3 单元和每 beta/每噪声列配对见 cross_matrix.csv、target_column_audit.csv、target_column_noise_summary.csv。No-response 与 Post-scale 分别使用各自检查点的 F1；Post-scale 比例只用 C1 与 C_alpha 拟合。解析参照使用编辑前 C/输出，不等同当前 C-only 模型输入合同，不计新模型生成。No-response 固定列应为 A=0、严格胜=0；同列方向正控结果见 tests/v2_scoring_cpu_tests.json。

本报告控制信号筛查要求正确列/三列全胜出现增益、同输入通吃不增加，且逐曲线绝对斜率误差不增加并在两噪声保持方向一致；不是单独看平均 A 或平均斜率。质量退化项逐项披露，不将任何数值容差当成允许质量损失的阈值。

## 七个问题与下一步

1. MOMENT 是否超过等预算 H_CONT：开发性几何候选=NA；主终点差=NA，配对差中位数=NA，改善=NA/8；两噪声主终点同向改善=NA。
2. 是否保留 START：MOMENT−START 主终点差=NA；完整 GT→F、F→GT 及质量变化单列，不以相对续训胜出代替起点保真。
3. 是否改善条件正确性：控制一致性筛查=NA；两噪声控制一致=NA。严格列、三列全胜、同输入通吃和绝对响应误差共同判断。
4. 质量代价：MOMENT−H_CONT 各独立 A/C 分组内可见守门项退化 0 项。此数不是独立实验次数，也不声称统计显著；小角分布与 GT 参照需结合相交/退化解释。
5. padding 修复是否有证据：实际 same_shape_padding、variable_padding、零梯度及逐层定位以上表/原表为准；新归一化接受与旧逐元素 flags 并列，缺失为 NA。
6. 等价优化成本：每任务 forward 与 F+B 的实测重复、范围、M/H 比及旧实现参照均已列出；门禁=NOT_RUN，不能用尚未测到的值作性能结论。
7. 优先下一步：先完成本轮已授权且尚缺失的门禁或固定终点评估。未训练或缺失的三比较保持 NA；不据此判断方法质量。 检验假说：只有完整固定面板及配对证据，才能检验关系通道是否超过等预算续训并保留共同起点质量。

本轮仅一对训练随机流的有限预算 pilot。训练 loss、成功生成、宽度变化或通道梯度非零不能单独证明几何收益。固定任务结束即停止；不会自动扩大预算、改 loss、改 λ、切换默认模型或启动下一实验。

## 可复核产物

- paired_results/per_output.csv、per_parent.csv、paired_per_output.csv、paired_per_parent.csv：全 432 登记行及逐父配对。
- primary_paired_parents.csv、primary_noise_pairs.csv：8 父×两噪声的原始前值、后值与差值。
- width_per_output.csv、width_per_curve.csv、noise_pairs_per_parent.csv：模型与两个解析参照的完整曲线。
- cross_matrix.csv、target_column_audit.csv、target_column_noise_pairs.csv：实际行列方向及所有目标列。
- quality_paired_summary.csv、coverage_partition.csv：双方向实值、覆盖与尾部分区、接口及相交代价。
- numerical_comparisons.csv、precheck_cost_summary.csv：实际数值合同和成本重复；不存在额外 I/P 矩阵。
- evaluation/<label>/scoring/scores/：原实际评分明细和身份；source_code/、registration/：评分源码、协议及来源 hash。
