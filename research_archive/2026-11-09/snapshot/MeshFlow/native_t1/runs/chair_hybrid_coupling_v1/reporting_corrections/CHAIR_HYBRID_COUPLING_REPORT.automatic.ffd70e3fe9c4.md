# Chair Hybrid Coupling Pilot V1

QUALITY_RESULT：**GEOMETRY_SIGNAL_WITH_TRADEOFF**。本次评分与配对汇总完成状态：EVALUATED DONE。

Native+Geo 从 chair_domain_control_v1 phase3000、conditional_total_step4000 共同起点分叉；OT_BASE λ=0，OT_HYBRID λ=.25。新终点为 pilot_step1000、conditional_total_step5000，不能与历史预算实验 step5000 混称。

## 执行账本

实际训练更新、采样、额外诊断前向及失败/缓存以 accounting.json 和 ledger 原账为准；下列为当前账本快照：

```json
{
  "attempts": {
    "diagnostic_bf16": 2400,
    "diagnostic_fp32": 72,
    "optimizer_update": 2000,
    "precheck_backward": 2,
    "precheck_forward": 10,
    "sampling": 292,
    "sampling_forward": 14600,
    "training_backward": 16000,
    "training_forward": 16000
  },
  "failed": {},
  "limits": {
    "diagnostic_bf16": 2400,
    "diagnostic_fp32": 72,
    "optimizer_update": 2000,
    "precheck_backward": 4,
    "precheck_forward": 32,
    "sampling": 434,
    "sampling_forward": 21700,
    "training_backward": 16000,
    "training_forward": 16000
  },
  "pending": [],
  "returns": {
    "diagnostic_bf16": 2400,
    "diagnostic_fp32": 72,
    "optimizer_update": 2000,
    "precheck_backward": 2,
    "precheck_forward": 10,
    "sampling": 292,
    "sampling_forward": 14600,
    "training_backward": 16000,
    "training_forward": 16000
  },
  "summary": {
    "actual_full_sampling_attempts": 292,
    "actual_full_sampling_returns": 292,
    "applicable_unique_generation_requests": 414,
    "extra_diagnostic_bf16_forwards": 2400,
    "extra_diagnostic_fp32_forwards": 72,
    "failed_actual_calls": 0,
    "generation_not_applicable": 18,
    "historical_raw_cache_hits": 122,
    "optimizer_updates": 2000,
    "precheck_forwards": 10,
    "precheck_zero_update_backwards": 2,
    "registered_unique_generation_requests": 432,
    "sampling_forwards": 14600,
    "total_model_forwards": 33082,
    "training_backwards": 16000,
    "training_microbatch_forwards": 16000,
    "unresolved_calls": 0
  },
  "total_forward_attempts": 33082,
  "wall_seconds_by_kind": {
    "diagnostic_bf16": 103.5547483280088,
    "diagnostic_fp32": 3.1906465450012718,
    "optimizer_update": 28.62193315699966,
    "precheck_backward": 0.1794030720000066,
    "precheck_forward": 1.451810940999998,
    "sampling": 791.1357667339964,
    "sampling_forward": 645.5762975639566,
    "training_backward": 1268.0407195899984,
    "training_forward": 751.4479912869764
  },
  "wall_time_note": "CPU wall time around actual call; aggregate nested sampling and its forwards separately, never sum nested categories."
}
```

## 主终点

`fixed_query_MSE = (1/512) Σ_q [d(q,F)/L]^2`。q 是原 coverage seed9973 冻结自由 GT 点，d 是点到三角表面的欧氏距离；L 为该输入完整目标 bbox 对角线。该量无量纲，是 L 归一距离的平方，沿用最近归因独立审计。A 的 alpha=1 时 L=L0；新 `fixed_query_MSE_L0` 另列。原双向表面 RMS 使用 seed9911 的另一组点和预测面积采样，两者独立列出。

先在每个验证父内平均两份噪声，再对 8 父等权。配对差中位数在逐父差上计算。改善计数以严格 <0 判断；两份生成噪声不是两个训练随机种子。

|比较|父均值差|逐父配对差中位数|改善父/8|最差父|最差差值|开发性筛选|
|---|---|---|---|---|---|---|
|OT_BASE-START|-8.987423e-05|2.247139e-05|3/8|c1b8c2f0c4f3cc43aff7c08b060f5ed6|0.001234083|0|
|OT_HYBRID-OT_BASE|-0.001019942|-0.0003549652|6/8|866e7248434968051cc7750d0afca796|0.0001329398|1|
|OT_HYBRID-START|-0.001109816|-0.0005315532|6/8|866e7248434968051cc7750d0afca796|0.0006510748|1|

|比较|噪声|均值差|改善父|有效父|
|---|---|---|---|---|
|OT_BASE-START|0|0.0007155663|1|8|
|OT_BASE-START|1|-0.0008953147|4|8|
|OT_HYBRID-OT_BASE|0|-0.001634837|7|8|
|OT_HYBRID-OT_BASE|1|-0.0004050472|6|8|
|OT_HYBRID-START|0|-0.000919271|4|8|
|OT_HYBRID-START|1|-0.001300362|7|8|

筛选要求：HYBRID−BASE 父均值差<0、逐父配对差中位数<0，且至少6/8父改善。缺少任一父/噪声不能宣布通过。这是开发性规则，不是显著性检验。

## 条件控制与两噪声

宽度为 free 三角角点 X 的 q95−q05。逐曲线先计算 |b_self−1|、|b_GT−1|，再父内噪声平均、父等权。基准宽度比例 r_base 与响应斜率分列；不能用平均斜率相互抵消欠响应和过响应。

|比较|划分|指标|父均值差|配对差中位数|
|---|---|---|---|---|
|OT_BASE-START|train|b_self_abs_error|-0.115383|-0.1591482|
|OT_BASE-START|train|b_GT_abs_error|-0.08378564|-0.08284567|
|OT_BASE-START|train|baseline_abs_ratio_error|0.01665089|0.0002200023|
|OT_BASE-START|train|mean_A_column|-0.0002955311|-7.923718e-05|
|OT_BASE-START|train|all_three_strict_win|0.125|0|
|OT_BASE-START|validation|b_self_abs_error|-0.2273869|-0.1842282|
|OT_BASE-START|validation|b_GT_abs_error|-0.1503208|-0.1031522|
|OT_BASE-START|validation|baseline_abs_ratio_error|-0.0001202965|-0.001446221|
|OT_BASE-START|validation|mean_A_column|2.789019e-05|0.0002088746|
|OT_BASE-START|validation|all_three_strict_win|0|0|
|OT_HYBRID-OT_BASE|train|b_self_abs_error|-0.09378413|-0.0995001|
|OT_HYBRID-OT_BASE|train|b_GT_abs_error|-0.06413222|-0.06632195|
|OT_HYBRID-OT_BASE|train|baseline_abs_ratio_error|-0.05271986|-0.04889793|
|OT_HYBRID-OT_BASE|train|mean_A_column|0.00216598|0.0009645279|
|OT_HYBRID-OT_BASE|train|all_three_strict_win|0|0|
|OT_HYBRID-OT_BASE|validation|b_self_abs_error|-0.05324138|-0.04017568|
|OT_HYBRID-OT_BASE|validation|b_GT_abs_error|-0.05319081|-0.06095685|
|OT_HYBRID-OT_BASE|validation|baseline_abs_ratio_error|-0.04434489|-0.03380489|
|OT_HYBRID-OT_BASE|validation|mean_A_column|0.000374428|7.500327e-05|
|OT_HYBRID-OT_BASE|validation|all_three_strict_win|0|0|
|OT_HYBRID-START|train|b_self_abs_error|-0.2091671|-0.234623|
|OT_HYBRID-START|train|b_GT_abs_error|-0.1479179|-0.1909712|
|OT_HYBRID-START|train|baseline_abs_ratio_error|-0.03606897|-0.01167666|
|OT_HYBRID-START|train|mean_A_column|0.001870449|0.001064915|
|OT_HYBRID-START|train|all_three_strict_win|0.125|0|
|OT_HYBRID-START|validation|b_self_abs_error|-0.2806283|-0.2865786|
|OT_HYBRID-START|validation|b_GT_abs_error|-0.2035116|-0.1708494|
|OT_HYBRID-START|validation|baseline_abs_ratio_error|-0.04446519|-0.03684344|
|OT_HYBRID-START|validation|mean_A_column|0.0004023182|0.0003268843|
|OT_HYBRID-START|validation|all_three_strict_win|0|0|

`E[alpha,beta]` 行为生成输入，列为评分目标。正确条件优势固定 beta 列：其他两个输入误差均值减正确输入误差。严格胜出必须同时小于两个错误输入；相等不算胜。所有 3×3 单元、逐 beta、逐噪声和同一输入是否三目标通吃均有独立 CSV。

|HYBRID−BASE|划分|beta|列优势变化|严格列胜率变化|
|---|---|---|---|---|
|OT_HYBRID-OT_BASE|train|0.9|0.01901699|0.5|
|OT_HYBRID-OT_BASE|train|1|-0.002116803|0.125|
|OT_HYBRID-OT_BASE|train|1.1|-0.01040225|-0.125|
|OT_HYBRID-OT_BASE|validation|0.9|0.001836794|-0.125|
|OT_HYBRID-OT_BASE|validation|1|-0.0008060138|0|
|OT_HYBRID-OT_BASE|validation|1.1|9.25035e-05|0|

No-response 与 Post-scale 仅由各检查点自身 F1 构造，Post-scale 只从 C1 与 C_alpha 的对应 X 坐标拟合比例。它们拥有编辑前条件和旧输出，输入合同多于只读当前 C 的生成器。解析输出不计模型生成，不替代原 raw。No-response 的每列优势必须为0且无严格列胜；数学正控另见 tests/scoring_cpu_tests.json。

## 覆盖尾部、接口与相交守门

A 训练/验证、20%/40%分别统计；原不可构造40%保持NA。C是已观察留出的回归披露，不能称全新独立测试，也不参与模型选择。下表给出实际父配对变化；完整字段及每父变化见 quality_paired_summary.csv / paired_per_parent.csv。

|表/划分/C|指标|HYBRID−BASE均值差|改善父|退化父|有效/登记父|
|---|---|---|---|---|---|
|A/train/0.2|Gamma_to_free_edges_RMS_bbox_pct|-0.4644272|6|2|8/8|
|A/train/0.2|Gamma_to_free_edges_coverage_0.001_fraction|-0.001585073|4|3|8/8|
|A/train/0.2|CF_unresolved_or_degenerate_pair_count|0|0|0|8/8|
|A/train/0.2|FF_unresolved_or_degenerate_pair_count|0|0|0|8/8|
|A/train/0.2|CF_proper_affected_free_fraction|-0.01633229|5|3|8/8|
|A/train/0.2|FF_illegal_affected_free_fraction|-0.03492874|7|1|8/8|
|A/train/0.2|free_minimum_angle_lt5_fraction|0.05139835|2|6|8/8|
|A/train/0.2|free_near_degenerate_fraction|0|0|0|8/8|
|A/train/0.2|fixed_query_p95_L|-0.01052926|6|2|8/8|
|A/train/0.2|coverage_fraction|0.15625|7|1|8/8|
|A/train/0.4|Gamma_to_free_edges_RMS_bbox_pct|0.4476407|3|4|7/8|
|A/train/0.4|Gamma_to_free_edges_coverage_0.001_fraction|-0.05997169|1|6|7/8|
|A/train/0.4|CF_unresolved_or_degenerate_pair_count|0|0|0|7/8|
|A/train/0.4|FF_unresolved_or_degenerate_pair_count|0|0|0|7/8|
|A/train/0.4|CF_proper_affected_free_fraction|-0.001665172|2|4|7/8|
|A/train/0.4|FF_illegal_affected_free_fraction|-0.002053011|3|4|7/8|
|A/train/0.4|free_minimum_angle_lt5_fraction|-0.008125656|5|2|7/8|
|A/train/0.4|free_near_degenerate_fraction|0|0|0|7/8|
|A/train/0.4|fixed_query_p95_L|0.03061652|3|4|7/8|
|A/train/0.4|coverage_fraction|-0.08928571|1|6|7/8|
|A/validation/0.2|Gamma_to_free_edges_RMS_bbox_pct|-1.107964|6|2|8/8|
|A/validation/0.2|Gamma_to_free_edges_coverage_0.001_fraction|6.389498e-05|3|4|8/8|
|A/validation/0.2|CF_unresolved_or_degenerate_pair_count|0|0|0|8/8|
|A/validation/0.2|FF_unresolved_or_degenerate_pair_count|0|0|0|8/8|
|A/validation/0.2|CF_proper_affected_free_fraction|-0.01256763|5|3|8/8|
|A/validation/0.2|FF_illegal_affected_free_fraction|-0.007648275|5|3|8/8|
|A/validation/0.2|free_minimum_angle_lt5_fraction|0.04966999|2|6|8/8|
|A/validation/0.2|free_near_degenerate_fraction|0|0|0|8/8|
|A/validation/0.2|fixed_query_p95_L|-0.0155135|7|1|8/8|
|A/validation/0.2|coverage_fraction|0.02490234|5|3|8/8|
|A/validation/0.4|Gamma_to_free_edges_RMS_bbox_pct|-0.006473101|3|3|6/8|
|A/validation/0.4|Gamma_to_free_edges_coverage_0.001_fraction|0.002508588|4|2|6/8|
|A/validation/0.4|CF_unresolved_or_degenerate_pair_count|0|0|0|6/8|
|A/validation/0.4|FF_unresolved_or_degenerate_pair_count|0|0|0|6/8|
|A/validation/0.4|CF_proper_affected_free_fraction|0.007960662|3|3|6/8|
|A/validation/0.4|FF_illegal_affected_free_fraction|-0.03703173|4|2|6/8|
|A/validation/0.4|free_minimum_angle_lt5_fraction|-0.01004978|4|2|6/8|
|A/validation/0.4|free_near_degenerate_fraction|0|0|0|6/8|
|A/validation/0.4|fixed_query_p95_L|-0.002614159|4|2|6/8|
|A/validation/0.4|coverage_fraction|0.03531901|3|3|6/8|
|C/holdout/0.2|Gamma_to_free_edges_RMS_bbox_pct|-0.5170373|5|3|8/8|
|C/holdout/0.2|Gamma_to_free_edges_coverage_0.001_fraction|-0.0004693046|3|4|8/8|
|C/holdout/0.2|CF_unresolved_or_degenerate_pair_count|0|0|0|8/8|
|C/holdout/0.2|FF_unresolved_or_degenerate_pair_count|0|0|0|8/8|
|C/holdout/0.2|CF_proper_affected_free_fraction|0.01351695|4|4|8/8|
|C/holdout/0.2|FF_illegal_affected_free_fraction|-0.03463754|7|1|8/8|
|C/holdout/0.2|free_minimum_angle_lt5_fraction|0.004787625|4|4|8/8|
|C/holdout/0.2|free_near_degenerate_fraction|0|0|0|8/8|
|C/holdout/0.2|fixed_query_p95_L|-0.01047938|7|1|8/8|
|C/holdout/0.2|coverage_fraction|-0.01013184|5|3|8/8|

coverage_partition.csv 在相同冻结查询点上拆成 both_in/enter/leave/both_out，贡献为各组平方距离差之和除原完整Q，四组精确闭合总MSE差。覆盖上升不能代替尾部误差下降；靠近C不自动意味着错误重复生成。

## 响应分解与解释边界

逐步 I/P/S、余弦、Euler恒等式残差和 FP32 冻结状态复核保存于各 evaluation/<label>/diagnostics/。I/P 是有限编辑的对称代数分配，不是唯一因果归因或无穷小Jacobian。零范数余弦必须标UNDEFINED；仅凭范数大不能宣称控制正确或刚性。应把逐父/噪声诊断与 width_per_curve、target_column_audit 中同 UID/noise 配对再解释。

本轮是一对训练随机流的有限预算开发试验。训练loss不能替代生成质量；宽度正确不能替代完整几何正确；未做严格全局水密审计，保持NOT_RUN。C保持是执行合同，不是模型学到的能力。证据不允许额外训练、换loss、改λ或追加求解器扫描。

## 产物

- `paired_results/primary_paired_parents.csv`：8父主终点配对原值、差值。
- `paired_results/per_output.csv` / `per_parent.csv`：A/B/C全请求与父均值，失败与NA保留。
- `paired_results/width_per_output.csv` / `width_per_curve.csv` / `noise_pairs_per_parent.csv`：模型和解析参照的真实逐曲线控制。
- `paired_results/cross_matrix.csv` / `target_column_audit.csv`：完整矩阵及固定目标列审计。
- `paired_results/coverage_partition.csv` / `quality_paired_summary.csv`：覆盖分区与守门配对变化。
- 每检查点 `scoring/scores/*.json` 保留实际原评分器全质量细节与来源hash。


## ????????

## 响应诊断：逐父、逐噪声与 FP32 复核

BF16/FP32 在 1 个预选冻结状态的 I/P 相反方向判断不一致。目前不能据 BF16 单独判定注入不足或传播抵消为主要原因；逐曲线代数分解仍有效。

诊断对象固定为表B按UID排序前2训练父、前2验证父，各原2份噪声；每检查点8条曲线。I/P/S均在有效free标量坐标上计算RMS；零范数余弦留空且状态为UNDEFINED。

|检查点|完成曲线/8|曲线平均相反步比例|积分I/P相反曲线|最大闭合RMS|FP32方向不一致/可比状态|
|---|---|---|---|---|---|
|START|8/8|0.52551|4|2.02996e-06|0/4|
|OT_BASE|8/8|0.433673|3|2.07106e-06|1/4|
|OT_HYBRID|8/8|0.492347|5|2.22095e-06|0/4|

FP32只复核冻结的首训练/首验证父noise0、k=0/25/49四组合。CSV保留实际autocast关闭、Linear输入/输出/参数dtype、Geo实际重算与输入hash；不是把BF16输出转成FP32。其余状态仍是正式BF16 Euler50轨迹。

逐曲线表按相同检查点、UID和noise关联真实几何fixed_query_MSE/p95、宽度r_base/b_self/b_GT及各目标列优势。缺失评分关联标BLOCKED，不用另一噪声或另一父替代。父表先等权平均两份噪声，同时保留逐噪声方向，不能把两噪声当独立父对象。

较大I/P范数不等于控制更正确；相反方向证明代数抵消存在，不自动证明其为错误主要原因。没有额外阈值把“注入接近零”或“Jacobian爆炸”写成已证实。闭合残差用于核对实际Euler数值关系，不作质量容许退化额度。

数据：`diagnostic_results/response_per_step.csv`、`response_per_curve.csv`、`response_per_parent.csv`、`fp32_precision_records.csv`。每检查点各有一张8曲线I/P/S图和一张8曲线余弦图；原向量与全部dtype审计保留在evaluation/<label>/diagnostics/。


## ??????

1. HYBRID??????alpha=1???????????????????? ?????=-0.001019942?????????=-0.0003549652????=6/8?
2. ????????A20????????=0.02490234?p95/L??=-0.0155135????????????????
3. ?????????????????????????????=True???????????/?????????=False????????????=True????beta????????????
4. ??????????????HYBRID?BASE??????????12?????????????????????????????????????????
5. ?????BF16/FP32 在 1 个预选冻结状态的 I/P 相反方向判断不一致。目前不能据 BF16 单独判定注入不足或传播抵消为主要原因；逐曲线代数分解仍有效。
6. ?????????????????????????????????????????????????????HYBRID?
