# Chair Hybrid Coupling Pilot V1

QUALITY_RESULT：**IMPLEMENTATION_BLOCKED**。本次评分与配对汇总完成状态：EVALUATED INCOMPLETE；缺失父对象不缩小预登记分母。

Native+Geo 从 chair_domain_control_v1 phase3000、conditional_total_step4000 共同起点分叉；OT_BASE λ=0，OT_HYBRID λ=.25。新终点为 pilot_step1000、conditional_total_step5000，不能与历史预算实验 step5000 混称。

## 执行账本

实际训练更新、采样、额外诊断前向及失败/缓存以 accounting.json 和 ledger 原账为准；下列为当前账本快照：

```json
{
  "status": "PENDING_ACCOUNTING"
}
```

## 主终点

`fixed_query_MSE = (1/512) Σ_q [d(q,F)/L]^2`。q 是原 coverage seed9973 冻结自由 GT 点，d 是点到三角表面的欧氏距离；L 为该输入完整目标 bbox 对角线。该量无量纲，是 L 归一距离的平方，沿用最近归因独立审计。A 的 alpha=1 时 L=L0；新 `fixed_query_MSE_L0` 另列。原双向表面 RMS 使用 seed9911 的另一组点和预测面积采样，两者独立列出。

先在每个验证父内平均两份噪声，再对 8 父等权。配对差中位数在逐父差上计算。改善计数以严格 <0 判断；两份生成噪声不是两个训练随机种子。

|比较|父均值差|逐父配对差中位数|改善父/8|最差父|最差差值|开发性筛选|
|---|---|---|---|---|---|---|

|比较|噪声|均值差|改善父|有效父|
|---|---|---|---|---|

筛选要求：HYBRID−BASE 父均值差<0、逐父配对差中位数<0，且至少6/8父改善。缺少任一父/噪声不能宣布通过。这是开发性规则，不是显著性检验。

## 条件控制与两噪声

宽度为 free 三角角点 X 的 q95−q05。逐曲线先计算 |b_self−1|、|b_GT−1|，再父内噪声平均、父等权。基准宽度比例 r_base 与响应斜率分列；不能用平均斜率相互抵消欠响应和过响应。

|比较|划分|指标|父均值差|配对差中位数|
|---|---|---|---|---|
|OT_BASE-START|train|b_self_abs_error|NA|NA|
|OT_BASE-START|train|b_GT_abs_error|NA|NA|
|OT_BASE-START|train|baseline_abs_ratio_error|NA|NA|
|OT_BASE-START|train|mean_A_column|NA|NA|
|OT_BASE-START|train|all_three_strict_win|NA|NA|
|OT_HYBRID-OT_BASE|train|b_self_abs_error|NA|NA|
|OT_HYBRID-OT_BASE|train|b_GT_abs_error|NA|NA|
|OT_HYBRID-OT_BASE|train|baseline_abs_ratio_error|NA|NA|
|OT_HYBRID-OT_BASE|train|mean_A_column|NA|NA|
|OT_HYBRID-OT_BASE|train|all_three_strict_win|NA|NA|
|OT_HYBRID-START|train|b_self_abs_error|NA|NA|
|OT_HYBRID-START|train|b_GT_abs_error|NA|NA|
|OT_HYBRID-START|train|baseline_abs_ratio_error|NA|NA|
|OT_HYBRID-START|train|mean_A_column|NA|NA|
|OT_HYBRID-START|train|all_three_strict_win|NA|NA|

`E[alpha,beta]` 行为生成输入，列为评分目标。正确条件优势固定 beta 列：其他两个输入误差均值减正确输入误差。严格胜出必须同时小于两个错误输入；相等不算胜。所有 3×3 单元、逐 beta、逐噪声和同一输入是否三目标通吃均有独立 CSV。

|HYBRID−BASE|划分|beta|列优势变化|严格列胜率变化|
|---|---|---|---|---|
|OT_HYBRID-OT_BASE|train|0.9|NA|NA|
|OT_HYBRID-OT_BASE|train|1|NA|NA|
|OT_HYBRID-OT_BASE|train|1.1|NA|NA|

No-response 与 Post-scale 仅由各检查点自身 F1 构造，Post-scale 只从 C1 与 C_alpha 的对应 X 坐标拟合比例。它们拥有编辑前条件和旧输出，输入合同多于只读当前 C 的生成器。解析输出不计模型生成，不替代原 raw。No-response 的每列优势必须为0且无严格列胜；数学正控另见 tests/scoring_cpu_tests.json。

## 覆盖尾部、接口与相交守门

A 训练/验证、20%/40%分别统计；原不可构造40%保持NA。C是已观察留出的回归披露，不能称全新独立测试，也不参与模型选择。下表给出实际父配对变化；完整字段及每父变化见 quality_paired_summary.csv / paired_per_parent.csv。

|表/划分/C|指标|HYBRID−BASE均值差|改善父|退化父|有效/登记父|
|---|---|---|---|---|---|
|A/train/0.2|Gamma_to_free_edges_RMS_bbox_pct|NA|0|0|0/1|
|A/train/0.2|Gamma_to_free_edges_coverage_0.001_fraction|NA|0|0|0/1|
|A/train/0.2|CF_unresolved_or_degenerate_pair_count|NA|0|0|0/1|
|A/train/0.2|FF_unresolved_or_degenerate_pair_count|NA|0|0|0/1|
|A/train/0.2|CF_proper_affected_free_fraction|NA|0|0|0/1|
|A/train/0.2|FF_illegal_affected_free_fraction|NA|0|0|0/1|
|A/train/0.2|free_minimum_angle_lt5_fraction|NA|0|0|0/1|
|A/train/0.2|free_near_degenerate_fraction|NA|0|0|0/1|
|A/train/0.2|fixed_query_p95_L|NA|0|0|0/1|
|A/train/0.2|coverage_fraction|NA|0|0|0/1|
|C/holdout/0.2|Gamma_to_free_edges_RMS_bbox_pct|NA|0|0|0/1|
|C/holdout/0.2|Gamma_to_free_edges_coverage_0.001_fraction|NA|0|0|0/1|
|C/holdout/0.2|CF_unresolved_or_degenerate_pair_count|NA|0|0|0/1|
|C/holdout/0.2|FF_unresolved_or_degenerate_pair_count|NA|0|0|0/1|
|C/holdout/0.2|CF_proper_affected_free_fraction|NA|0|0|0/1|
|C/holdout/0.2|FF_illegal_affected_free_fraction|NA|0|0|0/1|
|C/holdout/0.2|free_minimum_angle_lt5_fraction|NA|0|0|0/1|
|C/holdout/0.2|free_near_degenerate_fraction|NA|0|0|0/1|
|C/holdout/0.2|fixed_query_p95_L|NA|0|0|0/1|
|C/holdout/0.2|coverage_fraction|NA|0|0|0/1|

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


## ??????

1. HYBRID??????alpha=1????????????????
2. ????????A20????????=NA?p95/L??=NA????????????????
3. ?????????????????????????????=False???????????/?????????=False????????????=False????beta????????????
4. ??????????????HYBRID?BASE??????????0?????????????????????????????????????????
5. ????????????????FP32????????????????
6. ????????????????????????????????????
