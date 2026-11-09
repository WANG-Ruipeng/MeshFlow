# Chair Finite Surface Attention V1

实际调用摘要：`{"actual_full_sampling_attempts": 0, "optimizer_updates": 0, "precheck_forwards": 88, "precheck_zero_update_backwards": 24}`。

IMPLEMENTED：IMPLEMENTATION_BLOCKED；TRAINED：NOT_RUN；EVALUATED：NOT_RUN；QUALITY_RESULT：**NA_NOT_EVALUATED**。

共同起点为 chair_hybrid_coupling_v1 / OT_HYBRID / pilot1000 / conditional_total5000 / λ=.25。新终点为 surface_step1000、conditional_total6000。H_CONT控制等预算续训；MOMENT_XY与SURFACE_XY同容量、同接入方式，区别仅为几何关系定义。

## 数值与成本门禁

门禁状态：**IMPLEMENTATION_BLOCKED**。未通过门禁时正式训练和质量比较保持NOT_RUN/NA；不能把未运行当成方法质量失败。

```json
{
  "status": "IMPLEMENTATION_BLOCKED",
  "numeric": {
    "status": "IMPLEMENTATION_BLOCKED",
    "reason": "schema-only fixture"
  },
  "real_model": {
    "status": "PASS",
    "precheck_forwards": 88,
    "precheck_backwards": 24
  },
  "cost": {
    "status": "PASS"
  }
}
```

192点候选、12288点参考与49152点收敛复核分别登记。关系准确度不是完整模型几何评分；CPU求积MSE不能与fixed_query_MSE或双向RMS混用。成本门槛是同任务完整forward及forward+backward中位耗时不超过H_CONT的3倍，并保留至少15%显存余量。

## 固定主终点

`fixed_query_MSE = mean([d(q,F)/L0]^2)`，q为原512个冻结GT free coverage查询点，点到三角表面距离，A20 alpha=1时L=L0。先父内平均两份原噪声，再8父等权。逐父配对差中位数不是两个组中位数之差；缺失任一父/噪声不能缩分母宣布通过。

|比较|父均值差|逐父差中位数|改善父/8|最差退化父|最差差值|几何候选|
|---|---|---|---|---|---|---|
|H_CONT-START|NA|NA|NA/8|NA|NA|NA|
|MOMENT_XY-H_CONT|NA|NA|NA/8|NA|NA|NA|
|MOMENT_XY-START|NA|NA|NA/8|NA|NA|NA|
|SURFACE_XY-H_CONT|NA|NA|NA/8|NA|NA|NA|
|SURFACE_XY-MOMENT_XY|NA|NA|NA/8|NA|NA|NA|
|SURFACE_XY-START|NA|NA|NA/8|NA|NA|NA|

|比较|噪声|均值差|改善父|有效父|
|---|---|---|---|---|
|H_CONT-START|0|NA|NA|0|
|H_CONT-START|1|NA|NA|0|
|MOMENT_XY-H_CONT|0|NA|NA|0|
|MOMENT_XY-H_CONT|1|NA|NA|0|
|MOMENT_XY-START|0|NA|NA|0|
|MOMENT_XY-START|1|NA|NA|0|
|SURFACE_XY-H_CONT|0|NA|NA|0|
|SURFACE_XY-H_CONT|1|NA|NA|0|
|SURFACE_XY-MOMENT_XY|0|NA|NA|0|
|SURFACE_XY-MOMENT_XY|1|NA|NA|0|
|SURFACE_XY-START|0|NA|NA|0|
|SURFACE_XY-START|1|NA|NA|0|

几何候选要求均值差<0、逐父差中位数<0且至少6/8父改善。这是单配对训练流开发性筛选，不是显著性检验。只有SURFACE同时胜过H_CONT与MOMENT，才给有限表面候选初步几何支持。

## 两个实际表面方向、覆盖与质量

GT→F与F→GT均给实际数值。它们使用原seed9911双向面积采样合同；fixed_query使用另一组seed9973点。下表方向指标单位为L归一RMS，保留原%L字段。

|检查点|表/划分/C|父数|GT→F RMS/L|F→GT RMS/L|双向 RMS/L|fixed MSE|覆盖|p95/L|预测小角率|同GT小角率|
|---|---|---|---|---|---|---|---|---|---|---|
|H_CONT|A/train/0.2|0|NA|NA|NA|NA|NA|NA|NA|NA|
|H_CONT|A/train/0.4|0|NA|NA|NA|NA|NA|NA|NA|NA|
|H_CONT|A/validation/0.2|0|NA|NA|NA|NA|NA|NA|NA|NA|
|H_CONT|A/validation/0.4|0|NA|NA|NA|NA|NA|NA|NA|NA|
|H_CONT|C/holdout/0.2|0|NA|NA|NA|NA|NA|NA|NA|NA|
|MOMENT_XY|A/train/0.2|0|NA|NA|NA|NA|NA|NA|NA|NA|
|MOMENT_XY|A/train/0.4|0|NA|NA|NA|NA|NA|NA|NA|NA|
|MOMENT_XY|A/validation/0.2|0|NA|NA|NA|NA|NA|NA|NA|NA|
|MOMENT_XY|A/validation/0.4|0|NA|NA|NA|NA|NA|NA|NA|NA|
|MOMENT_XY|C/holdout/0.2|0|NA|NA|NA|NA|NA|NA|NA|NA|
|START|A/train/0.2|0|NA|NA|NA|NA|NA|NA|NA|NA|
|START|A/train/0.4|0|NA|NA|NA|NA|NA|NA|NA|NA|
|START|A/validation/0.2|0|NA|NA|NA|NA|NA|NA|NA|NA|
|START|A/validation/0.4|0|NA|NA|NA|NA|NA|NA|NA|NA|
|START|C/holdout/0.2|0|NA|NA|NA|NA|NA|NA|NA|NA|
|SURFACE_XY|A/train/0.2|0|NA|NA|NA|NA|NA|NA|NA|NA|
|SURFACE_XY|A/train/0.4|0|NA|NA|NA|NA|NA|NA|NA|NA|
|SURFACE_XY|A/validation/0.2|0|NA|NA|NA|NA|NA|NA|NA|NA|
|SURFACE_XY|A/validation/0.4|0|NA|NA|NA|NA|NA|NA|NA|NA|
|SURFACE_XY|C/holdout/0.2|0|NA|NA|NA|NA|NA|NA|NA|NA|

同GT free角度使用原shape_quality公式，保存全部角点角度、每面最小角和分位数；零边沿用原cosine=1的数值约定，不把零面积面称作合法面积表面。该参照只用于评分，未进入模型或loss。

|比较|表/划分/C|指标|配对均值变化|改善父|退化父|有效/登记|
|---|---|---|---|---|---|---|
|SURFACE_XY-H_CONT|A/train/0.2|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/train/0.2|CF_proper_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/train/0.2|FF_illegal_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/train/0.2|free_minimum_angle_lt5_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/train/0.2|fixed_query_p95_L|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/train/0.2|F_to_GT_original_RMS_L|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/train/0.4|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/train/0.4|CF_proper_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/train/0.4|FF_illegal_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/train/0.4|free_minimum_angle_lt5_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/train/0.4|fixed_query_p95_L|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/train/0.4|F_to_GT_original_RMS_L|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/validation/0.2|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/validation/0.2|CF_proper_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/validation/0.2|FF_illegal_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/validation/0.2|free_minimum_angle_lt5_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/validation/0.2|fixed_query_p95_L|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/validation/0.2|F_to_GT_original_RMS_L|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/validation/0.4|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/validation/0.4|CF_proper_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/validation/0.4|FF_illegal_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/validation/0.4|free_minimum_angle_lt5_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/validation/0.4|fixed_query_p95_L|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/validation/0.4|F_to_GT_original_RMS_L|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|C/holdout/0.2|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|C/holdout/0.2|CF_proper_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|C/holdout/0.2|FF_illegal_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|C/holdout/0.2|free_minimum_angle_lt5_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|C/holdout/0.2|fixed_query_p95_L|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|C/holdout/0.2|F_to_GT_original_RMS_L|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/train/0.2|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/train/0.2|CF_proper_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/train/0.2|FF_illegal_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/train/0.2|free_minimum_angle_lt5_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/train/0.2|fixed_query_p95_L|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/train/0.2|F_to_GT_original_RMS_L|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/train/0.4|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/train/0.4|CF_proper_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/train/0.4|FF_illegal_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/train/0.4|free_minimum_angle_lt5_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/train/0.4|fixed_query_p95_L|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/train/0.4|F_to_GT_original_RMS_L|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/validation/0.2|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/validation/0.2|CF_proper_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/validation/0.2|FF_illegal_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/validation/0.2|free_minimum_angle_lt5_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/validation/0.2|fixed_query_p95_L|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/validation/0.2|F_to_GT_original_RMS_L|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/validation/0.4|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/validation/0.4|CF_proper_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/validation/0.4|FF_illegal_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/validation/0.4|free_minimum_angle_lt5_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/validation/0.4|fixed_query_p95_L|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/validation/0.4|F_to_GT_original_RMS_L|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|C/holdout/0.2|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|C/holdout/0.2|CF_proper_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|C/holdout/0.2|FF_illegal_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|C/holdout/0.2|free_minimum_angle_lt5_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|C/holdout/0.2|fixed_query_p95_L|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|C/holdout/0.2|F_to_GT_original_RMS_L|NA|NA|NA|0/8|

完整原方向、Gamma覆盖、未决相交、退化与有限值/C/N保持均在per_output、paired_per_parent和quality_paired_summary中。A训练/验证与20%/40%分开；原不可构造40%保持NA。C为已观察留出回归，不参与选择终点。

## 条件控制与解析参照

W为free角点X的q95−q05。r_base、b_self、b_GT分列；先逐曲线求|b−1|再平均，不能抵消欠响应与过响应。分母退化标UNDEFINED。矩阵E[alpha,beta]的行是输入、列是目标，A_column固定beta列；严格胜出需同时小于两个错误输入。

|比较|划分|指标|父均值变化|配对差中位数|
|---|---|---|---|---|
|H_CONT-START|train|b_self_abs_error|NA|NA|
|H_CONT-START|train|b_GT_abs_error|NA|NA|
|H_CONT-START|train|mean_A_column|NA|NA|
|H_CONT-START|train|all_three_strict_win|NA|NA|
|H_CONT-START|validation|b_self_abs_error|NA|NA|
|H_CONT-START|validation|b_GT_abs_error|NA|NA|
|H_CONT-START|validation|mean_A_column|NA|NA|
|H_CONT-START|validation|all_three_strict_win|NA|NA|
|MOMENT_XY-H_CONT|train|b_self_abs_error|NA|NA|
|MOMENT_XY-H_CONT|train|b_GT_abs_error|NA|NA|
|MOMENT_XY-H_CONT|train|mean_A_column|NA|NA|
|MOMENT_XY-H_CONT|train|all_three_strict_win|NA|NA|
|MOMENT_XY-H_CONT|validation|b_self_abs_error|NA|NA|
|MOMENT_XY-H_CONT|validation|b_GT_abs_error|NA|NA|
|MOMENT_XY-H_CONT|validation|mean_A_column|NA|NA|
|MOMENT_XY-H_CONT|validation|all_three_strict_win|NA|NA|
|MOMENT_XY-START|train|b_self_abs_error|NA|NA|
|MOMENT_XY-START|train|b_GT_abs_error|NA|NA|
|MOMENT_XY-START|train|mean_A_column|NA|NA|
|MOMENT_XY-START|train|all_three_strict_win|NA|NA|
|MOMENT_XY-START|validation|b_self_abs_error|NA|NA|
|MOMENT_XY-START|validation|b_GT_abs_error|NA|NA|
|MOMENT_XY-START|validation|mean_A_column|NA|NA|
|MOMENT_XY-START|validation|all_three_strict_win|NA|NA|
|SURFACE_XY-H_CONT|train|b_self_abs_error|NA|NA|
|SURFACE_XY-H_CONT|train|b_GT_abs_error|NA|NA|
|SURFACE_XY-H_CONT|train|mean_A_column|NA|NA|
|SURFACE_XY-H_CONT|train|all_three_strict_win|NA|NA|
|SURFACE_XY-H_CONT|validation|b_self_abs_error|NA|NA|
|SURFACE_XY-H_CONT|validation|b_GT_abs_error|NA|NA|
|SURFACE_XY-H_CONT|validation|mean_A_column|NA|NA|
|SURFACE_XY-H_CONT|validation|all_three_strict_win|NA|NA|
|SURFACE_XY-MOMENT_XY|train|b_self_abs_error|NA|NA|
|SURFACE_XY-MOMENT_XY|train|b_GT_abs_error|NA|NA|
|SURFACE_XY-MOMENT_XY|train|mean_A_column|NA|NA|
|SURFACE_XY-MOMENT_XY|train|all_three_strict_win|NA|NA|
|SURFACE_XY-MOMENT_XY|validation|b_self_abs_error|NA|NA|
|SURFACE_XY-MOMENT_XY|validation|b_GT_abs_error|NA|NA|
|SURFACE_XY-MOMENT_XY|validation|mean_A_column|NA|NA|
|SURFACE_XY-MOMENT_XY|validation|all_three_strict_win|NA|NA|
|SURFACE_XY-START|train|b_self_abs_error|NA|NA|
|SURFACE_XY-START|train|b_GT_abs_error|NA|NA|
|SURFACE_XY-START|train|mean_A_column|NA|NA|
|SURFACE_XY-START|train|all_three_strict_win|NA|NA|
|SURFACE_XY-START|validation|b_self_abs_error|NA|NA|
|SURFACE_XY-START|validation|b_GT_abs_error|NA|NA|
|SURFACE_XY-START|validation|mean_A_column|NA|NA|
|SURFACE_XY-START|validation|all_three_strict_win|NA|NA|

|比较|划分|beta|指标|变化|
|---|---|---|---|---|
|SURFACE_XY-H_CONT|train|0.9|A_column|NA|
|SURFACE_XY-H_CONT|train|0.9|column_strict_win|NA|
|SURFACE_XY-H_CONT|train|1|A_column|NA|
|SURFACE_XY-H_CONT|train|1|column_strict_win|NA|
|SURFACE_XY-H_CONT|train|1.1|A_column|NA|
|SURFACE_XY-H_CONT|train|1.1|column_strict_win|NA|
|SURFACE_XY-H_CONT|validation|0.9|A_column|NA|
|SURFACE_XY-H_CONT|validation|0.9|column_strict_win|NA|
|SURFACE_XY-H_CONT|validation|1|A_column|NA|
|SURFACE_XY-H_CONT|validation|1|column_strict_win|NA|
|SURFACE_XY-H_CONT|validation|1.1|A_column|NA|
|SURFACE_XY-H_CONT|validation|1.1|column_strict_win|NA|
|SURFACE_XY-MOMENT_XY|train|0.9|A_column|NA|
|SURFACE_XY-MOMENT_XY|train|0.9|column_strict_win|NA|
|SURFACE_XY-MOMENT_XY|train|1|A_column|NA|
|SURFACE_XY-MOMENT_XY|train|1|column_strict_win|NA|
|SURFACE_XY-MOMENT_XY|train|1.1|A_column|NA|
|SURFACE_XY-MOMENT_XY|train|1.1|column_strict_win|NA|
|SURFACE_XY-MOMENT_XY|validation|0.9|A_column|NA|
|SURFACE_XY-MOMENT_XY|validation|0.9|column_strict_win|NA|
|SURFACE_XY-MOMENT_XY|validation|1|A_column|NA|
|SURFACE_XY-MOMENT_XY|validation|1|column_strict_win|NA|
|SURFACE_XY-MOMENT_XY|validation|1.1|A_column|NA|
|SURFACE_XY-MOMENT_XY|validation|1.1|column_strict_win|NA|

完整矩阵、三目标全胜与同一输入三目标通吃、两噪声分别的曲线和列结果都有CSV。No-response和Post-scale只从各检查点自己的F1构造，Post-scale只由C1→C_alpha拟合；它们拥有旧条件与旧输出，不与单次生成冒称同输入合同。无额外I/P模型前向。

## 实际轨迹求积与成本

轨迹关系诊断状态：NOT_RUN。结果引用 `trajectory_quadrature_results.json`，仅使用预登记正式SURFACE轨迹k={0,25,49}，不重生成或改变主评分。

逐输出实际采样耗时及峰值见 sampling_cost_per_output.csv；完整账本见 accounting.json。每更新耗时与关系/注意力/模型耗时以原成本门禁、训练日志和阶段资源文件为准；缺失数值标NA。Euler50不等于推理成本不变；gradient checkpointing内部重算需与顶层forward分别解释。

## 四个决策问题

1. 总体有效性，SURFACE_XY vs H_CONT：NA。主比较原值与配对差见上表。
2. 有限表面机制，SURFACE_XY vs MOMENT_XY：NA；MOMENT_XY vs H_CONT：NA。同容量对照未被续训对照代替。
3. 条件能力：以逐曲线绝对斜率误差、逐beta列选择及两噪声分别变化判断。仅宽度响应改善不能宣布控制解决；门禁阻塞时均为NA。
4. 质量与成本：同时读取反向表面误差、尾部、接口、相交、小角面、留出及真实延迟。生成成功、MSE下降、条件选择改善和干净网格是不同结论。

**下一步判断：**数值或成本门禁未通过，正式质量比较均为NA。先处理本次实测阻塞；不以未运行结果判断有限表面或质心矩方法优劣。

本轮固定预算完成或门禁失败即停止，不调λ/带宽/层数/求积点，不改loss/学习率，不扩数据，不改默认模型。水密NOT_RUN；置换数值检查不是SE(3)等变证明；C硬保持是执行合同。


## Measured cost summary

|Checkpoint|Actual samples|Cache hits|Timed samples|Mean sample seconds|Median sample seconds|Train seconds/update|Train peak allocated bytes|Train peak reserved bytes|
|---|---|---|---|---|---|---|---|---|
|START|0|0|0|NA|NA|NA|NA|NA|
|H_CONT|0|0|0|NA|NA|NA|NA|NA|
|MOMENT_XY|0|0|0|NA|NA|NA|NA|NA|
|SURFACE_XY|0|0|0|NA|NA|NA|NA|NA|

Historical cache hits have no new sampling latency. Training phase wall time includes OT/checkpoint I/O; it is not isolated network time.

## Exact control curve counts

|Checkpoint|Split|Candidate|Noise|Valid/registered|All three targets strictly correct|One input strictly wins all targets|
|---|---|---|---|---|---|---|
|START|train|MODEL|all|0/8|NA|NA|
|START|train|MODEL|0|0/4|NA|NA|
|START|train|MODEL|1|0/4|NA|NA|
|START|validation|MODEL|all|0/8|NA|NA|
|START|validation|MODEL|0|0/4|NA|NA|
|START|validation|MODEL|1|0/4|NA|NA|
|H_CONT|train|MODEL|all|0/8|NA|NA|
|H_CONT|train|MODEL|0|0/4|NA|NA|
|H_CONT|train|MODEL|1|0/4|NA|NA|
|H_CONT|validation|MODEL|all|0/8|NA|NA|
|H_CONT|validation|MODEL|0|0/4|NA|NA|
|H_CONT|validation|MODEL|1|0/4|NA|NA|
|MOMENT_XY|train|MODEL|all|0/8|NA|NA|
|MOMENT_XY|train|MODEL|0|0/4|NA|NA|
|MOMENT_XY|train|MODEL|1|0/4|NA|NA|
|MOMENT_XY|validation|MODEL|all|0/8|NA|NA|
|MOMENT_XY|validation|MODEL|0|0/4|NA|NA|
|MOMENT_XY|validation|MODEL|1|0/4|NA|NA|
|SURFACE_XY|train|MODEL|all|0/8|NA|NA|
|SURFACE_XY|train|MODEL|0|0/4|NA|NA|
|SURFACE_XY|train|MODEL|1|0/4|NA|NA|
|SURFACE_XY|validation|MODEL|all|0/8|NA|NA|
|SURFACE_XY|validation|MODEL|0|0/4|NA|NA|
|SURFACE_XY|validation|MODEL|1|0/4|NA|NA|
