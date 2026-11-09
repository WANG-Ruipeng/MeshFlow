# Control MVP48 独立评价 — Session C

1. **T1是否跨出了双对象沙盒？** 矩阵有 1 个失败、94 个待完成请求，证据不完整。T1 在条件微调未参与的父对象上有效输出 1/32；排除 C 后双向表面 RMS NA（0/8 父对象） %L，GT coverage@1%L NA（0/8 父对象）。这些值及固定视图用于判断未知区域补全质量；仅输出有限值或复制 C 不能证明合理补全，本轮未登记绝对成功门槛，也未全面验证几何编辑控制。

2. **Geo是否有初步的独立父对象增量？** 矩阵有 1 个失败、94 个待完成请求，证据不完整。Geo−T1：双向表面 RMS（%L）均值差 NA，父对象配对中位差 NA，改善 0/0 父对象；GT coverage均值差 NA，父对象配对中位差 NA，改善 0/0 父对象。改善数以父对象为单位，个别条件和噪声波动见逐输出表。

3. **Full的接口/质量收益是否值得保留？** 矩阵有 1 个失败、94 个待完成请求，证据不完整。Full−Geo：有限边接口 RMS（%L）均值差 NA，父对象配对中位差 NA，改善 0/0 父对象；表面接口 RMS（%L）均值差 NA，父对象配对中位差 NA，改善 0/0 父对象；GT coverage均值差 NA，父对象配对中位差 NA，改善 0/0 父对象；CF 受影响 free 比例均值差 NA，父对象配对中位差 NA，改善 0/0 父对象；FF 非法受影响 free 比例均值差 NA，父对象配对中位差 NA，改善 0/0 父对象。是否保留 Full 需同时查看覆盖、接口与质量的方向和最差对象，不构造偏向 Full 的总分或自动切换模型。

4. **下一步应扩大数据、增加训练预算，还是先修数据/实现？** 先检查已记录的数据、身份、数值或评分失败，保留全部失败请求；不以扩训掩盖不完整评价。本次到冻结预算即停止；不自动扩训、换 checkpoint、扫描求解器或更改默认模型。

正式请求 **96**，实际尝试 **2/96**，评分有效输出 **1/96**；已尝试失败 **1**，待运行或待评分 **94**。所有请求均保留，登记占位行不计作失败尝试。
均值先在父对象内平均两块 C × 两噪声，再对 8 个父对象等权汇总。任何父对象少一个有效输出，该父对象完整均值为 NA；完整八对象总均值也为 NA。可用分母诊断另列于 JSON，不能替代主均值。

## 完整性、接口与质量

数值沿用原评分器的单位与阈值。C 本身排除于剩余几何分数，每块 C 使用自己的 GT 自由补集；不同 N 的相交对数需结合受影响 free 比例。

| 指标 | 单位 | T1 | Geo | Full | 有效分母 T1 / Geo / Full |
| --- | --- | --- | --- | --- | --- |
| remainder_symmetric_RMS_bbox_pct | % L | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| free_to_R12_RMS_bbox_pct | % L | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| R12_to_free_RMS_bbox_pct | % L | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| fixed_R12_coverage_RMS_bbox_pct | % L | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| boundary_RMS_bbox_pct | % L | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| boundary_p95_bbox_pct | % L | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| Gamma_to_free_edges_RMS_bbox_pct | % L | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| Gamma_to_free_edges_p95_bbox_pct | % L | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| fixed_R12_coverage_within_1pct_fraction | fraction | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| boundary_coverage_fraction | fraction | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| Gamma_to_free_edges_coverage_0.001_fraction | fraction | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| CF_proper_pair_count | count | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| CF_coplanar_pair_count | count | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| CF_unresolved_or_degenerate_pair_count | count | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| FF_proper_pair_count | count | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| FF_coplanar_pair_count | count | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| FF_illegal_pair_count | count | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| FF_unresolved_or_degenerate_pair_count | count | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| free_near_degenerate_count | count | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| free_exact_zero_area_count | count | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| free_nan_coordinate_count | count | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| free_inf_coordinate_count | count | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| CF_proper_affected_free_fraction | fraction | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| FF_proper_affected_free_fraction | fraction | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| FF_illegal_affected_free_fraction | fraction | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| free_minimum_angle_lt5_fraction | fraction | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| free_q_gt10_fraction | fraction | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| free_near_degenerate_fraction | fraction | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| free_exact_zero_area_fraction | fraction | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| free_angle_p05_degrees | degrees | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| free_q_mean | dimensionless | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| C_preserved | bool | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| actual_N | faces | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| actual_K | faces | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| actual_free_faces | faces | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| free_total_surface_area | model coordinate squared | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| GT_free_total_surface_area | model coordinate squared | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |
| free_area_over_GT_free_area | ratio | NA | NA | NA | 0/8父,1/32输出 / 0/8父,0/32输出 / 0/8父,0/32输出 |

## 父对象配对增量

差值方向固定为后者−前者：Geo−T1、Full−Geo、Full−T1。误差/非法相交越低越好，coverage 越高越好；描述性字段不判断改善。中位差和改善数的单位是父对象，不把 32 个 case 当作 32 个独立形状。

### Geo-T1

| 指标 | 均值差 | 父对象配对中位差 | 改善父对象/有效父对象 | 最差父对象(Δ) | 最差对象/C/噪声(Δ) |
| --- | --- | --- | --- | --- | --- |
| remainder_symmetric_RMS_bbox_pct | NA | NA | 0/0 | NA | NA |
| free_to_R12_RMS_bbox_pct | NA | NA | 0/0 | NA | NA |
| R12_to_free_RMS_bbox_pct | NA | NA | 0/0 | NA | NA |
| fixed_R12_coverage_RMS_bbox_pct | NA | NA | 0/0 | NA | NA |
| boundary_RMS_bbox_pct | NA | NA | 0/0 | NA | NA |
| boundary_p95_bbox_pct | NA | NA | 0/0 | NA | NA |
| Gamma_to_free_edges_RMS_bbox_pct | NA | NA | 0/0 | NA | NA |
| Gamma_to_free_edges_p95_bbox_pct | NA | NA | 0/0 | NA | NA |
| fixed_R12_coverage_within_1pct_fraction | NA | NA | 0/0 | NA | NA |
| boundary_coverage_fraction | NA | NA | 0/0 | NA | NA |
| Gamma_to_free_edges_coverage_0.001_fraction | NA | NA | 0/0 | NA | NA |
| CF_proper_pair_count | NA | NA | 0/0 | NA | NA |
| CF_coplanar_pair_count | NA | NA | 0/0 | NA | NA |
| CF_unresolved_or_degenerate_pair_count | NA | NA | 0/0 | NA | NA |
| FF_proper_pair_count | NA | NA | 0/0 | NA | NA |
| FF_coplanar_pair_count | NA | NA | 0/0 | NA | NA |
| FF_illegal_pair_count | NA | NA | 0/0 | NA | NA |
| FF_unresolved_or_degenerate_pair_count | NA | NA | 0/0 | NA | NA |
| free_near_degenerate_count | NA | NA | 0/0 | NA | NA |
| free_exact_zero_area_count | NA | NA | 0/0 | NA | NA |
| free_nan_coordinate_count | NA | NA | 0/0 | NA | NA |
| free_inf_coordinate_count | NA | NA | 0/0 | NA | NA |
| CF_proper_affected_free_fraction | NA | NA | 0/0 | NA | NA |
| FF_proper_affected_free_fraction | NA | NA | 0/0 | NA | NA |
| FF_illegal_affected_free_fraction | NA | NA | 0/0 | NA | NA |
| free_minimum_angle_lt5_fraction | NA | NA | 0/0 | NA | NA |
| free_q_gt10_fraction | NA | NA | 0/0 | NA | NA |
| free_near_degenerate_fraction | NA | NA | 0/0 | NA | NA |
| free_exact_zero_area_fraction | NA | NA | 0/0 | NA | NA |
| free_angle_p05_degrees | NA | NA | 0/0 | NA | NA |
| free_q_mean | NA | NA | 0/0 | NA | NA |
| C_preserved | NA | NA | 描述性 | NA | NA |
| actual_N | NA | NA | 描述性 | NA | NA |
| actual_K | NA | NA | 描述性 | NA | NA |
| actual_free_faces | NA | NA | 描述性 | NA | NA |
| free_total_surface_area | NA | NA | 描述性 | NA | NA |
| GT_free_total_surface_area | NA | NA | 描述性 | NA | NA |
| free_area_over_GT_free_area | NA | NA | 描述性 | NA | NA |

### Full-Geo

| 指标 | 均值差 | 父对象配对中位差 | 改善父对象/有效父对象 | 最差父对象(Δ) | 最差对象/C/噪声(Δ) |
| --- | --- | --- | --- | --- | --- |
| remainder_symmetric_RMS_bbox_pct | NA | NA | 0/0 | NA | NA |
| free_to_R12_RMS_bbox_pct | NA | NA | 0/0 | NA | NA |
| R12_to_free_RMS_bbox_pct | NA | NA | 0/0 | NA | NA |
| fixed_R12_coverage_RMS_bbox_pct | NA | NA | 0/0 | NA | NA |
| boundary_RMS_bbox_pct | NA | NA | 0/0 | NA | NA |
| boundary_p95_bbox_pct | NA | NA | 0/0 | NA | NA |
| Gamma_to_free_edges_RMS_bbox_pct | NA | NA | 0/0 | NA | NA |
| Gamma_to_free_edges_p95_bbox_pct | NA | NA | 0/0 | NA | NA |
| fixed_R12_coverage_within_1pct_fraction | NA | NA | 0/0 | NA | NA |
| boundary_coverage_fraction | NA | NA | 0/0 | NA | NA |
| Gamma_to_free_edges_coverage_0.001_fraction | NA | NA | 0/0 | NA | NA |
| CF_proper_pair_count | NA | NA | 0/0 | NA | NA |
| CF_coplanar_pair_count | NA | NA | 0/0 | NA | NA |
| CF_unresolved_or_degenerate_pair_count | NA | NA | 0/0 | NA | NA |
| FF_proper_pair_count | NA | NA | 0/0 | NA | NA |
| FF_coplanar_pair_count | NA | NA | 0/0 | NA | NA |
| FF_illegal_pair_count | NA | NA | 0/0 | NA | NA |
| FF_unresolved_or_degenerate_pair_count | NA | NA | 0/0 | NA | NA |
| free_near_degenerate_count | NA | NA | 0/0 | NA | NA |
| free_exact_zero_area_count | NA | NA | 0/0 | NA | NA |
| free_nan_coordinate_count | NA | NA | 0/0 | NA | NA |
| free_inf_coordinate_count | NA | NA | 0/0 | NA | NA |
| CF_proper_affected_free_fraction | NA | NA | 0/0 | NA | NA |
| FF_proper_affected_free_fraction | NA | NA | 0/0 | NA | NA |
| FF_illegal_affected_free_fraction | NA | NA | 0/0 | NA | NA |
| free_minimum_angle_lt5_fraction | NA | NA | 0/0 | NA | NA |
| free_q_gt10_fraction | NA | NA | 0/0 | NA | NA |
| free_near_degenerate_fraction | NA | NA | 0/0 | NA | NA |
| free_exact_zero_area_fraction | NA | NA | 0/0 | NA | NA |
| free_angle_p05_degrees | NA | NA | 0/0 | NA | NA |
| free_q_mean | NA | NA | 0/0 | NA | NA |
| C_preserved | NA | NA | 描述性 | NA | NA |
| actual_N | NA | NA | 描述性 | NA | NA |
| actual_K | NA | NA | 描述性 | NA | NA |
| actual_free_faces | NA | NA | 描述性 | NA | NA |
| free_total_surface_area | NA | NA | 描述性 | NA | NA |
| GT_free_total_surface_area | NA | NA | 描述性 | NA | NA |
| free_area_over_GT_free_area | NA | NA | 描述性 | NA | NA |

### Full-T1

| 指标 | 均值差 | 父对象配对中位差 | 改善父对象/有效父对象 | 最差父对象(Δ) | 最差对象/C/噪声(Δ) |
| --- | --- | --- | --- | --- | --- |
| remainder_symmetric_RMS_bbox_pct | NA | NA | 0/0 | NA | NA |
| free_to_R12_RMS_bbox_pct | NA | NA | 0/0 | NA | NA |
| R12_to_free_RMS_bbox_pct | NA | NA | 0/0 | NA | NA |
| fixed_R12_coverage_RMS_bbox_pct | NA | NA | 0/0 | NA | NA |
| boundary_RMS_bbox_pct | NA | NA | 0/0 | NA | NA |
| boundary_p95_bbox_pct | NA | NA | 0/0 | NA | NA |
| Gamma_to_free_edges_RMS_bbox_pct | NA | NA | 0/0 | NA | NA |
| Gamma_to_free_edges_p95_bbox_pct | NA | NA | 0/0 | NA | NA |
| fixed_R12_coverage_within_1pct_fraction | NA | NA | 0/0 | NA | NA |
| boundary_coverage_fraction | NA | NA | 0/0 | NA | NA |
| Gamma_to_free_edges_coverage_0.001_fraction | NA | NA | 0/0 | NA | NA |
| CF_proper_pair_count | NA | NA | 0/0 | NA | NA |
| CF_coplanar_pair_count | NA | NA | 0/0 | NA | NA |
| CF_unresolved_or_degenerate_pair_count | NA | NA | 0/0 | NA | NA |
| FF_proper_pair_count | NA | NA | 0/0 | NA | NA |
| FF_coplanar_pair_count | NA | NA | 0/0 | NA | NA |
| FF_illegal_pair_count | NA | NA | 0/0 | NA | NA |
| FF_unresolved_or_degenerate_pair_count | NA | NA | 0/0 | NA | NA |
| free_near_degenerate_count | NA | NA | 0/0 | NA | NA |
| free_exact_zero_area_count | NA | NA | 0/0 | NA | NA |
| free_nan_coordinate_count | NA | NA | 0/0 | NA | NA |
| free_inf_coordinate_count | NA | NA | 0/0 | NA | NA |
| CF_proper_affected_free_fraction | NA | NA | 0/0 | NA | NA |
| FF_proper_affected_free_fraction | NA | NA | 0/0 | NA | NA |
| FF_illegal_affected_free_fraction | NA | NA | 0/0 | NA | NA |
| free_minimum_angle_lt5_fraction | NA | NA | 0/0 | NA | NA |
| free_q_gt10_fraction | NA | NA | 0/0 | NA | NA |
| free_near_degenerate_fraction | NA | NA | 0/0 | NA | NA |
| free_exact_zero_area_fraction | NA | NA | 0/0 | NA | NA |
| free_angle_p05_degrees | NA | NA | 0/0 | NA | NA |
| free_q_mean | NA | NA | 0/0 | NA | NA |
| C_preserved | NA | NA | 描述性 | NA | NA |
| actual_N | NA | NA | 描述性 | NA | NA |
| actual_K | NA | NA | 描述性 | NA | NA |
| actual_free_faces | NA | NA | 描述性 | NA | NA |
| free_total_surface_area | NA | NA | 描述性 | NA | NA |
| GT_free_total_surface_area | NA | NA | 描述性 | NA | NA |
| free_area_over_GT_free_area | NA | NA | 描述性 | NA | NA |

## 失败、验证与控制边界

| 父对象 | C | 噪声 | 模型 | 状态 | 原因 |
| --- | --- | --- | --- | --- | --- |
| parent0 | c0 | 0 | Geo | SCORING_ERROR | RuntimeError('Synthetic scoring failure; no retry permitted') |

待运行/待评分状态：{"NOT_RUN": 94}。逐项身份保留在 per_output.csv 与 results.json 的 pending_requests 中。

验证 teacher 请求 **24**；已登记 **24**、表内记录 **24**、实际尝试 **0/24**、有效检查 **0/24**；登记或待运行记录不是已执行的前向。检查固定 t=.5、第一块 C、冻结 teacher 噪声及配对。teacher 的 GT/OT 状态与自由生成严格分开，仅解释学习状态，不替代完整生成、不挑 checkpoint、不推断完整控制能力。

同 N/K 正确/交换条件比较：**NOT_RUN**。Synthetic fixture

同一父对象的两块 C 来自同一完整目标，输出不同不等于几何编辑成功，也不要求完整形状不同。若未完成冻结的跨父对象四格比较，本轮尚未全面验证几何编辑控制。

精确拓扑与水密认证：**NOT_RUN**。C 保持、实际面数、退化、NaN/Inf、UNRESOLVED、角质量以逐输出原记录为准，未运行的认证不填 0。

## 协议、身份与实际账本

协议目标终点：T1_step1000、Geo_step1000、Full_step1000；实际统一严格加载及身份核对状态详见 results.json 的 identity，共同官方初始化、数据 hash、任务步数、分支来源按 protocol 核对。无 step500 选择，无 raw/EMA 混选。

原 Euler50；BF16 前向、FP32 参数/状态/积分；严格确定性、TF32 关闭、SDPA math、eval/no_grad；每条 51 状态保持 C。raw known 速度不要求恒零。没有训练 loss、自由生成 GT/OT 初始化、Guidance、Pair 或几何修补。

正式 NFE 请求 4800；显式记录实际 NFE 的输出 2/96，已记录实际前向合计 100；完整实际合计 NA。teacher、必要重复和失败尝试另见实际账本，不混入 96 条。

```json
{
  "sampling": {},
  "teacher": {},
  "scoring": {
    "synthetic_score_calls": 2
  },
  "scoring_wall_seconds": 0.023547303999976066,
  "GPU_status": "STOPPED",
  "scoring_status": "STOPPED",
  "CPU_preparation_OT_attempts": 0,
  "CPU_preparation_OT_returns": 0,
  "CPU_scoring_workers": 1,
  "GPU_workers": 1,
  "repeats_requested": 0,
  "repeats_actual": 0,
  "generation_GT_OT_calls": 0,
  "postprocessing_repairs": 0,
  "retries": 0,
  "phase_timing": {},
  "resources": {}
}
```

GPU 阶段要求使用共同锁且只有一个 worker；CPU 评分可与后续推理重叠。实际资源快照、峰值及分阶段 wall 以账本为准，并行争用下的 wall 不作为隔离性能比较。

## 交付与限制

[results.json](results.json)、[逐请求 96 行](per_output.csv)、[逐父对象 8 行](per_parent.csv)、[24 请求 teacher 表](teacher_validation.csv)、[固定视图](../figures/)。正式视图目标是固定第一份噪声，8 个对象的两块 C 均展示 GT、C、T1、Geo、Full，同视角同范围，不删除失败面板。准备阶段的 GT/C 预览与 NOT_RUN 模型占位面板不代表已生成模型输出；实际图件状态见 results.json 的 figures。

本轮证据范围是条件微调未参与的 8 个父对象，不等于官方预训练从未见过；没有论文级泛化或统计定论。1000 步无效果仅能称该预算下未成立，不能否定整个方法；持续学习的终点不能称充分收敛。没有临时 10% 成功门槛或偏向 Full 的总分。

正式 96 条完成后停止；不自动扩训、扫 loss、改求解器或切换默认模型。
