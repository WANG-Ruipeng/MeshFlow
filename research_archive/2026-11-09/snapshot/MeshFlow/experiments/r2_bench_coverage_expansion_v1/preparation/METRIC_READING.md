# Bench coverage 终点评分解读口径

2026-10-21；只读核对本轮 collector、原 scorer、指标字典后整理。本文不改指标、不增加评分，不包含本轮尚未完成的结果。

比较是 `COVERAGE_S1500 − A_S1500`，即同一 A step0/fresh AdamW、1,500 更新，在固定 48 请求上的条件覆盖策略整体差异。旧 A 是新臂的精确初始化/更新预算对照，不能将本轮直接称为“超过所有 Bench 历史配置”。冻结 TRAIN 321 条/79 父相对 A 的 195 条/58 父，同时恢复闭合/无 owner 条件及加入新父；因此不能把结果单独归因于新增父数、腿类型或单个条件比例。

## 1. 收集方式和单位

原始 `row.json` 为准。collector 不重新评分；复制全部原字段，增加标签与元数据，并认证原 row/receipt 和共同评分源码哈希。旧错误 collector 覆盖 whole-mesh `boundary_edges` 的行为没有继承：本轮 `boundary_edges` 始终为网格指标，`C_boundary_edges` 单独保存输入 C 的边界数量。

`collect.numeric_fields` 选取数值字段，排除 bool 以及 `noise_index/N/K/C_boundary_edges`。旧 A corrected completion 实际列出 **177 个数值聚合字段**；其中包括面积、计数和分母等上下文字段，不等于 177 个有“越大/越小越好”方向的性能分数。原始布尔与状态字段仍在逐输出表，例如 `C_preserved_exact`，需要按合同单独核查；不会自动进入均值表。最终本轮字段数以 `results/completion.json.metric_fields` 为准，不预设179列数值。

- `*_fraction`、`A_VMR__*`、`PMR__*`、`ConflictCov_C*` 通常是 0–1；报告百分比时乘100，配对差乘100为**百分点 pp**，不是相对百分比。
- `*_count` 是面对/面/边计数，不能乘100；`*_length`、`*_distance_*_model` 为模型单位；面积为模型单位平方。
- 固定长度标尺 `L=sqrt(3)*2/0.3762≈9.2081382646`，若另展示距离占比则模型距离除 L；不对每份输出重新用 bbox 归一化。
- 负 delta 在下降方向字段表示改善；正 delta 在上升方向字段表示改善。上下文字段不能自动染红/染绿或计入“赢了几个 metric”。

## 2. 主要质量和接续字段

| 常用简称/字段 | 方向及实际分母 | 必须同时说明的限制 |
|---|---|---|
| CF：`CF_illegal_affected_free_fraction` | 低；参与至少一个非法 C/F 面对的不同 free 面数 / 当前 free 面数 | 不等于面对数占所有可能面对的比例；同时保留 `CF_illegal_pair_count`。`CF_proper_*` 仅穿插，不等于全部非法（还包括共面正面积重叠）。 |
| FF：`FF_illegal_affected_free_fraction` | 低；参与至少一个非法 F/F 面对的不同 free 面数 / 当前 free 面数 | 不是 whole-mesh self-intersection ratio，也不是 `FF_illegal_pair_count / FF_tested_pair_denominator`；后处理增删面改变分母。 |
| I：`ConflictCov_C` | 低；固定 C 上2048个面积加权查询点被局部穿插/重层覆盖的权重比例 | 是 **C 域冲突覆盖**，不能表述为全椅子冲突面积；其较低不能排除 bulk 内部 FF 或主体缺失。 |
| `ConflictCov_C_cross/dup` | 低；原定义的穿插附近覆盖/内部近共面覆盖 | 两项可重叠。`I=cross+dup-cross_and_dup`；`cross_and_dup` 是交集账本，不单独追求下降。保留 upper、status、uncertain。 |
| `free_minimum_angle_lt5_fraction` | 低；最小内角<5°的 free 面占比 | 是三角质量诊断，不把所有小角面自动叫视觉碎片；与面积、图像、连通分量一起看。 |
| `free_q_gt10_fraction`、`free_q_mean` | 低；q=(三边平方和)/(4√3×面积)，等边 q=1 | 面退化时有无穷/未定义情况；原 scorer 的 per-face status 保留。 |
| `free_angle_p05_degrees` | 高；free最小内角分布的5%分位 | 不能代替主体形状判断。 |
| `free_near_degenerate_*`、`free_exact_zero_area_*`、`free_exact_duplicate_excess_fraction` | 低，按原阈值/精确检测 | 退化、重复和非法相交是不同定义，不合并成一种坏面。 |
| `nonmanifold_edges`、`C_F_nonmanifold_edge_count` | 低；实际索引边的非法容量计数 | 无非流形不等于水密、无交或全局可定向。 |
| `C_F_valid_edge_count/total_length`、`C_F_has_valid_edge` | 高是接续描述性收益 | 通过原合法性检查的实际共边；只接一张孤立 seam 也可能取高值，不能认证接入主体/用途正确。 |
| `C_F_valid_opposite_winding_edge_count`、`C_F_has_opposite_winding_edge` | 高；合法且绕序相反的 C/F 接口 | 来源角序已恢复；无向合法与方向一致仍分开。 |
| `free_only_reached_bulk_area_fraction` | 高；从合法 C/F seam 种子出发，只沿合法 F/F 共边到达的 bulk 面积 / 全部保留 bulk 面积 | BFS不穿越 C；BRIDGE可作路径但不计入bulk分子/分母。这是本轮 seam→bulk 主体接续描述，不等于整圈边界封合。 |
| `bulk_reach_ge_0_30_descriptive` | 高；上述比例≥0.30的逐输出0/1标签 | 阈值是已登记描述，不是通用控制成功/水密/可用标准。 |
| `legal_seam_bulk_edge_count/length` | 高；全部合法 seam/bulk 共边 | `seed_connected_legal_seam_bulk_*` 才只数实际从C种子可达的部分。 |
| `opposite_winding_reached_free_area_fraction` | 高；C出发方向一致图到达的全部free面积占比 | 与 free-only bulk 可达域、分母不同，不能互换同一列。 |
| `hinge_diag_isolated_1face_edge_components` | 低；合法共边图的单面free分量数 | 单面分量可共点挂在主体上，也可能几何重叠；不是“所有漂浮碎片”的充分检测。另看 `...unfiltered_edge_components` 和图像。 |
| `A_VMR__*`、`PMR__*` | 高；全部C来源开放边界顶点对 **F角点** 的距离匹配 | A-VMR为匹配顶点比例；PMR逐输出全匹配0/1。只匹配顶点，不验证整条边或真实接入主体；SEAM硬锚点可使它天然很高。 |
| `boundary_distance_mean/median/p95/max_model` | 低；上述边界顶点到F角点的最近距离 | 这里不是有限线段/有限三角面距离，不能用于证明长边中段没有缺口。 |

AVMR/PMR 各阈值均保留：模型单位 `raw_tau_1e-6`，以及固定 L 比例 `1e-6、1e-4、.001、.002、.005、.01、.02`；匹配判据严格 `< threshold`。不能选一个最有利阈值后当成原文统一指标。全部来源开放边界是诊断域，不自动声明每条边都必须封闭。

## 3. 后处理、分母与非单调字段

`actual_free_faces`、`retained_bulk_faces/area`、`retained_seam_faces/area`、`free_area`、`free_total_surface_area`、`whole_total_surface_area`、`boundary_edges`、`hinge_diag_open_edge_count`、`new_bridge_faces`、`bridge_area` 都是上下文；减少面/面积可能是清理收益，也可能是主体缺失，不能单独判好。

`deleted_free_faces` 按丢失的原 RAW 槽身份计数，**不是 N减最终面数**；新增BRIDGE不能抵消原槽丢失。`deleted_original_POST_faces` 则相对本请求 BASE_POST；两者阶段基准不同。各模型的BASE_POST free总数本身可不同。

最终级联新增桥面会扩大当前free分母。务必配合 `fixed_parent_CF/FF_illegal_affected_fraction` 和 `fixed_parent_*_fraction` 看原BASE_POST面身份集合的变化，避免新增干净桥面稀释冲突/小角比例。`fixed_parent_free_face_denominator` 是 **当前模型此请求** 的原BASE_POST free面数，不能误解为两个模型共同的原始训练父分母；跨模型不能只拿 fixed_parent 比例忽视其分母差异。

桥面相关新增字段按实际含义读：`bridge_*illegal*`、`illegal_pairs_bridge-*`、`bridge_*degenerate*`、`bridge_minimum_angle_lt5_count`、`bridge_q_gt10_count/q_mean` 是下降方向；`bridge_minimum_angle_degrees` 是上升方向；桥面数/面积及桥面可达面积是增补代价和结构描述，不构成独立成功。无桥面时 `bridge_shape_status=NA_NO_BRIDGE`。

`hinge_diag_vertex_attached_only_*`、`hinge_diag_fully_C_disconnected_free_area_fraction` 沿用原方向字典为上下文；例如单点挂接减少可能是变成共边，也可能是彻底断开，不能自动算改善。`RAW_SOUP_*`、`EXACT_COORD_*` 是特定拓扑域的描述；RAW_SOUP不代表已保存的实际SEAM共享索引图，不能用汤表示的共边0覆盖真实接口。

质量分区 `illegal_pairs_bulk-bulk / seam-bulk / seam-seam / C-bulk / C-seam` 保留面对数，可定位代价变化；它们不会单独证明动力学/语义因果。`C_*` 质量字段来自固定输入，是上下文，不能把输入本来小角面算成模型改进。

## 4. 聚合、闭合C与失败

逐输出→同一C两噪声均值→同一父两C均值→split内父等权。这是 `per_condition`、`per_parent`、`parent_macro_summary` 的实际顺序。TRAIN4父/16输出，DEV8父/32输出。不能把两噪声、两条件或多张图片当成独立父扩大样本量。

闭合C有8份输出：TRAIN2、DEV6；对应原 scorer 的 seam→bulk、AVMR/PMR 等为空边界/不适用项为NA。开放C有效接口诊断最多TRAIN14、DEV26。不能把NA补0，也不能写成所有48请求都需要SEAM接口。即使 closed C 的数值图统计可计算，也没有“必须接通开放边”的任务要求。

`aggregate()` 按指标分别过滤非有限值，输出 `<metric>__valid_n`。它可能在多层聚合中表示有效条件数或有效父数，不是原输出数量；需要逐层读分母。尤其某父的一条条件闭合时，其父接口均值只来自另一条适用条件。总览必须同时报告具体适用输出/父数、NA原因和失败状态。`ConflictCov_C_status=LOW_SUPPORT/PARTIAL` 的数值不会被collector自动筛掉；原上下界和状态需单独披露。

`results/paired_per_output.csv` 保存 `__reference/__candidate/__delta`；后两张配对表先对同一固定请求做差再按条件/父聚合。报告优先看DEV逐父改变量、改善/恶化父数与极端尾部；TRAIN另列。不能用TRAIN大幅提升抵消DEV恶化，也不能只报一个总体均值。

collector没有自动构造置信区间、图审标签或通用综合分。原始bool保留但不数值聚合，因此 `C_preserved_exact` 必须显式核验全部为true；各阶段 source restoration/anchor receipts也需通过。不能仅看均值CSV声称硬合同全部通过。

## 5. 报告应该回答的问题

先依据固定图集和真实非盲 `AI_PROXY` 记录，分别回答“C是否被合理采用”和“主体是否完整、是否错层/悬片/穿插”；不要求匹配唯一来源款式。图审不是神经评分器，也不能从已算CF/FF直接填标签。截面线段端点本身不能认证裂缝，应结合两视角、多个阶段及实际索引连接。

再看RAW变化是否已出现，BASE_POST是否放大/缩小差异，CLEAN_FINAL_POST是否以桥面/删面等代价改善。保留所有阶段，不能只选最终最漂亮阶段。

可以支持继续的信号是：多个DEV父的采用及整体质量同时改善，重要冲突/缺面/碎片代价可接受；接续数字单独上涨不够。未改善也仅否定本轮起点、固定1,500步和这套扩覆盖策略，不能证明原始数据无问题、数据覆盖已足够或所有增加数据都无效。

依据文件：`native_t1/docs/control_metrics.md`；`r2/tools/hinge_opposite_v1/metric_policy.json`；`r2/tools/hinge_v2_cpu_v1/scoring.py`；`r2/pilots/bench_clean_multipart_seam_v1/post_cpu.py`；`r2/pilots/bench_adaptation_strength_v1/collect_v2.py`；本轮 `collect.py`。保持旧来源与结果只读。
