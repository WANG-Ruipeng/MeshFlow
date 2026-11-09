# 冻结Native模型：几何探针与有门禁的预测—修正验证

执行状态：**PASS_EXECUTION**。本轮零新增训练；原V/S公式审计、一次离线R/S修正和完整预测—修正采样分别报告。
真实几何方向只读当前free、C及已知尺度L；完整GT仅用于离线评分。L来自旧训练对象bbox元数据，不是未知目标部署能力。
开发9201–9204是重复使用的回归噪声；确认噪声只有按门禁实际执行时才计入新噪声结果。

## 1. 原顶点斥力漏掉什么，能量下降是否等于缺陷下降？

32态审计记录2566个state-pair穿刺观测，其中原能量不激活2318个、局部梯度为零2318个；三个顶点距C平面均在带外仍穿刺2180个。原势能下降但同对仍proper穿刺228个，不能将能量下降等同于缺陷消除。k=15时已有proper相交的轨迹为8/8。原proper对在实际FP32一步后可确认合法28个，转共面重叠0个，未定/近退化0个；旧proper分类消失不能直接记作安全清除。

## 2. 整三角形R是否减少相交，是否推远或转移缺陷？

范围：完整预测—修正采样；R离线晋级门禁：PASS。P0（4/4条）穿刺面率19.25%→12.25%，FF非法pair均值122.2500→125.2500，R12 RMS 1.6109→1.6277% bbox，边界RMS 1.0179→1.0249% bbox；P1（4/4条）穿刺面率13.75%→8.75%，FF非法pair均值97.0000→96.5000，R12 RMS 0.7557→0.7574% bbox，边界RMS 0.2298→0.2205% bbox。同一终点对比保留CF与FF的新增/移除pair集合、位移和两个表面误差方向；即使总数下降，也不能隐藏新pair或质量代价。

## 3. S改善整段边界还是仅改善已有近邻点？

S离线门禁：PASS。P0（4/4条）穿刺面率19.25%→17.50%，FF非法pair均值122.2500→125.0000，R12 RMS 1.6109→1.6080% bbox，边界RMS 1.0179→1.0082% bbox；P1（4/4条）穿刺面率13.75%→13.25%，FF非法pair均值97.0000→98.2500，R12 RMS 0.7557→0.7627% bbox，边界RMS 0.2298→0.2172% bbox。主结果是固定整段Gamma样点的长度权重RMS/p95/coverage；near筛选顶点距离仅作局部候选诊断，不替代完整边界。

## 4. R+S协同、抵消，还是只增加成本？

P0的RS相对R/S在穿刺、边界、R12和FF四项均值均不差：未通过；穿刺面率R/S/RS=12.25/17.50/13.00%，修正秒均值R/S/RS=43.8069/6.1786/51.9775；P1的RS相对R/S在穿刺、边界、R12和FF四项均值均不差：未通过；穿刺面率R/S/RS=8.75/13.25/8.00%，修正秒均值R/S/RS=46.6612/6.3270/54.7404。这是固定样本上的指标组合与成本比较，不等于已证明协同机制；若任一项未改善，不能仅凭另一项下降包装整体成功。 P0（4/4条）穿刺面率19.25%→13.00%，FF非法pair均值122.2500→121.0000，R12 RMS 1.6109→1.6264% bbox，边界RMS 1.0179→1.0221% bbox；P1（4/4条）穿刺面率13.75%→8.00%，FF非法pair均值97.0000→99.2500，R12 RMS 0.7557→0.7658% bbox，边界RMS 0.2298→0.2214% bbox。离线单次改进不能冒充完整生成中的协同；主干每一步会重新影响几何。

## 5. P0的9201、9204改善的是覆盖还是错误表面？

R/9201：两方向均未改善（覆盖Δ+0.04177、free→GT Δ+0.02083个bbox百分点）；R/9204：覆盖未改善、反向改善（覆盖Δ+0.00111、free→GT Δ-0.00070个bbox百分点）；S/9201：两方向均未改善（覆盖Δ+0.00000、free→GT Δ+0.00000个bbox百分点）；S/9204：两方向均改善（覆盖Δ-0.00061、free→GT Δ-0.00052个bbox百分点）；RS/9201：两方向均未改善（覆盖Δ+0.04177、free→GT Δ+0.02083个bbox百分点）；RS/9204：覆盖未改善、反向改善（覆盖Δ+0.00229、free→GT Δ-0.00147个bbox百分点）。两噪声不被预设为同一根因；表面距离方向只说明几何误差变化，不认证形状语义或拓扑。

## 6. 新噪声复现收益了吗？

未执行新噪声确认（NOT_RUN_NO_CANDIDATE_PASSED）：没有候选通过全部预登记门槛；因此没有新噪声复现证据，也没有强选不合格的最好组。

## 7. 哪些按构造保持、哪些仅实测、哪些未运行？

按实现设计保持C原FP32值、112总面/100自由面、冻结T1参数和50次原Euler；实测C逐位与调用账本另列。修正接受只约束回写FP32的静态端点及实际检查的pair集合，不保证下一次主干预测无碰撞。CCD=NOT_RUN；没有索引焊接、删面、重三角化或拓扑缝合，水密/流形未认证。本轮optimizer更新、模型参数更新、主干反向均必须为0；未启动原Phase3。

## 实际运行配置与冻结检查

| 检查 | 记录值 |
| --- | --- |
| 真实原V epsilon/L（用户明确选择） | 0.0050 |
| 解析fixture epsilon | 0.1000 |
| 严格加载状态 | PASS |
| 严格加载 missing/unexpected | [] / [] |
| 可训练参数数 | 0 |
| 模型state运行前SHA256 | 37e31fe5925b8f870b9a4d3bbbbc84edc7bf30177946a6f67f4529a7f8e97172 |
| 模型state运行后SHA256 | 37e31fe5925b8f870b9a4d3bbbbc84edc7bf30177946a6f67f4529a7f8e97172 |
| 前后state逐字节哈希相同 / frozen检查 | True / True |
| 关闭修正完整51态重放 | PASS_BITWISE_51_BOTH_PARENTS |
| 两条重放最大绝对差 | [0.0, 0.0] |
| 实际前向 CUBLAS_WORKSPACE_CONFIG | :4096:8 |
| 实际 deterministic / warn_only | True / False |
| 实际 autocast启用 / dtype | True / torch.bfloat16 |
| 实际参数dtype / FP32积分状态 | torch.float32 / True |
| 实际SDPA math / flash / efficient / cudnn | True / False / False / False |
| 实际TF32 matmul / cudnn | False / False |
| 实际全部eval / gradient_enabled | True / False |
| 协议SHA256 | 87ec904ca39826aa93357512992145520f0335436f85cae0a2bc024309191676 |

运行配置取第一次实际完整重放内的前向上下文，不使用加载前尚未进入autocast/SDPA上下文的快照替代。原V真实epsilon=0.005L与fixture epsilon=0.1分别记录；原V h=.02、lambda=20只用于原公式一步审计，不是正式R/S修正参数。

## 资产与保护

| 资产 | 值 |
| --- | --- |
| T1本地路径 | /mnt/e/MeshFlow/experiments/mf_h1_local0/outputs/native_patch_20260929_010000/Native_correct/endpoint_step500.pt |
| T1 SHA256 | 7928d77507bf1fe610876f61b5b11e614263323b5fc0f595cc6740ffd24af634 |
| 父对象 | ['2af09bd8df40506c9e646678ef50aa3d', '5b0e833cf2ea465e42bd82e95587d8af'] |
| N/K/free | 112/12/100 |
| 历史元数据无改变 | PASS |
| 历史文件数 | 787 |
| 关键资产字节哈希 | PASS |

使用混合K4/8/12的Native_correct step500；不复制1.56GB终点或官方EMA。输入源面ID和Gaussian哈希由资产审计记录。

## 原公式与独立检测器探针

| fixture | 状态 | 观察 |
| --- | --- | --- |
| outside_band_vertices_inside_triangle_penetration | PASS_COUNTEREXAMPLE | Full triangle intersects even though original V has zero energy and gradient |
| lower_original_energy_persistent_penetration | PASS_COUNTEREXAMPLE | Energy decrease is not defect decrease |
| exact_plane_abs_autograd | PASS_COUNTEREXAMPLE | Positive original V has zero local autograd gradient at d=0 |
| step_interior_crossing_clear_endpoints | PASS_COUNTEREXAMPLE | Endpoint acceptance cannot certify a collision-free full step |
| legal_contacts | PASS_DETECTORS | 详见JSON |
| coplanar_positive_area_overlap | PASS_DETECTORS | 详见JSON |
| T_junction_zero_gap_not_shared_discrete_edge | PASS_COUNTEREXAMPLE | Geometric line coverage does not split AB or weld the indexed connectivity |
| full_attraction_projection_collapses_area | PASS_COUNTEREXAMPLE | Counterexample full attraction step; not the registered bounded S beta=.25 corrector |
| repulsion_attraction_conflict_and_nonzero_equilibrium | PASS_COUNTEREXAMPLE | Opposing forces can balance at nonzero gap; they are not assumed orthogonal |

解析反例被复现代表探针成功，并非算法保证成立。原V是真实的双侧顶点带罚项；原S是near顶点到有限线段的平方距离。真实轨迹原V只做h=.02、lambda=20的一步；它不是正式R的整面有限平移。精确平面abs梯度、步内穿越、合法接触、共面重叠、T-junction和吸附退化均独立保留。

## 原始轨迹对账与32态审计

| 父对象 | proper自由面累计 | proper pairs | 边界RMS %bbox | 全边界coverage % |
| --- | --- | --- | --- | --- |
| P0 | 77/400 | 100 | 1.0179 | 48.2021 |
| P1 | 55/400 | 81 | 0.2298 | 53.4162 |

| Raw父/seed | proper面 | CF pairs | FF非法pairs | R12 RMS % | free→R12 % | R12→free % | 边界RMS % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0/9201 | 16 | 26 | 148 | 3.5064 | 3.3010 | 3.7004 | 3.5203 |
| P0/9202 | 13 | 19 | 123 | 0.0913 | 0.0886 | 0.0940 | 0.1265 |
| P0/9203 | 33 | 37 | 110 | 0.1642 | 0.1352 | 0.1889 | 0.3628 |
| P0/9204 | 15 | 18 | 108 | 2.6815 | 3.7892 | 0.1505 | 0.0622 |
| P1/9201 | 13 | 16 | 91 | 0.0739 | 0.0681 | 0.0794 | 0.1294 |
| P1/9202 | 16 | 21 | 86 | 1.0222 | 0.9239 | 1.1119 | 0.4363 |
| P1/9203 | 12 | 16 | 107 | 0.9097 | 1.1206 | 0.6321 | 0.1146 |
| P1/9204 | 14 | 28 | 104 | 1.0168 | 1.4193 | 0.2313 | 0.2388 |

原指标以同一历史分类器逐pair对账；新增独立分类器的UNRESOLVED单列，不能悄悄改定义再与19.25%/13.75%比较。实际已记录8/8个终点的完整4950个free/free无序pair检查；已记录32/32态原势能审计，更新另检查所有CF及涉及实际改动面的FF。

| 父/seed/k | proper对/面 | 原E不激活 | 三个顶点带外 | E下降仍proper | proper前→后 | 新增CF/FF |
| --- | --- | --- | --- | --- | --- | --- |
| P0/9201/15 | 128/52 | 126 | 126 | 2 | 128→129 | 1/0 |
| P0/9201/30 | 82/46 | 72 | 69 | 10 | 82→83 | 1/0 |
| P0/9201/40 | 65/39 | 60 | 57 | 4 | 65→64 | 0/0 |
| P0/9201/50 | 26/16 | 26 | 26 | 0 | 26→26 | 0/0 |
| P0/9202/15 | 145/68 | 137 | 130 | 8 | 145→145 | 0/0 |
| P0/9202/30 | 113/58 | 106 | 99 | 6 | 113→112 | 1/0 |
| P0/9202/40 | 67/43 | 61 | 59 | 6 | 67→67 | 0/0 |
| P0/9202/50 | 19/13 | 7 | 0 | 10 | 19→13 | 0/5 |
| P0/9203/15 | 143/61 | 138 | 133 | 5 | 143→143 | 0/1 |
| P0/9203/30 | 110/54 | 100 | 93 | 10 | 110→110 | 0/0 |
| P0/9203/40 | 71/43 | 65 | 61 | 6 | 71→71 | 0/1 |
| P0/9203/50 | 37/33 | 6 | 2 | 24 | 37→37 | 8/6 |
| P0/9204/15 | 150/63 | 144 | 142 | 5 | 150→149 | 0/0 |
| P0/9204/30 | 106/53 | 99 | 95 | 6 | 106→105 | 0/0 |
| P0/9204/40 | 64/41 | 56 | 54 | 8 | 64→64 | 0/0 |
| P0/9204/50 | 18/15 | 5 | 0 | 12 | 18→17 | 0/12 |
| P1/9201/15 | 129/45 | 127 | 122 | 2 | 129→129 | 0/0 |
| P1/9201/30 | 93/39 | 90 | 87 | 3 | 93→93 | 0/0 |
| P1/9201/40 | 58/31 | 52 | 49 | 6 | 58→59 | 1/0 |
| P1/9201/50 | 16/13 | 3 | 0 | 13 | 16→16 | 0/0 |
| P1/9202/15 | 112/42 | 108 | 105 | 4 | 112→112 | 0/0 |
| P1/9202/30 | 91/36 | 83 | 75 | 8 | 91→91 | 0/0 |
| P1/9202/40 | 58/32 | 53 | 47 | 5 | 58→58 | 0/0 |
| P1/9202/50 | 21/16 | 5 | 0 | 14 | 21→19 | 1/0 |
| P1/9203/15 | 128/43 | 125 | 124 | 3 | 128→128 | 0/1 |
| P1/9203/30 | 101/39 | 96 | 93 | 5 | 101→101 | 0/0 |
| P1/9203/40 | 62/35 | 58 | 54 | 4 | 62→62 | 0/0 |
| P1/9203/50 | 16/12 | 4 | 0 | 9 | 16→12 | 0/3 |
| P1/9204/15 | 131/44 | 129 | 123 | 2 | 131→131 | 0/0 |
| P1/9204/30 | 108/44 | 103 | 98 | 5 | 108→108 | 0/0 |
| P1/9204/40 | 70/38 | 65 | 57 | 5 | 70→70 | 0/0 |
| P1/9204/50 | 28/14 | 9 | 0 | 18 | 28→28 | 1/4 |

以上是state-pair和state-free-face观测；同一轨迹面在不同时刻重复出现，不能当成独立新对象。独立复核实际记录32项；预定顺序要求≥16个proper与≥16个非proper/接触（不足全检），没有按改进效果挑案例。

## 离线门禁与逐阶段指标

R/S/RS各自从同一个Raw终点独立开始；离线一次调用不是50步生成。R至少2/8减少proper且不新增非法/退化，S至少2/8改善完整Gamma距离且不新增非法/退化。未过门禁的完整采样组标NOT_RUN，不能以离线结果冒充。

| 阶段/组/父 | n | proper面累计 | 均值/中位数 % | 逐输出<1% | CF overlap/未定 | FF非法/未定 | 边界RMS/p95 % | coverage % | R12双向RMS % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offline_one_call/B/P0 | 4 | 77/400 | 19.25/15.50 | 0/4 | 0.0000/0.0000 | 122.2500/0.0000 | 1.0179/1.5156 | 48.2021 | 1.6109 |
| offline_one_call/B/P1 | 4 | 55/400 | 13.75/13.50 | 0/4 | 0.0000/0.0000 | 97.0000/0.0000 | 0.2298/0.5334 | 53.4162 | 0.7557 |
| offline_one_call/R/P0 | 4 | 60/400 | 15.00/12.00 | 0/4 | 0.0000/0.0000 | 117.7500/0.0000 | 1.0148/1.5079 | 54.2048 | 1.6092 |
| offline_one_call/R/P1 | 4 | 40/400 | 10.00/10.50 | 0/4 | 0.0000/0.0000 | 94.2500/0.0000 | 0.2295/0.5351 | 53.9579 | 0.7544 |
| offline_one_call/S/P0 | 4 | 77/400 | 19.25/15.50 | 0/4 | 0.0000/0.0000 | 120.5000/0.0000 | 1.0113/1.5021 | 51.2947 | 1.6099 |
| offline_one_call/S/P1 | 4 | 54/400 | 13.50/13.50 | 0/4 | 0.0000/0.0000 | 95.2500/0.0000 | 0.2169/0.5143 | 59.4593 | 0.7546 |
| offline_one_call/RS/P0 | 4 | 60/400 | 15.00/12.00 | 0/4 | 0.0000/0.0000 | 116.7500/0.0000 | 1.0082/1.4933 | 57.1105 | 1.6084 |
| offline_one_call/RS/P1 | 4 | 40/400 | 10.00/10.50 | 0/4 | 0.0000/0.0000 | 93.0000/0.0000 | 0.2188/0.5191 | 58.7575 | 0.7534 |
| development_sampling/B/P0 | 4 | 77/400 | 19.25/15.50 | 0/4 | 0.0000/0.0000 | 122.2500/0.0000 | 1.0179/1.5156 | 48.2021 | 1.6109 |
| development_sampling/B/P1 | 4 | 55/400 | 13.75/13.50 | 0/4 | 0.0000/0.0000 | 97.0000/0.0000 | 0.2298/0.5334 | 53.4162 | 0.7557 |
| development_sampling/R/P0 | 4 | 49/400 | 12.25/11.50 | 0/4 | 0.0000/0.0000 | 125.2500/0.0000 | 1.0249/1.5270 | 49.0879 | 1.6277 |
| development_sampling/R/P1 | 4 | 35/400 | 8.75/8.50 | 0/4 | 0.0000/0.0000 | 96.5000/0.0000 | 0.2205/0.5113 | 55.0552 | 0.7574 |
| development_sampling/S/P0 | 4 | 70/400 | 17.50/13.50 | 0/4 | 0.0000/0.0000 | 125.0000/0.0000 | 1.0082/1.4984 | 57.5040 | 1.6080 |
| development_sampling/S/P1 | 4 | 53/400 | 13.25/14.00 | 0/4 | 0.0000/0.0000 | 98.2500/0.0000 | 0.2172/0.5165 | 58.0398 | 0.7627 |
| development_sampling/RS/P0 | 4 | 52/400 | 13.00/13.50 | 0/4 | 0.0000/0.0000 | 121.0000/0.0000 | 1.0221/1.5327 | 53.6482 | 1.6264 |
| development_sampling/RS/P1 | 4 | 32/400 | 8.00/7.50 | 0/4 | 0.0000/0.0000 | 99.2500/0.0000 | 0.2214/0.5392 | 61.1333 | 0.7658 |

每输出只有100个free面，穿刺率严格<1%即0张受影响面；跨四输出累计<1%不能代替每输出零穿刺。R12 RMS单位为完整目标bbox百分比；coverage固定严格小于1e-3L。p95按Gamma长度权重，max仅采样最大值。

## 两个P0困难样例与质量代价

| 阶段/组/seed | free→R12 % | R12→free % | 双向RMS % | 全表面RMS % | proper面 | FF非法 | 边界RMS % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| offline_one_call/B/9201 | 3.3010 | 3.7004 | 3.5064 | 2.7403 | 16 | 148 | 3.5203 |
| offline_one_call/B/9204 | 3.7892 | 0.1505 | 2.6815 | 0.0915 | 15 | 108 | 0.0622 |
| offline_one_call/R/9201 | 3.3013 | 3.6933 | 3.5028 | 2.7383 | 16 | 146 | 3.5195 |
| offline_one_call/R/9204 | 3.7893 | 0.1522 | 2.6816 | 0.0942 | 3 | 98 | 0.0552 |
| offline_one_call/S/9201 | 3.3010 | 3.7004 | 3.5064 | 2.7403 | 16 | 148 | 3.5203 |
| offline_one_call/S/9204 | 3.7891 | 0.1496 | 2.6814 | 0.0902 | 15 | 105 | 0.0561 |
| offline_one_call/RS/9201 | 3.3013 | 3.6933 | 3.5028 | 2.7383 | 16 | 146 | 3.5195 |
| offline_one_call/RS/9204 | 3.7893 | 0.1512 | 2.6816 | 0.0929 | 3 | 96 | 0.0483 |
| development_sampling/B/9201 | 3.3010 | 3.7004 | 3.5064 | 2.7403 | 16 | 148 | 3.5203 |
| development_sampling/B/9204 | 3.7892 | 0.1505 | 2.6815 | 0.0915 | 15 | 108 | 0.0622 |
| development_sampling/R/9201 | 3.3219 | 3.7421 | 3.5382 | 2.7616 | 16 | 148 | 3.5329 |
| development_sampling/R/9204 | 3.7885 | 0.1516 | 2.6810 | 0.0947 | 2 | 112 | 0.0808 |
| development_sampling/S/9201 | 3.3010 | 3.7004 | 3.5064 | 2.7403 | 16 | 148 | 3.5203 |
| development_sampling/S/9204 | 3.7886 | 0.1499 | 2.6811 | 0.0912 | 11 | 113 | 0.0564 |
| development_sampling/RS/9201 | 3.3219 | 3.7421 | 3.5382 | 2.7616 | 16 | 148 | 3.5329 |
| development_sampling/RS/9204 | 3.7877 | 0.1528 | 2.6805 | 0.0912 | 3 | 110 | 0.0845 |

| 阶段/组/父 | 窄长面% | 角度p05 | 退化比例 | 总面积 | 最小面积 | 精确边界边 | 共边分量 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| offline_one_call/B/P0 | 10.5000 | 5.2631 | 0.0000 | 50.3909 | 0.0518 | 312.0000 | 101.0000 |
| offline_one_call/B/P1 | 15.7500 | 4.7634 | 0.0000 | 32.5568 | 0.0023 | 314.0000 | 101.0000 |
| offline_one_call/R/P0 | 10.5000 | 5.2631 | 0.0000 | 50.3909 | 0.0518 | 312.0000 | 101.0000 |
| offline_one_call/R/P1 | 15.7500 | 4.7634 | 0.0000 | 32.5568 | 0.0023 | 314.0000 | 101.0000 |
| offline_one_call/S/P0 | 10.5000 | 5.2617 | 0.0000 | 50.3844 | 0.0518 | 312.0000 | 101.0000 |
| offline_one_call/S/P1 | 15.7500 | 4.7634 | 0.0000 | 32.5613 | 0.0023 | 314.0000 | 101.0000 |
| offline_one_call/RS/P0 | 10.5000 | 5.2617 | 0.0000 | 50.3844 | 0.0518 | 312.0000 | 101.0000 |
| offline_one_call/RS/P1 | 15.7500 | 4.7634 | 0.0000 | 32.5616 | 0.0023 | 314.0000 | 101.0000 |
| development_sampling/B/P0 | 10.5000 | 5.2631 | 0.0000 | 50.3909 | 0.0518 | 312.0000 | 101.0000 |
| development_sampling/B/P1 | 15.7500 | 4.7634 | 0.0000 | 32.5568 | 0.0023 | 314.0000 | 101.0000 |
| development_sampling/R/P0 | 10.2500 | 5.3022 | 0.0000 | 50.2619 | 0.0515 | 312.0000 | 101.0000 |
| development_sampling/R/P1 | 15.5000 | 4.7651 | 0.0000 | 32.5463 | 0.0022 | 314.0000 | 101.0000 |
| development_sampling/S/P0 | 10.5000 | 5.2592 | 0.0000 | 50.4009 | 0.0516 | 312.0000 | 101.0000 |
| development_sampling/S/P1 | 14.7500 | 4.8182 | 0.0000 | 32.5638 | 0.0021 | 314.0000 | 101.0000 |
| development_sampling/RS/P0 | 10.5000 | 5.2979 | 0.0000 | 50.2708 | 0.0514 | 312.0000 | 101.0000 |
| development_sampling/RS/P1 | 15.2500 | 4.7851 | 0.0000 | 32.5591 | 0.0021 | 314.0000 | 101.0000 |

评分器不修补输入。外层B/R/S/RS表示实际输入是否经过推理修正；旧评分字段geometry_postprocessing=NONE仅指评分时未二次修补。未合并索引或改变三角连接；边界接近、非流形边数0、C精确均不能推出水密。

## 固定free四格绑定、晋级与确认

| 阶段 | 组 | seed | R0/C0 | R0/C1 | R1/C0 | R1/C1 | 绑定0 | 绑定1 | 双向正 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development | B | 9201 | 3.5064 | 8.1333 | 4.7665 | 0.0739 | 4.6269 | 4.6926 | PASS |
| development | B | 9202 | 0.0913 | 8.1360 | 9.4326 | 1.0222 | 8.0446 | 8.4104 | PASS |
| development | B | 9203 | 0.1642 | 8.3616 | 9.3513 | 0.9097 | 8.1974 | 8.4415 | PASS |
| development | B | 9204 | 2.6815 | 8.1996 | 8.9860 | 1.0168 | 5.5181 | 7.9692 | PASS |
| development | R | 9201 | 3.5382 | 8.1075 | 4.6960 | 0.0643 | 4.5692 | 4.6317 | PASS |
| development | R | 9202 | 0.1374 | 8.1454 | 9.5459 | 1.0370 | 8.0079 | 8.5089 | PASS |
| development | R | 9203 | 0.1540 | 8.3371 | 9.3613 | 0.9096 | 8.1832 | 8.4517 | PASS |
| development | R | 9204 | 2.6810 | 8.2065 | 8.9900 | 1.0186 | 5.5255 | 7.9713 | PASS |
| development | RS | 9201 | 3.5382 | 8.1074 | 4.6960 | 0.0650 | 4.5692 | 4.6310 | PASS |
| development | RS | 9202 | 0.1344 | 8.1327 | 9.5511 | 1.0385 | 7.9982 | 8.5126 | PASS |
| development | RS | 9203 | 0.1523 | 8.3505 | 9.3813 | 0.9411 | 8.1982 | 8.4402 | PASS |
| development | RS | 9204 | 2.6805 | 8.2058 | 8.9868 | 1.0187 | 5.5253 | 7.9682 | PASS |
| development | S | 9201 | 3.5064 | 8.1283 | 4.7665 | 0.0703 | 4.6219 | 4.6962 | PASS |
| development | S | 9202 | 0.0871 | 8.1343 | 9.4199 | 1.0232 | 8.0472 | 8.3967 | PASS |
| development | S | 9203 | 0.1575 | 8.3552 | 9.3539 | 0.9416 | 8.1977 | 8.4124 | PASS |
| development | S | 9204 | 2.6811 | 8.2249 | 8.9657 | 1.0157 | 5.5439 | 7.9499 | PASS |

交叉评分沿用同一输出的固定free数组，没有重新挑面或用GT改变修正。晋级必须两父对象穿刺面率相对B各降≥25%，剩余RMS和边界RMS各不恶化>5%，无新增退化/非有限、平均FF非法及UNRESOLVED不增加，C保持且成本有记录。这些是预登记操作性门槛，不是硬安全保证。

晋级记录：`{"status": "NO_CANDIDATE_MET_ALL_PREREGISTERED_GATES", "candidates": [{"mode": "R", "parents": [{"parent": 0, "base": {"proper_face_fraction": 0.19250000000000003, "boundary_rms": 1.0179422355668075, "R12_symmetric_rms": 1.6108610810670028, "free_free_illegal": 122.25, "free_free_unresolved": 0.0, "degenerate": 0.0}, "corrected": {"proper_face_fraction": 0.1225, "boundary_rms": 1.024858162680178, "R12_symmetric_rms": 1.6276516908916725, "free_free_illegal": 125.25, "free_free_unresolved": 0.0, "degenerate": 0.0}, "gates": {"proper_reduction_at_least_25pct": true, "R12_worsening_at_most_5pct": true, "boundary_worsening_at_most_5pct": true, "no_systematic_FF_transfer": false, "no_increased_FF_unresolved": true, "no_added_degenerate": true, "C_all_exact": true}, "pass_gate": false}, {"parent": 1, "base": {"proper_face_fraction": 0.1375, "boundary_rms": 0.22975372136961505, "R12_symmetric_rms": 0.7556630216336697, "free_free_illegal": 97.0, "free_free_unresolved": 0.0, "degenerate": 0.0}, "corrected": {"proper_face_fraction": 0.0875, "boundary_rms": 0.22045630583765435, "R12_symmetric_rms": 0.7573828273078538, "free_free_illegal": 96.5, "free_free_unresolved": 0.0, "degenerate": 0.0}, "gates": {"proper_reduction_at_least_25pct": true, "R12_worsening_at_most_5pct": true, "boundary_worsening_at_most_5pct": true, "no_systematic_FF_transfer": true, "no_increased_FF_unresolved": true, "no_added_degenerate": true, "C_all_exact": true}, "pass_gate": true}], "pass_gate": false, "selection_key": [0.105, 1.192517259099763, 361.872157614001]}, {"mode": "S", "parents": [{"parent": 0, "base": {"proper_face_fraction": 0.19250000000000003, "boundary_rms": 1.0179422355668075, "R12_symmetric_rms": 1.6108610810670028, "free_free_illegal": 122.25, "free_free_unresolved": 0.0, "degenerate": 0.0}, "corrected": {"proper_face_fraction": 0.17500000000000002, "boundary_rms": 1.0082318440149938, "R12_symmetric_rms": 1.608007888541667, "free_free_illegal": 125.0, "free_free_unresolved": 0.0, "degenerate": 0.0}, "gates": {"proper_reduction_at_least_25pct": false, "R12_worsening_at_most_5pct": true, "boundary_worsening_at_most_5pct": true, "no_systematic_FF_transfer": false, "no_increased_FF_unresolved": true, "no_added_degenerate": true, "C_all_exact": true}, "pass_gate": false}, {"parent": 1, "base": {"proper_face_fraction": 0.1375, "boundary_rms": 0.22975372136961505, "R12_symmetric_rms": 0.7556630216336697, "free_free_illegal": 97.0, "free_free_unresolved": 0.0, "degenerate": 0.0}, "corrected": {"proper_face_fraction": 0.1325, "boundary_rms": 0.21716746411213714, "R12_symmetric_rms": 0.7627130211780768, "free_free_illegal": 98.25, "free_free_unresolved": 0.0, "degenerate": 0.0}, "gates": {"proper_reduction_at_least_25pct": false, "R12_worsening_at_most_5pct": true, "boundary_worsening_at_most_5pct": true, "no_systematic_FF_transfer": false, "no_increased_FF_unresolved": true, "no_added_degenerate": true, "C_all_exact": true}, "pass_gate": false}], "pass_gate": false, "selection_key": [0.15375, 1.185360454859872, 50.022316824000086]}, {"mode": "RS", "parents": [{"parent": 0, "base": {"proper_face_fraction": 0.19250000000000003, "boundary_rms": 1.0179422355668075, "R12_symmetric_rms": 1.6108610810670028, "free_free_illegal": 122.25, "free_free_unresolved": 0.0, "degenerate": 0.0}, "corrected": {"proper_face_fraction": 0.13, "boundary_rms": 1.0221148924947019, "R12_symmetric_rms": 1.6263554392935782, "free_free_illegal": 121.0, "free_free_unresolved": 0.0, "degenerate": 0.0}, "gates": {"proper_reduction_at_least_25pct": true, "R12_worsening_at_most_5pct": true, "boundary_worsening_at_most_5pct": true, "no_systematic_FF_transfer": true, "no_increased_FF_unresolved": true, "no_added_degenerate": true, "C_all_exact": true}, "pass_gate": true}, {"parent": 1, "base": {"proper_face_fraction": 0.1375, "boundary_rms": 0.22975372136961505, "R12_symmetric_rms": 0.7556630216336697, "free_free_illegal": 97.0, "free_free_unresolved": 0.0, "degenerate": 0.0}, "corrected": {"proper_face_fraction": 0.08, "boundary_rms": 0.22135127078904582, "R12_symmetric_rms": 0.7658220747698337, "free_free_illegal": 99.25, "free_free_unresolved": 0.0, "degenerate": 0.0}, "gates": {"proper_reduction_at_least_25pct": true, "R12_worsening_at_most_5pct": true, "boundary_worsening_at_most_5pct": true, "no_systematic_FF_transfer": false, "no_increased_FF_unresolved": true, "no_added_degenerate": true, "C_all_exact": true}, "pass_gate": false}], "pass_gate": false, "selection_key": [0.10500000000000001, 1.1960887570317058, 426.8716238839992]}], "selected": null, "selected_before_confirmation": true, "evaluated_seed_scope": [9201, 9202, 9203, 9204], "not_generalization": "Both parents were training parents; new seeds only test noise robustness"}`。

## 成本、位移与实际执行账本

| 阶段/组/父 | 修正秒均值 | 接受均值 | 拒绝均值 | 位移RMS | 最大位移 |
| --- | --- | --- | --- | --- | --- |
| offline_one_call/R/P0 | 0.7698 | 15.7500 | 47.5000 | 0.0012 | 0.0073 |
| offline_one_call/R/P1 | 0.5979 | 16.7500 | 47.2500 | 0.0010 | 0.0061 |
| offline_one_call/S/P0 | 0.6297 | 30.2500 | 17.7500 | 0.0008 | 0.0062 |
| offline_one_call/S/P1 | 0.4789 | 29.5000 | 23.7500 | 0.0011 | 0.0061 |
| offline_one_call/RS/P0 | 1.0435 | 46.0000 | 65.2500 | 0.0016 | 0.0073 |
| offline_one_call/RS/P1 | 0.7940 | 45.0000 | 75.2500 | 0.0014 | 0.0061 |
| development_sampling/R/P0 | 43.8069 | 654.7500 | 1582.0000 | 0.0014 | 0.0073 |
| development_sampling/R/P1 | 46.6612 | 655.2500 | 1584.0000 | 0.0012 | 0.0061 |
| development_sampling/S/P0 | 6.1786 | 75.5000 | 61.7500 | 0.0002 | 0.0073 |
| development_sampling/S/P1 | 6.3270 | 81.7500 | 78.5000 | 0.0003 | 0.0061 |
| development_sampling/RS/P0 | 51.9775 | 719.2500 | 1651.0000 | 0.0014 | 0.0073 |
| development_sampling/RS/P1 | 54.7404 | 721.7500 | 1689.7500 | 0.0012 | 0.0061 |

离线位移为一次调用的真实净位移；采样行的RMS为50步净修正RMS的均方根，max为任一步最大角位移，不能当作50步终点累计位移。累计接受路径长度另存JSON。R+S共享同一次预测Xhat的rho=1e-3L，不能分别多用一次预算。eta=1e-4L、delta=.005L、beta=.25、每模块64候选检查与固定回溯1/.5/.25/.125不变。有限轴搜索不保证全局最小平移；部分移动和拒绝不等于已解决。

| 模型及运行计数 | 实际 |
| --- | --- |
| model_forward_attempts | 1300 |
| model_forward_returns | 1300 |
| model_backward_calls | 0 |
| optimizer_updates | 0 |
| model_parameter_updates | 0 |
| complete_generation_attempts | 26 |
| complete_generations | 26 |
| baseline_replays | 2 |
| development_guided_generations | 24 |
| confirmation_generations | 0 |
| euler_updates | 1300 |
| archived_baseline_trajectories_reused | 8 |
| offline_corrector_calls | 24 |
| trajectory_corrector_calls | 640 |
| model_cuda_seconds | 67.3260 |
| geometry_cpu_seconds | 856.0208 |
| transfer_seconds | 0.9998 |
| geometry_observation_cpu_seconds | 136.1911 |
| peak_cuda_allocated | 574452224 |
| trajectory_corrector_returns | 640 |
| trajectory_R_module_calls | 560 |
| trajectory_S_module_calls | 160 |

| 独立账本域 | primary测试 | 独立测试 | 双检测验收 | 几何求导 | AABB粗检 | FP32候选 | 接受 | 拒绝 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| formal_probe_audit_endpoint_score | 564520 | 415420 | 415388 | 107 | 0 | 0 | 0 | 0 |
| formal_corrector_fixtures_and_offline | 28616 | 28616 | 28616 | 0 | 309473 | 1852 | 742 | 1110 |
| trajectory_corrector | 1316745 | 1316745 | 1316745 | 0 | 6233922 | 38221 | 11633 | 26588 |
| trajectory_observation | 212114 | 212114 | 212114 | 0 | 823200 | 0 | 0 | 0 |

正式合计：`{"primary_pair_tests": 2121995, "independent_pair_tests": 1972895, "verified_pair_tests": 1972863, "geometry_autograd_calls": 107, "AABB_pair_checks": 7366595, "candidates": 40073, "accepted": 12375, "rejected": 27698}`；额外实现期CPU检查另列：`{"primary_pair_tests": 71, "independent_pair_tests": 71, "geometry_autograd_calls": 12, "candidates": 12, "accepted": 9, "rejected": 3}`。双检测验收是一次primary加一次独立检测，不将它作为第三次判交再相加；AABB粗检也不等于完整triangle判交。

| 采样阶段/组/父/seed | 模型CUDA秒 | 修正CPU秒 | 观察判交CPU秒 | 搬运/积分同步秒 | 墙钟秒 | 峰值CUDA GiB | R/S模块调用 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| development/R/P0/9201 | 3.0753 | 35.9914 | 2.9524 | 0.0499 | 42.2614 | 0.5350 | 35/0 |
| development/R/P0/9202 | 2.9740 | 45.6973 | 3.3577 | 0.0448 | 52.2609 | 0.5350 | 35/0 |
| development/R/P0/9203 | 3.1027 | 44.8305 | 3.3243 | 0.0539 | 51.4896 | 0.5350 | 35/0 |
| development/R/P0/9204 | 2.7879 | 48.7082 | 3.3867 | 0.0391 | 55.0998 | 0.5350 | 35/0 |
| development/R/P1/9201 | 2.8259 | 42.6924 | 3.5010 | 0.0467 | 49.2430 | 0.5350 | 35/0 |
| development/R/P1/9202 | 3.1283 | 45.0207 | 3.0040 | 0.0470 | 51.4038 | 0.5350 | 35/0 |
| development/R/P1/9203 | 3.1624 | 46.9001 | 3.4553 | 0.0402 | 53.7527 | 0.5350 | 35/0 |
| development/R/P1/9204 | 2.6732 | 52.0316 | 3.2433 | 0.0399 | 58.1626 | 0.5350 | 35/0 |
| development/S/P0/9201 | 2.0254 | 4.7385 | 6.6343 | 0.0202 | 13.6114 | 0.5350 | 0/10 |
| development/S/P0/9202 | 2.4012 | 6.7728 | 8.6685 | 0.0196 | 18.1000 | 0.5350 | 0/10 |
| development/S/P0/9203 | 2.4395 | 6.4767 | 8.4429 | 0.0259 | 17.5986 | 0.5350 | 0/10 |
| development/S/P0/9204 | 2.4737 | 6.7262 | 8.7200 | 0.0308 | 18.1963 | 0.5350 | 0/10 |
| development/S/P1/9201 | 2.1916 | 6.1976 | 8.4156 | 0.0227 | 17.0456 | 0.5350 | 0/10 |
| development/S/P1/9202 | 2.5301 | 5.8774 | 7.5566 | 0.0281 | 16.2142 | 0.5350 | 0/10 |
| development/S/P1/9203 | 2.3309 | 6.6297 | 8.1023 | 0.0249 | 17.3177 | 0.5350 | 0/10 |
| development/S/P1/9204 | 2.4017 | 6.6033 | 7.9773 | 0.0175 | 17.2264 | 0.5350 | 0/10 |
| development/RS/P0/9201 | 2.5826 | 42.9188 | 3.6059 | 0.0469 | 49.3753 | 0.5350 | 35/10 |
| development/RS/P0/9202 | 2.4836 | 54.5151 | 3.8427 | 0.0561 | 61.1205 | 0.5350 | 35/10 |
| development/RS/P0/9203 | 2.4681 | 53.4676 | 4.0133 | 0.0647 | 60.2342 | 0.5350 | 35/10 |
| development/RS/P0/9204 | 2.4709 | 57.0083 | 4.2127 | 0.0662 | 63.9643 | 0.5350 | 35/10 |
| development/RS/P1/9201 | 2.6272 | 52.0946 | 4.2037 | 0.0441 | 59.1767 | 0.5350 | 35/10 |
| development/RS/P1/9202 | 2.1721 | 52.0045 | 3.5142 | 0.0444 | 57.9530 | 0.5350 | 35/10 |
| development/RS/P1/9203 | 2.3298 | 54.9813 | 3.9876 | 0.0525 | 61.5590 | 0.5350 | 35/10 |
| development/RS/P1/9204 | 2.3582 | 59.8814 | 3.6642 | 0.0592 | 66.1820 | 0.5350 | 35/10 |

| 阶段/组/父/seed | 接受/拒绝 | 修正新增CF/FF非法pair | 主干新增CF非法pair |
| --- | --- | --- | --- |
| development/R/P0/9201 | 743/1497 | 0/0 | 86 |
| development/R/P0/9202 | 621/1619 | 0/0 | 148 |
| development/R/P0/9203 | 544/1696 | 0/0 | 104 |
| development/R/P0/9204 | 711/1516 | 0/0 | 119 |
| development/R/P1/9201 | 605/1632 | 0/0 | 102 |
| development/R/P1/9202 | 648/1592 | 0/0 | 180 |
| development/R/P1/9203 | 775/1465 | 0/0 | 108 |
| development/R/P1/9204 | 593/1647 | 0/0 | 149 |
| development/S/P0/9201 | 0/4 | 0/0 | 58 |
| development/S/P0/9202 | 103/75 | 0/0 | 111 |
| development/S/P0/9203 | 94/93 | 0/0 | 93 |
| development/S/P0/9204 | 105/75 | 0/0 | 108 |
| development/S/P1/9201 | 85/58 | 0/0 | 88 |
| development/S/P1/9202 | 72/69 | 0/0 | 145 |
| development/S/P1/9203 | 104/85 | 0/0 | 106 |
| development/S/P1/9204 | 66/102 | 0/0 | 129 |
| development/RS/P0/9201 | 743/1497 | 0/0 | 86 |
| development/RS/P0/9202 | 706/1702 | 0/0 | 153 |
| development/RS/P0/9203 | 653/1775 | 0/0 | 109 |
| development/RS/P0/9204 | 775/1630 | 0/0 | 119 |
| development/RS/P1/9201 | 678/1731 | 0/0 | 101 |
| development/RS/P1/9202 | 709/1677 | 0/0 | 178 |
| development/RS/P1/9203 | 857/1573 | 0/0 | 106 |
| development/RS/P1/9204 | 643/1778 | 0/0 | 147 |

上表按步记录pair创建事件；相同pair跨时刻可能重复出现。几何模块未新增不代表主干下一步也未新增，不把整个生成过程描述为穿刺单调下降。几何修正、额外观察判交、模型CUDA以及搬运/积分同步分别计时；CPU耗时不能冒充模型速度，RS包含两个模块的实际成本。模型CUDA时间由CUDA event记录，几何修正/观察以及搬运字段为墙钟时间；模型CPU派发时间未单独profile，不从墙钟残差推算。几何GPU工作为0（全部CPU）。峰值是Torch记录的CUDA已分配内存，不是整个进程或驱动的全量峰值；进程CPU profiling未执行。阶段时间记录：{'cpu_stages_seconds': 98.75483046500001, 'sampling_seconds': 1236.869115436}。离线派生、实际轨迹内修正、评分复用与额外核验分别保留于JSON。失败/重试不能从调用账本删除。报告自身模型、几何检测/求导和优化器调用均为0。

## 固定样本图与停止

![offline_one_call](fixed_P0_P1_seed9201_offline_one_call.png)

图SHA256：`049889d1de6c6f6a27511968209c0c72e6cd9aa4152cb7f33d418d1f41697ff7`。固定两父对象及9201，所有已执行组保持同视角/完整范围；实际CF非法free面按已存检测高亮。蓝色C，红色CF proper穿刺，紫色CF共面正面积重叠，橙色线框为FF非法对涉及的自由面。未执行组保留NOT_RUN格。

![development_sampling](fixed_P0_P1_seed9201_development_sampling.png)

图SHA256：`0fe50f5445512db88fba12d32b4a7bde4d738bb18e05ee92225aa347ce5a0061`。固定两父对象及9201，所有已执行组保持同视角/完整范围；实际CF非法free面按已存检测高亮。蓝色C，红色CF proper穿刺，紫色CF共面正面积重叠，橙色线框为FF非法对涉及的自由面。未执行组保留NOT_RUN格。

原始/修正NPZ与逐输出检测证据见[唯一结果JSON](results.json)；[本轮代码diff](code.diff)。B引用既有归档，不复制历史权重或数据。
到预定门禁停止条件或固定预算结束即停止。本轮没有宏观吸引、删面、重网格、索引焊接、优化器更新或模型反向；不自动扩参数、加训练或启动Phase3。
