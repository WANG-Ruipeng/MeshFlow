# 来源 M=C+Y 的资产与评分合同（只读核查）

状态：READ_ONLY_CONTRACT_REVIEW_COMPLETE。主范围为194训练条件＋64 NEW32条件，合计258条件、99独立父（70个 official_raw_indexed、29个 existing_repair_v2）。旧6不属于此主范围。本核查没有运行接口评分、几何面对检测、D0/GEO、模型、训练、生成或新图审，也未修改MINI/旧资产。

## 1. 唯一身份入口及坐标构造

- 训练：E:/MeshFlow/experiments/r2_user_data_scaling_32_67_v1/data/train_manifest.json；parents[uid]保存父NPZ/hash，tasks[condition_id]保存194条件及source_face_ids/free_source_ids。
- DEV：同目录panel_manifest.json；dev_parents为32父完整元数据，conditions为64条件。
- 这些清单指向上一轮只读NPZ：E:/MeshFlow/experiments/r2_user_data_scaling_12h_v1/data/{parents,conditions}/。
- 每条件元数据同时带quality_path/quality_sha256，指向同一父、同一版本、同一模型坐标的完整既有面对记录。
- C条件NPZ **没有Y**。必须从其登记父的full_target按free_source_ids取补集；不得拿另一个父版本的Y、重新OT、宽度增强或重新规范化。

父NPZ的权威数组：

| 数组 | 实际意义 |
|---|---|
| full_target，FP32[N,9] | 当前训练的canonical模型坐标；reshape(N,3,3) |
| model_vertices，FP32[V,3] | 已在相同模型坐标的原索引顶点 |
| source_face_vertex_ids，int64[N,3] | canonical角序下的原顶点ID |
| source_faces，int64[N,3] | 保留面顺序、来源原角序下的原顶点ID |
| source_corner_permutations，int64[N,3] | P[j]选择来源角：canonical[j]=original[P[j]] |
| retained_to_original_face | 当前保留M行到原始面ID；仅用于追溯，不是当前数组下标 |
| pre_ot_full | 仅辅助合同：pre_ot_full*FP32(2)逐位等于full_target |
| source_vertices / C_original_raw_coordinates | 原始坐标；**不能**当模型坐标参与本次距离评分 |

可用的只读数组组织（不是新增网格修复）：

~~~python
idsC = condition["source_face_ids"]
idsY = condition["free_source_ids"]
order = np.concatenate((idsC, idsY))      # C前K，Y在后
canonical_M = parent["full_target"].reshape(-1, 3, 3)
V = parent["model_vertices"]             # 原索引保持
F = parent["source_faces"][order]         # 来源角序，非canonical绕序
source_M_Cfirst = V[F]
C = source_M_Cfirst[:K]
Y = source_M_Cfirst[K:]
~~~

每条正式测量前做身份/数组合同检查即可，不需要重新几何审计：
1. 登记NPZ及质量JSON的hash、uid/variant/N吻合；C/Y ID不重、不交且并为0..N-1。
2. model_vertices[source_face_vertex_ids]逐位等于canonical_M。
3. canonical_M[idsC]逐位等于condition.C。
4. V[source_faces[idsC]]逐位等于condition.C_original；顶点ID逐位等于C_original_source_vertex_ids。
5. 对整M用argsort(P)恢复canonical角序，逐位等于V[source_faces]；C的P须等于条件已保存P。
6. condition.C_source_vertex_ids与canonical C配对；C_original_source_vertex_ids与来源C配对，不能交叉使用。

不要用original_source_faces以“当前保留面ID”直接索引；existing_repair_v2可能保留更大的原始数组。优先使用source_faces与显式retained_to_original_face。也不要将旧版canonical_to_source_face等名称当成无需检查的当前面重排合同。

本次仅抽查三条代表资产的上述坐标/角序/ID关系，全部逐位PASS：
- existing_repair_v2训练：103b75dfd146976563ed57e35c972b4b__BACK_FULL。
- official_raw_indexed训练：1997d5827caf81df67d783b4714d4324__BACK_HALF。
- official_raw_indexed DEV：69735def3b0f009eb098f5649ee97d24__SEAT_FULL。
这是构造方式核验，**不是258条已完成接口测量**。

## 2. 原索引、坐标合并、方向必须分层

主记录采用上述来源原索引V/F：同坐标但不同source vertex ID仍是不同顶点。它直接回答来源答案的原连接。

坐标合并可作为独立诊断。若要与旧RAW的精确索引逻辑一致，使用FP32角坐标的uint32三元组np.unique；这不是带容差焊接，也不是GEO_C。必须标EXACT_UINT32_COORDINATE_INDEX（包括+0/-0位模式区别），保留来源ID到坐标索引映射及合并量。不同ID合并可能增加连接、重复面、非流形度或合法边拒绝数，不能覆盖原索引结果。

生成POST图指标在它实际的POST vertices/faces上计算。来源原索引和来源坐标合并是两个明确域；不能把某个域称为“执行过官方POST”，本轮没有D0/GEO。

方向主记录使用来源C与来源Y的原角序，避免再次锁死canonical的任意绕序。若注册“固定来源C、重新定向Y”诊断，应另存原方向和定向后结果及solver状态；不能静默改写来源方向或把方向修正当成新增连接。可用正式CPU函数meshflow_control.winding_solver.orient_fixed_C(vertices,faces,K)，但本核查未调用，是否测该支路由本轮协议锁定。

## 3. 可复用既有面对分类，不能复用面对数充当接口数

每父quality_path的schema为r2_user_scaling_source_quality_v1。它已对同一full_target执行完整independent self-pair扫描，保留非disjoint的pair_contacts及classification、原始面ID。复用前检查status=COMPLETE、N、full_target_sha256、input_npz_sha256、L和detector_source_hashes；不需要重扫完整M。

本次核对源质量冻结检测器与32/67生成评分实际legacy实现，三个源码hash完全相同：
- pair_audit.py：2e1e1b5520d808d6f11e3660373118d309dfadf18dd2e3cf90f15a7856ae3d8d
- intersections.py：23a2e8c8deefcceb51ace60ad8d8a58d352d7c38bdcba2e8f85d6ceabd447d86
- seams.py：9ce6927710d80eedec19aa10e2100ade582cf00e6942e9de4ef987618a005e4b

源缓存位置来自r2/data_tools/user_conditions_v1/quality.py::frozen_detector（84–99）；生成侧由E:/MeshFlow-Control/src/meshflow_control/evaluation/geometry.py::scan_geometry（7–23）调用同hash的pair_scan。统一L=sqrt(3)*2/.3762=9.208138264587333；不再按每父包围盒缩放。

映射规则：
- 原pair=(source_i,source_j)，先按idsC/idsY分区，保留完整classification。
- CF contacts的pair必须是[known_local,free_local]；若原pair第一面是Y，先交换角色再映射。
- FF contacts的pair必须是两个Y-local下标（排序），**不是Cfirst全网格下标，也不是原M行ID**。
- CC不传给当前actual_valid_graph，但应保留资格/方向来源记录。
- 缺失shared-edge facing pair时评分器默认UNKNOWN_MISSING_SHARED_EDGE_PAIR并拒绝，不能把未保存条目补成合法。
- 缓存记录继承的是已保存的canonical无序几何分类；绕序变化不改变点集。应记录REUSED_SOURCE_PAIR_CLASSIFICATION，不冒称已经在恢复后角序重新运行过检测器。

坐标合并不移动三角点时，同一几何面对缓存仍可复用；但新索引下的重复面、边incidence和连通图必须重新按该索引计算。shared_edge_contact只是检测器面对类别；它本身不等于唯一共边或合法接口。

## 4. 与已生成POST同一数学定义的API

本次逐字节核对下列当前小源码与完整审阅ZIP内实际快照一致；宜将冻结快照及依赖常量复制到本轮独立评分目录。不要调用旧score_one/run/aggregate或denoise入口，它们带旧OUT/期限/写入逻辑。

| 功能 | 实际函数 |
|---|---|
| 合法索引共边与无方向可达 | r2/pilots/anchor_aware_denoiser_v1/evaluation.py::actual_valid_graph(vertices,faces,K,pairs)，251–319 |
| 方向一致可达 | r2/tools/c_restore_coupling_v1/evaluation.py::opposite_reach(tri,K,edges)，186–199 |
| 来源C开边界 | r2/tools/avmr_pmr_v1/run.py::topology(C,source_ids)，102–134 |
| C边界点到Y顶点最近邻 | 同文件nearest(points,free)，137–145 |
| A-VMR/PMR | 同文件metric(distances,threshold)，148–155；SETTINGS保留全部8阈值 |
| 来源映射检查 | r2/src/meshflow_control/source_winding.py::validate_source(source,C_canonical)，174–237 |

actual_valid_graph要求唯一索引边恰好两个incident faces、无零长/零面积、两侧不属于索引重复面组，且面对classification在LEGAL集合。每条边只计一次；从全部C面为根沿合格双面边BFS，free面积分母为Y全表面积。它允许合法但同向边进入无方向图；opposite_reach额外要求winding_opposite。局部连通可达不认证整个来源物体无交/水密。PURE-C边不构成Y接续。

A-VMR的来源C边界按原source IDs统计count==1的边；同坐标异ID不能先合并。nearest比较的是Y所有角点的**顶点距离**，不是点到有限三角表面；阈值使用严格distance<threshold，PMR为全部边界顶点匹配，空边界返回NA。共有raw 1e-6模型单位和L比例1e-6、1e-4、.001、.002、.005、.01、.02八档。应保存分子、分母和逐边界点最近Y face/corner身份，以免后续浮点均值残差被当成改善。

这些来源接口是“参考完成是否示范接续”的诊断，不是推理用户强制指定接口，也不是要求生成恢复唯一来源椅型。

## 5. 科学解释与复用边界

- 不重置C，不删除重复面、不修源网格、不把UNKNOWN变合法；图定义自身对重复/非流形边的拒绝要保留。
- source_quality='ILLEGAL'先于UNKNOWN，是FF条件标签，不代表该条件没有UNKNOWN事件。COMMON32的723问题对含721确认非法＋2UNKNOWN；EXP_ONLY35的843含840＋3。必须分开事件类型，不能把两个总数称为确认非法。
- 把共享边数、合法边数/长度、方向可达与A-VMR/PMR分别保存；它们不能互相替代。
- 可在99父只读缓存上按258条件重映射，既有全M分类无需按条件反复执行全对几何扫描；实际新增必要计算仅来源索引图、面积/长度、方向可达和边界顶点最近邻。任何额外支路先由root登记，不从本说明自动扩任务。
