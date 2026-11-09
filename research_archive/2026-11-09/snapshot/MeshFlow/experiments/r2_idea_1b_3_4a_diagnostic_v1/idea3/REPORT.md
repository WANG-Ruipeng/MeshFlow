# Idea 3：共享 C 原边槽位的真实监督可行性诊断

状态：COMPLETE，纯 CPU；没有训练、模型前向、生成、新噪声、生产 OT 或来源几何重评分。47条既有训练条件＋6条固定 FULL 面板条件，共50条去重来源条件；面板3条TRAIN重复身份单独展示，3条DEV只诊断，不进入训练。

## 核心结果

训练池的 400 条原 C 开放边中，366 条有原评分判据支持的合法 C/Y 共边（91.50%）。47/47条件有真实接口，全部端点原ID、C来源绕序及唯一第三顶点标签可复核；不存在“只有合成玩具能造标签”的问题。

| 训练标签性质 | 计数 |
|---|---:|
| 原 C 开放边 / 支持边 / 无合法来源Y邻面 | 400 / 366 / 34 |
| 不同接口 Y 三角面 | 360 |
| Y 面接0/1/2/3条合法C边 | 8695 / 354 / 6 / 0 |
| 多边同面涉及条件 | 4 / 47 |
| 每边独立生成一张面会重复的实例 | 6 |
| 单边面占接口面 / 占合法边 | 98.33% / 96.72% |
| 第三顶点本身已经是C原顶点的边标签 | 18 |
| 来源绕序相反的合法边 | 366 / 366 |
| 仍有来源Y内部非法面对的条件 | 31 / 47 |

可见真实标签足以支持一个有边界的**表征合同验证**；不能直接推出未来训练收益。保留所有47条既定条件，没有根据新统计删父，31条已有Y风险也不改成“全网格干净”。开放边全部只是描述域，未升级成用户必须封闭的接口。

## 单边槽位必须补齐的离散合同

1. 使用原 C 开放边ID和两个原顶点ID；令边端点在各时刻直接引用C，端点不进入独立自由角点积分。仅预测轮廓弧长位置不能保证源边整段索引共享：可能形成T-junction。
2. 每个来源Y面最多输出一次。本诊断保存按字典序最小合法边确定的FACE_OWNER；其余对应边是ALIAS_TO_OWNER_NO_SECOND_FACE，不是再实例化一张重复面。这个规则只是可复算的标签分配示例，不是已确定的最佳学习方案。
3. Null用于该来源没有合法邻面的边；多边同面的别名不能混成“此边没有接续”。若第三顶点也是另一个C顶点，连续预测其坐标不能保证其ID一致，必须另有C顶点引用/面组机制或明确只承诺一条锚边。
4. 单槽最多一个占用面；不能把多个slots绑定同一原边。固定反向端点顺序能保证该边的局部绕序，仍不能保证第三点不退化、不向C内部翻折、不穿过其他面，或与其余free组成主体。
5. 带Null的槽位也改变面数合同：保留固定N时，需要将未占用的边槽返回bulk或显式分配计数；不能把退化三角形当Null导出。多边owner＋bulk复原来源集合时，本次全部50条件面数守恒。

## 六父条件（来源标签，不是生成成绩）

| 父 | split | K/N | C开放边 | 合法原边 | 接口Y面 | 多边Y面 | Null |
|---|---|---:|---:|---:|---:|---:|---:|
| 73b369 | TRAIN | 20/231 | 22 | 22 | 22 | 0 | 0 |
| 3f7417 | TRAIN | 14/168 | 16 | 13 | 13 | 0 | 3 |
| 85b169 | TRAIN | 12/166 | 10 | 10 | 10 | 0 | 0 |
| 2d08a6 | DEV | 14/238 | 16 | 16 | 13 | 3 | 0 |
| 610b16 | DEV | 2/228 | 4 | 4 | 4 | 0 | 0 |
| 608b46 | DEV | 2/140 | 4 | 3 | 3 | 0 | 1 |

DEV三条件有23条来源合法边；这是标签可用性的既有来源诊断，不是DEV输入的真实用户必接答案。父对象来源几何不是开放生成的唯一GT。

## 原OT为什么不能直接照搬

原流程：free面先置换、各面循环角序→free-only nested OT（面Hungarian与全部6种角排列）→目标转入噪声槽→HYBRID→全C/free槽再次置换。原三维自由槽没有固定边角色。OT的奇排列还可改变内部绕序，导出必须使用原C与来源映射。

本次读取原第一批8条实际缓存，只重放离散置换、不抽噪声、不重新求OT。8/8的端点标记及原顶点ID都能往返恢复；若丢掉source_face_ids/corner_permutations，旧位置mask会错。3×3固定成本例中，无约束最小分配交换两条C边角色；受限分配才保留角色。**仅把mask搬到原OT结果，不足以维持固定边槽表征。** 未来必须冻结按原边角色允许的face匹配、端点角序及Null/bulk的分配合同。

## 原合法边判据的正反控

以下9个显式小网格调用原verified_pair与原actual_valid_graph；没有改变检测器阈值，没有扫描生产输出。独立探针不能替代训练或全局品质证据。

| 控制 | 合法C/F边 | 反向合法边 | 意义 |
|---|---:|---:|---|
| legal_strip | 1 | 1 | Shared original endpoints, consistent direction and a second free face; still no whole-object guarantee. |
| interior_arc_T_junction | 0 | 0 | Both endpoints lie on the original boundary segment, but C lacks their split vertices: no indexed whole-edge sharing. |
| coordinate_copy_without_ID_share | 0 | 0 | Byte-equal endpoint positions using new IDs do not share the C edge. |
| two_slots_same_edge | 0 | 0 | Two independent occupied slots plus C produce three-incidence nonmanifold edge. |
| degenerate_third_point | 0 | 0 | Third point on the anchor line creates zero area, rejected despite copied IDs. |
| isolated_tiny_flap | 1 | 1 | One legal tiny attached flap coexists with a detached large free component; endpoint sharing does not ensure attachment to body. |
| same_winding_shared_edge | 1 | 0 | Exact original edge and legal pair do not enforce opposite winding unless slot corner order is fixed. |
| duplicate_C_face | 0 | 0 | Copying all C vertices into an existing C triangle is duplicate/overlap, not useful attachment. |
| inward_coplanar_overlap | 0 | 0 | Third point goes into C interior: correct anchored edge and order still allow positive-area overlap. |

小翻片可以通过局部合法边而没有接入远处主体；重复C、重叠、退化和三入射边会被拒绝。硬共享的确定性质是指定原边的索引复用，不是“100%生成合理、流形、水密、完整椅子”。

## 与R10的区别、是否值得训练

R10在teacher状态配对正确率100%，但自由生成中伙伴可靠性、锚冲突与大位移让闭合均0/40。本设计的实质差异是**从生成状态初始化就共享明确原边的两个端点ID**，不在末端猜伙伴再搬动现有角点；因此没有被R10等价覆盖。但Null占用、第三点、与bulk的连接仍要学习，不能拿本次源标签覆盖宣称rollout已经解决。

本次支持将Idea3列为可实施候选，先把离散所有权与受限匹配合同作为实现的一部分；不支持“8–16个任意弧长槽一定接好”或直接把原无约束OT用于硬锚槽。以后真实训练的通过条件应在同请求同时成立：RAW采用、较长合法原边连接和整体主体，并披露CF/FF与退化代价；仅合成控通过/teacher复原/合法小翻片增加不够。本轮不启动该训练。

文件：per_condition.csv、per_C_open_edge.csv、per_source_interface.csv、per_source_Y_face.csv、ot_transport_examples.csv、constrained_matching_toy.json、controls.csv、control_pairs.json、control_arrays.npz、source_hashes.json、registration.json、completion.json。run.py --stage all可核验完成状态，已完成不会重复运行；--stage status只查状态。原资产只读，所有新文件限idea3目录。

## 计数兼容修正与独立复核

来源pair记录的category使用proper、coplanar_positive_overlap、UNKNOWN等名称；初版汇总误按illegal/unknown名称取值，造成FF风险显示0。已改为原LEGAL/ILLEGAL classification集合统计：TRAIN47有1138条FF非法面对，涉及31父；另3条FF未决，涉及3父，未当成无风险。修正前CSV/summary/报告/completion保存在before_category_fix/。

合法原边从始至终由原actual_valid_graph读取classification判断，不使用错误category计数；此次独立核对全部389条去重来源接口，均为原saved legal shared_edge_contact且实际仅一C/一Y双面入射，与TRAIN旧proof/mask完全一致。无需也没有重跑任何来源triangle检测。另保存每条接口Y面与其他面的既有非法/未决关联，局部锚边合法不代表该Y面全局无冲突。

注册前写路径变量遮蔽被输出目录保护拦截，原文件未写；初次CPU执行在第一合成控后的status重复字段序列化失败，保留failed.json及原源码。工程修正后完成全部9控。实际合成verified_pair尝试共32次（失败前5＋完成27），其中退化/重复控的未决由原图判据拒绝；生产检测0，模型调用0。原50来源图统计因恢复执行两遍，共100次图组装，非100份数据。
