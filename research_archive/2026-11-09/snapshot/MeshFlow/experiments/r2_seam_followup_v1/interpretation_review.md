# SEAM 后续四个 Idea：独立解释核对

2026-10-17。本文件核对附件 `3701b7dc-17e4-4786-848f-526b132f75fc/Pasted text.txt`、已封存 ABC 报告、实际实现及研究路线记忆。核对阶段零模型加载、零前向、零训练、零生成；原结果不改写。额外计算仅为已有 47 TRAIN + 3 DEV 小型来源数组的拓扑候选/匹配计数，不是新的几何评分或完整数据审计。

**建议实际执行附件末尾的两项冻结检查。它们能分离“后处理能否接上已有 seam”和“B 的推理占用布局影响有多大”，但附件把多个尚未验证的解释写成了定论，不能照搬其因果叙事。** Idea 3 值得继续设计，当前一槽一伴生面的合同不能覆盖全部来源拓扑；Idea 4 与 R07 同源副本一致性高度重合，不应称为全新且已知能降低相交的机制。

## 1. 对两条“破案线索”的修正

### C 的训练/推理不一致是真实问题，具体裂口因果尚未定位

实际代码在训练时把来源同顶点的 bulk 角点设为 tip 别名，复制其路径和速度，并从独立自由标量 FM 分母排除；推理 28 请求全部 Null，没有执行引用。因此训练与推理约束分布确实不同。[shared_tip.py](../../r2/pilots/seam_abc_v1/shared_tip.py)；[ABC 报告第 5 节](../r2_seam_abc_v1/REPORT.md)。

但“这批角点从没学过独立运动，故坍缩删掉，正好造成全部 FF 下降”并未证实。模型参数是共享的，还继承了原有生成器，来源角点经 OT 和排列进入不同槽位；它们不是各自拥有一个从未受训的专用速度网络。训练掩码变化、相关起点、上下文输入及独立变量分母同时变化。需要逐面身份与反事实配对才能把特定删面或裂口归因给某批别名角点。

附件把两个分组混在了一句话中：C 的 free 面积 **TRAIN 减少 12.1%，DEV 减少 8.8%**；DEV 平均删除面 44.5→58.5。TRAIN 平均删除面仅 30.4375→32.0625，不能用 DEV 的大删面解释 TRAIN 的全部 FF 降幅。TRAIN bulk–bulk 非法对 330→175 是实测；“交界一圈被挖空导致这一下降”仍是假说。A 的 TRAIN 可见裂口 YES 总数 2→2、碎片 YES 1→0，不等于整个面板“绝对零新增裂口、零碎片”：DEV A 裂口 YES 3→5，碎片 YES 2→3。[原逐输出表](../r2_seam_abc_v1/evaluation/per_output.csv)；[原图审](../r2_seam_abc_v1/evaluation/visual_review.csv)。

### t=0 学对应并非信息论上不可能

固定生成器的输出是 `(C,N,Z)` 的函数；Gaussian 的各向同性不意味着观察到的具体噪声数组没有区分度，也不构成任何预测器都不能从初态推断终态对应的证明。当前证据只支持 **这个 3970 参数、局部 pair/Null MLP、此标签与训练预算没有学成对应**。

实现中每个查询看到自己的噪声角点、所在面的噪声均值、C 的中心/范围、N/K、候选 C 边特征和 tip 噪声，没有直接读取整组 bulk 槽的联合 OT 竞争或生成器隐藏状态。训练标签又由混噪前的 Z 与来源 Y 的全局 OT 得到，头读到的却是 HYBRID 混合后的初态；这可能增加标签条件歧义，不能凭现有结果量化成“彩票”或证明必然全 Null。[pointer.py](../../r2/pilots/seam_abc_v1/pointer.py)；[layout_features / prepare_training](../../r2/pilots/seam_abc_v1/shared_tip.py)；[OT 后 HYBRID 实现](../../r2/pilots/seam_abc_v1/slot_data_extension.py)。

### DEV 的 8 条是跨父均值，K=2 不是每份 8 条

LONG/A/C 的 DEV 原占用分别为 2d=16、610=4、608=4，宏均恰为 8。610/608 每份是 4 张 seam，不是 8；2d 为 K=14。单张 seam 平均面积必须按各父实际数量另算，不能把 pooled 6.11 面积和一个 TRAIN 平均数量拼成“所有 DEV 五倍巨型面”的解释。

可确认的是：这些方法在 DEV 上已有结构化 C/seam 合法共边，但很少进入 bulk；这不认证所有用户所需边界已经接完或主体用途正确。LONG/A/C 熟悉微调池均为四父，短边/长边覆盖差异值得关注，但 **B 的 47 父中已有 24 个 K=2 父**，B 不能被解释成仍然从未见过两三角完整输入。[B_SOURCE_CONDITIONS.csv](../r2_seam_abc_v1/B_SOURCE_CONDITIONS.csv)。

### B 的布局干预有价值，不能先宣布只有占用头有罪

833 参数占用头为 `24→32→1` 的 SiLU MLP，不是线性分类器。实际继承 class weights `[negative=8.25, positive=66/124]`，按有效批的实际边数平均。输入为 C-only 24 维边特征；标签是独立 owner 槽，不是所有来源支持边。头特征 detach、优化器独立，BCE 不反传到生成器；**生成器训练一直用真实 owner 布局，不用头的预测布局**。[occupancy.py](../../r2/src/meshflow_control/seam/occupancy.py)；[train.py](../../r2/pilots/seam_abc_v1/training/train.py)；[B 数据构造](../../r2/pilots/seam_abc_v1/b_pool.py)。

47 父 400 条开放边有 366 条来源支持，但只有 360 个唯一 owner；其余 6 条是同一来源面的 alias，不能重复占一个新面槽。2d 的来源支持是 16/16，独立 owner 是 13/16，不应把三条 alias 改称真实 Null。由 B 预测改为 LONG 原布局，分离的是**冻结生成器在两套推理布局下的反事实变化**。若恢复，支持占用布局贡献；若不恢复，说明旧布局替换不足。两种结果都不能独自证明占用头容量是唯一瓶颈，更不能等同于“冻结旧头重新训练 47 父”的实验。

## 2. 两项即将执行的零训练检查边界

### Idea 1：固定终态的 seam/bulk 局部绑定

最近先例是 [Partial-edge binding](../r2_partial_edge_binding_v1/REPORT.md) 和 [SEAM 容量保护](../r2_seam_geometry_guard_v1/REPORT.md)。前者 B2 实测无旧合法 FF 边损失，但新增非法 FF 面对仍有 4 对；“零撕裂”不等于“零新穿插”。后者保住 C/F 但有旧 FF 丢失及新 seam–bulk 穿插。**这次新变量是从固定 C 目标转向已生成 seam 的自由 tip/侧边，并允许两个无 C 锚的真实 GEO 簇中点协调；不新跑 D0，不重新聚类或改变原合并半径。**

建议冻结 LONG/A 每个 28 请求，在 1.5%L、2.0%L 两档共 112 个 CPU 单元。0.015L/0.02L 必须明确是候选簇代表点的距离门槛，还是每个原角点最终位移上限；中点使代表点各走一半，并不保证簇里最远原角点也只走一半。保留每一原角点相对 PRE_GEO 与原 POST 代表点的实际位移。

点共享与边共享必须分别登记：`TIP` 合并只保证一个共享顶点；`LATERAL_EDGE` 要两端点配对，才可能新增合法共边/进入边图。已属于 C 锚的簇不得中点移动；任何已有 C/F、seam/bulk、bulk/bulk 共边保留或丢失均按原来源面身份核账。受影响面包括两个簇关联的全部面。新增同面坍缩、形状恶化、法向变化、容量冲突、新 CF/FF 对分别记录并按冻结规则拒绝，不能把拒绝当成已成功修复。最终仍须原合法边图和实际图审，不能由空间点更近推出“彻底焊通”。

### Idea 2：B 生成器换用 LONG 原冻结布局

28 个相同 C/N/已存噪声；冻结 B_STEP1000 生成器，用 LONG 逐边占用 mask，不能仅匹配激活条数或根据结果手选边。复用旧 LONG/B 原结果；新增 28 Euler50=1400 生成器前向和 28 单次官方 D0，零更新。来源标签只作结果诊断，不进入新布局；不扫阈值、不额外 all-on/oracle。保持原 D0/GEO_C/定向合同。

以 TRAIN16/DEV12、逐父逐噪声报告：合法 C/seam、seam/bulk 及主体可达、FF/CF/I、小角、删面、面积、碎片/裂口。对 3f 由无 seam 到有 seam 的恢复，要同时判断新接续与代价；不能只由头激活数恢复宣布 B 被平反。该项不是纯 CPU，附件末尾把它并进“纯 CPU 几分钟”不准确。

## 3. Idea 3 的来源拓扑可行性：现有缓存的小查询

查询只读 `B_DATA_REGISTRATION.json` 记录的 47 TRAIN + 3 DEV `training_labels.npz`。对每个已登记 owner 源面 s，bulk 集合为 `free_ids - 所有 owner 面`：

- tip 候选：bulk 面包含 `edge_source_triple[s,2]` 源顶点。
- 侧边候选：同时包含 tip 和底边两个锚顶点之一。
- 对 seam-owner→bulk-face 二分图运行确定性增广路最大匹配，要求不同 companion 使用不同来源面，防重复目标。
- 仅核拓扑身份，不重新计算面对合法性；“有候选”不认证该局部几何无交或可定向。

| 既有池 | owner 数 | 有 tip 候选 | tip 单射最大匹配 | 有侧边候选 | 侧边单射最大匹配 |
|---|---:|---:|---:|---:|---:|
| TRAIN47 | 360 | 360 | 356 | 325 | 313 |
| DEV3（诊断） | 20 | 18 | 18 | 18 | 17 |
| 3f TRAIN | 13 | 13 | 13 | 13 | 13 |
| 73 TRAIN | 22 | 22 | 22 | 13 | 11 |
| 85 TRAIN | 10 | 10 | 10 | 10 | 10 |
| 8c TRAIN | 17 | 17 | 17 | 17 | 17 |

所有 47 TRAIN 父的每个 owner 都有 tip 候选，但 `893c689b` 的 12 owner 最多只能分配 8 个不同 tip companion（侧边最大 6）；因此“每 seam 固定消耗一张不同 companion 源面”的合同不能全覆盖。侧边单射可覆盖全部 owner 的父只有 40/47。DEV 的 2d 两个 owner（边 ID 8、10）在排除其他 seam owner 面后没有 bulk tip 候选；610 的 4、608 的 3 个来源 owner 均有独立侧边 companion。

这支持先研究确定性伴生表示，但必须处理无伙伴、伙伴竞争、多 seam 同源 tip 和预测误激活 Null。只共享一个 tip 不等于四边形条带，更不等于 legal-edge 可达面积必增；若伴生面仍与其余 bulk 断开，只是把孤立小翻片变大。共享噪声也不会单独保证整条积分路径和最终索引一致。当前两项检查不启动该新训练。

查询代码核心（实际执行为一线程 CPU NumPy，未导入 torch）：

```python
owners = d['edge_owner_face'][d['occupied_labels']].tolist()
triples = d['edge_source_triple'][d['occupied_labels']]
faces = d['source_face_vertex_ids']
bulk = [int(f) for f in d['free_ids'] if f not in set(owners)]
adj_tip, adj_edge = [], []
for tri in triples:
    tip, anchors = int(tri[2]), set(map(int, tri[:2]))
    adj_tip.append([f for f in bulk if tip in faces[f]])
    adj_edge.append([f for f in bulk if tip in faces[f]
                     and anchors.intersection(map(int, faces[f]))])

def maximum_matching(adj):
    owner = {}
    def augment(i, seen):
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False
    for i in range(len(adj)):
        augment(i, set())
    return len(owner)
```

## 4. Idea 4 与已有路线重合

`sum ||Yhat_i-Yhat_j||²` 对同源 bulk 顶点副本的一致性，本质上属于 [R07](../../r2/docs/research_route_registry.json) 的 L_FF。旧 [关系诊断](../mf_h1_local0/outputs/c_relation_diagnosis_20260929_204330/C_RELATION_DIAGNOSIS_REPORT.md) 已分析一步终点的共享组离散、位置与梯度时间分布，后续做过时间门控、FC-only/FF-only 以及后置 FC；不是第一次使用 Yhat。

本次 ABC A 只作用 seam-tip→bulk 一环，扩展到 bulk–bulk 是一个实际作用域变化，但它不是相交检测或排斥项：它能让同源副本靠近，并不阻止不同源面相交，错误整体位置、局部覆盖或收缩取舍仍可能存在。若未来重测，应明确新 SEAM 表示和监督域是重测理由，冻结 group/pair 权重与时间因子；不要把“邻接一致性”改名“防碰撞”就宣称新机制。当前不追加第四臂训练。

## 5. 本核对的文件身份

以下为本次实际读取的小文件 SHA256；来源 NPZ 的原登记 hash 保留在原 B_DATA_REGISTRATION 中，本次没有重复全量 hash 审计，也不冒称全部旧资产重新验真。

| 文件 | SHA256 |
|---|---|
| `r2_seam_abc_v1/B_DATA_REGISTRATION.json` | `7609dbd1217d93684aa2d49ae11e1387a047bbfb14646924b7f371b19249093f` |
| `r2_seam_abc_v1/B_SOURCE_CONDITIONS.csv` | `7543509d3c08e786cc2c9e01760dc74072c80d7394cb0c6f39d95753595dc82c` |
| `r2_seam_abc_v1/REPORT.md` | `7a209bb9a2f9413bf196c43bb6abfee17a1f7018332dead99fe35761caa23e93` |
| `r2/pilots/seam_abc_v1/shared_tip.py` | `6cb38fc96956a1982411772435ab42a1aaf7b2f43e24f728d0a6709063c62811` |
| `r2/pilots/seam_abc_v1/training/train.py` | `3ef10b8869b4a246e0ac700e08d9be1aeec7b98cdf56af274a5d1605e9ba701a` |
| `r2/src/meshflow_control/seam/occupancy.py` | `dd291d602a05d7113ccc2ac2641f4ec252d90e9566e50eb7823caa5e67c35120` |

本文件是执行前解释审阅和已有源数据的小查询回执，不是两项新实验的结果报告。后续结果出来前，不预设 Idea 1 安全、B 可恢复、Idea 3 必然泛化或 Idea 4 必然降 FF。

## 6. 执行后的独立绑定代码与账本核对

此补充在 CPU 112 单元完成后添加。审阅了 `binding.py`、旧 `partial_edge_binding_v1` 账本与合法边实现、ABC 原来源绕序/锚索引合同。原 GEO 簇全员并集合并、所有受影响面检查、原 PRE_GEO 累计位移限制、C 固定根与最终字节检查、原半边来源身份及统一评分域均符合冻结意图。点候选不直接记为合法边，FF 新穿插按协议记录而非前置拒绝，二者不能混为“零裂口/零相交”。

审阅发现一项代码边界：初始候选为 free tip，但排队后如果其簇被此前合并锚定到 C，后续候选仍需核当前根。作者在冻结前加入了 TIP 双侧当前根检查与 LATERAL partner→C 阻断；最终冻结 LATERAL 分支没有额外显式检查其自身 tip 当前根是否已为 C。末项意见到达时正式 CPU 矩阵已开始，因此没有静默修改代码或重跑。**对全部实际账本的独立重放确认：这个未显式检查的分支在本轮没有触发。** 这只支持本次结果，不证明任意未来输入上该分支可省。

独立核对只用 NumPy/JSON 读取旧 POST、PRE_GEO、已保存候选与输出；没有调用绑定解码、任何评分器或模型。每条接受操作按其 `original_cluster_ids → new_root` 顺序重建并查验当前 C 根。最终从原 corner→vertex 映射枚举全部实际移动角点，重新计算相对 PRE_GEO 的位移；同时检查来源面过滤映射及 C 的 FP32 字节。

| 检查 | 实际结果 |
|---|---:|
| 完成单元 | 112 |
| 已登记候选总数 | 2592 |
| 接受候选总数 | 251 |
| 排队 LATERAL 的原 free tip 已变为 C 根 | 0 |
| 接受操作新将任何原 free seam tip 锚定到 C | 0 |
| 账本重建根映射与保存映射一致 | 112/112 |
| 最终 C FP32 字节与原 POST 一致 | 112/112 |
| 全部实际移动原角点通过对应 1.5%/2%L 上限 | 112/112 |
| 来源面身份与保存 keep 掩码一致 | 112/112 |
| 所有单元最大实际 PRE_GEO 位移 | 0.18298949095987024 模型单位 |

核对对象 SHA256：`binding.py=c184cae7e66fe7ffa65f8f2d027a5c208b8121220fc2dc64e9a00f4fc87f1d60`；`binding/registration.json=56a525041461fec99155dcef886045206d88c8b44be9b412e785b858af5df87f`；`binding/completion.json=e5ade7ceea7ed5be55ba51f755b5fd88383df9de9b2f6dbabd23e1b02c6131cd`。质量、接续与图审结果以本轮主报告为准，这项检查不代替它们。
