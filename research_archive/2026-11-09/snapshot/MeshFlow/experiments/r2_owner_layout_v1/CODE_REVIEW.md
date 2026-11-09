# Frozen head / binding adapter：只读实现审查

2026-10-17。审查时两份 CPU 头已在固定预算训练。没有执行头前向、模型加载、解码、评分或任何补充实验，没有修改已冻结代码/协议。以下审查不代替终态账本或结果。

## 1. head.py 主要合同核对

被审查版本 `r2/pilots/owner_layout_v1/head.py` 的登记 SHA256 为 `70ea65e646514c937b497fbf1c4a5847cc92953ab45400b0273279cd70631218`。

| 项目 | 静态代码核对 |
|---|---|
| loss 按父平均 | `BCE * valid_mask` 沿边求和，除该父有效边数，再47父求均值；没有误变成全池按边平均 |
| padding | BCE忽略padding；attention每层只允许读取有效key，LayerNorm按token，不让padding变成上下文信息 |
| 标准化 | 均值/标准差仅由TRAIN47的原26维特征计算；无DEV值参与 |
| C-only输入 | 原C/N的24维特征，加C边序rank与E/32；forward不收Y、来源面ID、父ID或标签 |
| owner语义 | owner为1，alias和真Null为0；评估单独保留两个负类的specificity |
| 训练预算 | 每头固定2000次full47批前向/反向/AdamW；无额外训练探针、无early stopping，保存最终step2000 |
| 主候选 | `CONTEXT`固定；MLP只对照；代码没有根据DEV选择模型或回退MLP |
| DEV | 优化器只用train.npz；DEV特征在终点评价创建，标签在该batch前向之后只用于计数与固定gate |
| 阈值 | `z>=0`与sigmoid≥0.5等价；没有阈值/温度搜索 |
| 统计 | 先滤掉padding，再算TP/FN/FP/TN、分角色specificity、Brier、稳定logaddexp NLL和固定10-bin二分类可靠性ECE |
| 失败/重跑 | 已有started标记阻止静默重试；complete与checkpoint hash核对；训练不调用生成器或D0 |

实际登记TRAIN最大22边、DEV最大16边。准备代码虽然用二者的最大值确定padding宽度，但本次宽度22完全由TRAIN决定；且有效key/mask使padding不参与任务输入。准备/contract确实提前读取DEV标签文件的字节以验hash，这不是提前用标签优化；“DEV标签只在terminal使用”应准确指其**内容只用于terminal指标/gate**。

预计终态保存2×(400TRAIN+24DEV)=848条逐边记录、2×(47+3)=100条逐父记录。终态真实前向为每模型TRAIN一个packed batch、DEV一个packed batch，共4次。单条件prediction receipt的 `head_batch_calls=1` 表示该条件参与一个batch，不能把50份receipt相加冒充50次CONTEXT前向；CONTEXT本身终态只前向2次。

没有发现要求停止当前训练的实现错误。rank明确依赖局部输入编号；本轮不能据此认证任意顶点重编号不变。TRAIN完美拟合和低ECE也不认证开放生成或跨父校准。

## 2. 冻结协议中的一处算术文字矛盾

原协议同时写了“各DEV父owner recall≥90%”与“这意味着20条全部召回”。后者不成立：2d有13个owner，12/13≈92.31%已满足90%；610的4条和608的3条则必须全部召回。因此逐父90%的实际代码门槛允许最多漏掉2d一条，最少19/20，而不是必需20/20。

这一问题已即时告知根会话。应保留已冻结代码/原协议，由独立勘误解释实际执行的是各父≥90%，不因最终结果把门槛事后加严或放松。源检查见 [head.py](../../r2/pilots/owner_layout_v1/head.py)、[原冻结协议](PROTOCOL.md)。

## 3. 非致命维护与报告边界

- `prepare`与`contract`保护原输入、label文件和train.npz，冻结head.py与协议hash。C特征实际依赖旧 `slot_data.py`；建议最终归档其实际源码/hash并与旧登记比对。训练读已冻结train.npz，不会受后续特征函数改动影响；terminal DEV生成特征仍需保持旧定义不变。
- per-parent gate的familiar选择依赖已登记四个父ID前缀；当前数据确实有这四父，故不是空集合通过。未来模块复用应显式断言恰好四个，不把本轮成功泛化成任意缺失输入都安全。
- 全池micro recall与分父门槛同时报告；训练loss是parent mean，summary NLL/ECE是有效边micro统计，二者分母不同但实现并不冲突，不能直接把数字视作同一loss。
- gate失败只支持“这套固定头/训练/门槛未获准进入生成”，不支持新布局一定使mesh退化。gate通过也只允许既定生成矩阵，不自动证明几何质量提升。

建议最终报告顺序：①本次是否通过固定门槛及实际执行范围；②两头逐角色/逐父分类与校准、所有失败边；③若放行则LONG/B原布局→新布局→绑定三阶段配对几何与图审；④费用/预算与所有保留代价；⑤固定结论和停止条件。若不放行，明确三个下游阶段NOT_RUN，不填零结果。

## 4. 新 binding_adapter.py guard / clone 只读核对

应另一执行者请求，额外静态检查新适配器，没有运行其自检或几何处理。

- 私有namespace只复制旧binding模块自有函数；`decode`、`candidate_inventory`、`worker`在新namespace解析路径、方法标签与`state_merge`。没有向旧B模块的global写入，也不改旧源文件。
- 新 `state_merge` 对每个pair重新取当前roots；当seam侧原始簇为free时，双方当前root必须仍free。检查位于pair循环内，覆盖同一LATERAL候选的前一个pair刚把相关簇并入C的情况。
- 固定C簇并合时保持C root作为代表，故`root in fixed`可识别后续C归属；不同C根仍拒绝。两自由簇中点赋值、全部原簇成员PRE_GEO位移、全部受影响面检查与最终C/累计位移核验继续由旧核心执行。
- 新标签的seam→bulk评分显式适用，读取实际`raw_seam_slot_mask`，没有因标签名不在旧allow-list而默认为NA。
- 没有发现阻止新适配器冻结的问题。仍须保留实际几何结果和连接账本；新增FF不是该版本的硬否决项，不能写成零新增自交或零损伤。

该适配器尚须父会话正式gate、完成采样/基准POST评分后才可冻结与派发。本审查不授权绕过门槛。
