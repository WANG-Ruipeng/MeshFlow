# Bench 生产 C-only 布局合同：源码与现存数组核对

2026-10-22。纯 CPU，零神经网络/占用头前向，零 GPU、训练、OT、新噪声及生成评分。来源文件只读。主审计约75秒，补充数组合同检查约1秒；不是再次进行几何质量审计。

**训练的 source-owner 确实改变模型可见输入，不只是监督匹配的名字不同。但本次没有证明这种差异导致 Bench 生成失败。** 现有91条训练数据不能直接声明为“已经与生产 C-only 完全对齐”：其中85条没有精确 C/N 对应的生产头缓存。这里交付的是可执行合同及阻塞清单，没有偷偷用来源标签补齐缺失预测。

## 1. 源码结论

| 路径 | 核实到的实际行为 |
|---|---|
| `bench_parts_convergence_v1/train.py:64` | COMMON/PLAIN调用普通Native；SEAM调用角点适配。COMMON的普通free布局不使用来源owner决定锚边。 |
| `bench_relation_training_v1/data.py:30` / `bench_user_v2_training_v1/data.py` | 条件及完整目标一起缩放，然后调用原SEAM监督打包。 |
| `seam_abc_v1/slot_data_extension.py:24` | 直接以`case['occupied_labels']`作为`inference_layout`的occupied输入。它来自来源完整网格的唯一owner定义。 |
| `meshflow_control/seam/slot_data.py:83` | 每个激活边对应free槽的两个角点被设为C顶点索引；Null/未激活槽返回三自由角bulk。 |
| `meshflow_control/seam/model_adapter.py:83` | 角点known掩码直接选择role embedding。:101及:114选择known-time的MLP和最终调制；非C面attention仍取free t，不能说整个seam面都是known。 |
| `seam_abc_v1/slot_data_extension.py:58` | x0锚角先写入C；xt、目标速度与有效FM分母都随锚角变化。源布局不是仅在loss末端使用。 |
| `bench_relation_training_v1/eval_plan.py:98` | 推理从已保存OLD_S15000头预测读取occupied，再从C/N重建布局，核对掩码和锚边；不是读取source-owner。 |
| `bench_trajectory_diagnostic_v1/gpu.py:29` | 先前4条当前TRAIN缓存来自既有占用头，阈值logit≥0；本次只读缓存。 |

因此，少开一条边会多出6个自由标量，同时改变该槽起点、轨迹、角点role/time和监督分母。即使激活总数相同、但选择另一条边，也不等价于普通face permutation。C-face Geo上下文掩码和N不因此变化。

这属于“来源布局监督＋预测布局推理”的实际条件分布差异；它可以是分解式生成设计的局限，不自动等同于程序bug或推理偷看Y。当前采样确实没有偷看Y；问题是是否需要改变训练任务使两端使用同一可部署布局策略。

## 2. 当前26父、91C能核对到哪里

来源91条共有1,515条开放边：1,428 owner、14 alias、73 Null；全部来源owner布局均通过原N、原面身份和FP32角点逐位重建。owner与alias来自同一个Y面时，只实例化一次，其余槽回bulk。

| 当前训练条件状态 | 条数 | 本次可执行结果 |
|---|---:|---|
| 有精确C/N生产预测缓存 | 4 | 1布局相同、3有真实role/anchor差异 |
| 闭合C、无需头预测 | 2 | C-only唯一空布局，原目标精确可表示 |
| 没有精确C/N预测缓存 | 85 | BLOCKED；不运行头，不以source-owner代替 |

| 已缓存条件 | 来源owner→生产激活 | 忠实端点打包 |
|---|---:|---|
| 35948e SEAT_SURFACE | 4→2 | 可表示；关掉的两张原接口面完整返回bulk |
| 838c48 PART_ASSEMBLY | 27→25 | 不兼容：误开的edge0、18没有任何原Y面包含所要求的两端点，25槽至多匹配23原面 |
| 943dde BACK_SURFACE | 4→4 | 相同布局，原样可表示 |
| bf504b LEG_SUPPORT | 13→8 | 可表示；遗漏owner不需要删除原面 |

缓存复用要求C顶点、来源绕序、模型角序及N数组签名全部一致，不能按父ID或条件名模糊复用。扫描552个历史Bench布局文件，124种文件内容；全部历史版本保存在清单。多历史头产生的mask没有按Y标签择优。

另一个限制是增强：现存4个缓存仅覆盖这4条未增强C。原训练会改变C的世界坐标、长度和特征；不能未经核实就把旧mask声称为增强后生产头的预测。新日程须明确“先C-only布局再联合变换”还是“变换C后重新运行生产规则”，并保持训练/部署定义一致。

## 3. 实际48请求面板与历史变体分开

按原冻结`eval_manifest.json`的`old_layout_sha256`精确选择，共24C×2noise。`actual_production_panel.csv`是这24条，不能用全历史102个布局变体的统计代替。

- 11/24模型可见布局完全相同，13/24不同；不存在仅face排列差异的例子。
- 来源133 owner、生产105 active；38条边的角色改变，其中5条误开来源Null。
- 20/24满足既有**合法来源owner**合同，可无复制/删面/改N精确编码原目标。
- 其余4/24都在历史TRAIN的1dd740/20222f两父；实际DEV16/16都可在合法owner合同下编码来源目标。

“不满足合法owner合同”须与“绝对无法表达原网格”分开。独立的源顶点—原Y面二分匹配发现，这4个旧TRAIN条件虽然被原合法owner判定拒绝，仍存在含所需锚端点的原Y面，可以几何表示但不认证接口合法。它们不是4个数学上不可能的目标。当前TRAIN的838c则连原面端点匹配都不成立。

所以，本次不能把DEV失败解释成“所有DEV target都因布局不匹配而不可能”，也不能根据可表示就推断模型一定会学好。

## 4. 可执行的后续合同

1. **输入层仅接收C与部署可得的N、可逆C角序/顶点身份和冻结布局策略版本。** 不读取来源父ID、Y、source-owner、part语义来决定occupied。
2. 从该输入生成或读取可验证的occupied/anchor/mask。保存输入签名、策略/头hash、实际预测和边顺序；缺失则BLOCKED。闭合C可直接产生唯一空mask，无模型调用。
3. **输入布局冻结以后才读取Y做监督分配。** 每个激活边必须分配一张含锚边的原Y面，每张原面最多实例化一次。alias可代替其owner边承担同一张面，但不能两槽复制同一面；改变owner边仍是模型可见布局变化。
4. 所有剩余原Y面进入bulk。原总N、C、原面坐标、source-face ID全集逐位/一一核验，保证没复制、删难面、制造连接帆或退化槽。当前`compatible_endpoint`已实际完成这一步，不新增OT。
5. 激活Null且无原目标邻面、或多槽争同一原面时，返回表示不兼容。禁止用Y悄悄关边再声称C-only；禁止把“该样本不兼容”升级为整父坏数据。需要另行定义可部署策略、明确任务筛选范围，或改变表示，不能通过打包遮掩。
6. 将来若继续原free-only FM，只有通过一一分配的记录才可做角色受限OT、原HYBRID和随机face排列。本次只保存确定性端点合同示例，**不是已生成训练样本、不是训练入口替换**。

ALL_ON是可由C决定的另一种规则，但只是本次代数可行性诊断，未采用为生产策略。91条中56满足合法owner唯一分配、35不满足；这不支持“全开所有边即可直接训练”。ALL_OFF虽可回到普通Native，也不能冒称保留了SEAM的接续机制，故未作为自动回退。

## 5. 文件与恢复

- `per_condition_layout.csv`：91当前条件及全部旧面板缓存变体。
- `per_edge_layout.csv`：逐边owner/alias/Null与预测角色。
- `actual_production_panel.csv`：实际部署版本的24条件。
- `compatibility_details.json` / `blocked_exact_source_incidence.csv`：合法owner限制与一般源面端点匹配分别记录。
- `endpoint_contract_examples/`：成功布局的原面一一端点数组，无噪声、OT或训练前向。
- `COMPLETE.json`：主审计的原资产/脚本/产物hash；`contract_checks.json`：补充三个合同反例检查与实际生产版本统计。

CPU复核：`python -B -m r2.data_tools.bench_deployment_tasks_v1.layout_contract`（已完成时仅核hash）；补充复算：`python -B -m r2.data_tools.bench_deployment_tasks_v1.test_layout_contract`。

本项不证明数据是唯一根因，也不自动启动占用头、改模型、生成或训练。独立结构覆盖建设可以继续；在上述缺失/不兼容关闭前，不能把完整新增池标成生产C-only训练就绪。
