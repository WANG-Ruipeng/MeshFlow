# Bench 完整座面覆盖：实际数据制作记录

本轮冻结 **24 个训练父、每父 1 条完整座面条件**：原课程的 22 父全部保留，新增 `a8e987` 与 `27eb79` 两父；`df6fd3` 单独留出。这里的增量是两种既有来源资产上的完整座面任务，不是大规模扩展，也不是三个从未在项目中出现过的新父。

本文件记录已经完成的数据工作。训练计划来自已冻结的 [training_registration.json](training_registration.json) 与 [PROTOCOL.md](PROTOCOL.md)；数据报告不预报训练结果，也不把计划曝光写成已经完成的曝光。

## 实际制作与纳入范围

从现有已核验的低面数完整目标提取真实原面构成 C，令 Y 为同一目标的精确面补集。没有重建高面目标、修改完整网格、移动顶点、重三角化、添加连接面，或为满足条件数量删掉困难面。每条新候选保存来源面 ID、原角序和可逆角序映射，以及 `case/input/paired/supervision_only` 文件及哈希。

| 父对象 | N | K | 完整座面候选 | 本轮决定 |
| --- | ---: | ---: | --- | --- |
| `a8e987a6a6486b962d6c8e51365a5a87` | 189 | 39 | 完整木条组及周围座面框条上表面 | NEW 训练 |
| `27eb79ddf26fd7ac7898580cb9d16864` | 255 | 29 | 弧形完整座碗，上表面止于转入靠背的弯折处 | NEW 训练 |
| `df6fd36040a5719662701a18243b45d3` | 255 | 38 | 中央背隔板两侧的完整双向座面 | 本轮留出，仅 1 父 |
| `3e4f4c54420bdcae75becd8a4c3f1866` | 158 | 16 | 带不规则中部切口的来源座面 | HOLD：本轮规则完整座面任务不采用 |
| `d9de3b770e53a41d5b7bc62e486d578c` | 152 | 2 | 狭窄局部座带 | HOLD：不视为完整座面 |

原 22 条训练记录完整继承，仅添加 `replay_group=OLD`；原条件、目标路径与哈希未替换。新训练两父为 `replay_group=NEW`。后两条 HOLD 仍有完整候选文件与验收记录，不删除、不伪装为数据损坏，也不因已有生成好坏调整准入。

实际根代理查看了全部五张 M/C/Y 三视图，然后冻结上述选择。评阅是非盲 **AI_PROXY**，不是独立人类语义标注。完整记录和图片哈希见 [root_input_review.json](data/root_input_review.json)。

| 输入检查图：M / C / Y，每张含正斜、反斜和俯视 | 决定 |
| --- | --- |
| [a8 木条完整座面](data/gallery/a8e987a6a6486b962d6c8e51365a5a87__COMPLETE_SEAT_CANDIDATE__COVERAGE_V1.png) | NEW 训练 |
| [27 弧形座碗](data/gallery/27eb79ddf26fd7ac7898580cb9d16864__COMPLETE_SEAT_CANDIDATE__COVERAGE_V1.png) | NEW 训练 |
| [df6 双向座面](data/gallery/df6fd36040a5719662701a18243b45d3__COMPLETE_SEAT_CANDIDATE__COVERAGE_V1.png) | 留出 |
| [3e 不规则切口](data/gallery/3e4f4c54420bdcae75becd8a4c3f1866__COMPLETE_SEAT_CANDIDATE__COVERAGE_V1.png) | HOLD |
| [d9 局部座带](data/gallery/d9de3b770e53a41d5b7bc62e486d578c__COMPLETE_SEAT_CANDIDATE__COVERAGE_V1.png) | HOLD |

## 谱系与泛化边界

新增训练的 a8、27 和留出的 df6 都不属于原完整座面课程 22 父，也没有出现在本轮初始化所继承的 COMMON12000 父池；因此可以称为**新于当前 COMMON→Native 起点谱系的父/任务**。三者均出现在项目的其他历史适配分支：a8、df6 在 RELATION26 中出现，三者均在历史 NEWDATA27 分支出现。它们不是项目首次见到的 ID。

三者来自官方 **train** 来源划分。没有官方预训练逐父曝光日志，故它们是否被官方实际检查点训练过为 UNKNOWN，不能宣称预训练未见。不同 UID 和已有精确去重检查也不证明款式家族完全独立。df6 只提供 **1 个本轮未训练父**的观察，两个固定噪声仍不是两个独立父，不足以支持宽泛的跨结构泛化结论。

3e 和 d9 已属于 COMMON12000，且完整座面范围不满足本轮选择，未纳入训练或新增评测。本轮没有把旧 DEV 父混入训练，也没有用旧 DEV 坏例的生成结果挑选这两条新训练 C。

## T-junction 合同与原判定保留

a8 的 C 有 2 个非流形顶点链接，df6 有 1 个。来源面在局部边界上存在**部分共线接触、端点分段不同的非共形 T-junction**：共享一个来源顶点或几何线段，不等于两面具有同一条索引共边。a8 对应来源顶点 76、89 的局部接触；df6 对应来源顶点 111。有限共线接触段分别约为 0.110779、0.110779、1.166955 模型单位。

这两条候选原 `QUARANTINE / C_NONMANIFOLD_VERTICES` 状态完整保留，没有改成旧规则的 PASS。本轮为 Native 输入独立记录一项准入政策：语义上完整的座面有限三角形可包含保留的来源非共形装配接触；不要求 C 为共形流形，但仍要求有限面没有非法穿插、正面积重叠、未决面对、退化面和精确重复面。

五条新候选的这些有限面检查均为 0；a8/df6 的非流形顶点限制仍然存在。**不得称其“拓扑干净”，也不认证原索引连接完整，更不把该 Native 准入自动解释为未来 SEAM 布局兼容。** 没有为了通过旧拓扑门槛而删面、拆点或改变来源身份。细项见 [vertex_link_audit.json](data/vertex_link_audit.json) 和各条件 `link_interpretation.json`。

## 已实际完成的 CPU 验收

对 5 条候选各执行一次原生产 Native CPU OT，共 **5/5 通过**，使用独立的 CPU 验证噪声；这些噪声不是评测噪声。检查了 C/Y 精确分区、每个目标面恰好出现一次、OT 面与角点排列、C 原样保持、free-only FM 分母，以及先 OT 后 HYBRID0.25 的路径。没有神经网络前向或 GPU 工作。

完整目标几何证据直接复用已有 receipt，没有重复执行全目标昂贵审计。制作前后来源目标哈希一致，原始文件只读。旧 22 条复用原 CPU gate。见 [validation/summary.json](data/validation/summary.json)、[cpu_gate.json](data/cpu_gate.json) 和 [protected_source_hashes.json](data/protected_source_hashes.json)。该 gate 的 PASS 指新 Native 有限面政策与数组合同，不是抹去旧隔离原因。

## 曝光与固定评价

已注册候选为新增 1,500 更新，每步 batch8：两个 OLD 父、两个 NEW 父，每父各一条 identity 与宽度增强实例。计划总实例曝光为 12,000，其中 OLD 合计 6,000，原 22 父每父约 **272—274 次**；两个 NEW 父各 **3,000 次**。这是对两种结构的密集适配，NEW 单父曝光约为 OLD 单父的 11 倍；不能把它包装成大规模结构覆盖，也不能将训练收益单独归因于新增独立父数量。实际完成曝光以训练账本为准。

保留旧 24 请求原 C/N/噪声，新增 a8、27、df6 各 2 噪声，共 6 请求，合计每模型 30 请求选择。a8/27 是本轮训练条件评价，df6 是仅一个本轮未训练父，旧 DEV 仍是历史开发回归。来源完整体只作为构造、监督与展示参考，不作为开放生成唯一正确款式。

数据阶段的 [eval_selection.json](data/eval_selection.json) 只冻结条件身份；其后正式注册的 [eval_manifest.json](eval_manifest.json) 冻结实际推理文件、噪声与查询。CPU 验证噪声未替代任何评测噪声。

数据与步数同时改变，且没有本轮等预算 OLD-only 续训臂。因此这轮能检验整个“原池回放＋两条完整座面密集适配”版本是否有用，不能证明独立的数据因果效应。

## 可复算入口与证据

- [冻结数据清单](data/manifest.json)，SHA256：`8a3a3cbaab4a7fa2bb8e60001eac42913f3735ec3e72cc2cbba4ba83dffc8148`。
- [五候选原始清单](data/candidate_manifest.json)、[图库元数据](data/gallery_manifest.json)、[制作账本](data/build_receipt.json)。
- 数据代码：`r2/data_tools/bench_seat_coverage_v1/`，包括 `run.py`、`validate_native.py`、`link_audit.py`、`finalize.py`。现有产物为不可覆盖保存；恢复入口会检查已有身份，不自动训练。

原数据、旧评分、旧准入记录、原报告和已有权重保持不变。本次数据制作没有删除资产、切换默认模型、提交或推送 Git。
