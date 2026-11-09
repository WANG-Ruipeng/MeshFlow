# Bench 与 Chair 对标核查（2026-10-19）

**结论：本轮是 Chair 训练阶段与基础优化配方的迁移，以及 Bench 内部 Native/SEAM 的同起点比较；不是 Chair 已保留最佳完整系统的跨类别复现。** 把它表述成“除类别外全部相同”不成立。纯 CPU 核验未发现 Bench 数据的整体尺度、C/Y 分区或角点规范化错误。

## 已证实的对标问题

| 项目 | Chair 实际保留链路 | 本轮 Bench | 影响 |
|---|---|---|---|
| `PLAIN` 含义 | `PLAIN_S2000` **仍是 SEAM**，只是没有 couple loss；不是 Native | `PLAIN` 是普通 Native，`SEAM` 是另一臂 | 同名不能视为同架构基线；必须在报告中写 Native/SEAM |
| 训练谱系 | 官方 Chair→EXP67 6000→FULL +6000→四 C SEAM200→B_POOL47 +800→PLAIN +2000；最终 global15000，SEAM 累计3000 | 官方 Bench EMA→multipart Native12000→同模型/AdamW/RNG 分叉，两臂各9000；最终 global21000 | 最终预算、课程、父曝光均不同；Bench15000只是阶段预算较接近，也不是全部历史一致 |
| 推理布局 | PLAIN_S2000 使用冻结 LONG 的 C-only 缓存布局 | 833 参数头在 Bench SEAM 全程独立优化，TRAIN 平衡 BCE，logit≥0 | 跨类别比较同时改变布局学习政策；不能将差异只归于生成器 |
| 最终后处理 | 已保留 S2000＋HINGE 1.5%L 反向绕序守卫，并归档后续 CPU 候选 | 单次官方 Bench D0→原 GEO_C→来源定向；**没有 HINGE 或后续 CPU 接合** | 当前 POST 与 Chair 最佳最终 POST 不是相同系统；这是对标范围缺口，不是偷偷漏执行注册步骤 |

依据：`r2/pilots/seam_pool47_couple_v1/protocol.py:31–36`；本实验 `registration.json`；`r2/pilots/bench_parts_convergence_v1/postprocess.py:71–115`；`WORKSPACE.md:1–7`；已存 `CHAIR_COMPARABILITY.md`。这些差异不会取消 Bench 内部同起点对照的价值，但限制其外部比较解释。

## 已执行的数据与类别检查

独立按官方归一化公式从原 archive 重建 **72 父**，并核对 **316 条条件**：源顶点/面、归一化坐标、规范化目标、`pre_ot_full×2`、C 源面/索引坐标、Y 补集与全体分区 **全部精确一致**。所有父归一化最大轴跨度均为1，模型坐标跨度均为5.316321；C 没有独立归一化。TRAIN/DEV 目标 scalar RMS 中位数1.49567/1.50466，没有“DEV 整体被多乘一个尺度”的证据。生成器与 D0 使用 Bench 类别02828884，未见 Chair 权重回退；D0 输入缩放 `.95×.3762` 与官方生成器/去噪器坐标合同相容。Bench 官方 D0 `fixed_std=.05` 与 Chair `.02` 是不同类别模型配置，并非本次额外注入噪声。

Bench 数据为64 TRAIN/8 DEV父、277/39条件，包含座面、腿、靠背、支撑和组合；其中49条是闭合 C，采用普通 free 分支。Chair 的主体比较任务是47父完整座面。父选择、K/N、开放边率、语义难度和原始质量均不相同，不能称跨类别单变量。Bench 父集合还继承了早期“存在合格连贯平面区域”的筛选，并非整个官方 Bench 分布。官方 Bench 预训练配置无旋转/缩放增强、使用排列增强；本轮沿用 Chair 的 x 宽度0.9–1.1增强，两 Bench 臂一致。

## Step0 固定 probe 异常的边界

按现成逐项 **SSE/count** 重算（未新增前向或 OT）：TRAIN ALL_free MSE **0.157445**，DEV **0.879781**，为5.59倍。按官方拆分，TRAIN 全为 official-train；DEV 的7个 official-test 父合计 **1.016429**，唯一 official-train 父为 **0.182999**。t=.2/.5/.8，TRAIN 为 **.38504/.06603/.02126**，DEV 为 **.79862/.52258/1.31814**。现成噪声 RMS 为.9691–1.0607，且数据尺度/分区核验通过。

**这是训练开始前已经存在、与官方预训练拆分相关的差异，不能证明由预训练记忆造成，也不能将它解释成这轮训练制造的泛化差距。** 官方拆分、父形状、C类型与难度互相混杂；此次没有重新运行 OT/模型，因此 OT 数值正确性与预训练熟悉度的独立因果贡献仍为 UNKNOWN。没有证据支持“归一化错了”或“只要延长训练便能修复”。

可复算材料：本目录 `comparability_check.py`、`coordinate_parent_checks.csv`、`coordinate_condition_checks.csv`、`coordinate_probe_summary.json`、`step0_probe_rows.csv`、`step0_probe_decomposition.json`。本次0模型加载、0前向、0新噪声、0新OT；未改生产代码或原资产。
