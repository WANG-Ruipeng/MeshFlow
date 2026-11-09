# Bench 完整部件与曲面条件重构

已实际构造11条候选：10条通过几何资格、逐图审阅和原生产CPU打包，1条隔离。合格输入来自4个来源，其中只有9e429b是旧25父池之外的新父。独立的合并提案为 **26父、91条条件**；原25父/81条件和39条DEV原件保持不变，未切换活动池。

本轮是数据准备，零训练、零模型前向、零GPU计算。不能据此宣布生成质量提升或数据覆盖已经充分。

## 实际构造

从已核验完整目标M的现有面选择C；Y为精确补集。没有移动顶点、裁断三角形、修补M、重新三角化或按面数比例强凑。目标仍为原N189/256、原规范化和角序版本。下面的面ID指冻结target_case.source_faces行号；历史QEM之前的来源关系沿原lineage.json查询，不冒称原始高面数网格的一一对应面ID。

| 来源 | 条件 | K | 开放边 | owner/alias/null | 录用 |
|---|---|---:|---:|---:|---|
| a8e987 | SLATTED_TOP_FULL | 39 | 39 | 31/0/8 | HOLD |
| a8e987 | SINGLE_SLAT_TOP_FULL | 2 | 4 | 3/0/1 | ACCEPT |
| a8e987 | END_FRAME_FULL | 71 | 39 | 36/0/3 | ACCEPT |
| 7b58a3 | SINGLE_SLAT_TOP_FULL | 2 | 4 | 3/0/1 | ACCEPT |
| 7b58a3 | END_WALL_FULL | 72 | 12 | 10/0/2 | ACCEPT |
| 9e429b | CURVED_THREE_BAY_SEAT_SURFACE | 26 | 28 | 28/0/0 | ACCEPT |
| 9e429b | COMPLETE_LEFT_END_OUTER_SURFACE | 20 | 22 | 21/1/0 | ACCEPT |
| 9e429b | CURVED_SEAT_PLUS_LEFT_END | 46 | 48 | 47/1/0 | ACCEPT |
| 943dde | TRUE_CURVED_SEAT_UPPER_SURFACE | 24 | 26 | 26/0/0 | ACCEPT |
| 943dde | COMPLETE_INNER_CURVED_BACK | 15 | 17 | 17/0/0 | ACCEPT |
| 943dde | CURVED_INNER_SEAT_BACK | 39 | 41 | 41/0/0 | ACCEPT |

a8的39面完整木条组虽然视觉完整，却有非流形顶点，保留为HELD_GEOMETRY，不进入新训练清单。没有通过删掉局部面伪装成合格完整输入。7b已有28面木条组与旧条件完全重复，未计新增。两条K=2输入是整根木条的真实上表面，长宽比分别约23和84，不是随机细条；它们仍含自然小角，不能宣称所有新输入都低小角。端部和曲面条目按表面边界明确命名，开放外曲面不声称是闭合实体端块。

943新增真正的上座面、内曲背及连续组合；9e新增三湾座面、端部外曲面及组合。它们补充了以前只含平座底/外背、截断下端块等输入时缺少的结构关系。四父每父2或3条，并不等于新增10种独立Bench款式。

## 核验与数据合同

所有新C保持原顶点、面身份及可逆来源角序。C/Y源面ID互斥且并集等于M，数组逐位还原；10条合格C均为一个边连通分量，原有限三角面检查通过。完整目标复用已通过的原几何与视觉审阅，不重复跑全目标审计。

实际查看了11张最终打包数组的M/C/Y/C+Y双视图，并为窄木条补顶视图。记录为AI_PROXY，不冒称人工评审，也没有把几何PASS直接当作语义标签。

对最终10条各执行一次原生产SEAM数据打包和CPU受限OT：10/10 PASS。含C/N推理文件检查、来源owner/alias引用、目标面各出现一次、固定角点与FM掩码、OT后HYBRID0.25、联合尺度增强检查。仅生成20个CPU数据验证噪声数组，不是训练/评测噪声，不运行生成器、去噪器或新评价面板。正式推理输入只含C/N所需字段；部位名称、Y、完整M和owner/alias来源标签仅用于整理与训练监督。

39条原DEV记录（8父）逐条不变，TRAIN/DEV来源隔离检查通过。旧文件与正式生产代码hash复核未变。评测来源是一种参考完成，不作为开放生成唯一GT。

## 交付与使用

- `manifest.json`：本轮10条录用输入及1条隔离证据，配`DATASET_READY.json`和`validation/latest.json`使用。manifest封存于CPU验收前，最终PASS由带hash的外部回执确认。
- `proposed_merged_manifest.json`：旧81＋新10＝91条、26父的独立合并提案；未覆盖旧活动池。
- `conditions.csv`：全11条的K/N、面ID、拓扑、来源接口、描述与录用状态。
- `conditions/`：每条实际C.obj、Y.obj、input.npz、paired.npz、case.npz、supervision_only.npz、quality.json与selection.json；隔离条目也保留。
- `review.html`：最终图册；`visual_review.json`记录实际图审和图片hash。
- `protected_source_hashes.json`、`source_hashes.json`、`delivery_hashes.json`：输入/代码/交付完整性；`ledger.jsonl`记录阶段。
- `r2/data_tools/bench_user_relation_rebuild_v1/resume.ps1`：默认只验证封存结果，不重算OT、不训练。

资源记录在resources文件中。两条独立区域设计并行，几何打包和正式CPU OT各单worker、每进程一个数值线程；此规模CPU工作不需要GPU。没有用额外GPU探针，不干预用户的GPU进程。构图/图审耗时不能当作模型训练吞吐。

## 这一步能支持的下一项决定

现在已有一批可直接进入现有数据入口的完整部件和非平面输入，值得作为后续配对数据覆盖实验的增量。应继续用原R2/SEAM模型与训练预算检验其作用；不能把一父三个条件算作三倍独立数据，也不宜只用这10条替换旧81条。训练时继续按父组织曝光，并单独统计新区域出现次数。本轮没有注册或启动下一次训练。
