# MeshFlowCtrl：当前实验结果、保留配方与训练谱系

日期：2026-10-28。性质：已有本地证据的只读整理，不是一次新实验。没有训练、模型前向、生成、重新评分、重新图审、清理、默认切换或 Git 提交/推送。

## 1. 先给结论

**我们已经有值得保留的训练路线，但还没有一个同时在 Chair、Bench、熟悉条件和未见父上全面胜出的设置。** 目前的选择应分成三项：

| 用途 | 目前最值得保留的设置 | 证据支持到哪里 |
|---|---|---|
| Chair 的 SEAM 方法与论文正结果 | `PLAIN_S2000` 生成器＋固定 `LONG` C-only head；单次官方 D0、GEO_C、来源定向、`HINGE_OPP015`；`ALPHA_HOLE`另列版本 | 有实际共享连接、主体可达和局部去碎片收益；是成熟的有限研究组合，未解决全部 DEV 接续 |
| Bench 下一轮生成器研究起点 | `EXPOSURE / M_PLUS3000`，global25500 | 扩展的自然 N 任务继续训练后，RAW 主体/采用有新增，固定 FINAL 的总体 FF 也下降；**不是完整流程已晋级** |
| Bench 必须保留的比较参照 | 窄池自然 N 的 `B / NATURAL_HIGH`，global22500；另保留原 `Native6000/SEAM3000` 历史研究参考 | B 保留两新增结构的学习收益，W/M 未整体超过它；不同面板与后端损失使单一“总冠军”说法不成立 |

这里的“最好/最有希望”均指**本地已测候选及其限定用途**，不是与外部论文统一评测后的领域 SOTA。维护 CLI 默认仍是 Chair HYBRID START；研究选择不等于默认入口已经切换。

**共同值得继续保留的配方是：官方预训练先验 → 真实、完整、连贯的人类式 C 任务 → Native 条件 FM 在固定预算内持续适配 → 按父组织曝光、保留旧任务 → 在已验证的类别/任务上使用 SEAM。** 新增复杂 loss、RL、不断延长同一个小池，或只改后处理，均没有形成可通用替代这一主线的证据。

## 2. 本次整理覆盖与证据口径

- 历史研究记忆已登记 R01—R30 共30个未形成稳定升级的机制家族、2个训练前门禁阻塞和单列 R31 SEAM 有限正结果。它们不等于30次训练，也不表示每一种方法在所有设置都无效。
- 本次另建立现行 `experiments/r2_*` 的 **97个目录、102份顶层报告文件**索引，包含部分草稿/阶段报告。其数量不与历史87份来源相加；有重叠，也有一个实验多份报告。
- 本次重点复核保留模型的训练谱系、直接配对的正式结果、后续复验及负结果收口；报告索引不意味着本次对102份报告进行了全量重新审计。
- 数值沿用原评分器、原输出、原面板、原分母。CF/FF通常指受非法面对影响的 free 面比例；I 为原有效支持域的 C 冲突覆盖，不是全网格自交率。LOW_SUPPORT/NA 不记成零。
- 图审沿用实际历史 AI_PROXY 标签，不是本轮新看图或独立人评。“三轴交集”通常指同一输出的 C 合理采用、整体合理、主要支撑存在；不同时认证无交、无碎片或合法索引接续。
- 不把不同父/噪声/阶段的分数拼成排行榜，不把同一缓存跨轮图审的标签漂移算成模型改进。

入口：[报告索引及小文件 SHA256](report_inventory.csv)、[索引范围](inventory_scope.json)、[历史机制逐条表](historical_route_summary.csv)、[历史原登记](../../r2/docs/research_route_registry.json)、[指标政策](../../native_t1/docs/control_metrics.md)。

## 3. Chair：真正有证据的训练顺序

```text
官方 Chair 生成器 EMA
  └─ EXP67：fresh AdamW，USER 条件训练 6000，global6000
      └─ FULL：完整座面 Native 续训 6000，global12000
          └─ SEAM_C200：4 TRAIN 条件，200，global12200
              ├─ LONG：同4条件再800，global13000
              │          └─ 保留其 C-only occupancy head 供下面推理使用
              └─ B_POOL47：扩至47父，再800，global13000
                  └─ PLAIN_S2000：同47父，再2000，global15000
```

R2 是所用模型/优化配方来源，**不是这条权重链必须先继承的历史 R2 checkpoint**。EXP67 从官方初始化开始。LONG 与 B_POOL47 是并行分叉；PLAIN 不是从 LONG 生成器继续训练。

S2000是原PLAIN3000轨迹中保存的step2000，之后经过开发比较与用户选择保留；不是原3000步实验预登记的唯一主终点。原Wave1完整晋级门未全部通过，不因后来保留一个有用设置而倒改历史。

### 3.1 PLAIN 的含义必须纠正

`PLAIN_S2000` **仍是 SEAM 表示**：条件开放边上的固定角点身份、SEAM 自由第三点、bulk 与受限匹配仍在。PLAIN 仅表示这条臂没有加 couple 辅助损失。它的生成器累计条件更新为15000，其中 SEAM 阶段累计3000；不是从官方只训2000步。

| 配置 | 实际做法 |
|---|---|
| 原条件骨架 | Native/T1＋role＋Geo；原 C 硬保持 |
| 普通阶段 | free-only FM；先 free OT 匹配，再 HYBRID0.25 |
| SEAM 阶段 | source-owner 布局用于训练，受限 OT；固定边角点不进入自由坐标损失分母，第三点及 bulk 可动坐标监督 |
| 数据 | USER 完整座面，最后47 TRAIN父；当前重点应保留规则、连贯、K≥2 输入合同 |
| 优化 | 沿用匹配 AdamW 状态与 RNG；batch8、microbatch1、lr1e-5、betas(0.9,0.95)、wd0、clip1 |
| 精度/采样 | BF16 前向、FP32 参数与积分；固定 Euler50、标准 Gaussian free 起点 |
| 推理布局 | 固定 LONG 的833参数 C-only MLP，logit≥0；不能替换成来源 oracle 后继续称部署成绩 |
| PLAIN 本段曝光 | 2000步×8＝16000实例，47父各338/340/342次；加上此前各阶段历史，不能直接用本段曝光比较所有模型 |

这里保留的是**产生有限正结果的实际训练历史**，并未独立证明每个阶段都不可删、这是最省预算的重训方法，或相同步数必然适合其他类别。

生成主干为约1.304亿参数的全参数适配，不是只训练833参数小头；共同骨架为12层、hidden768、12 heads。官方EMA用于最初初始化，后续恢复的是live参数和配套优化器；不将另一份EMA与不匹配动量拼接。保持原FM、无训练EMA/学习率scheduler与原微批归一化，是这些已测配方的实际组成。

### 3.2 哪些正结果是实测成立的

最初 SEAM200 的同起点同预算对照，在4熟悉条件×4噪声上：合法 C/F 接口覆盖13/16→16/16，合法边131→175；13/16进入 bulk，11/16的 bulk 可达≥30%。CF3.751%→1.704%，但 FF12.841%→14.563%、I0.671%→3.302%。因此它建立了真实接续能力，同时暴露几何质量成本；不能只复述前半句。[SEAM200](../../experiments/r2_seam_slot_c200_v1/REPORT.md)

成熟 PLAIN_S2000 固定28请求的后处理配对如下。这里比较的是**同一个生成器的后端**，不是重新训练收益：

| 同28请求指标 | PLAIN 原 POST | ＋HINGE_OPP015 |
|---|---:|---:|
| FF | 14.3104% | 13.4902% |
| CF | 2.9116% | 2.7878% |
| I | 1.6863% | 1.6846% |
| 单面索引孤岛总数（该报告未滤非法边口径） | 150 | 121 |
| 合法共边总数 | 4624 | 4729 |
| TRAIN bulk可达≥30% | 13/16 | 14/16 |
| DEV bulk可达≥30% | 0/12 | 0/12 |

原合法边全保留，零新增非法面对、零删原面，是本面板明确的工程正结果；仍有个别 I/连续形状量小幅变差，不能称所有指标逐例零退化。[HINGE反向守卫](../../experiments/r2_hinge_opposite_v1/REPORT.md)

后续 `ALPHA_HOLE` 在这个 HINGE 基础上，固定原面分母的 FF13.4902%→13.0463%，增加合法连接；608/n2接入原bulk面积28.82%，值得保留。其实际 FF12.9955%还包含16张新桥面的分母效应，I1.68457%→1.69678%，DEV≥30%仍0/12。它是可选归档版本，不把补面全部算成旧自交消除。[ALPHA/HOLE](../../experiments/r2_hinge_v2_cpu_v1/REPORT.md)

Chair BACK 的 `MULTIPART_S1500` 另有正结果，属于从 S1000 分出的多部位分支；不能把它直接接在上述 S2000 后面写成同一权重。其完整谱系、头结构和证据见[Chair专项整理](chair_evidence.md)。

同BASE_POST对比，BACK熟悉条件的FF21.455%→18.965%、I18.475%→6.409%、平均bulk可达2.072%→44.812%、bulk可达≥30%从0→5/8；所以不是全部收益来自新增后端。BACK DEV的I24.042%→16.992%，但平均bulk可达14.390%→9.604%、≥30%仍1/8，不能包装成广泛迁移。此分支同时改了部位数据、头和预算，不是单独验证其中一项。

## 4. Bench：最有希望的是数据/曝光驱动的 Native 主线

### 4.1 两个 B 和几条分叉不要再混用

```text
官方 Bench → COMMON12000
  └─ 完整座面 Native6000 → global18000（22父）
      ├─ SEAM3000 → global21000（保留的历史研究参考）
      └─ Coverage1500 → global19500（22旧父＋a8/27，共24父）
          └─ 曝光 B_75_25 +1500 → global21000（下称 P0）
              ├─ CONTROL_OLD +1500 → global22500（24父）
              ├─ 窄 B / NATURAL_HIGH +1500 → global22500（26父，HIGH2父）
              │    └─ 原版 SEAM +1500 → global24000（未整体晋级）
              ├─ W / WIDE_HIGH +1500 → global22500（30父，HIGH6父）
              └─ M / W_MATCHED +1500 → global22500（同W池，改组MSE系数）
                   └─ EXPOSURE +3000 → global25500
                       ├─ PLAIN200 +200 → global25700（未晋级）
                       └─ FLIP200  +200 → global25700（未晋级）
```

W/M 并不从窄 B 的22500终点继续训练；它们与窄 B 共享 P0 起点。M 当时没胜过 B，并不意味着其后增加3000步的 EXPOSURE没有新增证据。反过来，EXPOSURE的正结果也不能倒推 M 的组权重修改单独有效。

### 4.2 当前值得保留的具体训练配方

| 项目 | 窄 B | EXPOSURE |
|---|---|---|
| 表示 | Native/T1＋role＋Geo，无SEAM | 同左 |
| 条件任务 | 完整、连贯、人类式座面；C/Y来自同一合格目标版本 | 继承M已有30父池；本段未增数据 |
| 父池 | 26＝OLD22＋FOCUS2＋HIGH2 | 30＝OLD22＋FOCUS2＋HIGH6 |
| N | 独立运行合同128..799；实际训练128..384 | 同合同，实际最高776 |
| 曝光 | OLD75%、FOCUS12.5%、HIGH12.5% | 同样实例份额；续训3000步HIGH六父各追加500次，含M阶段累计各750次 |
| 组内采样 | 按父组织；每步3 OLD＋1交替FOCUS/HIGH父 | 沿用已冻结流组织 |
| 增强 | 每父CLEAN与X_WIDTH U[0.9,1.1]各一例 | 同左；不加入本轮未晋级的200步内部翻边增强或联合平移 |
| 损失 | 原整batch有效free坐标平均 FM | 继承M的固定组MSE系数版本，无新增几何辅助loss |
| 噪声/配对 | free OT 后 HYBRID0.25 | 同左 |
| 优化 | batch8/micro1、lr1e-5、AdamW(0.9,0.95)、wd0、clip1，匹配live/optimizer/RNG续训 | 同左 |
| 累计步数 | global22500＝COMMON12000＋Native10500 | global25500＝COMMON12000＋Native13500 |

**自然 N 不等于统一把所有输入设成799，也不等于“更多面总是更好”。** 训练中实际准入的是合格目标的自然面数。低面数版本不合格就不强造；这几轮没有证明多LOD混合有效。N800不在当前已核验合同中。

**75/12.5/12.5是实例曝光比例，不是loss或梯度比例。** 窄 B 的目标是 `ΣSSE / Σ有效free标量数`（登记名 `whole_effective_batch_free_coordinates`），HIGH占12.5%实例却占21.418%累计自由坐标。M/EXPOSURE的HIGH步使用固定窄池参考：`rho=m_H,B/m_H,W`，`L=(SSE_OLD+rho*SSE_HIGH)/(m_OLD+m_H,B)`；FOCUS步不变，登记名为 `narrow_reference_group_MSE_coefficients_v1`。这是实际继承设置，尚无证据证明它是通用最优归一化。[自然N报告](../../experiments/r2_bench_natural_n_recipe_v1/REPORT.md) · [M的公式与对照](../../experiments/r2_bench_group_weight_match_v1/REPORT.md)

### 4.3 为什么保留窄 B

在同38请求与等预算 CONTROL_OLD 对照中，新增两HIGH父的RAW三轴交集0→4/4，两父两个噪声都有收益；既有OLD的FINAL FF10.385%→9.182%，DEV FF21.137%→11.494%。这是数据方案能改变真实生成的证据。

代价同时存在：相对CONTROL_OLD，DEV CF5.872%→6.039%、I10.376%→11.719%；相对冻结起点P0，更是CF2.939%→6.039%、I1.044%→11.719%。新增HIGH的FINAL三轴仍0/4，面积与支撑明显损失，I为NA而不是零。自然N同时改变了独立父、条件与曝光，不能单独归功于N。

后续宽池W、组权重M均未整体替代它，所以它仍是必要的较稳参照。不能只因EXPOSURE总体FF更低，就把窄B及其好例删除。

### 4.4 为什么把 EXPOSURE 作为前端优先候选，而不称整链冠军

下列为同46请求、同后端、M→EXPOSURE的正式配对：

| 指标 | M | EXPOSURE |
|---|---:|---:|
| RAW三轴交集（本轮AI_PROXY） | 29/46 | 33/46 |
| FINAL FF，父等权 | 14.260% | 9.044% |
| FINAL CF，父等权 | 7.602% | 7.350% |
| FINAL I，共同19个PASS父 | 6.755% | 4.828% |
| FINAL有合法CF的输出 | 14/46 | 19/46 |
| FINAL三轴交集（本轮AI_PROXY） | 17/46 | 15/46 |
| DEV FINAL CF | 5.694% | 8.013% |
| DEV FINAL I | 11.237% | 12.744% |

RAW新增5份、丢1份，其中两个DEV父各一份噪声改善；不是稳定跨噪声泛化证明。FINAL没有新增三轴交集，反而丢2份，DEV仍0/8。因此可以说**继续训练改善了部分生成能力与全组某些几何量，后端损伤和DEV代价没有解决**。[四方向正式报告](../../experiments/r2_bench_four_routes_v1/REPORT.md)

原版 SEAM 接在窄 B 上的另一个1500步候选，使开放C合法接口覆盖42/42，但FINAL FF13.056%→18.994%、I4.247%→7.588%，bulk可达≥30%仅5/42、DEV0/8。这直接限制了“Chair成功，所以Bench现在也应一律加SEAM”的建议。

最新同物理C翻边快测未推翻上述选择：原6请求FINAL FF为EXPOSURE12.414%、普通200步18.808%、翻边200步23.099%；翻边虽减少单面孤岛，仍出现主体错置和旧采用丢失。两新端点不晋级。[翻边快测](../../experiments/r2_bench_c_edgeflip_v1/REPORT.md)

## 5. 已做路线如何收口：保留贡献，避免换名重跑

| 实验系列 | 已获得的结果 | 当前处理 |
|---|---|---|
| 早期注入、Q/G、上下文读取、有限表面/MOMENT、READOUT | 条件响应/激活可存在，但未形成稳定整体升级；部分因精度或成本门阻塞 | 不把梯度非零当成功；门阻塞不冒充训练失败 |
| Native/T1、role、Geo、HYBRID/R2 | 建立条件任务、硬保持与可用适配基线 | 保留共同骨架；零出口不意味着有C时等同无条件前向 |
| B辅助表征日程、CAPPED/梯度预算、早期FM加权 | 个别距离/采用/质量改善伴随代价，无稳定通用替代 | 不继续凭总FM或单张好图扫日程 |
| 真实USER、32→67父、完整座面长训 | 部分覆盖与采用改善；FULL形成足训练参照 | 真实部件输入与独立父覆盖值得保留；数据仍不是唯一已证根因 |
| CREF、完整K≥2、边界归因、局部几何读取 | 排除部分“输入太难/只是高度”解释；局部定位收益未稳定解决轮廓与接续 | 保留诊断，未晋级的模块不自动加入主配方 |
| 边界/partner/锚定、终态硬绑、RBF/BLEND | RBF有少量真实局部连接；普遍吸附受错位/尺度/冲突限制 | 有限正结果与源绕序工程修正分开；不放大半径冒充学习 |
| NFT、SELF_REFLOW、窗口策略RL | 当前奖励配置没有稳定增加严格联合成功，部分共边长度或质量变差 | 两版已收口，不延长同配方；不等于所有RL无效 |
| SEAM200分叉LONG与B_POOL47→PLAIN；A/B/C与couple | SEAM有真实主体接续；PLAIN推理借LONG头；额外couple/头校准并未形成普适升级 | Chair保留PLAIN；source oracle只作诊断，不替代C-only成绩 |
| 单次D0、来源定向、HINGE_OPP、ALPHA/HOLE | 去交/导出正确性/局部连接有可复算正结果 | 维持版本和阶段，保留微小代价；不认证普遍水密 |
| 去重、匹配N、法向门禁、容量保护 | 清理能减部分冲突，但删面/覆盖/接口有代价；角度门会误伤合法来源 | 不将“重复/双面薄壁”全部等同废数据；不叠加未经验证的保护loss |
| Bench COMMON、多部位SEAM、低LR/官方短重启、PERT/ASYNC | 条件能学习，但未整体获得泛化与bulk接续 | 不宣称低LR或短重启已解释根因 |
| Bench USERv2/RELATION、新父27、SHIFT | 真实条件有局部学习；SHIFT质量代价与跨父门不足 | SHIFT交叉未运行；不是交叉训练失败。旧文件证据保留 |
| Bench完整座面Native→SEAM、Coverage、曝光75/25、自然N | 当前最清楚的前端正证据：学会一些完整结构，保留旧任务曝光但仍有性能取舍 | 作为数据与训练顺序主线；不同阶段不盲目合并 |
| Bench扩宽W、组系数M、EXPOSURE | W/M短预算不升级；M延长曝光后有新增RAW能力与全组FF收益 | EXPOSURE保留前端候选，窄B继续作参照；不宣称收敛已充分 |
| Bench GEO_ONLY、SHAPE500、RELDEF500、真实RAW修复 | 绕过D0能保支撑但FF大涨；SHAPE有有限视觉收益及质量代价；RELDEF不升级；真实修复门未通过 | 无已验证统一替代D0后端；修复未训练不写成训练负结果 |
| Bench HYBRID/BESTOF4、区域ASSIST/AUTO、LOCAL_SHAPE | 辅助已知选区可保支撑，但带回RAW冲突；自动选区与微拟合未升级 | 不把人工已知选区成绩写成自动全46输出成绩；未测EXPOSURE组合不拼接收益 |
| Bench同物理C细分/翻边与200步增强 | 确认划分敏感，局部稳定有正例；未减少整体碎面/质量代价 | 保留EXPOSURE；PLAIN200/FLIP200不晋级 |

逐机制原证据见[历史路线记忆](../../r2/docs/research_route_memory.md)，最近Bench逐项数据和权重见[Bench专项整理](bench_frontend_evidence.md)。本表将诊断、数据制作、门禁和实际训练分开，不把所有报告均记成失败的模型候选。

## 6. 现在能判断什么，不能再简单归因什么

1. **模型确实能用C，问题是采用、完整主体、真实接续和几何质量不能总在同一输出上同时成立。** Chair SEAM是关键正证据，Bench多次真实训练也有RAW采用增益；“完全没学到”不符合记录。
2. **碎片有两个来源。** RAW已经错层、主体错置或相交时，后端不能替代正确生成；RAW本来有的细支撑又可能在D0/GEO中收缩、删除。不能把两者统一叫一个loss问题。
3. **FT破坏了直线flow不是已证根因。** 训练直线桥的定义没有因权重更新改变；Coverage诊断未见普遍轨迹恶化。自由rollout仍可能不够好，但不据此自动换OT或求解器。[flow核查](../../experiments/r2_bench_flow_integrity_diagnostic_v1/REPORT.md)
4. **仅“多数据/多面/多步”都不是已证通用解法。** 扩父会减少每父曝光，高N会改变loss份额，更多训练可能改善一个面板却损坏其他请求。应同时记录独立父、条件数、实际N、每父曝光与loss分母。
5. **低FF可以伴随缺面；孤岛更少也可以伴随主体崩坏。** 必须同时保留绝对非法面对、有效free面积/面数、删面原因、支撑和同位置截面，不仅看百分比。
6. **数据清洁与部署合同仍重要，但已有数值不能证明它们是全部根因。** 来源owner与生产C-only布局的模型可见差异确有必要披露；来源完成不是开放生成唯一真值，不能用DEV来源标签替部署预测。

## 7. 后续若沿当前证据选训练方式

这不是本次开训指令，而是对当前配方的选择结论：

- **Chair：保留SEAM表示与PLAIN路线。** 用它作为方法主体，不重新堆B、RL、边界吸引或盲目硬角度门禁。BACK分支单列，不冒充同权重跨所有部位。
- **Bench：优先保留EXPOSURE继续前端研究的资格，同时固定窄B为必要比较。** 任务保持完整人类式C、同版目标、合格自然N、按父曝光与旧任务回放；不将最新翻边/局部混合候选混入“已成功配方”。
- **后处理独立验收。** 当前D0有大幅去交作用也有细结构损伤。已知选区恢复不能直接铺成统一自动流程；尚未实测的EXPOSURE＋自动区域恢复不得预写收益。
- **判断训练是否值得继续，看真实生成学习曲线和每父曝光，而非只看总FM或固定200步。** 但EXPOSURE之后普通200步也退化了部分请求，所以这不是无限续训建议；新预算仍需固定终点与评价。

真正应争取的下一项升级，是同一请求的RAW采用/主体/支撑改善能保留到FINAL，并且DEV冲突不恶化。当前没有足够证据承诺改某一个模块就能做到。

## 8. 文件与保留边界

| 资产 | 当前位置 |
|---|---|
| Chair PLAIN完整状态 | `experiments/r2_seam_pool47_couple_v1/checkpoints/PLAIN3000_step2000.pt` |
| Chair PLAIN推理生成器 | `experiments/r2_seam_quality_wave1_wave2_v1/inference/PLAIN_S2000_generator.pt` |
| Chair固定LONG头 | `experiments/r2_seam_quality_four_v1/inference/LONG_occupancy.pt` |
| Bench窄B完整状态 | `experiments/r2_bench_natural_n_recipe_v1/training/NATURAL_HIGH/checkpoint_step1500.pt` |
| Bench EXPOSURE完整状态 | `experiments/r2_bench_four_routes_v1/exposure/training/M_PLUS3000/checkpoint_step3000.pt` |
| Bench EXPOSURE推理生成器 | `experiments/r2_bench_four_routes_v1/exposure/training/M_PLUS3000/generator_step3000.pt` |

本次核对文件存在与大小，不加载权重，不重新哈希大权重；已登记哈希参照[10/28保留终点](../weight_cleanup_20261028/retained_endpoints.csv)、[10/25保留模型](../storage_cleanup_20261025/RETAINED_MODELS.md)。历史中间点已删除时，不恢复或改写原报告。维护默认、Git和原资产均不变。

本次交付：本文件；[Chair详细配方](chair_evidence.md)；[Bench详细配方](bench_frontend_evidence.md)；[历史机制表](historical_route_summary.csv)；[现行报告索引](report_inventory.csv)。
