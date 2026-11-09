# MeshFlowCtrl：研究问题、已验证正结果与云端交接

整理日期：2026-11-01（Asia/Shanghai）。最新纳入实验：2026-10-30完成的 R2_BENCH_LAYOUT_ORACLE_V1。

这是一份基于本地既有报告、方法代码说明和保留资产记录的交接，不是新实验。本次没有训练、模型前向、采样、重新评分或重新图审。下面的图审数字沿用原实验实际 AI_PROXY 记录；不冒充独立人类评审。引用编号对应文末本地证据索引，另附小文件 SHA256 清单。

## 1. 先读这一页：我们目前走到了哪里

**目标：给定用户真实三角面 C，保持其几何与已声明的连接身份，生成围绕 C 合理组织的完整网格。输入要成为成品的组成部分，而不只是被复制进输出。**

目前已有三类不同的正结果：

1. **Chair 的共享原边 SEAM 表示有真实接续收益。** 它让生成从一部分确定的 C 边上生长，并在熟悉条件中实际接入较大 bulk 主体；不是仅仅点距变小。
2. **Chair 的 HINGE_OPP015，以及后续 ALPHA/HOLE，有可核账的局部后处理收益。** 在固定面板上增加合法共边、减少部分自交与单面孤岛，但没有解决全部未见父的主体接续。
3. **Bench 的人类式条件、自然面数准入、结构覆盖与曝光训练有生成侧正信号。** 当前保留 EXPOSURE 为生成器研究参照；它能学到部分新的 C—主体关系，但固定后端仍会毁掉一些已经长出的支撑。

**我们还没有得到一个同时在 Chair、Bench、熟悉条件和未见父上全面胜出的统一完整流程。** “本地优先保留”不等于与外部论文统一评测后的 SOTA。

当前最关键的两个问题：

- **生成端：** 在未见结构、小 K 或某些噪声下，灰色主体另造座面、包住/绕过蓝色 C；有时 C 侧接好了，却没接入主体。
- **后处理端：** RAW 里已经合理存在的腿、端框和薄支撑，在 D0/GEO 后变成细丝、缺口或残片。降低 FF 可能伴随删除了有用结构。

最新粗布局试验也已完成：**两臂各1000步，来源布局没有新增可辨别的采用—完整—支撑联合成功，因此没有启动 C→布局预测头。** 这约束的是本版“小粗框读取分支”，不是证明所有结构化生成表示无效。[S01]

## 2. 任务合同：不能在后续讨论中悄悄改掉

输入是 C 的真实顶点坐标、三角面连接/绕序及必要元数据，加显式总面数 N 和随机噪声。常规推理不提供未知完整来源模型 Y、不提供来源 owner 标签或语义部位名称。

训练可从完整 M 构造 C 和补集 Y，利用 Y 做监督和匹配；但来源 M 只是一个合理完成，**开放生成不要求复刻它的唯一款式**。同一 C 可以有多个合理完成。

- 优先规整、连贯、人实际可能输入的部位/组合；通常 K≥2。不是继续随机抽几张悬空三角面凑数。
- 完整座面、靠背、支撑部件、组合输入都可以有意义；不能强制所有对象都是平板座面或四腿拓扑。
- C 精确保留是合同，不是控制成功证明。还要看 C 是否被合理采用、主体是否完整，以及任务确实需要的接续。
- 不能默认所有 C 开放边都必须封闭；闭合 C 的开放边接续指标应为 NA。
- 原始 RAW、单次 D0/GEO 的 BASE_POST、完整后端 FINAL 必须分开。
- DEV 长期参与开发；“后训练未见父”也不保证官方预训练未见。论文独立测试尚需明确建立。
- 保留完整指标和坏尾部；不只挑有利的三个分数，也不以来源重建误差代替开放生成质量。[S02]

## 3. 在原 MeshFlow 上到底增加了什么

### 3.1 原先验与条件接口

沿用原 MeshFlow 的 EquiDiT 三角汤生成主干。当前 Native 条件模型约 **130,401,155 个参数（约1.304亿）**：不是5.2亿参数；约5.2亿字节是 FP32 权重文件量级。主干12层、hidden768、12 heads。官方生成器和官方 D0 是两个模型；Chair D0 约4.56亿参数，不能与生成器大小混同。[S03,S04]

增加的基础机制：

| 部分 | 实际作用 | 不应夸大的地方 |
|---|---|---|
| Native/T1 | 将干净 C 与当前 free 状态一起输入主干；C 使用终点时间1，free 使用 t | 不保证新建条件接口的前向与原无条件模型逐位等价 |
| known/free role | 区分输入条件与待生成变量 | 只是角色信息，不是用途语义标签 |
| Geo | 读取 C 的13维局部几何特征，零出口残差注入；再经原注意力传递 | 现行基础 Geo 使用 P=I，不是已实现完整邻接图传播 |
| free-only FM | 只监督真实可动 free 变量 | 总 FM 降低不保证局部接口或自由生成改善 |
| C 硬保持 | Euler 每步保持输入 C；导出保存/恢复其来源映射 | 保坐标不自动证明原索引拓扑、水密或用途合理 |

这些主线是**全参数条件后训练**，不是仅训练一个小头、冻结全部生成器。

### 3.2 R2 的含义

历史 R2 是“20%/40% 条件混合＋全程 HYBRID0.25”的官方初始化后训练配方。后来的 USER/FULL/Bench 继承其模型和优化算术，但已更换任务数据，不能都声称仍在训练原 C20/C40 池。

普通 Native 配对先把 Y 匹配到固定 free 噪声槽位，再混噪：

    z_H = sqrt(0.75)·z + 0.5·epsilon
    F_t = (1-t)·z_H + t·Y_OT
    U = Y_OT - z_H

训练起点混噪发生在 OT 之后，不重匹配；推理仍是标准高斯、Euler50。常用设置 batch8/microbatch1、AdamW lr=1e-5、betas=(0.9,0.95)、wd=0、clip=1，BF16前向、FP32参数/积分，严格确定性。后期高 N 分支另明确固定 OLD/FOCUS/HIGH 的组 FM 系数。

“HYBRID0.25 训练噪声”与后来“RAW/D0 后端 HYBRID 混合”是两件完全不同的事。[S03,S05]

### 3.3 SEAM：目前最明确的方法正信号

对 C 真实开放边建立候选槽：

- active seam 面前两个角引用对应 C 边的反向端点，只生成第三个角；
- inactive 候选槽继续作为普通 bulk 三角面，不删槽、不增加总 N；
- 训练来源 owner 面固定到对应 seam，剩余 bulk 目标做受限 OT，然后 HYBRID；
- 固定角不计入 FM 自由标量分母。若 S 个单边 seam 活跃，分母是 9(N−K)−6S；
- 推理时由 C-only 布局规则/预测头决定激活，不得拿来源 oracle 偷换部署成绩；
- Euler 和 D0 后恢复锚角，最终索引解码核验真正共享 C 顶点身份。

**C—seam 底边共享属于表示保证；真正额外的能力证据是 seam 能否沿合法 free 共边连接到 bulk。** 共享底边不等于大主体已经接好，owner/alias/Null 多解性仍要明确。[S06]

### 3.4 后端

常用研究链：

    RAW
      → 单次官方类别 D0
      → 恢复 C（SEAM 还恢复其锚角）
      → GEO_C
      → 恢复用户 C 原绕序、固定 C 整理 free 方向
      → HINGE_OPP015
      → 可选 ALPHA
      → 可选 HOLE
      → FINAL

GEO_C 做索引化合并及退化/重复清理；来源方向恢复是已验证的工程修正。HINGE/ALPHA/HOLE 是独立版本的几何后处理，默认 CLI 不会因为研究结果自动启用。后端需要簇、槽、来源身份等元数据，不是对任意 PLY 无条件适用。[S04,S07]

## 4. 当前保留的模型与准确训练谱系

### Chair 研究主线

    官方 Chair EMA
      → EXP67：USER适配6000（global6000）
      → FULL：47父完整座面 Native再6000（global12000）
      → SEAM_C200：4个熟悉条件再200（global12200）
         ├─ LONG：4条件再800（global13000），保留其 C-only 头
         └─ B_POOL47：47父再800（global13000）
              → PLAIN_S2000：47父再2000（global15000）

**PLAIN_S2000 仍然是 SEAM。** PLAIN 只表示没有 couple 辅助损失；其 SEAM 累计更新3000，全部条件更新15000。它的生成器不是从 LONG 生成器接着训练，推理却配用 LONG 头。

S2000 是原3000步轨迹中的保存点，后经开发比较与用户选择保留；不是原实验预登记的唯一终点，论文中不能掩盖这项选择历史。

### Bench 研究主线

    官方 Bench → COMMON12000
      → 22父完整座面 Native6000（global18000）
         ├─ SEAM3000（global21000）：历史研究参照
         └─ Coverage1500：22旧父＋a8/27，共24父（global19500）
              → 曝光 B_75_25 再1500（global21000；后称P0）
                 ├─ 窄 B / NATURAL_HIGH：26父，再1500（global22500）
                 │    → SEAM1500（global24000；未整体晋级）
                 ├─ W / WIDE_HIGH：30父，再1500（global22500）
                 └─ M / W_MATCHED：30父、组FM权重匹配，再1500（global22500）
                      → EXPOSURE / M_PLUS3000（global25500）

B、W、M 是并行分叉，**不是 B→W→M 串行训练**。Bench 当前优先保留的 EXPOSURE 是 Native，不是 SEAM；累计 Native 适配13500步，之前还有 COMMON12000。

EXPOSURE 当前实际30父/30条物理 C，N128—776、K2—84，模型合同上限799。自然 N 表示保留合格完整体需要的面数，不意味着给任意坏例多发面槽就会变好。OLD22/FOCUS2/HIGH6 曝光比例75%/12.5%/12.5%；不是30父均匀。新增几何版本或多个 C 不等于新增独立父。

Chair 和 Bench 是各自官方类别先验出发的适配，**不是一个共享权重模型已经完成跨类别零样本泛化**。同一方法框架可以共用，但当前预算、历史与覆盖并不完全一致。[S08,S09]

### 用途区分

| 用途 | 保留设置 | 地位 |
|---|---|---|
| 维护 CLI 默认 | Chair HYBRID START | 历史维护合同；不是最新实验冠军 |
| Chair 接续研究 | PLAIN_S2000＋LONG头＋单次D0/GEO/来源定向＋HINGE_OPP015 | 最清楚的限定任务正结果 |
| Chair 可选进一步整理 | 上述流程＋ALPHA/HOLE | 局部收益，保留桥面与代价账本 |
| Chair 跨部位研究 | MULTIPART_S1500 BACK分支 | 熟悉BACK正信号，另一个训练分叉 |
| Bench 生成器研究 | EXPOSURE / M_PLUS3000 | RAW有新增能力，不是完整流程全面晋级 |
| Bench 重要对照 | 窄池B、Native6000、SEAM3000 | 避免只和弱/错误历史对照比较 |
| 最新粗布局 | LAYOUT1000 | 不晋级，不训练下一阶段预测头 |

## 5. 正结果：哪些已被真实实验支持

### 5.1 Chair

| 实验与比较 | 实际正结果 | 必须同时保留的限制 |
|---|---|---|
| FULL完整座面学习曲线，24请求 | RAW采用12→16/24；有限边距离4.762→3.282%L | 采用与新增接口不重合；DEV FF17.939→25.855% |
| SEAM_C200 vs等预算普通续训，4熟悉C×4噪声 | 合法接口13→16/16；合法边131→175；13/16进入bulk，11/16 bulk可达≥30% | FF12.841→14.563%、I0.671→3.302%；不是DEV泛化 |
| 成熟PLAIN_S2000，28请求 | TRAIN主体/采用各16/16，bulk≥30%为13/16；DEV主体12/12、采用11/12 | DEV bulk≥30%仍0/12；可辨认用途不等于实际接续 |
| 同S2000＋HINGE_OPP015 | FF14.3104→13.4902%、CF2.9116→2.7878%；单面索引孤岛150→121；合法边4624→4729 | 原合法边全保留、新非法面对/删面均0，但个别I/形状量仍有小代价；DEV仍0/12 |
| HINGE再加ALPHA/HOLE | 固定原面分母FF13.4902→13.0463%；新增111合法边；608/n2首次接入38张原bulk面、28.82%面积 | 实际FF12.9955%还含16张新桥分母影响；I1.68457→1.69678%；DEV≥30%仍0/12 |
| BACK多部位适配 | 同BASE熟悉BACK bulk平均2.072→44.812%，≥30%0→5/8；I18.475→6.409%、FF21.455→18.965% | 未见BACK平均可达14.390→9.604%；数据/头/预算一起变，不是单因素归因 |

[S10—S15]。这些是不同实验各自的直接配对，不能把其最好数字拼成一个从未运行的模型。

HINGE 的1.5%L是累积位移限制；它保护已有合法关系，并要求新双面共边反向绕序。ALPHA 在有限位置中选合格合并；HOLE 仅补实际索引围成且满足合同的三边环，新增面标 BRIDGE。这些机制支持“局部已有几何可被整理成更好的连接”，不支持“能修复宏观错误主体”。

来源 C 方向恢复也应保留：旧RBF的9条已有合法接口得以全部一致定向，但没有新增连接，也没有改变FF/I。工程修正与RBF算法收益必须分开。[S07]

### 5.2 Bench：EXPOSURE是当前最值得保留的生成侧进展

在到达EXPOSURE之前，以下几项也有独立价值，不能因最终未全面晋级而抹掉：

| 方案 | 同轮正信号 | 限制 |
|---|---|---|
| Chair式完整座面 Native→SEAM 顺序 | 同24请求的TRAIN RAW采用：COMMON/Native/SEAM为2/14/15（各16份）；FINAL FF34.58→14.57→10.25% | DEV没有稳定升级；TRAIN最终整体合理反而14→8→9/16，后端损伤保留 |
| Coverage新增a8/27 | 两个新增父RAW三轴2→4/4，新增均来自a8 | 同组FINAL3→0/4；没有等预算旧池臂，不能单独归因扩父 |
| 曝光B_75_25对A_50_50 | 旧父RAW三轴12→14/16；DEV FINAL FF21.683→11.783%，4父均下降 | 不是所有组都改善；FOCUS FINAL仍0/4 |
| 自然N窄池B对同预算旧合同对照 | 新两个高N父、四请求RAW三轴0→4/4 | HIGH FINAL仍0/4，高N留出未出现新增采用；任务数据方案一起变，不是N单因素因果证据 |

[S28—S31]。它们共同支持“有效任务与曝光能学入新的关系”，没有证明照搬Chair相同步数就必然泛化。

同23父、46请求、同后端，冻结 M 与 M再训3000步的 EXPOSURE 比较：

| 指标 | M | EXPOSURE |
|---|---:|---:|
| FINAL CF，23父等权 | 7.602% | 7.350% |
| FINAL FF，23父等权 | 14.260% | 9.044% |
| FINAL I，共同PASS19父 | 6.755% | 4.828% |
| FINAL有合法C/F接口的请求 | 14/46 | 19/46 |
| RAW视觉三轴联合 | 29/46 | 33/46 |
| FINAL视觉三轴联合 | 17/46 | 15/46 |

三轴是同一输出的“整体合理∩C合理采用∩主支撑存在”，不是无交、水密或真实共边的替代指标。

RAW新增5、丢失1；包括两个DEV父各一份噪声的采用改善。DEV RAW联合0→2/8，FINAL仍0/8；DEV CF5.694→8.013%、I11.237→12.744%仍有代价。

所以应说：**扩大到合格自然N结构后，给予足够曝光，确实能学到部分新的主体关系；这些收益尚未稳定兑现为最终网格。** 不能说数据扩展无效，也不能说已经解决泛化。[S09]

### 5.3 后端局部保支撑：存在可行组合，尚无自动通用方案

固定 M 的3父/6请求，零训练/零D0重跑：

| FINAL流程 | 支撑存在 | 三轴联合 | FF | I |
|---|---:|---:|---:|---:|
| 原D0完整后端 | 0/6 | 0/6 | 7.777% | 0.016% |
| ASSIST11：预先辅助指定两端区域回RAW | 6/6 | 4/6 | 13.677% | 0.130% |
| 全RAW走几何后端 | 6/6 | 5/6 | 27.901% | 2.100% |
| AUTO_COVER自动选区 | 0/6 | 0/6 | 9.888% | 0.016% |

ASSIST11仅23.81%的面回RAW。27父恢复支撑且FF7.143→1.818%；a8恢复支撑但有表面残缺；459仍错层，FF升到35.244%。

这是**辅助选区的局部可行性证据**：不是无人工知识自动算法，不是EXPOSURE＋该后端已经跑完，更不是严格物理上界。其原合法FF边丢23条、新增非法FF面对37对，必须保留。[S16]

## 6. 当前研究问题：事实与待验证解释

| 层次 | 已观察事实 | 还不能宣布的结论 |
|---|---|---|
| C的用途/主体组织 | 459/f495等RAW中灰座替代或包住蓝C；同C换噪声可一好一坏 | “就是注意力被小K稀释”“只是数据少”尚非唯一根因 |
| 接续 | SEAM能保证C侧底边，但可能只长小翻片；Bench新SEAM开放接口42/42，bulk≥30%仅5/42、DEV0/8 | “接口满分等于主体接好”“只校准占用头即可泛化”不成立 |
| bulk质量 | Chair早期SEAM非法FF343对中338对为bulk–bulk | 不能把所有FF归于seam直接内翻；也不能排除seam对生成的间接影响 |
| 训练数据 | 有重复/反向薄壁、交叉与部件缺失风险；独立结构覆盖很有限 | 不能把全部坏面都删掉便保证提升，或把每份薄壁副本判成无意义垃圾 |
| 后端 | 合理RAW支撑会在D0/GEO后缩成细丝或被删；FINAL低FF有时伴随结构消失 | “D0修好了一切，只是硬恢复C毁了它”“去掉D0就都好”均不受支持 |
| 表示敏感性 | 同物理C的三角划分、面数嵌入桶能改变轨迹；固定K/N的内部翻边也有反应 | 不代表简单增加C面数或做重三角增强就能修好 |
| 泛化 | 熟悉条件可学会，独立父/结构和噪声稳定性不足 | 训练loss下降、DEV差不自动证明经典过拟合；小模型/降lr也未解决 |
| 新条件信息的使用 | 本版来源粗框P对输出作用很弱，无新增联合成功 | 尚不能区分信息粗糙、读取弱、优化弱、预算不足；不能说“正确结构已用上仍无效” |

### 已经做过、值得停止重复猜测的诊断

1. **“后训练普遍破坏直线FM”没有获支持。** 生产OT/掩码/排列和线性桥自洽；Coverage在新增a8/27上真实轨迹更接近直行，共享角点误差与全局速度误差一起下降。旧6b有局部代价，df6未改善，结论只限该诊断。[S17]
2. **单一刚体恢复D0不能修复主要内部冲突。** Chair固定16份中1852个bulk–bulk非法面对在无约束D0、整体刚体、硬恢复三个阶段身份完全相同；224/224重复来源C顶点组被D0拆散，刚体不能精确合回。[S18]
3. **简单去重不是已经获证的整体解法。** 200步去重能降POST FF，但有更多删面、外壳破口；匹配N诊断未给出自动合流SEAM的充分依据。目标清洗应保几何任务，不能靠删结构获得低冲突。[S19]
4. **更多增强也不是已经成功的修复。** C内部翻边200步与多LOD1500步都真实跑过，均未整体晋级；见下一节。[S20,S21]

## 7. 近期实际试验的结论，避免重新从零提同一方案

| 路线 | 实际预算/性质 | 当前结论 |
|---|---|---|
| 固定C多LOD | 30父新增9份合格LOW；两臂各1500步；LOW实际13.30%曝光 | 相对PLAIN最终接口17→21/46，CF改善；FF8.913→9.273%、I2.506→6.344%，RAW三轴35→34、FINAL17→16；新增接口只在熟悉父 |
| C内部翻边 | 同K、N、物理表面及free目标；两臂各200步 | 有表示鲁棒局部信号，无稳定减碎面；原6请求FF18.808→23.099%，接口2→1/6 |
| 共享双面PATCH | 同EXPOSURE起点，200步，对PLAIN200 | 保证内部共边，但FINAL FF9.785→21.009%，方向可达12.032→4.281%；不是整个学习关系头的完整测试 |
| 训练式LOOP | 200步，对PLAIN200及近似等block的PLAIN267 | a8两噪声新增7条合法C/F边；对PLAIN267 FF10.662→13.409%，I7.373→2.360%，free面积12.560→9.213；局部信号，不全面晋级 |
| 局部RL16 | 16次机会×4轨迹，14更新/2跳过；对同随机过程RL_ZERO | FF/CF/I均小幅恶化；合法接口身份/条数不变 |
| 冻结官方motif guidance | 38请求确认；随后冻结内部loop小试 | G3 RAW/FINAL联合5/4，G1为7/6，EXPOSURE为25/15；冻结loop小面板未改善，没有有效教师可蒸馏 |
| 后端全局/局部混合 | HYBRID几乎全回RAW；ASSIST与AUTO分别核验 | 可保支撑但冲突代价高，自动选择尚无有效升级 |
| 历史NFT/窗口RL | 已有正式预算与冻结复验 | 没有稳定奖励独有增益；不等于所有RL无效 |

[S20—S24]。历史 R01—R30、两项训练前门禁和 R31 SEAM另有登记；这个计数是历史机制归类，不是最新所有实验总数，也不是30条方法被普遍证伪。[S25]

早期B表征对齐、CAPPED/梯度限制、READOUT、有限表面/MOMENT、边界/partner损失、局部锚定、denoiser微调/蒸馏/多步，以及后来时间权重、平移、曝光与后端修复，都应先查原报告的后续复验。不要把历史建议误当成尚未执行。

## 8. 最新结果：来源粗布局为什么没有放行预测头

云端小实验提出层次生成：

    p(F | C) = ∫ p(P | C) · p(F | C,P) dP

小实验的“物理部件参数Flow＋已知解码器”有正信号，但其部件数量、拓扑和参数含义已知，且不是本地真实MeshFlow生成器。因此本地先检验来源 P 能否帮助，再考虑学习 C→P。

实际本地实现：

- 两臂从EXPOSURE同live权重/AdamW分叉，各1000步；仍是30TRAIN父。
- P最多两个区域框：BACK_REGION、SUPPORT_REGION；相对C中心的框中心、跨度、类别；没有可靠区域标UNKNOWN。
- 111,424参数、64维4头cross-attention，主干前单处注入，零出口；新分支lr与主干同为1e-5，约20% P dropout。
- 只增加条件残差，**没有把自由几何重新参数化成部件变量，没有显式部件连接图或分部件几何解码器**。
- 四模式：旧EXPOSURE、PLAIN1000、LAYOUT1000、同LAYOUT权重但遮P的NULL。
- 固定6父12请求，36新RAW/36单次D0，144评分/实际图审单元，177字段；不改后端。

| 同12请求 | EXPOSURE | PLAIN1000 | LAYOUT1000 | LAYOUT_NULL |
|---|---:|---:|---:|---:|
| RAW三轴联合 | 6/12 | 6/12 | 6/12 | 6/12 |
| FINAL三轴联合 | 0/12 | 0/12 | 0/12 | 0/12 |
| FINAL FF | 8.438% | 12.154% | 10.797% | 11.696% |
| FINAL CF | 8.040% | 5.835% | 5.434% | 5.892% |
| FINAL I | 9.611% | 7.825% | 7.487% | 7.804% |
| FINAL平均free面数 | 64.250 | 61.333 | 59.083 | 62.917 |
| FINAL方向合法可达面积 | 14.687% | 13.921% | 13.177% | 13.346% |

四模式RAW六个成功身份相同，均来自熟悉父；P没有独有的新增采用。LAYOUT对NULL的RAW自由角点集合双向最近邻差异均值仅0.035190%L；这只是响应量，不是质量分数。训练FM曲线也近乎重合。

**结论：不晋级LAYOUT，不启动布局预测头。** 首先缺少“当前生成器能够有效利用P”的证据，训练更好的P预测器不能绕过这一缺口。

但不要将它写成“结构化变量已经完整验证失败”：本地小框残差分支与云端 PART6 的生成变量/解码方式不是同一种实现。“来源oracle”也只是诊断称谓，不是严格最优性能上界。[S01]

## 9. 评估该怎么读

| 量 | 本地含义 | 常见误读 |
|---|---|---|
| CF | 涉及非法C/F面对的free面比例 | 不是合法接触率；不是所有面对的概率 |
| FF | 涉及非法F/F穿插或正面积重叠的free面比例 | 不等于I，不是全网格流形认证 |
| I / ConflictCov_C | 固定C查询支持域中的穿越/近重层覆盖 | 不是整个网格自交率；LOW_SUPPORT不能补零 |
| A-VMR / PMR | 边界顶点与新free顶点匹配比例/全部匹配率，越高通常越好 | 近点或匹配全部顶点不保证整段共边或主体接合 |
| 合法共边 | 真正索引共边并满足该版本合法性/方向合同 | 存在一条边不意味着完整边界封合 |
| seam→bulk可达 | 从合法seam沿不经C的free图进入bulk，分母排除seam | 与包含seam的C可达free面积不同；30%不是通用质量标准 |
| 小角/孤岛/面积 | 退化、细片、连接与删面代价 | RAW独立角点soup的孤岛数不能直接当肉眼碎片数 |
| 视觉三轴联合 | 同一输出整体合理、C采用、主要支撑存在 | AI_PROXY描述汇总，不是无交/水密/人类认证 |

每项先按原合同聚合噪声/条件/父，不能把大父多噪声当更多独立数据。历史不同面板的成功计数不能直接前后相减；同一缓存在不同图审中也有标签分歧，必须保留。

来源GT距离可以做受控诊断，但开放生成的成功标准要看实际完成。后处理低CF/FF/I必须同时看保留面、原slot参考面积、实际面积、支撑完整性和合法关系损失。三轴正例也可能仍有碎片和相交。[S02,S09,S21]

## 10. 交给云端继续处理的具体问题

请以本地证据为前提，帮助选择**下一项可检验的机制变化**，不要直接罗列很多新模块或把小型合成正结果当真实Bench已成功。

1. **统一框架应该怎样划分职责？** Chair已验证的共享身份/受限匹配，Bench仍缺的主体布局，以及保结构后处理，怎样组成一套跨类别的原则，而不是每类堆一套技巧？哪些需要学习、哪些是输入/表示合同？
2. **新的结构变量怎样真正影响生成？** 如果继续布局方向，解释它与刚失败的“两粗框＋弱残差读取”有什么实质区别；是否需要改变生成变量/匹配/部件分工，而不是只再加一个预测头？不要默认当前分支已经有效使用了来源布局。
3. **前端与后端哪个改动能形成可解释增量？** RAW已错用C的案例和RAW正确、POST毁支撑的案例应分开；不能靠后端保留结构替代学习用途，也不能靠删掉表面降低冲突。
4. **怎样检验跨父迁移而不重复过去？** 说明最近的历史路线、已有反例、新假说、一个固定的小规模配对实验，以及什么观察会让它停止。
5. **若引用论文，请核验任务与实现边界。** 区分任意真实三角面条件、点云/草图条件、固定拓扑、已知部件参数、从头训练和后训练；区分公开结果与云端自制toy试验。不要把输入信息更强的方法当同任务胜负。

建议输出：一个优先方向＋一个备选；每项给机制/计算图、输入信息边界、与已测路线差别、最小对照和停止标准。当前未授权新增训练、清理、切默认或发布。

需要尊重用户偏好：论文主线是“如何在MeshFlow上加入有用的control”，希望跨类别方法统一；重视真实人类式输入、整体视觉和截面中的支撑/碎片，不只追一个分数。实验可并行，但要按实测GPU/CPU/RAM调度且不扩大预算。

## 11. 关键资产与代码入口

下面的权重hash引自原保留/完成收据，本次未重新读取大权重计算hash；旧清理后不得按历史路径自动补下载或重训。[S26,S27]

### Chair

- 生成器：experiments/r2_seam_quality_wave1_wave2_v1/inference/PLAIN_S2000_generator.pt
  - SHA256：46be9a29d6d9e150205cc46cf8c952ae1cd32b2aa58bff24cb879eb451932f43
- 配套头：experiments/r2_seam_quality_four_v1/inference/LONG_occupancy.pt
  - SHA256：6e8f9ab606403c9c2b96d7f6bf6a2c67dacd11d4c45179369414e420a75982df
- 完整恢复：experiments/r2_seam_pool47_couple_v1/checkpoints/PLAIN3000_step2000.pt
- BACK分支：experiments/r2_chair_multipart_seam_v1/inference/MULTIPART_S1500_generator.pt

### Bench

- EXPOSURE推理：experiments/r2_bench_four_routes_v1/exposure/training/M_PLUS3000/generator_step3000.pt
  - SHA256：70e51d7913f99ad017d023e47965387b28692075ed9964b61d9d55dd1f0ff6b3
- 同目录完整恢复：checkpoint_step3000.pt
  - SHA256：3c1e9e1db0f9a462632bd70a91b5acaa4bef1dd40ad20ad90a194ab17bbd49fd
- 最新布局证据：experiments/r2_bench_layout_oracle_v1/
  - REPORT.md、METHODS.md、results/per_output.csv、results/paired_per_parent.csv、gallery/index.html。
  - RUNBOOK.md、completion.json、source_hashes.json、code.diff、artifacts/retained_weights.json。

### 代码

- 原MeshFlow：models/equidit.py、flow_matching.py、train.py、inference.py。
- 条件基础：r2/src/meshflow_control/models/native.py、models/geometry.py。
- SEAM核心：r2/src/meshflow_control/seam/；原实验入口 r2/pilots/seam_slot_c200_v1/。
- HINGE：r2/tools/hinge_opposite_v1/、r2/tools/hinge_v2_cpu_v1/。
- 最新布局：r2/pilots/bench_layout_oracle_v1/。
- 维护说明：WORKSPACE.md、AGENTS.md、r2/docs/research_route_memory.md。

公共CLI、历史实验入口和最新研究代码有不同适用合同；不要直接拿旧R2的C20/C40训练命令启动新Bench方案。本次没有核验GitHub远端是否包含每项最新代码，不能把“本地存在”写成“云端已经同步”。

## 12. 本地证据索引

以下相对路径均以 E:/MeshFlow 为根。只发本MD也能读懂主结论；需要复算时，再按编号向本地索取逐例表/源码/少量图，不必先传大权重。

- [S01] experiments/r2_bench_layout_oracle_v1/REPORT.md
- [S02] native_t1/docs/control_metrics.md
- [S03] r2/docs/training_structure.md；experiments/r2_bench_adaptation_strength_v1/REPORT.md
- [S04] r2/docs/denoising.md
- [S05] r2/README.md
- [S06] r2/docs/seam_slot.md
- [S07] experiments/r2_source_winding_export_v1/REPORT.md
- [S08] maintenance/research_synthesis_20261028/RESEARCH_SYNTHESIS.md
- [S09] experiments/r2_bench_four_routes_v1/REPORT.md；experiments/r2_bench_four_routes_v1/exposure/REPORT.md
- [S10] experiments/r2_full_seat_learning_curve_v1/REPORT.md
- [S11] experiments/r2_seam_slot_c200_v1/REPORT.md
- [S12] experiments/r2_seam_quality_wave1_wave2_v1/REPORT.md
- [S13] experiments/r2_hinge_opposite_v1/REPORT.md
- [S14] experiments/r2_hinge_v2_cpu_v1/REPORT.md
- [S15] experiments/r2_chair_multipart_seam_v1/REPORT.md；experiments/r2_chair_back_transfer_v1/REPORT.md
- [S16] experiments/r2_bench_regional_mix_dual_v1/REPORT.md
- [S17] experiments/r2_bench_flow_integrity_diagnostic_v1/REPORT.md
- [S18] experiments/r2_d0_restore_quality_v1/REPORT.md
- [S19] experiments/r2_target_cleanup_c200_v1/REPORT.md；experiments/r2_dedup_n_sensitivity_v1/REPORT.md
- [S20] experiments/r2_bench_c_edgeflip_v1/REPORT.md；experiments/r2_bench_c_tessellation_v1/REPORT.md
- [S21] experiments/r2_bench_multilod_training_v1/REPORT.md；experiments/r2_bench_multilod_data_v1/REPORT.md
- [S22] experiments/r2_bench_three_mechanisms_v1/REPORT.md
- [S23] experiments/r2_bench_full_prior_motif_guidance_v1/REPORT.md；experiments/r2_bench_loop_motif_mini_v1/REPORT.md
- [S24] experiments/r2_nft_negative_result_closeout_20261015/NEGATIVE_RESULT_REPORT.md；experiments/r2_window_policy_rl_closeout_20261016/NEGATIVE_RESULT_REPORT.md
- [S25] r2/docs/research_route_memory.md；r2/docs/research_route_registry.json
- [S26] maintenance/storage_cleanup_20261025/RETAINED_MODELS.md；maintenance/storage_cleanup_20261025/FINAL_CLEANUP_RECEIPT.json
- [S27] experiments/r2_bench_four_routes_v1/WEIGHTS.md；experiments/r2_bench_layout_oracle_v1/artifacts/retained_weights.json
- [S28] experiments/r2_bench_full_seat_curriculum_v1/REPORT.md
- [S29] experiments/r2_bench_seat_coverage_v1/REPORT.md
- [S30] experiments/r2_bench_exposure_allocation_v1/REPORT.md
- [S31] experiments/r2_bench_natural_n_recipe_v1/REPORT.md
- [外部附件，未独立复算] C:/Users/warpwang/.codex/attachments/b3ce7ea4-88d8-4d51-87a9-66d60c06ee74/Pasted text.txt：云端九类候选小实验文字，不能当本地生产结果。

本交接保留了正结果及其限制；未把历史负结果改写成普遍不可能，也没有将多个未组合运行的优势拼成一个已完成系统。

