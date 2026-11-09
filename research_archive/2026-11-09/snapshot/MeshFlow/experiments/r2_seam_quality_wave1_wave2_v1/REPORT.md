# R2_SEAM_QUALITY_WAVE1_WAVE2_V1 实验报告

**本轮五个候选已完成固定预算。没有候选同时通过预定的质量、接续和图审条件，因此不升级默认模型。最值得保留的局部结果是 CLEAN 的 DEV FF/I 下降；它不是 TRAIN/DEV 一致的改进。SURGICAL 没有实现本轮预期的 DEV 保形收益。**

Wave1 的真实普通续训曲线也说明：更多步数、训练父上的更低 FF、权重中点，并不自动带来更好的未见条件完成。保留全部权重和逐例证据，结束本轮，不自动加步数、组合机制、扫描系数、切默认、commit 或 push。

## 实际完成范围

| 阶段 | 实际执行 |
| --- | --- |
| 前置清理 | 仅任务书白名单16份旧权重；释放25,041,197,776字节，即23.3214 GiB；36份保护权重前后哈希验证 |
| Wave1 | 实际PLAIN +1000/+2000检查点、B与PLAIN+3000的FP32等权中点；零训练；84 RAW、84次D0、84 BOUND |
| Wave2 | CLEAN、SURGICAL各1000次更新，每臂8000次训练前向及8000次反向；56 RAW、56次D0、56 BOUND |
| 固定评测 | 4个熟悉TRAIN父×4噪声=16；3个DEV父×4噪声=12；五个新模型共140请求、7000次Euler网络前向 |
| 完整数值 | 8模型×28请求×RAW/POST/BOUND015=672行，保留124数值字段；新增420评分单元；三份历史参照只读复用 |
| 实际图审 | 140条新模型POST全覆盖；另28条B参照与40条预登记关键BOUND；208条规范记录，AI_PROXY、非盲 |
| 限制 | 占用头前向0、新OT调用0、历史参照重生成0、额外训练臂0；没有模型失败、OOM或静默重试 |

任务原文：[TASK.md](TASK.md)。固定合同：[protocol.json](protocol.json)、[analysis_policy.json](analysis_policy.json)。完成凭据以本目录最终 `completion.json` 及独立 FULL 审计为准；数值、渲染等子阶段凭据保留其完成当时的状态文字，不覆盖历史记录。

## 比较口径

五个候选共享已冻结的C、N、四份噪声、查询和LONG的缓存C-only布局。训练池是原TRAIN47；评测中的“TRAIN16”只是其中四个熟悉条件，不代表完整47父的生成评价。DEV仅三个已反复观察的父，属于开发证据；四噪声不当成四个独立父。不给来源唯一款式排名。

Wave2主对照固定为 **PLAIN_S1000**，不是挑结果更有利的LONG、B或PLAIN3000。两臂从相同B活生成器、匹配AdamW及RNG分叉，global13000→14000，batch8/microbatch1、lr1e-5、clip1、原受限OT后HYBRID0.25不变。复用相同8000条缓存及面/角点匹配；47父曝光34父×170、4父×168、9父×172。

每波全部RAW落盘后才执行单次官方D0；恢复C与seam前两锚点，原GEO_C清理，恢复来源C角序并固定C定向free。BOUND015再使用真实GEO_C簇、PRE_GEO逐成员总位移不超过1.5%L的既有绑定器。没有第二次D0。

下表FF为**参与非法F/F面对的保留free面比例**；CF为参与非法C/F面对的free面比例；I为固定C查询上的 `ConflictCov_C`，不是整个mesh冲突面积。全部显示为百分数。可达≥30%是合法共边图上的描述阈值，不能代替完整接缝、用途或水密性。完整分母、均值/中位数和面对数见逐例CSV。

## POST主表：质量与接续

| 模型 | TRAIN FF% | CF% | I% | TRAIN bulk≥30% | DEV FF% | CF% | I% | DEV bulk≥30% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LONG | 14.351 | 1.507 | 0.266 | 11/16 | 16.127 | 5.548 | 2.315 | 0/12 |
| B_LONG_LAYOUT | 15.904 | 1.420 | 0.409 | 12/16 | 15.164 | 6.235 | 3.434 | 0/12 |
| PLAIN_S1000 | 16.282 | 1.246 | 0.229 | 11/16 | 14.183 | 4.649 | 3.463 | 0/12 |
| PLAIN_S2000 | 12.987 | 0.886 | 0.177 | 13/16 | 16.075 | 5.612 | 3.699 | 0/12 |
| PLAIN3000 | 13.712 | 1.411 | 0.308 | 11/16 | 19.119 | 4.497 | 2.600 | 0/12 |
| WISE_B_PLAIN_050 | 15.033 | 1.340 | 0.385 | 11/16 | 16.514 | 6.414 | 3.137 | 0/12 |
| CLEAN_BULK_FM_S1000 | 17.177 | 1.825 | 0.443 | 11/16 | **12.572** | 4.857 | 3.048 | 0/12 |
| SURGICAL_LATE4_S1000 | 14.447 | 0.970 | 0.317 | 11/16 | 17.210 | 6.436 | 4.553 | 0/12 |

完整三阶段主表：[KEY_TABLES.md](results/final/KEY_TABLES.md)。[质量轨迹图](results/final/quality_trajectory.png) 已实际检查；WISE画在横轴1500处仅用于排版，它不是实际第1500步。

## Wave1：找到了不同取舍，没有找到完整替代

**PLAIN_S1000** 的DEV FF低于B约0.981个百分点，也明显低于PLAIN3000；平均保留154.083张free面，高于B的145.500，不是靠更多删面得到这个均值。但逐父方向相反：2d与610改善，608的FF比B增加7.627个百分点。实际DEV可见裂口YES从B的3/12变为4/12，610/n3仍有明显大折片。因此没有建立“裂口/碎片不恶化”的完整条件。

**PLAIN_S2000** 的熟悉条件较好：TRAIN POST FF12.987%、I0.177%，BOUND达15/16，均优于同轨迹+1000。但DEV POST FF为16.075%，严格高于任务书16.0%线，不能四舍五入判通过；I3.699%、小角33.169%，平均保留free降至135.417。尤其610平均少32.5面，伴随小角恶化。它是TRAIN上的取舍点，不是DEV整体升级。

**WISE0.5** 也不是免费融合：DEV FF16.514%，连通分量21.000，高于B的14.583；610分量增加14.25。它没有通过预设门禁。更详细证据见 [WAVE1_RESULTS.md](WAVE1_RESULTS.md)。

## Wave2：CLEAN的局部收益及其代价

相对等预算PLAIN_S1000，CLEAN的DEV POST FF **14.183→12.572%（−1.611个百分点）**，I **3.463→3.048%（−0.415个百分点）**。DEV BOUND的FF也较低：15.362→12.654%。这是真实配对信号，应保留。

但不能晋级为稳定质量改进：

- TRAIN POST FF **16.282→17.177%**，CF1.246→1.825%，I0.229→0.443%；与“TRAIN和DEV都降低FF及bulk-bulk面对”预期不符。
- DEV CF4.649→4.857%，小角28.149→29.094%，连通分量15.500→16.083，开放边124.583→126.500。降FF没有同时减少散裂指标。
- DEV平均保留free153.750对154.083，平均多删0.333面；未出现上一轮去重实验的大幅删面变化，但不能写成“完全没有额外删面”。TRAIN平均少删0.688面。
- DEV逐父FF变化：2d **+0.898**、608 **−2.929**、610 **−2.802**个百分点；608的CF反而+1.106个百分点。它不是三个父一致改善。
- POST及BOUND的≥30%身份与PLAIN_S1000一致：TRAIN分别11/16和13/16，DEV均0/12。DEV平均可达从0.089%到1.770%，中位数仍0，不是接续能力突破。

绝对面对账本与比率方向相符，但收益集中。TRAIN非法FF总对数从392增至427，其中bulk-bulk **366→394**、seam-bulk26→33；DEV总对数205→184，其中bulk-bulk **157→142**、seam-bulk43→37、seam-seam5→5。DEV的I降低主要来自dup **0.7406→0.3784%**，cross反而 **2.7588→2.7832%**；cross与dup有交集，不能简单相加当union。

逐噪声CLEAN FF下降/上升/平局：TRAIN **7/7/2**，DEV **5/6/1**，不是多数请求一致改善。最大DEV收益610/n0为22.9008→6.8182%；最大DEV恶化2d/n2为10.8434→19.8758%；TRAIN的85/n1从37.5887%增至53.2847%，仍是显著坏尾。不能只引用−1.611个百分点的DEV均值。[绝对面对与I组成](report_preparation/wave2_evidence_summary.csv)、[逐噪声明细](report_preparation/wave2_clean_vs_plain_per_noise.csv)。

CLEAN没有删数据或减N，只把参与来源非法/UNKNOWN面对的1241张bulk源面权重置0.2；7454张其他bulk面和360张owner的自由第三点权重仍1。任务书加权有效自由标量分母照常执行。这会使未降权坐标的绝对归一化系数改变，倍率1至1.688466，因此本轮不能解释成“只有脏面梯度改变而seam绝对梯度严格不变”，也没有证明来源污染是生成自交的唯一根因。

## Wave2：SURGICAL未实现DEV保形目标

SURGICAL的TRAIN FF **16.282→14.447%**、CF1.246→0.970%，85/n1图中也出现了更正常的方背细腿椅，原大侧板和落地折片不再明显。这是局部正结果。

但DEV相反：FF **14.183→17.210%**、CF **4.649→6.436%**、I **3.463→4.553%**；连通分量 **15.500→23.000**，开放边 **124.583→150.417**，非流形边7.917→11.333。它保留更多free（160.333对154.083）且小角更低（24.221%对28.149%），这两项不能抵消冲突和散裂代价。

逐父结果不是统一失效：608 FF下降5.237个百分点；2d增加9.623、610增加4.695个百分点。610/n2实际出现明显大三角背部折片。TRAIN POST新增85/n1≥30%但丢73/n2；BOUND还丢73/n3，最终12/16对PLAIN的13/16，其中73/n2为29.7066%的近阈值例，完整连续值保留。

本轮实际冻结 `backbone.layers.0..7`，前后逐字节不变；82个可训练参数状态精确继承AdamW动量与step13000，112个冻结状态移出优化器。前处理梯度保持。**参数冻结不等于内部表征冻结**，因为坐标/时间/角色/Geo输入嵌入仍在训练；本轮结果不支持该固定掩码，但不构成所有分层微调方案都无效的证据。

## 训练误差与自由生成的区别

以下为最后100更新、同800个正常训练前向实例中，按相应有效自由标量数聚合的未加权速度MSE。不是重新前向，不是用带权loss互相比数值。

| 类别 | PLAIN_S1000 | CLEAN | SURGICAL |
| --- | --- | --- | --- |
| 全部自由坐标 | 0.112796 | 0.113970 | 0.113110 |
| seam第三点 | 0.095839 | 0.090318 | 0.150539 |
| 全部bulk | 0.113029 | 0.114295 | 0.112594 |
| clean-bulk | NA：旧日志未记录 | 0.113191 | 0.114038 |
| dirty-bulk | NA：旧日志未记录 | 0.121228 | 0.103531 |

CLEAN的seam误差略低，没有兑现为新增DEV高可达；SURGICAL的seam误差明显较高，训练门禁本身也未满足。三者全free MSE很接近，不能据此说生成质量相同。曲线为不同正常训练样本窗口，非固定验证集单调学习曲线；同窗口三臂才有严格输入配对。[完整学习曲线](training_summary/unweighted_learning_curves.png)、[逐更新记录](training_summary/per_update_unweighted.csv)、[时间段拆分](training_summary/window100_time_bin_unweighted.csv)。

RAW与POST必须分开：DEV RAW FF为PLAIN88.595%、CLEAN88.133%、SURGICAL88.558%，远高于各自POST；CLEAN在RAW的FF只降低0.462个百分点。实际收益包含生成输出与固定后处理的共同作用，不能把POST差值全部称为RAW表面已经干净。RAW全部模型的精确seam-bulk共边可达均为0；固定C/seam底边与长出完整接续是两回事。

## BOUND015：保持旧连接，但仍有FF代价

| 模型 | TRAIN bulk≥30% POST→BOUND | DEV bulk≥30% | DEV FF% POST→BOUND | DEV新增/消除非法FF对 |
| --- | --- | --- | --- | --- |
| LONG | 11→16 /16 | 0/12 | 16.127→15.569 | 19/19 |
| B | 12→14 /16 | 0/12 | 15.164→15.845 | 19/6 |
| PLAIN_S1000 | 11→13 /16 | 0/12 | 14.183→15.362 | 26/1 |
| PLAIN_S2000 | 13→15 /16 | 0/12 | 16.075→16.854 | 30/16 |
| PLAIN3000 | 11→13 /16 | 1/12 | 19.119→19.176 | 6/3 |
| WISE0.5 | 11→13 /16 | 0/12 | 16.514→16.160 | 17/18 |
| CLEAN | 11→13 /16 | 0/12 | 12.572→12.654 | 4/1 |
| SURGICAL | 11→12 /16 | 0/12 | 17.210→17.382 | 16/17 |

所有已记录BOUND格子的新增非法CF、原合法CF/FF边丢失、额外删源面均为0。这个保护合同**不保证零新增FF**，总面对数净减少也不保证受影响面比例必降。CLEAN和SURGICAL的DEV各仅接受1次整侧边绑定，TIP分别15/22次，仍未产生≥30%请求。历史PLAIN3000的608/n3 BOUND可达61.7988%仍是唯一DEV≥30%例；本轮没有把它藏进总体均值，也没有把它扩写为普遍泛化。

## 实际图审

模型名可见，均为 **AI_PROXY_NONBLIND**；双透视、正侧有限剖面使用历史固定相机和尺度。每条保存图片SHA、YES/NO/UNCERTAIN及文字观察。图审不由评分自动生成，不把设计开孔自动当裂缝，也不把剖面断线自动当三维断裂。

| POST图审 | TRAIN：主体YES / 采用YES / 碎折YES / 裂口YES | DEV：主体YES / 采用YES / 碎折YES / 裂口YES |
| --- | --- | --- |
| B参照 | 16 / 15 / 1 / 2 | 12 / 10 / 1 / 3 |
| PLAIN_S1000 | 15 / 15 / 3 / 6 | 12 / 10 / 1 / 4 |
| PLAIN_S2000 | 16 / 16 / 1 / 2 | 12 / 11 / 2 / 5 |
| WISE0.5 | 16 / 15 / 1 / 2 | 12 / 10 / 2 / 5 |
| CLEAN | 15 / 15 / 3 / 7 | 12 / 10 / 1 / 5 |
| SURGICAL | 16 / 15 / 3 / 6 | 12 / 10 / 1 / 5 |

TRAIN分母16、DEV分母12；未列的项不能都当作NO。完整三分类计数见 [visual_summary.csv](visual_summary.csv)。例如CLEAN的TRAIN碎折为3 YES、8 NO、5 UNCERTAIN；SURGICAL的DEV碎折为1 YES、7 NO、4 UNCERTAIN。仅看YES计数相同，不足以宣称视觉不退化。

具体图例：CLEAN的85/n0出现明显背板斜裂口；85/n1仍保留大侧板与落地折片，SURGICAL在该例更完整。SURGICAL的610/n2出现大背部三角折片；CLEAN的610/n3原大折片仍在。73/n1的上层替代座面没有因绑定消失。关键BOUND复核没有发现新增宏观主体缺失，但原裂口/折片大多保持，不能用共边指标提升代替视觉修复。

图集：[Wave1 POST](gallery/wave1_POST/index.html)、[Wave2 POST](gallery/wave2_POST/index.html)、[Wave1 BOUND](gallery/wave1_BOUND/index.html)、[Wave2 BOUND](gallery/wave2_BOUND/index.html)。[208条实际图审](visual_review.csv)。

## 资源、恢复与资产

RTX5080总显存16303 MiB；开始每波均记录GPU使用、进程、CPU和RAM。先运行已授权的CLEAN真实更新，观察前向/反向/AdamW分配后再启动SURGICAL。第二worker放行时空闲9795 MiB、额外保留2048 MiB，按4096 MiB保守worker估计及1.2倍余量核算。没有为调度新增探针训练。

两条训练进程实际重叠1687秒；完整训练计时CLEAN约1868秒、SURGICAL约1687秒。两臂allocator reserved峰值分别3184/1860 MiB；设备总峰值8647 MiB包含并发及其他用户进程，**不能当作某条进程峰值**。WDDM逐进程VRAM值为NA。CLEAN并发区间平均1.603秒/更新，非重叠区间1.371秒；不与历史孤立耗时作未经控制的加速比。

采样按实测资源逐个加载，最多两worker；Wave2 D0实际未形成并发执行重叠，因为第一组处理在第二组加载就绪前已结束。CPU评分/绑定均两worker、各1线程：Wave1约163.13/85.05秒，Wave2约109.23/59.96秒。资源时序与阶段范围见 [phase_resources.json](training_summary/phase_resources.json)、[concurrency_phases.json](training_summary/concurrency_phases.json)、`resources/`及各评测阶段账本。

前置清理的删除清单、哈希及保护验证保存在 [cleanup/completion.json](cleanup/completion.json)。报告编码、三条图审缺失枚举、独立审计器对配对CSV布尔文本的解析遗漏，以及收口元数据重复arm键曾作工程修正；失败稿、原记录和修正理由均保留。审计解析修正只重查受影响结果段及图审，收口修正只重写完成元数据，不重跑模型或评分，详见 [EXECUTION_NOTES.md](EXECUTION_NOTES.md)。

新训练完整权重与独立推理导出均保留：

- [CLEAN完整检查点](checkpoints/CLEAN_BULK_FM_step1000.pt)；[CLEAN推理权重](inference/CLEAN_BULK_FM_generator.pt)。完整检查点文件SHA256：`3d07de861ffcf64afb9c53c47eaa815cf040e9abad7673eda597ae8ed4ad2bb2`。
- [SURGICAL完整检查点](checkpoints/SURGICAL_LATE4_step1000.pt)；[SURGICAL推理权重](inference/SURGICAL_LATE4_generator.pt)。完整检查点文件SHA256：`9d8f862b7883cd98068f9721709c6f9ff0203f01e5c7a1fbcf4fb28e7c8d08ed`。
- Wave1三份推理权重和逐张量194键验证见 `inference/`、[wave1_interpolation_tensor_audit.json](wave1_interpolation_tensor_audit.json)。

可复算交付：[逐输出124字段](results/final/per_output.csv)、[逐父配对](results/final/paired_per_parent.csv)、[逐噪声配对](results/final/paired_per_noise.csv)、[角色非法面对](results/final/illegal_pairs_by_role.csv)、[删面身份](results/final/deleted_faces.csv)、[绑定账本](results/final/binding_ledger.csv)、[绑定逐输出差值](results/final/binding_effect_per_output.csv)。所有RAW、PRE_GEO、POST及BOUND实际网格位于 `evaluation/`、`binding/`。

隔离实现位于 `r2/pilots/seam_quality_wave1_wave2_v1/`。状态与恢复总入口：

```text
python -B -m r2.pilots.seam_quality_wave1_wave2_v1.resume status
```

使用现有WSL环境、仓库根及 `r2/src` 的PYTHONPATH。完成后此入口只允许状态检查；完整执行凭据、源文件SHA及独立审计由最终 `completion.json` 引用。

**独立执行审计：FULL / PASS。** [AUDIT_FINAL.json](independent_audit/AUDIT_FINAL.json) 核验了实际权重/AdamW/冻结张量、8000条共同输入身份、140份RAW的7140次积分状态锚点约束、7000次生成前向和140次D0调用、672×124结果矩阵、166656项配对差值复算以及全部实际图审身份。核验文件17755个；保护旧权重优先复用本轮清理时两次实际SHA并验证当前文件身份，不重复无关大盘扫描。原审计布尔解析失败件单独保留，修正后只增量闭合受影响段，未追加模型或评分调用。审计文件SHA256：`7ba4045e6f2b5b05e0b7e878c09149dafed8379044ef98ac7b5b6e0913bc7aaf`。

[报告数值复核](report_preparation/report_review.json) 对独立复核稿的主表、逐父/逐噪数据、预算、图审及解释边界给出PASS_WITH_SCOPE_LIMITS；其后仅补充审计说明和明确完整检查点SHA措辞，未更改数值结论。

## 预定判定与停止

| 候选 | 本轮判定 |
| --- | --- |
| PLAIN_S1000 | DEV FF较低但逐父和图审有代价；没有完整通过Wave1晋级条件 |
| PLAIN_S2000 | TRAIN取舍较好；DEV严格超过16% FF线且删面/小角/裂口有代价 |
| WISE0.5 | 未满足DEV FF及表面完整性条件 |
| CLEAN | 保留DEV FF/I降低的局部信号；TRAIN反向、CF/小角/裂口有代价，不满足两组一致改善门禁 |
| SURGICAL | TRAIN有局部改善；DEV散裂、FF/CF/I及seam训练误差未达目标，不晋级 |

没有结果支持自动把CLEAN与其他损失组合、延长SURGICAL或切换默认。SEAM既有连接正结果仍保留，本轮回答的是在固定布局和预算下，这五个候选能否提供更好的质量取舍；答案是**有局部收益，尚无满足完整合同的统一升级**。
