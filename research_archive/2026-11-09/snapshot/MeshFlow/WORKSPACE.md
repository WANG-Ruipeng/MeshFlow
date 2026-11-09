## 2026-10-21 Bench USER v2 1500步已完成：有限局部收益，未整体晋级

25父81条USER v2实际完成1500更新、每父480实例、48RAW/48单次D0和144新增三阶段评分；训练12000前向/12000反向/1500更新账本闭合，无失败，冻结33训练代码/219输入hash检查PASS。实际AI_PROXY查看12张逐父联系图覆盖144个A/新模型配对页面，另看5页原分辨率；完整报告与完成状态见 [REPORT](experiments/r2_bench_user_v2_training_v1/REPORT.md)、[CLOSEOUT](experiments/r2_bench_user_v2_training_v1/CLOSEOUT.json)。旧自动流水线“待图审”状态是历史阶段，不需重新派发。

DEV最终相对A：FF26.97→29.04%、I8.54→9.91%、CF6.08→6.97%，非流形3.281→2.375，但单面孤岛15.094→21.344；适用请求bulk可达≥30%仍0/26。座面8请求/4父子组FF17.56→15.18%、I9.79→8.37%，个别靠背/组合有局部外观收益；长蓝座板上另长灰板、短支撑悬在主体内部等旧坏例仍在。保留有限正信号，不晋级USER_V2_S1500为综合baseline。

本轮目标、父池、条件分布和曝光共同改变，COMMON12000含旧历史；不能单独归因于干净数据、数据数量或过拟合。旧面板TRAIN4父均不在当前25父，DEV是反复使用的开发面板。实际池未达此前47–67父扩充目标。本轮正式终点权重和全部数据/图/原评分保留，不自动加训、删权重、改默认、commit/push。

## 2026-10-21 Bench USER v2 单臂1500步已启动（历史记录，已完成见上）

用户随后授权用封存25父81条开始训练。独立实验 `experiments/r2_bench_user_v2_training_v1/` 已冻结并启动：复用A/COVERAGE的COMMON12000 live及fresh AdamW共同起点，原SEAM/free-only FM/OT后HYBRID0.25、batch8/micro1、lr1e-5，新增1500步至global13500，每父480实例曝光。未新增占用头训练、loss或架构。数据与固定48请求CPU检查PASS，首批正式前后向及更新正常，实测余量约9GiB；仅一个授权GPU worker。

该条记录为启动状态，不是完成或正结果。后台有限流水线训练后自动执行48RAW、单次D0、原统一几何处理、144新增评分和同位置截面图；实际图审与研究结论仍须等产物完成后另记。原面板TRAIN4父均不在当前25父池，必须区分历史标签与本轮成员身份；DEV8父保持隔离。本次是目标质量、来源与条件分布、曝光一起改变的数据方案比较，不是纯去重因果对照。原资产及默认不动，不清权重、不commit/push。

[协议](experiments/r2_bench_user_v2_training_v1/PROTOCOL.md) · [运行与恢复说明](experiments/r2_bench_user_v2_training_v1/RUNBOOK.md) · [实时状态](experiments/r2_bench_user_v2_training_v1/workflow/state.json)。不要重复派发已有进程或静默回滚失败更新。

## 2026-10-21 Bench USER v2 首批数据已实际制作

按用户“先造数据看看”完成独立首批：32父目标提案、107条条件，6父／18条通过完整目标几何检查、实际AI_PROXY图审与18/18生产CPU OT打包验收；26父目标几何BLOCKED，另2条局部座面/组合C因不规则切口暂存。实际数据、OBJ/C/Y、来源映射、逐条件记录和171页图册在 [Bench v2报告](experiments/r2_bench_user_data_v2/REPORT.md) 与 [有效图册](experiments/r2_bench_user_data_v2/index.html)。机器入口 `experiments/r2_bench_user_data_v2/DATASET_READY.json` 绑定manifest和PASS收据，恢复入口 `python -B -m r2.data_tools.bench_user_v2.run verify`。

6个通过父仅删除41张精确零面积面，所有正面积与原坐标保留；去重/细分/翻面的复杂提案没有通过，不能算修复成功。18条包含4靠背、4座面、3支撑侧表面、7组合；N133..215、K2..99，全部开放且有owner。本批不认证闭合C覆盖，不新增独立父、不证明训练质量改善，也不以6父取代原79父。待审44条已补查，17通过原两项门槛、27有已知问题；原v1清单保持历史版本。旧39 DEV/8父及2248个已登记资产hash核对未变。全程零模型/GPU/训练，无物理删除、默认切换、commit或push。

## 2026-10-21 Bench 后续训练池条件级筛除

用户已明确选择“从后续训练池剔除不合适样本，保留原件和实验记录”。新Bench数据任务以 [后续数据池约定](r2/docs/bench_data_pool_policy.md) 及机器入口 `r2/data_tools/bench_future_pool.json` 为准：原321条TRAIN中，163条有已知C/Y整面精确重合或有效Y精确零面积，隔离待重建；44条缺相同口径目标审计，待核验；114条/30父通过这两项检查，保留为候选。旧39条DEV/8父、原训练清单和所有资产原样保留，零物理删除；按条件筛除，不整父删除。

这不是已认证干净池：114条中仍有43条/15父的Y内部反向重复，全量自交等尚未核验；靠背只剩3条/3父。后续需补覆盖并重新冻结训练采样计划，不能自动沿用321条旧日程，也不能将覆盖变化的效果单归因于清洗。新构造优先完整规整的人类部件输入，不随机凑面；闭合腿及合理多部件组合不自动排除。历史实验加载器和维护模型默认不变。本次仅CPU清单筛选，未训练/模型调用/几何重评分/commit/push。

清单与执行回执：`experiments/r2_bench_data_retirement_20261021/future_pool_v1/`；核验入口：`execute_future_filter.py --verify`。隔离与待核验条件须在新版本完成重建/审计后才可恢复，不覆盖原件。

## 2026-10-21 当前主线：Bench 生成质量与泛化

用户明确当前优先目标仍是提升 Bench 数据集上的整体生成质量和未见父对象泛化：主体完整、少错层/穿插/破损，并合理使用和接续输入 C。局部 C/F 接口或单个指标提升不能替代这些目标；Chair 已有正结果继续保留。

用户当前更怀疑数据质量与条件构造。后续优先区分完整来源 M/Y 的几何质量、C→free 任务构造是否合理、独立父对象/部位/几何形态的覆盖，以及后训练专门化；不能把这些问题统称为样本少。Bench CLEAN1500 的有效接口条件筛选不等于完整来源网格清洗；数据因素仍是待验证假说，不能据此宣布模型/表示因素已排除，也不能把 Chair 的源缺陷统计当作 Bench 统计。

本次仅检索论文、复核既有本地证据和记录研究优先级；不启动训练/生成/重评分/清理，不改变维护默认，不 commit/push。后续实验需保持固定面板与完整质量代价记录。

## 2026-10-18 保留 S2000 + HINGE 反向绕序守卫

用户明确选择保留 `PLAIN_S2000 + HINGE 1.5%L`，附2.5%L对照。独立版本对整簇合并引入的全部双面边（含旁生CF/FF）要求反向，保持旧缺陷身份、不翻面/删面。固定28请求×2阈值CPU复跑、原124＋11拓扑字段及独立审计通过；原1172资产hash未变、零模型调用。

主档全组FF14.3104→13.4902%、CF2.9116→2.7878%、单面孤岛150→121，新增105条合法边；原4624条合法边全保留、新增同向边0、新非法CF/FF面对0、额外删面0。仍有610/n1的I+0.048828pp、少数q/p05退化；DEV bulk≥30%仍0/12。归档为有限正结果，不称全指标零退化，不切换维护CLI默认。原S2000权重与旧HINGE结果保护保留，不自动清理。

报告与恢复账本：`experiments/r2_hinge_opposite_v1/REPORT.md`；实际代码：`r2/tools/hinge_opposite_v1/`；可发布源码、135字段完整CSV和固定截面图：`r2/research/hinge_opposite_v1/`。本轮用户已明确授权验证正结果后commit/push，权重不上传。

## 2026-10-16 SEAM 几何保护：CPU实测完成，未进入条件RL

后续反馈勘误：60°条件正确为`r < -abs(h)/sqrt(3)`；原报告写成乘sqrt(3)错误。实际代码按cos≤0.5判拒，缓存表复核50/62、341/389等统计不变。原件不覆盖，[勘误及核验](experiments/r2_seam_geometry_guard_v1/errata_normal_60deg_20261016/ERRATUM.md)另存。容量版与原GEO是同PRE_GEO两终态对照；旧FF关系未重现不等于直接切开前置已有连接。

R31后续固定16份D0提议容量保护已运行，原GEO每份94项数组重放一致；全部175条原合法CF保留、新增17条，CF非流形18→0，但丢13条原合法FF、新增11对seam–bulk非法FF，FF14.563→15.348%，I3.3020→3.29895%基本不变；13/16非零bulk与11/16≥30%保留。3f/n2精确共面CF三对消失而I仍42.4805%，说明避免精确吸附不等于消除近重层。容量保护保留独立候选，不自动晋级默认；共享槽原正结果仍保留。

占用解析去权重恰变为全占用：62/62召回但Null0/4，不能称修好分类器。所提seam/C法向夹角<60度会拒绝50/62合法来源接口（正常侧壁90度），未加入训练。原SEAM POST343个非法FF对中338为bulk–bulk，不能用seam直接碰撞解释全部FF；不能从此排除间接生成影响。按用户“前三项稳定后”条件，RL NOT_RUN_GATE_NOT_MET。零模型前向/训练、新PRE_GEO评分16及候选POST16；全部配对图已实际非盲AI_PROXY审阅。

[报告](experiments/r2_seam_geometry_guard_v1/REPORT.md) · [逐例/连接账本](experiments/r2_seam_geometry_guard_v1/capacity/per_output.csv) · [图集](experiments/r2_seam_geometry_guard_v1/gallery/index.html) · [只读核验/CPU恢复入口](r2/pilots/seam_geometry_guard_v1/resume.py)。原资产、默认与Git未改变。

## 2026-10-16 SEAM_SLOT_C200 有限正结果已加入科研手记

按用户要求将共享原边200步配对实验单列为研究路线 **R31 / BOUNDED_POSITIVE_SIGNAL**，不并入R01—R30未形成稳定升级路线。两臂各200步、四条已见TRAIN C、每臂16固定新噪声；32RAW/32POST及64实际评分/图审完成。相对原架构FULL4_C200，合法接口13/16→16/16、合法边131→175；候选13/16从合法seam经不含C的free共边图到达bulk，11/16 bulk可达面积≥30%。73/n2、85/n3有实际接续增益，用户认为图上整体效果已经较好。

保留CF与小角改善，也保留FF12.841%→14.563%、I0.671%→3.302%、新增非流形及方向一致可达略降；3f/n2是已定位的seam共面内翻尾部。占用漏26/62正边；73/n1新增用途但bulk可达0。后续关注是保住外观与已有接续，利用现存逐面证据定位FF/I来源，不预设加loss、不自动新训；四TRAIN原型不认证DEV或默认升级。

[科研手记R31](r2/docs/research_route_memory.md) · [完整报告](experiments/r2_seam_slot_c200_v1/REPORT.md) · [图集](experiments/r2_seam_slot_c200_v1/evaluation/gallery/index.html) · [保留权重身份](experiments/r2_seam_slot_c200_v1/WEIGHTS.json)。两臂step200完整恢复权重、推理生成器和占用头、原始几何/评分/报告/审阅包均保留。本次仅文档记忆更新，无模型调用、清理、默认切换或commit/push。

## 2026-10-16 窗口轨迹RL收口与权重清理

R2_WINDOW_POLICY_RL_V1完成标定及64批机会（54次实际AdamW更新/10次跳过）。固定44请求上，FULL_DET/FULL_STOCH/RL_STOCH的合法接口与严格联合均TRAIN9/32、DEV0/12，成功身份完全相同；RL相对同随机采样器的冻结FULL没有新增成功，共边总长度减少4.90861，TRAIN FF +0.65345pp，CF/I有局部下降。当前配置记为负结果，不扩大，不否定所有RL，不把标定通过/非零梯度当终态收益。

按用户授权已删除batch_000..062共63份中间权重，605,818,757字节（577.75MiB）；保留batch_063最终适配器、FULL_S6000、官方模型及全部RAW/POST/buffer/图审/评分/原报告。历史逐批模型重放及旧训练控制器全链恢复退役，不自动补训。原报告/ZIP/最终权重哈希未变。当前维护零训练/模型调用/重评分，不切默认，不commit/push。

[负结果收口](experiments/r2_window_policy_rl_closeout_20261016/NEGATIVE_RESULT_REPORT.md) · [清理回执](experiments/r2_window_policy_rl_closeout_20261016/cleanup_receipt.json) · [生命周期说明](experiments/r2_window_policy_rl_v1/CHECKPOINT_RETIREMENT_20261016.md)。研究登记新增R30，来源报告86份；FULL_S6000仍为方法比较参照。

## 2026-10-15 NFT负结果收口与中间权重清理

当前NFT配置停止扩大。训练后终点评价面板36个配对请求三模型明确联合均9/36；八条奖励TRAIN C×四份新噪声的冻结检查，FULL/NFT/SELF严格联合11/12/11，RAW采用均26/32。唯一新增请求也由无奖励SELF实现，NFT相对SELF的净差是保住另一旧正例。共同已知28对/7C的NFT原始奖励相对FULL -0.003113；第8条C奖励UNKNOWN，不填零。FF均值局部改善与I/尾部代价同时记录，不否定全部RL，也不认为改奖励已被证明是首要答案。

用户授权删除30个NFT/SELF round00..14中间.pt，逻辑空间287,949,354字节（274.61 MiB）；保留两份round15、FULL_S6000、官方模型及全部RAW/POST/图审/评分/源码/日志/旧报告。历史逐轮权重重放已退役；不自动补回、不续训。实际磁盘变化、完整路径与SHA见[清理回执](experiments/r2_nft_negative_result_closeout_20261015/cleanup_receipt.json)。[负结果报告](experiments/r2_nft_negative_result_closeout_20261015/NEGATIVE_RESULT_REPORT.md)；研究记忆新增R29，来源索引83+2=85份。FULL仍为方法比较参照，维护CLI默认不变；零新模型调用，无commit/push。

## 2026-10-14 Partial-edge binding CPU pilot completed

R2_PARTIAL_EDGE_BINDING_V1 executed all 24 FULL_S6000 outputs x A/B x 1/2/3.5%L = 144 cells, zero model/GPU calls. Original POST representatives and actual GEO_C clusters were frozen; no radius reclustering. A reassigns selected corners; B assigns the original free vertex cluster to exact C indices. All six policies preserve the original 6 joint positives and all 62 legal CF edges; zero face deletion and zero newly illegal CF pairs.

Joint screening counts A=8/8/9, B=7/8/8 of24. Original legal FF edge losses A=12/16/27, B=0/0/0. Newly illegal FF pairs A=0/1/4, B=0/4/4. B2's two conversions are TRAIN73 n0/n1, DEV remains0/12. Only A3.5 adds DEV2d n1 with tearing costs. All608/610 requests have zero eligible candidates even at3.5%; this is a candidate-coverage limit of whole-edge endpoint correspondence, NOT evidence that attempted binding destroyed them. Do not infer a physical upper bound, global post-hoc impossibility, or pointer learnability. Whole-edge one-to-one matching does not test edge-subsegment bindings.

34 paired images actually reviewed by nonblind AI_PROXY (6 contact sheets plus3 full-size cases), including all12 new joint arm/request entries. C's seat role remains visible; local cracks/full seams are not visually certified. Formal binding/scoring151.24s,2CPU workers; this excludes preparation/review. Independent144-cell audit and9880 cached scalar comparisons PASS,611 protected files unchanged. One before-decode import failure and hash amendments retained. Current default unchanged; no training/commit/push. Report: [REPORT.md](experiments/r2_partial_edge_binding_v1/REPORT.md). Resume: r2/tools/partial_edge_binding_v1/resume.py (completed runs verify only).

## 2026-10-14 全历史研究路线查重记忆

按用户要求回看本地全部主要实验阶段，建立[r2研究路线记忆](r2/docs/research_route_memory.md)、[机制登记JSON](r2/docs/research_route_registry.json)和[83份主报告及SHA256索引](r2/docs/research_history_sources.md)。按机制合并28条已运行但未形成稳定升级的候选、2条训练前门禁阻塞；另列128父数据门槛阻塞和应保留的局部正结果，不把报告数当失败方法数。

特别记录：FC/FF与时间门控、六loss筛查、JEdge/UCorr、Pair伙伴闭合、边界标记、MOMENT、READOUT及FP32复验、B早晚日程、CAPPED复现/梯度预算/半幅、课程、求解器、锚点去噪、蒸馏及多轮去噪、反馈/BLEND，以及最新FULL轮廓读取。新实验先查最接近的旧路线、实际失败边界和本次实质变化，避免改名重跑；无需因此先做一轮大审计。

FULL6000确有采用和轮廓改善，不能称FM完全学不动；同预算CONTOUR终点仍16/24采用、8/24合法接口、6/24联合正例，新增成功0，DEV联合0/12。Native/Geo/HYBRID/R2、USER构造、单次D0/GEO_C及来源方向修正的有限正证据保留。历史GT唯一款式评价、旧C20/C40回归否决规则不作为当前USER主任务的默认合同。

本轮仅报告/源码/维护记录阅读与文档整理；没有重评网格、重新图审、模型加载、训练、前向、生成或权重清理，没有commit/push或切默认。独立目录experiments/research_history_audit_20261014保存来源清单、读取范围、文档变更和完成核验；原83份报告hash保持。读取范围日志不冒充全量原始数组复算。
## 2026-10-14 FULL座面6000步训练曲线完成

R2_FULL_SEAT_LEARNING_CURVE_V1已实际完成。从EXP67 step6000的live生成器、匹配AdamW及RNG恢复，47个合格TRAIN SEAT_FULL父/条件，原世界坐标Native/T1+role+Geo、free-only FM、OT后HYBRID0.25，新增6000步至global12000。三条C不连贯输入在冻结前排除，其余31条来源Y-only风险保留记录；没有训练DEV、修原几何或按生成失败删条件。48000曝光，每父1021/1022次；零额外诊断前向或loss。

预定0/1000/3000/6000四点完整评价六父×四固定噪声，6000为主终点，不挑best。96逻辑RAW/96POST，其中88新增rollout=4400生成前向、88单次官方D0；另两冻结参照48RAW/48POST。全288原质量与边界单元、192主图审完成，AI_PROXY非盲，参照图审另存。RAW采用YES为12/12/13/16；最终新增4请求来自3父，无原YES丢失。RAW有限边长度加权距离4.762→3.282%L，18/24下降；POST合法接口输出7/6/6/8，合法边53/40/53/62，总合法边长47.955→46.266模型单位。不能将1000/3000全部共享边41/55冒称合法边。

最终采用改善与新增合法接口不重合。新增接口是2d n0/n2（采用UNCERTAIN），73 n1丢失；610虽新增两份采用仍无合法接口，608全部采用但仍无合法接口。POST CF/FF/I宏均增加1.217/3.864/0.765个百分点；DEV FF17.939→25.855%，不能认定代价可控或直接升级默认。接口训练vMSE在固定日志窗口间小幅下降，属于变化输入上的训练诊断，不是固定探针或自由生成误差。结论是原FM在FULL任务上仍有可兑现改善，但稳定接续与几何代价未同时解决；不自动延长训练、加loss、改模块或放宽焊接。

独立最终执行审计PASS；保留2d/610共8请求继承HALF描述字段的元数据sidecar，实际FULL数组、K/N、查询及评分身份核验正确，原冻结记录不重写。早期训练与0点采样阶段包络重叠69.24秒、逐调用区间交集47.14秒；这不是CUDA kernel并行证据。三训练段总墙钟153.37分钟，包含加载保存。GPU科学任务全部结束。

报告 experiments/r2_full_seat_learning_curve_v1/REPORT.md；图集evaluation/gallery/index.html与sections_gallery/index.html；最终checkpoint FULL_SEAT_step6000.pt（5f41df3d…af33）、推理FULL_S6000_generator.pt（86b38d35…c5cc），1000/3000恢复点保留。实际逐例CSV、曲线、图审及源码在同实验目录；小审阅包9,827,092字节，SHA256 d3cd93bedd7c652455ee60a7dba47015c5297c728847773ed1d0a5f261489a40；archive/latest.json为COMPLETE，独立schema_v2归档核验PASS（1143成员SHA、192主图审、118源码、26原评分定义、48预览）。原验证器对可选N/K空列的解析错误与修正另存；包未修改。三个冗余原日志/旧审计账本留本地，包中列原路径/hash，全部CSV与失败案例保留。恢复入口r2/pilots/full_seat_learning_curve_v1/resume.ps1默认只读；控制器阶段状态保留图审前的COMPLETE_PENDING_ACTUAL_IMAGE_REVIEW_AND_REPORT，整轮封存状态另见completion.json。未切默认、commit或push；固定实验结束，不自动启动下一轮。

## 2026-10-14 FULL训练前清理旧权重

用户要求在训练前清空不再用的旧检查点。已按逐文件白名单删除66文件，逻辑92.065 GiB、实际可用空间增加约87.7 GiB；23个保留大权重哈希复核。EXP67 step6000匹配恢复点、SMALL/CREF终点、官方Chair/D0、维护START/R2等保留。旧anchor/student/ABC、中间点、重复latest及退休续训对照按清单删除；对应旧分支不再可恢复。RAW、原数据、评分、报告和日志未动。清单与账本：experiments/r2_full_seat_learning_curve_v1/cleanup/。原报告只读；不回写旧结论。后续继续已授权的FULL座面6000步训练，无自动commit/push。
## 2026-10-13 FULL边界离线归因完成

R2_FULL_BOUNDARY_ATTRIBUTION_V1已完成：固定上轮11请求（3f n0–n3、73 n2、2d n2/n3、85 n0–n3）×SMALL/CREF×RAW/POST=44既有几何；源边界等弧段中点与单独顶点共27112查询。两个CPU worker，零训练/模型前向/新噪声/生成/几何修改/原质量重评分。独立逐点和聚合审计PASS，593原文件hash不变；11页实际非盲AI_PROXY、44行图审完成，旧采用标签不变。

主要发现：3f n0/n1/n3 CREF的轮廓错位已存在RAW，n1/n3以切向错位为主；2d n2/n3用途改善同时仍有显著轮廓间隙，不是单纯导出焊接问题。SMALL 3f n0 POST虽16/16边界顶点精确匹配，但仍有长边中段缺口；83.9%轮廓在.005L内不等于整圈接好。73 n2是近表面/局部点匹配与冲突混合情形。85 n0/n2/n3保留流程能形成局部接续的正例，n1单列。来源法线侧别、最近面CF/FF注释及多次法线命中均不可直接认证层语义或非法点。

优先建议后续研究RAW局部边界对应，不以高度接近替代接续，不自动放宽后处理；本轮未新增loss或训练。全边界距离为描述性中点近似，另存Lipschitz采样界限，S仍NA_NO_REQUIRED_INTERFACE，不恢复SJoin。此11请求按观察结果选例，不能替代全24请求成功率或新父泛化。

报告 experiments/r2_full_boundary_attribution_v1/REPORT.md；图集gallery_v2/index.html；逐点/边/面、配对表和法线命中均在同目录。小包R2_FULL_BOUNDARY_ATTRIBUTION_V1_REVIEW_SMALL.zip为8591321字节，含44几何、全部27112逐点CSV行、实际分析/原评分源码、hash/diff及11图预览；可解压纯NumPy复算全部44单元且已验证。恢复 r2/tools/full_boundary_attribution_v1/resume.ps1 已实际验证命中缓存，不重新生成。未切默认、commit或push；固定任务停止。

## 2026-10-13 完整规整面片冻结模型检验完成

R2_REGULAR_PATCH_CONTROL_V1已实际完成。六条既有SEAT_FULL，K=20/14/12/14/2/2；610b16与608b46为原生两三角矩形，3f7417后缘有凹口需单列。四份固定原噪声n0..3，冻结SMALL_C200/CREF_C200；32新RAW、1600生成前向、32单次官方D0，另严格复用16RAW/16POST；48RAW+48POST、96逐输出分数和实际AI_PROXY已交付。零训练，无新高斯、RBF、loss、蒸馏或多次D0。

最终V2实际图审：SMALL采用11/24、CREF14/24；POST合法接口输出7/24与3/24。真正改变FULL的16请求：SMALL采用5、合法接口3；CREF采用8、合法接口0。3f SMALL共边15→35，但有接口仍3/4；三个其他新FULL父均0/12合法接口，610两三角矩形未获接合。CREF的用途/布局改善仍与接续及CF/FF/I代价分离，未晋级或切换默认。HALF→FULL同时改变K/面积/边界/free/中心，不能称纯K消融；两原FULL是恒等缓存，不能记新增条件收益。

本轮发现Matplotlib分立C/F Poly3DCollection的平均深度排序会错遮蓝C。V1图及96原审阅保留；另以同几何逐像素CPU深度输出全部24页，统一实际重看，形成完整V2并逐项记录变化。显示纠正本身不算模型收益。后续图审不要仅因蓝色未显示便判未采用，优先深度正确的图并结合有限剖面；共面重层、次像素间隙和实际共边仍需独立数据。

报告 experiments/r2_regular_patch_control_v1/REPORT.md；优先看 evaluation/depth_gallery/index.html，原剖面在evaluation/gallery/index.html。恢复 python -m r2.pilots.regular_patch_control_v1.run --stage status/all；科学入口拒绝静默重跑未完成cell。全部新代码独立放r2/pilots/regular_patch_control_v1。输入/执行/数值审计通过；GPU调度进程曾重叠，但实际前向无重叠，不能宣称GPU并行加速；CPU评分2worker。无commit/push，固定任务后停止。

## 2026-10-13 CREF层结构离线诊断完成

R2_CREF_LAYER_AUDIT_V1已实际完成：预选11请求×SMALL/CREF×RAW/POST=44缓存几何，另6来源结构参照；两个CPU worker，零训练/模型前向/生成/新噪声/旧质量评分。102400逐点记录、全部有限法线命中与侧壁邻接、28张原图和44非盲AI_PROXY已交付。44旧dup逐点一致，210原文件hash不变；独立数值、聚合、报告审计PASS。

结论分情况：73b/n3 CREF支持正常厚座；3f/n2有近重合额外上皮；73b/n4、2d/n2、610b/n0混合；73b/n6薄垫/厚框/冗余解释仍不唯一。I_dup不能全部当重复面，也不能全部豁免；FF和合法接口代价保持。85b/n6有局部正例，n7接口5→0且采用仍YES。此为按历史结果选例的机制诊断，不是新泛化成功率。CREF不默认晋级，不添加dup惩罚loss。

报告 experiments/r2_cref_layer_audit_v1/REPORT.md；恢复 r2/tools/cref_layer_audit_v1/resume.py --verify；小包R2_CREF_LAYER_AUDIT_V1_REVIEW_SMALL.zip共9873148字节，含全部例/图/逐查询CSV/实际代码及精简几何，完整派生表本地保留。SHA256 0f407ffafda6f9f808a9a049eee4b36ea0379a0a324b7fc19f2eb4a0b8529c09。固定实验完成停止，无默认切换、commit/push。后续仍优先规整连贯K≥2的用户式输入；未自动构造或运行下一轮。

## 2026-10-13 后续 USER 输入优先规整、连贯的人类式局部表面

用户明确提醒：后续新构造的条件 C 尽量采用至少两个三角形组成、形状规整、连贯且真实用户可能提供的局部表面。例如两个相邻三角形组成完整四边形座面，或多个相邻三角形组成有清楚轮廓的座面／靠背表面。重点是有意义的完整局部区域，不是机械地凑 K≥2；不为满足面数而把孤立面、狭窄边条或无关部位拼在一起，也不预设所有合理输入都必须是平面矩形。

这是一项后续数据构造及主评价的默认偏好，不是宣布 K=1 不可生成或模型必须改为四边形 token。未来单三角、残缺角落等输入可作为单独登记的压力测试，不再默认以这些极简条件主导当前架构选择。现有冻结实验中的 K=1 输入、正负例、图审和评分继续完整保留；不回头删行、换父、改分母或将历史收益重新筛选。新主面板应在生成前按输入可用性确定。

此次同时审阅 CREF 反馈：它提出的是“布局更靠近 C，但可能仍生成了额外近平行表面”的待检验机制；已有 I_dup 不能区分正常薄底面与多余座层。后续若获授权核层，应比较已保存 SMALL/CREF 的 RAW/POST，保留原 I、查询和阈值，结合多层候选、局部延续和侧壁，不能只凭最近面或法向正负认证薄壁合法。7个明确收益＋2个YES→UNCERTAIN共9请求的36份缓存几何是可用诊断范围，本次尚未执行该新诊断。

本次只记录用户偏好及分析边界，不构造新数据、不训练／生成／重评分、不改loss或默认模型，不commit/push。


## 2026-10-11 来源方向恢复正式接入 denoise，已提交并推送 GitHub

用户明确授权将来源 C 角序恢复＋固定 C 的 free 定向固化进 denoise，并整理正结果上传。正式入口位于 r2/src/meshflow_control/，包版本0.2.0；新增 prepare-condition / denoise / orient。新 denoise 默认单次官方 D0 → 恢复C → GEO_C → 来源C恢复 → free定向；RBF只通过 --coordination rbf 显式启用，使用固定 --reference-length；旧 sample 仍输出RAW，旧START/native_t1入口没有静默切换。后续需要此合同的去噪使用 r2 正式入口，不再依赖旧实验tools导入。

prepare-condition 在规范化前保存用户C原角序/顶点ID及可逆排列，不需要完整GT；历史保存映射标 SAVED_MAPPING_ONLY。坐标相等不替代来源连接身份：GEO引入合并/拆分需报 CHANGED_BY_GEO_C，无来源ID报 UNKNOWN；未知属性拒绝静默丢弃。方向矛盾/非流形状态保留，不承诺水密、全局朝外或新增接缝。保留 proposal / canonical / restored_C / final 和逐阶段hash账本；不自动重试模型。

验证：115 CPU tests PASS；36仅定向＋24缓存D0/RBF链，4320旧数组逐字节一致，199旧文件hash不变；单次真实D0完整proposal与旧缓存maxabs=0，canonical18及final46数组一致；独立wheel安装验证128数组另PRE_GEO2一致，无旧工程导入。维护验证合计仅1次真实D0，0训练/生成器/新噪声/新科学评分；CPU迁移额外60定向24GEO12RBF，wheel额外2定向1GEO，GPU全链额外1定向1GEO，单元测试使用合成输入另计。GPU资源及峰值scope有记录，不把allocated/reserved说成进程显存。1次D0有readonly NumPy提示，未写输入/未重试。

公共研究综述 r2/docs/research_results.md 汇总R2、USER、单次D0、RBF、来源方向修正及学生/多轮的有限收益与代价，含30 CSV、来源hash与验证小收据；CF proper/illegal按原字段区分，完整选定面板不按输赢删行。RBF九条已有连接可一致定向，座面仍0/4、FF/I代价仍在；RBF非默认升级。完整入口文档 r2/docs/denoising.md，验证 r2/docs/denoise_validation.md。本地工程账本 experiments/r2_denoise_publish_v1/。

提交及origin/master已实核同一SHA：1c6d15b0e6abe5c7053590b7c80d292315e2b346。共68文件约1.21MB，权重/RAW/原始网格/旧完整实验和无关未跟踪代码未上传，旧报告只读；GitHub https://github.com/WANG-Ruipeng/MeshFlow/commit/1c6d15b0e6abe5c7053590b7c80d292315e2b346 。当前任务完成并停止，未自动启动后续实验。

# 当前工作区：R2 条件研究（保留历史 START 采样入口）

## 2026-10-11 来源C角序恢复与free定向导出完成

已完成R2_SOURCE_WINDING_EXPORT_V1：只读同36份D0/RBF/BLEND POST，按保存的来源排列取逆恢复C，再固定来源C整理free；保存36恢复及36最终NPZ/PLY，原坐标、无序索引面、源面ID/连接、面数及逐角追溯保持。六条件原45条双面C边均反向，规范化后28同向；恢复后36实例的168条同向C/C边全部消失。新合同恢复原输入角序，不再锁死内部规范化角序；没有改模型输入。

D0/RBF/BLEND原合法共边1/9/3条不变，方向一致共边1/3/2→1/9/3；RBF方向可达free面积宏均0.2643%→19.9459%，BLEND0.1584%→6.8409%，均达到各自无方向上限。RBF86ec n0/n1及bdd2 n0分别85.333/77.262/76.756%。这是已有连接的导出表示修正，没有新接缝；每组四座面仍无合法共边。CF/FF/I、A-VMR/PMR、距离及小角值在几何等价核验后继承，未重新评分。

每组各一份f596 n0仍有177面纯free奇偶环冲突；分量未改，不能归因于C。非流形边80/84/80保留，每组10/12涉及；没有水密/全局外向或流形认证。独立来源逆映射、XOR并查集、72 PLY、108图阶段及864VM行审计PASS；独立审计首次ASCII读取器遇二进制PLY，工程修正/失败版保留，无科学重放。2314保护文件hash不变。

执行36来源恢复/36定向，2 CPU worker；模型加载/前向/训练/生成/新噪声/GEO/完整质量评分/新图全部0。科学阶段墙钟37.469秒含末尾hash，不含初始准备核验。[报告](experiments/r2_source_winding_export_v1/REPORT.md)、[逐父噪声](experiments/r2_source_winding_export_v1/paired_per_parent_noise.csv)、[独立审计](experiments/r2_source_winding_export_v1/independent_audit/completion.json)。源码r2/tools/source_winding_export_v1，实际代码/diff/hash和小审阅包保留。仅独立导出实验，D0比较基准不变，未改默认/训练/loss、未commit/push；固定任务已完成停止。


## 2026-10-11 RBF有界对照与固定C方向审计完成

完成R2_C_RESTORE_BOUNDED_AND_ORIENTATION_V1：同12 USER_MIX缓存D0，12纯CPU非负局部BLEND、同GEO_C和原质量评分；D0/RBF质量24复用，36 POST均执行固定C绕序的free-only方向审计。零训练/模型加载/前向/新生成/新噪声；24三组两视角图、36非盲AI_PROXY和独立审计完成。1408保护文件/原权重hash保持，独立来源审计另96文件不变。

D0/RBF/BLEND严格AVMR为1.389/13.492/7.077%，FF12.019/12.550/13.410%，I3.687/4.390/3.988%，小角17.062/17.455/16.930%，有效共边输出1/3/2（各12）。BLEND压低866e n0与bdd29 n0的I尾部，但12/12边界距离均比RBF差，86ec两噪声代价增加；不替代RBF或D0。四座面仍无有效共边，PMR全部0。

方向解释需修正：6 DEV条件原C所选面的45条双面共享边原本相反，保存的逐面角点规范化使28条变同向（官方sort_vertices_and_faces逐面sorted）。5/6条件、10/12输出的固定C方向合同已矛盾；并非原始来源面本身有方向缺陷，也非RBF或GEO才制造。全冲突分量保持导致三组接口反向可达均无新增；不等于所有可达差异都是物理缝隙，也未否定局部free翻面可能。原无方向共边/相交/错层证据保留。未自动改变C合同或旧评分。

[报告](experiments/r2_c_restore_bounded_v1/REPORT.md)、[图集](experiments/r2_c_restore_bounded_v1/gallery/index.html)、[逐例](experiments/r2_c_restore_bounded_v1/decision_per_output.csv)、[六条件方向来源](experiments/r2_c_restore_bounded_v1/review/source_C_orientation_conditions.csv)。代码r2/tools/c_restore_bounded_v1；实际源码/diff/hash及小审阅包保留。优先建议澄清保持C几何是否允许重排角点，再独立验证全网格方向整理；该新合同未执行。没有默认切换、commit/push；固定任务完成停止。


## 2026-10-11 C协调恢复与有限反馈实验完成：保留局部信号，不晋级

已完成R2_C_RESTORE_COUPLING_V1：同12 USER_MIX RAW，官方D0与GEO_C固定；12 RBF离线协调恢复、24真实D0反馈查询（两更新，F_RAW恒定），零训练/新生成/新噪声。24新POST评分、D0/D3缓存、24两视角四组图和48非盲AI_PROXY审阅完成，独立链/数值审计PASS。

RBF严格A-VMR1.389→13.492%，有效共边输出1→3/12，但相反绕序仅1→2/12、完整路径可达面积仍小；FF12.019→12.550%、I3.687→4.390%，两主要I尾部为866e座面n0与bdd29背板n0。FEEDBACK的C误差12/12下降（宏均值0.043324→0.017103），F平均仅变0.001772，接合改进很小；四座面均无新增有效共边。RBF新增边在原GEO之后出现，不能称为自己生成了精确接口。维持单次D0参照，两个候选均不默认采用，不自动追加求解器/训练/蒸馏。

[完整报告](experiments/r2_c_restore_coupling_v1/REPORT.md)、[图集](experiments/r2_c_restore_coupling_v1/gallery/index.html)、[逐父噪声](experiments/r2_c_restore_coupling_v1/decision_per_parent_noise.csv)、[恢复说明](experiments/r2_c_restore_coupling_v1/README_RESUME.md)。代码r2/tools/c_restore_coupling_v1，原数据/模型/评分器/旧报告hash保持，零commit/push；固定任务已停止。


## 2026-10-10 锚点感知去噪器适配完成：未证明role增量，暂不晋级

已完成R2_ANCHOR_AWARE_DENOISER_PILOT_V1。官方Chair去噪EMA同起点，112条通过局部C/CY筛查条件/25父（59条仍有Y内部非法、10条Y内部UNKNOWN，不是全网格干净集），clean C＋逐角点独立sigma0.02噪声free，唯一free MSE；D_CONT_NO_ROLE和D_ANCHOR各1000更新、batch8/micro1。ANCHOR额外零初始化1024D known向量加在PE投影后；梯度与BF16实际加法均已核验，不能归因于角色完全没参与计算。生成器、既有48RAW、GEO_C和原评分器固定。

实际完成16000F/16000B/2000正式更新；另32F/16B/2丢弃预检更新；96真实＋54合成最终推理。累计GPU阶段3939.52秒（65.66分钟），单GPU worker顺序，无GPU重试。144 POST、全8阈值、逐边界距离、实际共边图、FF源面事件、56张原PNG及144 AI_PROXY审阅均完成。独立最终审计PASS，927保护文件/hash不变。

USER_MIX的D0/CONT/ANCHOR有效共边输出1/2/2（各12份），仍只在86ec一父；新增n0可达free面积约78.46%，但新增边均非相反绕序，要求相反绕序的覆盖仍1/12。四座面在2%L仍零匹配/无有效共边。FF12.019/13.534/13.411%，ANCHOR比D0+1.392pp、9/12更差；I3.6865/3.6580/3.6580%，严格A-VMR1.3889/2.3148/2.3148%，PMR全0。ANCHOR合成noisy free/interface MSE相对D0+22.40%/+21.33%，clean不必要移动也增加。两个适配器有共有保腿/去局部突刺正例，但无重复明确的role独有接续改善。

全48有效共边输出2→4→4，新增OLD/bdd29父；相反绕序有效边数仍2→2→2。CONT_USER的FF有改善，FROZEN有大退化尾部，完整四来源/父/噪声结果均保存。一次CPU缓存身份int/string断言在任何新quality调用前触发，只修元数据规范化；无GEO/模型/质量重放，原始错误与diff/hash保留。

[完整报告](experiments/r2_anchor_aware_denoiser_v1/REPORT.md)、[图集](experiments/r2_anchor_aware_denoiser_v1/gallery/index.html)、[逐输出表](experiments/r2_anchor_aware_denoiser_v1/report_tables/real_per_output.csv)、[恢复说明](experiments/r2_anchor_aware_denoiser_v1/README_RESUME.md)。[审阅包](experiments/r2_anchor_aware_denoiser_v1/R2_ANCHOR_AWARE_DENOISER_REVIEW.zip)9,540,620字节，小于10MB；全部CSV/56JPEG/55份实际源码/diff及账本保留，CRC/相对链接PASS，CSV采用ZIP-LZMA、其余DEFLATE。重复大JSON留本地并列OMITTED_JSON.csv路径/hash。代码位于r2/pilots/anchor_aware_denoiser_v1；两份最终完整权重/推理导出和500恢复点留本地。维持D0单次为比较基准，不自动晋级、蒸馏、放大role、改loss或默认模型；无commit/push。固定任务完成后停止。


## 2026-10-10 USER_MIX 多次去噪对照完成：暂不采用3/5次默认

同一12 USER_MIX RAW复用D1，实际新增48官方去噪前向得到D2..D5，每轮C原字节恢复、N不变，无中途Geo；D3/D5终点各一次同配置GEO_C、24 CPU评分与24两视角图全部完成。零训练/蒸馏/新生成/新噪声。ALL D1/3/5：FF12.02/8.02/6.12%，CF4.88/4.49/4.32%，I3.69/3.71/3.77%，小角17.06/16.28/16.06%；严格A-VMR1.39/2.31/3.37%，PMR全0，共边输出1/2/2，但新增共边未通过双面反向绕序检查。四座面在2%L仍0。

D5 FF改善10/12、退化2；I改善4/12、持平3、退化5。33ec seat n0在D3/D5两视角整组腿消失，D1/3/5 free数136/85/66，D5 FF=0却小角54.55%，C错层未解决。其他11请求未见同等级支撑消失，但图审非盲AI_PROXY，不能认证无损。实际FF改善不仅靠删面：140个D1非法面对两面保留后变合法接触/分离，也有新非法对。保留一次为比较基准，未自动改默认或蒸馏目标。

[结论](experiments/r2_multistep_denoise_v1/FINDINGS.md)、[完整报告](experiments/r2_multistep_denoise_v1/REPORT.md)、[全部图集](experiments/r2_multistep_denoise_v1/gallery/index.html)、384指标行/496顶点距离/48质量与拓扑/逐父噪声/24新PLY及实际源码diff/hash均保存。独立数值、网格和48前向链审计PASS，895受保护文件/官方权重hash一致。固定任务完成后停止；没有commit/push、新loss、模型切换或后续训练。


## 2026-10-10 完整后处理复评：官方Chair去噪＋C保护GEO_C

用户要求按完整生成流程处理既有mesh并复评。已对同一4模型×6USER条件×2噪声的48份RAW，执行48次官方Chair去噪前向、原C写回、固定GEO_C及48次CPU复评分；无新生成器前向/训练/噪声。48份C逐位保持，48PLY/NPZ一致。真实官方0030000.pt（约6.79GiB，SHA256 6a31f4c7…）及匹配db86bc3源码已找到并保存在本轮独立目录；不再把历史“缺denoiser资产”当作当前状态。固定缩放0.95×0.3762；最终C保护几何导出替代原无约束量化导出，属于官方发布权重的C保持适配，不是原脚本逐位复现。

48/48 FF均下降，FROZEN/OLD/USER/MIX从90.36/91.97/88.96/89.65%降到16.11/15.30/15.36/12.02%。MIX CF5.60→4.88%、I3.77→3.69%、小角17.18→17.06%；座面小角31.93→33.28%。FF改善包含1886对MIX原非法面对两面均保留后变为合法接触/分离，不能只归为删面，也不能当体积或完整形状质量。

POST严格模型1e-6 A-VMR四组为1.85/3.90/3.51/1.39%，PMR仍全0；1%L的MIX从6.75降到5.56%。实际共享C边仅OLD与MIX各1/12；MIX四份seat在2%L仍0。后处理有很大free内部几何整理收益，仍未普遍接好C；今后评价保留RAW/同一POST并列，不自动采用新默认。24张两视角图全部AI_PROXY审阅，记录33ec细腿损伤；不能只看较低FF。

[结论](experiments/r2_postprocess_avmr_pmr_v1/FINDINGS.md)、[完整报告](experiments/r2_postprocess_avmr_pmr_v1/REPORT.md)、[图集](experiments/r2_postprocess_avmr_pmr_v1/gallery/index.html)、逐例CSV、最终48PLY/NPZ、原评分代码快照/diff/hash和资源账本均已保存。独立复算768指标/496距离/192汇总/96拓扑一致。准备阶段误写1份父NPZ，已按历史ZIP原SHA256恢复，77原输入终态hash均一致；事故和恢复有独立记录，正式48前向无重试。未改loss/默认模型/原评分器，无commit/push；任务完成并停止。


## 2026-10-10 A-VMR／PMR 首轮离线估计完成

用户后续授权实际估计；已在独立目录 experiments/r2_avmr_pmr_offline_v1 读取4模型×6USER条件×2噪声的48份既有RAW，零模型加载/前向/训练/新生成/POST/S_join。代码 r2/tools/avmr_pmr_v1/run.py 为固定面板CPU入口，公式和8档阈值先冻结；77原文件hash未变，8项指标正反例通过，独立实现核对384评分行、496顶点距离、96汇总与288配对均0差异。

全部C原始顶点ID开放边界作为描述性评分域，尚无用户必接接口；论文评分尺度兼容性UNKNOWN，因此不是论文表数值复现或接合成功率。严格模型单位1e-6下四组A-VMR/PMR均0；固定L的1%档FROZEN、CONT_OLD、CONT_USER、MIX分别为22.72/8.33、17.76/0、17.03/0、6.75/0（%）；2%档分别41.27/25.00、37.00/8.33、33.99/16.67、17.13/0。MIX4份座面在2%档边界顶点匹配仍0，但不等于整片表面都远离F。低I不代表顶点匹配改善，匹配高也不保证无重层或真实共边。

[结果解读](experiments/r2_avmr_pmr_offline_v1/FINDINGS.md)与[定义及全阈值表](experiments/r2_avmr_pmr_offline_v1/REPORT.md)；逐输出、逐顶点、逐父/噪声CSV、实际代码/diff/hash与CPU账本均保留。固定任务完成后停止；未改原评分器/RAW/旧报告，未改变loss或默认模型，未commit/push。下方NOT_IMPLEMENTED/NOT_RUN仅描述此前“先记下来”的历史阶段。

## 2026-10-10 评价选择：记录 A-VMR／PMR，不推进 S_join

用户明确要求先记下 MeshFIM 的 A-VMR（Average Vertex Matching Ratio，平均边界顶点匹配率）和 PMR（Perfect Matching Ratio，全部边界顶点匹配的样本比例），不继续推进新增 S_join 指标。优先保留论文原定义与来源，后续采用时明确边界集合、坐标尺度和容差；这不是已冻结的完整论文主表。

定义及适用限制见 [指标库补充记录](native_t1/docs/control_metrics.md#meshfim-boundary-metrics)。匹配对象只能是新生成 F 的顶点，不能包含复制的 C；A-VMR／PMR 衡量边界顶点匹配，不认证实际共边、无穿插或完整主体接合。原论文容差为 1e-6，不能不核对坐标尺度就直接套入本地模型单位。

本次仅更新持久化文档：两项本地接入状态 NOT_IMPLEMENTED，运行状态 NOT_RUN；未新增评分、训练、模型前向、生成或后处理。旧 S、I、评分源码、RAW和历史报告保留；不自动改loss、切换默认模型或commit/push。

## 2026-10-10 USER主导混合从官方初始化实验完成，保留候选

已完成R2_USER_MIX_FROM_OFFICIAL_V1：单臂从官方EMA/fresh AdamW step0训练6000步，Native/T1+role+Geo、FM和全程HYBRID0.25；25父117USER+395OLD，每批6USER/2OLD、父曝光平衡。实际48,000F/48,000B/6000更新全部成功，无OOM/重跑；训练8556.64秒。12新增USER Euler50/600采样F，核验复用36历史同请求结果，共48评分、24PNG和48逐输出/36配对AI_PROXY审阅。

ALL结果MIX CF5.60%、FF89.65%、I3.77%、小角<5°17.18%；相对FROZEN为−2.72pp、−0.71pp、−6.06pp、+6.81pp。座面小角31.93%，不可称全面更好。866e完整座面相对FROZEN/OLD双噪声C采用改善，33ec座面一好一坏，四靠背无明确C用途提升。对CONT_USER，I−8.65pp但FF+0.70pp、小角+1.37pp。未以旧随机任务退化作门槛，未做唯一GT排名。

24项CPU测试PASS，执行审计10部分PASS，2236数值单元独立回算一致，1059份保护文件未变。正式全量权重、推理导出和5500恢复点保留；没有默认切换/自动commit/push。当前候选用于继续研究，不认定为整体更优基线；六父开发面板与历史26/25父、6000/7000步差异限制因果及泛化结论。后续优先候选是固定权重在未看过的USER条件上验证采用收益和小角代价，本轮没有追加执行。

报告：[R2_USER_MIX_REPORT.md](experiments/r2_user_mix_from_official_v1/R2_USER_MIX_REPORT.md)；图集：[gallery/index.html](experiments/r2_user_mix_from_official_v1/gallery/index.html)；完整CSV、原始评分代码小文件、diff/hash、资源与调用账本均在同目录。代码仍为r2/pilots/user_mix_from_official_v1/独立实验入口。固定任务已完成并停止。


## 2026-10-10 USER 主任务单版从官方初始化实验已授权

用户明确要求按 USER 为主的方向实际训练一版。冻结 R2_USER_MIX_FROM_OFFICIAL_V1：单臂 R2_USER_MIX，从同一官方 Chair EMA 初始化 Native/T1+role+Geo、fresh AdamW step0，FM和HYBRID0.25全程6000更新。25共同父，117 USER条件＋395旧条件；每批8不同父、6USER/2OLD、4clean/4width，父曝光严格平衡。75%/25%为本次探索选择，不称最优，也不是梯度或loss贡献的固定权重。

仅新增现有6 USER DEV×2冻结噪声的12份Euler50 RAW；此前R2_FROZEN、CONT_OLD、CONT_USER各12份同请求RAW/评分核验后复用，共48结果。旧随机裁剪回归不新增生成、不作为门槛。无新loss、B、教师、POST或唯一GT排名。历史参照父数和总步数不同，只回答新配方是否值得继续，不作混合比例的独立因果证明。

独立代码 r2/pilots/user_mix_from_official_v1/；独立产物 experiments/r2_user_mix_from_official_v1/。run.py提供可恢复分阶段入口，按账本计实际调用；失败不自动重放或降低设置。固定6000终点评价，不按中间图选点。任务完成后停止，不自动更换默认模型或commit/push。当前执行状态以该目录status/账本和最终报告为准。

## 2026-10-10 后续主任务以 USER 型部件条件完成为准

用户明确调整后续研究目标：主要研究有实际用途的 USER 型输入，当前为座面／靠背等局部面片，后续可扩展不同靠背、椅腿等部件。后续主评价围绕这些输入展开；旧随机裁剪 C20/C40 任务的保持能力不再作为晋级或否决门槛，不为保留旧任务而强制安排旧面板生成预算。

此前用户提出把 USER 条件混入原 R2 数据作为增强，这与评价目标的调整分别处理。旧条件可以作为可选训练辅助，但是否保留、混入多少，只根据其对 USER 条件采用和生成质量的帮助决定。此前讨论的 75% 旧条件／25% USER 只是未执行建议，不是已确定配方，不因维持旧任务而锁定该比例。继续采用 R2 训练路线；具体混合比例、更新预算及实验注册另行确定。

主判断仍包括：C 是否合理参与完整形状、整体是否完整合理，以及相交／重叠候选、退化面与明显碎片等代价；不恢复唯一 GT 重建排名。既有 pilot 的原始结果和报告保留，旧面板退化仍是历史观察，但不能单独否决面向新任务的方法；新 USER 面板自身的质量问题仍需如实评估。该条是研究方向记录，没有启动训练、生成或评分，没有切换默认模型或自动 commit/push。

## 2026-10-10 R2 用户式条件续训 pilot 完成，暂不晋级 USER

已按用户授权实际执行 R2_USER_CONDITION_FINETUNE_PILOT_V1：同主R2 full step6000/完整AdamW，25共同父、OLD395旧C与USER117活动C，各1000更新（global7000）；Native/T1+role+Geo、FM、全程HYBRID0.25。正式2000更新/16000F/16000B；小K预检8F/8B/0更新；68新增Euler50+16核验复用RAW，84统一CPU评分、56PNG、84逐输出及28配对AI_PROXY审阅均完成。训练双worker波次1818.62秒，设备VRAM观测峰9202MiB，无OOM/模型调用失败/重跑。579执行核验PASS，3846保护文件未变。

结果不支持用USER替换主R2：新DEV只有2条座面各一个噪声C用途更相容，无双噪声一致收益，4靠背未见清楚C用途改善。USER相对OLD：新DEV CF −0.30pp、FF −3.02pp，但I +4.77pp、free小角<5° +5.35pp；旧回归CF +1.12pp、I +0.60pp、小角 +1.83pp。图审旧面板整体1改善/13接近/2退化，AI_PROXY非人评。I_dup为近并行覆盖候选，可能包含合法薄壳；不能直接当非法重复或采用率。

两份full终点、两份generator、两份仅恢复500检查点均保留。结果报告：[R2_USER_CONDITION_FINETUNE_REPORT.md](E:/MeshFlow-Control-artifacts/experiments/r2_user_condition_finetune_pilot_v1/R2_USER_CONDITION_FINETUNE_REPORT.md)；图集：[gallery/index.html](E:/MeshFlow-Control-artifacts/experiments/r2_user_condition_finetune_pilot_v1/gallery/index.html)。完整逐例CSV、来源源码/diff/hash、账本及DELIVERY在同独立目录；新代码r2/pilots/user_condition_finetune_v1/。没有自动commit/push或默认模型切换。

后续训练继续R2路线，当前117池是保留的局部适配证据，未定为新默认数据。若继续，优先候选为新旧条件混合回放：检验完全替换C20/C40的分布偏移是否造成部分回归，同时能否保留局部C采用信号；此假说未启动、未追加比例/预算。固定任务已经结束，不从500点重放已消费更新。


## 2026-10-09 R2 用户式条件续训 pilot 已获授权并开始执行

用户明确批准执行 R2_USER_CONDITION_FINETUNE_PILOT_V1；此前数据准备交接中的 training_authorized=false 仅描述旧准备阶段，不再是本次执行的授权状态，原交接文件保持只读。
新代码位于 r2/pilots/user_condition_finetune_v1/，独立输出 E:/MeshFlow-Control-artifacts/experiments/r2_user_condition_finetune_pilot_v1/。冻结 registration.json（SHA256 e0ad17e15915efba7dd5e20c668e99b27ed40e7630fcdaf8bd8690325abd5961）。
固定主实验 R2 full step6000（SHA256 89f49e014615eb4a41e052ed90719a2580ead46418997c268e35e09ff37a7541）及完整 AdamW/RNG；25共同父，OLD395旧条件、USER117活动条件，各1000更新/8000曝光，Native/T1+role+Geo、纯FM、全程HYBRID0.25，新增辅助头/教师/后处理均0。
评价固定6新DEV×2噪声与8旧请求×2噪声，三模型共84输出，正常新增68条Euler50、旧冻结R2的16RAW核验后统一CPU重评分。预检上限16F/8次零更新B，500步只作恢复，不按图片选终点。新旧面板分别汇总，完整定性审阅，不做唯一GT重建排名。
当前为执行中，实际完成以该输出目录的各阶段账本/最终报告为准。固定预算完成后停止；不自动增加训练、切换旧采样默认模型或commit/push。

## 2026-10-09 R2 用户式条件数据准备完成（后续训练路线）

后续训练统一使用 R2：Native/T1 + role + Geo、FM、全程 HYBRID lambda=0.25。新条件池替换旧 C20/C40 构造，不将区域面积25/50/100%混同于父面数比例。未来分支起点固定主实验完整 R2 step6000 及对应 AdamW；本轮没有加载模型、训练、OT、抽噪声或生成mesh。

实际数据位于 E:/MeshFlow-Control-artifacts/experiments/chair_r2_user_conditions_v1/。权威活动清单为 active_conditions.json，交接为 dataset_handoff_active.json，查看 DATA_PREPARATION_REPORT.md 和 gallery/index.html。25共同训练父、47区域、117训练C/Y（56座面/61靠背），6新DEV条件；8旧DEV条件×原2噪声继续保留。原122train+7DEV候选已完整保存；6条极窄边条在实际看图后标HOLD（5train/1DEV），不进入主池。400a两类区域未能可靠分区，因此两条未来续训须同步排除该父；其余已知14个源缺陷父没有自动删除。语义为AI_PROXY，源相交/重叠/UNKNOWN仍逐例记录，不能称作无交干净数据。

129份C/Y都只由原FP32面与严格补集构成；6154项原导出验证、669项活动子集验证均通过，289个保护文件hash未变。图集包含每候选两视角M/R/C/Y/C+Y，原CSV、面ID、源映射、完整面对诊断和代码快照均保留。准备代码在 r2/data_tools/user_conditions_v1/；run.py --stage check 可回读核验并记账。

新DEV仅6条件，未来三模型新旧合计84逻辑输出；冻结R2旧16 RAW的身份和评分合同满足复用条件时新增68。新噪声只是未物化槽位。公共R2入口仍绑定旧26父/C20-C40/6000步，下一次需独立续训合同与入口，不能绕过恢复检查直接塞入此清单。此次停止在数据准备，等待数据审阅后再启动两条1000步；不自动改旧START采样默认，不commit/push。

## 2026-10-09 用户式条件数据准备与后续训练路线

用户明确要求后续训练统一走 R2。这里指以 R2 的 Native/T1 + role + Geo、FM、全程 HYBRID lambda=0.25 为配方基础；当前任务仅准备新的条件池，不把新座面/靠背条件强行凑成旧20%/40%面数。未来续训候选起点固定主实验 R2 完整 step6000，不自动选择复验权重，不恢复 B 或其他辅助loss。

本次已授权构造：现有26训练父，每父座面/靠背各至多一个核验区域、区域内25/50/100%面积参考的嵌套真实面子集，最多156条，去重/NA后如实计数；8DEV每父至多1新C，另登记8旧C及既有两噪声作回归。数据和审阅图独立放 E:/MeshFlow-Control-artifacts/experiments/chair_r2_user_conditions_v1/，源码在 r2/data_tools/user_conditions_v1/。零模型加载、零前向、零训练、零OT、零新Gaussian、零完整网格生成；只从原父网格取原面及严格补集，不修复或重三角化。先交付区域/条件清单、质量证据和检查图，待数据清单确认后才启动两条1000步续训。旧默认START采样入口与保留权重不自动切换；本轮不commit/push。


## 2026-10-09 R2 代码独立整理（已按用户要求提交并推送）

用户确定下一步研究配方为 R2 全程混合 C20/C40 + 全程 HYBRID，并选择现有 GitHub 仓库内独立的 r2/ 目录。使用说明见 [r2/README.md](r2/README.md)，三条训练/采样路径见 [训练结构](r2/docs/training_structure.md)。

实际 GitHub master 核查为 8961b7a41fc153b5b45f0c8903370c6bfd1cf04f，与本地 HEAD 一致；此前 R2 位于无提交、无 remote 的 E:/MeshFlow-Control。此次复制计算依赖并保留其源码字节，新增只公开 R2 的入口；旧项目及外部权重/数据/RAW 不修改。主实验和确认实验两套 R2 均保留，没有自动选定默认终点，native_t1 的 START 采样默认继续保留。

当前 R2 训练接口仍绑定历史 26 父条件池、6000 步、20%/40% 面数占比；新 C 构造、洗数据与追加训练尚未执行。此轮只做 CPU 测试、权重严格加载和冻结输入抽查，不代表重训或生成等价性已经验证。证据位于 native_t1/maintenance/r2_port_20261009/，公开验证说明见 r2/docs/validation.md。

用户随后明确要求推送 R2，已将 72 个 R2 源码/测试/文档及首页入口文件提交并推送到 origin/master：573ad4dc960de26b1569595a4bfa736abe6d1e39。远端 SHA 已再次核实一致。入库 blob 保留 51 个源文件的原始 SHA256；未提交权重、数据、其他未跟踪文件，未运行新训练或生成。推送记录见 native_t1/maintenance/r2_port_20261009/push_receipt.json。

## 2026-10-08 GEO_C 可选后处理正式入库

保持 C 的固定 GEO_C 几何整理已迁入正式模块 native_t1/postprocessing.py，入口为 `python -B -m native_t1 postprocess --input RAW.npz --known-faces K --out NEW_DIR`。NPZ默认键output，也可显式--input-key raw；输入FP32、前K面为C、使用原模型坐标。仅需NumPy，CPU执行，不读取权重；另存post.npz、indexed post.ply和postprocess.json，保留来源映射、删面原因和C位模式回读。默认START采样仍输出RAW，不自动执行后处理。

本轮完成并推送13个源码/测试/文档文件到origin/master：8961b7a41fc153b5b45f0c8903370c6bfd1cf04f。公共CPU测试130/130通过；32个原后处理案例416项数组逐字节一致，统计和逐面记录一致（耗时除外），97个旧输入/结果hash未变。真实模型加载/前向、训练实验、新生成和重评分均0；原实验报告和评分器未改。C保持不意味着原vertex-ID图、水密、无相交或稳定接缝。

用法和固定尺度见 [正式后处理文档](native_t1/docs/postprocessing.md)，本地迁移账本位于 native_t1/maintenance/postprocess_port_20261008/。此次入库未包含任何权重、网格、实验数据或此前无关的未跟踪文件。独立MeshFlow-Control仓库没有在本轮提交或设置remote。

## 2026-10-07 持久评价记忆：完整指标库

按用户要求审阅并保存 [Chair Control 完整指标库](native_t1/docs/control_metrics.md)，共35类指标，包含定义、适用任务、GT依赖、尺度、失败/未决和汇总规则。AGENTS.md 已加入后续评价前的阅读入口。

任务目标是给定真实面片 C 的多样化合理补全；来源重建是辅助诊断。按已授权预算冻结适用项并保存完整结果，主文可选择三个有贡献证据的指标，但目前未确定具体三项。无GT、无指定接口、无外露要求的任务分别不强算来源重建、S、E；优先客观量，人评另列。

已修订附件中的实现状态与关键口径：C坐标/绕序保持不等于原索引图保持；旧CF比例只计proper，FF非法比例另有并集定义；已有共享边数覆盖不等于新Gamma弧长覆盖；S角度为≤20°；E确定参考域与全局未知域上下界分开。新定义和文献项不冒称已实现，旧评分源码/结果不回写。

本轮仅写文档记忆，未运行训练、模型或特征前向、生成、评分，未改默认START、loss、权重、RAW及历史报告，未commit/push。

## 2026-10-07 当前决策：回到原始 START，结束 CAPPED 探索

按用户明确要求，以原始 Chair HYBRID START / total5000 继续作为 baseline。正式默认 profile 一直为 chair-hybrid，已实际核验解析及一次 CPU strict load；零模型前向、零训练、零生成。START 两条硬链接路径及完整 AdamW/RNG/输入流状态均保留。

已删除 chair_control_feedback_auto_v1、chair_capped500_gradbudget_v1 和 chair_capped500_half_v1 的22份实验权重，以及两处正式 OT 派生缓存的16,000个NPZ，共33,426,755,776逻辑字节（约33.43GB）。OT尝试日志、训练计划、小型测试缓存、原始RAW、冻结条件/噪声/查询、反馈池、全部评分/报告/CSV/图集/源码和审阅ZIP保留，43,405个保留文件前后hash一致。

相关历史代码留作评分依赖和证据，不自动运行旧恢复入口，也不因权重/缓存缺失而重训或重生成。原报告中的旧文件存在状态以本轮删除清单为准。官方权重、Geo4000、START、旧显式FM/JEdge5及其它既定保留checkpoint不在本轮删除范围。没有修改正式代码或commit/push。

详见 [清理报告与完整账本](native_t1/maintenance/capped_retirement_20261007/CLEANUP_REPORT.md)。

维护入口为 `native_t1/`。2026-10-05 按用户明确要求退回并正式接入 START：原 OT_HYBRID pilot1000 / conditional_total5000，Native T1 + C-only Geo + FM，训练 coupling lambda=0.25。默认采样 profile 现为 `chair-hybrid`；MOMENT V2 及其 finite-surface 前序活动包已封存并移除。

## 当前使用方式与边界

```bash
python -B -m native_t1 sample --condition /path/to/C.npy --num-faces 163 --seed 2452526625 --out native_t1/runs/chair_start_sample
```

总 N 必须显式指定 128..256；C 为原坐标尺度的有限 FP32 数据，0<K<N。使用原 MT19937 free-face 噪声、50 步 Euler、BF16 forward / FP32 参数与积分，已知 C 在 51 个状态均精确保留。详见 [START 说明](native_t1/docs/chair_start.md)。

正式本地权重为 `native_t1/checkpoints/chair_hybrid_start_total5000.pt`，是原 `chair_hybrid_coupling_v1/checkpoints/OT_HYBRID_pilot1000_total5000.pt` 的硬链接。文件 SHA256：`d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80`。原完整 AdamW、RNG 和输入流状态保留；推理不恢复优化器。Git 只纳入正式源码、测试、文档及身份 manifest，权重继续本地保存。

旧 `--profile fm-geo`、`edge5` 等保持显式可用的 N112 路径。`train` / `train-edge5` 仍是显式预算、fresh AdamW 的旧 N112 训练入口，不是 START 续训入口。任何新训练需要独立的数据、coupling 和预算合同，不因改变默认模型自动开始。

## 保留的正式和历史权重

|角色|路径|累计步数|
|---|---|---:|
|START 默认|native_t1/checkpoints/chair_hybrid_start_total5000.pt|5000|
|旧 FM 显式 profile|native_t1/checkpoints/ablations/fm_cumulative3000.pt|3000|
|JEdge5 显式可选|native_t1/checkpoints/jedge5_cumulative3000.pt|3000|
|旧共同起点 G0|native_t1/checkpoints/tgeo_g0_cumulative2500.pt|2500|
|surface 旧参照|native_t1/checkpoints/ablations/surface_cumulative3000.pt|3000|
|Edge-only 旧参照|native_t1/checkpoints/ablations/edge_only_cumulative3000.pt|3000|

## 2026-10-05 MOMENT 封存完成

删除 MOMENT/H_CONT 正式 V2 权重各 5 份、8000 个派生 OT NPZ 和三个退役活动源码包，共 8061 文件、15,677,075,950 bytes（约 15.68 GB 逻辑字节）。源码先行归档；原始结果、报告、逐例 CSV、日志、评分源码快照、决策 ZIP 和小型合成 CPU 测试 fixture 保留。

11,488 个保留文件前后 SHA256 一致。START CPU strict load 通过，公共 CPU 测试 109/109 通过，删除后 CLI 导入通过；真实模型前向、训练更新和生成均 0。根据本轮用户明确授权完成本地 Git 提交，没有 push。详见 [清理报告](native_t1/maintenance/moment_retirement_20261005/CLEANUP_REPORT.md) 和其中的账本。

后续仍遵守 [AGENTS.md](AGENTS.md) 的资源调度与保护边界。以下是按日期保留的历史记录；其中旧的默认模型、HEAD、实验状态及已删除路径以其记录日期为准。

## 2026-10-02 Pair清理

删除两处隔离目录：`experiments/mf_h1_local0/geometric_pair_closure_pilot/`、`native_t1/runs/geometric_pair_closure_20261001_171356/`。四份Pair step250/500 checkpoint、诊断及正式原始数组、连接候选、活动实验源码均退役。

报告、results、逐输出表、训练记录、固定图、协议和code.diff保存在 [精简档案](native_t1/maintenance/pair_cleanup_20261002_100559/CLEANUP_REPORT.md)。档案内历史报告的checkpoint/raw路径是执行时记录，现已删除；code.diff可恢复实验代码，但不恢复已删除的权重/数组。

官方权重、原数据、G0/FM/JEdge5和其他历史实验不在删除范围。本轮没有模型前向、训练更新、完整生成、reset/clean、commit或push；公共代码保持HEAD `5c11eb2`。权重、数据、runs、maintenance及本地WORKSPACE说明仍Git忽略。

本地已准备输入为 `output/t1_publish_20260930/chair_n112.npz`；公开代码可依据官方数据和固定recipe重新准备。当前证据仍来自两个观察过的父对象，不代表新对象泛化、水密或全指标优势。

## GPU任务调度约定（2026-10-02）

后续每次实验及新增GPU任务前按 [AGENTS.md](AGENTS.md) 检查显存、GPU忙碌率、已有进程及CPU/RAM；独立任务优先尝试有资源约束的并行，不再默认逐臂串行。并发数来自同类任务实际峰值和吞吐，保留显存余量，逐个启动并动态复核。

未知峰值时先观察一个已授权任务，不额外新增探针训练。训练、评价、采样分开估算；不改变microbatch、精度、随机流、确定性或实验预算来塞入更多任务。CPU匹配/OT/离线审计可与独立GPU工作重叠，独立输出和固定seed不随完成顺序变化。

本次只读取GPU状态和记录执行偏好，没有启动训练、采样或新的调度服务。检查时RTX5080总显存16303 MiB、可用13080 MiB、GPU利用率0%；这是空闲快照，不是运行峰值或固定并发数依据。

已有显存参考：`experiments/mf_h1_local0/outputs/tgeo_m10_20260930_173200/results.json` 中两臂 `arms.*.peak_cuda_allocated` 为约2.520/2.521 GiB；其72条 `generated[].sampling.peak_cuda_allocated` 为约0.536 GiB。这只是历史PyTorch allocated峰值，不含完整CUDA进程占用，也不是当前JEdge5所有loss的峰值；不能按16GB直接相除定并发。当前采样入口记录allocated峰值，公共训练入口尚无完整reserved/进程峰值记录，后续须随首个已授权任务补齐观测。allocated属于reserved的一部分，不把两者相加重复计算。

## 2026-10-02 正式多对象实验前的数据盘点

本地 `experiments/mf_h1_local0/assets/shapenet-03001627` 实际已有 3135 个官方 UID、3125 组精确几何；官方 train/test 为 2822/313，未发现 UID 或精确几何跨划分重复。近重复与同源家族尚未排查。当前两对象 N112 是维护训练入口的限制，不代表本地只有两个对象。正式实验应区分条件微调未见与官方预训练未见，不直接把旧双对象模型作为公平的纯 T1 / Geo 比较起点。

面数为 28–800，中位 470；原始索引拓扑/退化面诊断及逐对象记录见 [数据扫描](native_t1/maintenance/temp_cleanup_20261002_160601/dataset_scan.json)。先准备现有椅子数据即可推进；跨类别可补下载 MeshFlow 的 `shapenet-cls.tar.gz`（25.48 MB，历史归档清单确认包含 bench/chair/lamp/table），语义部件控制可申请 PartNet。具体链接、体积与适用边界见 [数据下载清单](native_t1/maintenance/temp_cleanup_20261002_160601/DATASET_DOWNLOADS.md)。本轮未下载数据、未启动训练或采样。

历史 checkpoint 已按用户明确批准的固定清单完成清理：删除 27 份退役权重及 30 个临时文件，共 35,971,879,861 bytes（约 35.97 GB）。official、无 Geo T1 500/1000/1500、canonical 五份及关键 Base/Geo 和 EMA 终点继续保留。3,711 个关键文件 SHA256 校验一致，原盘点中清单外缺失为 0；实验报告、指标和原始采样证据均保留。已删历史终点不能直接重新采样，相关分支若重新研究需重训；历史报告内旧 checkpoint 路径以 [清理报告](native_t1/maintenance/temp_cleanup_20261002_160601/CLEANUP_REPORT.md) 的删除记录为准。
## 2026-10-02 已接收 ShapeNet class split 归档

用户下载的 `shapenet-cls.tar.gz` 已从 Downloads 移至 `downloaded_data/archives/shapenet-cls.tar.gz`（25,483,419 bytes）；SHA256 `06707caff4ce465f402e6b69f81056204c579921d371eb2c9e4e9a3929f7aef2` 与官方元数据一致。来源及校验记录在同目录 `shapenet-cls.manifest.json`。原始归档已保留，尚未解压；既有 chair 数据不变，后续可从归档提取缺失的 bench/lamp/table 类别。
## 2026-10-04 Chair 第一档旧恢复点清理

经用户批准，删除 chair_domain_control_v1 的 T1/Geo/Full_latest.pt（phase2800、total3800），以及 control_mvp48_budget_single_v1 的 T1/Geo/Full_latest.pt（total4800）和 T1_fit2_probe_latest.pt（probe800），共 7 份、10,950,114,685 bytes（约10.95 GB）。只执行已评估的第一档；第二、第三档未删除。

全部固定中间评价状态和终点继续保留。下一步 Hybrid Coupling Pilot 的指定起点仍为 native_t1/runs/chair_domain_control_v1/checkpoints/Geo_phase3000_total4000.pt，模型与完整优化器状态未改变，文件 SHA256 为 190a16e2aec96a1f84f63460b5f97b00e23499338102f82e7530218ecb185ebf。

10,773 个受保护文件前后 SHA256 一致，清单外意外缺失为0；原数据、冻结C/噪声/GT查询点、raw、源码、报告和归因包保留。旧清单与同名JSON保留为历史证据，其中上述七个PT路径已明确退役；不能再用它们恢复旧步数。详见[清理报告与账本](native_t1/maintenance/weight_cleanup_stage_a_20261004/CLEANUP_REPORT.md)。本轮没有模型加载、前向、训练或生成，没有启动下一步试验。


## 2026-10-04 Chair Hybrid Coupling 有限试验完成

`native_t1/runs/chair_hybrid_coupling_v1/` 已实际完成同源 Geo4000 两臂各1000更新、固定A/B/C与响应诊断。两新终点为本实验 pilot1000/total5000，身份见 checkpoint_manifest.json；不要与历史预算实验5000步混称。未切换FM默认模型或JEdge5可选模型。

实际总计2000更新、432登记请求（18原始NA）、292完整采样、122历史raw缓存、2400 BF16交叉诊断、72 FP32诊断及10预检前向；总前向33082，模型失败/未决0。866个保护依赖前后hash一致；冻结新实现未改变，训练曝光、主指标算术及三个检查点轨迹独立审计PASS。

结果为 GEOMETRY_SIGNAL_WITH_TRADEOFF：HYBRID相对BASE验证A20固定查询MSE下降约40%，6/8父改善，覆盖与p95改善；但小角面增加、跨噪声控制分歧、40%上下文部分指标及已观察留出退化。优先下一步是现有raw的小角面收益归因，检验误差收益是否主要由新增细长面贡献；该后续未执行，没有追加训练、loss或求解器扫描。

详见本轮 CHAIR_HYBRID_COUPLING_REPORT.md 与 DELIVERY_INDEX.md。纯CPU审计读取器/报告文案修正单独保留原版、diff和哈希；正式实验没有静默重跑。当前固定任务已停止。


## 2026-10-04 Chair Finite Surface Attention 门禁停止

`native_t1/runs/chair_surface_attention_v1/` 已完成资产登记、独立实现、CPU门禁及受限真实预检，结论为 IMPLEMENTATION_BLOCKED。共同起点固定为上一轮 OT_HYBRID pilot1000/conditional_total5000（SHA256 d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80）。没有新增正式训练权重，没有切换默认模型。

192点求积在4个真实训练任务均未通过0.01精度门禁，7504个普通query/known对中1459个失败；误差集中在t=0.9/1。96个密集参考对子集中另有1个未达到0.001收敛要求，故保留求积规则差异的表述，不声称真积分误差界。

实际真实预检仅10前向、1次零更新反向；新增显存日志器重复传入time_unix/pid导致停止，模型调用失败0。没有修正后重跑，真实三臂成本比较与MOMENT/SURFACE接线检查均未完成。H_CONT参数、AdamW与RNG恢复通过。三臂正式更新均0，所有新质量比较NA。登记576请求，START只读复用138个完整raw，零新采样；隔离CPU小模块1次测试更新另记。

固定任务已停止，不应自动恢复训练或把控制器DONE解释为真实预检通过。优先后续假说是晚期窄核相对长三角边的n8子网格欠分辨，先在同一冻结失败输入上建立误差受控参考；本轮未执行该后续。日志接口缺陷是未来重新审计的工程前置修正。

详见本轮 CHAIR_FINITE_SURFACE_ATTENTION_REPORT.md、DELIVERY_INDEX.md 和 RUN_AND_RESUME.md。原失败源码、自动报告、完整CSV、实际评分源码与hash均保留，最终呈现修正位于postrun_review，未改旧结果。


## 2026-10-04 Chair Finite Surface Attention 明确授权修复续跑完成

用户明确要求继续下一步、不要因实现缺失停止后，在独立 `chair_surface_attention_v1r1` 代码与结果目录修复了显存日志保留字段重复问题。CPU回归、真实只读nvidia-smi及实际三臂遥测均通过，旧v1所有文件未改。

本次实际新增76前向/20次零更新反向，与旧轮累计86/21，仍在原96/24硬上限内。三臂全部完成预定检查，无调用/遥测失败、无未决调用；原参数、AdamW与RNG恢复。H/S各任务2次成本测量，M每任务1次限制已登记；未用余量不授权追加任务。MOMENT/SURFACE真实FP32及BF16零初始化均与H输出逐位一致。

独立CPU自适应积分完成固定8例，34次积分、1472625核点、24次旧求积CPU调用，无重试。两档精度自洽通过；192点最坏均值差约0.111808，确认晚期窄核欠分辨。12288点参考在部分难例也有偏差；49152点在本8例低于0.001，但不是全任务或严格认证。

真实成本已完成：SURFACE/H完整forward在普通40%/最大40%为3.05084/3.13547，略超冻结3倍门槛；F+B四任务均低于3倍。严格置换容差H本身已失败，M/S在最大N另有padding数值差异，全部保留、未放宽阈值。实现已实际跑通，当前阻塞由数值精度、成本及数值不变性合同构成。

正式训练和新采样仍为0，没有改变默认模型。优先后续是保持数学定义、在冻结难例上验证能解析边/角窄核的有限三角积分实现；该新生产算法与训练本次未启动。修订固定任务已完成，详见 `native_t1/runs/chair_surface_attention_v1r1/CHAIR_FINITE_SURFACE_ATTENTION_R1_REPORT.md` 及 `DELIVERY_INDEX.md`。

## 2026-10-05 Chair Moment Channel Learning V2 实际训练对照完成

`native_t1/runs/chair_moment_channel_v2/` 已实际完成从同一 HYBRID total5000 起点恢复的 H_CONT/MOMENT_XY 两臂各1000更新，终点均 moment1000/total6000。20个冻结新实现文件和2464个保护依赖hash不变；旧AdamW延续、新通道参数与状态单列。未切换FM默认模型或JEdge5可选模型，未覆盖旧START。

实际16000训练forward/backward、2000 optimizer更新、120预检forward/44零更新backward、276新终点Euler50与2初始回放、138 START raw缓存、18原NA；采样NFE13900，总forward30020，模型失败/未决0。两臂8000训练样本与1000批次逐项配对，完整检查点/raw/账本独立审计PASS。孤立CPU模块仅1次测试更新另记。padding压缩修复在四真实fixture的变长padding逐层与输出均bitwise通过；原严格逐元素置换FAIL仍保留，新归一化FP32合同通过，BF16非零置换仅作披露。

质量结论 NO_CLEAR_LEARNING_SIGNAL：验证A20主MSE START/H/M分别0.0015310739/0.0015961439/0.0016433661，M比H高2.96%、仅3/8父改善，比START高7.33%。验证正确条件列H/M均9/24、三目标全对均0/8；通吃稍减与平均斜率误差稍降未形成跨噪声一致控制收益。M训练循环约43.8分钟、H约25.3分钟，采样均值4.6352秒/2.7163秒；不宣称已证明可靠内核加速。

暂留简单HYBRID/START，不扩大本通道训练。唯一后续建议是用既有raw与冻结查询定位父c1b8c2f0c4f3cc43aff7c08b060f5ed6的noise1两臂共同退化区域，检验是否局部缺面/错位而非整体宽度偏差；该后续本轮未启动。科学协议偏离0；训练前恢复状态修复、1次独立审计元数据工程错误修复、报告采样计时字段与编码修订均保留原件/diff。固定任务已停止。详见本轮 `CHAIR_MOMENT_CHANNEL_V2_REPORT.md`、`DELIVERY_INDEX.md` 和 `delivery_hashes.json`。


## 2026-10-06 保留 RAW / START，局部锚定路线退役

用户明确要求停止当前局部锚定路线并清理不用的代码、权重和缓存。`chair_bounded_attachment_v1` 的活动源码与运行目录已完整归档后移除；其 16 次 CPU 修正只改变 1 例、strict 接缝父均提高约 1.035 个百分点，未过冻结推进门槛，TRAJECTORY 未执行。继续使用原始 RAW / START，不启用局部后处理。

删除此前边界接合实验 19 份不用的权重（A 的 4 个中间恢复点、B/C/D 各 5 份），共 29,736,512,413 bytes；另删除 16,000 个派生 OT 缓存文件。A_FM attachment1000/total6000 按旧报告继续保留为等预算对照证据，不作为默认模型。START 两条硬链接路径、官方及维护权重、原始 RAW、C/噪声/查询点、四格实验报告和评分证据保持不变。

合计删除 16,655 文件、29,803,808,433 逻辑字节（约 29.80 GB）；局部锚定的全部代码与逐例结果封存在 17.66 MB ZIP，未保存已删除的大权重。578 个受保护文件前后哈希一致，删除后正式采样 CLI 帮助检查通过。模型加载、前向、训练和新生成均为 0，没有 commit/push。旧实验记录里的已删除路径以 [清理报告](native_t1/maintenance/local_anchor_retirement_20261006/CLEANUP_REPORT.md) 为准；不要自动恢复该路线。

## 2026-10-06 两三角形干净面片 control 小试

按用户要求清理本轮论文图目录 `deliverables/chair_paper_figures_v1/`，删除394文件、32,675,398逻辑字节（包含该轮ZIP/预览/面板缓存）。509个旧来源依赖hash不变，原RAW/数据/权重保留。清理账本见 `native_t1/maintenance/paper_figures_cleanup_20261006/`；旧论文图链接已失效。

随后固定START做两三角形共面座片小试：K2/N160，座面宽2.4与3.0、深2.4、y=-0.55，两配对种子20261006/20261007，共4次Euler50、200前向，零训练/反向/额外重采样。全部51态C精确保留，checkpoint、正式代码及模型state不变。输入增宽25%时，自由部分X宽分别增约1.47%与8.02%，边界接近比例均下降；图中可见生成座面位于已知面上方，不能称可靠接合。此为两噪声的局部观察，不代表总体性能。

完整新输入、RAW、51态、逐例/配对CSV、清晰输入与control/侧视PNG和独立审计保存在 `native_t1/runs/chair_two_triangle_control_v1/`，说明为 `REPORT.md`。不改默认模型/loss，没有自动训练、重放、修补或Git提交；该固定小试已完成停止。

## 2026-10-14 FULL Contour Readout 固定6000步完成

用户批准局部几何读取候选后，已实际完成 `r2/pilots/full_contour_readout_v1/` 的单臂6000步配对训练，起点为EXP67 global6000 live+匹配AdamW/RNG；复用FULL的47条完整座面条件、48,000曝光及6,000原OT，只有block6后的128维4头17D来源真实轮廓读取为方法变化（412,032参数）。原FM/R2、单次D0/GEO_C/来源定向合同不变。

同预算终点CONTOUR vs FULL：RAW明确采用均16/24，有合法接口均8/24，“采用且合法接口”均6/24，保留6、新增0、丢失0，DEV均0/12联合正例。合法边62→64，仅85既有接口增长；RAW轮廓边距3.28205→3.34608%L，POST CF4.45378→4.75748%、FF19.97185→19.72589%、I2.19320→2.34578%。本候选不晋级；FULL_S6000及其曲线继续作为充分训练的方法比较基线，维护默认模型未切换。S3000的7/24不替代预定终点。

实际6000更新、48000F/B、72新Euler50输出（3600F）、72单次D0、144新增AI_PROXY图审，全部完成；S0及原FULL结果明确缓存/继承。一次显存余量安全暂停在1114，另一次控制器状态读取竞态均按已保存状态恢复，无重复科学更新。最终审计PASS，0错误/缺失；旧194参数optimizer=12000、新8参数=6000，官方/历史资产及冻结源hash通过。归档器筛选尝试日志的工程错误已修复并留痕，无缺失OT。

完整报告、权重、逐例/逐父配对、图集、调用和恢复入口在[交付索引](experiments/r2_full_contour_readout_v1/DELIVERY_INDEX.md)。9,602,480-byte小包包含CSV、实际评分代码、code.diff、hash与48全组压缩预览；原分辨率图和大权重本地保留。固定任务完成停止，没有自动追加实验、默认切换或Git commit/push。

## 2026-10-18 非Chair试验前负结果权重清理

用户授权在约4小时跨类别试验前释放旧负结果权重。按精确白名单删除13个已完成候选的完整优化器/中间检查点，释放逻辑空间17,602,302,735字节（16.3934 GiB）：CONTOUR完整曲线及两个中间推理导出、CLEAN_EDGE_COH/SMOOTH_PRIOR完整终点及rolling、CLEAN_BULK_FM/SURGICAL_LATE4旧rolling、COUPLE3000完整终点。负结果最终推理导出仍保留；官方、START、EXP67、FULL、SEAM_C200及配对对照、LONG、B_POOL47、PLAIN_S2000等正结果与依赖不动，24项保护资产SHA256复核一致。全部RAW/POST、数据、报告、图审、评分和源码保留。只退休被删文件对应的优化器续训/中间权重重放，不改旧科学结论，不自动恢复这些权重。没有递归删除或删除外部路径；一次删除前PowerShell字段汇总类型修正已记录，修正前删除数为0。

精确清单、保护哈希及收据：[cleanup_receipt.json](experiments/r2_bench_control_transfer_v1/cleanup/cleanup_receipt.json)。此次清理没有模型调用或默认切换；非Chair实验采用独立冻结合同。

## 2026-10-18 Bench 跨类别条件适配配对完成

用户给予约4小时后，实际完成官方Bench EMA同初始化、fresh AdamW的PLAIN/SEAM两臂各3000更新。冻结64 TRAIN+8 DEV父，每父一条连贯近共面K≥2来源区域；新评测4 TRAIN+8 DEV各2噪声，共48 RAW/48单次官方Bench D0+GEO_C POST、96评分单元和96实际非盲AI_PROXY图审。7个DEV父来自官方test，1个来自官方train但未进条件训练，分层报告。此为类别专用后训练迁移，不是Chair权重零样本泛化。

局部C/F覆盖DEV由PLAIN0/16变SEAM16/16，但SEAM全部24份POST的合法seam–bulk共边与bulk可达面积均为0，已按保存索引/角色/BFS独立复核。DEV POST FF20.009→21.577%、CF8.185→5.395%、I6.522→10.864%、free小角53.727→43.334%。局部接口可迁移，不能写成整体接续或质量升级；官方test分层仍有FF/I成本。大量删面和部分POST腿部损失两臂共有，不能归咎单一SEAM机制或直接认定D0/GEO唯一根因。

保留两臂step1000/step3000完整检查点、原数据/噪声/RAW/POST/轨迹/评分及图审；终点审计后删除两份本轮旧rolling备份2.915GiB，另有本节前述旧负结果清理16.393GiB（均逻辑字节，非新增权重扣除后的净空间收益）。新Bench D0严格类别核验，Chair官方模型和维护默认不变。固定任务完成停止，无自动延长、commit或push。

[完整报告](experiments/r2_bench_control_transfer_v1/REPORT.md) · [全组表面/截面图](experiments/r2_bench_control_transfer_v1/gallery/paired/index.html) · [交付与恢复入口](experiments/r2_bench_control_transfer_v1/DELIVERY_INDEX.md)。后续不要将本轮16/16局部边包装成主体成功，不要恢复已退役rolling或混用Chair/Bench去噪器。

## 2026-10-18 Bench 多部位条件数据准备完成

按用户要求，复用原Bench TRAIN64/DEV8父与原网格版本，构造座面、腿/支撑、靠背和组合输入，零训练/模型调用。342候选经过独立C几何审计及逐条实际AI_PROXY图审，保留316条（TRAIN277/DEV39）：座面65、腿/支撑107、靠背22、座面+支撑54、座面+靠背16、其他多部件组合50、其他明确部位2。26条排除记录保留，未修复来源M/Y；不把更多条件算成更多独立父。

全部K>=2，K中位数10，保留原坐标、源面/顶点身份和原角序。独立检查316条C/Y逐字节重建、C-only输入字段隔离及72源hash通过。实际范围包括完整闭合腿、腿侧面、端架、单块座板等，必须读extent_description，不从历史候选文件名猜部位。

255条符合当前Bench入口与single-edge SEAM数据合同；61条需入口/布局扩展，其中49条闭合C没有开放边，不能宣称现SEAM入口已经支持。闭合有效腿保留，不人为切开。来源Y的重复/自交风险未清洗；源owner/alias只作训练/诊断，严禁放入C-only推理输入。后续应按父组织采样，冻结新评测面板后再决定训练，不能自动沿用旧单平面条件分布或把此次准备称为模型质量升级。

[报告](experiments/r2_bench_parts_v1/REPORT.md) · [多部位图库](experiments/r2_bench_parts_v1/gallery/index.html) · [逐条件CSV](experiments/r2_bench_parts_v1/data/conditions.csv) · [恢复入口](r2/data_tools/bench_parts_v1/run.py)。维护默认与历史试验不变，无commit/push。

## 2026-10-19 Bench 多部位完整训练与评价收口

按用户要求优先完成训练，`r2_bench_parts_convergence_v1` 已从官方 Bench EMA、新 AdamW 完成 COMMON12000，再从同一完整状态分叉 PLAIN／SEAM 各9000步，终点均 global21000。有效更新合计30000；三次明确留痕的恢复重算额外325步。固定规则结果为 `BUDGET_EXHAUSTED_NOT_CONVERGED`：PLAIN达到有限平台，SEAM tip最后1000步仍下降16.736%，不能称全部收敛或自动延长。

活动数据为64 TRAIN／8 DEV父、277／39条多部位条件；闭合C通过普通条件路径保留，不人为切边。终点每模型48请求，合计96 RAW＋96 POST、124字段及192条实际非盲AI_PROXY图审，含固定位置截面。全部过程共312 RAW／312单次官方Bench D0与GEO，原资产、逐例结果、曲线、完整恢复权重保留。

同预算POST：有合法C/F边的请求TRAIN由3/16变14/16、DEV由0/32变22/32，但SEAM free-only bulk可达面积仅TRAIN0.9715%、DEV0%，≥30%为0/14、0/26适用请求。DEV CF/FF/I由8.119/29.543/6.673%升至9.628/32.296/8.452%；局部接口收益不能包装成整体接续或质量升级。闭合C的接缝指标为NA，图审不认证精确拓扑。此为Bench类别专用后训练，非Chair零样本迁移，也非只换类别的因果对照。

旧Bench快测4份冗余完整检查点已按用户授权清理，逻辑释放5.830GiB，收据在 `maintenance/bench_old_checkpoint_cleanup_20261018/`。先前工作区条目中的旧Bench完整权重保留描述已由该清理收据更新；原评价证据仍在。当前维护默认、Chair正结果与官方权重不变，没有commit/push或启动后续实验。

[最终报告](experiments/r2_bench_parts_convergence_v1/REPORT.md) · [终点双臂表面与截面](experiments/r2_bench_parts_convergence_v1/gallery/NATIVE_S21000__SEAM_S21000_full/index.html) · [交付与恢复索引](experiments/r2_bench_parts_convergence_v1/DELIVERY_INDEX.md)。

## 2026-10-20 Bench新任务准备：旧中间权重清理，未开实验

用户要求先不写实验代码，先清理不用的权重并评审 R2_BENCH_CLEAN_MULTIPART_SEAM_V1 任务书。已按逐文件白名单删除29份 .pt：Bench COMMON1000..11000的11份推理快照；PLAIN/SEAM各13000、14000、16000..20000的14份非关键推理快照；Bench三份重复rolling；Chair多部位local1250旧rolling。释放19,302,132,919字节（17.9765 GiB），本次磁盘可用空间增量同值，24项关键资产SHA256前后一致。

保留Bench COMMON12000完整生成器/AdamW/RNG、两臂15000/21000完整与推理权重、官方生成器/D0、Chair S1000/S2000和MULTIPART最终完整/推理/占用头；所有数据、RAW/POST、噪声、查询、曲线、固定探针、评分、图审、报告和取证检查点不动。Chair被删rolling是1250中间态，不是1500终点副本。

生命周期边界：上述中间权重重采样及依赖rolling/中间导出的旧Bench整链控制器重放退役；不能声称旧run_all --execute --resume-controller仍可原样从头执行。命名完整起点和终点仍可供新授权分支使用，历史数值与图像证据有效；不自动补回已删除文件。

任务书只读核查发现：49闭合条件是TRAIN44+DEV5；按开放且有owner过滤为58父195条件，仍含69多连通条件。新模型两种布局各48输出应计96RAW/96D0，另计固定探针；Chair1500与Bench1500训练历史不等同。完整评审与建议顺序存于下列JSON，本轮没有编写实验代码、模型调用、训练、重新采样/评分、默认切换或commit/push。

[清理回执](maintenance/bench_clean_preparation_cleanup_20261020/cleanup_receipt.json) · [逐文件清单与删除前hash](maintenance/bench_clean_preparation_cleanup_20261020/prepared_manifest.json) · [任务书评审](maintenance/bench_clean_preparation_cleanup_20261020/taskbook_review.json)。

## 2026-10-20 Bench 完成实验后冗余权重清理

用户要求清理多余权重，已逐文件删除16份 `.pt`，逻辑释放18,780,888,428字节（17.4911 GiB）：adaptation_strength三臂各自step500、step1000、旧1250 rolling及S0/S500推理导出；CLEAN旧1250 rolling。三臂step0完整初始参考、各自step1500完整与推理终点、CLEAN最终生成器/优化器/occupancy、COMMON12000及历史PLAIN/SEAM15000/21000均保留；官方模型、Chair正结果和全部数据/RAW/POST/曲线/图审/评分/报告不动。

22份保留权重与69份原文档前后SHA256核验，另对两个清理目录的36,178份非权重文件核对大小和修改时间；完整结果见回执。已删中间点的直接模型重放退役，原全库存审计含已删导出，清理后不能按旧完整库存重跑；历史PASS与当前资产可用性分别记录。四条训练均COMPLETE，未回退游标、补训或自动补回导出。本轮零模型调用/训练/评分，无默认切换、commit或push。

[清理说明](maintenance/bench_weight_cleanup_20261020_2241/CLEANUP_REPORT.md) · [清单与哈希](maintenance/bench_weight_cleanup_20261020_2241/prepared_manifest.json) · [完成回执](maintenance/bench_weight_cleanup_20261020_2241/cleanup_receipt.json)。两实验目录均新增独立生命周期说明，原报告和完成收据保持只读。

## 2026-10-21 Bench USER v2 扩充数据封存，47–67父目标未达

按用户认可的 Chair 规模参照继续实际制作数据，最终保留25个TRAIN来源父／81条人类式部件条件，含首批6父18条及新增19父63条。旧79父之外检查本地官方归档新增83个低面数来源，并从341个高面数来源中筛至22个可尝试简化父；两次独立固定QEM提案及失败记录均保留，最终仅6个简化父通过几何、保真及图审。最终25父距47父还差22，不能称已完成规模对齐或独立款式覆盖。

81条含座面21、靠背10、腿／支撑22、座背7、座面支撑17、其他组合3、其他1；N128..256、K2..134。78条有来源owner，2条闭合及1条开放无owner保留普通条件路径，不强造seam。新增63条生产CPU OT打包全部PASS，首批18条复用封存收据；原DEV8父39条及保护资产哈希不变。仅做数据制作和CPU核验，模型调用/训练/GPU计算均为0，没有权重清理、默认入口切换或commit/push。

新数据独立封存于 [完成收据](experiments/r2_bench_user_data_v2_expansion_v1/DATASET_READY.json)，以该收据绑定的manifest为正式池，不加载candidate全集。[报告](experiments/r2_bench_user_data_v2_expansion_v1/REPORT.md) · [图册](experiments/r2_bench_user_data_v2_expansion_v1/index.html) · [逐条件表](experiments/r2_bench_user_data_v2_expansion_v1/accepted_conditions.csv) · [只读恢复复核](r2/data_tools/bench_user_v2_expand/resume.ps1)。更新后的边界见 `r2/docs/bench_data_pool_policy.md`；训练预算和评测尚未登记，不能把本次准入结果当成生成质量改善。

## 2026-10-22 Bench真实轨迹交叉完成：熟悉条件收益更依赖USER前缀

完成 `R2_BENCH_TRAJECTORY_CROSSOVER_V1`：复用上一轮16请求的真实t=.5状态，在COMMON12000与USER_V2_S1500之间交换完整模型后半程；新增CU/UC各16份RAW，共800次已返回生成器前向，零训练/D0/新噪声/新OT/来源Y注入，零GPU重放。CC/UU原结果只读复用；64行全指标、16页四组合图及同位置截面全部实际AI_PROXY评阅。

CURRENT_TRAIN明确采用CC/CU/UC/UU为1/8、1/8、5/8、6/8；UC靠背943dde/n0为PARTIAL/YES边界，宽松计入则6/8。熟悉支撑与组合两条C的四个改善请求均由USER前缀保留，COMMON前缀仅换USER后缀未获得同样改善。四组DEV均0/8明确采用。支持所测熟悉条件的前半程组织作用，不证明经典过拟合、晚期无作用或低t加权必然有效；交叉兼容性与DEV布局因素仍未排除。

UC的TRAIN RAW CF/FF受影响面率/I/小角均值低于UU，但非法FF面对数397.25高于356.875；DEV未新增采用。不得自动升级为默认pipeline或将RAW质量与历史POST混比。本轮没有D0/POST交叉结果。评分包装的两次目录/收据冲突与一次归档路径错误全部记录；仅一次CPU评分重算，原评分器和保护资产不变。固定任务已停止，无新训练、清理、commit或push。

[报告](experiments/r2_bench_trajectory_crossover_v1/REPORT.md) · [完成收据](experiments/r2_bench_trajectory_crossover_v1/COMPLETE.json) · [图集](experiments/r2_bench_trajectory_crossover_v1/gallery/index.html) · [全部逐输出指标](experiments/r2_bench_trajectory_crossover_v1/analysis/per_output_all_metrics.csv) · [小包](experiments/r2_bench_trajectory_crossover_v1/BENCH_TRAJECTORY_CROSSOVER_REVIEW.zip)。恢复入口 `python -B -m r2.pilots.bench_trajectory_crossover_v1.resume` 完成后仅做只读核验，不调模型。

## 2026-10-22 Bench早期FM加权完成：未扩大C采用范围

完成 `R2_BENCH_EARLY_FM_V1`：从与USER v2完全相同的COMMON12000/A step0及fresh AdamW出发，25 TRAIN父/81条件、相同12000实例流、batch8/micro1，新训练1500步。唯一科学变量为原FM在t<.5时乘1.5，否则乘.5，保留整批原坐标分母；不是修改时间采样、轨迹或前750步训练日程。原t为sigmoid正态；5994/6006实例、平均权重.9995。首批8次前向统计逐位相同，全曝光身份配对通过。

实际训练12000F/12000B、1500更新，35.68分钟；固定16请求新增800生成器F、16单次官方Bench D0、零新评价噪声、零GPU失败/重放。采样按实测资源逐个启动2进程并重叠；RAW评分与D0可独立重叠，CPU评分3进程、渲染2进程。训练设备显存采样峰值5185MiB含后台，进程显存NVIDIA-SMI不可得；不以allocator峰值冒充全进程。

USER→EARLY的明确采用TRAIN6/8→6/8、DEV0/8→0/8，RAW/FINAL逐请求标签相同；16页固定视角/截面、64单元均实际AI_PROXY评阅。TRAIN最终FF19.540→18.985%、I3.687→2.698%，但bulk平均可达19.130→18.474%；≥30%计数2→3仅bf504/n0由29.581→30.070%的跨线。DEV最终FF25.803→24.014%，同时保留free面79.875→75.625、GEO删面114.625→118.875、小角48.618→49.097%；I仅PASS6份13.452→13.664%，2d两份LOW_SUPPORT不当成功。943dde/n0与9d5608/n1有更明显表面损失；f495/n0可达0→19.945%为局部收益，仍PARTIAL。低t seam训练MSE略降，bulk未改善，高t误差升高。该权重配方不晋级，不否定早段组织的重要性，不自动延长或扫描其他权重。

保护资产和原评分源码hash通过。完整权重、16 RAW/POST、144逐输出数值行及178原数值字段均保留；6,996,128字节小包含逐例CSV、实际评分代码、diff/hash、图审与16页图，不含大权重。恢复入口 `python -B -m r2.pilots.bench_early_fm_v1.resume` 完成后仅只读核验。未清理、切默认模型、commit或push。

[报告](experiments/r2_bench_early_fm_v1/REPORT.md) · [完成收据](experiments/r2_bench_early_fm_v1/COMPLETE.json) · [图集](experiments/r2_bench_early_fm_v1/gallery/index.html) · [逐输出全指标](experiments/r2_bench_early_fm_v1/analysis/per_output_all_metrics.csv) · [小包](experiments/r2_bench_early_fm_v1/BENCH_EARLY_FM_REVIEW.zip)。

## 2026-10-22 Bench 数据扩展训练启动与存储清理

按用户要求完成有限清理：删除五个已完成分支的7份旧恢复中间权重、80个逐哈希相同的验包副本和72个可重建build文件，共10,977,147,265字节（10.2233 GiB）。95个保留文件SHA256前后不变；受影响旧实验28,381个非权重文件大小/mtime不变。官方与最终权重、来源数据、RAW/POST、评分、图审及报告保持。旧1250 rolling不是1500终点副本；对应中间点重放退役，不能按旧完整权重库存验证器补回。详见 [清理报告](maintenance/storage_cleanup_20261022/CLEANUP_REPORT.md) 与 [收据](maintenance/storage_cleanup_20261022/cleanup_receipt.json)。

新的 Bench 关系/部件条件池已通过原件封存复核：在原25父81条件上新增10条、仅新增1个独立父，合计26 TRAIN父91条件，N128..256、K2..134；旧8 DEV父39条件不变。364个训练NPZ引用共2,733,928字节（2.61 MiB），不含模型、图与共享来源资产。不得将这次增量描述成大规模扩父。数据见 experiments/r2_bench_user_relation_rebuild_v1/DATASET_READY.json。

用户已授权开始训练；独立实验 `r2_bench_relation_training_v1` 已冻结1500更新，从与USER v2/A相同的COMMON12000 live及fresh AdamW起点开始，原SEAM/FM/HYBRID0.25、batch8/micro1、lr1e-5；不搜索超参、不增加loss或训练占用头。26父曝光460或462实例，最终固定local1500/global13500；复用旧48请求、布局、噪声及完整后处理和原位置截面，主参照USER_V2_S1500、次参照A_S1500。旧面板历史TRAIN均不属于当前26父，不应写成该池训练集表现。当前状态和有限恢复入口见 [RUNBOOK](experiments/r2_bench_relation_training_v1/RUNBOOK.md)；本条仅记录启动，不预判结果或收敛。保持默认CLI，不commit/push。

## 2026-10-22 Bench关系覆盖1500步完成：混合结果，不晋级

RELATION_V1_S1500实际完成1500更新、12000训练前向/反向、48 Euler50 RAW、48单次官方D0；固定完整后处理、432评分行/177原数值字段、144页固定位置截面及432条非盲AI_PROXY图审齐全，调用、输入、源码、权重与图片hash核对PASS。训练无失败/重放；原自动workflow状态不回写，完整收口由 [COMPLETE.json](experiments/r2_bench_relation_training_v1/COMPLETE.json) 记录。

相对USER v2，最终ALL FF28.119→27.416%、CF6.373→5.980%、I仅PASS12.157→10.174%；但绝对FF面对1269→1298、删free面88.958→92.792，DEV FF29.044→29.457%，座面DEV子组FF15.18→24.26%。三个模型均无≥30% bulk可达（40适用/8闭合NA）；RELATION DEV合法seam/bulk边为0。实际图审FINAL C采用PLAUSIBLE USER/A/RELATION=15/17/13，DEV=7/10/7；保面局部收益与错层、尖片、支撑损失并存，不替换默认，不据此否定所有数据增强。

26父91条件中仅1新父/10新C；所有48评价请求在本适配池之外，历史TRAIN4父已在COMMON见过，DEV为反复使用的开发面板。完整结果、逐父配对与图审边界见 [报告](experiments/r2_bench_relation_training_v1/REPORT.md)、[图集](experiments/r2_bench_relation_training_v1/gallery/index.html)、[全指标](experiments/r2_bench_relation_training_v1/results/per_output.csv)。完整终点和推理导出保留，未继续清理新权重、commit或push。

用户随后明确授权若本轮仍不解决则检索并再派发一次实验。新候选独立登记为 `r2_bench_joint_shift_v1`：同26父91C/同起点与1500步，仅在原后半缩放实例追加C/Y共同三维平移，每轴±0.04L，Gaussian不移动；原前半identity、噪声/时间/父流不变，重新生产OT。PDR共同增强提供方法先例，非效果保证；与旧Chair CREF和Bench PERTURBED_STATE不同。此条只记录授权与准备，是否正式启动/完成看该目录实际账本；不因这条记忆自动发任务。

## 2026-10-22 Bench共同平移1500步完成：采用改善，几何代价上升

按上述追加一次实验授权，`JOINT_SHIFT_S1500` 已完成1500更新、12000训练前向/反向、48固定Euler50及48单次官方Bench D0。数据仍为26父91条人类式部件条件；与RELATION同COMMON12000初始快照及fresh AdamW、同父/条件/噪声/时间流，batch8/micro1、lr1e-5、原FM/HYBRID0.25。唯一变化为6000个原缩放实例同时平移C/Y，每轴±4%L（实测向量最大6.819%L），另6000实例零平移；Gaussian不平移，变换后重做原欧氏OT。不从RELATION终点续训，不新增loss/头/独立父。

训练2177.71秒，单worker；实测整设备峰5259MiB含后台，进程独立显存不可得，allocator reserved峰3368026112字节。零失败、零重放，25500对调用闭合；首更新前四个identity micro的40项原loss/SSE/分母与RELATION逐位相同，但不外推全训练重复性。两次后续CPU评分阶段分别2/4 worker，RAW评分与D0重叠；不把并发墙钟当隔离性能对比。全部144页三列图/固定位置截面实际评阅，432行原评分、177原数值字段与432条非盲AI_PROXY图审齐全。

同一最终处理下，主对照RELATION→SHIFT的DEV FF29.457→34.431%、CF6.122→10.031%、I仅PASS父宏9.525→12.354%（30/32通过，2条LOW_SUPPORT不当成功）；绝对FF面对892→1024、CF面对251→469，因此代价不只是分母变化。DEV >=30% bulk可达仍0/26，6个闭合C请求为NA；新增5条seam–bulk边仅涉及2请求，不能称主体接续突破。

本轮并排AI_PROXY图审的DEV C用途PLAUSIBLE为8/32→17/32：6个REPLACED_OR_IGNORED及3个UNCERTAIN转为PLAUSIBLE，原8个保留，新增分布于5父。ALL C采用16/48→26/48，同时主体REASONABLE27→24、支撑损坏15→17。459f弯座板/端支撑条件在RAW即有采用改善并保留到FINAL；但其中n1的CF20→52.17%、free65→46，不能包装为几何全面成功。该条件旧文件名含LEG，实际part_type=SEAT_SUPPORT；9d相关LEG文件实际为PART_ASSEMBLY，按manifest解读。BASE同时含D0、硬恢复与GEO，阶段性损失不能单独归因D0。

旧参考网格在本轮被重新非盲评阅，标签存在跨轮变化（旧RELATION/USER报告的ALL采用13/15，本轮同参考16/17）；保留完整差异账本，不能跨轮拼接成稳定客观成功率。当前局部采用证据值得保留，但质量代价明显，结论为混合结果、不晋级默认，不声称已证明平移等变或跨款式泛化。论文方法依据与本地已试路线见 [检索记录](experiments/r2_bench_joint_shift_v1/LITERATURE_AND_ROUTE_CHECK.md)。

完整权重、数据/随机流、RAW/POST、曲线、逐输出/逐父表及真实图审均保留：[报告](experiments/r2_bench_joint_shift_v1/REPORT.md) · [图集](experiments/r2_bench_joint_shift_v1/gallery/index.html) · [完成收据](experiments/r2_bench_joint_shift_v1/COMPLETE.json)。原自动workflow停止在数值/图集完成，其pending-review字段是历史阶段记录；独立完成收据记录实际收口，不覆盖旧状态。恢复入口 `python -B -m r2.pilots.bench_joint_shift_v1.resume` 完成后仅只读核验。此次授权的追加实验已用完，不自动延长、再开其他候选、切默认、清理新权重或commit/push。

## 2026-10-22 SHIFT→RELATION候选：匿名复核未过门，GPU前停止

用户授权启动一轮有限后续，独立登记 `R2_BENCH_SHIFT_RELATION_CROSSOVER_V1`：拟从48份SHIFT真实state[25]接续RELATION后25步，固定同一噪声、布局和完整后处理。48份输入与24+24分片CPU核对PASS；源码、原资产hash及实际diff保存。启动前门槛先于匿名图审锁定：SHIFT明确采用总数必须更高，且至少2个不同DEV父出现RELATION明确未采用→SHIFT明确采用。并非看结果临时改变采样预算。

现成32对DEV FINAL以原尺度、两视角及两位置截面重新匿名渲染，两个未继承历史的AI线程分别一次查看16页，合计64列五轴记录。左右/页序预先随机，身份键分离；这是AI_PROXY匿名观察，不是人工评价或重复独立评分。揭盲后明确采用RELATION17/32、SHIFT21/32，但严格改善只覆盖9fd5b67a2b20b5d1b19874039c8f3e05一个父，第二项未通过。原非盲8/32→17/32标签与本轮记录并列保留，不能拼接成稳定客观成功率。

因此本轮状态为 `COMPLETE_GATE_STOP_BEFORE_GPU`：新增训练、生成器前向、D0、RAW/POST及评分均0。**交叉候选没有执行，不能登记为交叉采样负结果，也不能据此否定平移增强。** 此结论仅为采用改善的跨父视觉证据未达到本轮预设继续门槛。没有改门、重标、补采样、清理、切默认或commit/push；之后若另行授权改变该门，应新建记录，不覆盖本轮。

[报告](experiments/r2_bench_shift_relation_crossover_v1/REPORT.md) · [门判与固定协议](experiments/r2_bench_shift_relation_crossover_v1/PROTOCOL.md) · [逐请求匿名配对](experiments/r2_bench_shift_relation_crossover_v1/blind_review/gate_analysis/paired_per_request.csv) · [完成收据](experiments/r2_bench_shift_relation_crossover_v1/COMPLETE.json)。恢复入口 `python -B -m r2.pilots.bench_shift_relation_crossover_v1.resume` 只读展示收口状态，不调用模型。


## 2026-10-22 Bench生产条件数据建设：源任务已产出，新增低面训练池未就绪

SHIFT交叉保持关闭。纯CPU核对source-owner确实改变可见锚点/角色/轨迹；但旧DEV16C均可忠实表示，不把它认定为失败根因。当前26父91C仅4缓存+2闭合可核对，85缺生产预测。复用560来源供给记录，实际复核16干净高面来源，15结构可解释、1HOLD；保全部边界的共面内部重三角化16父8011面不减，独立核查内部顶点为0，不证明其他低面重建不可能。

父/家族先划分后制作30条真实C/Y，23条通过语义和C拓扑源任务准入，原面全覆盖逐位往返；全部仍超N128..256，新增可训练父为0。13父TRAIN_STAGING、2父适配未见暂存、1HOLD；同父预留没有已准入训练C时另列PENDING，官方预训练未见不认证。零模型/头/D0前向、训练、新噪声，不切默认/清理/commit/push。未把旧26父重新包装开训。

[报告](experiments/r2_bench_deployment_task_data_v1/REPORT.md) · [数据状态](experiments/r2_bench_deployment_task_data_v1/DATASET_STATUS.json) · [源任务图集](experiments/r2_bench_deployment_task_data_v1/dataset/source_tasks_pending_adapter/gallery/index.html)。恢复入口 `python -B -m r2.data_tools.bench_deployment_tasks_v1.run --resume` 完成后仅核验hash。

## 2026-10-22 Bench三家族低面与生产输入小样完成：低面成功1父，新增训练0父

固定27eb79/945214/585673三个TRAIN来源、最多6条新C/Y；未动617fa8/756108适配未见暂存和750cec HOLD。严格共享共线边界阶段均无可删点；一次部件提案使27从271降到255面，固定探针最大0.09392%L、面积比0.999382、组件/开放轮廓及有限几何通过。945保持448面阻塞；585的12部件144面提案虽无非法面对，却增加5.81%面积、探针偏差1.16%L，继续HOLD，未放宽阈值或换父。

实际查看3份完整体四视角与6页C/Y，AI_PROXY非盲。生产OLD_S15000_HEAD仅1次CPU BF16前向：27座背壳R1的31条开放边中激活20条，但原Y逐条无包含端点的面，固定布局无法忠实编码原目标；未用来源标签纠正mask。闭合扶手腿框R2走空SEAM路径，255面身份/坐标/来源绕序完整往返；因R1未准入，R2仍预留pending，不转为训练。最终新增可训练父/C和有效同父迁移C均0。此不兼容只针对指定目标与布局，不证明C没有其他完成，也不是全体Bench失败根因。

静态N审计确认128–256来自当前适配合同，非官方主干天然上限；已有源任务R1在384/512上限下分别覆盖3父3家族、6父4家族，但384档均无正owner监督。N800碰CFG面数类别槽，不能直接放行；小头N/256特征和运行资源也尚未高N验证。建议先明确可忠实表示开放Null/alias的生产布局合同，容量扩展另立验收；本记录不自动启动实验。

原资产/评分器保留，生成器/D0/训练/OT/新噪声均0，无清理、默认变更或commit/push。15项CPU单元检查、3目标6条件最终数组核对通过。权威状态为根COMPLETE与tasks/final_manifest，中间pending记录不回写。恢复入口 `python -B -m r2.data_tools.bench_three_family_v1.run --resume` 仅核验hash。

[报告](experiments/r2_bench_three_family_v1/REPORT.md) · [正式完整画幅图集](experiments/r2_bench_three_family_v1/gallery/review_index.html) · [逐条件](experiments/r2_bench_three_family_v1/results/per_condition.csv) · [完成收据](experiments/r2_bench_three_family_v1/COMPLETE.json)。

## 2026-10-22 Bench Null后续核验：纠正通用数据准入，不回写三家族冻结结果

现有SEAM已经支持Null：未激活槽返回三自由角bulk，N不减。本轮3真实条件、7唯一CPU端点检查确认27 R1在来源全Null布局下255面/坐标/原绕序完整往返；原缓存20条误激活布局仍失败。已有正owner样本全关也能打包但失去硬共享边，alias可替换owner而不可同时复制原面。零模型前向/训练/OT/噪声/几何重评分。

**“来源布局下有效监督数据”与“当前冻结生产头下原目标可编码”必须分列。** 来源布局teacher forcing＋C-only头学习标签是合法两阶段监督；不能把“旧头必须先预测正确”升级为所有未来训练数据的准入条件。来源布局确实改变生成器输入，方法描述不能声称仅影响OT；推理仍不得用Y纠正预测。三家族固定旧头合同下训练增量0保留，27仅新列为将来联合适配的监督候选，活动池仍26父91C，R2保持预留。

OLD_S15000_HEAD为旧64父277C的SEAM stage3000/global15000头、2996次更新；USER v2/RELATION均只训生成器，未在当前91C上更新该头。当前91C元数据1428owner/73真Null/14alias；新27是另一个31边全Null壳，但一个父不代表泛化覆盖。单独续训生成器不能取消旧头部署时的20条硬锚。未据此重开已试过的去权重、上下文分类器或角部规则；旧DEV16C原本可表示，不能把所有Bench失败都归于此。

[后续报告](experiments/r2_bench_null_layout_followup_v1/REPORT.md) · [逐端点结果](experiments/r2_bench_null_layout_followup_v1/endpoint_results.json) · [候选而非活动池](experiments/r2_bench_null_layout_followup_v1/SUPERVISION_CANDIDATES.json)。恢复入口 `python -B -m r2.data_tools.bench_null_layout_followup_v1.run` 完成后仅核验hash；未修改生产布局/默认/旧报告或commit/push。

## 2026-10-23 Bench 新座背壳联合适配1500步完成：局部学会Null，整体不晋级

用户授权新数据1500步快测并要求评价覆盖相似训练输入。独立实验 `R2_BENCH_NEWDATA_JOINT_V1` 实际使用原26父91C＋新27eb79的R1座背壳（27父92C），R2扶手腿框同父留出。与RELATION同COMMON12000/A-step0、fresh AdamW；原FM/SEAM/OT→HYBRID0.25、batch8/micro1/lr1e-5，原833参数头同时独立适配。训练source-owner，推理C-only预测；这是新数据＋头适配整体配方，不能单因果归数据。

完成1500更新、12000生成器前向/反向、11334头前向/反向；新R1曝光444（3.7%）。固定14C×2噪声，含当前训练18、同父留出2、旧DEV8请求；两模型56输出，实际48新RAW/48单次D0、8旧参考复用。168评分行/177字段、84配对页与固定位置截面、168条真实非盲AI_PROXY图审齐全；C精确保留168/168，最终账本与独立汇总复算PASS。首次基准入口CUDA导入错误发生在前向之前，真实0模型调用、1额外加载，独立v2恢复并保留失败记录，训练零重放。

新R1原头31 Null误开20→1，但旧当前训练95 owner召回55→21（全面板118 owner召回73→39）。新R1 FINAL FF12.67→0%，却每请求删面83.5→150.5、保留87→19.5、free面积8.905→0.932，实际图审腿框从RAW到BASE大量损失，不能作为干净完整成品正结果。闭合R2 n1也只剩8张free，无法把全部损失归头。旧DEV8请求FINAL父宏FF32.24→40.81%、I4.73→9.86%；小角40.24→27.31%、支撑PRESENT6→7并存。当前训练FINAL整体REASONABLE11/18→5/18、支撑16→8、采用14→11；DEV采用4→5但主体5→4。此处面板不同于旧48请求，勿混用旧全组统计。

上段的“保留87→19.5”指最终free面数，包含新增桥面；参照原面保留实际86.5＋桥面0.5。占用头计数按唯一条件布局计边，两份噪声不重复计入。

保留终点及完整证据，不晋级、无清理/默认切换/commit/push。若后续授权，优先隔离现成生成器与新旧头增量；本记录不自动派发。数据只新增1父，不认证充分结构扩展或独立测试泛化。
[报告](experiments/r2_bench_newdata_joint_v1/REPORT.md) · [图集](experiments/r2_bench_newdata_joint_v1/gallery/index.html) · [逐输出](experiments/r2_bench_newdata_joint_v1/results/per_output.csv) · [恢复入口](experiments/r2_bench_newdata_joint_v1/RUNBOOK.md)。

## 2026-10-23 Bench完整座面课程完成：熟悉条件采用改善，DEV未晋级

用户确认6000步Native＋3000步SEAM预算。`R2_BENCH_FULL_SEAT_CURRICULUM_V1` 从保留COMMON12000 live及匹配AdamW/RNG出发，同22父22条经图审完整座面连续训练，累计18000→21000；原FM/OT→HYBRID0.25、batch8/micro1、lr1e-5、identity/x-width增强，SEAM推理固定旧833参数C-only头。训练9000更新、72000前向/反向，零失败/重放/额外探针；训练墙钟合计约3小时31分，部分与登记采样重叠，不是独占benchmark。此为Chair阶段逻辑的Bench复用，不是数据/前史完全相同的类别对照。

固定当前训练8父16请求＋既有DEV4父8请求，三模型COMMON/Native/SEAM共72新RAW、72单次官方Bench D0；同GEO_C来源定向及HINGE_OPP015→ALPHA→HOLE，216评分行、72页双视角双原位截面、216条实际非盲AI_PROXY图审齐全。TRAIN RAW用途PLAUSIBLE为2/16→14/16→15/16；FINAL FF34.58%→14.57%→10.25%、CF13.23%→9.33%→2.09%。但FINAL整体REASONABLE14/16→8/16→9/16、支撑PRESENT16/16→11/16→11/16，一部分新模型损伤在RAW→BASE已发生；不能把低FF直接视为保面质量提升。

DEV用途PLAUSIBLE4/8→1/8→2/8，FINAL FF26.87%→24.05%→27.21%、小角30.13%→58.01%→55.88%。SEAM全24输出均有合法C/F共边，主体bulk可达≥30%却仅TRAIN2/16（同一91e169父）、DEV0/8；不能将硬边槽接住C等同接入主体。COMMON/Native没有SEAM分区，其专属可达量是NA。SEAM训练I高尾359/n0为42.68%，DEV f495/n1为81.40%，矛盾代价完整保留。

实际输入、训练流、权重、源码与评分核验通过，独立只读复算全部层级数值/配对表及图审计数一致；149个源码归档条目、26项封口产物哈希见完成收据。controller READY_REVIEW是历史交接快照，随后实际图审完成由根COMPLETE记录。保留两段最终恢复权重、推理导出与全部证据；不晋级默认、不延长、不删除旧资产、不commit/push。

[报告](experiments/r2_bench_full_seat_curriculum_v1/REPORT.md) · [三阶段图集与截面](experiments/r2_bench_full_seat_curriculum_v1/gallery/index.html) · [逐输出](experiments/r2_bench_full_seat_curriculum_v1/results/per_output.csv) · [真实图审](experiments/r2_bench_full_seat_curriculum_v1/visual_review/summary_v1/actual_observations.csv) · [完成收据](experiments/r2_bench_full_seat_curriculum_v1/COMPLETE.json)。只读恢复入口见该实验RUNBOOK。

## 2026-10-23 用户指定当前 Bench 研究参考，授权完整座面扩覆盖分支

用户认可完整座面 Native→SEAM 路线，要求作为当前 Bench 优选研究参考保留。独立登记见 [当前参考](r2/docs/bench_current_reference.md) 和 [保护清单](r2/maintenance/bench_reference_20261023/retained_assets.json)：Native6000/global18000 的完整恢复与推理权重为扩覆盖起点；SEAM3000/global21000 的完整恢复、推理权重及固定 C-only 头为冻结现状参照。原报告和全部正负证据不回写，不将“本地当前研究参考”说成已证明独立父全面 SOTA，不切 maintained CLI 默认。

用户本轮授权新增结构数据实验，独立目录 `experiments/r2_bench_seat_coverage_v1/`；先构造与审阅完整座面，再冻结旧池回放、父级留出、有限预算和固定评价。准备状态不是训练已经完成；以该目录 registration、实际账本和最终收据为准。旧资产不清理、不commit/push，不自动开展修正器、顶点引用或新SEAM训练。

## 2026-10-23 Bench完整座面扩覆盖1500步完成：保留RAW局部采用，不替换SEAM3000

`R2_BENCH_SEAT_COVERAGE_V1` 从保留Native6000/global18000及匹配AdamW/RNG接出一条Native分支，原22父全回放＋a8e987/27eb79两个新增当前谱系父，固定1500更新到global19500。OLD/NEW各6000实例，两个NEW父各3000次，原22父各272或274次；无等预算OLD-only臂，不能将增量单因果归数据。df6单父留出，a8/27/df6均属官方train来源且曾在项目其他分支出现，非全项目新父；a8/df6的来源T-junction单独按Native有限面合同准入，不认证SEAM兼容。

实际1500更新、12000前向/反向，0头/探针、0失败重放；36新RAW、1800采样前向、36单次Bench D0，加48旧输出只读复用；252评分行/177字段、90页252条真实非盲AI_PROXY图审完成。执行、原资产保护及数值独立复算PASS。训练派发至退出45.24分钟，pipeline53.24分钟不含数据准备/图审，部分阶段重叠。

本轮同输出采用＋整体＋支撑交集：NEW RAW 2/4→4/4（新增两份均a8），FINAL3/4→0/4；候选四份细支撑在BASE已大幅丢失。NEW FINAL FF20.87%→1.79%、I2.73%→0，伴随原free留存57.75→30.75、删除130.25→157.25、面积20.52→3.80、小角57.28%→71.14%；629删除中627为合并后重复顶点坍缩、2为重复free面。不能以低FF判升级，也不能只据RAW→BASE指定D0或GEO单一根因。

OLD FINAL FF14.57%→15.80%、I1.87%→5.07%；DEV FF24.05%→22.26%、I12.85%→11.42%，但小角58.01%→65.32%、支撑6/8→5/8。DEV三项仅0/8→1/8（fd355/n0，保留SEAM3000已有同例），df6仍0/2。新增结构密集适配保留有限RAW采用信号，不是整体跨父升级或新增数据路线的全面否定。

旧Native/SEAM几何与原评分不变，但本轮并排重审720轴标签有113改变；历史CSV原样保留，原Native TRAIN RAW采用14/16、本轮16/16，不把评级漂移算模型收益。保留原Native6000和SEAM3000作为当前Bench研究参考；新终点与数据证据归档，不自动接SEAM、清理、切默认或commit/push。

[报告](experiments/r2_bench_seat_coverage_v1/REPORT.md) · [图集和截面](experiments/r2_bench_seat_coverage_v1/gallery/index.html) · [全字段逐输出](experiments/r2_bench_seat_coverage_v1/results/report_tables_v1/all_177_per_output.csv) · [完成收据](experiments/r2_bench_seat_coverage_v1/COMPLETE.json)。

## 2026-10-25 当前轮次归档与旧权重退役

用户本轮明确要求归档当前实验、清除不再使用的旧权重和数据、保留当前优选结果。已实际按封存白名单删除390份旧权重及1,022份与保留ZIP逐字节一致的审阅包展开副本；逻辑大小137.098GiB，E盘可用空间实测增加135.639GiB（逻辑大小包含硬链接；卷空间变化与逻辑长度分别记账）。第一阶段842项、补充清理后实际保留605项资产SHA256复核通过，当前7份清单中的4,301条非权重引用仍在，最新轮次159项原始归档证据hash不变。

保留Chair PLAIN_S2000及其完整恢复点、LONG C-only头、官方D0和HINGE/ALPHA/HOLE CPU代码；保留MULTIPART_S1500靠背正结果完整/推理/头、FULL_S6000完整/推理、START默认及全部官方模型。Bench保留COMMON12000、Native6000、SEAM3000、Coverage1500的完整/推理终点、固定小头及SHAPE500后端完整/推理；SHAPE、Coverage仅保留各自有限正证据，不宣称已全面替换当前Bench研究参考。

旧R2、N112、CONTOUR、CREF/SMALL、NFT/窗口RL剩余终点、SEAM旧候选、Bench旧失败分支、RELDEF及重复恢复点等的具体退役路径/hash见账本。此前“保留旧终点”的历史文字反映当时状态，本次较新的清理授权以独立退役告示更新当前可用性；不修改原报告、COMPLETE、历史分数或原hash。依赖已删权重的旧完整训练重放/采样不再可用，不自动恢复或重训。现存最终模型恢复状态仍保留；官方数据、当前数据依赖、RAW/POST、固定噪声、来源身份、图审、失败证据不删。

最近R2_BENCH_RAW_LOCAL_REPAIR_V1完成12请求有限CPU目标试点，6份非空EDIT未达到预先固定的7份开训门槛，4份保守数值资格不等于已获通用后端；本轮无新训练权重。完整实验原件继续保留，小审阅包3,508,889 bytes，含逐例CSV、全部12份目标/掩码、实际评分源码与图审，159科学成员已逐项校验。

[清理报告](maintenance/storage_cleanup_20261025/CLEANUP_REPORT.md) · [当前保留模型及准确路径](maintenance/storage_cleanup_20261025/RETAINED_MODELS.md) · [逐文件删除账本](maintenance/storage_cleanup_20261025/deletion_ledger.jsonl) · [最终核验回执](maintenance/storage_cleanup_20261025/FINAL_CLEANUP_RECEIPT.json) · [本轮小审阅包](deliverables/R2_BENCH_RAW_LOCAL_REPAIR_V1_REVIEW_SMALL.zip)。本轮不训练、不生成、不重新评分、不切默认、不commit/push。

## 2026-10-25 Bench固定池曝光比较完成：B保留有限收益，未替换当前研究参考

用户授权按附件推进 `R2_BENCH_EXPOSURE_ALLOCATION_V1`。从同Coverage/global19500完整live/AdamW/RNG分叉，24父24完整座面C不变；A每步2 OLD+a8+27，B每步3 OLD+均衡FOCUS，batch8/micro1，原Native FM/OT→HYBRID0.25。两臂各1500更新至global21000/native_total9000；各12000前向/反向，A OLD/FOCUS=6000/6000，B=9000/3000，零头调用、零科学失败/重放。训练经正式A前25步观察后并行，两进程跨度47.33分钟，整卡峰8953MiB，逐进程WDDM峰NA。另完成60新RAW/3000采样前向/60单次官方Bench D0，P0只读复用。

同30请求三阶段270评分行/177字段、90页双视角双截面、270实际非盲AI_PROXY观察完成；同输出整体合理∩C合理采用∩主要支撑保留：OLD RAW P0/A/B=12/12/14（16请求），FINAL=10/11/12；FOCUS RAW4/4/4、FINAL0/0/0；DEV RAW0/0/0、FINAL0/0/1（8请求），df6三个阶段0/2。B保住重点两父RAW，OLD FINAL增加4cd/n1和6b/n0且无交集丢失；唯一DEV新增459/n0仍有碎条。A的FINAL交集丢失为91e/n0。此交集不包含无碎片/无折层认证。

FINAL OLD FF P0/A/B=15.801/14.158/15.057%，I=5.069/.610/1.349%，保留free=87.50/89.94/81.88；A在这些量更有利。DEV FF=22.259/21.683/11.783%，B相对A四父均降且多保留3.875张free，但比P0少11.875张free、面积少4.092；DEV合理主体5/3/3，碎片/裂口4/6/7。不能以B低FF认定全流程升级。a8/27在BASE已大幅丢支撑，FINAL未恢复。P0同几何450五轴标签有47漂移，历史保留，本轮配对用本轮同页重评P0，不能把标签变化记模型收益。

同步CPU制作28b9/abf250两独立来源C/Y，N384/308；一次预登记保结构共线边界方法未降面，新增可训练父0，BLOCKED_N，不混入A/B。保留两臂终点/恢复状态，B仅为曝光配比候选；当前Bench Native6000/SEAM3000研究参考与CLI默认不变。不扫比例、不自动接SEAM/新loss/后端训练，不清理、commit或push。

[完整报告](experiments/r2_bench_exposure_allocation_v1/REPORT.md) · [逐输出177字段](experiments/r2_bench_exposure_allocation_v1/results/per_output.csv) · [实际图审](experiments/r2_bench_exposure_allocation_v1/visual_review/summary_v1/actual_observations.csv) · [图集](experiments/r2_bench_exposure_allocation_v1/gallery/index.html) · [完成/核验入口](experiments/r2_bench_exposure_allocation_v1/RUNBOOK.md)。

## 2026-10-26 Bench自然N配方完成：新结构RAW可学，最终流程未整体晋级

用户确认两臂各1500更新。`R2_BENCH_NATURAL_N_RECIPE_V1` 从同曝光B/global21000的live/AdamW/RNG分叉；A保持旧22父75%＋a8/27 25%，B为75%＋12.5%＋新增28b9/abf250 12.5%。实际24/26父、完整座面、N128..384；新增自然N384/K10与N308/K44，无合格低版本。独立Native包装允许128..799，参数和原算术不变，N800禁止；没有SEAM/新loss/头调用。每臂1500更新至global22500/native10500，各12000前向/反向，科学失败/回放0。HIGH占12.5%实例、21.418%累计自由坐标、18.954%实际累计FM贡献，后者不是梯度份额。两个新父各750实例曝光。实测后两臂并行各约42分钟，整卡观测峰10248MiB含用户进程，不是模型独占峰。

旧30请求＋高N8请求，P0旧30只读复用；新增84RAW/4200Euler50前向/84单次官方Bench D0，全部RAW完成后才D0。同原GEO_C来源定向及HINGE_OPP015→ALPHA→HOLE。342原177字段评分、114页两视角双截面、342实际非盲AI_PROXY记录完成。HIGH_TRAIN三轴交集（整体合理∩C合理采用∩支撑保留）RAW P0/A/B=0/0/4，2父各2noise均有B独有收益；FINAL均0/4，28b9采用/接触不明、abf支撑受损。OLD FINAL=8/10/10，DEV=0/0/0，FOCUS=0/0/0，df6=0/0/0；HIGH_HOLDOUT=2/2/2，均来自原已可采用的闭合617且仍有重层，756仍错层。617 N724超过训练最高384、且闭合厚垫改变输入类型，不能单解为父泛化。P0同几何五轴66/450标签漂移已披露，不记为模型变化。

FINAL B相对A：OLD FF10.385→9.182%，定向可达面积11.085→9.021%；DEV FF21.137→11.494%四父均降，I10.376→11.719%，冻结P0 DEV I仅1.044%，f495为主要高尾。HIGH_TRAIN FF16.026→1.584%，但I因原1%L边界排除后两父均0/2048内部支持而NA，不能称I为0。B的HIGH_TRAIN自由面积RAW/preGEO/FINAL=23.577/8.437/8.412，1072删除槽全为GEO重复角点；大部分面积缩减先于最终删面，不能单用低FF或删面数认证质量。完整矛盾代价、NA、逐槽账本均保留。

保留自然N准入、两臂终点和B的已见新结构学习证据，不替换当前Bench研究参考/维护默认，不自动延长或接SEAM。153实际源码hash、全部预算/输入/结果/图审终检PASS。小审阅包9,242,140字节，含完整逐输出CSV、真实图审、实际评分源码、全部8高N请求三阶段24页图；大权重/RAW/POST及114页全图仍在本地。没有清理、commit或push。

[报告](experiments/r2_bench_natural_n_recipe_v1/REPORT.md) · [模型索引](experiments/r2_bench_natural_n_recipe_v1/MODELS.md) · [图集](experiments/r2_bench_natural_n_recipe_v1/gallery/index.html) · [逐输出](experiments/r2_bench_natural_n_recipe_v1/results/per_output.csv) · [小审阅包](experiments/r2_bench_natural_n_recipe_v1/R2_BENCH_NATURAL_N_RECIPE_REVIEW_SMALL.zip) · [核验/恢复入口](experiments/r2_bench_natural_n_recipe_v1/RUNBOOK.md)。

## 2026-10-27 Bench HIGH扩覆盖1500步完成：局部RAW可学，宽池未晋级

用户授权按附件8be846fe执行 `R2_BENCH_HIGH_JOINT_COVERAGE_V1`。保留原26父，新增既有合格945214 N448/K14、af264 N636/K84、bcb N520/K2、77aa N776/K68，训练共30父/HIGH6。原可信资产没有新增bin15/18/19父，实际新bin22/26/31/38各一父；这不是同桶多父检验，不因N范围变大宣称617/756计数桶已覆盖。77aa闭合C开放接口NA。原M/C/Y及独立留出不改。

与旧窄NATURAL_HIGH B共用曝光B_75_25的global21000 live/AdamW/RNG起点；只新增WIDE_HIGH1500更新至global22500/native10500。原Native/FM/OT→HYBRID.25、batch8/micro1、lr1e-5不变；OLD/FOCUS流逐行一致，HIGH总1500实例从两父各750变为六父各250。HIGH平均每更新坐标权重19.852→24.307%，实际累计FM贡献18.954→24.729%，不是梯度份额；不能将差异单因归数据数量。实际12000前向/反向，0科学失败/重放/头调用；训练进程36.34分钟，WDDM进程显存NA，整卡峰8863MiB含背景及短采样重叠。

固定46请求/模型：旧B38只读复用，新B8＋W46=54新RAW/2700Euler50前向/54单次官方Bench D0，全部RAW完成后才D0。276原177字段评分、138页两视角双截面、276真实非盲AI_PROXY记录，执行/身份/NA终检PASS。新增四父RAW三轴B/W均6/8（945n1新增、bcbn1丢失），FINAL4/8→1/8；afn1和77两噪声有RAW形态组织但POST支撑损失。OLD RAW15/16→14/16、FINAL均6/16；原HIGH4/4→3/4（abf n1退化）；DEV仍0/8、df6仍0/2、HIGH留出仍2/4（原617两份，756未改善）。三轴不认证无碎片/无交。

FINAL FF：新四父34.823→21.612%，伴随CF6.623→10.047%、free面积28.312→10.189及原槽107→94.125；旧DEV11.494→20.139%（三父上升），OLD9.182→13.890%。I保留PASS口径，窄条LOW_SUPPORT仍NA。旧B同几何570个轴标签103漂移；尤其本轮主图审原HIGH POST4/4与专项/历史0/4来自轴阈值偏宽，已确认原hash一致并明确披露，不当作模型改善，旧标签均保留。

独立四RAW CPU GEO_ONLY完成，0模型/D0：可见28座条侧壁和abf端撑保留更多，专项三轴D0 FINAL0/4→GEO_FINAL4/4，但FF1.584→22.675%、CF0→4.260%、小角31.414→53.164%、非流形0→10.5；abf n1 FF41.126%坏尾保留。保形正证据与冲突代价并存，不切后端默认。

保留自然N接口及W完整/推理终点，不替换B或当前Bench研究参考，不自动延长、清理、commit或push。178个执行源码文件已归档；完整审阅包16,512,374字节，小包9,361,593字节，含全逐输出177字段、主要逐父配对、实际图审、全部新增8请求三阶段24页及四RAW后端图。小包只省略11份详细重复账本/放大图，原件完整保留。

[报告](experiments/r2_bench_high_joint_coverage_v1/REPORT.md) · [模型索引](experiments/r2_bench_high_joint_coverage_v1/MODELS.md) · [图集](experiments/r2_bench_high_joint_coverage_v1/gallery/index.html) · [逐输出](experiments/r2_bench_high_joint_coverage_v1/results/per_output.csv) · [小审阅包](experiments/r2_bench_high_joint_coverage_v1/R2_BENCH_HIGH_JOINT_COVERAGE_REVIEW_SMALL.zip) · [核验/恢复](experiments/r2_bench_high_joint_coverage_v1/RUNBOOK.md)。

## 2026-10-27 Bench宽池组权重匹配1500步：局部FF改善，整体不晋级

用户批准 `R2_BENCH_GROUP_WEIGHT_MATCH_V1` 唯一W_MATCHED臂。M从共同曝光B_75_25/global21000的live/匹配AdamW/RNG重新分叉，非B/W22500终点续训；逐条重放W30父12000实例、时间和随机身份，只在750个HIGH更新按冻结窄B计数恢复组MSE系数，750个FOCUS公式不变。全1500更新平均HIGH系数W24.307%→M19.852%（B19.852%）；实际HIGH loss贡献24.729%→20.149%，不是梯度份额，六父各250次曝光未变。Native/T1+role+Geo、FM/OT→HYBRID.25、N128..799实际最高776、batch8/micro1、lr1e-5及后端不变，无SEAM/新头/新loss。

实际1500更新到global22500/native10500、12000前向/反向、0科学失败/重放/探针；训练37.23分钟，整卡观测峰7106MiB含背景、进程显存WDDM不可得。46新RAW/2300Euler前向、46单次官方Bench D0，全部RAW先保存；B/W276原评分只读复用，共414行177字段。138页双视角双截面、276格V2实际非盲AI_PROXY完成；执行、系数/随机流、评分NA和原资产hash审计PASS。

M相对W的FINAL：新增四父FF21.612→18.163%（3父两噪声均降），但RAW FF约+0.905pp；OLD FF13.890→12.879%、CF6.723→5.928%，I却1.553→5.664%；DEV FF20.139→20.588%且相对窄B仍高9.094pp，HIGH留出I4.822→8.508%。OLD I增量几乎全为4cd/n1：FINAL3.369→69.629%、RAW3.271→95.508%，不归因于最终导出才出现。

本轮V2三轴交集W/M：OLD RAW14/13、FINAL10/11；FOCUS RAW4/4、FINAL0/0；DEV0/8、df6 0/2不变；原HIGH RAW3/3、FINAL2/2；HIGH留出2/4不变；新增四父RAW6/8、FINAL1/8不变。唯一联合身份转移4cd/n1为RAW退回不确定、POST端撑改善，但I坏尾并存；交集不认证无交/无碎片。新增父原free均553，W/M删458.875/458.750，FINAL面积10.189/10.788，后处理支撑损失没有解除。I原HIGH两父LOW_SUPPORT仍NA，新四父只2/4有PASS支持域；Native SEAM不适用。

旧W同几何重审出现V2计数差异（如OLD FINAL历史6/16、本轮10/16），原标签不覆盖，不跨版本计算模型收益。保留M推理及完整恢复、全部逐例/图审和181实际源码归档；降低“宽池仅因HIGH系数增加而退化”的充分解释优先级，不自动扫系数、延长、替换B或当前Bench研究参考/CLI默认。未清理、commit或push。

[报告](experiments/r2_bench_group_weight_match_v1/REPORT.md) · [模型身份](experiments/r2_bench_group_weight_match_v1/MODELS.md) · [图集与截面](experiments/r2_bench_group_weight_match_v1/gallery/index.html) · [完整逐输出](experiments/r2_bench_group_weight_match_v1/results/per_output.csv) · [小审阅包](experiments/r2_bench_group_weight_match_v1/R2_BENCH_GROUP_WEIGHT_MATCH_REVIEW.zip) · [核验与恢复入口](experiments/r2_bench_group_weight_match_v1/RUNBOOK.md)。

## 2026-10-27 Bench四方向固定预算完成：保形与冲突仍有取舍

用户批准`R2_BENCH_FOUR_ROUTES_V1`四项：固定M的CPU RAW/D0混合；M/global22500再训3000步至25500（六HIGH父累计各750曝光）；冻结窄B每C四固定噪声的几何选优；窄B/global22500接原SEAM1500步至24000、旧C-only头冻结、不加共享噪声。实际4500更新/36000训练前向与反向、138新RAW/6900生成前向、105单次官方D0、21头前向、440新原评分，固定23父46请求面板（选优92候选），无科学失败/重放、无扩预算。四个AI_PROXY身份实际审391页/920格，原固定双视角及位置截面保留；额外46页BASE展示不算已审。

EXPOSURE相对M的FINAL父等权CF7.602→7.350%、FF14.260→9.044%、共同PASS19父I6.755→4.828%，但DEV CF5.694→8.013%、I11.237→12.744%。本分支RAW三轴29→33/46（新增5、丢1，含两个DEV父各一噪声），FINAL17→15（无新增、丢2）；保留生成侧候选，不升级完整流程。

HYBRID相对同M后端，实际平均93.77%free面回RAW、29/46请求全free回RAW。FINAL图审三轴15→30/46、主支撑29→46，原RAW本已30；合法C/F接口14→23，原RAW参考删面面积32.519→4.968%。同时FF14.260→33.075%、I6.755→9.667%、非流形边均0.978→16.326；CF比例虽7.602→6.642%，非法CF面对却438→1271。保形正证据与严重几何代价并存，不当作精细局部混合或默认后端。原CPU运行墙钟3.70小时，含一次保内存暂停及最多2活跃worker分流5个未开始请求，原队列仅核哈希复用；暂停约3949.59秒，无科学重算。不得重跑一次性PID控制器。

BESTOF4的FINAL三轴固定n0/n1/选择=11/12/10（23C），I2.832/5.661/9.244%，本选择器不晋级。SEAM开放C接口42/42，但seam→bulk可达≥30%仅5/42、DEV0/8；FINAL FF13.056→18.994%、I4.247→7.588%、三轴21→18/46，不能用C侧固定底边代替主体接续。没有运行EXPOSURE+HYBRID组合，不把两个方向优点相加。

复用B、M各138份资产SHA和177字段完全一致；重复非盲图审有分歧（B FINAL21/23，M FINAL17/15），完整保留原标签及身份对照，不跨分支拼视觉排名。新终点和匹配AdamW/RNG完整保留；当前研究参考和CLI默认不变，未清理、commit/push或自动延长。

[总报告](experiments/r2_bench_four_routes_v1/REPORT.md) · [四项实际方法](experiments/r2_bench_four_routes_v1/METHODS.md) · [新权重与恢复状态](experiments/r2_bench_four_routes_v1/WEIGHTS.md) · [最终调用账本](experiments/r2_bench_four_routes_v1/audits/four_routes_execution_ledger_final.json) · [复核与恢复](experiments/r2_bench_four_routes_v1/RUNBOOK.md)。

## 2026-10-28 Bench局部RAW/D0混合双路线：辅助保支撑成立，自动覆盖未兑现

用户同意两份附件思路都试，执行`R2_BENCH_REGIONAL_MIX_DUAL_V1`：固定M的a8/27/459各n0/n1，六请求N189/255/198；提前固定左右下支撑P1/P2，完整评估10/01/11，与质心到实际存活BASE表面>2R、种子共边分量≥2面且≥2%RAW面积的AUTO_COVER并行比较。整面RAW/D0二选一，无扩张/平滑/GEO豁免，原GEO_C→HINGE_OPP015→ALPHA→HOLE不变；本轮不处理EXPOSURE或全46面板。

30逻辑格中实际23新CPU后端、46原完整评分；六整RAW精确复用＋一空自动复用D0。共72行177字段、36逐父、120数值配对；24页双视角/原位置截面实审、42唯一五轴AI_PROXY记录。最多2CPU worker，调度283.87秒；零模型/D0/训练/新噪声/失败/重放。原输入与源码hash、C字节/来源角序和原177字段一致核验通过。

双端ASSIST11平均23.81%free面回RAW，选区原RAW参考面积保留D0 26.31%→99.08%；主要支撑0→6/6、三轴交集0→4/6（a8/27各两噪声），整RAW为5/6。FINAL父等权D0/ASSIST11/RAW_ALL FF=7.777/13.677/27.901%，CF=.505/1.727/1.791%，共同PASS全部三父I=.016/.130/2.100%。代价：原77合法FF边丢23、新非法FF37/CF7对；原D0存活槽18丢失，虽新保留217槽，不能称无损。全体实际free面积仅原RAW31.87%，板条侧壁/上框仍损坏；459 FF升到35.244%，n1灰座替代蓝C未变。三轴不是无交/无裂认证。

AUTO平均只回RAW4.59%，支撑和三轴均0/6，FF9.888%高于D0；区域实际面积仅原RAW14.37%。本版自动规则不晋级；辅助组合保留有限可行性，不能当作自动成绩、严格上界或高N泛化。完整原表、逐槽/连接账本、图审和小源码归档。默认模型/后端不变，不自动扩样、训选择器、清理、commit或push。

[报告](experiments/r2_bench_regional_mix_dual_v1/REPORT.md) · [24页图册](experiments/r2_bench_regional_mix_dual_v1/gallery/index.html) · [全177字段](experiments/r2_bench_regional_mix_dual_v1/analysis/per_output_full177.csv) · [逐父](experiments/r2_bench_regional_mix_dual_v1/analysis/per_parent.csv) · [恢复入口](experiments/r2_bench_regional_mix_dual_v1/RUNBOOK.md)。

## 2026-10-28 Bench局部混合缓存归因：漏选与RAW内部缺陷分开

用户要求继续查局部混合；仅复用上述三父六请求，零模型/训练/生成/后端/评分/新求交/新图审。265张辅助区域面有226张未成为距离种子，39张种子中30张又被分量过滤，最终9张仅覆盖该参考区域RAW面积3.034%；a8的81张区域面零种子，单改2%面积门不能修复。93张已有七点缓存呈质心近而其他点远，其中78张原槽在D0 BASE已删；该离散证据不认证完整连续表面空洞。

ASSIST11 FINAL新增FF37对中RAW/RAW34、RAW/D0仅2、D0/D0 1；27对在整RAW同链FINAL已有，新增CF7对全部已有。459占新增FF34对，27零新增冲突。故不能将代价统称混合接缝损伤。23条丢失合法FF边按来源14/5/4；原CF边零丢失。旧缓存证明未选中的D0角点也会因GEO重新选代表点而改变最终坐标/保留状态。仅修改局部提议不保证局部影响。

后续优先假说为固定选区的局部保形去噪，需同时避免带回RAW内部缺陷、丢支撑及改变远处D0；本次只有设计未执行。自动选区另行验证，不复活全局面积缩减加一环扩张，也不把R27 BLEND改名重跑。三父范围、不变旧AI_PROXY与评分口径，默认不变。

[归因说明](experiments/r2_bench_regional_mix_followup_v1/REPORT.txt) · [机器结论](experiments/r2_bench_regional_mix_followup_v1/FINDINGS.json) · [逐面选择](experiments/r2_bench_regional_mix_followup_v1/selector/per_face.csv) · [逐事件账本](experiments/r2_bench_regional_mix_followup_v1/boundary/per_event.csv)。

## 2026-10-28 Bench局部保形融合实测：保住支撑，但本候选不晋级

用户批准继续固定选区方向，完成 `R2_BENCH_LOCAL_SHAPE_MIX_V1`：同M的a8/27/459各n0/n1、原P1/P2及查询。两臂均采用独立RAW/D0自由顶点身份域的索引组装，仅原C身份跨域共享，外部D0 BASE存活面硬锁；LOCKED_RAW直接恢复，LOCAL_SHAPE一次稀疏边向量正则拟合到缓存D0_PRE，λ=1、最大2R、面积≥0.8、法向余弦≥0.5、区跨度≥0.9，固定回退规则。无GEO重跑；后接原HINGE_OPP015→ALPHA→HOLE。它与旧ASSIST11组装不同，主因果比较只在两新臂之间。

实际12新处理、24新增原完整评分＋36复用评分，60行177字段、12页双视图/原位置截面，54唯一五轴AI_PROXY实审。零训练/模型/D0/新噪声，13项CPU检查通过，2单线程CPU worker调度97.85秒、渲染39.49秒。科学失败/重放0；汇总RAW_ALL/HYBRID生产者标签问题用独立v2兼容，原结果不动。完整脚本、原评分源码、逐例CSV和可恢复入口保留。

FINAL父等权LOCKED→LOCAL：FF14.7541→15.1606%，CF1.7372%不变，I0.130208→0.113932%，小角52.8506→58.9636%，q>10 36.1434→37.1983%。416/1142原free槽保留完全相同、无拟合新增删面；5合法CF边和305合法FF边全保留，459n0后端新形成1合法FF边但仍无C接续；459n1消1增3非法FF面对，FF受影响面25→27/82，不是分母假象。可达面积21.8882→22.0260%不能作新接口证据，主要涉及实际面积缩小；FINAL实际总面积−2.2805%。

6/6拟合接受，目标下降但实际最大位移仅0.0814–0.2099%L。两新臂与旧ASSIST11视觉支撑均6/6，LOCAL对LOCKED无明确宏观改善；459n1灰面替代蓝C仍在。本轮同资产ASSIST11三轴5/6对旧4/6为AI_PROXY标签漂移，不能记作新几何收益。BASE外部156槽/臂逐位不变；FINAL原后端各移动16槽、无丢失，相对原同阶段D0另有2槽不同，不能宣称全链外部硬锁。C来源角序坐标与索引关联保持。

不晋级LOCAL_SHAPE，不自动扫λ/放宽保护/训选择器。旧辅助局部恢复的有界正证据保留；本轮削弱“只需向D0保形小幅融合即可去掉RAW内部冲突”的假说，不否定所有局部混合。默认模型/后端不变，未清理、commit或push。
[完整报告](experiments/r2_bench_local_shape_mix_v1/REPORT.txt) · [已审阅图集入口](experiments/r2_bench_local_shape_mix_v1/REVIEW.html) · [逐输出177字段](experiments/r2_bench_local_shape_mix_v1/analysis/per_output_full177.csv) · [恢复入口](experiments/r2_bench_local_shape_mix_v1/RUNBOOK.txt)。

## 2026-10-28 用户授权清理八份旧滚动权重

实际删除exposure_allocation A/B、natural_n CONTROL_OLD/NATURAL_HIGH、high_joint WIDE_HIGH、group_weight W_MATCHED及four_routes EXPOSURE/SEAM的8份 `checkpoint_latest.pt`，共12,520,489,368字节（11.661GiB）。均是正式终点前250步的中间恢复点，不是字节重复。所有16份正式终点（完整AdamW/RNG及推理导出）前后SHA256核验通过；其他426份权重大小/时间与328项原证据hash检查通过。P0、窄B、M、EXPOSURE及旧Chair/Bench/官方保留资产不变。未删数据/RAW/POST/评分/图片，未训练、切默认或commit/push。

各训练目录新增中间权重退休说明，原latest JSON/报告不改；已删除中间点不再支持历史重放，不自动补训或伪装路径。后续获授权续训须显式选保留终点。
[清理报告](maintenance/weight_cleanup_20261028/REPORT.txt) · [实际收据](maintenance/weight_cleanup_20261028/cleanup_receipt.json) · [保留终点](maintenance/weight_cleanup_20261028/retained_endpoints.csv) · [删除账本](maintenance/weight_cleanup_20261028/deletion_ledger.jsonl)。

## 2026-10-28 冻结EXPOSURE：C三角划分敏感性实测，未形成稳定质量升级

用户同意沿云端机制排除后的方向实测。`R2_BENCH_C_TESSELLATION_V1`冻结EXPOSURE global25500/local3000，459/f495/bcb三父各原n0/n1；同一C表面中点1→4细分，free面数与原噪声不变。K/N从3/198→12/207、5/160→20/175、2/520→8/526；仅459跨count bin9→10，因此另做原/细C×bin9/10的完整2×2诊断。真实N始终等于valid数，诊断钩子只改实例查表，不修改旧源码/参数或生产入口。

实际10新RAW＋1原输出逐位复现校验，550生成前向；6原RAW/评分复用，16唯一比较格。零训练/反向/D0/GEO/POST/新噪声，无科学失败或重算；所有51状态C精确保留，权重与保护原资产hash通过。10次原完整177字段评分，8次共同物理C额外CF/I诊断，12条轨迹配对/612逐步记录，6页双视图和原位置截面、16格实际非盲AI_PROXY。两采样进程先后完成、GPU并发峰1；CPU评分双worker与独立渲染重叠，资源账本保留。

同bin的f495终点free角点位移约11.06/11.05%L；bcb为2.29/26.31%L。bcb/n1细分后C与异向主体关系变差，n0组织仍保留；f495有更清楚采用的局部视觉信号，但CF/FF增加。459只改count也产生8.86/11.00%L响应，自然细分的小角下降不能全归K。它证明真实生成对划分及count敏感，不证明softmax数量稀释是唯一根因，也不要求开放生成维持同一款式。

全部为RAW，不能与旧FINAL FF9.04%拼接比较。原I依赖查询所属三角面截交线，故主表另固定原C和原2048物理查询；新C实际177字段完整保留。f495共同C I均值40.67→14.43%，掩盖n0从1.86→15.28%的退化；不只挑有利均值。原通用score.stage写入臂名的问题只在独立metrics_v2中声明实际RAW并保留source字段，旧原件/v1不动。

本轮不晋级细分，不替换EXPOSURE/后端/默认；后续仅保留“同表面合法划分训练能否改善采用稳定性”的待检验假说，未自动启动，不能直接复活R16/R28或计数偏置loss。代码、实际评分源码、逐例CSV、图审、哈希和恢复入口归档；未清理、commit或push。

[报告](experiments/r2_bench_c_tessellation_v1/REPORT.md) · [图集与截面](experiments/r2_bench_c_tessellation_v1/gallery/index.html) · [逐输出177字段](experiments/r2_bench_c_tessellation_v1/analysis/metrics_v2/metrics_per_output_full177.csv) · [恢复入口](experiments/r2_bench_c_tessellation_v1/RUNBOOK.md)。

## 2026-10-28 同物理C内部翻边＋两臂200步：局部稳定信号，不晋级

用户明确批准 `R2_BENCH_C_EDGEFLIP_V1` 继续普通续训/内部重三角化增强各200步。先固定459(K3/N198)、bcb(K2/N520)原顶点、K/N/count、有限表面和有向外边界，合法内部对角线翻转；f495非共面等几何门禁BLOCKED，仅保留原输入回归。四新冻结RAW没有上一轮中点细分的明显bcb崩坏，但无稳定质量收益；实际评分与共同原C诊断分开保存。

同保留EXPOSURE global25500 live/AdamW/RNG分叉PLAIN200、FLIP200，原FM/OT→HYBRID0.25、batch8/micro1、lr1e-5、确定性BF16，30 TRAIN条件中28可翻/2原样保留；按完整CLEAN/X_WIDTH配对交替，共728翻转、52阻塞回退。1600逐实例流、60次实际CPU OT、free目标/随机身份/组系数核对PASS。两臂各200更新/1600F/1600B、终点global25700，无科学失败重算；不训练新头或新loss。

三模型各6原输入＋4翻边输入，旧六RAW/FINAL复用；共24新RAW/1200Euler前向、24单次官方D0，同GEO_C/来源定向→HINGE_OPP015→ALPHA→HOLE。72新完整评分、24共同原C额外CF/I、20页双视图原位置截面及60格实际非盲AI_PROXY完成。原输入6例FINAL：EXPOSURE/PLAIN/FLIP的FF12.414/18.808/23.099%，CF8.268/4.307/4.598%，I14.950/12.866/13.590%，小角59.076/66.699/63.600%；合法CF输出2/2/1。FLIP单面索引孤岛更少但bcb/n1变L形错置主体、合法边丢失；不能据孤岛计数晋级。459/n0翻边后FLIP比PLAIN保住宽端架/靠背是局部稳定信号；459/n1两续训共同丢合理采用且FINAL面积降至约2.84/2.88（原19.93）。原面板三轴RAW交集3/2/1、FINAL1/0/0，均本轮AI_PROXY而非独立人类认证。

不晋级两新候选，不自动延长或转下一机制；保留EXPOSURE证据地位。训练起点/终点、原数据、RAW/FINAL、完整CSV和实际代码均保留，旧资产无删除，默认不变，无commit/push。[报告](experiments/r2_bench_c_edgeflip_v1/REPORT.md) · [图集与截面](experiments/r2_bench_c_edgeflip_v1/candidate_evaluation/gallery/index.html) · [逐输出177字段](experiments/r2_bench_c_edgeflip_v1/candidate_evaluation/analysis/native_full177_per_output.csv) · [恢复入口](experiments/r2_bench_c_edgeflip_v1/RUNBOOK.md)。

## 2026-10-28 当前研究配方与全部路线整理入口

用户要求重新整理实验、明确当前有希望的训练配方。本次仅汇总现有证据，未训练/调用模型/重评分/新图审/清理/切默认/commit或push。新增[研究综述](maintenance/research_synthesis_20261028/RESEARCH_SYNTHESIS.md)、[Chair谱系](maintenance/research_synthesis_20261028/chair_evidence.md)、[Bench谱系](maintenance/research_synthesis_20261028/bench_frontend_evidence.md)、历史机制表及97个现行实验目录的102份顶层报告索引；文件索引含草稿与阶段报告，不是102个独立模型实验，也不与旧87来源相加。

Chair保留PLAIN_S2000＋LONG头及已归档CPU后端的有限正结果；PLAIN仍是SEAM，生成器累计global15000，LONG与B_POOL47是并行分叉，不串行继承；S2000是开发比较后保留的轨迹点，不倒改原PLAIN3000主终点或Wave1门禁。BACK MULTIPART为S1000的独立分支。Bench最新前端优先候选为EXPOSURE(global25500＝COMMON12000＋Native13500)，仍未成为完整流程全面升级；保留窄池NATURAL_HIGH B作参照及Native6000/SEAM3000原研究参考。W/M均从曝光B_75_25/global21000分叉，EXPOSURE才从M续3000。不能把尚未实测的EXPOSURE＋局部恢复拼成新SOTA。原文件与分数保持，只新增证据索引；后续选择权重先看该谱系，默认START不变。

## 2026-10-29 冻结 Bench motif 与 loop 小试收口

官方完整先验的 motif guidance 未形成当前条件任务升级：原确认38请求 G0/G3 的 RAW J均5、FINAL均4，G1为7/6，EXPOSURE为25/15。随后固定旧开发4父8请求，仅新增 L0/L3 共16份，零训练；零基4–7层各重复两次整层半残差，原Euler50、lambda0.3和0.005L限幅不变。r1真实输出与输入VJP逐位一致；完成800采样向量场、360输入VJP及16单次D0，另门禁2+2。L0/L3 RAW与FINAL均0/8联合正例；同面板G1为1/8，EXPOSURE为8/8 RAW及4/8 FINAL。L3相对G3 FINAL FF−1.835pp、小角−7.289pp，但CF+1.297pp、I+8.162pp。360/360引导步仍限幅，shadow误差约40.06%L，无改善。

本版冻结loop＋motif到此关闭，不自动训练循环修正器、蒸馏G3或扫描层段/强度。此结论只针对该固定开发小试，不否定专门训练的循环模型。报告、32个新增实际AI_PROXY图审、逐例177字段和完整对照截面保留；默认及EXPOSURE研究地位不变。

[实验报告](experiments/r2_bench_loop_motif_mini_v1/REPORT.md) · [收口与清理](maintenance/loop_motif_closeout_20261029/CLOSEOUT.md) · [图集](experiments/r2_bench_loop_motif_mini_v1/gallery/index.html)。本次用户授权清理只处理已替代的旧滚动权重和可重建字节码，封存评分JSON采用NTFS无损压缩，原路径/内容/历史hash不变；准确数量以清理回执为准。正式终点、官方资产、原数据、RAW/POST与所有证据继续保留。

## 2026-10-29 Bench固定C多LOD：数据阶段完成，训练收益待检验

用户同意先参考官方离线减面增强，再比较生成，最后研究关系预测头。已从EXPOSURE当前30父/30物理C构造9份合格LOW几何版本，总39版本；没有新增独立父或物理C。保持原坐标变换、C字节/源顶点身份/角序与原CF共边邻面；固定面积/分量/有限表面探针/非法面对准入。9份实际三视图AI_PROXY通过，最大双向探针偏差0.149034%L、总面积比例0.997632—1.000127，无新增检测到的非法CF/FF。9父受N128下限阻塞，1父减面不足，11候选质量拒绝，不通过者保留原版，原资产不改。距离是每面13探针的采样界，不是严格Hausdorff。

v1额外冻结所有锐边端点造成19/21父无候选；独立engineering_v2只为20个减面不足父修正此工程限制，并允许自由端并入原坐标不动的固定端，未改独立质量门槛。累计41条记录路径，v1全部保留；不对质量失败继续调参。CPU双worker/各1线程；模型前向、生成、更新均0，新增产物约9.7MB。

CPU版本选择器保留原OLD/FOCUS/HIGH父曝光，先选父再按整对CLEAN/X_WIDTH选NATURAL/LOW。真实历史3000步24000行的身份/种子核验PASS，LOW占13.3417%（FOCUS无LOW）；此为日程映射检查，不是新增训练。原trainer仍未接入，需版本化case缓存并按实际N重算mask/OT/噪声/FM系数，禁止直接平铺39行或复用旧系数。所有315项新产物及原30父/39case终检PASS。生成器比较预算待冻结，尚未接预测头；默认/EXPOSURE地位不变，无清理、commit、push。

[数据报告](experiments/r2_bench_multilod_data_v1/REPORT.md) · [逐父CSV](experiments/r2_bench_multilod_data_v1/data/per_parent.csv) · [数据清单](experiments/r2_bench_multilod_data_v1/data/manifest.json) · [实际图审](experiments/r2_bench_multilod_data_v1/REVIEW.html) · [恢复入口](experiments/r2_bench_multilod_data_v1/RUNBOOK.md)。

## 2026-10-29 Bench固定C多LOD：两臂1500步完成，未形成全面升级

用户接受固定PLAIN1500/MULTILOD1500配对预算后，已从同一EXPOSURE global25500 live生成器/AdamW/RNG各续1500步，终点global27000/Native15000。原Native/T1+role+Geo、free-only FM、OT后HYBRID0.25、batch8/micro1、精度/优化器不变，无新头/新loss。30父的9份合格LOW按同父整对CLEAN/X_WIDTH交替；实际LOW1596/12000=13.30%，未增加独立父或C。按实际N重算mask/噪声/OT及free分母，同时固定原OLD/FOCUS/HIGH组MSE系数。每臂1500更新/12000前向/12000反向，无失败、重算或额外probe；测量正式更新峰值后两worker并行，每臂约44.1分钟，时间重叠。

两新终点各46请求，全部92 RAW齐全后才执行92单次官方Bench D0，同GEO_C/来源定向→HINGE_OPP015→ALPHA→HOLE。旧EXPOSURE138行评分只读复用；新增276行，合计414行177字段。138页双视角与原位置截面、414单元实际非盲AI_PROXY完成，保留UNCERTAIN及旧EXPOSURE标签差异，不把评阅差异解释成模型变化。

FINAL父等权EXPOSURE/PLAIN/MULTILOD：CF7.350/7.343/6.438%，FF9.044/8.913/9.273%，I4.828/2.506/6.344%（三模型三阶段共同PASS19父38请求），小角44.706/47.652/45.317%，合法CF输出19/17/21。MULTILOD相对PLAIN新增4份接口、0份丢失，均熟悉父；DEV及两类留出仍无合法CF共边。实际图审三轴交集RAW33/35/34、FINAL15/17/16；MULTILOD相对PLAIN为2d/n0 NO→YES、fd355/n0 YES→NO、617/n0 YES→UNCERTAIN。增强没有超过普通续训的整体视觉结果。

DEV FINAL MULTILOD相对PLAIN CF3.996→8.721%、FF8.960→12.509%、I2.942→16.821%。f495/n1和2d/n0约占全组I净增量73.35%，保留尾部不剔除；PLAIN f495/n1低I同时伴随106/155原free槽删除和free总面积比13.728%，不能当全面质量胜出。free面积/各自RAW面积含变形、删面与桥面，不是原表面覆盖保留率。三模型RAW支撑PRESENT均46/46，FINAL仅22/19/21，后端损伤仍在。

保留9份LOW、两臂完整终点/推理导出与全部证据；不晋级任一新臂为统一默认，不据此否定所有增强。预测/关系头本轮未启动；固定范围已完成停止。2719项文件身份终检PASS、171份实际小型源码归档；无清理、默认切换、commit或push。

[完整报告](experiments/r2_bench_multilod_training_v1/REPORT.md) · [图集及截面](experiments/r2_bench_multilod_training_v1/gallery/index.html) · [逐输出177字段](experiments/r2_bench_multilod_training_v1/results/per_output.csv) · [实际图审](experiments/r2_bench_multilod_training_v1/visual_review/summary_v1/actual_observations.csv) · [恢复入口](experiments/r2_bench_multilod_training_v1/RECOVERY.md) · [完成及权重身份](experiments/r2_bench_multilod_training_v1/completion.json)。


## 2026-10-30 Bench三机制快筛：未形成整体升级

`R2_BENCH_THREE_MECHANISMS_V1`已完成用户授权的共享双面PATCH200、训练式LOOP200和局部RL16；同EXPOSURE/global25500起点、30 NATURAL父，普通续训PLAIN200/267对照。监督667更新；RL16机会、64策略轨迹、14更新/2跳过。6父各原2噪声，72新RAW、72单次官方D0、相同GEO_C/来源定向→HINGE_OPP015→ALPHA→HOLE；PATCH另有明示的D0后共享恢复。252行177指标、36对照页和固定截面、252实际非盲AI_PROXY记录齐全。

FINAL父宏均值：PLAIN200/PATCH200的FF9.785/21.009%、I(C)9.078/6.779%，方向C可达12.032/4.281%；PATCH平均30条RAW强制内部边，到FINAL仅11.917条合法，不能把结构保证算成学到的接续。LOOP相对近似等block PLAIN267，FF10.662→13.409%、CF5.078→2.992%、I7.373→2.360%，free面积12.560→9.213；新增a8两噪声7条合法C/F边，DEV/df6仍无合法C/F边。f495/n0后处理清除了替代灰座，RAW仍替代；不将删面后的低I冒称生成器RAW采用提升。RL16相对同随机过程RL_ZERO，FF10.204→10.632%、CF7.424→7.708%、I9.416→9.623%，合法C/F接口身份与条数不变。

三臂均不替换EXPOSURE或正式默认；保留LOOP的有限连接/后处理用途信号，不自动长训。PATCH不等于完整学习关系头，LOOP不等于旧冻结半残差或完整LiFT，局部RL不同于旧R29/R30但本配置无稳定收益。短预算、小开发面板不否定整个机制家族。CF/FF按受影响free面，I是C查询域；RAW soup单面分量不直接等于视觉碎片。原标签差异、支持域/NA、质量成本和工程失败均保留。

新增实际权重/AdamW/RNG、逐父配对、实际评分代码、code.diff、source_hashes及只读总核验入口齐全。一次PLAIN工程错误多1成功前向、0反向/更新，显式计账；评测资源结构和RL文件/状态hash修复均在模型调用前完成。无OOM、重抽、默认切换、清理、commit或push。固定范围已完成停止。

[完整报告](experiments/r2_bench_three_mechanisms_v1/REPORT.md) · [全图集与截面](experiments/r2_bench_three_mechanisms_v1/gallery/index.html) · [逐输出177字段](experiments/r2_bench_three_mechanisms_v1/results/per_output.csv) · [实际图审与独立不确定性修订](experiments/r2_bench_three_mechanisms_v1/visual_review/summary_v2/actual_observations.csv) · [恢复与核验](experiments/r2_bench_three_mechanisms_v1/RUNBOOK.md)。
