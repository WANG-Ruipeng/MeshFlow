# Chair 保留主线与实际训练配方核对（2026-10-28）

本次只读核对报告、注册、实际源码及训练曝光日志；对大权重只作文件 stat，没有加载模型、重算权重 hash、训练、采样或重评分。旧报告中的“权重全部保留”须服从 2026-10-25 后来的清理记录；当前可用资产以 [RETAINED_MODELS](../storage_cleanup_20261025/RETAINED_MODELS.md) 和实际 stat 为准。

目前最有理由保留的 Chair 主研究组合是 **PLAIN_S2000 生成器 + LONG C-only 占用头 + 单次官方 Chair D0 + GEO_C/来源绕序恢复 + HINGE_OPP015**。它的优势集中在熟悉的完整座面条件，尚不能称为通用 Chair SOTA。**MULTIPART_S1500 是独立的 BACK 正结果分支**；FULL_S6000 保留为充分训练的方法比较参考；维护 CLI 默认仍为 START。ALPHA/HOLE 有归档的局部正证据，可在对应保存身份/布局合同下使用；后来的防坍缩 UNIFIED_RESCUE 护栏不应自动加入保留主链。

## 权重血缘和步数：不是一条简单的 R2→LONG→B 直线

| 阶段 | 实际起点和增量 | 数据与曝光 | 累计身份 |
| --- | --- | --- | --- |
| R2 配方家族 | 官方 Chair EMA 初始化，Native/T1+role+Geo、fresh AdamW；并非 START 继续训练 | 历史 R2 为 26 父的 C20/C40 混合任务，6000 步 | 配方来源，不是 EXP67 权重祖先 |
| EXP67 / EXPANDED67 | 再从官方 EMA + Native/T1+role+Geo、fresh AdamW 开始，6000 更新；不是旧 R2/USER_MIX 权重续训 | 67 父、194 条 USER FULL/HALF 座面/靠背条件；48000 实例，每父716–717 | global6000 |
| FULL_S6000 | EXP67 live 非 EMA 生成器 + 匹配 AdamW/RNG，再加6000 | 47 个合格 TRAIN SEAT_FULL 父/条件；48000 实例，每父1021/1022 | global12000 |
| SEAM_SLOT_C200 | FULL_S6000 live + 匹配 AdamW/RNG，加200；新独立占用头 | 73/3f/85/8c 四条已见 SEAT_FULL；1600 实例，各父400 | global12200，SEAM200 |
| LONG | SEAM200 分叉再加800；生成器和原小头都继续训练 | 同四父；新增6400 实例，各父1600 | global13000，SEAM1000 |
| B_POOL47 | **从同一 SEAM200 分叉**再加800；并非从 LONG 生成器接着训 | 扩至47父；新增6400，43父136、4父138；同步训练 B 自己的占用头 | global13000，SEAM1000 |
| PLAIN_S1000 / S2000 / PLAIN3000 | B_POOL47 的 live/AdamW/RNG 续训轨迹；分别 +1000/+2000/+3000 | 原47父与原来源 owner 训练布局；头冻结且不调用；推理换用固定 LONG 布局 | global14000/15000/16000；SEAM累计2000/3000/4000 |
| MULTIPART_S1500 | **PLAIN_S1000/global14000** 的完整状态分叉，+1500；不是从 S2000 接训 | 47 SEAT + 31 BACK 条件；12000 实例，SEAT/BACK各6000；新 C-only 注意力头 | global15500，SEAM累计3500 |

PLAIN_S2000 是原 PLAIN3000 训练轨迹的真实 step2000 恢复点，随后在 Wave1 被完整评价，并作为一种开发取舍保留；不能称为“原3000步实验预注册的唯一2000步终点”，也不能称为独立训练出来的新2000步随机流。正式固定3000步主终点并未因此改写。

对实际 `training/PLAIN3000/exposure.jsonl` 只读统计：local≤2000 共 **16000 实例、47父，4父338次、29父340次、14父342次**。S2000 沿当前祖先生成器路径共15000更新、120000正式训练实例；其中 FULL 的47父阶段6000更新 + B扩池800 + PLAIN2000，合计8800更新/70400实例，另有四父SEAM200的1600实例。67父早期阶段包含更多部位，不能说模型只见过47父、或S2000仅训练2000步。LONG 的800更新没有计入 S2000 生成器血缘；其头是另一个被借用的已训练组件。

证据：[EXP67 报告](../../experiments/r2_user_data_scaling_32_67_v1/REPORT.md)、[FULL 报告](../../experiments/r2_full_seat_learning_curve_v1/REPORT.md)、[SEAM200 报告](../../experiments/r2_seam_slot_c200_v1/REPORT.md)、[LONG 报告](../../experiments/r2_seam_quality_four_v1/REPORT.md)、[A/B/C 报告](../../experiments/r2_seam_abc_v1/REPORT.md)、[PLAIN 3000 合同与结果](../../experiments/r2_seam_pool47_couple_v1/REPORT.md)、[Wave1 真实轨迹检查](../../experiments/r2_seam_quality_wave1_wave2_v1/WAVE1_RESULTS.md)、[MULTIPART 报告](../../experiments/r2_chair_multipart_seam_v1/REPORT.md)。

## 当前保留设置实际是什么

生成主干为世界坐标 Native/T1+role+Geo（Geo 的 P=I，非新邻接传播/局部 readout）；模型配置是 EquiDiT v3，12层、hidden768、12 heads、face_bin20、max_length800、coord encoding/pe_freq20、RMSNorm/QKNorm。EXP67 初始化后为194个可训练参数张量、130401155参数；官方主干继承182张量，role/Geo出口从零初始化。max_length800 是架构容量，不能据此声称 Chair 在高N800区域有充分训练证据；EXP67 实际 N133–256。

共同生成器优化：全参数训练，AdamW lr1e-5、betas(.9,.95)、eps1e-8、weight_decay0、clip1，无训练 EMA、无 scheduler；batch8/micro1，每批四clean+四x-width U(.9,1.1)，扩池阶段按父均衡且clean/width成对。模型 eval 模式保留梯度；BF16前向，FP32参数/优化器/积分，严格确定性、TF32关闭、SDPA math。训练先用原 Gaussian 做原匹配 OT，再做 HYBRID0.25，不对混噪结果重新 OT。来源 Y 是训练目标，评测推理只读取 C/N 与噪声。

**PLAIN 并非普通 triangle-soup FM。** SEAM 为每条来源 C 开放边设置 owner seam 槽，前两角引用原 C 顶点，第三角自由；Null 槽回 bulk，总 N 固定。固定 C/固定角不进入自由角 FM 分母；有 S 个激活槽时自由标量数为 `9(N−K)−6S`。混合 seam 面 attention 用 t，原 C 面用1，role/MLP/final 的角时间为[1,1,t]；受限 OT 和固定锚的 model adapter 仍在。`PLAIN` 仅表示不加 COUPLE3000 的 λ=.1 同源 tip/bulk 一步终点 MSE；没有 B 蒸馏、RL、新几何损失或 LOCAL_NOISE。

47父来源有400条 C 开放边、366条来源有邻面；一源面只占一个生成槽，最小 C 边 ID 做 owner，得360 owner、6 alias。18条来源接口第三点属于C，其中12 owner仍作为自由变量；这也是 B 扩池兼容规则的组成部分，不能把 B 当成只有样本数一个因素变化。

LONG 小头为 **24→32→1、SiLU、833参数**，每边只读 C 几何与N/K，不读取Y、父ID或查表标签；每C一次 logit≥0 后冻结布局，四噪声共享。24维含边中点/方向/绝对方向、C法向、包围盒跨度/中心、边长、K/N、开放边端点度数、常数。头独立 AdamW lr.001、betas(.9,.95)、wd0，class-weighted BCE 的[Null,occupied]权重=[8.25,66/124]；梯度不进入生成器。LONG 经过四父累计1000步，TP48/62、Null4/4，仍漏14条。B 自训头在部分熟悉条件严重少开槽；因此后续冻结 LONG 布局，不能把 `B_LONG_LAYOUT` 与 B 原始头输出混同。

实际代码：[原训练初始化](../../r2/pilots/user_data_scaling_32_67_v1/training.py)、[SEAM200 训练](../../r2/pilots/seam_slot_c200_v1/train.py)、[PLAIN/COUPLE update，显式 seam_forward 和角FM](../../r2/pilots/seam_pool47_couple_v1/training.py)、[C-only 小头](../../r2/src/meshflow_control/seam/occupancy.py)、[MULTIPART 注册](../../experiments/r2_chair_multipart_seam_v1/training_registration.json)。

## 同面板证据：应保留的正结果和反证

下面 POST 使用相同单次官方D0、GEO_C、来源绕序及固定C free定向；同28请求（四熟悉TRAIN父×4、三长期观察DEV父×4）和固定 LONG C-only 布局。FF/CF 是受非法面对影响的保留 free 面比例；I 是固定 C 查询的 ConflictCov_C。bulk≥30% 是合法 free-only 共边图的可达面积描述标记，不能当成整椅成功率、水密性或采用率。

| 模型 | TRAIN FF / CF / I (%) | TRAIN bulk≥30% | DEV FF / CF / I (%) | DEV bulk≥30% |
| --- | --- | --- | --- | --- |
| LONG | 14.351 / 1.507 / .266 | 11/16 | 16.127 / 5.548 / 2.315 | 0/12 |
| B_LONG_LAYOUT | 15.904 / 1.420 / .409 | 12/16 | 15.164 / 6.235 / 3.434 | 0/12 |
| PLAIN_S1000 | 16.282 / 1.246 / .229 | 11/16 | 14.183 / 4.649 / 3.463 | 0/12 |
| **PLAIN_S2000** | **12.987 / .886 / .177** | **13/16** | **16.075 / 5.612 / 3.699** | **0/12** |
| PLAIN3000 | 13.712 / 1.411 / .308 | 11/16 | 19.119 / 4.497 / 2.600 | 0/12 |

S2000 保留依据是熟悉条件上的质量/接续取舍，不是所有指标第一。它仍严格超过 Wave1 的 DEV FF<16%门槛；DEV小角33.169%、平均删60.583张free面（S1000为41.917），610父明显失面。Wave1五候选没有通过完整晋级条件；之后用户选择保留 S2000+HINGE 是后来的有限正结果保留决定，不倒改原实验胜负。[完整同面板表](../../experiments/r2_seam_quality_wave1_wave2_v1/REPORT.md)。

更早的两个独立对照不能与上表拼成统一排行榜：

- FULL_S6000 在六父×四噪声面板，EXP67→FULL 的 RAW明确采用12→16/24，POST有合法接口7→8/24；FF16.108→19.972%、CF3.237→4.454%、I1.428→2.193%，有代价。FULL/CONTOUR最终“明确采用且合法接口”均6/24、DEV0/12；FULL是充分训练参照，不是已解决接续。
- SEAM200 对同起点、同200步 FULL4：合法接口13→16/16、合法边131→175，13/16有seam→bulk路径、11/16可达≥30%；FF12.841→14.563%、I.671→3.302%。LONG普通延长后 I3.302→.266，但主体门槛仍11/16且旧正例有换出。原SEAM200大I坏尾是3.302%，不能误写成1.302%。

## 后处理保留范围

主链为标准Gaussian → Euler50 → 单次官方Chair D0 → 恢复C/seam硬锚 → 原GEO_C、来源C角序与固定C free定向 → HINGE_OPP015。这些操作依赖保存的来源 face/corner 身份、PRE_GEO全簇成员及原GEO簇；**不是拿任意PLY即可等价运行的通用修复**。2.5%L为保留对照，不是预注册主档。L固定约9.208138模型单位，不随每个输出bbox重定标。

HINGE_OPP015 的关键是整簇合并所带来**全部新增 incidence=2 边**都反向，而非只检被提名的hinge。相对原POST的28请求，FF14.3104→13.4902%、CF2.9116→2.7878%，新增105条合法边；原4624条合法边全部保留，零新非法CF/FF、零额外删面。bulk≥30%为13→14/28（TRAIN14/16、DEV0/12）；610/n1 I仍+0.048828pp，局部q/角度可退化。不是全指标零退化。[保留合同](../../experiments/r2_hinge_opposite_v1/retained_profile.json)、[报告](../../experiments/r2_hinge_opposite_v1/REPORT.md)。

ALPHA 在相同候选/排序先试原位置，再有限α={0,.25,.5,.75,1}；PRE_GEO全簇总位移仍≤1.5%L，含C锚簇只能固定原锚。HOLE 只填真实索引围成的三开放边环，至少邻接一张SEAM、所有邻面free；新面angle≥5°、q≤10、零新非法面对，记BRIDGE、来源父槽−1。BRIDGE可作为图路径，但不计原BULK面积，不假装旧面细分。有限队列一次扫描，不无限补洞或阈值搜索。

同28缓存 ALPHA_HOLE 的固定旧面FF13.49022→13.04628%，含新桥实际FF12.99550%；16张新桥本身降低FF部分只是分母增加。DEV608/n2接入38张原bulk面、28.820738%，仍低于原30%门槛；整体DEV仍0/12。I1.684570→1.696777%，原4729合法边全保留。该独立候选没有替换维护正式HINGE，不过后续 BACK 迁移合同确实采用 HINGE→ALPHA→HOLE。不同报告的“孤岛”有合法图/全部共边图差别，不混用121与221等计数。[ALPHA/HOLE报告](../../experiments/r2_hinge_v2_cpu_v1/REPORT.md)、[BACK流程](../../experiments/r2_chair_back_transfer_v1/REPORT.md)。

UNIFIED_RESCUE 又在 D0/GEO 前加最短边阻尼和禁止面自合并，保住更多面同时丢旧合法FF边、加新非法FF；这两个护栏不是保留主链默认。MULTIPART的UNIFIED_FINAL表必须双方都跑同链比较，不能拿其FINAL比S2000旧BASE，也不能把新增后处理收益算成训练收益。

## MULTIPART：更适合保留为 BACK 分支

训练从 S1000/global14000 新增1500，47 SEAT+31 BACK（条件数，不是78独立父），每批四SEAT+四BACK，clean/width配对，原SEAM/FM/OT后HYBRID不变。新头18049参数：原24维+C bbox相对中点xyz+实际y-up高度共28维，2层width32、4-head attention、dropout.1，unweighted BCE，fresh AdamW lr.001、betas(.9,.95)、wd.01、clip1，阈值logit≥0，无DEV校准；头梯度不进入生成器。生成器恢复匹配AdamW/RNG，原lr1e-5等不变。

原SEAT28 + BACK16，共15C/44请求。BACK FAMILIAR四条C本轮进入训练，不能称零样本；DEV官方预训练曝光未知。下表同一 UNIFIED_FINAL、旧S2000→MULTIPART：

| 子集 | FF% | CF% | I% | bulk可达面积% | ≥30%请求 |
| --- | --- | --- | --- | --- | --- |
| SEAT TRAIN16 | 12.858→13.113 | .611→1.312 | .168→.464 | 66.245→62.823 | 15→15 |
| SEAT DEV12 | 19.458→16.626 | 4.933→4.394 | 3.650→3.524 | 2.759→4.972 | 1→1 |
| BACK FAMILIAR8 | 22.475→22.256 | 4.072→2.404 | 18.640→6.909 | 5.677→37.803 | 0→5 |
| BACK DEV8 | 19.261→16.147 | 9.553→3.947 | 23.938→17.554 | 28.388→20.596 | 2→3 |

同BASE_POST也有真实训练正信号：BACK FAMILIAR FF21.455→18.965%、I18.475→6.409%、bulk2.072→44.812%、门槛0→5/8；BACK DEV I24.042→16.992%，但bulk14.390→9.604%、门槛仍1/8。因此 BACK 收益不全是新后处理，但未见BACK平均接续面积下降、SEAT有换例/小角/删面成本。BACK整体FINAL门槛2→8/16；I约12.23%仍未达<8%目标。该配方同时改数据、头和预算，不是混合部位单因素消融。[完整报告及179字段](../../experiments/r2_chair_multipart_seam_v1/REPORT.md)。

## 当前权重 stat（全部存在；未重新 hash/加载）

以下路径相对 `E:/MeshFlow/`，字节数来自本次文件系统 stat。大权重 hash 可从保留索引查既有身份；本次不把 stat 冒充内容哈希复验。

| 角色 | 当前实际路径 | 字节 |
| --- | --- | ---: |
| S2000完整恢复 | `experiments/r2_seam_pool47_couple_v1/checkpoints/PLAIN3000_step2000.pt` | 1565057323 |
| S2000推理 | `experiments/r2_seam_quality_wave1_wave2_v1/inference/PLAIN_S2000_generator.pt` | 521665979 |
| 配套LONG头 | `experiments/r2_seam_quality_four_v1/inference/LONG_occupancy.pt` | 6045 |
| MULTIPART完整恢复 | `experiments/r2_chair_multipart_seam_v1/training/checkpoint_step1500.pt` | 1565305291 |
| MULTIPART推理 | `experiments/r2_chair_multipart_seam_v1/inference/MULTIPART_S1500_generator.pt` | 521665851 |
| MULTIPART配套头 | `experiments/r2_chair_multipart_seam_v1/inference/MULTIPART_S1500_occupancy.pt` | 80975 |
| FULL完整恢复 | `experiments/r2_full_seat_learning_curve_v1/checkpoints/FULL_SEAT_step6000.pt` | 1565284779 |
| FULL推理 | `experiments/r2_full_seat_learning_curve_v1/inference/FULL_S6000_generator.pt` | 521666683 |
| 官方Chair D0 | `experiments/r2_postprocess_avmr_pmr_v1/assets/chair_denoiser_0030000.pt` | 7289979546 |
| 维护START完整点 | `native_t1/checkpoints/chair_hybrid_start_total5000.pt` | 1565074939 |

这里的“最有希望”应解释为目前可用、已观察到有限正结果的研究组合。三DEV父反复观察、四噪声不是四独立父、官方预训练曝光未知、来源Y仍有缺陷、非盲AI_PROXY不等于人评，均限制泛化结论。合法CF边、坐标保留、用途采用、bulk可达、FF/I和表面完整性必须分开；没有一个单一总分证明通用最优。本次未提出或启动任何新实验，未改变默认、清理、commit或push。
