# CHAIR_CONTROL_FEEDBACK_AUTONOMOUS_OPTIMIZATION_V1

**结论：本轮已实际完成有限自主研究与训练；没有候选达到联合晋级条件，保留 START。** 完成 3 条登记假设、4 个独立探索分支，各 1,000 更新，共 4,000 更新。两个候选是实际轨迹重放 TR 与其损失值限幅版 CAPPED；另有原 FM 续训对照和同组织 teacher 重放对照。它们均从指定 START 分叉，终点均为本轮 1,000 / 累计 6,000，不能把不同分支的更新相加成一个模型的训练深度。

实际 Euler50 尝试 225 次，保存并完整评分 224 份输出：DEV 144 份，TRAIN_FEEDBACK 80 份。反馈生成尝试 81 次，含 1 次已完成模型计算但保存失败的尝试；随后只按相同冻结输入恢复。真实模型调用失败、非有限、C 改写、面数错误均为 0。DEV 的所有 500/1000 检查点完成；独立训练重复和 CONFIRM 为 **NOT_TRIGGERED_NO_QUALIFIED_DEV**，不是资产缺失，也不是已经通过确认。CONFIRM 的输入和新噪声已在训练前冻结，没有生成其输出或查看其分数/图。

本轮在用户约 4 小时窗口内完成。起始为 2026-10-06 09:18:49 UTC（北京时间 17:18:49），预设最晚结束 13:18:49 UTC。最后一次真实模型调用返回于 11:59:10 UTC。研究收尾时间及实际总墙钟见 `delivery/completion_receipt.json`，打包验证时刻另见交付收据，不能把窗口上限写成实际 GPU 运算时间。1个FM对照加3个实验配方的名额用完且无人过门槛，按停止合同结束，不为凑足时间追加训练。公共代码、旧数据、原评分器、旧 raw、旧报告的 167 个保护文件 hash，以及另行核验的 START 权重 hash 全部未变；没有切换默认模型、commit 或 push。

**直接回答九个问题。**

1. **C 是否完整保持？是，在明确的模型坐标合同内。** 全部 224 份留存输出的 51 个采样状态、最终 NPZ、二进制 PLY 导出重读通过。原 C 的 FP32 坐标、面数、绕序、重复面多重集及映射保留，free 面数为 N−K。条件文件另存原始 `C_world`、transform、源面/顶点关联；交付 PLY 是模型坐标，不冒称近似逆变换后的 world 坐标逐位相同。精确保留不代表没有遮埋、穿插或非流形。
2. **哪个方案改善了接合？没有一个已证明在全部约束下稳定优于 START 与匹配对照。** CAPPED500 有真实接合正信号：S 相对 teacher +6.0072 个百分点，6/8 父改善；但形状和视觉约束失败。TR1000 有冲突减少、外露增加及相对 teacher 的形状误差下降，伴随 S 明显下降，不能称接合成功。
3. **S/I/E 是否交换？是。** TR1000 对 teacher：S −4.6770、I −0.9613、E +8.8078 个百分点；CAPPED1000 对 teacher：S −0.8159、I +0.8118、E +6.1696 个百分点。S/E 越高越好、I 越低越好，不能相加成一个总分。E 的平均收益高度集中于 f596 父。
4. **椅子主体是否合理？存在合理候选，也存在明显退化，不能整体认证。** 341 的不同噪声出现主体缺失，33ec/c1b 的背部碎裂、f596 的长下挂薄片等没有被局部指标充分惩罚。未观察到全组都塌到 C 上的行为，但“全组无新增主体崩坏”门槛不满足。
5. **收益对两个参考都成立吗？不成立。** TR1000 的 Dsym/MSE 对 teacher 改善 1.19%/16.81%，对 START 却恶化 13.40%/17.06%。CAPPED1000 的 MSE 对 teacher 改善 10.49%，对 START 恶化 25.95%。优于继续训练的对照不能替代优于 START。
6. **独立训练流与确认是否支持同方向？尚无此证据。** 由于 DEV 联合门槛未过，按协议没有独立重复或 CONFIRM。CPU repeat verifier 的测试与原 teacher 的限幅不激活证明均不是独立训练重复。
7. **代价是多少？** 35,000 次训练前向、35,000 次训练反向、4,000 次 optimizer 更新；11,250 次采样前向，总样本等价前向 46,250。GPU worker 持锁占用合计 7,623.575 秒（2.118 小时），包含该 worker 内加载/保存/CPU 等待，不能等同纯 GPU kernel 时间。预检整网前向/反向均为 0。
8. **应保留谁？保留 START 为正式默认。** 本轮新权重作为已验证可加载的研究终点保留，CAPPED500 的局部正结果与全部负结果一并归档；没有“获胜终点”。
9. **尚未建立哪些能力？** 普遍接合改善、对新噪声/独立训练的稳定迁移、严格水密/可编辑性与真实人类可用率均未建立。已建立的是可追溯的反馈训练、固定评分、C 硬保持和对这些有限假设的实际否定/局部支持证据。DEV 是反复观察的开发面板，不能称独立测试。

**实际学习方案与反馈链。**

基座严格核对为 Native/T1 + 实际 Geo（P=I）+ HYBRID-FM，`OT_HYBRID_pilot1000_total5000.pt`，SHA256 `d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80`。恢复原 AdamW step5000 与动量：lr=1e-5、betas=(.9,.95)、weight_decay=0、clip=1，无新 EMA/scheduler/参数。主训练保持原 32 父、453 个 READY C 任务、repair_v2、自然 N=128..245、原 20%/40% 条件；effective batch8、micro1、8 个不同父，4 个 alpha=1 与 4 个独立 U[.9,1.1]。保留 logit-normal t、原 pre-scale free-only nested OT、全 batch 有效 free 标量归一、混噪后不再 OT。严格确定性、BF16 forward/FP32 参数和几何积分、SDPA math、TF32 关闭。

| 登记项/实际分支 | 学习干预 | 实际更新 | 每更新模型 F/B | 角色与结果 |
|---|---|---:|---:|---|
| FM_CONTROL | 原 HYBRID-FM 继续训练 | 1000 | 8/8 | 基础续训对照；形状对 START 退化 |
| H1 TRAJECTORY_REPLAY（TR） | 主 FM + 0.1×真实轨迹状态的 GT 目标速度 MSE | 1000 | 9/9 | 候选；训练池收益未稳定迁移，DEV 不晋级 |
| H2 REPLAY_FM_CONTROL（teacher） | 相同队列/父/C/Z/t/额外 F/B，辅助项仍为原 HYBRID teacher-line FM | 1000 | 9/9 | 必要匹配对照，占一个配方名额 |
| H3 TRAJECTORY_REPLAY_CAPPED | 同 H1，仅限制实际辅助损失值相对当前主 FM 的比例 | 1000 | 9/9 | 根据已完成 H1/H2 反馈登记；500 局部正信号，最终不晋级 |

START 在预先登记的 8 个训练父×2 噪声上完成真实 Gaussian→Euler50，保存 51 态和 50 速度，并计算固定 S/I/E、质量与失败状态。重放按 S/I/E 三队列循环，每队列遍历全部 8 父，在父内选两份噪声中对应指标最坏者（S lower/I upper/E global lower，平局选低噪声编号）；t=.6/.8 每完整队列周期交替。实际辅助反向覆盖 12/16 条轨迹、8 父、24 个唯一状态目标；S/I/E 各 334/333/333 更新，两个 t 各 504/496。不能把已生成 16 条全部说成参与梯度。

TR 将真实终点 free 与原训练 GT free 做合法面/角双射，在对应终点槽位形成 `u=(Y_GT_matched−X_real_t)/(1−t)`，对真实中间态的模型速度反传。GT 没有被自身生成结果替换；硬 S/I/E 通过确定性选例队列影响学习，并非把 CPU 浮点分数伪称可微 loss。它同时改变辅助状态与终点 GT 对应关系，因此是复合干预，不能单独归因于“真实状态”一项。推理没有 GT/Gamma/父 UID/目标 bbox 输入，网络结构与 Euler50 未变。

H3 在前 3 臂及其完整 DEV/终点反馈已结束后冻结，理由是 TR 辅助梯度占优、频繁 clip 与训练/DEV 分离；并非中途更换同一条训练曲线的目标。公式为 `lambda_eff=min(0.1, stopgrad(FM_sum)/max(stopgrad(aux_MSE),1e-30))`，实际 loss 仍有原主 FM 正权重。限幅的是 **loss 数值，不是梯度范数**。teacher 原 1000 步全部严格不触发此 cap，逐步 FP32 乘法和系数分支等价证明通过，所以它可作为该探索流的匹配 teacher 对照；这不是重新训练了一条对照。H3 与 TR 比较作为单系数规则的额外描述对照保存，未篡改原 A/B 门槛。

完整开始前决策、500/1000 反馈与冻结身份见 `hypothesis_register.md`、`decisions/`、`protocol.json`。H3 只使用剩余的第 3 个实验配方名额。原始 protocol/hypotheses 保留，H3 扩展另存。

**固定 DEV 的完整主结果。**

每标签均为原 A20 全 8 父×原两噪声；先父内均值，再父等权。表内 S/I/E 为百分数，E 用原保守下界；完整上下界、未决比例、资格与逐例值保留在 CSV/JSON。原裁判为 `chair_boundary_attachment_control_metrics_v1r1`，没有修改数学定义或将 NA 置零。E 的 PARTIAL 继续保留，所有 gate 用保守区间；表内四舍五入不参与判定。Dsym/512 MSE 均仅评分 free，不用 C 到自身的零距离稀释误差。

| 模型/步 | S % ↑ | I % ↓ | E % ↑ | Dsym2048 ↓ | MSE512 ↓ | p95/L0 ↓ | CF % | FF % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| START | 14.2690 | 4.2511 | 84.4864 | .00161980 | .00153107 | .0688447 | 6.0135 | 92.6270 |
| FM500 | 12.8969 | 5.6885 | 85.5008 | .00172044 | .00195596 | .0752805 | 6.1447 | 91.5714 |
| FM1000 | 12.2617 | 3.9978 | 85.8689 | .00186535 | .00216486 | .0780196 | 5.6359 | 91.6464 |
| teacher500 | 12.7354 | 3.7689 | 85.0493 | .00175094 | .00198533 | .0763590 | 6.1931 | 91.5410 |
| teacher1000 | 13.6907 | 4.2664 | 85.3727 | .00185903 | .00215434 | .0764469 | 6.1657 | 92.0667 |
| TR500 | 5.5284 | 3.0823 | 94.1719 | .00178461 | .00196956 | .0750159 | 6.5882 | 89.9232 |
| TR1000 | 9.0137 | 3.3051 | 94.1805 | .00183688 | .00179229 | .0747134 | 6.5000 | 88.3747 |
| CAPPED500 | 18.7426 | 4.4769 | 95.3601 | .00158225 | .00172166 | .0677851 | 6.7028 | 89.4060 |
| CAPPED1000 | 12.8748 | 5.0781 | 91.5423 | .00191475 | .00192839 | .0732064 | 7.8226 | 89.3431 |

| 候选 | 相对 teacher 的 ΔS / ΔI / ΔE（百分点） | 控制 A | 控制 B | 双参考形状 | 完整视觉 | 联合 |
|---|---|---|---|---|---|---|
| TR500 | −7.2070 / −0.6866 / +9.1226 | FAIL | FAIL | FAIL | FAIL | FAIL |
| TR1000 | −4.6770 / −0.9613 / +8.8078 | FAIL | FAIL | FAIL | FAIL | FAIL |
| CAPPED500 | +6.0072 / +0.7080 / +10.3108 | PASS | PASS | FAIL | FAIL | FAIL |
| CAPPED1000 | −0.8159 / +0.8118 / +6.1696 | FAIL | FAIL | FAIL | FAIL | FAIL |

A/B 是已冻结的控制子门槛，必须与形状、硬约束和视觉全部满足才能晋级。CAPPED500 不是联合 PASS；其 MSE 对 START +12.4476%，超过 5%，同时有主体退化。它相对 START 的 S +4.4735、I +0.2258、E +10.8736 个百分点值得保留，但不能为选这个中间点改变预先指定的 1000 步终点。最终 CAPPED 对 START 的 Dsym/MSE/p95 分别 +18.2094%/+25.9502%/+6.3354%。

全输出的两个 Dsym 方向、512 coverage、面积比、零面积/近退化/小角面、CF/FF 原定义分项均在 `evaluation/*/DEV/per_output_metrics.csv` 与 `report_tables/DEV_all_models_metrics.csv`。例如 CAPPED1000 的两个 Dsym 方向为 .0019232941 / .0019062151，coverage=.3391113，面积/GT 比=1.1007081；没有新增零面积或近零面积面不等于没有细长薄片、相交或主体破损。原 FF 比例普遍很高，下降只说明该原指标下相对改善，不能据此宣布网格拓扑合法。

**完整父分布、两噪声与尾部。**

逐父、逐输出和原噪声配对全部保留，含区间，不只围绕最坏父叙事。详见 `comparisons/DEV/`、`derived/attribution/DEV/latest.json` 指向的完整归因表，以及 `report_tables/FINAL_PARENT_DIGEST.md`。

TR1000 对 teacher：S 仅 2/8 父改善，中位差 −.031311；I 6/8 父有利，中位减少 .019897；E 5/8 父有利，中位增加 .005127。E +.088078 的净均值收益中 f596 贡献 94.61%，去掉它仍有 +.005423，但远小于全组值；E 的两噪声方向在 3/8 父相反。TR 对 START 的 E 收益 f596 贡献 88.23%。S 的下降并非只由一个坏父造成：对 teacher 5 父下降、2 父上升、1 父持平，8 个留一父均值都为负（−.063416 至 −.023909）。

CAPPED1000 的两噪声 S 差方向在对 START 的3/8父、对teacher的4/8父相反；对 START 两噪声组均值分别为 +0.6578/−3.4462pp，对teacher为 −3.4849/+1.8531pp。这些方向翻转不能描述为稳定进步。CAPPED1000 对 teacher：S 仅 1/8 父改善，中位 −.006420；I 虽有 5/8 父改善、中位改善 .009277，均值却恶化 .008118，说明尾部成本不能由改善父数量掩盖。E 仅 3/8 父改善，中位 −.004532；f596 贡献净 E 收益的 107.59%，去掉该父剩余 7 父均值为 −.005350。相对原 TR，CAPPED 的 S +3.8611 个百分点，I 恶化 +1.7731，E 下降 −2.6382，Dsym 恶化 4.2394%，是取舍而非一致改进。

严重回退按原定义单列：534b 父的 I 对 teacher 在 CAPPED500 增加 **13.9160 个百分点**，CAPPED1000 增加 **17.7002 个百分点**，不能由其他父或 E 的改善抵消。TR500/TR1000 对 START/teacher，以及 CAPPED500/1000 对 START，均无父均值超过10pp的严重 I/E 代价；全部八个比较均无确认或可能的 >10pp E 下降。完整清单与两噪声值见上述父分布摘要和原 gate JSON。严重父不会从平均值分母中删除。

**训练池收益真实存在，但迁移证据不足。**

四个 1000 终点均从磁盘重新加载，在原 TRAIN_FEEDBACK16 条件/噪声上重新完整 Euler50，随后冻结评分；没有把这些终点输出重新加入训练。它们是受监督池内的生成诊断，不是未见噪声测试。父对象不同，训练池与 DEV 的增益差是描述性迁移差，不是同父因果估计。

| 1000 终点相对 teacher | TRAIN ΔS / ΔI / ΔE pp | DEV ΔS / ΔI / ΔE pp | TRAIN S 改善父 | DEV S 改善父 |
|---|---|---|---:|---:|
| TR | +7.9714 / −3.6072 / +2.8815 | −4.6770 / −0.9613 / +8.8078 | 3/8，中位0 | 2/8 |
| CAPPED | +8.0275 / −3.8940 / +2.8689 | −0.8159 / +0.8118 / +6.1696 | 6/8，中位+.036150 | 1/8 |

TR 在训练池相对 START 的 S +21.1476pp、I −1.1902pp、E +3.4917pp；Dsym/MSE/p95 分别下降 57.40%/50.95%/44.60%。但训练 S 对 teacher 的增益仍集中于 1fe0 和 55740，两父合计贡献净增益的 116.06%，其他父有抵消。CAPPED 相对 START 的训练 S +21.2037pp、I −1.4771pp、E +3.4790pp，控制改善的父分布较广，却仍未迁移到 DEV。完整四臂双池表见 `derived/transfer_attribution/adbb7d1495a8e0fb/TRANSFER_EVIDENCE.md`，没有将训练池改善改名为泛化成功。

**梯度、代理误差与数值限幅的实证。**

| 分支 | 主 FM 均值 | 实际加权辅助 loss 均值 | clip比例 | 辅助/FM 梯度范数比中位 | 梯度夹角余弦为负 |
|---|---:|---:|---:|---:|---:|
| FM | .109428 | 0 | 1.8% | NA | NA |
| teacher | .109389 | .001226 | 1.8% | .072897 | 9/41 |
| TR | .123230 | .519569 | 85.5% | 2.822769 | 23/41 |
| CAPPED | .113675 | .095751 | 28.0% | .850085 | 25/41 |

梯度来自真实辅助 backward 的 leaf hooks，在 step1 和每25步记录，共每辅助臂41次，不是两条范数相减推算，也没有额外整网反向。CAPPED 有751/1000次限幅，系数均值 .04524070，中位 .03073394，范围 [.000568566,.1]；梯度范数比仍可达到9.3146，故不能称梯度被固定比例限制。各臂内部匹配相同24个唯一状态并等权（TR与CAPPED彼此使用同24状态，teacher使用不同的teacher-line状态），前96步到末96步的未加权辅助 MSE：TR 15.610978→2.054055（−86.84%），CAPPED16.704400→4.880052（−70.79%），teacher .014502→.009767（−32.65%）；每臂24状态均下降。实际队列加权口径不同，另表保存，不能混用百分比。

CAPPED 激活时，加权辅助 loss 近似等于当前 FM 是公式的直接结果，不能据此把这条 loss 的下降当成独立学习证据。它后96步 cap激活降至51.04%，clip反而升至48.96%；训练 dynamics 并未单调变温和。这些日志支持干预有实际梯度作用、限制了其相对强度，但没有证明梯度竞争是 DEV 失败的唯一原因。全部24目标、6000组相同步主输入比较、74,000组调用 BEGIN/RETURN（35k F+35k B+4k updates）的训练审计通过；此数与含采样的总账本口径分别报告。

**实际视觉复核与代表案例。**

全量500/1000图集按同父固定相机、画幅和材质展示 C / START / FM / teacher / TR / CAPPED / GT；GT也按同一C蓝色、free橙色。每父两噪声、四视角加线框及 S/I/E 叠图，原网格没有平滑、补洞、焊接或裁异常。完整七列每个检查点480个mesh视图单元、240个反馈叠图单元，见 `visuals/INDEX.html` 与各 manifest。全组候选的实际看图结论是 **AI_VISUAL_REVIEW**，human usability始终NOT_RUN。审阅记录绑定原图片hash，不以自动渲染完成冒充已看图。

| 父/噪声/检查点 | 实际观察及解释 |
|---|---|
| 341 n0，1000 | FM/teacher已有相对START的箱体侧面丢失；CAPPED进一步呈压缩、碎裂的自由主体与known横条分离。该失败不能全部归给新增模块，但不能通过双参考视觉门槛。 |
| 341 n1，500→1000 | CAPPED500出现明显下部缺失；CAPPED1000恢复为连贯箱体，不能把500坏例沿用到终点。TR1000仍有严重前下部破碎。 |
| 33ec n1，1000 | CAPPED减少TR在known座面一侧的重复/遮挡，但有破碎座面和背侧尖片；teacher整体更完整。局部 I/E 改善不代表整体质量变好。 |
| 534b n0/n1，1000 | 椅子大体完整；n0 CAPPED有新增长细座下尖片，n1 I叠图有明显紫色重复覆盖。仅凭轮廓难以判断C相容。 |
| 866 n0/n1，1000 | 继续训练的各臂均恢复圆孔背部，改善不是TR/CAPPED独有；CAPPED减小n1大下挂薄片，但小地面三角仍在。 |
| f596 n0，1000 | CAPPED出现抬高扶手状薄片和近地长下挂三角，明显差于TR；known上的I/E仍大体灰/绿。E均值主贡献父也可有视觉代价。 |
| 86ec n1 / c1b n1，1000 | CAPPED分别有背部覆盖C与座裙加厚、碎冠破背；不得只展示外露分数改善。 |
| bdd n1，500→1000 | CAPPED500的大下挂片在1000已消失，终点座面比teacher连续。正向局部变化同样保留。 |

最佳 S、父中位附近、最差 S/I/E（含并列）及指标改善但视觉变差的代表选择由现存全组配对表确定，来源在 `visuals/representatives/**/selection_manifest.json`。这些代表图是全量图集的索引，不代替完整16请求。不能从技术看图推出人类可编辑率。

**预算、调度、恢复与偏离说明。**

| 项目 | 实际 | 上限/口径 |
|---|---:|---|
| optimizer更新 | 4000 | 6000；探索4×1000，未触发重复2000 |
| 训练F/B | 35000 / 35000 | 所有模型batch1，含辅助；无activation checkpointing重算 |
| Euler50尝试 / 保存 | 225 / 224 | 512尝试；含1次保存失败和相同输入恢复 |
| TRAIN_FEEDBACK尝试 / 保存 | 81 / 80 | 128尝试；5模型×16份已保存 |
| DEV保存 | 144 | START16 + 4臂×2检查点×16 |
| 采样NFE | 11250 | 25600；225×50 |
| 样本等价总F | 46250 | 160000；35000+11250 |
| 额外真实预检F/B | 0 / 0 | 64/32 |
| GPU worker持锁占用 | 7623.575秒 | 8小时；已闭合10个区间，最大并发1 |
| 新权重 | 12份，18,780,889,156字节 | 8个500/1000永久终点 + 4个latest900恢复点 |

遵守本任务明确的GPU单worker合同；调度前与各波重新测量资源，没有因空闲显存擅自并发GPU。RTX5080共16,303MiB，主机24逻辑CPU，CPU评分/OT/可视化各限2线程，可与独立GPU阶段重叠。整个设备监测峰6,601MiB；WDDM/WSL的本进程显存为NA，不能将设备总占用或PyTorch allocated当作进程峰值，allocated和reserved也不相加。保留至少2GiB/10%余量原则，未杀其他用户进程。

训练worker阶段墙钟约 FM1731.7秒、TR1840.6秒、teacher1830.9秒、CAPPED1849.2秒（含该阶段采样/加载保存等，以receipt为准）；输入加更新的逐步中位约1.4278/1.5321/1.5267/1.5501秒。它们不是纯GPU算力基准，CPU并行竞争和首次8000 OT/后续复用8000缓存都会影响耗时。占用总计取闭合锁区间，不再叠加其内部forward耗时。

科学定义、数据、noise、优化器、主FM、评分尺子与停止门槛没有静默改变。以下实际工程修复和身份修正保留原件、diff、失败日志与收据：

- 首次 START 已完成50个有限前向且C保持，但旧 `save_npz(path, **data)` 与 `path=states` 字段命名冲突，保存抛TypeError；计入1次已返回完整采样损失，仅同冻结输入恢复一次。没有抽新噪声寻找好结果。
- 数据库C元信息适配、评分导出嵌套字段修正；旧adapter把新生成START记作CACHED的来源标记保留原文并在独立provenance修订中更正。START DEV16与历史相同输入raw逐位一致，最大差0，但本轮32份START确为新生成。
- 梯度leaf hook在baseline进程载入后加入；baseline实际执行源码另存并hash，主训练数学AST核对不变，没有额外F/B。CPU guard、日志hash/布尔读取检查器的工程修正另存，不是科学失败重试。
- 原3臂总入口在所有3000更新和DEV完成后，遇到缓存图manifest的JSON list/runtime tuple类型差异而退出1；规范化JSON后语义与hash一致，独立完成后处理，**零GPU重跑**。恢复收据为 `decisions/original_exploration_completion.json`。
- 原teacher探索流的cap不激活证明用于避免无意义重训，H3原recipe与源码hash在开跑前冻结；独立重复验证器只有CPU测试，实际NOT_RUN。

工程恢复没有跳过坏batch、重置预算或修改输出几何。完整调用对账无pending，模型层失败0、OOM0；最终核验与全部12新权重的实际全文件hash完成。保留完整checkpoint模型/optimizer/RNG/游标/池版本；四个1000终点均实际重新加载完成生成。它们使用本轮 `checkpoint.load` schema，不能假定公共START旧schema入口直接接受，复現路径见 `REPRODUCE.md`。

**交付清单、复核路径与冻结身份。**

本地完整成果根目录为 `E:\MeshFlow\native_t1\runs\chair_control_feedback_auto_v1`。唯一轻量review ZIP小于9,900,000字节；为延续云端小包要求，不塞入18.78GB新权重、raw数组与全分辨率PNG，均在本地保留并提供精确路径/hash。逐例CSV和实际评分源码必须物理入包。完整大型图集manifest在包内以 `.json.xz` 无损保存，恢复后核对原SHA；实际审阅/代表选择JSON及逐例CSV直接可读。重复的完全相同源码/证据仅以hash别名压缩，`RESTORE_ALIASES.py`可校验恢复，省略清单明确区分“重复文件”与“未嵌入大资产”，不会把路径清单冒称原数据已入包。

| 任务所需产物 | 实际位置 |
|---|---|
| 1 主报告 | `CONTROL_FEEDBACK_AUTONOMOUS_REPORT.md`（本文件） |
| 2 冻结协议 | `protocol.json`、`registration/`、`code_snapshots/` |
| 3 ≤3假设 | `hypothesis_register.md`、原 `hypotheses.json` |
| 4 前后决策 | `decisions/research_decisions.jsonl`、`decisions/training_starts.jsonl`、H3recipe/500review/final_stop |
| 5 四套清单与反馈池 | `manifests/{TRAINING,TRAIN_FEEDBACK,DEV,CONFIRM}.json`、`feedback/pool_v1/manifest.json` |
| 6 起点/全部权重身份 | `derived/checkpoints/checkpoint_manifest.json` 及CSV |
| 7 可加载研究终点 | `checkpoints/explore_*_{500,1000}.pt`及其实际manifest路径；四个1000终点重载PASS，未晋级 |
| 8 raw/C映射/导出审计 | `runtime/`、`inputs/conditions/`、`derived/audit/final_latest.json`内逐文件精确路径/hash |
| 9 S/I/E逐例/父/噪声/区间 | `evaluation/<label>/<pool>/per_output_metrics.csv`、`per_parent_metrics.csv`、`per_noise_metrics.csv`；`comparisons/DEV/` |
| 10 形状/面积/CF/FF/成本 | 同逐例表、`report_tables/`、`training_diagnostics/final/`、`derived/transfer_attribution/` |
| 11 全图集/代表/来源 | `visuals/INDEX.html`、`visuals/gallery_catalog.json`、图集manifest、reviews、representatives |
| 12 实际账本 | `ledger/`、`resources/`、`derived/audit/final_latest.json` |
| 13 最小实际入口 | `REPRODUCE.md`；独立包 `experiments/mf_h1_local0/chair_control_feedback_auto_v1/` |
| 14 实际代码/diff | `delivery/code.diff`、`delivery/source_code/`、`delivery/source_scoring/`及review ZIP对应成员 |

关键SHA256：

| 文件 | SHA256 |
|---|---|
| 原任务合同 | `68138316a1d073487eced2781c3a99c49f07a87a96c8407df0b970eea3b1109e` |
| protocol.json | `5cb91c82d922396cfcee42355e9dadc79424a4dccab2f6501f4eb5e2e68961e8` |
| DEV manifest | `f5fb15ec102d953c075513ff17ed280c79f33b4b4f90ed2958d67c926c45d779` |
| TRAIN_FEEDBACK manifest | `db92918feb88389fa0e5765044e123c27d465bb429527bb3dc6e2afd72f127a6` |
| CONFIRM manifest | `bc2434aba65de60f22063369f07caefa83316793cad0c66069b848a5b0c0f132` |
| 反馈池manifest | `bbe0dacda4b32d3c5718fcb4855276642a2f1fc62ca5e49dc69698adbb6e5209` |
| H3 recipe | `d66d4822a4d6164fcc9883b2eac2fd7e923d7443036ca56dfb3f8d7e964c6c21` |
| 原控制评分源码 | `a6036ccd2f6063a8a863591d4aa79e5358de7ab617449a4c19b133f095ecf9ab` |
| 原质量评分源码 | `c287c97f0e76bf59ec99fe05f7fe494f27d69799d8ae0a938844d3a3663be01c` |
| 最终严格账本审计 | `4eb474b74b6d6411dbc4997853cf43f4eaa9b9eae4655040756a129eb19b61e5` |
| 最终新权重清单 | `376d75e19e8da359f8d92873ca88301d76e598e706859cb19e5c08505d76c26f` |

**下一步优先级仅作为未执行建议。** 若另开获批研究，优先检验“反馈池状态/时间覆盖不足导致收益停留在已选轨迹”，而不是继续放大同一辅助项。可检验预测是：在固定总更新/辅助F/B预算和原GT/主FM/DEV门槛下，扩大受监督的实际状态覆盖应提升未参与反馈的训练条件与固定DEV的广泛S改善，同时控制I/E及主体代价；若只改善旧24状态或同噪声训练父，则否定此预测。本轮尚不能区分池覆盖、目标对应关系、时间分布与一般续训漂移的因果份额，也不据此自动改变loss或启动下一轮。当前固定任务结束，保留START与全部证据。
