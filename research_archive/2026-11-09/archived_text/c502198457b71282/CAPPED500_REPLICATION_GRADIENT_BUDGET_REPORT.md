# CAPPED500 Replication and Gradient Budget Report

任务：`CHAIR_CAPPED500_REPLICATION_AND_GRADIENT_BUDGET_V1`。执行日期：2026-10-07（Asia/Shanghai）。固定三臂、两个终点已实际完成；所有生成和权重保留在本独立目录。

## 实际执行账本

| 项目 | 实际完成 | 登记上限/范围 |
| --- | --- | --- |
| 正式optimizer更新 | T/L/G各1000，共3000 | 3000 |
| 正式训练模型前向/反向 | 27000 / 27000 | 每更新9F/9B |
| 真实接线预检 | 12F、12B；1次隔离optimizer副本更新并丢弃 | ≤24F/12B；不保留预检更新 |
| 新完整Euler50 | 240次尝试/240成功；12000 NFE | ≤290次尝试/14500 NFE |
| 历史OLD缓存 | 48条直接核验复用 | 没有重生成 |
| 最终评价 | 288条：9标签×8父×4噪声 | OLD144/FRESH144 |
| 顶层模型前向 | 39012 | ≤41524 |
| 模型反向总数 | 27012（含12预检） | CPU玩具测试另列 |
| CPU小模型测试 | 累计180F/184B/18玩具更新；最终15项通过 | 真实模型0、CUDA false |
| 模型调用失败/未闭合调用 | 0 / 0 | 加载/展示工程故障另列 |
| GPU累计持锁 | 6501.261100s / 108.354min | ≤14400s；最大1 GPU worker |
| 反馈刷新/新增池/CONFIRM评价 | 0 / 0 / 0 | 固定原反馈池和DEV父 |


调用原件为 `ledger/calls_*.jsonl`，阶段组织为 `ledger/orchestrator.jsonl`，持锁原件为 `ledger/gpu_occupancy.jsonl`。总前向39012=27000训练+12000采样+12预检；正式更新3000与预检副本1次明确分列。CPU审计自身的 model_calls=0 不代表本实验未训练。

最终运行审计PASS：17段占用全部闭合，855个登记保护来源hash一致；直接重读288个NPZ/PLY与51状态路径全部PASS。最终训练审计438533项检查通过，三组各1000更新、3000组跨臂主输入配对及1000组L/G辅助身份配对一致，每臂42个真实参数更新快照。没有默默换坏batch、补噪声或中途修改科学定义。

设备RTX 5080，总显存16303MiB，设备显存监测峰值6320MiB；CPU24逻辑核，起始Windows可用RAM68.326GiB、WSL可用RAM观测58.018–61.138GiB。GPU遵守本任务单worker锁，尽管16GiB卡还有余量，也未扩大并发；CPU匹配、评分与审计独立重叠，进程各限2线程。设备显存峰值含桌面/其他用户占用，WSL/WDDM未给出可靠进程显存时记NA，未拿权重大小或PyTorch allocated代替进程VRAM。持锁时间包含加载、保存及持锁CPU等待；不同阶段墙时不能当成孤立kernel基准。

评分原始调用合计339.0924秒（48历史分数复用、240新输出评分），0评分失败；完整OLD/FRESH图集CPU渲染69.1483/72.4487秒。时间来源 `status/cpu_completion_receipt.json` 及各原summary；恢复后watcher的completed数组为空是展示计数缺口，18个完成面板与288 cells均已独立核验，不据此误计为0。

后半程4000个已登记主FM OT匹配在CPU预计算，原OT定义、输入与随机流未变；4000次均返回、无失败/重试。计算109.2325秒，含启动及保护核验180.9830秒；三臂后续实际输入hash和cache使用均一致。这不是新的反馈池匹配或额外训练样本。

### 各GPU阶段实际持锁及设备峰值

| 阶段 | 状态 | 持锁s | 设备峰值MiB | 进程峰值MiB |
| --- | --- | --- | --- | --- |
| preflight_real_1000_BOTH | PASS | 30.189 | 6119 | NA |
| sample_START_1000_FRESH | FAILED | 0.135 | NA | NA |
| sample_START_1000_FRESH | PASS | 60.841 | 3355 | NA |
| sample_HIST_TEACHER500_1000_FRESH | PASS | 53.115 | 3349 | NA |
| sample_HIST_CAPPED500_1000_FRESH | PASS | 54.828 | 3362 | NA |
| train_T_500_BOTH | PASS | 967.750 | 6320 | NA |
| sample_T500_1000_BOTH | PASS | 103.838 | 3366 | NA |
| train_L_500_BOTH | PASS | 949.247 | 6211 | NA |
| sample_L500_1000_BOTH | PASS | 104.897 | 3357 | NA |
| train_G_500_BOTH | PASS | 936.356 | 6215 | NA |
| sample_G500_1000_BOTH | PASS | 95.299 | 3358 | NA |
| train_T_1000_BOTH | PASS | 946.665 | 6200 | NA |
| sample_T1000_1000_BOTH | PASS | 102.960 | 3342 | NA |
| train_L_1000_BOTH | PASS | 933.357 | 6200 | NA |
| sample_L1000_1000_BOTH | PASS | 109.933 | 3344 | NA |
| train_G_1000_BOTH | PASS | 944.057 | 6203 | NA |
| sample_G1000_1000_BOTH | PASS | 107.794 | 3356 | NA |


失败加载段0.135382秒无模型调用；该失败短段的设备峰值记NA；同名成功恢复段的观测不借用于失败段。

### 保护身份与实现范围

共同START的SHA256为 `d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80`。全参和AdamW step5000、动量与二阶矩均恢复。结构仍是Native/T1 +实际Geo(P=I)+HYBRID-FM；无新增参数。旧teacher500和CAPPED500仅冻结推理参考，不作本轮训练起点。

三臂共享新的主流seed **3594377816**，由登记SHA命名空间推导且不同于旧流。原32父/453 READY、effective batch8/microbatch1、每批8父、4个alpha=1与4个U[.9,1.1]、logit-normal t、free-only nested OT、HYBRID .25、整批free坐标归一、x1头到velocity转换均保留。原24状态/12轨迹、S/I/E队列及父内最坏噪声不刷新；teacher-line辅助epsilon继续使用冻结旧seed20261008。

三臂同载体、BF16前向和FP32参数/积分，严格deterministic且warn_only=False、TF32关闭、SDPA math、原eval/dropout约定；AdamW 1e-5/(.9,.95)/wd0，总clip1。没有新scheduler/EMA、时间加权、端点残差、guidance或几何后处理。生成器输入仍只含当前C/N/Gaussian/mask；GT、Gamma、UID和目标尺度不输入网络。

协议规范JSON SHA为 `c155f532569415880a13830e4c13f45be5238369c25015a077c3dc4c36c7c8bb`；协议文件字节SHA为 `fd6f43284da0ea914e8f669dc142ba613a6e362c721bef5a23ddc3dde7ab6410`，二者是不同hash定义。训练计划、Gaussian实际数组hash、辅助队列及每步输入身份在 `registration/`、`manifests/` 和完整训练CSV。

### 接线数值检查与工程修复

真实预检前向逐位一致，8微批主FM归一残差全部0；T/L合成梯度相对L2差为0.00037981/0.00669546，余弦0.9999999282/0.9999776751，最大绝对差3.0518e-5/3.3490e-4。原加权aux在BF16反向中引入系数，新载体在未加权反向后于FP32乘系数，数学相同而舍入位置不同。预先冻结的combined相对L2≤.01、cos≥.9999、aux相对L2≤.05、归一残差≤1e-8均通过，未在看到结果后放宽。隔离副本真实更新范数0.02614091，相对参数范数2.3500e-5；之后原参数和AdamW均恢复。

保留的工程故障：注册hash误含正在写入的log；CPU测试初稿的1ULP期望与重复JSON参数；总入口冻结前的账本缩进错误；原START硬链接别名不被历史严格加载路径识别（0模型调用），改用同SHA原路径恢复；最终相机冻结时Python tuple/JSON list比较不等，做JSON规范化后恢复，冻结相机文件字节SHA仍完全相同。没有改模型数学、loss、rho、数据、评分定义或相机几何。原件、回执及diff保存在 `engineering/`、`preflight/`、`failures/`。

文献仅作研究背景与边界说明，见 `LITERATURE_SCOPE.md`；未运行这些方法，不声称完整算法复现或套用理论保证。

## 研究结论及适用范围

**保留 START。此次没有得到同时支持控制收益与完整 Chair 质量的新晋级候选。** 新 L500 相对新 T500 的接缝效应没有复现；G 的完整辅助梯度预算确实改变了更新输入，但没有通过预登记的 500 步质量保护筛选。1000 步上存在覆盖距离方向的恢复证据，不能据此回写 500 步主终点，也不能概括为整体控制稳定性改善。

历史 CAPPED500 的 OLD 接缝信号仍是实际观察，不应抹掉；但同一历史权重在新噪声上没有保持相对历史 teacher 的收益，新续训流也没有复现它。保留其权重、RAW和证据供复核，将它视为受旧开发噪声及续训条件限制的未确认观察；本轮不新增一个已获支持的控制候选，不切换默认模型。

这是同一 START 基础模型、同一冻结反馈池上的一份新续训流。FRESH 是相同 8 个 DEV 父对象上的新 Gaussian，不是新父泛化；未使用封存 CONFIRM。新三臂采用统一的 FP32 梯度收集载体，历史训练则在 BF16 反向中的系数放置不同。数学定义相同但实测有小舍入差，因此不能把旧/新差异全部因果归于 seed，更不能声称做了多个独立预训练重复。

## 1. 原 CAPPED500 相对 teacher 的效应是否复现

主比较固定为 L500−T500。OLD 的 S 父均差 **−7.1538 pp**，父中位差 **−1.9661 pp**，仅 **1/8** 父改善；FRESH 分别为 **−3.8318 pp、−2.3546 pp、1/8**。预登记 A 的 OLD、FRESH 三项条件均 FAIL。不是 OLD_PANEL_ONLY，也不是只有均值而分布不足的弱正信号。

OLD 两份噪声各自均差为 −10.7713/−3.5363 pp；FRESH 为 −2.1611/−5.5025 pp。两份噪声都改善的父数在两面板均为 0/8。任意留一父后均差仍为负：OLD [−8.7544,−4.3329] pp，FRESH [−4.6994,−2.2296] pp。完整 8 父及噪声配对附后，不围绕最坏一父解释全组。

L500−START 也为负：OLD −8.9114 pp（0/8 父改善），FRESH −4.8265 pp（1/8）。另一方面，L500 相对 T500 的 GT→F2048 均值下降 12.13%/21.97%，相对 START 下降 1.94%/11.33%。这是距离方向上的真实收益，但没有同时达到研究要求的接缝收益。OLD 的 GT→F2048 仅 3/8 父改善，父中位差反而变差；FRESH 则 7/8 改善，两者不能只合并成一个均值。

## 2. 新噪声对历史信号的检验

冻结历史 CAPPED500−冻结历史 teacher500 的 OLD S 均差为 **+6.0072 pp**，中位差 +3.4719 pp，6/8 父改善。复用其原 OLD，不重新生成或挑噪声。

同两份历史权重在本次预登记 FRESH 上的 S 均差转为 **−5.7789 pp**，中位差 −1.2245 pp，仅 2/8 父改善；两份新噪声均差均为负，0/8 父在两噪声上同时改善。其 FRESH GT→F512/2048 相对历史 teacher 分别恶化 18.25%/16.61%，Dsym 恶化 14.03%；相对 START 的 S 为 −0.2431 pp，GT→F2048 恶化 7.28%，Dsym 恶化 9.28%。

所以有两类范围有限的检验证据：历史权重换新评价噪声没有维持优势；新主训练流下 L−T 也没有复现接缝收益。不能把这两条混称为不同基础模型或不同反馈池上的重复。

## 3. 梯度预算实际改变了什么

三臂均执行同样的 8 次主 FM 前反向、保存 FP32 主梯度、清空 .grad、1 次未加权辅助前反向、合成完整梯度、global clip=1、原 AdamW 更新。无第三次网络计算，无重复乘系数，无新参数、时间权重、PCGrad、KL 或优化器重置。

L 与 G 的冻结START轨迹输入状态 X_real_t、匹配 GT、队列和逐次主输入一致；起始模型、AdamW与RNG一致，训练后的模型状态会分岔。G 唯一科学变化为 rho=1 的完整参数梯度范数上限。辅助 loss cap 与梯度 cap 分开记录，None、零范数和实际 FP32 舍入均在审计范围。具体触发率、前后半程变化、真实更新快照和实现成本见后面的最终训练审计表。

预算约束的是 **裁剪前合成时的辅助梯度范数**。原 AdamW 动量、二阶矩和预条件仍作用于合成梯度；即使 G 的辅助/FM 比被压到约 1，实际参数更新也不必处处小于 L。只在登记的 step1、每25步及501记录了真实更新，不能把这些快照插值成全部1000步的更新约束。

## 4. G 是否保留接缝收益并减少覆盖损失

500 步固定 B 筛选在 OLD、FRESH **均 FAIL**。G500−L500 的 S 均差虽然为 +1.5681/+2.2413 pp，GT→F512 却恶化 **20.07%/20.41%**，GT→F2048 恶化 **21.44%/20.39%**，Dsym 恶化 **6.36%/17.25%**。GT→F2048 改善父数 OLD 为 5/8，FRESH 为 0/8；OLD 的均值损失由尾部主导，FRESH 的父均方向更普遍。

G−L 的 S、I、E容差在两面板通过，两个 GT 距离改善条件和 Dsym 条件均不通过；不能用控制容差通过代替整项 B 通过。G500−T500 的 S 仍为 **−5.5857/−1.5905 pp**，仅 1/8、2/8 父改善，没有保留相对 teacher 的接缝增益。FRESH 的 G−L S 均差为正但父中位差约 −0.0273 pp，也应保留这个分布差别。

需要实际看图的距离尾部包括：OLD 的 341b/n1，G500−L500 GT→F2048 差为 +0.00630739；FRESH 的 534b/n2 为 +0.00732154，而同父 n3 为 −0.00048856。两噪声不能互相抵消后宣称该父没有风险。GT 距离增大只能说明参考表面距离变大；是否主体缺失、薄片或结构仍合理，由后面的实际 AI 视觉记录另判，不能从一个距离数自动推断。

## 5. 后500步回退及差分变化

L 的质量距离回退在本轮出现：1000−500 的 GT→F512 在 OLD/FRESH 分别 +5.09%/+7.72%，GT→F2048 +4.94%/+9.42%，Dsym +7.27%/+20.27%。但接缝变化并不一致：OLD S −0.2661 pp，FRESH **+2.5711 pp**。E 在 OLD −3.9075 pp，在 FRESH +0.5685 pp；I 均增加约0.37–0.38 pp。不能称本轮 L 在所有控制维度都重现了同一后期回退。

G 的 GT→F2048 从500到1000下降 13.08%/24.51%，Dsym 下降 3.07%/18.22%，有距离方向上的恢复。但 S 为 −1.4714/−0.1683 pp，E −3.3030/−2.3678 pp，I +0.2167/+1.4313 pp。G 的距离变好伴随控制代价，不能只称曲线更稳。

差分变化 DID=(G1000−G500)−(L1000−L500)：S为 **−1.2052/−2.7394 pp**，I为 −0.1648/**+1.0620 pp**，E为 +0.6045/**−2.9362 pp**；GT→F2048 为 −0.00030843/−0.00100960，Dsym 为 −0.00016787/−0.00089565。距离负值有利，S/E负值不利，I负值有利，不能统一解释为“下降更好”。

G1000 相对 L1000 的 FRESH Dsym 均值下降20.28%，但只有 **1/8** 父改善；GT→F2048 均值下降16.94%，只有 **2/8** 父改善。534b 的 Dsym 父均差 −0.00509749，其中 n2 −0.01044733、n3 +0.00025234；其余7父 Dsym 父均差均为正。这个尾部恢复有研究价值，但不是分布普遍改善。G1000−T1000 的 S仍为 −6.4552/−2.6223 pp，不能用1000的距离收益改写500结论或提出控制晋级。

## 6. 控制及完整 Chair 质量代价

固定评分保持原 v1r1 的 S、I、E、查询点、阈值与资格；S↑、E↑、I↓。所有8父、每父两噪声固定分母，另提供四噪声 ALL 表。PARTIAL 保留上下界，NA未补有利值；表中E点估计不代替阈值审计使用的上下界。

距离方向同时报告 GT→F512、GT→F2048、F→GT2048、Dsym（两向算术平均）和 p95，不拿 Dsym 掩盖GT表面距离损失。平方距离项（MSE及Dsym）除以原 L0²，p95除以L0。原 coverage、CF/FF受影响free面比例、退化、小角、free面积和面数同时保留。FF数值很高是原检测定义下的受影响面比例，不是“失败输出占比”或人类不可用率。

OLD L500 的 I 比 T500 低1.7120 pp、E高7.2713 pp，但 S低7.1538 pp。暴露更多且冲突更少也可能伴随接合不足，不能仅凭 I/E 方向称贴合更好。FRESH L500 的 I低2.8198 pp、E低0.7524 pp；G500相对L的I/E均值变化很小，但GT→F距离代价约20%。完整数值、尾部和实际视觉解释分别保留。

联合晋级仍与 START 和匹配 teacher 双参考比较，保留原形状/控制门槛。两面板的历史 CAPPED500、L500、G500、L1000、G1000全部数值联合 FAIL。L500 的形状子项可以 PASS，但控制子项未通过；G1000 FRESH 的形状子项PASS也没有控制晋级依据。

原评分聚合源码将 I/E 超10pp或区间未能排除越阈值的严重父回退单列为 `severe_cost_caveat`，没有把该布尔值直接并入 `numeric_status`。本报告保留原定义和严重父明细，不把数值门槛当成足够晋级条件；实际 AI 视觉审阅及严重父风险另列。没有人类用户研究，不报告人类可用率。

## 7. 下一步判断与停止状态

正式使用继续保留 START；历史 CAPPED500 保留为未确认的 OLD 研究观察，不能继续按已复现优越候选描述。G 的执行证据说明完整范数预算确实改变了辅助梯度，但 rho=1 的这个单因素约束不足以同时保证接缝收益、参考表面覆盖与后期稳定性。本轮没有支持增加一个新的综合控制候选。

本轮 rho=1 干预未达到预登记联合目标。由于新L500本身没有先复现接缝收益，本轮不能单独识别G对“已经复现的收益”的保留能力，也不能据此断言真实轨迹纠偏在任何流、目标或反馈池上必然无效。固定任务到此停止；不扫描rho、不扩大反馈池、不新增种子、不改变loss或时间权重、不启用MOMENT、不自动commit/push。


## 最终梯度与真实更新证据

| 臂/阶段 | loss cap% | grad cap% | clip% | aux/FM均值 | aux/FM中位 | cos均值 | 真实更新均值 | 真实更新快照 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T/1-500 | 0.0 | 0.0 | 3.2 | 0.089232 | 0.078624 | +0.046702 | 0.02501406 | 21 |
| T/501-1000 | 0.0 | 0.0 | 1.0 | 0.082774 | 0.072420 | +0.070123 | 0.02402321 | 21 |
| L/1-500 | 89.6 | 0.0 | 18.6 | 0.853924 | 0.656127 | -0.060144 | 0.02964441 | 21 |
| L/501-1000 | 61.8 | 0.0 | 40.8 | 1.677667 | 1.238829 | -0.051018 | 0.03057415 | 21 |
| G/1-500 | 90.6 | 31.6 | 8.8 | 0.667353 | 0.677949 | -0.052560 | 0.02856075 | 21 |
| G/501-1000 | 64.8 | 59.4 | 8.6 | 0.844831 | 1.000000 | -0.063267 | 0.02834991 | 21 |


T在本新流实际1000步均未触发loss cap。L后半程aux/FM均值1.677667、最大10.80486，说明loss cap不等于梯度cap。G总1000步触发455次，理论预算最大残差0；实际FP32乘法/范数归约有104步正残差，最大1.1920928955e-7，实际比最大1.0000001781，不能写成浮点逐位严格≤1。

42对注册的真实参数更新中，G比L小37次、大4次、相等1次；G/L比均值0.948312、中位0.955519；前后半中位0.977050/0.936787。step501 G/L=0.02970094/0.03424206；step1000=0.02810124/0.02982294。它是特定配对快照的实测结果，不是所有步更新界，也不能由这个单一新流作普遍因果推断。

全部步额外主梯度缓冲521604620 bytes（497.441MiB），同样用于T/L/G；各臂/半程capture均值10.50–11.22ms，scalar reduction14.13–14.43ms，combine13.36–13.85ms，载体总37.99–39.35ms。原始LF、raw LA、加权LA、系数、范数、夹角、clip、输入/辅助身份、有限性与阶段时间均在3000行逐步表。

![完整训练诊断](delivery_compact/training_overview.webp)

## 全量实际视觉审阅

标识为 **AI_VISUAL_REVIEW**。288/288输出均实际查看了同父共同相机下四视角+wire和S/I/E叠图；不是生成图片后直接打勾。四个并行审阅范围各72条，逐条记录合并去重后为288。GT及输出均为C蓝/free橙，未平滑、补洞、裁异常或改变几何。共同相机含全部标签/噪声的所有顶点；OLD/FRESH各800个原mesh面板（含C与GT）及432个叠图，全部顶点入镜和几何未变核验PASS。

完整本地图集入口：[REVIEW_INDEX.html](visuals/REVIEW_INDEX.html)。逐输出观察为 `visuals/reviews/all_output_observations.csv`，每条有请求、标签、主体/重叠/薄片/合理性、确定程度及图页hash。没有人类用户研究；不报告人类可用率。图像能确认层叠、尖片和碎片化外观，不能单凭投影把所有疑似重叠断言为精确共面重复三角面。

明确坏例：OLD 341b/n1/G500相对START/T500/L500的座下承托变成尖碎片和交叉长三角；G1000恢复较完整箱式外形。FRESH 534b/n2，历史CAPPED500与G500的free支撑进一步缩成悬空碎片，L1000呈错位板与纠缠箱状承托；G1000恢复更多高背/腿，但仍有大尖片。FRESH 341b/n3的历史CAPPED500也出现相对START/历史teacher明确新增的座面/下部碎片化与背部裂解。上述判断依据实际碎片化与支撑不连贯，不以椅型、背高或腿数不同于GT为失败理由。

其他父也保留：866e的FRESH局部座面交叉片/尖片，86ec/n2/G1000新增腿间大三角薄片；bdd历史CAPPED OLD n1腿间薄片变大；f596历史CAPPED FRESH n2背部不规则孔；c1b FRESH n2各标签主体基本保持。确定的局部缺陷与是否构成重大主体失败分开写，possible不偷换成confirmed。T自身的座面尖片、悬挂片和START已有缺陷也记录在案。

| 面板 | 候选/匹配T | 视觉判断 | 实际依据 |
| --- | --- | --- | --- |
| OLD | HIST_CAPPED500/HIST_TEACHER500 | INCOMPLETE | 全16请求五视图与S/I/E已审阅。bdd n1历史CAPPED的腿间大三角悬挂片比START/历史teacher更明显，主体仍可辨；是否达到重大主体失败的程度不确定。保留possible，不把椅型差异自动判失败，也不将本比较判PASS。 |
| OLD | L500/T500 | PASS | 全16请求五视图与S/I/E已审阅。两参考比较未见明确新增重大主体丢失或崩坏；534b座面层叠/C间隙、c1b背部薄片以及其余父的裂缝/细长片仍存在。此PASS只对应新增重大主体失败检查，非几何健康或数值控制通过。 |
| OLD | G500/T500 | FAIL | 341b n1的G500在正面、侧面及两斜视下，座下承托破碎为交叉尖片；START、T500和L500有更连贯的下部主体。该确定新增重大主体失败由实际多视图确认，非从GT距离或腿数推断。其余15请求亦已审阅。 |
| OLD | L1000/T1000 | PASS | 全16请求五视图与S/I/E已审阅。未见相对START及匹配T1000明确新增重大主体崩坏；局部座面层叠、薄片与裂缝仍记录。T1000的f596 n0座上大斜片也保留作参照风险。此PASS不表示接缝或人类可用性通过。 |
| OLD | G1000/T1000 | PASS | 全16请求五视图与S/I/E已审阅。341b n1相较G500恢复较连贯箱式外形，未见相对START及T1000明确新增重大主体崩坏；c1b等父局部细长片、层叠与接口仍有缺陷。此PASS仅为新增重大主体失败检查。 |
| FRESH | HIST_CAPPED500/HIST_TEACHER500 | FAIL | 534b n2相对START及历史teacher的free支撑进一步缩为座下悬空碎片，主体连贯性明确退化；341b n3亦有相对两参考明确新增的座面/下部碎片化与背部裂解。f596 n2背部不规则大孔另记possible。全16请求已审阅，不用另一噪声抵消坏例。 |
| FRESH | L500/T500 | INCOMPLETE | 全16请求五视图与S/I/E已审阅。534b n2有高背及较长支撑但新增大尖片/分离背顶片；866e n2/n3背座扭曲交叉片、86ec n3座下大三角片明确可见。主体仍可辨，这些局部缺陷达到重大主体失败的程度尚不确定，保留possible而不判PASS。 |
| FRESH | G500/T500 | FAIL | 534b n2相对START/T500的free支撑变成座下纠缠碎三角，背部亦碎裂，确定新增重大主体失败；866e的局部座面交叉层和长薄片另列。全16请求已审阅。 |
| FRESH | L1000/T1000 | FAIL | 534b n2的背座排列退化为错位抬升板与纠缠箱状短支撑，相对START及T1000明确新增主体连贯性失败；不是仅因背高或腿数不同于GT。866e等父局部薄片亦保留，全16请求已审阅。 |
| FRESH | G1000/T1000 | INCOMPLETE | 全16请求五视图与S/I/E已审阅。534b n2较G500/L1000恢复更多高背和长支撑，但大尖片未消失；866e n2座上尖片及座洞、86ec n2新增腿间大三角薄片、n3悬挂长条明确可见。主体仍可辨，重大主体失败程度保留possible；不能把距离恢复等同健康或PASS。 |


视觉INCOMPLETE表示全量已看完但重大缺陷程度仍有不确定性，不表示漏看请求；它不会被当成PASS。视觉PASS也仅指本次比较未见明确新增重大主体缺陷，不保证无裂缝/相交或人类可用。数值联合门槛保持独立。

![500步控制与覆盖方向](delivery_compact/primary500_control_and_shape.webp)
![持续训练变化](delivery_compact/stability500_to1000.webp)
## 完整结果表

OLD 全标签绝对值

S/I/E 为 %；MSE 与 Dsym 均除以原 L0²，表内乘 1000。8 父等权，每父固定两噪声。

| 标签 | S↑ | I↓ | E↑ | GT→F512↓ | GT→F2048↓ | F→GT2048↓ | Dsym↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| START | 14.269 | 4.251 | 84.486 | 1.531 | 1.511 | 1.729 | 1.620 |
| HIST_TEACHER500 | 12.735 | 3.769 | 85.049 | 1.985 | 1.932 | 1.569 | 1.751 |
| HIST_CAPPED500 | 18.743 | 4.477 | 95.360 | 1.722 | 1.660 | 1.504 | 1.582 |
| T500 | 12.511 | 4.425 | 87.923 | 1.761 | 1.686 | 1.568 | 1.627 |
| L500 | 5.358 | 2.713 | 95.194 | 1.527 | 1.482 | 1.706 | 1.594 |
| G500 | 6.926 | 2.859 | 94.904 | 1.834 | 1.799 | 1.591 | 1.695 |
| T1000 | 11.909 | 2.673 | 89.019 | 1.540 | 1.488 | 1.719 | 1.604 |
| L1000 | 5.091 | 3.094 | 91.287 | 1.605 | 1.555 | 1.865 | 1.710 |
| G1000 | 5.454 | 3.076 | 91.601 | 1.607 | 1.564 | 1.723 | 1.643 |

FRESH 全标签绝对值

S/I/E 为 %；MSE 与 Dsym 均除以原 L0²，表内乘 1000。8 父等权，每父固定两噪声。

| 标签 | S↑ | I↓ | E↑ | GT→F512↓ | GT→F2048↓ | F→GT2048↓ | Dsym↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| START | 11.967 | 2.603 | 92.598 | 2.913 | 2.925 | 1.502 | 2.213 |
| HIST_TEACHER500 | 17.503 | 4.868 | 92.229 | 2.682 | 2.691 | 1.551 | 2.121 |
| HIST_CAPPED500 | 11.724 | 5.557 | 94.394 | 3.172 | 3.138 | 1.699 | 2.419 |
| T500 | 10.972 | 5.133 | 95.583 | 3.288 | 3.324 | 1.590 | 2.457 |
| L500 | 7.140 | 2.313 | 94.830 | 2.554 | 2.594 | 1.708 | 2.151 |
| G500 | 9.382 | 2.390 | 95.463 | 3.075 | 3.123 | 1.921 | 2.522 |
| T1000 | 11.836 | 2.576 | 96.747 | 3.082 | 3.123 | 1.664 | 2.393 |
| L1000 | 9.711 | 2.682 | 95.399 | 2.751 | 2.838 | 2.336 | 2.587 |
| G1000 | 9.213 | 3.821 | 93.096 | 2.315 | 2.357 | 1.767 | 2.062 |

接缝配对分布

差值单位为百分点；两噪声按 OLD 0/1、FRESH 2/3 排列。留一父只是描述性敏感性，原门槛仍用全部8父。

| 面板 | 比较 | 均差 | 父中位差 | 改善父 | 两噪声均差 | 两噪声均改善父 | 噪声反向父 | 留一父均差范围 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD | HIST_CAPPED500−HIST_TEACHER500 | +6.007 | +3.472 | 6/8 | +3.188 / +8.827 | 2/8 | 4/8 | +3.038 ~ +6.901 |
| OLD | HIST_CAPPED500−START | +4.474 | -0.963 | 2/8 | +9.137 / -0.190 | 2/8 | 4/8 | -1.087 ~ +6.126 |
| OLD | L500−T500 | -7.154 | -1.966 | 1/8 | -10.771 / -3.536 | 0/8 | 2/8 | -8.754 ~ -4.333 |
| OLD | L500−START | -8.911 | -4.158 | 0/8 | -11.034 / -6.789 | 0/8 | 2/8 | -10.062 ~ -4.180 |
| OLD | G500−L500 | +1.568 | +0.211 | 4/8 | +4.609 / -1.473 | 1/8 | 2/8 | +0.157 ~ +2.690 |
| OLD | G500−T500 | -5.586 | -2.510 | 1/8 | -6.162 / -5.009 | 0/8 | 2/8 | -6.545 ~ -3.430 |
| OLD | G500−START | -7.343 | -4.314 | 1/8 | -6.425 / -8.262 | 0/8 | 2/8 | -9.749 ~ -3.277 |
| FRESH | HIST_CAPPED500−HIST_TEACHER500 | -5.779 | -1.224 | 2/8 | -8.707 / -2.850 | 0/8 | 5/8 | -7.108 ~ -2.430 |
| FRESH | HIST_CAPPED500−START | -0.243 | -0.522 | 2/8 | +0.606 / -1.093 | 0/8 | 5/8 | -1.417 ~ +1.010 |
| FRESH | L500−T500 | -3.832 | -2.355 | 1/8 | -2.161 / -5.502 | 0/8 | 3/8 | -4.699 ~ -2.230 |
| FRESH | L500−START | -4.827 | -1.342 | 1/8 | -4.151 / -5.502 | 0/8 | 3/8 | -5.531 ~ -2.709 |
| FRESH | G500−L500 | +2.241 | -0.027 | 3/8 | +3.883 / +0.600 | 2/8 | 2/8 | +0.300 ~ +2.883 |
| FRESH | G500−T500 | -1.590 | -0.500 | 2/8 | +1.722 / -4.903 | 1/8 | 3/8 | -2.130 ~ -0.709 |
| FRESH | G500−START | -2.585 | -1.425 | 1/8 | -0.268 / -4.902 | 0/8 | 3/8 | -3.074 ~ -1.241 |

500步相对质量与控制代价

距离列为候选均值/参考均值−1的百分比，负值较好；I/E 为百分点。父改善数专指 GT→F2048。

| 面板 | 比较 | GT512 % | GT2048 % | F→GT % | Dsym % | p95 % | GT2048改善父 | I pp | E pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD | HIST_CAPPED500−HIST_TEACHER500 | -13.281 | -14.094 | -4.143 | -9.634 | -11.229 | 4/8 | +0.708 | +10.311 |
| OLD | HIST_CAPPED500−START | +12.448 | +9.873 | -12.973 | -2.318 | -1.539 | 3/8 | +0.226 | +10.874 |
| OLD | L500−T500 | -13.240 | -12.134 | +8.779 | -2.056 | -3.838 | 3/8 | -1.712 | +7.271 |
| OLD | L500−START | -0.239 | -1.937 | -1.304 | -1.599 | -1.903 | 3/8 | -1.538 | +10.708 |
| OLD | G500−L500 | +20.070 | +21.440 | -6.730 | +6.363 | +5.384 | 5/8 | +0.146 | -0.290 |
| OLD | G500−T500 | +4.173 | +6.704 | +1.458 | +4.176 | +1.339 | 2/8 | -1.566 | +6.981 |
| OLD | G500−START | +19.783 | +19.087 | -7.946 | +4.662 | +3.378 | 3/8 | -1.392 | +10.418 |
| FRESH | HIST_CAPPED500−HIST_TEACHER500 | +18.253 | +16.610 | +9.558 | +14.032 | +6.498 | 5/8 | +0.690 | +2.165 |
| FRESH | HIST_CAPPED500−START | +8.892 | +7.276 | +13.182 | +9.279 | -1.803 | 5/8 | +2.954 | +1.796 |
| FRESH | L500−T500 | -22.321 | -21.968 | +7.432 | -12.455 | -15.238 | 7/8 | -2.820 | -0.752 |
| FRESH | L500−START | -12.321 | -11.326 | +13.767 | -2.814 | -16.504 | 6/8 | -0.290 | +2.232 |
| FRESH | G500−L500 | +20.411 | +20.389 | +12.472 | +17.245 | +11.206 | 0/8 | +0.076 | +0.633 |
| FRESH | G500−T500 | -6.466 | -6.059 | +20.831 | +2.642 | -5.739 | 6/8 | -2.744 | -0.119 |
| FRESH | G500−START | +5.575 | +6.753 | +27.957 | +13.945 | -7.148 | 5/8 | -0.214 | +2.865 |

持续训练变化及差分变化

前三列为百分点，后三列为归一化 MSE 差×1000。S/E 正值较好；I/MSE/Dsym 负值较好。DID不能代替先判断L是否实际回退。

| 面板 | 比较 | S pp | I pp | E pp | GT512×1000 | GT2048×1000 | Dsym×1000 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OLD | T1000−T500 | -0.602 | -1.752 | +1.096 | -0.220 | -0.198 | -0.024 |
| OLD | L1000−L500 | -0.266 | +0.381 | -3.908 | +0.078 | +0.073 | +0.116 |
| OLD | G1000−G500 | -1.471 | +0.217 | -3.303 | -0.227 | -0.235 | -0.052 |
| OLD | G变化−L变化 | -1.205 | -0.165 | +0.605 | -0.304 | -0.308 | -0.168 |
| FRESH | T1000−T500 | +0.864 | -2.557 | +1.164 | -0.205 | -0.201 | -0.064 |
| FRESH | L1000−L500 | +2.571 | +0.369 | +0.568 | +0.197 | +0.244 | +0.436 |
| FRESH | G1000−G500 | -0.168 | +1.431 | -2.368 | -0.760 | -0.765 | -0.460 |
| FRESH | G变化−L变化 | -2.739 | +1.062 | -2.936 | -0.958 | -1.010 | -0.896 |

交叉、退化、面积与面数

p95为距离/L0；各比例为%；面积为原模型坐标平方的父均值，面积比是逐例比值的等父均值。CF/FF沿用原检测定义。平均free面数不代表每例相同，逐例N/K见原表。

| 面板 | 标签 | p95/L0 | coverage | CF受影响 | FF受影响 | 近退化 | 零面积 | 角<5° | free面积 | free/GT面积 | free面数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD | START | 0.069 | 32.300 | 6.013 | 92.627 | 0.000 | 0.000 | 14.676 | 51.441 | 1.082 | 149.625 |
| OLD | HIST_TEACHER500 | 0.076 | 32.690 | 6.193 | 91.541 | 0.000 | 0.000 | 10.902 | 49.229 | 1.041 | 149.625 |
| OLD | HIST_CAPPED500 | 0.068 | 35.071 | 6.703 | 89.406 | 0.000 | 0.000 | 12.067 | 43.067 | 0.979 | 149.625 |
| OLD | T500 | 0.070 | 30.664 | 5.497 | 92.858 | 0.000 | 0.000 | 12.802 | 49.650 | 1.049 | 149.625 |
| OLD | L500 | 0.068 | 28.259 | 6.320 | 88.631 | 0.000 | 0.000 | 9.725 | 47.426 | 1.063 | 149.625 |
| OLD | G500 | 0.071 | 30.310 | 6.480 | 88.680 | 0.000 | 0.000 | 9.150 | 41.952 | 1.017 | 149.625 |
| OLD | T1000 | 0.065 | 34.192 | 6.387 | 92.617 | 0.000 | 0.000 | 10.426 | 49.501 | 1.064 | 149.625 |
| OLD | L1000 | 0.070 | 30.811 | 6.809 | 87.796 | 0.000 | 0.000 | 9.105 | 53.635 | 1.144 | 149.625 |
| OLD | G1000 | 0.072 | 31.067 | 6.495 | 88.851 | 0.000 | 0.000 | 9.879 | 51.790 | 1.102 | 149.625 |
| FRESH | START | 0.097 | 28.638 | 6.753 | 90.986 | 0.000 | 0.000 | 10.911 | 42.809 | 0.919 | 149.625 |
| FRESH | HIST_TEACHER500 | 0.090 | 32.886 | 8.419 | 88.864 | 0.000 | 0.000 | 9.120 | 44.899 | 0.971 | 149.625 |
| FRESH | HIST_CAPPED500 | 0.096 | 32.068 | 6.604 | 88.205 | 0.000 | 0.000 | 6.496 | 39.827 | 0.953 | 149.625 |
| FRESH | T500 | 0.096 | 28.247 | 7.197 | 90.540 | 0.000 | 0.000 | 9.516 | 37.648 | 0.896 | 149.625 |
| FRESH | L500 | 0.081 | 29.004 | 5.692 | 87.877 | 0.000 | 0.000 | 5.698 | 42.855 | 1.013 | 149.625 |
| FRESH | G500 | 0.091 | 27.087 | 5.619 | 87.799 | 0.000 | 0.000 | 6.602 | 39.709 | 0.940 | 149.625 |
| FRESH | T1000 | 0.093 | 29.700 | 6.433 | 89.752 | 0.000 | 0.000 | 8.538 | 39.000 | 0.901 | 149.625 |
| FRESH | L1000 | 0.083 | 31.042 | 6.728 | 88.009 | 0.000 | 0.000 | 7.408 | 46.048 | 1.056 | 149.625 |
| FRESH | G1000 | 0.076 | 29.932 | 6.811 | 87.935 | 0.000 | 0.000 | 6.752 | 44.649 | 1.033 | 149.625 |

完整8父主终点配对

S写为父均差 [两个噪声各自差]，单位百分点。GT写为该父两噪声均值的相对变化%。不挑最佳噪声。

| 父UID | OLD L−T S | OLD L/T GT% | OLD G−L S | OLD G/L GT% | FRESH L−T S | FRESH L/T GT% | FRESH G−L S | FRESH G/L GT% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 33ec1e64a02ff718faaadcdc0221eb72 | +0.000 [+0.000/+0.000] | +26.169 | +1.131 [+2.262/+0.000] | -18.072 | +2.242 [+0.000/+4.484] | -50.482 | -0.055 [+2.013/-2.122] | +38.708 |
| 341bcae1e7f366a233d93f57729688db | -0.858 [+0.000/-1.715] | -6.291 | +0.000 [+0.000/+0.000] | +123.551 | +0.000 [+0.000/+0.000] | -6.508 | +0.000 [+0.000/+0.000] | +0.523 |
| 534b0e6e2c53a8e1a7cdfd3874ed3723 | -2.529 [-5.058/+0.000] | +77.119 | -0.256 [-0.512/+0.000] | -3.086 | -0.208 [+0.702/-1.117] | -49.976 | -0.482 [-0.965/+0.000] | +120.485 |
| 866e7248434968051cc7750d0afca796 | -16.933 [-30.079/-3.787] | +95.457 | +11.446 [+22.892/+0.000] | -31.062 | -1.525 [+1.175/-4.226] | -46.500 | -2.250 [-4.500/+0.000] | +22.402 |
| 86ec9c796a6c1789e792598ed58dd1cc | -26.900 [-47.401/-6.400] | +28.881 | +6.221 [+9.879/+2.563] | +1.116 | -6.751 [+1.636/-15.139] | +16.903 | -1.006 [+2.091/-4.104] | +2.424 |
| bdd29e651e5f6fb2b079317292bdc5d4 | -1.403 [+11.729/-14.535] | +4.310 | -0.133 [-0.801/+0.535] | +3.326 | -3.184 [-6.239/-0.129] | -3.906 | +2.874 [+5.490/+0.258] | +1.276 |
| c1b8c2f0c4f3cc43aff7c08b060f5ed6 | -12.658 [-23.596/-1.719] | -37.557 | +0.422 [+0.844/+0.000] | -2.381 | -6.181 [-8.624/-3.737] | -4.093 | +3.019 [+6.037/+0.000] | +1.410 |
| f59627a077dcc2566c63b8b48495261a | +4.051 [+8.236/-0.134] | -62.688 | -6.286 [+2.310/-14.882] | -6.836 | -15.047 [-5.938/-24.155] | -18.759 | +15.831 [+20.898/+10.764] | +6.030 |

原联合门槛

匹配对照与START双参考均保留；严重父代价按原实现单独flag，必须阅读逐父表和实际视觉审阅。数值条件、视觉判断与晋级建议分别解释。

| 面板 | 候选 | 匹配对照 | 数值 | 形状 | 控制A | 控制B | 严重父标记 | 含视觉状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLD | HIST_CAPPED500 | HIST_TEACHER500 | FAIL | FAIL | PASS | PASS | True | FAIL |
| OLD | L500 | T500 | FAIL | PASS | FAIL | FAIL | False | FAIL |
| OLD | G500 | T500 | FAIL | FAIL | FAIL | FAIL | False | FAIL |
| OLD | L1000 | T1000 | FAIL | FAIL | FAIL | FAIL | False | FAIL |
| OLD | G1000 | T1000 | FAIL | FAIL | FAIL | FAIL | False | FAIL |
| FRESH | HIST_CAPPED500 | HIST_TEACHER500 | FAIL | FAIL | FAIL | FAIL | True | FAIL |
| FRESH | L500 | T500 | FAIL | PASS | FAIL | FAIL | True | FAIL |
| FRESH | G500 | T500 | FAIL | FAIL | FAIL | FAIL | True | FAIL |
| FRESH | L1000 | T1000 | FAIL | FAIL | FAIL | FAIL | False | FAIL |
| FRESH | G1000 | T1000 | FAIL | PASS | FAIL | FAIL | True | FAIL |


## 严重父控制代价

# Severe parent I/E costs (existing-result extraction)

**7 original joint-gate parent/comparison rows carry a severe-cost warning; these must be reported explicitly even if a numeric gate were to pass.**

All deltas below are candidate minus reference, in percentage points (pp). Positive I = worse conflict; negative E = worse exposed retention. Original severe thresholds are I increase >10 pp or E decrease >10 pp. Bounds are the original saved PARTIAL-aware bounds. CONFIRMED means the whole bound exceeds the adverse threshold; possible/unresolved flags remain separate.

OLD and FRESH each average the two registered noises within each parent (eight parents per panel). ALL, when shown as a supplementary CSV result, averages four registered noises within the same eight parents; it is not another independent parent panel.

The original `severe_cost_caveat` is a separate warning, not an extra undocumented numeric veto. Neither original `numeric_status` nor overall `status` incorporates it. Therefore numeric PASS alone does not establish a cost-free or acceptable promotion. Original definitions and statuses are unchanged.

|Panel|Candidate|Reference|Parent UID|I delta [lower, upper] pp|I gate|E delta [lower, upper] pp|E gate|Risk|
|---|---|---|---|---:|---|---:|---|---|
|FRESH|G1000|START|534b0e6e2c53a8e1a7cdfd3874ed3723|+3.8818 [+3.8818, +3.8818]|PASS|-13.9909 [-13.9909, -13.9909]|FAIL|CONFIRMED|
|FRESH|G1000|T1000|341bcae1e7f366a233d93f57729688db|+6.9092 [+6.9092, +6.9092]|PASS|-10.9478 [-10.9478, -10.9478]|FAIL|CONFIRMED|
|FRESH|G1000|T1000|534b0e6e2c53a8e1a7cdfd3874ed3723|+3.8086 [+3.8086, +3.8086]|PASS|-21.3135 [-21.3135, -21.3135]|FAIL|CONFIRMED|
|FRESH|G500|T500|534b0e6e2c53a8e1a7cdfd3874ed3723|+1.0498 [+1.0498, +1.0498]|PASS|-12.8334 [-12.8334, -12.8334]|FAIL|CONFIRMED|
|FRESH|HIST_CAPPED500|START|f59627a077dcc2566c63b8b48495261a|+20.6787 [+20.6787, +20.6787]|FAIL|+7.9390 [+7.9390, +7.9390]|PASS|CONFIRMED|
|FRESH|L500|T500|534b0e6e2c53a8e1a7cdfd3874ed3723|+1.5137 [+1.5137, +1.5137]|PASS|-10.3674 [-10.3674, -10.3674]|FAIL|CONFIRMED|
|OLD|HIST_CAPPED500|HIST_TEACHER500|534b0e6e2c53a8e1a7cdfd3874ed3723|+13.9160 [+13.9160, +13.9160]|FAIL|+9.8138 [+9.8138, +9.8138]|PASS|CONFIRMED|

The accompanying [CSV](severe_parent_costs.csv) also retains 3 supplementary flagged rows already present in the frozen paired CSV (ALL and control-versus-START comparisons). Each row identifies whether its source is the original joint gate or the paired CSV. No new scores or point samples were generated.

Original joint-gate parent I/E values and bounds were cross-checked against the paired CSV: 960 scalar comparisons passed (absolute difference <1e-12).

Resource evidence:

- GPU: NVIDIA GeForce RTX 5080, 16303 MiB total. Maximum monitored device-wide usage: 6320 MiB. Own-process peak: NA under WSL/WDDM; device usage includes unrelated desktop processes.
- CPU: 24 logical cores; registered two threads per worker. Initial Windows CPU load [11]%; free RAM 68.326 GiB. WSL total 62.515 GiB; available RAM across recorded dispatch snapshots 58.018 to 61.138 GiB. These are snapshots, not continuous CPU/RAM peak monitoring.
- GPU lock: 6501.261100 seconds (108.354 minutes), 17 closed intervals, one worker; limit 14400 seconds. Includes the preserved 0-call loader failure, loading, saving and in-worker CPU waiting.
- Stage-specific observed peaks and lock duration are in [resource_costs.csv](resource_costs.csv). Sampled peaks are not continuous hardware maxima; no process/allocator equivalence is claimed.

|Stage|Device peak MiB|GPU utilization max %|Min free MiB|Lock seconds|
|---|---:|---:|---:|---:|
|preflight_real_1000_BOTH|6119|17|9861|30.189|
|sample_G1000_1000_BOTH|3356|24|12624|107.794|
|sample_G500_1000_BOTH|3358|26|12622|95.299|
|sample_HIST_CAPPED500_1000_FRESH|3362|26|12617|54.828|
|sample_HIST_TEACHER500_1000_FRESH|3349|26|12630|53.115|
|sample_L1000_1000_BOTH|3344|26|12636|109.933|
|sample_L500_1000_BOTH|3357|23|12623|104.897|
|sample_START_1000_FRESH|3355|23|12625|60.977|
|sample_T1000_1000_BOTH|3342|26|12638|102.960|
|sample_T500_1000_BOTH|3366|23|12614|103.838|
|train_G_1000_BOTH|6203|34|9777|944.057|
|train_G_500_BOTH|6215|32|9765|936.356|
|train_L_1000_BOTH|6200|39|9780|933.357|
|train_L_500_BOTH|6211|30|9769|949.247|
|train_T_1000_BOTH|6200|30|9780|946.665|
|train_T_500_BOTH|6320|35|9660|967.750|

Extraction source and all input SHA256 bindings are retained under `audit/severe_parent_costs_receipt.json`. The extraction imports no scoring, model, mesh, or training module.


## 检查点与导出合同

每个500/1000权重保存参数、完整AdamW、RNG、主流游标、辅助next step、recipe和protocol hash；500=conditional_total5500，1000=6000。每100步恢复保存不增加评价终点；最后latest恢复状态仍保留，不据其挑最好分数。以下为源文件真实SHA；详细模型/优化器/RNG/游标hash见 `audit/checkpoint_manifest.csv`。

| 标签 | 条件总步 | bytes | SHA256 |
| --- | --- | --- | --- |
| HIST_CAPPED500 | 5500 | 1565084587 | 058168b3126d26707ac57470613ac4e914ccc9fc2a34d119816cc1327b358c35 |
| HIST_TEACHER500 | 5500 | 1565071035 | 935ae8caf37c82ff6fa6080ffdc693f30997caa335bef2f49a455eeefbadd596 |
| START | 5000 | 1565074939 | d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80 |
| G1000 | 6000 | 1565061051 | 8c0fbb64d342cd97a9805bd1db3f6856735d4713d91c492c8d81bad691a9a590 |
| G500 | 5500 | 1565060267 | ee0d1262bc94f4eee2a23e1abb14be987febcb0f689d0de35922c0e25a2726a9 |
| L1000 | 6000 | 1565061051 | 493bc1fb1730f7c6962c6a79eb7864926bc63d23f4bc677b29a04a8499bdd83b |
| L500 | 5500 | 1565060267 | 916a4baffcbeb0b8f945de7bc6e368cc67dc5c810bccdf4bea506f27a882653b |
| T1000 | 6000 | 1565061051 | 9e495970e17f47459e1199b7723dc70e4c712c1383cf4e76e5df4e1ecd481535 |
| T500 | 5500 | 1565060267 | 0f58421e41987f29883b0d3db5b332769f783577c5c8772f775de3414e9109f7 |


所有输出N、K按输入固定，N范围129–246、K26–49，平均free面数149.625；288例的51个模型坐标状态中C逐位保持。最终NPZ以及导出PLY重读核对C坐标、定向面多重集（包括重复面）、映射和绕序，全部PASS。模型坐标FP32路径合同与world导出坐标合同分开核验，不声称任意文件格式逐字节相同。逐输出回执在 `audit/runtime_final_outputs.csv`。

## 文件导航与复核

| 材料 | 路径/用途 |
| --- | --- |
| 冻结定义 | protocol.json、hypotheses.json、recipes/、task_contract.txt |
| 输入及身份 | registration/、manifests/；训练/辅助/OLD/FRESH Gaussian hash |
| 逐步梯度与实际更新 | training_diagnostics/final/per_step_training.csv、actual_parameter_updates.csv、G_vs_L_actual_updates.csv |
| 严格日志核验 | training_audit/final/、audit/runtime_final.json |
| 全部288原逐输出评分 | evaluation/{label}/{OLD,FRESH}/per_output_metrics.csv，18份各16行 |
| 配对/逐父/噪声/留一父 | analysis/per_output.csv、per_parent.csv、per_noise.csv、leave_one_parent_out.csv、summary.csv；ALL四噪声也保留 |
| 持续训练及DID | analysis/drift_difference_in_differences_{per_output,per_parent,per_noise,summary}.csv |
| 原双参考晋级裁判 | analysis/original_joint_gates/；原定义与严重父flag不改 |
| 可复核数值图 | report_figures/：500效应、持续训练图，配套displayed_values.csv及hash |
| 全量图与AI观察 | visuals/REVIEW_INDEX.html、visuals/reviews/；完整图本地保留 |
| 实际源码与差异 | review_sources/、SOURCE_HASHES.json、code.diff；含原评分依赖及新梯度实现 |
| 最小入口与恢复 | REPRODUCE.md；run/worker/evaluation/analysis/audit模块 |
| 紧凑审阅包 | delivery/CAPPED500_REPLICATION_GRADIENT_BUDGET_REVIEW.zip；REVIEW_INVENTORY.json逐文件hash和CSV行数 |


审阅ZIP保留全逐例/逐父/逐噪声CSV、逐步梯度和真实更新、实际评分源码、协议、审计与数值图及接触图（无损WebP，解码RGBA像素逐字节相同）；为保持紧凑不装RAW、大权重或全分辨率图集。完整RAW、权重、原PNG图集、逐mesh绘图回执和逐调用原始账本均保留本地；ZIP的delivery_compact/保存全图集manifest摘要及原文件hash，不含重复QA裁片；ZIP不是脱离登记资产就可独立重现模型的环境。未清理受保护资产，未切换默认模型，未自动Git提交或推送。
