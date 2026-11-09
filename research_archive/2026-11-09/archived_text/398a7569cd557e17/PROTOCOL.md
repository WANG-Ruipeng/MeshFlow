# R2_BENCH_GROUP_WEIGHT_MATCH_V1：固定实验合同

用户已批准一条W_MATCHED1500快速训练及完整固定评价。本轮只检验：在W相同宽数据池/数据流下，恢复旧窄B逐更新组MSE系数，能否减少W的旧能力和质量退化，同时保住新增关系。B和W为只读已完成参照，不再训练或生成。

## 1. 起点、数据和唯一变化

共同P0为 `r2_bench_exposure_allocation_v1/training/B_75_25/checkpoint_step1500.pt`，global21000/native9000，恢复live生成器、匹配AdamW/RNG；不从B22500或W22500继续。原W30父、30完整USER条件、12000实例、序列、mask、源面身份、N/K、增强、OT/noise/time随机身份逐条复用；没有新增或删改来源数据、没有新抽噪声。

原模型Native/T1＋role＋Geo，free-only FM、先OT再HYBRID0.25，batch8/micro1、lr1e-5、AdamW(.9,.95)、wd0、clip1；C硬保持、N128..799（实际最高776）、精度和全部采样/后端规则保持不变。无SEAM、新头、新辅助几何loss、shift、LOD或课程。

含HIGH的更新s：mO是本步共同OLD有效自由标量数，mHB为窄B该步HIGH自由标量数，mHW为W对应数；rho=mHB/mHW，DB=mO+mHB，DW=mO+mHW。目标为：

`L = (SSE_OLD + rho*SSE_HIGH) / DB`。

实现沿用原微批FM helper的DW分母，再乘 `DW/DB`（OLD）或 `DW/DB*rho`（HIGH）。八次微批累加后直接一步优化，不额外除8。FOCUS更新原公式不变。全部系数仅依赖冻结mask/计数，不读取DEV或当前模型误差调参。

这恢复B逐更新组内平均坐标MSE的组系数，不恢复B的样本、组内父比例、实际loss份额、梯度或AdamW后续轨迹。HIGH仍6父各250实例，未恢复窄B各750次曝光。记录原unweighted_FM与实际objective，不能把两种loss混成同一学习曲线。

最近的直接对照为Bench自然N与HIGH宽池实验，前者HIGH逐更新坐标权重19.852%，后者24.307%；本轮首次在相同W序列下控制该组权重差。区别于旧曝光比例、R14辅助梯度限制和R20条件比例课程：本轮不改抽样、不加辅助项，只按明确代数对照归一化。旧机制负结果不被改写。

## 2. 固定预算与工程门槛

唯一W_MATCHED新增1500更新，终点global22500/native10500，12000前向/12000反向，0头/0GPU探针。固定1500终点，不挑best、不延长、不扫系数。每250步滚动完整恢复点，最终推理及匹配恢复状态保留。

开训前完成CPU代数和小张量autograd核验：逐更新引用计数/数据流身份、8微批loss与梯度等价、known/padding零梯度、等计数/FOCUS退回原公式、实际backward目标与账本一致。复用已完成W的N776/runtime/OT资格，不重复模型探针或CPU OT。先冻结源码hash、来源权重/AdamW/RNG、数据/评测/系数和预算，再由root测资源后放行。

当前可比W曾用训练allocated/reserved峰4.03/4.37GB，整卡含背景与短并发峰8863MiB；WDDM进程峰NA，不能当独占进程估算。本轮只有一条训练任务，GPU序列依赖决定先训练后新采样；CPU准备/审计可并行且限线程。每次GPU派发记录实际nvidia-smi/进程、CPU/RAM/负载，保留max(2GiB,整卡10%)初始额外余量。采样/D0按各自实测，不依据权重文件尺寸或训练allocator单独决定并发。

保持CUBLAS_WORKSPACE_CONFIG=:4096:8、strict deterministic warn_only=False、禁TF32、SDPA math、BF16前向和FP32参数/积分。失败/OOM停止新增派发，保留attempt与真实调用，不静默回滚、重抽或更改配置；本轮不含额外科学重放额度。CPU路径/报告工程问题可在零新增科学调用条件下修复并留痕。

## 3. 评价输入、指标与后端

完全复用W的46固定请求：OLD16、FOCUS4、DEV8、df6 HOLDOUT2、原HIGH_TRAIN4、HIGH_HOLDOUT4、NEW_WIDE_TRAIN8。相同N/K、C、噪声数组、查询点、尺度、相机和截面位置，不按结果更换。

仅新M46 RAW（2300 Euler50前向）及46次单次官方Bench D0；全部M RAW完成保存后方可D0。与原同GEO_C/来源定向及HINGE_OPP015→ALPHA→HOLE。B/W共92既有生成及276原评分行按hash只读复用；M三阶段138新评分行，总414行/原177字段。RAW、BASE_POST、CLEAN_FINAL_POST分开保存。

主比较M−W，B为窄池参照。分组分别报父等权与逐请求配对：采用/主体/主要支撑、CF/FF/I、合法接口、碎片、小角、非流形、删槽及三角面积变化。保留完整177字段，不只汇总有利项。I仅原PASS支持域；LOW_SUPPORT仍NA；617与77aa闭合C开放接口NA，不用空边界判成功。三角面积和不是唯一GT表面覆盖率，来源几何不作开放生成款式的唯一答案。

## 4. 图审与决定方式

主图审同页M/W，138页两视角＋两固定位置截面、276个实际AI_PROXY观察；B不重做全面图审，历史标签只读，不跨版本拼成功率。五轴仍分开，V2在采样前用少量既有锚例冻结：支撑仅剩细线/开角窄残片不算“主要支撑完整”；C可见不自动算采用；轮廓小缝与整体支撑损失分开；不确定就记UNCERTAIN。该澄清另立版本，不覆盖旧记录，不声称独立人类评价。三轴交集仅为描述性辅助计数，不能替代无交/完整性证据。

冻结观察重点但不替代全组：945/n1收益是否保住；abf/n1、6b/n1、bcb/n1是否恢复；新增四父的支撑是否到FINAL仍保留；DEV/原HIGH/OLD代价是否在多个父改善。不得只按总FM下降或FINAL FF变低晋级。

若旧能力/质量回退减轻并保住新增关系，支持组权重漂移是因素之一；若旧组恢复但新增学习变弱，记录预算取舍；若基本无变化/更差，降低这一解释优先级而不继续扫系数；若只降FF且结构继续损失，不晋级。即使正结果，也不把本小面板解释为独立泛化定论。

## 5. 交付和停止

独立目录保存真实权重/完整恢复状态、原数据和流的引用hash、逐步系数与loss账本、逐输出/逐父配对、完整RAW/POST截面图与实际评阅、报告、实际源码/hash/diff和可恢复总入口。网页小包尽量小于10MB，原大资产留本地。完成上述固定任务即停止，不自动清理、换默认、commit或push。
