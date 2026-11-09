# Bench真实轨迹后半程交叉：实际实验报告

实验：`R2_BENCH_TRAJECTORY_CROSSOVER_V1`；日期：2026-10-22；状态：完整执行、固定预算结束。

**本轮给“USER前半程形成更有用的主体布局”增加了直接干预证据。** 在当前四条熟悉C、各两份噪声上，用USER前半程接COMMON后半程，保守图审仍有5/8明确采用；反过来只接USER后半程仍为1/8，与COMMON全程相同。完整USER为6/8。支撑和组合两条C的四个改善请求都保留在USER前缀组合中。

**这不是一个已经解决Bench泛化的新生成方案。** 四个DEV父、各两噪声的明确采用仍全部为0/8。后半程依然影响局部几何、冲突和少数采用判断，不能写成“晚期已解决”或“COMMON后半程全面优于USER”。

## 1. 为什么执行这项实验

附件对[上一轮诊断](../r2_bench_trajectory_diagnostic_v1/REPORT_FINAL.md)的纠偏成立：从含来源Y的训练插值状态继续积分，只能证明这些状态附近的恢复能力，不能将真实生成失败定位到前半程。

这次实际交换**完整模型**的后半程。前缀来自已保存的真实噪声生成轨迹，没有混用来源布局，没有注入Y，没有重新做OT、投影或重置噪声。最接近的历史路线是上述轨迹诊断；本轮也没有重开共享副本loss、轨迹奖励、CREF或Bench seam提前完成训练。

## 2. 冻结合同与调用账本

C代表COMMON12000；U代表USER_V2_S1500。表中的C字母是模型简称，蓝色输入条件仍称条件C。

| 组合 | t=0至0.5 | t=0.5至1 | 本轮生成器前向 |
|---|---|---|---:|
| CC | COMMON | COMMON | 0，原16份RAW复用 |
| CU | COMMON | USER | 400 |
| UC | USER | COMMON | 400 |
| UU | USER | USER | 0，原16份RAW复用 |

每种新增组合16请求×25步。**实际STARTED=RETURNED=800，GPU重放0，训练更新0，D0前向0，占用头前向0，新噪声0，新OT0。** 两组各保存16份RAW、26个后半程状态和25份速度。原CC/UU结果与评分只读复用。

起点严格为原轨迹`states[25]`；绝对时间`.50,.52,…,.98`，Euler dt=.02。原生产layout、角点锚定、mask、N和噪声身份保持。BF16前向、FP32参数与积分、严格确定性、TF32关闭、math SDPA及CUBLAS配置与原实验一致。新模型加载后没有运行额外校准前向。

CPU核验了原32条轨迹全部100步/请求的两模型Euler递推、来源C导出的逐数组往返，以及32条新轨迹的25步递推。所有新状态中的固定角点精确保留，终点与RAW对应；推理前后Torch CPU/CUDA随机状态相同。来源文件只读取C顶点身份及角序元数据，没有读取来源free坐标作为生成输入。

COMMON权重：`experiments/r2_bench_adaptation_strength_v1/training/A_COMMON_LR1E5/checkpoint_step0.pt`。

USER权重：`experiments/r2_bench_user_v2_training_v1/training/USER_V2_SEAM/generator_step1500.pt`。

路径、SHA256、实际起点和全部请求见[registration.json](registration.json)；实际调用见[call_ledger.csv](analysis/call_ledger.csv)，逐轨迹核验见[integrity.csv](analysis/integrity.csv)。没有更换检查点、切分点或请求。

## 3. 完整面板与真实图审

面板保持八个父各一条C、每条两噪声：四个CURRENT_TRAIN父分别为座面35948e、靠背943dde、支撑bf504b、组合838c48；四个DEV父为5dda4e、9d5608、2d1e5e、f495bd。DEV 9d5608的旧文件名包含SEAT，但本面板登记部位是BACK_SURFACE，未更改输入。

实际打开并评阅了**全部16张四组合页面，共64个RAW输出**。每页含两个相同角度整体图及原来固定位置的正面、侧面截面。没有按结果改相机、选噪声或替换父。蓝色为输入C，橙色为SEAM，灰色为bulk。

评分为非盲`AI_PROXY`，不是人类评分。原CC/UU的32条RAW判断经当前图像复核后保留；新增CU/UC单独记录。YES仅表示宏观采用，不认证水密、精确共边、无相交或符合唯一来源款式。PARTIAL、UNCERTAIN不按YES统计。

| 父/部位 | 噪声 | CC | CU | UC | UU |
|---|---:|---|---|---|---|
| TRAIN 35948e 座面 | 0 / 1 | NO / YES | NO / YES | NO / YES | NO / YES |
| TRAIN 943dde 靠背 | 0 / 1 | PARTIAL / PARTIAL | PARTIAL / PARTIAL | PARTIAL* / PARTIAL | YES / PARTIAL |
| TRAIN bf504b 支撑 | 0 / 1 | NO / NO | NO / NO | YES / YES | YES / YES |
| TRAIN 838c48 组合 | 0 / 1 | PARTIAL / PARTIAL | PARTIAL / PARTIAL | YES / YES | YES / YES |
| DEV 5dda4e 座面 | 0 / 1 | NO / NO | NO / NO | NO / NO | NO / NO |
| DEV 9d5608 薄背上沿 | 0 / 1 | UNCERTAIN / PARTIAL | UNCERTAIN / PARTIAL | PARTIAL / PARTIAL | PARTIAL / PARTIAL |
| DEV 2d1e5e 支撑 | 0 / 1 | NO / NO | NO / NO | NO / NO | NO / NO |
| DEV f495bd 组合 | 0 / 1 | PARTIAL / PARTIAL | PARTIAL / PARTIAL | PARTIAL / PARTIAL | PARTIAL / PARTIAL |

*UC的943dde/n0是PARTIAL/YES边界：它保留了USER前缀的条形靠背和端架组织，但灰上沿及局部背层仍与蓝C错开，因此保守记PARTIAL。若把此例计YES，UC为6/8；“USER前缀保留主要收益、COMMON前缀仅换后半段不够”的判断不变。不可把5/8与6/8的一例主观差异当成统计显著结论。薄片没有输入部件名称，943dde/n1的侧裙式完成也不能仅因不符合靠背标签就判为完全无效。

四种组合的主体总体都可辨认，仍有局部薄片、重层、间隔或缺口；没有发现交叉引发整组主体崩溃。截面端点本身不证明拓扑裂缝；精确连接以原评分器为准。

[完整图集](gallery/index.html) · [逐输出图审](analysis/visual_review.csv) · [逐请求采用配对](analysis/adoption_paired.csv)。

## 4. RAW完整质量代价

下表均为逐请求宏均值。CF/FF为**至少受一对非法面对影响的free面比例**，不是面对数除以全部可能面对；I沿用原`ConflictCov_C`及固定2048个C查询。小角指free面最小内角小于5°。各组同请求N/K相同，没有删面或后处理，分母没有通过删面缩小。

| 子集/组合 | 明确采用 | CF % ↓ | FF % ↓ | I全部 % ↓ | 小角 % ↓ | 非法FF对/请求 ↓ | free面积/请求 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN CC | 1/8 | 2.581 | 91.548 | 19.403 | 35.165 | 470.125 | 42.949 |
| TRAIN CU | 1/8 | 3.185 | 91.073 | 11.346 | 32.535 | 475.875 | 39.964 |
| TRAIN UC | 5/8* | 4.487 | 88.252 | 4.352 | 26.190 | 397.250 | 45.368 |
| TRAIN UU | 6/8 | 7.445 | 89.245 | 8.832 | 28.607 | 356.875 | 41.642 |
| DEV CC | 0/8 | 5.933 | 95.305 | 17.554 | 38.115 | 565.000 | 80.632 |
| DEV CU | 0/8 | 8.017 | 94.594 | 14.661 | 38.156 | 559.875 | 82.123 |
| DEV UC | 0/8 | 5.439 | 95.222 | 15.076 | 40.429 | 443.125 | 66.646 |
| DEV UU | 0/8 | 7.061 | 94.829 | 17.120 | 40.560 | 455.375 | 68.430 |

**这里全部是RAW。** 不能把约88%—95%的RAW FF与上一轮单次D0/GEO后的约20%—36% FF混比。本轮未执行后处理，因此没有“交叉后POST质量/接续改善”的结果。

每个DEV组合中2d1e5e的两噪声I均为`LOW_SUPPORT`。保留原数值但不能当可靠零冲突成功；仅取其余六请求PASS时，CC/CU/UC/UU的I分别为**21.525/19.548/20.101/22.827%**。完整表保留每个状态、查询权重及面对分母。无必需接口、曝光要求、唯一GT时分别保存NA，不造新S/E/GT排名。

有三个不能省略的取舍：

1. UC在TRAIN的CF、FF受影响面率、I和小角宏均值都低于UU，但**非法FF面对数397.25高于UU的356.875**。其中943dde两噪声父均值为809.5对，高于UU的584.0对；该父的free面积43.485也高于UU的27.117，不能将面积大小单向解释为质量。总体“受影响面更少”不代表所有冲突账目更好。
2. CU让TRAIN的I从19.403%降到11.346%，但明确采用仍1/8；在838c48/n0还减少了跨过蓝顶面的灰片，却没有确认完整组合采用。**USER后半程有局部作用，只是没有独自改掉主要主体关系。**
3. DEV的四组都没有明确采用。特别是f495bd的父均值I为CC36.35%、CU43.55%、UC60.30%、UU68.48%，较低的总平均不能遮住这个坏尾部。5dda/2d仍有错层或包住C的宏观失败，9d的薄顶条关系仍不完整。

RAW的bulk连续坐标角点尚未焊接；图上的整体采用与共享索引接续是不同问题。完整连接、A-VMR、边界距离、分角色非法面对、退化及拓扑字段全部保留，但不拿RAW的稀少共边宣布后处理后接续已改善。

[64行完整逐输出表](analysis/per_output_all_metrics.csv)含205个字段（原字段加组合、出处与审计元数据）；[所有指标汇总](analysis/all_metric_summary.csv)含176个数值字段；[逐父两噪声配对](analysis/per_parent_paired.csv)；[逐请求因子差分](analysis/per_request_factorial.csv)保留CU−CC、UU−UC、UC−CC、UU−CU及交互项。差分仅作描述，不把两个噪声当两个独立父做显著性检验。

## 5. 轨迹数值是否也跟随前缀

图像之外，补充计算了同噪声、同槽位、非固定角点的端点向量RMS距离，归一化为原L。**这是事后描述性的轨迹差异量，不是新增主质量指标、形状距离或GT误差；槽位差异可包含三角划分变化。**

| 子集 | CU与CC | UC与UU | CC与UU |
|---|---:|---:|---:|
| TRAIN | 2.545%L | 3.614%L | 14.341%L |
| DEV | 1.853%L | 2.061%L | 13.008%L |

CU更接近CC、UC更接近UU的数值趋势与所见的宏观布局一致。每个请求从t=.5至1的完整两两差异保存在[paired_trajectory_distance.csv](analysis/paired_trajectory_distance.csv)，端点摘要见[paired_endpoint_distance.csv](analysis/paired_endpoint_distance.csv)。这个测量不认证某个几何更正确，也不消除交叉状态兼容性的限制。

## 6. 本轮到底定位了什么

**支持的判断：** 在所测熟悉支撑/组合条件上，USER前半程产生的真实状态包含了可被COMMON后半程继续保留的采用收益。该收益不是必须等USER最后25步才出现。用COMMON前缀再换USER后缀，没有得到同样的宏观改善。

**没有支持的判断：** 不能认为任意模型都能接好USER前缀，不能认为后半程永远无害或不需要学习，不能用这次切分证明低t加权是正确训练方法。943dde靠背边界例及FF面对数反例已经说明局部质量仍依赖后半程。另一模型的状态可能偏离接手模型的训练分布，混合失败不能单独给一个时段定罪。

**DEV仍未定位：** 四种组合的明确采用都为0，交换只改变部分款式和代价。数据覆盖、条件几何关系学习、起点历史以及DEV生产布局等解释均未被本轮排除。这不是经典过拟合的单独证明，更不是“多加父一定有效”的证明。四条熟悉C和四条DEV C都不足以估计一般成功率。

## 7. 下一步优先问题与停止边界

下一项应进入**针对前半程C—主体空间关系学习的单变量训练比较**，而不是继续用后处理或只改最后几步来救宏观错层。检验假说是：若训练改动确实改善这一能力，固定坏噪声与DEV的真实RAW应更经常围绕C形成相容主体，且保留已有支撑/组合正例；仅降低训练插值MSE、I或改变款式不算通过。

本轮还不足以在低t加权、模型生成态训练、条件读取结构或关系覆盖数据之间直接选定一个。因此不据这张表自动加loss、扫切分点、回退模块或启动训练。USER→COMMON可以保存为功能诊断组合，**不晋级为默认pipeline**：它有局部代价改善，但没有新增DEV明确采用，靠背有边界退化，也没有完整D0/POST对照。

此前D0对细结构的损伤证据继续独立保留，本轮零D0没有复验或修复它。下一次生成器改动仍需RAW/POST分开验收。

## 8. 执行与工程记录

资源准备：RTX5080总16,303MiB，空闲14,419MiB；24逻辑CPU，65.84GB可用RAM。第一个真实请求后整卡约2,526MiB、空闲13,454MiB，GPU利用率5%（Windows即刻复查20%）。按1.5GiB/worker的保守进程估计、1.2倍率与2GiB额外余量，允许第二个授权worker增量启动。

**实际400次前向段各约31.71秒和30.97秒，调用时间区间未重叠。** 第二任务的启动核验/加载跨过了第一任务结束，因此不能把“计划最多两个worker”写成取得了双worker推理加速。两个阶段均无反向/优化器，峰值allocated592,047,616、reserved648,019,968字节；这不等于完整进程显存。测得整卡最高约2.59GB，进程显存为N/A；无GPU错误，没有停止用户其他进程。

实际CPU评分样本约3.28秒、RSS101MiB，随后使用3个独立评分进程和2个渲染进程，各数值线程1。所有源评分函数未改，查询未重抽。

新评分包装发生两次文件合同错误：先在原评分器要求为空的输出目录中写了尝试记录，随后又与原评分器的`complete.json`重名。保护守卫均拒绝覆盖。第一次未落盘的评分明确重算一次；第二次原评分已完整落盘，直接验证复用，没有再次重算。包装元数据已移至独立父目录，原版/修正版、hash和说明保留在[engineering/](engineering/README.md)。**额外GPU计算为0，原评分定义和历史结果均未改变。**

归档首次另因原评分依赖位于相邻`E:/MeshFlow-Control`而停止；将该已知只读来源单独映射到`source_snapshot/external/`后完成，不涉及重新评分或前向。

## 9. 交付与复核

- [冻结清单](registration.json)、[完整图集与同位置截面](gallery/index.html)、[逐输出全指标](analysis/per_output_all_metrics.csv)、[真实图审](analysis/visual_review.csv)。
- 本地`evaluation/CU`、`evaluation/UC`保存32份RAW与全部后半程轨迹；旧CC/UU以原路径及hash引用，不复制权重。
- `source_snapshot/`包含实际运行包装及原评分源码；`source_hashes.json`、`code.diff`和`engineering/`保存版本证据。
- 恢复/验收入口：`python -B -m r2.pilots.bench_trajectory_crossover_v1.resume`。完成后仅执行只读核验，不增加模型调用。
- 小包`BENCH_TRAJECTORY_CROSSOVER_REVIEW.zip`包含报告、全部逐例CSV、全部16页图像、代码和账本；大权重、冗长面对缓存及轨迹NPZ保留本地，清单可追溯。

固定实验结束。未切换默认模型、训练、清理旧资产、commit或push。
