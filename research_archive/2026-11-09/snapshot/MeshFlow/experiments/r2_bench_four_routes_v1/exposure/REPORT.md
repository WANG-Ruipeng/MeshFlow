# M 新增3000步曝光：实际训练与固定面板结果

本分支已完成固定3,000次更新、46次Euler50、46次单次官方D0、完整177字段三阶段评分及276条实际图审。**增加曝光带来了RAW采用和总体FF/I的真实正信号，但固定后端没有把它兑现成更好的最终成品：RAW三轴交集29→33/46，FINAL反而17→15/46。** 保留本终点作为生成侧研究候选，不替代当前完整流程。

## 实验具体改变了什么

恢复宽池M的live生成器、匹配AdamW与RNG，从global22500继续到25500。没有更换数据、引入新损失或新网络，也不从官方权重重启。训练目标仍是原Native/T1＋role＋Geo、free-only FM、OT后HYBRID0.25和M原有组均方误差系数；batch8、microbatch1、lr1e-5及确定性精度设置保持不变。

新3,000步实际24,000次实例曝光：OLD18,000、FOCUS3,000、HIGH3,000。六个HIGH父各新增500次，累计各750次。原父序列与增强序列重放两遍，但时间、噪声、epsilon和排列使用新登记随机身份，未重复旧训练实例。OLD/FOCUS也获得额外曝光，因此这检验的是继续原配方的额外计算收益，不能单独证明HIGH曝光或数据规模的因果作用。

M使用30父，其中新增宽池四父也在训练内。评价的NEW_WIDE_TRAIN沿用旧命名，确为本臂熟悉父；它们对于窄池B/SEAM是未见父，不能将两个方向当成同起点排名。训练与评价的独立父、旧DEV及留出边界见根目录[METHODS.md](../METHODS.md)。

预先固定终点，不按中间结果选checkpoint。全部原资产与评分器保持只读；本轮没有改变默认模型。

## 评价与数值结果

原23父、每父一条C、两份固定噪声，共46请求。M的138行原三阶段结果经身份核验复用，EXPOSURE新增138行；RAW、单次D0/GEO后的BASE，以及相同HINGE_OPP015→ALPHA→HOLE后的FINAL分开。C的精确坐标保留检查276/276通过。

下表为父内两噪声平均后父等权，百分比已乘100。CF、FF的分母是实际free面，I是原ConflictCov_C；I只在两模型、三阶段共同PASS的19/23父（38/46请求）展示。其余四父的LOW_SUPPORT原值保留于完整表，不补零。

| 阶段 | 模型 | CF↓% | FF↓% | I↓%，19父 | 有合法C/F共边 | 方向合法路径可达free面积% |
|---|---|---:|---:|---:|---:|---:|
| RAW | M | 9.548 | 88.123 | 10.833 | 0/46 | 0.000 |
| RAW | EXPOSURE | 9.782 | 87.443 | 8.852 | 0/46 | 0.000 |
| BASE | M | 7.896 | 17.443 | 6.925 | 14/46 | 6.321 |
| BASE | EXPOSURE | 7.552 | 11.367 | 4.974 | 19/46 | 9.631 |
| FINAL | M | 7.602 | 14.260 | 6.755 | 14/46 | 7.968 |
| FINAL | EXPOSURE | 7.350 | 9.044 | 4.828 | 19/46 | 11.059 |

FINAL变化：CF−0.251个百分点、FF−5.216个百分点、I−1.927个百分点；小角面47.038%→44.706%，近退化面两者均0。合法C/F边均值0.761→1.065，单面合法边图孤岛25.543→27.000，非流形边0.978→0.957。不是所有指标都改善，也不能用新增接口数证明主体已接好。

| FINAL组别 | CF M→EXPOSURE % | FF M→EXPOSURE % | I M→EXPOSURE % |
|---|---:|---:|---:|
| OLD，8父 | 5.928→7.489 | 12.879→7.356 | 5.664→1.505 |
| FOCUS，2父 | 0.758→1.087 | 5.087→0.000 | 0.024→0.024 |
| DEV，4父 | 5.694→8.013 | 20.588→18.231 | 11.237→12.744 |
| HOLDOUT，1父 | 19.930→21.260 | 13.137→13.386 | 12.305→12.769 |
| HIGH_TRAIN，2父 | 1.316→3.704 | 6.256→5.093 | NA→NA |
| HIGH_HOLDOUT，2父 | 21.194→23.125 | 17.061→9.740 | 8.508→5.322 |
| NEW_WIDE_TRAIN，4父 | 9.544→0.000 | 18.163→8.297 | 4.358→2.625（2/4父） |

总体均值下降不能遮住DEV与独立HOLDOUT的代价。尤其DEV的CF与I升高，不能称为跨父整体质量升级。所有177字段原值、每父分布与配对差值分别见[完整逐输出](results/per_output.csv)、[逐组177字段](results/analysis_v1/full177_parent_macro.csv)、[配对177字段](results/analysis_v1/paired177_parent_macro.csv)、[主口径配对](results/analysis_v1/paired_primary_parent_macro.csv)。

## 面积、删面与已实际观察到的阶段问题

FINAL/RAW的free面积比按请求求比、再父等权：57.092%→58.814%；被删除原生成槽对应的RAW参考面积比例32.519%→30.703%；原槽保留比例33.114%→34.454%；平均删原槽219.783→216.304。实际最终free面积均值20.094→19.375模型单位平方。面积比包含D0变形、合并与桥面，不能当作GT完整性，也不能把全部变化归因于删面。

全组并非依靠“更多删面”得到低FF，但局部仍有严重损失。NEW_WIDE_TRAIN的FINAL/RAW面积比38.051%→35.803%，删除槽RAW参考面积50.053%→53.216%，最终free面积均值10.788→6.969。实际图中：945/n0的RAW从额外高座层转为直接使用蓝色C；bcb/n1从C穿过灰色后背转为C位于完整圆环座顶。这是熟悉父上的用途进步。两者经过相同后端后，细腿框或圆弧端撑仍大面积消失，不能计作最终成品成功。

af/n0的RAW由板背端板款式变为窄条靠背与四腿，C仍是座面；这种款式变化本身不作为输赢。但它的细结构在POST中大量变成断续线条。77的支撑从细框变成柱式也不按来源样式评分；POST柱体腰部仍有大缺口。

逐槽身份和两种参考坐标面积见[删面账本](results/analysis_v1/deletion_fate_per_slot.csv)、[逐输出面积](results/analysis_v1/deletion_area_per_output.csv)。图册包含相同双视角与固定front/side位置[截面图](gallery/index.html)。

对应原图（每页同时包含M/EXPOSURE与固定截面）：

- [945/n0 RAW](gallery/RAW__9452149917cb2e27f4fb4dee5181bee__COMPLETE_SEAT_UPPER_SURFACE_GROUP__HIGH_JOINT_V1__n0.png)
- [945/n0 BASE](gallery/BASE_POST__9452149917cb2e27f4fb4dee5181bee__COMPLETE_SEAT_UPPER_SURFACE_GROUP__HIGH_JOINT_V1__n0.png)
- [bcb/n1 RAW](gallery/RAW__bcb4f576747e10bdd7f6e86d7d335698__COMPLETE_FLAT_SEAT_TOP_SURFACE__HIGH_JOINT_V1__n1.png)
- [bcb/n1 BASE](gallery/BASE_POST__bcb4f576747e10bdd7f6e86d7d335698__COMPLETE_FLAT_SEAT_TOP_SURFACE__HIGH_JOINT_V1__n1.png)
- [af/n0 RAW](gallery/RAW__af264c02f69566cdf4fb4dee5181bee__COMPLETE_SLAT_SEAT_UPPER_SURFACE_GROUP__HIGH_JOINT_V1__n0.png)
- [af/n0 BASE](gallery/BASE_POST__af264c02f69566cdf4fb4dee5181bee__COMPLETE_SLAT_SEAT_UPPER_SURFACE_GROUP__HIGH_JOINT_V1__n0.png)

## 完整真实图审及逐请求转移

三位审阅者逐张打开138页双模型图，检查两视角和两条固定截面，共276条真实观察；未从指标或历史标签生成当前判断。五轴独立、保留UNCERTAIN，这是非盲AI_PROXY，不是独立人类认证。三轴交集仅是`REASONABLE ∧ PLAUSIBLE ∧ PRESENT`已有标签的描述统计，不等于无交、无裂或水密。

| 阶段 | M三轴交集 | EXPOSURE三轴交集 | 新增 | 丢失 |
|---|---:|---:|---:|---:|
| RAW | 29/46 | 33/46 | 5 | 1 |
| BASE | 17/46 | 15/46 | 0 | 2 |
| FINAL | 17/46 | 15/46 | 0 | 2 |

RAW新增身份是`6b/n1`（OLD）、`2d/n1`及`459/n1`（DEV）、`945/n0`及`bcb/n1`（NEW_WIDE_TRAIN）；丢失`4cd/n1`（OLD）。所以并非只有熟悉父有信号：两个DEV父各有一份固定噪声改善，但同父另一噪声和完整后处理尚未一起改善，不能写成稳定泛化。FINAL丢失`4cd/n1`和`af/n0`，没有新增交集。

| 组别 | RAW交集 M→EXPOSURE | FINAL交集 M→EXPOSURE |
|---|---:|---:|
| OLD | 14/16→14/16 | 11/16→10/16 |
| FOCUS | 4/4→4/4 | 0/4→0/4 |
| DEV | 0/8→2/8 | 0/8→0/8 |
| HOLDOUT | 0/2→0/2 | 0/2→0/2 |
| HIGH_TRAIN | 3/4→3/4 | 3/4→3/4 |
| HIGH_HOLDOUT | 2/4→2/4 | 2/4→2/4 |
| NEW_WIDE_TRAIN | 6/8→8/8 | 1/8→0/8 |

RAW的合理C采用29→33，整体合理42→43，主支撑存在两者46；可见碎片/裂口28→25、重层/折叠18→13。FINAL仍保留C采用29→33的变化，但主体不完整22→25、主支撑缺失/损伤20→24；可见碎片/裂口两者43/46，重层/折叠17→13。C用途变好与成品更完整是不同结果。

这里既有宏观失败，也有阶段损失：4cd/n1在RAW就另造高灰座；f495/fd多数仍错层；2d/459的新采用、6b的L形主体、945的端框、af的窄条靠背和bcb的圆弧支撑，分别存在后处理保留不足。不能把所有失败都解释为D0，也不能将这些已经实际看到的阶段损伤归咎于训练没有采用C。

[全部观察](visual_review/summary_v1/actual_observations.csv)、[逐轴转移](visual_review/summary_v1/paired_transitions.csv)、[三轴身份转移](results/delivery_v1/visual_joint_transitions.csv)、[数值及分组汇总](results/delivery_v1/summary.json)均已保存。RAW新增五例不是净增四例的随意替代解释，而是明确列出的逐请求配对。

## 实际训练曲线、成本与恢复

真实50步分箱中，首箱weighted objective0.09998、unweighted FM0.09854；末箱分别0.08237、0.08080，中间仍明显波动。曲线只是正常训练流的统计，不是固定验证集误差，也不证明收敛或因果机制。未增加teacher、probe或额外rollout。[曲线](closeout/loss_curve.png)、[逐箱CSV](closeout/loss_curve.csv)保留全部记录。

训练3,000更新／24,000前向／24,000反向，0失败、0重放、0额外探针。活动墙钟4,320.491秒（约72分钟），与另一条授权训练及短采样并发；整卡峰值12,156MiB、最低空闲3,824MiB，不能当作本进程独占显存。训练worker allocator峰值allocated4.035GB、reserved4.368GB，两者也不等于完整进程开销。

46次Euler50共2,300生成器前向、0头前向，142.828秒；全部RAW保存后执行46次单次D0及46次GEO，46.803秒。原RAW评分46行98.673秒，POST评分92行249.191秒，均3CPU worker；138页图册3worker、19.274秒。保持科学配置、未重抽噪声。图库继承的W/M副标题在首次渲染前由独立caption适配器修正，只有一个字符串常量变化；一次前置import断言失败发生于任何模型、评分、渲染调用之前，修正记录见[audit](audits/caption_adapter_cpu_check.json)。冻结科学源码未改。

实际终点：[推理权重](training/M_PLUS3000/generator_step3000.pt)，SHA256 `70e51d7913f99ad017d023e47965387b28692075ed9964b61d9d55dd1f0ff6b3`；[完整恢复状态](training/M_PLUS3000/checkpoint_step3000.pt)，SHA256 `3c1e9e1db0f9a462632bd70a91b5acaa4bef1dd40ad20ad90a194ab17bbd49fd`。身份、24,000条曝光及随机流核验见[训练收口](closeout/training.json)、[曝光核验](closeout/exposure.json)。

恢复入口见[总RUNBOOK](../RUNBOOK.md)，[评价收口](closeout/evaluation.json)已PASS；完成后的模型不继续训练。**本轮支持保留EXPOSURE的生成侧进步与FF/I改善，尚不支持把EXPOSURE＋现有后端晋级为全面更好的流程。** 不自动扩大训练、切默认或启动后端新方案。
