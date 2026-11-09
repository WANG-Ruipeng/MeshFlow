# NATIVE_MESH_DIRECT_CONTEXT_READOUT_V1 — 实验报告

本轮已经实际完成独立代码迁移、三臂各 1000 次训练更新、272 条新 Euler50 生成、16 条可信 START 缓存复核和全部 288 条评价。**实现成立，输入 C 始终精确保留；没有观察到 B/C 相对等预算 A 的一致质量收益，也不足以支持 CLEAN memory 比同容量 MIXED 更好。** 有局部指标改善，也有相反方向的代价。START 默认权重保持不变。

## 先回答本轮问题

| 问题 | 实际结果 |
| --- | --- |
| 整理了什么代码？ | 在 E:/MeshFlow-Control 建立独立可安装包；必要主干、Native/T1、Geo、HYBRID coupling 和评分模块迁入包内；三臂共用模型、trainer、CLI；新增版本化 checkpoint、按参数身份恢复 AdamW、外部资产接口、可恢复总入口和调用账本。原 E:/MeshFlow 未改。 |
| 三臂更新与完整生成？ | A/B/C 各 1000 更新；每臂 500 步生成 P1 32 条，1000 步生成 P1 32 + P2 16 条，即各 80 条完整 Euler50。另 START 32 条新生成 + 16 条可信缓存；正式结果共 288 条。 |
| C 是否原样保留？ | 全部 288 输出的所有 51 个状态、最终 NPZ、二进制 PLY 均通过坐标字节、顺序/映射、面片多重性、绕序核验。N/K、finite、实际 Gaussian 一并通过。这不等于 C 与 F 已焊接或接缝正确。 |
| B/C 是否超过等预算 A？ | 未形成一致证据。P1 的 Dsym 均值 B 比 A 低 0.554%、C 低 1.575%，但各自均为 4 父改善/4 父退化，且全 8 父都存在噪声间方向反转。冲突、覆盖、尾距和实际形状不支持普遍改善。 |
| C 是否超过 B？ | P1 FF 受影响面比例在 7/8 父下降；同时 CF/I 更高、参考 coverage 更低、p95 更高。Dsym 的平均优势对 f596 父敏感。无法认定 clean source 更优，也不能据此证明 known 特征“污染”。 |
| 来源重建与开放完成结论是否相同？ | 不同。P1 有较小参考误差/接口指标变化；P2 实际可见另生相对背板、座面错层和尖片。参考距离改善没有转化为明确开放完成改善；款式改变也没有一律判错。 |
| 能否脱离旧实验工作区运行？ | 已在独立安装环境、无旧 PYTHONPATH 下测试并实际训练/采样。运行需要外部权重与准备数据，不动态导入历史 experiments。GPU 调度当前使用 Linux/WSL 的 fcntl 锁。 |
| 未验证的部分？ | 新父泛化、多训练种子、human usability、通路差异的唯一因果机制、BF16 加回后的有效残差强度、完整流形/水密保证均未验证。没有新增实验来补这些结论。 |

## 实验与实现合同

保留的 START 文件 SHA256 为 d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80，模型状态 SHA256 为 166c71fef888f26494fc4a0af1776b77a484efa72a6f3ce06552c604fbecc972。三臂均恢复 194 个原参数张量和 AdamW step5000/moments；原参数量 130,401,155。B/C 额外 16 张量、2,359,296 参数，新 optimizer group 从 step0 开始。原 START 只读且没有复制多份。

代码核验确认：Chair 是变长入口；known time=1/free time=t；role 后加 Geo，Geo 实际 P=I、没有邻面传播；原 block 先对三角角均值形成 face token 做联合 attention，再广播回角；known hidden 在原主干内仍会更新，sampler 只硬保持坐标。公开旧 trainer 不能替代本轮多父 HYBRID 续训器。

A 无新增分支。B/C 在第 3/6/9/12 个完整 block 后，以有效 free 角查询 known 角；B 读取该层 mixed hidden，C 读取同次前向中 coordinate+role+Geo 后、主干前的 known H0 副本。四个独立 readout 的维度 192、4 heads、无 bias、non-affine LN、无 dropout/gate/额外 MLP。Q/K/V 由独立 RNG 同初始化，W_O 为零；C memory 不 detach、不跨更新或 Euler 步缓存。原主干 BF16、新分支内部 FP32，FP32 参数/积分，TF32 关闭、SDPA math、严格确定性。

32 训练父、453 READY 条件均迁移为独立 manifest。新流的 1000 批 × 8 样本对三臂逐例字节配对：每批 8 个不同父，4 个 alpha=1、4 个 U[.9,1.1]；每父每臂 250 次曝光。原 logit-normal t、pre-scaling/free-only nested OT 与混噪次序保留；OT 后 Z_h=sqrt(.75)Z+.5epsilon，不二次 OT；推理实际从标准 Gaussian 开始。唯一训练损失是全 batch 有效 free 标量归一的 FM。全参数训练，AdamW lr1e-5、betas(.9,.95)、wd0、clip1，无 EMA/scheduler/新 loss/后处理。

评分迁移也用旧 RAW 的 129 个共同数值标量核验，与历史缓存完全一致，未静默修改旧定义。预检完成 47F/11B 和 6 条完整初始化回放：A/B0/C0 同输入 FP32/BF16 与原模型一致；B0/C0 两输入 Euler50 所有状态与原输出逐位一致；原参数、原 optimizer、RNG 不变；FP32 非零置换误差在既定容差内；padding 有效输出不变、padding 梯度为零；clean memory 与 free/t 独立且 known encoder 梯度可达。隔离小模块更新不写入正式起点。[预检记录](preflight/real/preflight.json)、[逐步配对审计](audit/training_pairing.json)。

## P1：已观察 DEV 父的配对结果

8 个固定 DEV 父，各 OLD n0/n1 和新的 n4/n5；新噪声实际数组已与旧 n0..n3 核对。与当前 32 训练父交集为 0，但这仍是此前观察过的 DEV，不能称新父泛化。先对同父噪声平均，再对 8 父等权；不是把 32 条噪声当 32 个独立父。1000 为主要终点，500 为预先指定次要观察。

CF 指 proper C/free 交叉触及的 free 面比例；FF 指 proper 或正面积共面重叠触及的 free 面比例，二者分母都是 N−K。I 是 C 面积采样的 crossing/near-duplicate 并集，不是上述面数比例。I 两分项有交集，不能相加当总量。FF 不包含 C/C，不能冒称全网格 Ri。

S、E、GT 距离、p95 和 coverage 均为 **REFERENCE_COMPLETION_DIAGNOSTICS**。较高 S/E/coverage 表示更符合这个参考完成的相应定义，并非所有合理完成的必要条件。Dsym 使用双向有限三角面距离的归一均方值；本表乘 1000，p95 用 L 的百分比，百分比行是比例×100。多样性没有“越高越好”的箭头。

| 指标 | START | A1000 | B1000 | C1000 |
| --- | --- | --- | --- | --- |
| CF proper / free % | 5.7341 | 5.7890 | 5.6573 | 5.7942 |
| FF illegal / free % | 90.3916 | 88.7598 | 89.5551 | 88.8440 |
| C冲突覆盖 I % | 4.6478 | 3.2288 | 3.2883 | 3.3813 |
| I crossing % | 1.9348 | 1.6571 | 1.5427 | 1.6235 |
| I near-duplicate % | 3.0121 | 1.8082 | 1.9775 | 1.9562 |
| 参考 S % | 10.5071 | 13.3012 | 13.3857 | 13.0075 |
| 参考 E % | 88.0746 | 93.1008 | 93.1394 | 93.4568 |
| GT→F MSE/L² ×1000 | 1.6598 | 1.8205 | 1.8242 | 1.8194 |
| F→GT MSE/L² ×1000 | 1.5934 | 1.5321 | 1.5099 | 1.4805 |
| Dsym ×1000 | 1.6266 | 1.6763 | 1.6670 | 1.6499 |
| 固定查询 p95 / L % | 7.0308 | 6.7298 | 6.7403 | 6.7848 |
| 固定查询 coverage % | 28.8025 | 36.8225 | 36.3892 | 36.0962 |
| Diversity ×1000 | 1.1837 | 1.1273 | 1.1462 | 1.1812 |

这张表同时保留正反结果：

- B−A：CF 降 0.132 个百分点，FF 升 0.795 个百分点（7/8 父上升），I 升 0.060 个百分点；Dsym 降 0.554%，coverage 降 0.433 个百分点，p95 升 0.0106%L。不能只取 CF 和 Dsym 宣布成功。
- C−A：Dsym 降 1.575%，主要来自 F→GT 方向；GT→F 几乎不变。CF 基本不变，I 升 0.153 个百分点，S 降 0.294 个百分点，coverage 降 0.726 个百分点，p95 升 0.0550%L；p95 在 7/8 父变差。
- C−B：FF 降 0.711 个百分点，7/8 父下降，是一个实际、较一致的局部改善；同时 CF 升 0.137 个百分点、I 升 0.093 个百分点，coverage 降 0.293 个百分点，p95 升 0.0445%L。I 的 crossing 增 0.081 个百分点、near-duplicate 反降 0.021 个百分点，不能统称重复增加。这构成取舍，不支持整体晋级。

逐父 Dsym 相对差如下；负号只表示更接近该参考完成：

| 父ID前8位 | B−A Dsym相对差 | C−A | C−B |
| --- | --- | --- | --- |
| 33ec1e64 | +1.210% | -0.983% | -2.167% |
| 341bcae1 | +0.292% | +0.238% | -0.054% |
| 534b0e6e | -1.383% | +2.045% | +3.476% |
| 866e7248 | -2.964% | +0.792% | +3.870% |
| 86ec9c79 | +4.298% | -3.448% | -7.427% |
| bdd29e65 | +0.303% | +1.487% | +1.180% |
| c1b8c2f0 | -2.192% | -2.990% | -0.815% |
| f59627a0 | -1.632% | -10.797% | -9.317% |

OLD/NEW 的分开结果也不应隐去：B−A 的 Dsym 是 −1.456% / +0.323%，S 是 −0.752 / +0.922 个百分点；C−A 的 Dsym 是 −1.556% / −1.593%，S 是 −1.403 / +0.816 个百分点。C 的均值改善没有消除各父各噪声间方向反转。

去一父仅作敏感性说明，不删除任何案例：去掉 f596 后，C−B 的 Dsym 均值从 −1.712e−5 变为 +1.160e−6；其 E 均值增量也从 +0.317 个百分点变成 −0.154 个百分点。C−A 的 Dsym 去任何单父仍为负，但去 f596 后优势大幅缩小。C−B 的 S 去掉 bdd 后由 −0.378 个百分点变为 +0.533 个百分点。均值、父分布和噪声配对必须一起读。

S/I 的本批检测上下界、E 的 definite-reference 下/上界在这里重合；E 的全局 eligibility 上下界仍单独保留在完整表中。它们是检测/适用性界，不是统计置信区间。例：C−B 的 E 全局平均差界为 [0.0031419, 0.0032194]，仍不能代替跨父/训练种子不确定性。

500 步次要结果如下，完整披露，未事后改成最优终点：

| 指标 | START | A500 | B500 | C500 |
| --- | --- | --- | --- | --- |
| CF proper / free % | 5.7341 | 5.3258 | 5.3811 | 5.6629 |
| FF illegal / free % | 90.3916 | 89.6127 | 89.8512 | 89.7569 |
| C冲突覆盖 I % | 4.6478 | 3.2898 | 3.3997 | 3.3524 |
| I crossing % | 1.9348 | 1.4404 | 1.3870 | 1.5503 |
| I near-duplicate % | 3.0121 | 1.9913 | 2.1439 | 1.9623 |
| 参考 S % | 10.5071 | 12.2201 | 11.3514 | 11.9536 |
| 参考 E % | 88.0746 | 89.0337 | 88.0193 | 88.6046 |
| GT→F MSE/L² ×1000 | 1.6598 | 1.6018 | 1.5919 | 1.5988 |
| F→GT MSE/L² ×1000 | 1.5934 | 1.4637 | 1.4335 | 1.4338 |
| Dsym ×1000 | 1.6266 | 1.5327 | 1.5127 | 1.5163 |
| 固定查询 p95 / L % | 7.0308 | 6.7830 | 6.7523 | 6.7609 |
| 固定查询 coverage % | 28.8025 | 35.2966 | 35.5896 | 35.3271 |
| Diversity ×1000 | 1.1837 | 1.0217 | 0.9972 | 1.0158 |

500 的 Dsym 三臂都低于 START，1000 时三臂均略高于 START，而 S/E/coverage 等仍有改善。续训本身也存在不同诊断间的取舍。不能据此重命名 500 为预指定最优模型，或把唯一参考误差当质量否决器。

## P2：没有唯一完整 GT 的开放完成

训练前冻结 4 条件、每条件 4 Gaussian、N=192。座面两三角形/32 三角形是同一几何的不同细分，不能当两个独立形状；不同条件的噪声各自固定，不能把两个细分面的差值解释成严格同噪声因果对照。稀疏两面条件属于训练分布外压力测试。没有构造隐藏完整目标，S/E/来源距离为 NA。

以下为各条件四候选均值，全部候选均纳入。百分比定义同上，尺度为预登记常数 L=sqrt(3)*2/.3762；不得横向把 P1/P2 归一数值视为相同样本分布。

| 条件 | 指标 | START | A1000 | B1000 | C1000 |
| --- | --- | --- | --- | --- | --- |
| P2_seat2 | CF proper / free % | 10.395 | 10.789 | 11.316 | 11.184 |
| P2_seat2 | FF illegal / free % | 96.316 | 94.868 | 95.658 | 94.474 |
| P2_seat2 | C冲突覆盖 I % | 9.509 | 28.772 | 31.519 | 31.787 |
| P2_seat2 | I crossing % | 9.509 | 9.937 | 10.498 | 10.522 |
| P2_seat2 | I near-duplicate % | 0.000 | 19.067 | 21.326 | 21.326 |
| P2_seat32 | CF proper / free % | 20.938 | 15.781 | 15.000 | 15.156 |
| P2_seat32 | FF illegal / free % | 91.562 | 90.938 | 91.250 | 90.156 |
| P2_seat32 | C冲突覆盖 I % | 12.488 | 11.633 | 10.864 | 11.365 |
| P2_seat32 | I crossing % | 10.657 | 10.034 | 9.851 | 9.448 |
| P2_seat32 | I near-duplicate % | 2.197 | 2.344 | 1.636 | 2.661 |
| P2_back2 | CF proper / free % | 2.105 | 3.947 | 3.158 | 3.421 |
| P2_back2 | FF illegal / free % | 93.158 | 91.053 | 91.184 | 91.184 |
| P2_back2 | C冲突覆盖 I % | 7.434 | 9.314 | 9.119 | 7.642 |
| P2_back2 | I crossing % | 7.410 | 9.314 | 8.569 | 7.642 |
| P2_back2 | I near-duplicate % | 0.769 | 0.122 | 1.794 | 0.000 |
| P2_seat_back4 | CF proper / free % | 8.378 | 5.984 | 7.580 | 8.112 |
| P2_seat_back4 | FF illegal / free % | 94.681 | 94.415 | 94.282 | 93.617 |
| P2_seat_back4 | C冲突覆盖 I % | 5.176 | 14.551 | 14.746 | 15.027 |
| P2_seat_back4 | I crossing % | 4.431 | 3.516 | 3.479 | 4.187 |
| P2_seat_back4 | I near-duplicate % | 0.793 | 11.133 | 11.328 | 11.365 |

两三角形座面上，A/B/C 的 I 从 START 的 9.51% 增至 28.77%/31.52%/31.79%，主要来自 near-duplicate 项：0% → 19.07%/21.33%/21.33%。座背组合也有类似代价。near-duplicate 检测可能覆盖薄壁，并非每次命中都能语义判错；这里的图像同时确实显示生成座面与 C 接近重层或错层，不能忽略。C 在单背板条件的 I 比 A/B 低，但另生相对背板的问题仍在，说明低几何冲突不足以证明合理使用 C。

实际查看全部 P2 16 页、64 输出的四视图＋线框，记录于 [逐输出 AI 审阅](analysis/visual_review_root.csv)：

- seat2 n1 可形成清楚四腿椅；n0 续训后变矮宽沙发式主体，属于款式变化，同时背部斜片、开口是真实局部缺陷。n2/n3 常见蓝色座面向较窄主体外伸。没有为了贴近“原款式”而判错。
- seat32 n0 的续训臂可辨圆角背扶手椅，但 C 和座架有错层、座上杂片；n2 B1000 有明显座上悬片。细分更密本身没有建立可靠条件适配。
- back2 的各噪声都可见在蓝 C 对面的座边另生橙色靠背；seat_back4 同样出现双竖板位置，n1 还清楚出现 C 水平面低于另生座面。几何位置确定，语义相容性保留“可能不合理”，不假定唯一 GT。
- 大多数图像仍可辨椅形，不能把全部结果概括成彻底崩坏。B/C 对 A 的肉眼细差经常无法稳定区分，也没有一致消除这些具体缺陷。

## 完整视觉审阅、多样性与几何限制

P1 224 条输出已由三份分工记录覆盖：先完成各自全部无参考页并保存初次观察，再查看对应参考页。多数仍可辨椅形，500→1000 的款式/比例变化比同终点 A/B/C 的差别更显著。33ec 的侧板封闭变化、86ec 的实心/条栅差异不能仅因不似参考判错；bdd n4 的长三角片和 C1000 外伸片、f596 n1 C1000 更紊乱的折片则是独立局部问题。f596 n4 C1000 又较简洁，不能将单个坏例或好例推广。c1b n4 的高脚凳式完成属于款式变化。全部 288 条（P1 224 + P2 64）的实际审阅已按请求/模型身份合并，图像 hash 与原图集核对一致。

图库共有 48 个无来源参考页面，另有 32 个 P1 参考页面；所有注册输出都在其中。Input C 蓝色、free 橙色，统一条件内相机/尺度/材质，四视图＋线框，不独立居中、ICP、裁飞片或修网格。所谓 blind 目录只隐藏来源参考，仍显示模型标签；不是方法匿名盲评。AI 实际看图有逐输出记录，human usability=NOT_RUN，不生成“可用率”。

同 C 四候选的六个自由表面距离全部保留，P1 OLD/NEW 各自只有一个候选对；ALL 包含跨 OLD/NEW 的配对，因此不等于前两项简单平均。下表为 D_free ×1000：

| 面板/噪声 | START | A1000 | B1000 | C1000 |
| --- | --- | --- | --- | --- |
| P1/OLD | 0.71164 | 0.86377 | 0.87952 | 0.87456 |
| P1/NEW | 1.72441 | 1.68250 | 1.73258 | 1.58079 |
| P1/ALL | 1.18371 | 1.12726 | 1.14616 | 1.18118 |
| P2/OPEN | 1.78241 | 1.91847 | 1.90964 | 1.94041 |

这些只是变化量。既有合理款式变化，也有碎片/错层，不可把较大多样性计为质量加分。P2/OPEN 的均值仅用于描述这四个条件，seat2/seat32 并非独立几何证据。

全部输出的 exact-coordinate C/F 共享边为 0；P1 exact-coordinate face components 均值 150.625，P2 183，三臂没有改变这个事实。它描述 RAW triangle soup 的严格坐标连接，不是容差焊接后的组件数，也不能把每个组件都叫作视觉飞片。所有输出的 exact free duplicate excess、near-degenerate 与 exact-zero-area 比例为 0；这也不排除近重层、狭长面、交叉及断缝。RAW mesh 没有被焊接/修法向/删面/补洞。全网格流形与水密认证为 NOT_IMPLEMENTED。

## 机制和成本：确实学到了，但幅度不是质量证据

逐步配对 FM 均值 A/B/C 为 0.11702824 / 0.11701513 / 0.11700694；B−A −0.0112%，C−A −0.0182%。这点训练 loss 差异不支撑生成质量结论。Q/K/V 第一步因 W_O=0 梯度/更新为零，从第 2 至 1000 步可达；W_O 全部更新非零。新参数实际更新 L2 均值 B/C 为 0.002093/0.002511；旧参数梯度范数均值约 0.398，**旧/全参数实际更新 L2 没有测量**。

1000 步各层 readout residual/hidden 范数比，B 为 [4.36e−4,4.20e−4,2.37e−4,4.02e−5]，C 为 [3.78e−4,3.48e−4,1.47e−4,2.93e−5]。这是 FP32 残差转成 hidden dtype 之前的测量；没有测 BF16 加回后有效扰动，不能据此直接断言“被量化吃掉”或“作用太弱就是唯一失败原因”。attention 熵与梯度可达也不是接合质量证明。[完整机制记录与曲线](analysis/training_evidence.md)。

| 实测训练成本 | A | B | C |
| --- | --- | --- | --- |
| 1000 更新总壁钟秒 | 1447.20 | 1602.41 | 1590.92 |
| 平均每批 F+B 秒 | 1.25060 | 1.39338 | 1.38591 |
| PyTorch peak allocated GiB | 2.727 | 2.793 | 2.793 |
| PyTorch peak reserved GiB | 3.125 | 3.176 | 3.176 |

RTX 5080，驱动报告总显存 16303 MiB；WSL 24 CPU 核、约 62.5 GiB RAM。按任务固定单 GPU worker，遵守共享锁；CPU 准备/评分/审阅可并行，最终评分为 4 workers，每进程受线程限制。B/C F+B 约比 A 多 11%，小于预设 3 倍上限。训练期有 CPU 审计重叠，故成本不是严格隔离吞吐基准。allocated/reserved 不能相加，也不等于进程显存；WDDM 进程独占峰值为 NA，驱动监控及阶段资源记录另存，训练期设备空闲通常约 9.8–10 GiB，保留要求余量。

| 预算类别 | 实际 | 硬上限 |
| --- | --- | --- |
| 正式 microbatch forward/backward | 24000 / 24000 | 24000 / 24000 |
| optimizer updates | 3000 | 3000 |
| 单步预检 F/B | 47 / 11 | 64 / 24 |
| 完整 Euler50 尝试/完成 | 278 / 278 | 320 |
| sampling forward | 13900 | 16000 |
| 累计 GPU 持锁 | 5535.56 秒（92.26 分钟） | 6 小时 |

278 包括 272 正式新生成和 6 初始化回放，16 缓存不计新调用。采样 ledger 的逐调用 RETURN 为 13800；另 100 次原模型 reference forward 有 RESERVE 和两条完整 rollout 完成记录，没有逐步 RETURN，完整性审计已单独核对，未将其丢掉。训练/采样模型错误、坏 batch 跳过、重抽噪声与重试均为 0。CPU OT 8000 条一次完成（52.71 秒），最终 CPU 评分 104.54 秒；代码整理/CPU/人工或 AI 审阅不计 GPU 持锁。

工程修复包括导出审计字段别名、验证数组一致后的 START NPZ 无损重包兼容、controller 的 COMPLETE500 恢复/身份守卫、移除未使用依赖和许可证说明。这些没有改变模型/训练/评分数学，也没有追加实验。早期导入或审计脚本的失败记录保留，[工程变更](audit/engineering_changes.json)、[可移植性验收](audit/portability_review.json) 和 [调用/完整性总账](audit/delivery_integrity.json) 可复查。图像查看器曾遇 Windows 1385 登录错误，改为经授权命令读取同一 PNG 并无损显示，未把失败读取记为已审阅。

## 独立使用与交付索引

独立源码：**E:/MeshFlow-Control**。外部制品：**E:/MeshFlow-Control-artifacts**。权重、数据、RAW、日志和图集都在源码外，原 START 仍从原保留路径只读引用。新 repo 只初始化本地 Git，未 commit、未设 remote、未 push。

已在无旧 PYTHONPATH 的独立目录通过 28 项 CPU 合成测试，真实 GPU 预检和三臂运行分别留证。--help、配置检查不隐式启动 CUDA；新 checkpoint 在新进程 strict 加载后从独立 C/N/z 请求生成。模型实参不包含 GT/Gamma/source UID。安装采用已有依赖环境中的离线 editable 安装；没有宣称在空白机器上完全离线自动取得依赖。

最小依赖与可复制 prepare/train/sample/evaluate/render 命令见 [README](../MeshFlow-Control/README.md)、[复现说明](../MeshFlow-Control/docs/reproduction.md) 和 [pyproject.toml](../MeshFlow-Control/pyproject.toml)。必要声明：vendored backbone/utils.py 保留 NVIDIA 2024 express-license 通知，独立再分发授权没有核实；仓库 MIT 不能自动覆盖该文件。原作者来源与通知保留在 [THIRD_PARTY_NOTICES](../MeshFlow-Control/THIRD_PARTY_NOTICES.md) 和 [SOURCE_MAP](../MeshFlow-Control/docs/source_map.md)。

| 交付内容 | 路径 |
| --- | --- |
| 方法与集成说明 | [METHOD.md](METHOD.md)、[method](../MeshFlow-Control/docs/method.md)、[integration](../MeshFlow-Control/docs/integration.md) |
| 三臂配置 | [A](../MeshFlow-Control/configs/a_baseline.yaml)、[B](../MeshFlow-Control/configs/b_mixed_readout.yaml)、[C](../MeshFlow-Control/configs/c_clean_readout.yaml) |
| 六个含 optimizer/RNG/cursor 的 checkpoint | [checkpoints/](checkpoints/)，文件 SHA256 见 [integrity](audit/delivery_integrity.json) |
| 外部训练数据与固定流 | [train_manifest](data/train_manifest.json)、[train_plan](training_stream/train_plan.json) |
| 固定评价请求、全部 RAW/PLY 与轨迹 | [requests](data/evaluation/requests.json)、[all_completed](outputs/all_completed.json) |
| 全部逐输出数值与完整 registry | [per_output_metrics.csv](evaluation/per_output_metrics.csv)、[metric_registry_results.csv](evaluation/metric_registry_results.csv) |
| 逐噪声、逐父和终点比较 | [paired_noise](analysis/comparisons/paired_noise.csv)、[paired_parent](analysis/comparisons/paired_parent.csv)、[checkpoint distributions](analysis/comparisons/checkpoint_distributions.csv) |
| 全部多样性候选对 | [diversity_pairs.csv](evaluation/diversity_pairs.csv) |
| 图集、完整 AI 逐输出审阅 | [INDEX.html](gallery/INDEX.html)、[visual_review_all.csv](analysis/visual_review_all.csv)、[review receipt](analysis/visual_review_completion.json) |
| 机制、成本、实际账本 | [training evidence](analysis/training_evidence.json)、[budget calls](audit/budget/calls.jsonl)、[GPU occupancy](audit/budget/gpu_occupancy.jsonl) |
| 源文件 hash 与最终只读验收 | [source_map.json](../MeshFlow-Control/docs/source_map.json)、[final acceptance](audit/final_acceptance.json)、[source inventory](audit/final_source_hashes.csv) |

## 研究判断与停止边界

IMPLEMENTATION：正确执行，迁移和硬保持合同通过。READOUT_EFFECT：分支有实际学习和输出差异，但在这一个训练种子、1000 更新和固定小面板中，没有足以支持 B/C 晋级的整体优势。OPEN_COMPLETION_EVIDENCE：能生成可辨认且有款式变化的椅子；稀疏 C 的位置/尺度整合、局部碎片和严格连接仍有缺陷，未得到稳定改善。

“clean memory 必要”“known 混合污染是唯一病因”“只要放大残差就能解决”“稀疏条件分布不匹配是主因”都仍是假说。当前证据最多提示：后续决策应以 C 是否被合理使用的逐例问题为依据，不能用 FM 微降、参考距离微降或 attention 熵替代。没有自动加深/加宽、改变 loss、启动额外训练、扫描新条件、后处理或切换默认模型。本轮固定任务到此停止。
