# Bench 联合 C/Y 平移增强：1500 步固定配对实验

**结论：局部 C 采用有正信号，但几何质量代价明显，本轮不晋级。** 共同平移让若干请求的灰色主体更接近并更合理地使用蓝色 C；这一变化已在部分 RAW 中出现。但主体与支撑完整性没有同步全面改善，最终 DEV 的 FF、CF 和有效支持域 I 均上升。保留这条有限正信号及全部反例，不能写成纯全负，也不能宣布得到更好的整体生成基线。

实验：`R2_BENCH_JOINT_SHIFT_V1`。主对照为 RELATION，USER v2 为历史次对照；所有变化先按主对照判读。本报告只记录这一次固定预算的实际结果，不自动晋级、切默认、延长训练、commit 或 push。

## 1. 当前问题与唯一训练变量

此前模型能保留输入 C 的坐标，却可能在另一高度生成灰色替代座面，或把蓝色输入包在主体内部；RAW 主体完整性与 POST 的保面/接续也可能背离。这一轮检验共同移动 C 和目标 Y 能否改善位置条件的使用，不是用后处理移动用户输入，也不把唯一来源款式当生成成功标准。

与 RELATION 完全相同的 26 个训练父、91 条条件、起始 COMMON12000 live 权重、fresh AdamW、1500 更新及原采样流。这是从共同 COMMON12000 初始化重新分叉，**不是从 RELATION 终点续训**。本轮只增加这一条训练臂。每更新前四个实例为原 identity，后四个保持原 XYZ U[0.92,1.08] 缩放，再给 C/Y 同加独立平移，每轴 U[-0.04L,+0.04L]，固定 L=√3×2/0.3762。Gaussian 与 HYBRID epsilon 不随平移；按平移后目标重算原 Euclidean OT，不复用未平移匹配，不重新居中或裁剪。

Native/T1+role+Geo、owner-only SEAM、原角点 free-only FM、受限 OT 后 HYBRID0.25；batch8/micro1，AdamW lr1e-5、betas(0.9,0.95)、eps1e-8、wd0、clip1。BF16 forward、FP32 参数/积分，strict deterministic、SDPA math、TF32off、CUBLAS=:4096:8。没有新增网络、占用头更新、loss 或推理噪声；local1500/global13500 为唯一预定终点。

本轮没有增加独立父对象或新条件；真实训练池按部位保留 BACK_SURFACE 11、SEAT_BACK 8、SEAT_SUPPORT 18、SEAT_SURFACE 25、LEG_SUPPORT 25、PART_ASSEMBLY 3、OTHER 1 条，共91条。每父实际曝光460或462次；部位是构造/审阅元数据，模型输入仍为C/N，推理不读来源Y或owner真值。条件的版本与来源关系保留在data/manifest.json，不将旧文件名自动当作实际部位。

## 2. 输入一致性、训练曲线与计算账本

实际 1500 次优化器更新、12000 次曝光；训练前向/反向各12000，25500组调用闭合。新旧原 schedule 字段比对 132000 项，runtime N/K/t/角色分母等比对 108000 项。零平移 6000、非零平移 6000；起始模型/优化器/RNG登记身份匹配。首更新前四 identity micro 的 40 项 loss/SSE/分母精确相等检查：PASS。

训练单次尝试 `attempt_1792664559675368070`，所有25500次前向/反向/更新调用开始与结束闭合，无失败、无待定调用、无重试或额外重算；逐步日志恰为1—1500。平移最大向量范数为 0.62791869 模型单位，即 6.81917%L；单轴±4%L不能误写成总向量位移上限4%L。

首微批比较仅使用已记录日志，不额外运行 probe；它不证明训练后全部轨迹逐位相同。原输入流相同不代表目标值相同：后四实例坐标变了，因此训练 loss 不能当同目标的严格配对生成质量分数。

| 平移轴 | 最小模型单位 | 最大模型单位 | 均值 | RMS |
| --- | --- | --- | --- | --- |
| x | -0.368293 | 0.368276 | -0.002961 | 0.213044 |
| y | -0.368257 | 0.368242 | -0.001486 | 0.212157 |
| z | -0.368285 | 0.367926 | 0.004105 | 0.213491 |

| 训练量 | 前100步 | 后100步 |
| --- | --- | --- |
| 全部free速度MSE | 0.370799 | 0.135574 |
| bulk速度MSE | 0.357142 | 0.132396 |
| seam-tip速度MSE | 0.756787 | 0.227209 |
| 一步endpoint MSE | 0.117449 | 0.063542 |

每100步曲线严格为 SSE 总和/有效标量分母总和；endpoint 使用原记录 endpoint SSE，不由 velocity MSE 乘平均时间推算。预算完成不等于收敛或泛化成功。

![训练曲线](analysis_joint_shift_v1/training_curve_plots_v1/training_curves.png)

## 3. 固定评价与当前图审合同

原48请求=12父×2条件×2噪声；C/N、标准Gaussian、C-only布局、查询和来源绕序保持原身份。48请求均在本轮26训练父之外，但历史TRAIN的4父在COMMON阶段见过；DEV是反复使用的开发面板，不是独立论文测试集，官方预训练的对象级接触情况也未在本轮重新证明。

Euler50 RAW → 单次官方D0 → 恢复C/seam → GEO_C与来源定向导出为BASE_POST；再固定 HINGE_OPP015/ALPHA/HOLE 得到CLEAN_FINAL_POST。新增48 RAW、48 D0、144评分行，与两个旧参考合计432行、实际177数值字段。原结果只读，没有把旧预期179补成虚构指标。

**BASE_POST包含D0、C/seam恢复和GEO_C等多个步骤。RAW→BASE的损失只能归因于这条合成处理链，本轮没有中间阶段的独立干预，因此不能只归咎于D0。** FINAL是额外固定几何后处理，三模型合同相同。

主数值为噪声→条件→父对象逐层等权，ALL直接对12父平均。下面比例乘100，差为百分点；I支持域单列，LOW_SUPPORT不填成可信0。CF/FF比例、绝对非法面对、面数和表面积分开。

三模型均在本轮144张完整对照页重新查看，按 RELATION / USER / SHIFT 列序，每页两视角和固定front/side截面，432条非盲AI_PROXY五轴记录。没有沿用旧基线标签，也不从数值推断视觉标签；UNCERTAIN独立保留。

## 4. RAW / BASE / FINAL 的数值与代价

| 阶段 | 面板 | 模型 | CF% | FF% | I原值含LOW% | 小角<5°% | 删free面 | bulk可达% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RAW | ALL（12父/48请求） | RELATION | 6.95 | 93.85 | 14.73 | 37.16 | 0.00 | 0.00 |
| RAW | ALL（12父/48请求） | USER v2 | 7.09 | 93.11 | 14.24 | 36.99 | 0.00 | 0.00 |
| RAW | ALL（12父/48请求） | SHIFT | 9.66 | 93.76 | 19.26 | 37.13 | 0.00 | 0.00 |
| RAW | 历史TRAIN（4父/16请求） | RELATION | 7.01 | 94.11 | 18.37 | 38.66 | 0.00 | 0.00 |
| RAW | 历史TRAIN（4父/16请求） | USER v2 | 6.72 | 92.23 | 18.67 | 38.64 | 0.00 | 0.00 |
| RAW | 历史TRAIN（4父/16请求） | SHIFT | 8.07 | 93.50 | 23.95 | 37.26 | 0.00 | 0.00 |
| RAW | DEV（8父/32请求） | RELATION | 6.92 | 93.72 | 12.92 | 36.41 | 0.00 | 0.00 |
| RAW | DEV（8父/32请求） | USER v2 | 7.27 | 93.55 | 12.02 | 36.16 | 0.00 | 0.00 |
| RAW | DEV（8父/32请求） | SHIFT | 10.46 | 93.89 | 16.92 | 37.06 | 0.00 | 0.00 |
| BASE_POST | ALL（12父/48请求） | RELATION | 5.98 | 30.64 | 9.81 | 44.25 | 92.79 | 0.17 |
| BASE_POST | ALL（12父/48请求） | USER v2 | 6.46 | 31.98 | 11.80 | 46.33 | 88.96 | 0.06 |
| BASE_POST | ALL（12父/48请求） | SHIFT | 10.67 | 37.37 | 16.16 | 50.09 | 96.48 | 0.07 |
| BASE_POST | 历史TRAIN（4父/16请求） | RELATION | 5.69 | 26.88 | 11.40 | 42.06 | 106.12 | 0.51 |
| BASE_POST | 历史TRAIN（4父/16请求） | USER v2 | 5.19 | 29.72 | 15.53 | 46.19 | 100.44 | 0.18 |
| BASE_POST | 历史TRAIN（4父/16请求） | SHIFT | 11.72 | 34.41 | 24.59 | 52.40 | 114.25 | 0.07 |
| BASE_POST | DEV（8父/32请求） | RELATION | 6.12 | 32.52 | 9.02 | 45.35 | 86.12 | 0.00 |
| BASE_POST | DEV（8父/32请求） | USER v2 | 7.10 | 33.11 | 9.94 | 46.40 | 83.22 | 0.00 |
| BASE_POST | DEV（8父/32请求） | SHIFT | 10.15 | 38.86 | 11.94 | 48.94 | 87.59 | 0.07 |
| CLEAN_FINAL_POST | ALL（12父/48请求） | RELATION | 5.98 | 27.42 | 9.85 | 43.03 | 92.79 | 0.35 |
| CLEAN_FINAL_POST | ALL（12父/48请求） | USER v2 | 6.37 | 28.12 | 11.75 | 45.08 | 88.96 | 0.11 |
| CLEAN_FINAL_POST | ALL（12父/48请求） | SHIFT | 10.49 | 33.46 | 16.26 | 49.19 | 96.48 | 0.33 |
| CLEAN_FINAL_POST | 历史TRAIN（4父/16请求） | RELATION | 5.69 | 23.33 | 11.47 | 40.42 | 106.12 | 1.04 |
| CLEAN_FINAL_POST | 历史TRAIN（4父/16请求） | USER v2 | 5.19 | 26.27 | 15.42 | 45.29 | 100.44 | 0.33 |
| CLEAN_FINAL_POST | 历史TRAIN（4父/16请求） | SHIFT | 11.42 | 31.51 | 24.91 | 51.22 | 114.25 | 0.76 |
| CLEAN_FINAL_POST | DEV（8父/32请求） | RELATION | 6.12 | 29.46 | 9.04 | 44.33 | 86.12 | 0.00 |
| CLEAN_FINAL_POST | DEV（8父/32请求） | USER v2 | 6.97 | 29.04 | 9.91 | 44.97 | 83.22 | 0.01 |
| CLEAN_FINAL_POST | DEV（8父/32请求） | SHIFT | 10.03 | 34.43 | 11.94 | 48.18 | 87.59 | 0.12 |

此大表的I列仅为完整原值留档，包含LOW_SUPPORT，**不作为有效I主结论**；有效I见4.2的PASS_ONLY父宏均值。

### 4.1 相对主对照与历史次对照的配对差

| 参考 | 阶段 | 面板 | CF%差 | FF%差 | I原值含LOW%差 | 小角<5°%差 | 删free面差 | bulk可达%差 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RELATION | RAW | ALL（12父/48请求） | 2.71 | -0.08 | 4.53 | -0.03 | 0.00 | 0.00 |
| RELATION | RAW | 历史TRAIN（4父/16请求） | 1.06 | -0.60 | 5.58 | -1.40 | 0.00 | 0.00 |
| RELATION | RAW | DEV（8父/32请求） | 3.54 | 0.18 | 4.00 | 0.65 | 0.00 | 0.00 |
| RELATION | BASE_POST | ALL（12父/48请求） | 4.69 | 6.74 | 6.35 | 5.84 | 3.69 | -0.10 |
| RELATION | BASE_POST | 历史TRAIN（4父/16请求） | 6.02 | 7.53 | 13.20 | 10.34 | 8.12 | -0.44 |
| RELATION | BASE_POST | DEV（8父/32请求） | 4.03 | 6.34 | 2.92 | 3.60 | 1.47 | 0.07 |
| RELATION | CLEAN_FINAL_POST | ALL（12父/48请求） | 4.51 | 6.04 | 6.41 | 6.16 | 3.69 | -0.01 |
| RELATION | CLEAN_FINAL_POST | 历史TRAIN（4父/16请求） | 5.72 | 8.17 | 13.43 | 10.80 | 8.12 | -0.27 |
| RELATION | CLEAN_FINAL_POST | DEV（8父/32请求） | 3.91 | 4.97 | 2.90 | 3.85 | 1.47 | 0.12 |
| USER v2 | RAW | ALL（12父/48请求） | 2.57 | 0.65 | 5.02 | 0.14 | 0.00 | 0.00 |
| USER v2 | RAW | 历史TRAIN（4父/16请求） | 1.35 | 1.28 | 5.28 | -1.38 | 0.00 | 0.00 |
| USER v2 | RAW | DEV（8父/32请求） | 3.18 | 0.34 | 4.90 | 0.90 | 0.00 | 0.00 |
| USER v2 | BASE_POST | ALL（12父/48请求） | 4.21 | 5.40 | 4.36 | 3.76 | 7.52 | 0.01 |
| USER v2 | BASE_POST | 历史TRAIN（4父/16请求） | 6.53 | 4.69 | 9.07 | 6.21 | 13.81 | -0.11 |
| USER v2 | BASE_POST | DEV（8父/32请求） | 3.06 | 5.75 | 2.00 | 2.54 | 4.38 | 0.07 |
| USER v2 | CLEAN_FINAL_POST | ALL（12父/48请求） | 4.12 | 5.34 | 4.52 | 4.12 | 7.52 | 0.22 |
| USER v2 | CLEAN_FINAL_POST | 历史TRAIN（4父/16请求） | 6.23 | 5.24 | 9.48 | 5.92 | 13.81 | 0.44 |
| USER v2 | CLEAN_FINAL_POST | DEV（8父/32请求） | 3.06 | 5.39 | 2.03 | 3.21 | 4.38 | 0.11 |

这些差值不自动给出胜负；逐请求升/降/不变、逐父177字段在 `analysis_joint_shift_v1/`。与USER的比较包含此前数据池变化，只对RELATION的配对才隔离本轮联合平移。

### 4.2 I有效域、绝对FF及删面

| 阶段 | 面板 | 模型 | PASS | LOW_SUPPORT | PASS I% | LOW原I% |
| --- | --- | --- | --- | --- | --- | --- |
| RAW | ALL（12父/48请求） | RELATION | 46 | 2 | 15.89 | 0.00 |
| RAW | ALL（12父/48请求） | USER v2 | 46 | 2 | 15.09 | 0.00 |
| RAW | ALL（12父/48请求） | SHIFT | 46 | 2 | 20.64 | 0.00 |
| RAW | 历史TRAIN（4父/16请求） | RELATION | 16 | 0 | 18.37 | NA |
| RAW | 历史TRAIN（4父/16请求） | USER v2 | 16 | 0 | 18.67 | NA |
| RAW | 历史TRAIN（4父/16请求） | SHIFT | 16 | 0 | 23.95 | NA |
| RAW | DEV（8父/32请求） | RELATION | 30 | 2 | 14.66 | 0.00 |
| RAW | DEV（8父/32请求） | USER v2 | 30 | 2 | 13.29 | 0.00 |
| RAW | DEV（8父/32请求） | SHIFT | 30 | 2 | 18.99 | 0.00 |
| BASE_POST | ALL（12父/48请求） | RELATION | 46 | 2 | 10.12 | 0.00 |
| BASE_POST | ALL（12父/48请求） | USER v2 | 46 | 2 | 12.21 | 0.00 |
| BASE_POST | ALL（12父/48请求） | SHIFT | 46 | 2 | 16.43 | 0.00 |
| BASE_POST | 历史TRAIN（4父/16请求） | RELATION | 16 | 0 | 11.40 | NA |
| BASE_POST | 历史TRAIN（4父/16请求） | USER v2 | 16 | 0 | 15.53 | NA |
| BASE_POST | 历史TRAIN（4父/16请求） | SHIFT | 16 | 0 | 24.59 | NA |
| BASE_POST | DEV（8父/32请求） | RELATION | 30 | 2 | 9.48 | 0.00 |
| BASE_POST | DEV（8父/32请求） | USER v2 | 30 | 2 | 10.55 | 0.00 |
| BASE_POST | DEV（8父/32请求） | SHIFT | 30 | 2 | 12.35 | 0.00 |
| CLEAN_FINAL_POST | ALL（12父/48请求） | RELATION | 46 | 2 | 10.17 | 0.00 |
| CLEAN_FINAL_POST | ALL（12父/48请求） | USER v2 | 46 | 2 | 12.16 | 0.00 |
| CLEAN_FINAL_POST | ALL（12父/48请求） | SHIFT | 46 | 2 | 16.54 | 0.00 |
| CLEAN_FINAL_POST | 历史TRAIN（4父/16请求） | RELATION | 16 | 0 | 11.47 | NA |
| CLEAN_FINAL_POST | 历史TRAIN（4父/16请求） | USER v2 | 16 | 0 | 15.42 | NA |
| CLEAN_FINAL_POST | 历史TRAIN（4父/16请求） | SHIFT | 16 | 0 | 24.91 | NA |
| CLEAN_FINAL_POST | DEV（8父/32请求） | RELATION | 30 | 2 | 9.52 | 0.00 |
| CLEAN_FINAL_POST | DEV（8父/32请求） | USER v2 | 30 | 2 | 10.53 | 0.00 |
| CLEAN_FINAL_POST | DEV（8父/32请求） | SHIFT | 30 | 2 | 12.35 | 0.00 |

配对仅双方PASS的I另保留于 `paired_metric_summary.csv`，低支持项不借均值掩盖。

| 阶段 | 面板 | 模型 | FF面对总数 | CF面对总数 | 保留free面均值 | 删面均值 | free面积 | 非流形边 | 单面孤岛 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RAW | ALL（12父/48请求） | RELATION | 19345.0 | 919.0 | 180.25 | 0.00 | 61.44 | 0.00 | 175.88 |
| RAW | ALL（12父/48请求） | USER v2 | 18547.0 | 950.0 | 180.25 | 0.00 | 59.33 | 0.00 | 175.88 |
| RAW | ALL（12父/48请求） | SHIFT | 19900.0 | 1364.0 | 180.25 | 0.00 | 59.64 | 0.00 | 175.88 |
| RAW | 历史TRAIN（4父/16请求） | RELATION | 6871.0 | 336.0 | 202.25 | 0.00 | 72.85 | 0.00 | 195.62 |
| RAW | 历史TRAIN（4父/16请求） | USER v2 | 6975.0 | 338.0 | 202.25 | 0.00 | 67.12 | 0.00 | 195.62 |
| RAW | 历史TRAIN（4父/16请求） | SHIFT | 7267.0 | 353.0 | 202.25 | 0.00 | 63.20 | 0.00 | 195.62 |
| RAW | DEV（8父/32请求） | RELATION | 12474.0 | 583.0 | 169.25 | 0.00 | 55.73 | 0.00 | 166.00 |
| RAW | DEV（8父/32请求） | USER v2 | 11572.0 | 612.0 | 169.25 | 0.00 | 55.43 | 0.00 | 166.00 |
| RAW | DEV（8父/32请求） | SHIFT | 12633.0 | 1011.0 | 169.25 | 0.00 | 57.87 | 0.00 | 166.00 |
| BASE_POST | ALL（12父/48请求） | RELATION | 1451.0 | 384.0 | 87.46 | 92.79 | 42.41 | 2.52 | 30.90 |
| BASE_POST | ALL（12父/48请求） | USER v2 | 1458.0 | 429.0 | 91.29 | 88.96 | 41.50 | 2.48 | 33.38 |
| BASE_POST | ALL（12父/48请求） | SHIFT | 1687.0 | 654.0 | 83.77 | 96.48 | 39.78 | 1.83 | 37.96 |
| BASE_POST | 历史TRAIN（4父/16请求） | RELATION | 457.0 | 131.0 | 96.12 | 106.12 | 52.33 | 2.31 | 35.31 |
| BASE_POST | 历史TRAIN（4父/16请求） | USER v2 | 462.0 | 125.0 | 101.81 | 100.44 | 49.53 | 2.69 | 37.75 |
| BASE_POST | 历史TRAIN（4父/16请求） | SHIFT | 497.0 | 181.0 | 88.00 | 114.25 | 42.66 | 1.69 | 36.50 |
| BASE_POST | DEV（8父/32请求） | RELATION | 994.0 | 253.0 | 83.12 | 86.12 | 37.45 | 2.62 | 28.69 |
| BASE_POST | DEV（8父/32请求） | USER v2 | 996.0 | 304.0 | 86.03 | 83.22 | 37.48 | 2.38 | 31.19 |
| BASE_POST | DEV（8父/32请求） | SHIFT | 1190.0 | 473.0 | 81.66 | 87.59 | 38.35 | 1.91 | 38.69 |
| CLEAN_FINAL_POST | ALL（12父/48请求） | RELATION | 1298.0 | 382.0 | 87.48 | 92.79 | 42.73 | 2.52 | 20.98 |
| CLEAN_FINAL_POST | ALL（12父/48请求） | USER v2 | 1269.0 | 418.0 | 91.29 | 88.96 | 41.85 | 2.48 | 22.12 |
| CLEAN_FINAL_POST | ALL（12父/48请求） | SHIFT | 1443.0 | 643.0 | 83.81 | 96.48 | 40.08 | 1.83 | 26.69 |
| CLEAN_FINAL_POST | 历史TRAIN（4父/16请求） | RELATION | 406.0 | 131.0 | 96.19 | 106.12 | 52.80 | 2.31 | 22.75 |
| CLEAN_FINAL_POST | 历史TRAIN（4父/16请求） | USER v2 | 395.0 | 125.0 | 101.81 | 100.44 | 49.91 | 2.69 | 23.69 |
| CLEAN_FINAL_POST | 历史TRAIN（4父/16请求） | SHIFT | 419.0 | 174.0 | 88.12 | 114.25 | 43.11 | 1.69 | 24.62 |
| CLEAN_FINAL_POST | DEV（8父/32请求） | RELATION | 892.0 | 251.0 | 83.12 | 86.12 | 37.70 | 2.62 | 20.09 |
| CLEAN_FINAL_POST | DEV（8父/32请求） | USER v2 | 874.0 | 293.0 | 86.03 | 83.22 | 37.82 | 2.38 | 21.34 |
| CLEAN_FINAL_POST | DEV（8父/32请求） | SHIFT | 1024.0 | 469.0 | 81.66 | 87.59 | 38.56 | 1.91 | 27.72 |

面对总数是样本输出总数，不是独立父数或唯一物理缺陷数；面数、面积的变化不能被FF比例取代。

### 4.3 DEV逐部位最终结果

| 部位 | 模型 | 请求 | CF% | FF% | I原值含LOW% | 小角<5°% | 删free面 | bulk可达% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BACK_SURFACE | RELATION | 6 | 10.14 | 30.37 | 15.54 | 60.96 | 121.00 | 0.00 |
| BACK_SURFACE | USER v2 | 6 | 9.22 | 36.02 | 14.81 | 55.88 | 119.17 | 0.05 |
| BACK_SURFACE | SHIFT | 6 | 4.89 | 29.24 | 16.34 | 56.78 | 124.17 | 0.01 |
| LEG_SUPPORT | RELATION | 10 | 3.57 | 32.41 | 7.89 | 44.42 | 68.50 | 0.00 |
| LEG_SUPPORT | USER v2 | 10 | 6.15 | 30.48 | 8.82 | 43.31 | 73.10 | 0.00 |
| LEG_SUPPORT | SHIFT | 10 | 10.00 | 41.73 | 7.62 | 49.39 | 76.70 | 0.00 |
| PART_ASSEMBLY | RELATION | 4 | 2.27 | 17.47 | 8.06 | 39.36 | 130.50 | 0.00 |
| PART_ASSEMBLY | USER v2 | 4 | 3.80 | 23.99 | 9.74 | 52.94 | 102.50 | 0.00 |
| PART_ASSEMBLY | SHIFT | 4 | 3.16 | 23.99 | 23.68 | 54.88 | 102.75 | 0.93 |
| SEAT_SUPPORT | RELATION | 4 | 16.75 | 43.09 | 4.11 | 21.60 | 31.25 | 0.00 |
| SEAT_SUPPORT | USER v2 | 4 | 16.97 | 47.79 | 8.51 | 22.60 | 27.25 | 0.00 |
| SEAT_SUPPORT | SHIFT | 4 | 37.55 | 61.61 | 15.67 | 29.26 | 28.00 | 0.00 |
| SEAT_SURFACE | RELATION | 8 | 2.92 | 24.26 | 8.56 | 45.60 | 87.25 | 0.00 |
| SEAT_SURFACE | USER v2 | 8 | 2.87 | 15.18 | 8.37 | 46.05 | 87.25 | 0.00 |
| SEAT_SURFACE | SHIFT | 8 | 3.60 | 20.83 | 6.30 | 46.32 | 96.00 | 0.00 |

## 5. 五轴图审及逐请求配对变化

五轴列内枚举顺序如下，三项计数均保留：

- `overall_shape`：REASONABLE / INCOMPLETE / UNCERTAIN。
- `C_use`：PLAUSIBLE / REPLACED_OR_IGNORED / UNCERTAIN。
- `support_retention`：PRESENT / MISSING_OR_DAMAGED / UNCERTAIN。
- `visible_fragment_or_gap`：YES / NO_OBVIOUS / UNCERTAIN。
- `layering_or_fold`：YES / NO_OBVIOUS / UNCERTAIN。

| 阶段 | 面板 | 模型 | 请求 | overall_shape | C_use | support_retention | visible_fragment_or_gap | layering_or_fold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RAW | ALL（12父/48请求） | RELATION | 48 | 37/5/6 | 16/18/14 | 41/6/1 | 17/24/7 | 25/1/22 |
| RAW | ALL（12父/48请求） | USER v2 | 48 | 36/6/6 | 17/16/15 | 40/7/1 | 17/25/6 | 22/2/24 |
| RAW | ALL（12父/48请求） | SHIFT | 48 | 36/8/4 | 26/10/12 | 39/8/1 | 20/19/9 | 17/3/28 |
| RAW | 历史TRAIN（4父/16请求） | RELATION | 16 | 14/1/1 | 8/3/5 | 15/1/0 | 1/13/2 | 6/1/9 |
| RAW | 历史TRAIN（4父/16请求） | USER v2 | 16 | 15/0/1 | 9/2/5 | 16/0/0 | 1/13/2 | 4/2/10 |
| RAW | 历史TRAIN（4父/16请求） | SHIFT | 16 | 13/3/0 | 9/1/6 | 13/3/0 | 3/9/4 | 3/3/10 |
| RAW | DEV（8父/32请求） | RELATION | 32 | 23/4/5 | 8/15/9 | 26/5/1 | 16/11/5 | 19/0/13 |
| RAW | DEV（8父/32请求） | USER v2 | 32 | 21/6/5 | 8/14/10 | 24/7/1 | 16/12/4 | 18/0/14 |
| RAW | DEV（8父/32请求） | SHIFT | 32 | 23/5/4 | 17/9/6 | 26/5/1 | 17/10/5 | 14/0/18 |
| BASE_POST | ALL（12父/48请求） | RELATION | 48 | 27/17/4 | 16/18/14 | 32/15/1 | 33/10/5 | 24/5/19 |
| BASE_POST | ALL（12父/48请求） | USER v2 | 48 | 27/16/5 | 17/16/15 | 31/16/1 | 29/14/5 | 23/7/18 |
| BASE_POST | ALL（12父/48请求） | SHIFT | 48 | 24/19/5 | 26/10/12 | 29/17/2 | 32/10/6 | 15/9/24 |
| BASE_POST | 历史TRAIN（4父/16请求） | RELATION | 16 | 11/3/2 | 8/3/5 | 14/2/0 | 11/4/1 | 6/1/9 |
| BASE_POST | 历史TRAIN（4父/16请求） | USER v2 | 16 | 11/3/2 | 9/2/5 | 14/2/0 | 7/8/1 | 5/2/9 |
| BASE_POST | 历史TRAIN（4父/16请求） | SHIFT | 16 | 10/5/1 | 9/1/6 | 12/4/0 | 8/7/1 | 3/3/10 |
| BASE_POST | DEV（8父/32请求） | RELATION | 32 | 16/14/2 | 8/15/9 | 18/13/1 | 22/6/4 | 18/4/10 |
| BASE_POST | DEV（8父/32请求） | USER v2 | 32 | 16/13/3 | 8/14/10 | 17/14/1 | 22/6/4 | 18/5/9 |
| BASE_POST | DEV（8父/32请求） | SHIFT | 32 | 14/14/4 | 17/9/6 | 17/13/2 | 24/3/5 | 12/6/14 |
| CLEAN_FINAL_POST | ALL（12父/48请求） | RELATION | 48 | 27/17/4 | 16/18/14 | 32/15/1 | 33/10/5 | 24/5/19 |
| CLEAN_FINAL_POST | ALL（12父/48请求） | USER v2 | 48 | 27/16/5 | 17/16/15 | 31/16/1 | 28/15/5 | 23/7/18 |
| CLEAN_FINAL_POST | ALL（12父/48请求） | SHIFT | 48 | 24/19/5 | 26/10/12 | 29/17/2 | 32/10/6 | 15/9/24 |
| CLEAN_FINAL_POST | 历史TRAIN（4父/16请求） | RELATION | 16 | 11/3/2 | 8/3/5 | 14/2/0 | 11/4/1 | 6/1/9 |
| CLEAN_FINAL_POST | 历史TRAIN（4父/16请求） | USER v2 | 16 | 11/3/2 | 9/2/5 | 14/2/0 | 6/9/1 | 5/2/9 |
| CLEAN_FINAL_POST | 历史TRAIN（4父/16请求） | SHIFT | 16 | 10/5/1 | 9/1/6 | 12/4/0 | 8/7/1 | 3/3/10 |
| CLEAN_FINAL_POST | DEV（8父/32请求） | RELATION | 32 | 16/14/2 | 8/15/9 | 18/13/1 | 22/6/4 | 18/4/10 |
| CLEAN_FINAL_POST | DEV（8父/32请求） | USER v2 | 32 | 16/13/3 | 8/14/10 | 17/14/1 | 22/6/4 | 18/5/9 |
| CLEAN_FINAL_POST | DEV（8父/32请求） | SHIFT | 32 | 14/14/4 | 17/9/6 | 17/13/2 | 24/3/5 | 12/6/14 |

图审是非盲AI代理观察，不是独立人评；可见碎/豁和截面端点不认证精确拓扑裂缝。各部位五轴及所有枚举、转移表详见 `visual_review/summary.csv`、`paired_transitions.csv`。

### 5.1 最终 C采用变化：相对 RELATION

| 请求 | 历史组 | 原→新 | 参考观察 | SHIFT观察 |
| --- | --- | --- | --- | --- |
| 122a480cfcdd742650c626aa72455dae__LEG_SUPPORT_1__n0 | TRAIN | UNCERTAIN -> PLAUSIBLE | 宽底座和座背保留，扶手薄片缺口仍可见；蓝腿被底座包围，采用判不确定。 | 蓝腿保留在支撑位置；灰腿细线化、座面空缺仍明显，最终处理未补回RAW中的完整支撑。 |
| 122a480cfcdd742650c626aa72455dae__SEAT_SURFACE_1__n0 | TRAIN | REPLACED_OR_IGNORED -> UNCERTAIN | 最终保留高处灰顶与较低蓝C两套面，端部开口和橙斜片仍在，C被替代。 | 最终灰座板较完整并靠近蓝C平面，两侧板支撑保留；遮挡下不能区分融合和薄层替代。 |
| 20222f467d257022c8bbbd98dee48cb__COMBO_SEAT_SUPPORT__n0 | TRAIN | PLAUSIBLE -> UNCERTAIN | 最终灰支腿仍细线化，蓝端板和座条保留；主体座背可读但支撑不完整。 | 最终仍是竖长板/端板与蓝条组合，水平座面和完整支撑不足，C使用不确定。 |
| 20222f467d257022c8bbbd98dee48cb__COMBO_SEAT_SUPPORT__n1 | TRAIN | REPLACED_OR_IGNORED -> PLAUSIBLE | 最终高灰顶和蓝座条仍错层，顶侧豁口持续；并非C采用改善。 | 最终条带座背和单侧蓝支撑仍缺完整主体；C座条虽合理使用，不能作为整体成功。 |
| 459f90aa2162b1f1d46c340938e2ff1c__LEG_SUPPORT_1__n0 | DEV | REPLACED_OR_IGNORED -> PLAUSIBLE | Blue bent seat/support and gray opposite support make a bench silhouette, but a torn gray replacement seat floats above the blue seat; the front and side sections confirm the separate upper layer. | The blue surface now reads as the seat rather than sitting below an independent upper seat; gray patches still overlap it and lower strips remain. This is an adoption gain with visible residual defects. |
| 459f90aa2162b1f1d46c340938e2ff1c__LEG_SUPPORT_1__n1 | DEV | REPLACED_OR_IGNORED -> PLAUSIBLE | Gray upper seat remains separated above blue C and has patchy torn faces; both end supports exist, but the supplied seat is replaced by the elevated layer. | Blue seat is cleanly exposed and paired with a gray opposite support. Remaining gray under-seat lines may be lower structure or residual layers; the images do not establish clean topology. |
| 5dda4ef4ab9add9411ee8243f54292d6__SEAT_SURFACE_1__n1 | DEV | UNCERTAIN -> PLAUSIBLE | Bench frame and supports remain recognizable, with some thin or broken surface strips; blue upper strip is still paired with an orange folded sheet above a separate lower seat. | Blue C remains plausibly at seat height, but the newly generated legs have become extremely thin or missing after processing, leaving an incomplete supported bench. |
| 9d56087c9f98bf90717cca22daef45e7__LEG_SUPPORT_1__n1 | DEV | REPLACED_OR_IGNORED -> PLAUSIBLE | Generated supports become sparse after processing and gray seat/back strips still sit off the blue panel; no coherent use of the large C is visible. | The bench remains organized against the blue wall with two visible front legs, unlike the offset references. Layered gray patches across the blue back and lower sheets remain, so this is a local adoption/structure gain, not a clean-mesh certificate. |
| 9fd5b67a2b20b5d1b19874039c8f3e05__LEG_SUPPORT_1__n0 | DEV | REPLACED_OR_IGNORED -> PLAUSIBLE | 处理后下层座与大灰板的错位布局仍在，蓝端架顶未承接主要座面。 | 蓝端架与灰座的用途较顺，但另一端只余细弱悬腿，主体支撑未完整。 |
| 9fd5b67a2b20b5d1b19874039c8f3e05__SEAT_SURFACE_1__n0 | DEV | REPLACED_OR_IGNORED -> PLAUSIBLE | 最终仍是蓝上灰下两套座层，细腿未恢复，明显未成完整支撑。 | C作座面的位置改善保留，但灰腿仍成细线，不能认作完整成品。 |
| 9fd5b67a2b20b5d1b19874039c8f3e05__SEAT_SURFACE_1__n1 | DEV | REPLACED_OR_IGNORED -> PLAUSIBLE | 最终下层灰座与蓝座仍分开，细弱支撑和层间空隙未恢复。 | 箱板式结构和顶部C保留，侧板上缘开口与内部灰层仍有歧义。 |
| fd355d06238853c7f0d9805ad6c2059c__LEG_SUPPORT_2__n0 | DEV | UNCERTAIN -> PLAUSIBLE | 最终蓝框与灰凳高低关系仍不协调，短灰腿及细背条未恢复，C用途难定。 | 最终C框与灰座位置较协调，但前板上方大开口仍在，整体完整性未定。 |
| fd355d06238853c7f0d9805ad6c2059c__LEG_SUPPORT_2__n1 | DEV | UNCERTAIN -> PLAUSIBLE | 最终仍为短灰腿高层主体加低蓝框，背部裂口折层明显，C用途不确定。 | 最终C框位置较合理，但灰背斜折、座下开口与支撑歧义仍在。 |

### 5.2 最终 C采用变化：相对 USER v2

| 请求 | 历史组 | 原→新 | 参考观察 | SHIFT观察 |
| --- | --- | --- | --- | --- |
| 122a480cfcdd742650c626aa72455dae__LEG_SUPPORT_1__n0 | TRAIN | UNCERTAIN -> PLAUSIBLE | 弧底座、座面及靠背保持；蓝腿作用不清，多层侧截面不单独认证非法叠层。 | 蓝腿保留在支撑位置；灰腿细线化、座面空缺仍明显，最终处理未补回RAW中的完整支撑。 |
| 122a480cfcdd742650c626aa72455dae__SEAT_SURFACE_1__n0 | TRAIN | REPLACED_OR_IGNORED -> UNCERTAIN | 最终灰顶仍线条化，低处蓝C与上层灰顶分离；缺损和跨层橙片未消失。 | 最终灰座板较完整并靠近蓝C平面，两侧板支撑保留；遮挡下不能区分融合和薄层替代。 |
| 20222f467d257022c8bbbd98dee48cb__COMBO_SEAT_SUPPORT__n0 | TRAIN | PLAUSIBLE -> UNCERTAIN | 最终蓝座条/端板用途合理，灰腿缺损未恢复；不能仅看座背判断整体良好。 | 最终仍是竖长板/端板与蓝条组合，水平座面和完整支撑不足，C使用不确定。 |
| 2d1e5eeb255e79dcab53bf7fa22f3636__BACK_SURFACE_1__n1 | DEV | UNCERTAIN -> REPLACED_OR_IGNORED | Gray seat/back survive more clearly than the other columns, but legs are very thin or partly missing and blue C remains behind a parallel gray back. | Blue back is still crossed by the gray seat strip; processing leaves sparse rails and incomplete supports rather than resolving the embedded panel. |
| 459f90aa2162b1f1d46c340938e2ff1c__LEG_SUPPORT_1__n0 | DEV | REPLACED_OR_IGNORED -> PLAUSIBLE | A detached-looking gray sheet forms a second seat above the supplied curved blue surface, with ragged gaps and strips; supports remain visible. | The blue surface now reads as the seat rather than sitting below an independent upper seat; gray patches still overlap it and lower strips remain. This is an adoption gain with visible residual defects. |
| 459f90aa2162b1f1d46c340938e2ff1c__LEG_SUPPORT_1__n1 | DEV | REPLACED_OR_IGNORED -> PLAUSIBLE | Gray seat spans above the supplied surface with visible broken patches and strips; source blue seat/support is not the main upper sitting surface. | Blue seat is cleanly exposed and paired with a gray opposite support. Remaining gray under-seat lines may be lower structure or residual layers; the images do not establish clean topology. |
| 5dda4ef4ab9add9411ee8243f54292d6__SEAT_SURFACE_1__n1 | DEV | UNCERTAIN -> PLAUSIBLE | The seat and back frame persist but some supporting legs are sparse/broken-looking; the elevated blue/orange panel remains separate from the lower sitting surface. | Blue C remains plausibly at seat height, but the newly generated legs have become extremely thin or missing after processing, leaving an incomplete supported bench. |
| 9d56087c9f98bf90717cca22daef45e7__LEG_SUPPORT_1__n1 | DEV | REPLACED_OR_IGNORED -> PLAUSIBLE | Processed gray front layers remain incomplete and weakly supported, with blue C still behind them. | The bench remains organized against the blue wall with two visible front legs, unlike the offset references. Layered gray patches across the blue back and lower sheets remain, so this is a local adoption/structure gain, not a clean-mesh certificate. |
| 9fd5b67a2b20b5d1b19874039c8f3e05__LEG_SUPPORT_1__n0 | DEV | REPLACED_OR_IGNORED -> PLAUSIBLE | 裁切大灰板和蓝梁以下灰座仍保留，清理没有解决层位分离。 | 蓝端架与灰座的用途较顺，但另一端只余细弱悬腿，主体支撑未完整。 |
| 9fd5b67a2b20b5d1b19874039c8f3e05__SEAT_SURFACE_1__n0 | DEV | REPLACED_OR_IGNORED -> PLAUSIBLE | 最终蓝C与下层灰座分开，细弱腿和层间大空隙仍在。 | C作座面的位置改善保留，但灰腿仍成细线，不能认作完整成品。 |
| 9fd5b67a2b20b5d1b19874039c8f3e05__SEAT_SURFACE_1__n1 | DEV | REPLACED_OR_IGNORED -> PLAUSIBLE | 最终仍有独立下层灰座及受损细腿，C用途未改善。 | 箱板式结构和顶部C保留，侧板上缘开口与内部灰层仍有歧义。 |
| fd355d06238853c7f0d9805ad6c2059c__LEG_SUPPORT_2__n0 | DEV | UNCERTAIN -> PLAUSIBLE | 最终灰座背细条化与短腿仍在，蓝C能否合理承接主体不确定。 | 最终C框与灰座位置较协调，但前板上方大开口仍在，整体完整性未定。 |
| fd355d06238853c7f0d9805ad6c2059c__LEG_SUPPORT_2__n1 | DEV | UNCERTAIN -> PLAUSIBLE | 最终转角灰座与蓝框的支撑关系不明，细短腿和橙灰重层未解决。 | 最终C框位置较合理，但灰背斜折、座下开口与支撑歧义仍在。 |

### 5.3 五轴最终配对计数

下表沿用原五轴枚举，不新增综合成功定义。非正向→正向包含UNCERTAIN转入，不能把这部分全部叫作确定失败被修复；完整逐枚举转移见paired_transitions.csv。

| 参考 | 面板 | 轴 | 所列正向枚举 | 转入 | 转出 | 保留 | 参考UNCERTAIN | SHIFT UNCERTAIN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RELATION | ALL（12父/48请求） | overall_shape | REASONABLE | 3 | 6 | 21 | 4 | 5 |
| RELATION | ALL（12父/48请求） | C_use | PLAUSIBLE | 11 | 1 | 15 | 14 | 12 |
| RELATION | ALL（12父/48请求） | support_retention | PRESENT | 4 | 7 | 25 | 1 | 2 |
| RELATION | ALL（12父/48请求） | visible_fragment_or_gap | NO_OBVIOUS | 4 | 4 | 6 | 5 | 6 |
| RELATION | ALL（12父/48请求） | layering_or_fold | NO_OBVIOUS | 4 | 0 | 5 | 19 | 24 |
| RELATION | 历史TRAIN（4父/16请求） | overall_shape | REASONABLE | 1 | 2 | 9 | 2 | 1 |
| RELATION | 历史TRAIN（4父/16请求） | C_use | PLAUSIBLE | 2 | 1 | 7 | 5 | 6 |
| RELATION | 历史TRAIN（4父/16请求） | support_retention | PRESENT | 0 | 2 | 12 | 0 | 0 |
| RELATION | 历史TRAIN（4父/16请求） | visible_fragment_or_gap | NO_OBVIOUS | 3 | 0 | 4 | 1 | 1 |
| RELATION | 历史TRAIN（4父/16请求） | layering_or_fold | NO_OBVIOUS | 2 | 0 | 1 | 9 | 10 |
| RELATION | DEV（8父/32请求） | overall_shape | REASONABLE | 2 | 4 | 12 | 2 | 4 |
| RELATION | DEV（8父/32请求） | C_use | PLAUSIBLE | 9 | 0 | 8 | 9 | 6 |
| RELATION | DEV（8父/32请求） | support_retention | PRESENT | 4 | 5 | 13 | 1 | 2 |
| RELATION | DEV（8父/32请求） | visible_fragment_or_gap | NO_OBVIOUS | 1 | 4 | 2 | 4 | 5 |
| RELATION | DEV（8父/32请求） | layering_or_fold | NO_OBVIOUS | 2 | 0 | 4 | 10 | 14 |
| USER v2 | ALL（12父/48请求） | overall_shape | REASONABLE | 4 | 7 | 20 | 5 | 5 |
| USER v2 | ALL（12父/48请求） | C_use | PLAUSIBLE | 10 | 1 | 16 | 15 | 12 |
| USER v2 | ALL（12父/48请求） | support_retention | PRESENT | 5 | 7 | 24 | 1 | 2 |
| USER v2 | ALL（12父/48请求） | visible_fragment_or_gap | NO_OBVIOUS | 3 | 8 | 7 | 5 | 6 |
| USER v2 | ALL（12父/48请求） | layering_or_fold | NO_OBVIOUS | 2 | 0 | 7 | 18 | 24 |
| USER v2 | 历史TRAIN（4父/16请求） | overall_shape | REASONABLE | 2 | 3 | 8 | 2 | 1 |
| USER v2 | 历史TRAIN（4父/16请求） | C_use | PLAUSIBLE | 1 | 1 | 8 | 5 | 6 |
| USER v2 | 历史TRAIN（4父/16请求） | support_retention | PRESENT | 1 | 3 | 11 | 0 | 0 |
| USER v2 | 历史TRAIN（4父/16请求） | visible_fragment_or_gap | NO_OBVIOUS | 2 | 4 | 5 | 1 | 1 |
| USER v2 | 历史TRAIN（4父/16请求） | layering_or_fold | NO_OBVIOUS | 1 | 0 | 2 | 9 | 10 |
| USER v2 | DEV（8父/32请求） | overall_shape | REASONABLE | 2 | 4 | 12 | 3 | 4 |
| USER v2 | DEV（8父/32请求） | C_use | PLAUSIBLE | 9 | 0 | 8 | 10 | 6 |
| USER v2 | DEV（8父/32请求） | support_retention | PRESENT | 4 | 4 | 13 | 1 | 2 |
| USER v2 | DEV（8父/32请求） | visible_fragment_or_gap | NO_OBVIOUS | 1 | 4 | 2 | 4 | 5 |
| USER v2 | DEV（8父/32请求） | layering_or_fold | NO_OBVIOUS | 1 | 0 | 5 | 9 | 14 |

### 5.4 与前一轮主观标签的差异披露

固定参考网格在本轮重新评阅后，1440个五轴单项中有358项与前一轮标签不同。本轮模型间比较统一使用本轮同页标签，不能把基线主观计数变化写成模型变化。原标签与原观察不覆盖；逐项差异和当前观察在 `historical_reference_review_differences.csv`。这也表明非盲AI_PROXY的一致性有限，不能把旧报告的基线计数与本轮SHIFT直接拼成同一评阅比较。

例如dev_a组的48个请求×阶段单元中，RELATION和USER的C_use各有12/48项跨轮变化，并非只改变少量总计数。全48请求FINAL的RELATION/USER旧计数为13/15，当前为16/17；它们仍是相同的冻结网格，不是参考模型改进。本轮只用同页新评阅进行配对。

| 模型 | 阶段 | 轴 | 全48中变化项 |
| --- | --- | --- | --- |
| RELATION | RAW | overall_shape | 7 |
| RELATION | RAW | C_use | 12 |
| RELATION | RAW | support_retention | 4 |
| RELATION | RAW | visible_fragment_or_gap | 14 |
| RELATION | RAW | layering_or_fold | 13 |
| RELATION | BASE_POST | overall_shape | 15 |
| RELATION | BASE_POST | C_use | 11 |
| RELATION | BASE_POST | support_retention | 8 |
| RELATION | BASE_POST | visible_fragment_or_gap | 14 |
| RELATION | BASE_POST | layering_or_fold | 13 |
| RELATION | CLEAN_FINAL_POST | overall_shape | 15 |
| RELATION | CLEAN_FINAL_POST | C_use | 11 |
| RELATION | CLEAN_FINAL_POST | support_retention | 8 |
| RELATION | CLEAN_FINAL_POST | visible_fragment_or_gap | 14 |
| RELATION | CLEAN_FINAL_POST | layering_or_fold | 13 |
| USER v2 | RAW | overall_shape | 8 |
| USER v2 | RAW | C_use | 10 |
| USER v2 | RAW | support_retention | 5 |
| USER v2 | RAW | visible_fragment_or_gap | 14 |
| USER v2 | RAW | layering_or_fold | 14 |
| USER v2 | BASE_POST | overall_shape | 18 |
| USER v2 | BASE_POST | C_use | 10 |
| USER v2 | BASE_POST | support_retention | 10 |
| USER v2 | BASE_POST | visible_fragment_or_gap | 16 |
| USER v2 | BASE_POST | layering_or_fold | 14 |
| USER v2 | CLEAN_FINAL_POST | overall_shape | 18 |
| USER v2 | CLEAN_FINAL_POST | C_use | 10 |
| USER v2 | CLEAN_FINAL_POST | support_retention | 10 |
| USER v2 | CLEAN_FINAL_POST | visible_fragment_or_gap | 15 |
| USER v2 | CLEAN_FINAL_POST | layering_or_fold | 14 |

## 6. 直接判读与边界

| 面板 | 模型 | 最终C采用PLAUSIBLE | 整体REASONABLE | 支撑MISSING_OR_DAMAGED |
| --- | --- | --- | --- | --- |
| ALL（12父/48请求） | RELATION | 16 | 27 | 15 |
| ALL（12父/48请求） | USER v2 | 17 | 27 | 16 |
| ALL（12父/48请求） | SHIFT | 26 | 24 | 17 |
| 历史TRAIN（4父/16请求） | RELATION | 8 | 11 | 2 |
| 历史TRAIN（4父/16请求） | USER v2 | 9 | 11 | 2 |
| 历史TRAIN（4父/16请求） | SHIFT | 9 | 10 | 4 |
| DEV（8父/32请求） | RELATION | 8 | 16 | 13 |
| DEV（8父/32请求） | USER v2 | 8 | 16 | 14 |
| DEV（8父/32请求） | SHIFT | 17 | 14 | 13 |

**采用正信号与质量代价必须分开记录。** 当前同轮图审DEV的C采用PLAUSIBLE为RELATION/USER/SHIFT=8/8/17，ALL为16/17/26；但ALL整体REASONABLE为27/27/24，支撑MISSING_OR_DAMAGED为15/16/17。相对RELATION，DEV有6条REPLACED_OR_IGNORED→PLAUSIBLE、3条UNCERTAIN→PLAUSIBLE，原8条PLAUSIBLE保留且无转出；另9条仍REPLACED_OR_IGNORED、6条仍UNCERTAIN。九条转入分布在5个父上，其中三条来自不确定，不应概括成九个确定失败已修好。

DEV最终父宏FF 29.45697%→34.43097%，CF 6.12238%→10.03063%。PASS_ONLY的I父宏为 9.52454%→12.35352%，两组均30/32请求PASS、2条LOW_SUPPORT。这里先在有效请求内对噪声/条件/父等权，不是30请求直接均值（后者约9.64355%→12.73438%）；不同聚合口径不能混用。

绝对非法FF面对 892→1024，CF面对 251→469；保留free面总数 2660→2613。绝对冲突对也增加，不能把比例上升只归为删面改变分母。DEV ≥30% bulk可达仍为 0/26→0/26；另外6条closed C无seam接口，记NA，不补成0/32或暗示其都需要接缝。

### 6.1 有限正例与对应代价

`459f...__LEG_SUPPORT_1__n1` 的实际部位是 **SEAT_SUPPORT**，输入为蓝色弯座板及支撑，不能因旧ID含LEG而叫单腿。SHIFT在RAW已把灰色主体从蓝弯座板上方移到下方，FINAL仍保留；这是位置/用途的局部正证据。但最终CF由20.00%升到52.17%，保留free面65→46、删面31→50，合法CF共边仍0；该C闭合，seam接续指标NA，因此不是全面几何成功。

`9fd5...__LEG_SUPPORT_1__n0` 的RAW布局也更合理使用蓝腿，然而BASE/FINAL灰端腿受损，最终保留free面81→33、删面31→79。这里是RAW→合成后处理链的阶段观察，不能单独归因于D0。`5dda...SEAT_SURFACE n1` 同样有C采用转入，却从整体合理/支撑存在变为不完整/支撑受损。

历史TRAIN也有双向变化：122a座面请求的灰面更接近C、部分表面保留更完整，但122a腿n0在POST丢灰支撑；20222组合n0在SHIFT缺少完整座面与另一端支撑，n1相对RELATION改善C位置却仍不完整，USER在该例更好。1dd座面n1仍另生高灰座面，SHIFT只缩短间距，不能将接近当采用。

部位对齐另需注意：9d的旧LEG ID实际是PART_ASSEMBLY，旧SEAT ID实际是BACK_SURFACE。本报告分部位统计一律读manifest的part_type，不从文件名猜标签。

逐阶段证据：`459f90aa2162b1f1d46c340938e2ff1c__LEG_SUPPORT_1__n1`（实际部位 `SEAT_SUPPORT`）。

| 阶段 | 模型 | CF% | FF% | 保留free | 删面 | 合法CF边 | bulk可达% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RAW | RELATION | 35.42 | 100.00 | 96 | 0 | 0 | NA |
| BASE_POST | RELATION | 20.00 | 46.15 | 65 | 31 | 0 | NA |
| CLEAN_FINAL_POST | RELATION | 20.00 | 44.62 | 65 | 31 | 0 | NA |
| RAW | SHIFT | 66.67 | 96.88 | 96 | 0 | 0 | NA |
| BASE_POST | SHIFT | 52.17 | 50.00 | 46 | 50 | 0 | NA |
| CLEAN_FINAL_POST | SHIFT | 52.17 | 47.83 | 46 | 50 | 0 | NA |

[实际RAW两视角/截面](gallery/RAW__459f90aa2162b1f1d46c340938e2ff1c__LEG_SUPPORT_1__n1.png) · [实际FINAL两视角/截面](gallery/CLEAN_FINAL_POST__459f90aa2162b1f1d46c340938e2ff1c__LEG_SUPPORT_1__n1.png)

逐阶段证据：`9fd5b67a2b20b5d1b19874039c8f3e05__LEG_SUPPORT_1__n0`（实际部位 `LEG_SUPPORT`）。

| 阶段 | 模型 | CF% | FF% | 保留free | 删面 | 合法CF边 | bulk可达% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RAW | RELATION | 1.79 | 92.86 | 112 | 0 | 0 | NA |
| BASE_POST | RELATION | 1.23 | 24.69 | 81 | 31 | 0 | NA |
| CLEAN_FINAL_POST | RELATION | 1.23 | 16.05 | 81 | 31 | 0 | NA |
| RAW | SHIFT | 27.68 | 89.29 | 112 | 0 | 0 | NA |
| BASE_POST | SHIFT | 51.52 | 24.24 | 33 | 79 | 0 | NA |
| CLEAN_FINAL_POST | SHIFT | 51.52 | 21.21 | 33 | 79 | 0 | NA |

[实际RAW两视角/截面](gallery/RAW__9fd5b67a2b20b5d1b19874039c8f3e05__LEG_SUPPORT_1__n0.png) · [实际FINAL两视角/截面](gallery/CLEAN_FINAL_POST__9fd5b67a2b20b5d1b19874039c8f3e05__LEG_SUPPORT_1__n0.png)

**决定：本轮固定幅度与预算已经结束，不晋级，也不自动延长。** 位置条件的用途响应有有限改善，下一轮若另获授权应保留这一证据；当前没有证据证明它同时解决主体完整性、非法冲突和开放边接续。不能用训练loss下降、单项冲突下降或单张好图代替稳定C采用与完整性；也不能把这次质量代价推广成所有平移增强均无效。本次收口不自动改变默认模型，不追加训练、样本或重新挑选checkpoint。

## 7. 文献依据和适用边界

冻结前的检索日期为2026-10-22，原记录保留于[LITERATURE_AND_ROUTE_CHECK.md](LITERATURE_AND_ROUTE_CHECK.md)。[PDR附录B.3](https://arxiv.org/html/2112.03530v4#A2.SS3)及[作者实现](https://github.com/ZhaoyangLyu/Point_Diffusion_Refinement)为partial/complete点云共同平移提供启发，但其重建任务、预算和GT评价不同，作者也报告增强可能降低测试性能；不能借其结论保证本任务收益。

[Conditional Prior Distribution](https://openreview.net/forum?id=Teh9Bq4giF)讨论条件初始分布，[Simba](https://arxiv.org/abs/2511.16161)讨论变换分布与级联细化。两者只作方法范围比较，本轮没有运行它们，也没有改变标准Gaussian或引入级联网络。本地Chair CREF、Bench扰动插值态/异步SEAM、早期FM加权均有既有结果；本轮的实质变化是共同移动C/Y终点且保持起点噪声不动，不把旧路线改名当新方法。

共同平移只增加位置变化，不增加独立结构/款式，也不证明原点偏置是唯一根因。本轮只评价原位固定开发请求，不能推出严格平移等变、未见大幅平移鲁棒性、跨类别泛化或独立论文测试性能。既有图审多次使用该面板，其统计与主观一致性边界必须保留。

## 8. 资源、源保护与交付

启动资源原记录：`NVIDIA GeForce RTX 5080, 16303 MiB, 1663 MiB, 14317 MiB, 21 %`；CPU 24核，可用RAM 61.30GiB。实际训练GPU worker=1；保留余量 2048MiB。训练状态耗时 2177.71秒，allocator reserved峰 3.137GiB；它不是整个进程VRAM峰。

训练设备总占用峰为 5259MiB，1015次、间隔2秒采样，含后台进程。本worker进程峰为NA（UNAVAILABLE_NVIDIA_SMI），不从allocator或权重大小补推。只授权一条额外训练臂，故GPU训练worker为1；采样保持单写者，CPU评分/渲染按资源并行，没有为填满显卡增加未授权训练或探针。

| workflow阶段 | 实际秒 |
| --- | --- |
| resource_release | 0.58 |
| training | 2185.32 |
| raw | 180.85 |
| D0 | 63.25 |
| score_raw | 64.92 |
| score_post | 269.94 |
| collect | 52.60 |
| gallery | 26.02 |

并发采样/评分的实际阶段记录保留于workflow；并发墙钟不能当独占吞吐。本收口只做CPU账本/hash/汇总，零模型调用、零评分调用、零新增采样。

源保护核对：42份冻结科学文件，147份继承数据/源资产保护记录，最终完整权重和推理导出hash均检查。此前清理已单独留存账本，本轮closeout不删任何资产。

完整终点：`training/JOINT_SHIFT_SEAM/checkpoint_step1500.pt`，SHA256 `536c6adf11736261ce9e31274f13f9ded86b3fb62d830024d2301f745d63ff82`。
推理导出：`training/JOINT_SHIFT_SEAM/generator_step1500.pt`，SHA256 `b78fb5bd67e682387154b815b1393a6096e3bc434f9ae4cfb64029e09840a440`。

数据/流：`data/manifest.json`、`data/schedule.json`、`data/exposure_plan.json`。原逐输出：`results/per_output.csv`；完整派生：`analysis_joint_shift_v1/`。144页固定截面：`gallery/`；432条实际记录及配对：`visual_review/`。

只读恢复核对：`python -B -m r2.pilots.bench_joint_shift_v1.resume`。独立收口核对：`python -B -m r2.analysis.bench_joint_shift_v1.closeout --check-only`。不要重启workflow重复生成。原workflow自动状态不回写，由独立 `COMPLETE.json` 引用实际图审、报告、原状态hash和交付审计。没有commit或push。
