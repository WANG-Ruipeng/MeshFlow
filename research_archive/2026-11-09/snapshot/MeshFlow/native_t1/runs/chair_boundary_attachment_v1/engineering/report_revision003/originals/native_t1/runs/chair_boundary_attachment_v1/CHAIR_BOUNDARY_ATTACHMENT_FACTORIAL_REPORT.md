# CHAIR 边界附着四格实验报告

**实际执行：INCOMPLETE；判定：COMPLETED_WITH_EVALUATION_GAP。**

| 臂 | 已返回optimizer更新 | UPDATE_DONE行 | 同输入流核验 | 训练状态 |
| --- | --- | --- | --- | --- |
| A_FM | 400 | 400 | 是 | TRAINING |
| B_BOUNDARY | 0 | 0 | NOT_RUN | NOT_RUN |
| C_SEAM | 0 | 0 | NOT_RUN | NOT_RUN |
| D_BOUNDARY_SEAM | 0 | 0 | NOT_RUN | NOT_RUN |

登记400请求。实际新生成0，缓存76，原不适用20，失败0，其余尚未完成。完整采样尝试2；逐输出NFE 100。初始化回放与正式评价分别记账。

预检/校准/成本合计尝试44次顶层前向、16次真实网络零更新反向；其中校准16次前向、32次输出leaf梯度。leaf梯度不计为网络反向。总顶层前向尝试3344/52196，真实调用失败0，pending 0。

三个主指标实际状态：{"SeamCov": {"PASS": 74, "NOT_APPLICABLE": 2}, "ConflictCov_C": {"PASS": 76}, "ExposedRetain": {"PARTIAL": 64, "PASS": 12}}。指标版本：chair_boundary_attachment_control_metrics_v1r1。计算完成与筛查通过分开判断。

科学配置偏离：预检收据记录未改变科学配置；最终保护哈希核验以独立收据为准。
CPU数值实现修订与日志/读写故障单列工程记录，不改记为新的模型试验。

## 起点、方法与一次校准

共同起点为OT_HYBRID pilot1000/total5000，SHA256：d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80。各臂恢复完整旧AdamW，固定增加1000更新至total6000，不挑best。A=FM；B=FM+C-only开放边界角向量；C=FM+解析UCorr；D=二者组合。向量仅768维且零初始化，标记来自C自身精确FP32边计数，不等同GT真实接口。

沿用32训练父、effective batch 8、microbatch 1、每批8个不同父；4个alpha=1、4个Uniform[0.9,1.1]。原free-only OT后混噪sqrt(0.75)·Z+0.5·epsilon，不再次OT。FM按整批有效free标量分母归一；BF16前向，FP32参数、几何与积分，推理固定Euler50。

UCorr采用对应边平方残差解析积分 (||r0||²+r0·r1+||r1||²)/3；边间等权，共享角按公式累加。仅t>=0.5，辅助项仍除以8。OT后监督身份映射另核验；GT端点与完整尺度只用于监督/评分，不进入推理bank。

校准状态PASS；lambda_seam=14.5296290122；有效状态16，有效父8。首批8父×t=0.5/0.8，lambda=0.25×输出梯度范数比中位数。它不是完整网络参数梯度比或验证集最优值。详见[lambda_calibration.json](lambda_calibration.json)。

## 主控制表

P1每父先平均全部4个候选，再对8父等权。缺任一父或候选显示NA，不改分母。SeamCov为带方向、投影支撑与CF资格的容差覆盖；Conflict为C冲突影响带并集；Exposure为固定方向GT参考外露保留。都不是拓扑、水密或通用可用率。

| 标签 | SeamCov@1% ↑ | ConflictCov_C ↓ | ExposedRetain ↑ |
| --- | --- | --- | --- |
| START_HYBRID | 0.113705 | 0.0492096 | 0.877495 |
| A_FM_END | NA | NA | NA |
| B_BOUNDARY_END | NA | NA | NA |
| C_SEAM_END | NA | NA | NA |
| D_BOUNDARY_SEAM_END | NA | NA | NA |

保守主值、可能上界、GT参考差值、LOW_SUPPORT与未决比例均保留。PARTIAL不自动失败：最坏/最好情形结论一致才作稳健判定，跨阈值则INCOMPLETE。

## 整体free质量与成本

| 标签 | D_sym2048 ↓ | 原512 MSE ↓ | 原512 p95/L0 ↓ | FF受影响 ↓ | CF proper ↓ | free面积/GT |
| --- | --- | --- | --- | --- | --- | --- |
| START_HYBRID | 0.00199829 | 0.00224297 | 0.0785602 | 0.915259 | 0.0693304 | 1.0263 |
| A_FM_END | NA | NA | NA | NA | NA | NA |
| B_BOUNDARY_END | NA | NA | NA | NA | NA | NA |
| C_SEAM_END | NA | NA | NA | NA | NA | NA |
| D_BOUNDARY_SEAM_END | NA | NA | NA | NA | NA | NA |

近退化、小角、越框、双向覆盖以及旧Gamma/仅距离AttachCov保留在[per_output_metrics.csv](per_output_metrics.csv)。局部指标改善不能替代主体质量。

| 臂 | 训练墙钟秒 | P1生成秒/例 | 设备采样峰值MiB | 进程峰值MiB |
| --- | --- | --- | --- | --- |
| A_FM | NA | NA | NA | NA |
| B_BOUNDARY | NA | NA | NA | NA |
| C_SEAM | NA | NA | NA | NA |
| D_BOUNDARY_SEAM | NA | NA | NA | NA |

固定任务同步成本如下。F+B与分段时间不能重复相加；allocated包含于reserved；设备显存包含桌面/其他进程，WSL进程显存没有数字时为NA。单GPU worker为合同要求，CPU可并行；并发墙钟不视为隔离性能优势。

| 臂 | 任务 | 重复 | F秒 | 完整F+B秒 | reserved GiB |
| --- | --- | --- | --- | --- | --- |
| A_FM | ordinary | 0 | 0.0874952 | 0.218184 | 2.53906 |
| A_FM | ordinary | 1 | 0.0733504 | 0.178968 | 2.53906 |
| A_FM | large | 0 | 0.0559621 | 0.150123 | 2.53906 |
| A_FM | large | 1 | 0.0970827 | 0.234801 | 2.53906 |
| B_BOUNDARY | ordinary | 0 | 0.0816534 | 0.19267 | 2.53906 |
| B_BOUNDARY | ordinary | 1 | 0.0385327 | 0.150396 | 2.53906 |
| B_BOUNDARY | large | 0 | 0.0448289 | 0.2006 | 2.53906 |
| B_BOUNDARY | large | 1 | 0.0665913 | 0.177273 | 2.53906 |
| C_SEAM | ordinary | 0 | 0.0963671 | 0.2114 | 2.53906 |
| C_SEAM | ordinary | 1 | 0.044365 | 0.163898 | 2.53906 |
| C_SEAM | large | 0 | 0.0393438 | 0.12784 | 2.53906 |
| C_SEAM | large | 1 | 0.0414502 | 0.161315 | 2.53906 |
| D_BOUNDARY_SEAM | ordinary | 0 | 0.0861008 | 0.179991 | 2.53906 |
| D_BOUNDARY_SEAM | ordinary | 1 | 0.042316 | 0.14877 | 2.53906 |
| D_BOUNDARY_SEAM | large | 0 | 0.0447076 | 0.165434 | 2.53906 |
| D_BOUNDARY_SEAM | large | 1 | 0.0658934 | 0.168558 | 2.53906 |

## 逐父结果与四格分解

每行包含父内全部候选；Best@4仅为附加展示，不能替代主均值，见[aggregation/P1_best_of_four_descriptive.csv](aggregation/P1_best_of_four_descriptive.csv)。

| 父UID | 标签 | SeamCov | ConflictCov_C | ExposedRetain | 已评分/适用 |
| --- | --- | --- | --- | --- | --- |
| 33ec1e64a02ff718faaadcdc0221eb72 | START_HYBRID | 0.0168498 | 0.00854492 | 0.996349 | 4/4 |
| 341bcae1e7f366a233d93f57729688db | START_HYBRID | 0.00428862 | 0.00720215 | 0.997973 | 4/4 |
| 534b0e6e2c53a8e1a7cdfd3874ed3723 | START_HYBRID | 0.0928939 | 0.119019 | 0.721817 | 4/4 |
| 866e7248434968051cc7750d0afca796 | START_HYBRID | 0.0764053 | 0.050293 | 0.959595 | 4/4 |
| 86ec9c796a6c1789e792598ed58dd1cc | START_HYBRID | 0.386919 | 0.00878906 | 0.996338 | 4/4 |
| bdd29e651e5f6fb2b079317292bdc5d4 | START_HYBRID | 0.126579 | 0.0257568 | 0.999512 | 4/4 |
| c1b8c2f0c4f3cc43aff7c08b060f5ed6 | START_HYBRID | 0.0756651 | 0.125854 | 0.90332 | 4/4 |
| f59627a077dcc2566c63b8b48495261a | START_HYBRID | 0.13004 | 0.0482178 | 0.445058 | 4/4 |
| 33ec1e64a02ff718faaadcdc0221eb72 | A_FM_END | NA | NA | NA | 0/4 |
| 341bcae1e7f366a233d93f57729688db | A_FM_END | NA | NA | NA | 0/4 |
| 534b0e6e2c53a8e1a7cdfd3874ed3723 | A_FM_END | NA | NA | NA | 0/4 |
| 866e7248434968051cc7750d0afca796 | A_FM_END | NA | NA | NA | 0/4 |
| 86ec9c796a6c1789e792598ed58dd1cc | A_FM_END | NA | NA | NA | 0/4 |
| bdd29e651e5f6fb2b079317292bdc5d4 | A_FM_END | NA | NA | NA | 0/4 |
| c1b8c2f0c4f3cc43aff7c08b060f5ed6 | A_FM_END | NA | NA | NA | 0/4 |
| f59627a077dcc2566c63b8b48495261a | A_FM_END | NA | NA | NA | 0/4 |
| 33ec1e64a02ff718faaadcdc0221eb72 | B_BOUNDARY_END | NA | NA | NA | 0/4 |
| 341bcae1e7f366a233d93f57729688db | B_BOUNDARY_END | NA | NA | NA | 0/4 |
| 534b0e6e2c53a8e1a7cdfd3874ed3723 | B_BOUNDARY_END | NA | NA | NA | 0/4 |
| 866e7248434968051cc7750d0afca796 | B_BOUNDARY_END | NA | NA | NA | 0/4 |
| 86ec9c796a6c1789e792598ed58dd1cc | B_BOUNDARY_END | NA | NA | NA | 0/4 |
| bdd29e651e5f6fb2b079317292bdc5d4 | B_BOUNDARY_END | NA | NA | NA | 0/4 |
| c1b8c2f0c4f3cc43aff7c08b060f5ed6 | B_BOUNDARY_END | NA | NA | NA | 0/4 |
| f59627a077dcc2566c63b8b48495261a | B_BOUNDARY_END | NA | NA | NA | 0/4 |
| 33ec1e64a02ff718faaadcdc0221eb72 | C_SEAM_END | NA | NA | NA | 0/4 |
| 341bcae1e7f366a233d93f57729688db | C_SEAM_END | NA | NA | NA | 0/4 |
| 534b0e6e2c53a8e1a7cdfd3874ed3723 | C_SEAM_END | NA | NA | NA | 0/4 |
| 866e7248434968051cc7750d0afca796 | C_SEAM_END | NA | NA | NA | 0/4 |
| 86ec9c796a6c1789e792598ed58dd1cc | C_SEAM_END | NA | NA | NA | 0/4 |
| bdd29e651e5f6fb2b079317292bdc5d4 | C_SEAM_END | NA | NA | NA | 0/4 |
| c1b8c2f0c4f3cc43aff7c08b060f5ed6 | C_SEAM_END | NA | NA | NA | 0/4 |
| f59627a077dcc2566c63b8b48495261a | C_SEAM_END | NA | NA | NA | 0/4 |
| 33ec1e64a02ff718faaadcdc0221eb72 | D_BOUNDARY_SEAM_END | NA | NA | NA | 0/4 |
| 341bcae1e7f366a233d93f57729688db | D_BOUNDARY_SEAM_END | NA | NA | NA | 0/4 |
| 534b0e6e2c53a8e1a7cdfd3874ed3723 | D_BOUNDARY_SEAM_END | NA | NA | NA | 0/4 |
| 866e7248434968051cc7750d0afca796 | D_BOUNDARY_SEAM_END | NA | NA | NA | 0/4 |
| 86ec9c796a6c1789e792598ed58dd1cc | D_BOUNDARY_SEAM_END | NA | NA | NA | 0/4 |
| bdd29e651e5f6fb2b079317292bdc5d4 | D_BOUNDARY_SEAM_END | NA | NA | NA | 0/4 |
| c1b8c2f0c4f3cc43aff7c08b060f5ed6 | D_BOUNDARY_SEAM_END | NA | NA | NA | 0/4 |
| f59627a077dcc2566c63b8b48495261a | D_BOUNDARY_SEAM_END | NA | NA | NA | 0/4 |

D−A、B−A、C−A、D−B、D−C逐父配对保存在[aggregation/paired_parent_differences.csv](aggregation/paired_parent_differences.csv)。下表为同条件、同噪声D−B−C+A，再按父平均；符号须结合指标方向解释。

| 父UID | 指标 | D−B−C+A | 有效/登记 |
| --- | --- | --- | --- |
| 33ec1e64a02ff718faaadcdc0221eb72 | SeamCov | NA | 0/4 |
| 33ec1e64a02ff718faaadcdc0221eb72 | ConflictCov_C | NA | 0/4 |
| 33ec1e64a02ff718faaadcdc0221eb72 | ExposedRetain | NA | 0/4 |
| 33ec1e64a02ff718faaadcdc0221eb72 | D_sym2048 | NA | 0/4 |
| 341bcae1e7f366a233d93f57729688db | SeamCov | NA | 0/4 |
| 341bcae1e7f366a233d93f57729688db | ConflictCov_C | NA | 0/4 |
| 341bcae1e7f366a233d93f57729688db | ExposedRetain | NA | 0/4 |
| 341bcae1e7f366a233d93f57729688db | D_sym2048 | NA | 0/4 |
| 534b0e6e2c53a8e1a7cdfd3874ed3723 | SeamCov | NA | 0/4 |
| 534b0e6e2c53a8e1a7cdfd3874ed3723 | ConflictCov_C | NA | 0/4 |
| 534b0e6e2c53a8e1a7cdfd3874ed3723 | ExposedRetain | NA | 0/4 |
| 534b0e6e2c53a8e1a7cdfd3874ed3723 | D_sym2048 | NA | 0/4 |
| 866e7248434968051cc7750d0afca796 | SeamCov | NA | 0/4 |
| 866e7248434968051cc7750d0afca796 | ConflictCov_C | NA | 0/4 |
| 866e7248434968051cc7750d0afca796 | ExposedRetain | NA | 0/4 |
| 866e7248434968051cc7750d0afca796 | D_sym2048 | NA | 0/4 |
| 86ec9c796a6c1789e792598ed58dd1cc | SeamCov | NA | 0/4 |
| 86ec9c796a6c1789e792598ed58dd1cc | ConflictCov_C | NA | 0/4 |
| 86ec9c796a6c1789e792598ed58dd1cc | ExposedRetain | NA | 0/4 |
| 86ec9c796a6c1789e792598ed58dd1cc | D_sym2048 | NA | 0/4 |
| bdd29e651e5f6fb2b079317292bdc5d4 | SeamCov | NA | 0/4 |
| bdd29e651e5f6fb2b079317292bdc5d4 | ConflictCov_C | NA | 0/4 |
| bdd29e651e5f6fb2b079317292bdc5d4 | ExposedRetain | NA | 0/4 |
| bdd29e651e5f6fb2b079317292bdc5d4 | D_sym2048 | NA | 0/4 |
| c1b8c2f0c4f3cc43aff7c08b060f5ed6 | SeamCov | NA | 0/4 |
| c1b8c2f0c4f3cc43aff7c08b060f5ed6 | ConflictCov_C | NA | 0/4 |
| c1b8c2f0c4f3cc43aff7c08b060f5ed6 | ExposedRetain | NA | 0/4 |
| c1b8c2f0c4f3cc43aff7c08b060f5ed6 | D_sym2048 | NA | 0/4 |
| f59627a077dcc2566c63b8b48495261a | SeamCov | NA | 0/4 |
| f59627a077dcc2566c63b8b48495261a | ConflictCov_C | NA | 0/4 |
| f59627a077dcc2566c63b8b48495261a | ExposedRetain | NA | 0/4 |
| f59627a077dcc2566c63b8b48495261a | D_sym2048 | NA | 0/4 |

这只是单训练seed、同一配对训练流下的描述性因子分析，不是多种子显著性或普遍因果结论。

## A相对START：继续FM本身的作用

A不是新增附件方法，不能只看B/C/D gate后自动丢弃它。下表直接量化A相对START；A的接缝增量区间为[NA, NA]，质量/冲突/外露成本状态为INCOMPLETE。对A的保留建议是描述性判断，不冒称B/C/D相对A的预注册gate。

| 指标 | START | A | A−START |
| --- | --- | --- | --- |
| SeamCov | 0.113705 | NA | NA |
| ConflictCov_C | 0.0492096 | NA | NA |
| ExposedRetain | 0.877495 | NA | NA |
| D_sym2048 | 0.00199829 | NA | NA |
| fixed_query_MSE | 0.00224297 | NA | NA |
| fixed_query_p95_L | 0.0785602 | NA | NA |
| FF_illegal_affected_free_fraction | 0.915259 | NA | NA |
| CF_proper_affected_free_fraction | 0.0693304 | NA | NA |
| free_area_over_GT_free_area | 1.0263 | NA | NA |

## 五面板与敏感性

| 面板 | 每标签登记 | 原NA | 用途 |
| --- | --- | --- | --- |
| P1 | 32 | 0 | 8验证父×4候选，主结论 |
| P2 | 8 | 0 | 前4训练父×2噪声 |
| P3 | 16 | 4 | 全部8验证父A40，原两父NA保留 |
| P4 | 16 | 0 | 已观察留出前4父×2C×2噪声 |
| P5 | 8 | 0 | 快测两父C2/C3×2噪声，C1不重生 |

| 面板 | 标签 | SeamCov | ConflictCov_C | ExposedRetain |
| --- | --- | --- | --- | --- |
| P2 | START_HYBRID | 0.52976 | 0.0428467 | 0.93644 |
| P2 | A_FM_END | NA | NA | NA |
| P2 | B_BOUNDARY_END | NA | NA | NA |
| P2 | C_SEAM_END | NA | NA | NA |
| P2 | D_BOUNDARY_SEAM_END | NA | NA | NA |
| P3 | START_HYBRID | 0.243121 | 0.0233968 | 0.988903 |
| P3 | A_FM_END | NA | NA | NA |
| P3 | B_BOUNDARY_END | NA | NA | NA |
| P3 | C_SEAM_END | NA | NA | NA |
| P3 | D_BOUNDARY_SEAM_END | NA | NA | NA |
| P4 | START_HYBRID | 0.0227215 | 0.0227966 | 0.940164 |
| P4 | A_FM_END | NA | NA | NA |
| P4 | B_BOUNDARY_END | NA | NA | NA |
| P4 | C_SEAM_END | NA | NA | NA |
| P4 | D_BOUNDARY_SEAM_END | NA | NA | NA |
| P5 | START_HYBRID | NA | 0.0253296 | 0.984327 |
| P5 | A_FM_END | NA | NA | NA |
| P5 | B_BOUNDARY_END | NA | NA | NA |
| P5 | C_SEAM_END | NA | NA | NA |
| P5 | D_BOUNDARY_SEAM_END | NA | NA | NA |

P5原341b…C3整体Gamma歧义继续NA，仅接口项不适用；冲突、外露和整体质量继续计算。不能用局部好边替代完整Gamma。共同补集Y_common的固定查询只量到free，C1/C2/C3比较见[aggregation/nested_context_common_support.csv](aggregation/nested_context_common_support.csv)。

密度敏感性只限首两P1父候选0×START/D四单元：Gamma64→256，C2048→8192，方向26→74。不替换未完成的D。实际差异见[aggregation/sensitivity.csv](aggregation/sensitivity.csv)。

## 七个问题逐答

1. **哪个方案真正增加合理接缝，而非只缩小最近距离？** 当前没有方案通过完整SeamCov筛查。仍有缺失或影响判断的不确定性，不能据此作普遍否定。该筛查同时约束有限边距离、切向、投影支撑和对应known邻面CF资格；仅距离AttachCov保留为不同诊断，不能替代合理接合。

2. **是否增加C的重复覆盖或proper穿越？** B_BOUNDARY_END：相对A代价[NA, NA]，INCOMPLETE，已证实严重父例0；C_SEAM_END：相对A代价[NA, NA]，INCOMPLETE，已证实严重父例0；D_BOUNDARY_SEAM_END：相对A代价[NA, NA]，INCOMPLETE，已证实严重父例0。主值为cross/dup并集，分项、交集及GT参考有符号差异保留在逐例CSV。不能以事件数减少替代影响带面积比例，也不能把未决项当0。

3. **C是否仍作为参考外露表面参与最终物体？** B_BOUNDARY_END：相对A代价[NA, NA]，INCOMPLETE，已证实严重父例0；C_SEAM_END：相对A代价[NA, NA]，INCOMPLETE，已证实严重父例0；D_BOUNDARY_SEAM_END：相对A代价[NA, NA]，INCOMPLETE，已证实严重父例0。这里“代价”为外露下降；坐标硬保留不能保证外露。判据只针对GT至少一个登记方向可见的点，不要求内部面外露；重合free命中保守处理。各标签值见主表，逐点见第三叠图。

4. **是否保住START与等预算A的整体free质量？** B_BOUNDARY_END联合代价=INCOMPLETE；C_SEAM_END联合代价=INCOMPLETE；D_BOUNDARY_SEAM_END联合代价=INCOMPLETE。筛查分别比较两参照：D_sym2048/原512 MSE最多+5%，p95最多+10%，FF受影响比例最多+0.02，并要求N/C/finite无新增错误。明细见[aggregation/summary.json](aggregation/summary.json)。

5. **边界标记与UCorr分别贡献什么，组合是否必要？** B−A和C−A是两个单组件净变化，D−B/D−C与D−B−C+A检验额外组合价值；完整逐父数字已列。当前建议NO_PROMOTION_PENDING_EVIDENCE。若B或C已满足全部约束，先保留简单组件；不得凭D训练loss低宣布组合必要。

6. **单片与多片是否一致，还是局部受益？** 当前P5已评分8/40。P1/P5分开报告；P5仅两父且原C3有整体Gamma NA，因此不能自动宣称跨单片/多片一致。共同补集、冲突和外露需一起核对，局部好例不外推。

7. **值得保留A/B/C/D中的哪个，或没有新方案晋级？** 现有证据或不确定性不足以支持晋级。 建议记录为 NO_PROMOTION_PENDING_EVIDENCE，不自动修改默认模型。

| 标签 | 接缝筛查 | 联合代价 | 分类 |
| --- | --- | --- | --- |
| B_BOUNDARY_END | INCOMPLETE | INCOMPLETE | COMPLETED_WITH_EVALUATION_GAP |
| C_SEAM_END | INCOMPLETE | INCOMPLETE | COMPLETED_WITH_EVALUATION_GAP |
| D_BOUNDARY_SEAM_END | INCOMPLETE | INCOMPLETE | COMPLETED_WITH_EVALUATION_GAP |

## 限制、恢复与唯一优先下一步

三个控制量只覆盖冻结容差、离散方向与有限面板，不保证水密、拓扑、穿透体积或通用可用率。等边UCorr对细分和接口划分敏感，且与FM坐标监督重合。一次梯度校准不能证明生成收益。P4已被观察，P5仅两父；400请求不是400独立对象。

预检成本A首测未额外warmup，不据配对计时声称速度优势。正式训练墙钟也受并行CPU工作影响。设备采样峰值含桌面；WDDM/WSL的进程显存不可得时明确NA，不能拿设备峰值冒充模型独占峰值。

**唯一优先下一步：** 优先补齐影响结论的独立CPU评分或身份核验；使用已有raw，不追加生成。 **检验假说：** 补齐核验或收紧保守区间后，同一固定面板可得到不依赖删除困难父例的结论。 本报告不执行该下一步，不自动训练、改loss、扫描求解器、换默认模型或恢复MOMENT/SURFACE。

首次登记字段路由故障与修复保留原件、traceback/失败说明、diff。CPU数值revision002保留旧评分和派生表；阈值不变，极瘦射线不定项改为保守UNKNOWN，无效C法向的参考域采用保守上下界。新版只重评分同一raw；旧预检、校准、回放调用全部仍计账。详见[engineering/](engineering/)及[scoring/cpu_ledger.jsonl](scoring/cpu_ledger.jsonl)。

调用和pending/duplicate核对：[reporting_snapshot.json](reporting_snapshot.json)；起终点实际文件hash及完整恢复元数据：[checkpoint_manifest.json](checkpoint_manifest.json)。最终公共源码保护hash、实际源码与code.diff由主会话独立收据补齐，不把占位链接当完成，见[DELIVERY_INDEX.md](DELIVERY_INDEX.md)。

图库：[P1全部160输出](gallery/index.html)、[P5与三个叠图](gallery/nested.html)。原三角不修补、不焊接，C/free固定颜色，同父统一框和光照。

固定任务完成即停止。
