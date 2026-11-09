# CHAIR 边界附着四格实验报告

**实际执行：COMPLETE；判定：NO_ATTACHMENT_SIGNAL。**

| 臂 | 已返回optimizer更新 | UPDATE_DONE行 | 同输入流核验 | 训练状态 |
| --- | --- | --- | --- | --- |
| A_FM | 1000 | 1000 | 是 | DONE |
| B_BOUNDARY | 1000 | 1000 | 是 | DONE |
| C_SEAM | 1000 | 1000 | 是 | DONE |
| D_BOUNDARY_SEAM | 1000 | 1000 | 是 | DONE |

登记400请求。实际新生成304，缓存76，原不适用20，失败0，其余未完成0。完整采样尝试306；逐输出NFE 15300。初始化回放与正式评价分别记账。

预检/校准/成本合计尝试44次顶层前向、16次真实网络零更新反向；其中校准16次前向、32次输出leaf梯度。leaf梯度不计为网络反向。总顶层前向尝试47344/52196，真实调用失败0，pending 0。

三个主指标实际状态：{"SeamCov": {"PASS": 370, "NOT_APPLICABLE": 10}, "ConflictCov_C": {"PASS": 380}, "ExposedRetain": {"PARTIAL": 320, "PASS": 60}}。指标版本：chair_boundary_attachment_control_metrics_v1r1。计算完成与筛查通过分开判断。

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
| A_FM_END | 0.0931057 | 0.0384064 | 0.865697 |
| B_BOUNDARY_END | 0.0887556 | 0.046875 | 0.869893 |
| C_SEAM_END | 0.093804 | 0.0464325 | 0.865076 |
| D_BOUNDARY_SEAM_END | 0.0873807 | 0.042038 | 0.874543 |

保守主值、可能上界、GT参考差值、LOW_SUPPORT与未决比例均保留。PARTIAL不自动失败：最坏/最好情形结论一致才作稳健判定，跨阈值则INCOMPLETE。

## 整体free质量与成本

| 标签 | D_sym2048 ↓ | 原512 MSE ↓ | 原512 p95/L0 ↓ | FF受影响 ↓ | CF proper ↓ | free面积/GT |
| --- | --- | --- | --- | --- | --- | --- |
| START_HYBRID | 0.00199829 | 0.00224297 | 0.0785602 | 0.915259 | 0.0693304 | 1.0263 |
| A_FM_END | 0.0018554 | 0.00213167 | 0.0765122 | 0.911921 | 0.0680817 | 0.978413 |
| B_BOUNDARY_END | 0.00182677 | 0.00206456 | 0.0755451 | 0.90758 | 0.0691614 | 0.965485 |
| C_SEAM_END | 0.00180514 | 0.00204624 | 0.0749191 | 0.910928 | 0.0646693 | 0.976936 |
| D_BOUNDARY_SEAM_END | 0.00180992 | 0.00206976 | 0.0748295 | 0.910451 | 0.0654674 | 0.973108 |

近退化、小角、越框、双向覆盖以及旧Gamma/仅距离AttachCov保留在[per_output_metrics.csv](per_output_metrics.csv)。局部指标改善不能替代主体质量。

| 臂 | 训练墙钟秒 | P1生成秒/例 | 训练设备占用峰值MiB | 进程峰值MiB |
| --- | --- | --- | --- | --- |
| A_FM | 1549.29 | 2.68192 | 7015 | NA |
| B_BOUNDARY | 1531.63 | 2.99488 | 6605 | NA |
| C_SEAM | 1504.75 | 2.83324 | 6605 | NA |
| D_BOUNDARY_SEAM | 1522.05 | 2.94957 | 6615 | NA |

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
| 33ec1e64a02ff718faaadcdc0221eb72 | A_FM_END | 0.00562967 | 0.00402832 | 0.996098 | 4/4 |
| 341bcae1e7f366a233d93f57729688db | A_FM_END | 0 | 0.00610352 | 0.998606 | 4/4 |
| 534b0e6e2c53a8e1a7cdfd3874ed3723 | A_FM_END | 0.0398647 | 0.0408936 | 0.72383 | 4/4 |
| 866e7248434968051cc7750d0afca796 | A_FM_END | 0.143237 | 0.0354004 | 0.983276 | 4/4 |
| 86ec9c796a6c1789e792598ed58dd1cc | A_FM_END | 0.255606 | 0.00793457 | 0.98291 | 4/4 |
| bdd29e651e5f6fb2b079317292bdc5d4 | A_FM_END | 0.149582 | 0.0324707 | 0.988892 | 4/4 |
| c1b8c2f0c4f3cc43aff7c08b060f5ed6 | A_FM_END | 0.0666613 | 0.146362 | 0.904175 | 4/4 |
| f59627a077dcc2566c63b8b48495261a | A_FM_END | 0.0842648 | 0.0340576 | 0.347792 | 4/4 |
| 33ec1e64a02ff718faaadcdc0221eb72 | B_BOUNDARY_END | 0.00270919 | 0.0032959 | 0.995594 | 4/4 |
| 341bcae1e7f366a233d93f57729688db | B_BOUNDARY_END | 0 | 0.00671387 | 0.997339 | 4/4 |
| 534b0e6e2c53a8e1a7cdfd3874ed3723 | B_BOUNDARY_END | 0.0382308 | 0.0362549 | 0.724082 | 4/4 |
| 866e7248434968051cc7750d0afca796 | B_BOUNDARY_END | 0.167672 | 0.0339355 | 0.9823 | 4/4 |
| 86ec9c796a6c1789e792598ed58dd1cc | B_BOUNDARY_END | 0.239789 | 0.00842285 | 0.983521 | 4/4 |
| bdd29e651e5f6fb2b079317292bdc5d4 | B_BOUNDARY_END | 0.150526 | 0.0332031 | 0.991943 | 4/4 |
| c1b8c2f0c4f3cc43aff7c08b060f5ed6 | B_BOUNDARY_END | 0.0622566 | 0.138916 | 0.922119 | 4/4 |
| f59627a077dcc2566c63b8b48495261a | B_BOUNDARY_END | 0.0488607 | 0.114258 | 0.36225 | 4/4 |
| 33ec1e64a02ff718faaadcdc0221eb72 | C_SEAM_END | 0.00490292 | 0.00305176 | 0.995468 | 4/4 |
| 341bcae1e7f366a233d93f57729688db | C_SEAM_END | 0 | 0.00549316 | 0.997592 | 4/4 |
| 534b0e6e2c53a8e1a7cdfd3874ed3723 | C_SEAM_END | 0.0502926 | 0.0430908 | 0.72773 | 4/4 |
| 866e7248434968051cc7750d0afca796 | C_SEAM_END | 0.129784 | 0.0406494 | 0.985229 | 4/4 |
| 86ec9c796a6c1789e792598ed58dd1cc | C_SEAM_END | 0.248114 | 0.00939941 | 0.982788 | 4/4 |
| bdd29e651e5f6fb2b079317292bdc5d4 | C_SEAM_END | 0.155775 | 0.0368652 | 0.988281 | 4/4 |
| c1b8c2f0c4f3cc43aff7c08b060f5ed6 | C_SEAM_END | 0.0829869 | 0.139771 | 0.906372 | 4/4 |
| f59627a077dcc2566c63b8b48495261a | C_SEAM_END | 0.078576 | 0.0931396 | 0.337145 | 4/4 |
| 33ec1e64a02ff718faaadcdc0221eb72 | D_BOUNDARY_SEAM_END | 0.00641385 | 0.00390625 | 0.996098 | 4/4 |
| 341bcae1e7f366a233d93f57729688db | D_BOUNDARY_SEAM_END | 0 | 0.0057373 | 0.997719 | 4/4 |
| 534b0e6e2c53a8e1a7cdfd3874ed3723 | D_BOUNDARY_SEAM_END | 0.0464538 | 0.0360107 | 0.719678 | 4/4 |
| 866e7248434968051cc7750d0afca796 | D_BOUNDARY_SEAM_END | 0.144231 | 0.0297852 | 0.985962 | 4/4 |
| 86ec9c796a6c1789e792598ed58dd1cc | D_BOUNDARY_SEAM_END | 0.243588 | 0.00866699 | 0.983765 | 4/4 |
| bdd29e651e5f6fb2b079317292bdc5d4 | D_BOUNDARY_SEAM_END | 0.149349 | 0.0308838 | 0.990479 | 4/4 |
| c1b8c2f0c4f3cc43aff7c08b060f5ed6 | D_BOUNDARY_SEAM_END | 0.0768984 | 0.147583 | 0.911499 | 4/4 |
| f59627a077dcc2566c63b8b48495261a | D_BOUNDARY_SEAM_END | 0.0321119 | 0.0737305 | 0.411146 | 4/4 |

D−A、B−A、C−A、D−B、D−C逐父配对保存在[aggregation/paired_parent_differences.csv](aggregation/paired_parent_differences.csv)。下表为同条件、同噪声D−B−C+A，再按父平均；符号须结合指标方向解释。

| 父UID | 指标 | D−B−C+A | 有效/登记 |
| --- | --- | --- | --- |
| 33ec1e64a02ff718faaadcdc0221eb72 | SeamCov | 0.00443142 | 4/4 |
| 33ec1e64a02ff718faaadcdc0221eb72 | ConflictCov_C | 0.00158691 | 4/4 |
| 33ec1e64a02ff718faaadcdc0221eb72 | ExposedRetain | 0.00113293 | 4/4 |
| 33ec1e64a02ff718faaadcdc0221eb72 | D_sym2048 | -0.000108016 | 4/4 |
| 341bcae1e7f366a233d93f57729688db | SeamCov | 0 | 4/4 |
| 341bcae1e7f366a233d93f57729688db | ConflictCov_C | -0.000366211 | 4/4 |
| 341bcae1e7f366a233d93f57729688db | ExposedRetain | 0.00139382 | 4/4 |
| 341bcae1e7f366a233d93f57729688db | D_sym2048 | 3.15936e-05 | 4/4 |
| 534b0e6e2c53a8e1a7cdfd3874ed3723 | SeamCov | -0.00220489 | 4/4 |
| 534b0e6e2c53a8e1a7cdfd3874ed3723 | ConflictCov_C | -0.00244141 | 4/4 |
| 534b0e6e2c53a8e1a7cdfd3874ed3723 | ExposedRetain | -0.00830398 | 4/4 |
| 534b0e6e2c53a8e1a7cdfd3874ed3723 | D_sym2048 | -9.86188e-06 | 4/4 |
| 866e7248434968051cc7750d0afca796 | SeamCov | -0.00998788 | 4/4 |
| 866e7248434968051cc7750d0afca796 | ConflictCov_C | -0.00939941 | 4/4 |
| 866e7248434968051cc7750d0afca796 | ExposedRetain | 0.00170898 | 4/4 |
| 866e7248434968051cc7750d0afca796 | D_sym2048 | 1.77616e-05 | 4/4 |
| 86ec9c796a6c1789e792598ed58dd1cc | SeamCov | 0.0112903 | 4/4 |
| 86ec9c796a6c1789e792598ed58dd1cc | ConflictCov_C | -0.0012207 | 4/4 |
| 86ec9c796a6c1789e792598ed58dd1cc | ExposedRetain | 0.000366211 | 4/4 |
| 86ec9c796a6c1789e792598ed58dd1cc | D_sym2048 | 6.81032e-05 | 4/4 |
| bdd29e651e5f6fb2b079317292bdc5d4 | SeamCov | -0.00736976 | 4/4 |
| bdd29e651e5f6fb2b079317292bdc5d4 | ConflictCov_C | -0.00671387 | 4/4 |
| bdd29e651e5f6fb2b079317292bdc5d4 | ExposedRetain | -0.000854492 | 4/4 |
| bdd29e651e5f6fb2b079317292bdc5d4 | D_sym2048 | 4.90457e-05 | 4/4 |
| c1b8c2f0c4f3cc43aff7c08b060f5ed6 | SeamCov | -0.00168386 | 4/4 |
| c1b8c2f0c4f3cc43aff7c08b060f5ed6 | ConflictCov_C | 0.0152588 | 4/4 |
| c1b8c2f0c4f3cc43aff7c08b060f5ed6 | ExposedRetain | -0.0128174 | 4/4 |
| c1b8c2f0c4f3cc43aff7c08b060f5ed6 | D_sym2048 | 0.000123683 | 4/4 |
| f59627a077dcc2566c63b8b48495261a | SeamCov | -0.01106 | 4/4 |
| f59627a077dcc2566c63b8b48495261a | ConflictCov_C | -0.0996094 | 4/4 |
| f59627a077dcc2566c63b8b48495261a | ExposedRetain | 0.0595426 | 4/4 |
| f59627a077dcc2566c63b8b48495261a | D_sym2048 | 9.49799e-05 | 4/4 |

这只是单训练seed、同一配对训练流下的描述性因子分析，不是多种子显著性或普遍因果结论。

## A相对START：继续FM本身的作用

A不是新增附件方法，不能只看B/C/D gate后自动丢弃它。下表直接量化A相对START；A的接缝增量区间为[-0.0205993, -0.0205993]，质量/冲突/外露成本状态为PASS。对A的保留建议是描述性判断，不冒称B/C/D相对A的预注册gate。

| 指标 | START | A | A−START |
| --- | --- | --- | --- |
| SeamCov | 0.113705 | 0.0931057 | -0.0205993 |
| ConflictCov_C | 0.0492096 | 0.0384064 | -0.0108032 |
| ExposedRetain | 0.877495 | 0.865697 | -0.0117979 |
| D_sym2048 | 0.00199829 | 0.0018554 | -0.00014288 |
| fixed_query_MSE | 0.00224297 | 0.00213167 | -0.000111297 |
| fixed_query_p95_L | 0.0785602 | 0.0765122 | -0.00204796 |
| FF_illegal_affected_free_fraction | 0.915259 | 0.911921 | -0.00333736 |
| CF_proper_affected_free_fraction | 0.0693304 | 0.0680817 | -0.00124864 |
| free_area_over_GT_free_area | 1.0263 | 0.978413 | -0.0478863 |

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
| P2 | A_FM_END | 0.499653 | 0.062561 | 0.949821 |
| P2 | B_BOUNDARY_END | 0.498509 | 0.0622559 | 0.949149 |
| P2 | C_SEAM_END | 0.492011 | 0.0604248 | 0.95049 |
| P2 | D_BOUNDARY_SEAM_END | 0.491233 | 0.0617676 | 0.948659 |
| P3 | START_HYBRID | 0.243121 | 0.0233968 | 0.988903 |
| P3 | A_FM_END | 0.231482 | 0.0208333 | 0.99046 |
| P3 | B_BOUNDARY_END | 0.208533 | 0.0220947 | 0.990718 |
| P3 | C_SEAM_END | 0.215929 | 0.0224609 | 0.990714 |
| P3 | D_BOUNDARY_SEAM_END | 0.205091 | 0.0234782 | 0.991424 |
| P4 | START_HYBRID | 0.0227215 | 0.0227966 | 0.940164 |
| P4 | A_FM_END | 0.0354498 | 0.0308228 | 0.94854 |
| P4 | B_BOUNDARY_END | 0.0364631 | 0.0310669 | 0.947092 |
| P4 | C_SEAM_END | 0.0394597 | 0.0323792 | 0.947873 |
| P4 | D_BOUNDARY_SEAM_END | 0.028344 | 0.0336914 | 0.9471 |
| P5 | START_HYBRID | NA | 0.0253296 | 0.984327 |
| P5 | A_FM_END | NA | 0.0248413 | 0.977149 |
| P5 | B_BOUNDARY_END | NA | 0.0283813 | 0.977128 |
| P5 | C_SEAM_END | NA | 0.0303345 | 0.977094 |
| P5 | D_BOUNDARY_SEAM_END | NA | 0.0290527 | 0.977094 |

P5原341b…C3整体Gamma歧义继续NA，仅接口项不适用；冲突、外露和整体质量继续计算。不能用局部好边替代完整Gamma。共同补集Y_common的固定查询只量到free，C1/C2/C3比较见[aggregation/nested_context_common_support.csv](aggregation/nested_context_common_support.csv)。

密度敏感性只限首两P1父候选0×START/D四单元：Gamma64→256，C2048→8192，方向26→74。不替换未完成的D。实际差异见[aggregation/sensitivity.csv](aggregation/sensitivity.csv)。

## 七个问题逐答

1. **哪个方案真正增加合理接缝，而非只缩小最近距离？** 当前没有方案通过完整SeamCov筛查。该筛查同时约束有限边距离、切向、投影支撑和对应known邻面CF资格；仅距离AttachCov保留为不同诊断，不能替代合理接合。

2. **是否增加C的重复覆盖或proper穿越？** B_BOUNDARY_END：相对A代价[0.00846863, 0.00846863]，PASS，已证实严重父例0；C_SEAM_END：相对A代价[0.00802612, 0.00802612]，PASS，已证实严重父例0；D_BOUNDARY_SEAM_END：相对A代价[0.00363159, 0.00363159]，PASS，已证实严重父例0。主值为cross/dup并集，分项、交集及GT参考有符号差异保留在逐例CSV。不能以事件数减少替代影响带面积比例，也不能把未决项当0。

3. **C是否仍作为参考外露表面参与最终物体？** B_BOUNDARY_END：相对A代价[-0.00424429, -0.00417974]，PASS，已证实严重父例0；C_SEAM_END：相对A代价[0.000589743, 0.000637862]，PASS，已证实严重父例0；D_BOUNDARY_SEAM_END：相对A代价[-0.00887634, -0.00882939]，PASS，已证实严重父例0。这里“代价”为外露下降；坐标硬保留不能保证外露。判据只针对GT至少一个登记方向可见的点，不要求内部面外露；重合free命中保守处理。各标签值见主表，逐点见第三叠图。

4. **是否保住START与等预算A的整体free质量？** B_BOUNDARY_END联合代价=PASS；C_SEAM_END联合代价=FAIL；D_BOUNDARY_SEAM_END联合代价=PASS。筛查分别比较两参照：D_sym2048/原512 MSE最多+5%，p95最多+10%，FF受影响比例最多+0.02，并要求N/C/finite无新增错误。明细见[aggregation/summary.json](aggregation/summary.json)。

5. **边界标记与UCorr分别贡献什么，组合是否必要？** B−A和C−A是两个单组件净变化，D−B/D−C与D−B−C+A检验额外组合价值；完整逐父数字已列。当前建议START_HYBRID。若B或C已满足全部约束，先保留简单组件；不得凭D训练loss低宣布组合必要。

6. **单片与多片是否一致，还是局部受益？** 当前P5已评分40/40。P1/P5分开报告；P5仅两父且原C3有整体Gamma NA，因此不能自动宣称跨单片/多片一致。共同补集、冲突和外露需一起核对，局部好例不外推。

7. **值得保留A/B/C/D中的哪个，或没有新方案晋级？** 本轮没有值得晋级的新方案；保留START，A保留为等预算对照证据。 建议记录为 START_HYBRID，不自动修改默认模型。

| 标签 | 接缝筛查 | 联合代价 | 分类 |
| --- | --- | --- | --- |
| B_BOUNDARY_END | FAIL | PASS | NO_ATTACHMENT_SIGNAL |
| C_SEAM_END | FAIL | FAIL | NO_ATTACHMENT_SIGNAL |
| D_BOUNDARY_SEAM_END | FAIL | PASS | NO_ATTACHMENT_SIGNAL |

## 限制、恢复与唯一优先下一步

三个控制量只覆盖冻结容差、离散方向与有限面板，不保证水密、拓扑、穿透体积或通用可用率。等边UCorr对细分和接口划分敏感，且与FM坐标监督重合。一次梯度校准不能证明生成收益。P4已被观察，P5仅两父；400请求不是400独立对象。

预检成本A首测未额外warmup，不据配对计时声称速度优势。正式训练墙钟也受并行CPU工作影响。设备采样峰值含桌面；WDDM/WSL的进程显存不可得时明确NA，不能拿设备峰值冒充模型独占峰值。

**唯一优先下一步：** 优先离线分解未覆盖Gamma：无近邻边、切向不合格、投影区间不足或CF冲突。 **检验假说：** 剩余失败主要由有限边方向、支撑区间或冲突资格限制，而非仅最近距离过大；先验证主导类别再提出改动。 本报告不执行该下一步，不自动训练、改loss、扫描求解器、换默认模型或恢复MOMENT/SURFACE。

首次登记字段路由故障与修复保留原件、traceback/失败说明、diff。CPU数值revision002保留旧评分和派生表；阈值不变，极瘦射线不定项改为保守UNKNOWN，无效C法向的参考域采用保守上下界。新版只重评分同一raw；旧预检、校准、回放调用全部仍计账。详见[engineering/](engineering/)及[scoring/cpu_ledger.jsonl](scoring/cpu_ledger.jsonl)。

调用和pending/duplicate核对：[reporting_snapshot.json](reporting_snapshot.json)；起终点实际文件hash及完整恢复元数据：[checkpoint_manifest.json](checkpoint_manifest.json)。最终公共源码与旧资产保护核验已通过，实际源码、source_hashes.json 与 code.diff 均已交付；详见文末独立复核收据。

图库：[P1全部160输出](gallery/index.html)、[P5与三个叠图](gallery/nested.html)。原三角不修补、不焊接，C/free固定颜色，同父统一框和光照。

固定任务完成即停止。

## 独立复核与交付完整性

独立核算 **PASS**，共 764968 项检查；直接读取逐例表、训练日志与原调用账本，未调用主汇总函数。见 [独立核算收据](independent_audit.json)、[逐噪声配对差](paired_noise_differences.csv) 和 [独立父均](independent_parent_means.csv)。

受保护的实际依赖、原输入和缓存共 351 个文件，前后哈希全部一致；已跟踪公共代码无差异，无自动提交、push或默认模型变更。见 [保护核验](protected_after_audit.json)。

[实际实验与原评分源码](actual_source/)、[源文件哈希](source_hashes.json)、[新增代码完整diff](code.diff)、[400请求的raw完整性索引](raw_integrity_index.csv) 已保存。原评分修订、失败原件、diff和重复CPU评分记录保留于 engineering/，未重生成已有网格。

完整恢复入口：python -B -m experiments.mf_h1_local0.chair_boundary_attachment_v1.execute。固定任务至此结束；未执行报告建议的下一步。

## 完整图库与训练诊断复核

逐请求与图片hash复核通过：全部 **160个P1输出 + 40个P5输出** 均有四视角/线框和三指标叠图，共 **412张PNG**（含12张GT参考）。同父统一画框，未裁掉异常面；这属于技术核验，不是真实用户研究。

| 训练臂 | 末100步FM均值 | 加权接缝/FM比中位数 | 边界向量末步范数 | 向量梯度中位数 | 向量更新中位数 | 标记known角比例均值 |
|---|---|---|---|---|---|---|
| A_FM | 0.10906756525859236 | 0.0 | NA | NA | NA | NA |
| B_BOUNDARY | 0.1090501472807955 | 0.0 | 0.01398006733506918 | 0.00023229133512359113 | 6.093967203923967e-05 | 0.23596414906159044 |
| C_SEAM | 0.10911144084326224 | 0.010025581840993957 | NA | NA | NA | NA |
| D_BOUNDARY_SEAM | 0.10909317591984291 | 0.00997984224360382 | 0.015337633900344372 | 0.0002427039507892914 | 6.141310586826876e-05 | 0.23596414906159044 |

四组各1000个更新与8000个样本的输入、OT结果和监督身份全部配对一致。B/D每步边界向量梯度与更新均非零，8000例均未出现近常数标记。C/D加权辅助loss与FM之比约1%，这是训练loss数值比，不能与校准时输出梯度比例目标25%混同；这些训练量本身不证明生成收益。

详见[独立训练诊断](review_training_diagnostics.json)、[P1完整图库](gallery/index.html)、[P5多片图库](gallery/nested.html)。最终复核没有模型前向、训练更新或新生成。

密度敏感性的全部4个固定单元（首两父候选0，START/D）如下。差值只反映离散评分变化，未替换主结果；4个单元不能代表全部8父的数值误差。

| 请求 | 指标 | 主密度 | 加密 | 加密减主密度 |
|---|---|---:|---:|---:|
| START_HYBRID__P1__33ec1e64a02ff718faaadcdc0221eb72_C1__n0 | SeamCov | 0.06739923947514707 | 0.0680152546976573 | 0.00061601522251023 |
| START_HYBRID__P1__33ec1e64a02ff718faaadcdc0221eb72_C1__n0 | ConflictCov_C | 0.02490234375 | 0.023193359375 | -0.001708984375 |
| START_HYBRID__P1__33ec1e64a02ff718faaadcdc0221eb72_C1__n0 | ExposedRetain | 0.9884189325276939 | 0.9880247069204589 | -0.0003942256072350103 |
| START_HYBRID__P1__341bcae1e7f366a233d93f57729688db_C1__n0 | SeamCov | 0.0 | 0.0 | 0.0 |
| START_HYBRID__P1__341bcae1e7f366a233d93f57729688db_C1__n0 | ConflictCov_C | 0.0 | 0.0 | 0.0 |
| START_HYBRID__P1__341bcae1e7f366a233d93f57729688db_C1__n0 | ExposedRetain | 1.0 | 1.0 | 0.0 |
| D_BOUNDARY_SEAM_END__P1__33ec1e64a02ff718faaadcdc0221eb72_C1__n0 | SeamCov | 0.025655413013429537 | 0.02402110735516422 | -0.0016343056582653163 |
| D_BOUNDARY_SEAM_END__P1__33ec1e64a02ff718faaadcdc0221eb72_C1__n0 | ConflictCov_C | 0.00732421875 | 0.0059814453125 | -0.0013427734375 |
| D_BOUNDARY_SEAM_END__P1__33ec1e64a02ff718faaadcdc0221eb72_C1__n0 | ExposedRetain | 0.986404833836858 | 0.9967225513677045 | 0.010317717530846537 |
| D_BOUNDARY_SEAM_END__P1__341bcae1e7f366a233d93f57729688db_C1__n0 | SeamCov | 0.0 | 0.0 | 0.0 |
| D_BOUNDARY_SEAM_END__P1__341bcae1e7f366a233d93f57729688db_C1__n0 | ConflictCov_C | 0.0 | 0.0 | 0.0 |
| D_BOUNDARY_SEAM_END__P1__341bcae1e7f366a233d93f57729688db_C1__n0 | ExposedRetain | 1.0 | 1.0 | 0.0 |

四单元单输出密度变化最大值分别为：SeamCov 0.16343、ConflictCov_C 0.17090、ExposedRetain 1.03177 个百分点。33ec…候选0的D−START外露变化从主密度 −0.20141 个百分点翻为加密后的 [+0.84457,+0.86978] 个百分点；**该局部外露增减方向标为 LIMITED_NUMERICAL_SUPPORT**。两密度均未越过 −2 个百分点容许线，接缝对比也未翻转，因此这些检查没有推翻本轮 NO_ATTACHMENT_SIGNAL；不能以4个单元声称全部8父的评分密度均已验证。Gamma点数、C样本量与方向同时改变，未单独归因到某个因素。见[密度独立复核](review_density_interpretation.json)。

P5接缝汇总只覆盖能够完整计算的1/2父：另一个父的C3真实Gamma原本不适用，不能缩小分母后宣称两父均改善。可完整计算父的C2/C3平均SeamCov为START 14.54%、A 10.22%、B 10.21%、C 8.99%、D 9.11%；Conflict与Exposure仍对两父全部计算。P4留出上C相对A有约0.40个百分点的局部接缝增益，D却下降约0.71个百分点；P2训练父四组继续训练后的接缝均值均低于START。现有结果不支持跨面板一致的组合收益。

## 对结果的独立解释

本轮保留 START，是针对“生成自由部分合理接入 C”这一目标的决定，并非认为 START 的所有质量指标都最好。A/B/C/D 的 P1 D_sym2048 相对 START 分别改善约 7.15%、8.58%、9.67%、9.43%，但接缝覆盖均值分别下降 2.060、2.495、1.990、2.632 个百分点。

C 相对 A 的接缝均值确有 **+0.06982 个百分点**的有限改善；逐噪声为 13/32 改善、10/32 下降、9/32 持平。然而父均差的中位数为 −0.03634 个百分点，仅 3/8 父改善，且均值低于 START，四条预登记接缝条件均未满足。B、D 相对 A 的均值变化为 −0.43502、−0.57250 个百分点，改善父数为 2/8、4/8。

还有不能被总体均值抵消的代价：C 在父对象 `f59627a077dcc2566c63b8b48495261a` 上，相对 START 的外露保留下降保守区间为 **10.77813–10.83070 个百分点**，稳健超过 10 个百分点严重回退线。该父候选0的接缝覆盖提高 14.68221 个百分点，同时 C 冲突覆盖上升 26.61133 个百分点、外露保留下降 26.13039 个百分点；因此不能只展示这一例的接缝增益。图库中的该例已按此数值依据进行技术目视检查，不是人工可用性评测。

组合效应也不是全方向同号：D−B−C+A 的接缝交互为 −0.20731 个百分点，冲突交互为 −1.28632 个百分点，外露交互为 +0.52711 个百分点；后两项有利，接缝项不利。这不支持笼统宣称组合“有害”，也没有证明组合对本轮附着目标必要。现有四格只有一份共享训练流，不能外推成多种子显著性结论。

原 FF 非法相交受影响自由面比例在 START 和四终点均约 91%。即使相对质量代价门槛通过，也不意味着绝对网格质量、严格拓扑或水密性合格。C 的坐标与面数精确保留已核验；“写入并保留 C”与“自由部分正确接合、使 C 保持外露”仍是不同的实际能力。

**优先下一步仅建议：**基于现有 raw 与冻结 Gamma 样本，离线分解未覆盖接缝的原因：没有足够近的自由边、切向不合格、有限投影支撑不足、或 CF 冲突资格失败。检验假说是：本轮整体距离改善没有转化为接缝覆盖，主要瓶颈在自由边的方向、有限支撑或冲突资格，而不只是最近距离。应先查明各类在全部父与噪声中的占比，再决定是否需要新的训练改动；本轮没有执行该后续分析或新增实验。

完整独立数据见 [P1逐父逐噪声解释收据](review_P1_interpretation.json)；主评分、原始定义与原结果均保留。

