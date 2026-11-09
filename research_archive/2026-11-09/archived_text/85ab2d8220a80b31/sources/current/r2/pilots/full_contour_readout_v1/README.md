# R2_FULL_CONTOUR_READOUT_V1

本目录是一次固定预算的独立候选实验：在现有 Native/T1＋role＋Geo、完整座面条件和原 FM 下，加入一个来源轮廓感知读取模块，与已完成的 FULL 训练曲线配对比较。维护模型、旧评分器、旧结果与默认采样入口不变。

本说明记录实现与执行合同，不预先判断候选获胜。最终效果以实验目录的逐输出结果、实际图审和报告为准。

## 实现与历史定位

早期 G_closest 已经把有限三角形最近点的相对向量放入逐查询 value；本实验不把“几何进入 value”称为首次提出。这里重新检验的是：在当前 Native FULL 训练合同下，使用单个读取层，并加入由输入 C 自身顶点连接确定的、**逐 C 面归属的真实开放轮廓**。

- 保留全部原模型参数对象、名字及遍历顺序；新增名字均以 contour_readout. 开头。
- 在第 6 个**完整** block（attention 与 MLP 都结束）后注入一次，后续 6 个 block 不变。
- 查询是当前有效 free 角点隐藏特征。每个 C 面的 memory 是当前这次前向中，坐标＋role＋Geo 后、进入 block 前的 3 个角点 H0 的 FP32 均值。
- C memory 与查询都不 detach，不跨前向缓存；原主干、role、Geo 与新增支路均训练。
- Q/K/V 为 768→128、无 bias；4 头，每头 32。查询和面 memory 分别经过同一个无 affine 的 LayerNorm，eps=1e-5。
- 几何 MLP 为 17→128→SiLU→128，两个 Linear 有 bias。Q/K/V 使用 Xavier uniform，MLP 使用 Linear 默认初始化。
- 输出为 128→768、无 bias、权重严格零初始化。没有额外零初始化乘法门。新增可训练参数实际为 **412,032**。

每头只按 Q/K 点积构造权重，softmax 的域是全部有效 C 面。几何项保留 free 查询 i 和 C 面 j 两个索引：

    a_ij = softmax_j(q_i · k_j / sqrt(32))
    message_i = sum_j a_ij [W_V(memory_j) + phi(relation_ij)]

没有先沿 free 维度平均几何关系，没有额外几何 logit bias、top-k 或距离截断。因此这是“带局部几何关系的全 C 读取”，不是空间稀疏 attention。

所有支路运算显式关闭 autocast，并使用 FP32；写回定义为：

    updated = cast_original_dtype(float32(hidden) + delta_float32)

仅有效 free 角点接受新残差，C 与 padding 的隐藏特征按原值复制。坐标几何来自实际 FP32 的 C 与当前 F_t。该模块不移动 C、不吸附 F、不输出共享顶点身份，也不提供硬拓扑保证。

源码：[model.py](model.py)、[relations.py](relations.py)。

## 17 维关系与真实轮廓

固定尺度沿用已有模型单位：

    L = sqrt(3) * 2 / 0.3762 ≈ 9.208

它不取当前 C 或生成输出的 bbox。对于 C 面 T_j，B_j 只包含该面上在整个 C 中 source vertex-ID incidence=1 的边。面内边顺序是 (0,1)、(1,2)、(2,0)；两三角矩形的内部对角线不是开放轮廓。

| 索引 | 精确字段名 | 定义 |
|---:|---|---|
| 0 | surface_dx_over_L | 有限三角形最近点−free 点的 x 分量 / L |
| 1 | surface_dy_over_L | 同上 y 分量 / L |
| 2 | surface_dz_over_L | 同上 z 分量 / L |
| 3 | surface_distance_over_L | 到有限三角形的距离 / L |
| 4 | boundary_dx_over_L | 到本面 B_j 最近边的相对向量 x 分量 / L |
| 5 | boundary_dy_over_L | 同上 y 分量 / L |
| 6 | boundary_dz_over_L | 同上 z 分量 / L |
| 7 | boundary_distance_over_L | 到本面 B_j 的实际最小距离 / L |
| 8 | tangent_xx | 无向单位切线 dyad 的 xx |
| 9 | tangent_yy | dyad 的 yy |
| 10 | tangent_zz | dyad 的 zz |
| 11 | tangent_xy | dyad 的 xy |
| 12 | tangent_xz | dyad 的 xz |
| 13 | tangent_yz | dyad 的 yz |
| 14 | boundary_exists | 本面是否存在开放边 |
| 15 | boundary_edge_count_over_3 | 本面开放边数量 / 3 |
| 16 | free_time | 当前 free 时间 t |

无向切线使用 uuᵀ，边端点倒置不改变描述。最近开放边发生数值平局时，对最近点相对向量与切线 dyad 对称平均；距离仍保留实际最小距离，不取平均向量的范数。固定舍入容差为：

    8 * FP32_eps * max(min_distance_squared, FP32_eps * L_squared)

这是固定机器精度规则，不是可调空间邻域。内部计算用坐标规范排列消除浮点求和顺序差异；源 ID 数值不进入网络或几何平局规则。没有开放边时，边界向量、距离、dyad、存在标记和数量均为有限零，**该 C 面仍提供表面 memory**。

拓扑 sidecar 从实际 FULL 的 region_source_face_ids、对应源顶点映射及已核验身份建立；不直接读取历史残留 HALF task_id/source_face_ids 来推断轮廓。训练时随原 face/corner 排列和增强搬运，模型接口必须提供 bool[B,N,3] 的 boundary_edge_mask，C 以外全部为 false。缺失时拒绝运行，不按坐标自动焊接补造身份。来源 Y 的接口标签只用于原有训练误差诊断，不进入新读取特征。

源码：[topology.py](topology.py)；冻结身份见实验目录 inputs/topology_manifest.json。

## 固定训练与比较合同

| 项目 | 本轮约定 |
|---|---|
| 起点 | 原 EXP67 global_step=6000 的 live 生成器、匹配 AdamW 与 RNG |
| 训练池 | 原 FULL 基线的 47 个合格 TRAIN SEAT_FULL 父/条件，K≥2 |
| 数据流 | 原 48,000 条曝光计划逐字节复用，读取原 6,000 批 OT 缓存；无新增 OT |
| 目标 | 原 free-only FM；free-only OT 后 HYBRID 0.25 |
| 增强 | 每有效 batch 的 4 clean＋4 x-width U(0.9,1.1)，沿用原记录 |
| 优化 | batch8、microbatch1；lr=1e-5、betas=(0.9,0.95)、eps=1e-8、weight_decay=0 |
| 裁剪 | 全模型统一 gradient norm clip=1；记录 old/new 范数和实际裁剪系数 |
| 时钟 | 原参数 AdamW step=6000+k；新增组从空状态开始，实际 step=k |
| 预算 | 仅一个候选臂新增6000步；本地0/1000/3000/6000为固定评价点，6000为主终点 |
| 对照 | 复用原 FULL_S0000/1000/3000/6000；不重新训练基线，不挑 best |
| 运行精度 | 主干 BF16 前向，FP32 参数及采样积分；支路 FP32；TF32 关闭、SDPA math、严格确定性设置 |

新增模块在隔离的 CPU RNG 中初始化；完整模型与优化器构建、设备转移完成后恢复原 checkpoint RNG。先按原参数身份恢复旧 AdamW，再追加新增参数组；不把旧 optimizer state 按含新增参数的顺序错配。

首批检查使用正式训练的 8 次正常前向，不增加生成器探针：核对输入/目标/接口标签身份、零残差、隐藏特征不变及旧首批 FM。零出口意味着第一步 QKV/MLP 的支路梯度为零，而出口可以非零；出口更新后内部才能学习。新增出口梯度会改变全模型裁剪范数，所以零初始化前向相同不等于首次更新必然相同。

不加新 loss、B、CREF、RBF、READOUT 历史多层臂、蒸馏或多次去噪；不改变原冻结数据或以来源唯一 GT 款式为成功标准。

## 评价边界与预算

复用六父 FULL、每父四固定噪声的完整 24 请求。每个候选评价点保留 RAW 与单次官方 D0＋GEO_C＋来源 C 角序恢复/free 定向后的 POST；相同噪声、查询、评分版本与导出合同。

候选逻辑规模为 96 RAW＋96 POST。其中 step0 的 24 对结果根据零注入合同和首批实际检查复用旧缓存，不宣称重新测量过候选 step0 的24次 rollout。新增仅 72 次 Euler50 rollout（3,600 生成前向）与72次单次 D0。已完成 FULL 对照96对结果独立保留。

主要比较同一输出的“明确 RAW 采用 ∩ POST 合法接口”，记录原有六个 FULL_S6000 正例保留/丢失与新增父、噪声；UNCERTAIN 单独保留。超过6/24也不自动判定成功，还要看逐父合法共边长度、轮廓位置/长边中段、RAW 有限轮廓距离，以及 CF/FF/I、主体完整性与折层代价。TRAIN 与 DEV 分开，不能用单父收益抵消其他父退化。

来源完整网格仅服务训练答案和来源诊断，不做开放生成的唯一款式排名。A-VMR/PMR 不替代真实共边或完整接续；C 原坐标保留也不认证 C/F 接合。图审标为实际评阅者的 AI_PROXY；这批 DEV 已被反复观察，不能称独立新父泛化证据。单候选比较只能检验整个新模块，不能独立归因到轮廓、层位或新增容量。

## 入口与恢复

所有路径相对 E:\MeshFlow。主要产物根目录是 experiments/r2_full_contour_readout_v1；原基线目录 r2_full_seat_learning_curve_v1 只读。

查看状态，不启动科学任务：

    .\r2\pilots\full_contour_readout_v1\resume.ps1 -Stage status

恢复尚未完成、已授权的固定流程：

    .\r2\pilots\full_contour_readout_v1\resume.ps1 -Stage all

入口会核验冻结注册、已有收据与任务身份，遵守资源余量及已保存的安全进度；失败/OOM/孤儿任务不静默重试，不覆盖旧结果。已有控制器运行时先检查状态，不另行绕过 owner/账本启动重复任务。每个 checkpoint 必须保存完整24份 RAW 后才进入该点的 D0；GPU 阶段按本轮资源合同运行，独立 CPU 评分可在限额内重叠。

自动数值阶段完成仍须补真实图审、结论和最终归档；恢复入口不能替代实际看图。完成固定任务后停止，不切默认、不自动 commit/push。

主要文件：

- registration.json：冻结源权重、数据流、预算、评价请求与源码 hash。
- inputs/topology_manifest.json：实际 FULL 的来源拓扑身份。
- training/CONTOUR/：逐曝光、逐更新、调用账本、资源与可恢复状态。
- checkpoints/、inference/：固定节点完整恢复点与推理导出。
- evaluation/：逐请求 RAW/POST、原评分、轮廓表、图集和真实图审。
- audits/model_unit_tests.json：13项实际 CPU 小模块测试；未构造/前向完整生成器。
- audits/model_train_integration_fix_verified.json：step0安全恢复的首批观察接入复核。
- audits/zero_initialization.json：首批正式训练零出口与旧 FM 一致检查。

源码入口：[resume.ps1](resume.ps1)、[orchestrate.py](orchestrate.py)、[train.py](train.py)、[evaluation.py](evaluation.py)。几何/小模块测试源码：[test_model.py](test_model.py)。
