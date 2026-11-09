# R2_FULL_CONTOUR_READOUT_V1：方法与执行事实初稿

状态：DRAFT_METHODS_ONLY，版本2。证据截点为 2026-10-14，已取得 local_step3000 的安全完成收据；读取时后半段仍在运行（最近状态为 local_step=3503、global_step=9503）。前版已保存在 [v1历史副本](report_methods_versions/v1_4f34ca636fea9a60/report_methods_draft.md)，SHA256为 `4f34ca636fea9a60afe8d30ac5b76d2c6b045b99aa39aadda2108e22a28f0aa1`。本文件只记录已冻结的方法、已经发生的执行检查和工程修正；尚不构成最终训练完成证明、最终审计或质量结论。正式结果应引用结束后的账本与最终配对表，不能把下面的预算写成已经完成的次数。

## 1. 比较对象与数据

本轮只新增一条 CONTOUR 训练臂，与已经完成的 FULL 原结构学习曲线配对比较。起点是 EXP67 的 step6000 live 非 EMA 生成器及其匹配 AdamW/RNG；不是从 FULL_S6000 继续累加，也不是重新从官方权重初始化。复用 FULL 的 47 个 TRAIN 父、每父一条完整 SEAT_FULL 条件（K=2..20），以及原 48,000 条曝光计划和 6,000 份 OT 缓存。不加入 DEV，不因本轮生成表现替换父对象、条件、噪声或完整来源几何。

新增 6,000 次更新，固定记录 local step 0/1000/3000/6000，对应累计 global step 6000/7000/9000/12000。local step6000 是预先指定的主结果，不按图像或指标挑选 best checkpoint。各点使用相同六父、每父四份固定噪声的 24 请求面板。比较既保留同预算 FULL 曲线，也保留共同起点，不能把普通续训收益全部归到新增模块。身份和冻结哈希见[注册表](../registration.json)。

## 2. 实际模型

主干仍为世界坐标 Native/T1 + role + Geo；只在完整第六个 block 的 attention 与 MLP 之后插入一次 `contour_readout`。旧参数名称、对象顺序和继承值保留。新增参数精确为 **412,032**，代码在独立 pilot 目录，不修改维护版 Native forward。

每个有效 free 角点的当前隐藏状态作为 query。每个 C 面的 memory 来自本次 forward 的初始隐藏 H0，即坐标、role、Geo 编码后的三个角点 FP32 均值；不 detach，不跨 forward 缓存。query 与 memory 分别使用同一无 affine LayerNorm（eps=1e-5），经无 bias 的 Q/K/V 768→128 投影，四头、每头32维。对所有有效 C 面做 QK softmax，不使用 top-k、距离 attention bias 或邻域截断。

关系 MLP 为 17→128→SiLU→128，两层均有 bias；Q/K/V 使用 Xavier uniform，MLP 使用 Linear 默认初始化。每对 query/C 面的关系向量产生独立 pair-value，与投影后的 C value 一起按 attention 权重聚合。128→768 出口无 bias，权重精确零初始化，无额外门。实现显式关闭该分支 autocast，FP32 计算关系、投影、softmax 和聚合；先将主干 hidden 提升为 FP32 与 delta 相加，再 cast 回原 dtype。仅更新有效 free 隐藏角点，C 和 padding 原样保留。

这不是首次引入几何 pair-value：项目历史 G_closest 已使用有限表面关系。本轮的具体变化是 after-block6 的固定读取位置、每 C 面自己的真实来源开放轮廓，以及与有限表面并列的关系特征。单臂实验也不能单独区分新增容量、轮廓特征或层位的贡献。

### 17 维关系的固定顺序

| 索引 | 实际字段 |
| --- | --- |
| 0–2 | `surface_dx_over_L`, `surface_dy_over_L`, `surface_dz_over_L` |
| 3 | `surface_distance_over_L` |
| 4–6 | `boundary_dx_over_L`, `boundary_dy_over_L`, `boundary_dz_over_L` |
| 7 | `boundary_distance_over_L` |
| 8–13 | `tangent_xx`, `tangent_yy`, `tangent_zz`, `tangent_xy`, `tangent_xz`, `tangent_yz` |
| 14 | `boundary_exists` |
| 15 | `boundary_edge_count_over_3` |
| 16 | `free_time` |

固定长度单位为 `L=sqrt(3)*2/0.3762`，约9.208，与原 FULL 诊断一致，不是2，也不是逐对象 bbox 尺度。surface 是有限三角形的最近位置。boundary 是该 C 面自身三条边中，在整个 C 的原始顶点 ID 拓扑里 incidence=1 的边；共享内部对角线不算开放边。顶点 ID 仅决定拓扑，不作为数值特征，也不用于距离 tie-breaking。

边切向量使用无符号 dyad `u u^T`。等距候选对最近矢量和 dyad 作对称平均，距离仍为真实最小距离。数值等距容差固定为 `8*FP32_eps*max(min_distance_squared, FP32_eps*L^2)`，这是浮点舍入合同，不是可调几何邻域。没有开放边的面其 boundary 特征为有限零、mask=0，surface 仍参与读取。退化三角形采用有限边/点回退。没有预测端吸附、顶点焊接或额外几何修正。

额外 forward 输入 `boundary_edge_mask[B,N,3]` 对应局部边01/12/20；缺失、shape/device不符或非 C 位置置真均拒绝。原 `readout_mode=none` 保留兼容，新增支路由独立模型类提供。详见[模型源码](../../../r2/pilots/full_contour_readout_v1/model.py)、[关系源码](../../../r2/pilots/full_contour_readout_v1/relations.py)及[实现说明](../../../r2/pilots/full_contour_readout_v1/README.md)。

## 3. 训练合同与优化器时钟

目标仍只有原 free-only FM，free-only OT 之后 HYBRID0.25。effective batch8、microbatch1；每有效 batch 四条 clean、四条 x-width U(0.9,1.1)。原学习率1e-5，AdamW betas=(0.9,0.95)、eps=1e-8、weight_decay=0，全体参数共同 global norm clip1。沿用原数据流、增强、噪声、时间和 OT；不为新增分支重配目标或新抽噪声。

先恢复194个旧参数的匹配 AdamW 状态，再附加同超参数、空状态的新增参数组。local step k 后旧参数累计 AdamW 时钟为6000+k，新参数为k；不是把新增参数伪装成已有6000步状态，也不清空旧动量。模型构造、设备搬运及优化器组建立后恢复原 RNG。全部旧/新参数可训练，模型保持 eval 以沿用无 dropout 的原训练合同。主干 BF16 forward，参数 FP32；确定性算法、TF32关闭、SDPA math 沿用原执行定义。

仅顺带记录已有合法来源 C/Y 接口角点与其余 free 角点的误差，随原 OT 面和角排列搬运标记；这些标记不改变 loss 权重，不作为推理条件。端点误差由同一次训练前向的 `(1-t)^2` 速度 SSE 得到，不替代自由生成，也不增加 rollout。

## 4. 已执行的结构和初始化检查

13项 CPU 小模块/几何测试已经通过，不构造完整生成器、不调用 GPU。测试覆盖有限表面与退化回退、矩形内部对角线、角/面重排和 ID 重命名、空边、pair-value、FP32路径、free-only更新、零出口与后续梯度通路。实际 unittest 用时0.147秒、进程墙钟5.1608529秒。见[CPU测试收据](model_unit_tests.json)。

另有192项 CPU 拓扑检查通过，并核对47条训练条件、6条评测条件、24请求和124份来源文件。对8个继承了旧 HALF 元数据的请求，依据当前 FULL 的 `region_source_face_ids` 与来源顶点映射建立掩码，不直接采用过时字段。当前 C 没有检出的非流形来源边；这不认证完整来源或生成网格整体无交。见[拓扑准备](topology_prepare.json)和[拓扑测试](topology_tests.json)。

首个正常训练有效 batch 的8次 micro forward 实际记录了分支 delta 精确为零、加回后的 hidden 逐值不变、残差有限，并与原曲线首批8份 FM 逐项精确相等；首批均值为0.18918075459077954。该检查复用已授权的训练前向，没有新增完整模型 probe。见[零初始化检查](zero_initialization.json)。

第一次更新前旧参数梯度范数为0.44087570905685425，新参数梯度范数为0.0032997436355799437，新增梯度仅在零出口 W_O，其余分支梯度为零，符合零出口初始化。第二次更新已记录到 Q/K/V 与两层 MLP 的非零梯度，说明内部梯度路径进入训练；这项检查不说明学习到的几何关系有效。local step0 的安全恢复也会重新启用首批观察列表，修正收据单独保留，未覆盖旧测试记录：[训练集成修正](model_train_integration_fix_verified.json)。

## 5. 预算及缓存边界

下表是注册的整轮上限，不是当前执行完成数。

| 项目 | 注册数量与身份 |
| --- | --- |
| 新训练 | 6000 optimizer updates；48000 forward、48000 backward |
| 候选逻辑输出 | 四个时点共96 RAW、96 POST |
| 候选 S0 | 复用24 RAW、24 POST，不重新 rollout |
| 候选后续三个时点 | 72次 Euler50，即3600生成器前向；72次单次官方D0 |
| 旧 FULL 对照 | 四时点共96 RAW、96 POST，缓存引用 |
| 额外预算 | 新OT=0、新噪声数组=0、额外完整模型probe=0 |

S0引用建立在继承检查与零出口合同上，必须标为缓存引用，不能描述成重新完成24次推理的复现。所有方法保持同一单次官方D0、GEO_C、来源C角序恢复和固定C的free定向导出；本轮没有加入RBF、B、CREF、新loss、蒸馏或重复D0。

图审计划为候选三个新时点共144条实际 AI_PROXY（72输出×RAW/POST），旧 FULL 有192条继承图审，候选 S0 有48条映射引用。三类身份分开计数，不能把384条逻辑标签写成384次新增实际图审。

## 6. 已发生的资源中断与恢复

0→1000段已经完成：8000次训练forward、8000次backward、1000次更新，墙钟1586.994114317秒。后续运行在 local1114 / global7114 因设备可用显存暂时低于2048MiB阈值安全中断；不是OOM，不存在失败或挂起模型调用。停止时成功且已记账的累计次数为8912 forward、8912 backward、1114更新。没有终止用户其他进程，没有重放已完成更新。

中断保存的 checkpoint 文件 SHA256 为 `44fc8594c9fd129e212add9c7fb9f991d2fa8cb8f9da57d1444911530d08230c`；生成器状态为 `8b7a9955400a6fe5db45150cd00d25d136f7b73c0a83e73008db18868416c7c6`，优化器状态为 `7fd50982caa4873d920408d8271565105b68a6f81eaa0b3f72196a1bb4b2c67c`。恢复初始化收据确认生成器、优化器与RNG匹配，下一步为1115，科学设置未改变。见[资源恢复收据](resource_resume_1114.json)和[实际恢复初始化](../training/CONTOUR/attempts/orchestrated_1114_to_3000/initialization.json)。

恢复前资源重测为RTX5080、总16303MiB、已用8094MiB、空余7886MiB、利用率14%，24逻辑CPU、约61GiB可用RAM。这是恢复重测值，不是触发中断瞬间的最低显存。当前调度一个GPU任务，独立CPU评分可资源约束并行；每CPU进程限制线程。PyTorch reserved峰值只记录分配器范围，不冒充整个进程显存。并行CPU工作可能影响墙钟，不能将上述分段时间当成隔离吞吐基准。

### local step3000 的实际完成与控制器恢复

2026-10-14 04:20:45.625596 UTC，控制器会话11759在读取正在替换的 `training/CONTOUR/status.json` 时遇到短暂 `FileNotFoundError`。它停止新的派发，并安全等待已经派发的训练子进程完成，没有杀停训练或回滚已完成更新。[原始失败事件](orchestrator_status_fix/0bd1874d2b432ae6/failure_event.json)保留了 traceback、safe_drain=true、no_automatic_retry=true。

训练子进程于04:46:40.039828 UTC以returncode=0完成 local step3000 / global step9000；控制器会话随后以exit1结束。两种退出状态对应不同进程：控制器读取失败没有成为训练调用失败。此时累计24,000次forward、24,000次backward、3,000次optimizer update全部成功，failed为空、pending为空。1114→3000训练段墙钟为2948.649378584秒，调度job墙钟为2954.653651611秒，二者范围不同。见[训练段收据](../training/CONTOUR/attempts/orchestrated_1114_to_3000/final_status.json)及[子进程完成记录](../orchestration/jobs/train_1114_to_3000/done.json)。

local step3000完整checkpoint文件SHA256为 `e34e7e867177baf2b22adf287c7f34d017873be909378a48f3986cac536e02cd`，生成器状态为 `9cbabadb43bddb64e1411551e4748b88d0f32bba339236442cb7a47371dcc321`，AdamW状态为 `587e0f392c9886ab08de91f04e9479131c4daa64fe38142a30bc1e451756d787`。推理导出文件SHA256为 `5894cceedcf4eb050862a5ec01f5cf43d00294d3bddae27f2d3eae0650a106dd`。这些哈希来自现有收据，本次文档更新没有加载或重算权重。

已审核的控制器恢复记录为 `REVIEWED_CONTROLLER_ONLY_RESUME`，新会话96166按已完成游标接续：先处理预定S3000评价，再启动3000→6000训练段，下一训练更新为3001，科学调用重放数为0。该段实际初始化收据确认恢复的生成器、AdamW和RNG哈希匹配，累计旧优化器步数9000。见[控制器恢复](controller_resume_3000.json)和[后半段初始化](../training/CONTOUR/attempts/orchestrated_3000_to_6000/initialization.json)。这不是新增实验臂，也没有增加训练、采样或去噪预算。
## 7. 工程修正与审计状态

所有以下问题均保留原记录或源码，并与科学定义变化区分。

- 正式派发前静态复核修正了曝光列表变量被checkpoint哈希覆盖、首批观察列表缺少实际append、S0缓存路由误进生成流程、以及训练入口缺少已声明曝光/拓扑/旧注册哈希核验。修正发生在正式受影响调用前；[预训练复核](pre_training_review.json)和独立训练集成修正有记录。
- 并行收尾时，一个只涉及拓扑加载CPU缓存的 evaluation 编辑跨过注册冻结。已在任何候选评价模型调用前恢复冻结源码，恢复前候选评价模型调用数为0，训练未变；未注册版本被保留。[冻结源码恢复收据](pre_dispatch_overlap/receipt.json)。
- 汇总代码曾将真实键 `(request_id,label,stage)` 按 `(label,request_id,stage)` 访问。只修正未冻结的汇总工程代码，旧源码保留；使用真实旧结果身份加内存同值副本检验31841项配对差精确为零。该fixture不生成模型结果，不新增评分或图审，也不改变指标定义。新汇总哈希明确取代旧 readiness 收据中的对应源码哈希。[修正收据](analyze_key_order_fix.json)。
- 进度审计器曾拒绝合法 `ROLLOUT_COMPLETE`，并把尚未完成的最终配对表 PARTIAL 当成错误。原FAIL审计保留；限定修正检验了24份现有S1000账本，共1200已预约/返回采样调用、1560检查，以及5个内存负例/未完成例。这个修正通过不等于全项目最终审计通过。该截点仍需最终审计，不能删除原FAIL或把旧latest指针当作已重跑的PASS。[审计器修正](audit_source_fixes/schema_cf653be2eac2cb08/correction.json)。

- 控制器状态读取竞态只修正 `orchestrate.py`：仅对FileNotFoundError/JSONDecodeError最多读取6次、间隔100ms；只有不存在训练、checkpoint、推理、job或既往非零控制器状态证据时，持续缺失才允许报告NOT_STARTED。已有运行证据下持续缺失或持续JSON损坏会明确抛错，不回退到step0。七项CPU fixture通过，包括短暂缺失/部分JSON恢复、真实新任务、持久损坏拒绝，以及已有训练/checkpoint/控制器状态不得伪装step0。收据记录85份冻结科学源码哈希未变。旧控制器源码与diff保留，新控制器SHA256为 `ff268fd5213ee9297e64771a7f9dde7dcb10fbd5a28098979d71ab2d00d38803`。[工程修正收据](orchestrator_status_fix/0bd1874d2b432ae6/receipt.json)。CPU测试本身未重启控制器、操纵worker或执行任何模型/训练/OT/评分；随后正式恢复另有上节独立记录。
## 8. 评价、图审与可解释边界

评价同时保留RAW采用、RAW有限表面/轮廓关系、POST实际合法接口及共享边界长度，以及CF/FF/I、主体完整性和款式变化。采用与接口必须在同一请求关联，不以A-VMR或某一均值替代整个关系。TRAIN熟悉父与DEV分层、每父四噪声配对；已多次观察的DEV不能称为全新盲测。UNCERTAIN保留，不强行改成成功或失败。来源完整形状只帮助确认C身份和训练诊断，不是开放生成唯一正确款式。

实际图审可能针对按时点配对的独立图集，例如 `gallery_FULL_S1000_CONTOUR_S1000`，CSV保存当时真实查看的图像/剖面路径和SHA256。归档脚本允许这种身份；最终48张全方法预览（24张depth、24张sections）是另一个展示集合。不能把最终重排后的预览说成原先实际评阅的同一张图。保留原图审CSV与路径哈希，S0使用继承映射；归档不会因此要求重新补384次图审。当前归档脚本尚未实际打包，最终归档必须等训练、全部图审、报告及显式最终审计齐备。

## 9. 冻结来源核对索引

以下哈希引用已存在的注册/收据，不为撰写本文件重新加载大权重。

| 对象 | SHA256 |
| --- | --- |
| [本轮注册](../registration.json) | `d9c9aae17bc6df9d108fcbdaf0b072f1aefa627ddfef069520a80f205c13c41e` |
| EXP67 step6000 checkpoint文件 | `cfb18cfa0dd82dbe55c1e7dfb7c78e862f01647d8f044bd9002a46a8d3d1ee27` |
| 起点生成器状态 | `54ba75f915bddaf3486276efec381b7b64d926f17cced6c70d5a18b4b7a9df4e` |
| 起点AdamW状态 | `f46d8dd1f9e62120e966f0df8259a2268165d4be94b9591771140dc6bef96956` |
| 原FULL曝光replay plan | `52a48be0987c11511dea149829935e6af9baceb4ac70317052653b95c557f4c9` |
| [本轮拓扑清单](../inputs/topology_manifest.json) | `9153cd4a5038feaf900e9ec61f844b67063e366064a0cce225d5d12033bb6d01` |
| 新model.py | `7b61890258c91f7079bae0c6f6c1fd8b880d419736c9fe253fc6c19f09d005c2` |
| 新relations.py | `24ae70efab9b43bb85fc4aea9fe010aa3bb48c2f849ddbcfdebff3d3a577a889` |

最终应由完成后的账本核对实际更新、调用、曝光、checkpoint时钟、生成/POST身份、全部逐输出/逐父配对和图审完整性。本初稿不预判候选是否优于原FULL，也不授权新增训练、替换默认模型或commit/push。