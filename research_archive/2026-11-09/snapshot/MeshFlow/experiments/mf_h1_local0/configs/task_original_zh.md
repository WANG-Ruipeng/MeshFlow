# MF-H1-LOCAL0：官方 MeshFlow 上的查询相关原语控制验证

版本：2026-09-28 / v1.0。性质：**执行任务书，不是已运行的官方模型实验**。
主问题：在相同几何提示、同一冻结官方主干与匹配训练预算下，向条件消息加入“当前生成位置到有限原语的几何关系”，是否比普通查询相关消息及强采样点条件更有价值？

## 0. 边界、资产与执行权限

只使用 SIGGRAPH 2026 Qi Sun 等人的 `qiisun/MeshFlow`，不是其他同名模型。阅读当前仓库的 AGENTS.md 及适用的上级说明。不要覆盖既有 AGENTS.md。

只在新增 `experiments/mf_h1_local0/` 与独立输出目录工作。不得 reset/stash 用户改动，不改系统驱动、不升级原环境、不修改原数据与 checkpoint，不覆盖旧实验，不自动提交或推送。优先独立 wrapper/import，不改原训练/采样入口。发现必须修改核心代码时，先用可审阅的新文件/补丁记录必要修改与原因，不能静默替换。

审阅参考提交：`8cc74da013d3d2feec1853cccbb1f9ac37814b95`。它不是对本地 HEAD 的假定，也不是要求把本地 reset 到该提交。先记录本地 HEAD、dirty 状态、关键文件哈希；版本不同则逐项核对语义。

官方 chair checkpoint 入口：HF 数据集 `qsun2001/meshflow` / `v1/120m-ot-v-chair/checkpoints/last.pt`。参考配置为 `configs/snet/base-120m-ot-v-chair.yaml`。权重与数据的本地绝对路径未知，应从用户工作区、已有配置和已授权缓存定位；不要全盘扫描个人目录。

过去云端仅验证过自建 322,627 参数、64 维四层的合成代理；旧适配器为 14,688 参数。旧模型的强度调节、边差分损失和显式手工描述子没有成为质量解决方案。不得用旧合成代理、随机官方架构或重建大模型冒充官方权重结果。不能把小模型的尺寸、层号、学习率和数据协议直接复制为正式设置。

**默认授权范围是 Phase 0–2：预检、单测与接线短训。Phase 3 需要用户明确发送第二段执行提示；即使 Phase 2 成功，也不得自行启动扩大训练。** 无权重时可以完成自包含数学/几何单测，但所有 official 字段必须保留 NOT_RUN/BLOCKED。不得要求先向云端上传 2GB 文件；本地直接读取。

## 1. 本轮只验证 H1，不叠加其他路线

保留整物体 triangle-soup 输出、原嵌套 OT、原噪声先验、原 FM 速度监督和原主干。只训练新增适配器。不加图网络、面内刚性头、SDF、物理约束、新几何损失、ControlNet 全主干复制、部件生成、Agent 规划、反向优化或几何投影修补。

输入只包含少量无向表面线段/三角片及其分组/类型；不是物体内部骨架、关节、带符号法向、语义部件 ID 或文本。提示是软控制，不要求复制为输出中不可变的边/面，不宣称水密/无交/流形。

H1 的精确范围：静态单 Token 交叉注意力在无额外门控时有 softmax=1，因此该分支的聚合消息与 Query 无关；这不是整个主干没有局部控制能力的定理。不能仅靠单测证明新架构有效，也不能把所有 ControlNet 泛化为这种简化模块。

## 2. Phase 0：官方资产、运行环境与无控制基线

### 2.1 核查而不假定

输出 `environment.json` 与 `source_audit.md`：仓库绝对路径、HEAD/dirty/diff、Python/PyTorch/CUDA、GPU 型号/显存/当前空闲量、设备实际可用性、BF16 探针、attention backend、checkpoint 绝对路径/字节数/SHA256/来源、数据路径及 author train/test split 身份。不把旧记忆中的 GPU 型号当作当前事实，不抢占繁忙 GPU，不重启/终止他人进程。

在参考配置中，chair 为 version=3、12 层、768 宽、12 头、max_length=800、face_bin=20、QK/RMSNorm、坐标 Fourier、无面序号位置编码、velocity prediction。实际 checkpoint 的配置与 tensor shapes 必须核对；不一致时报告，不静默覆盖。

### 2.2 低内存、安全、严格加载

优先 `torch.load(path, map_location='cpu', weights_only=True, mmap=True)`，核对实际顶层 keys 后读取 `checkpoint['ema']`。键不是 `model_ema`。不自动回退到 weights_only=False；受信文件若存在不支持对象，记录具体对象并处理最小必要 allowlist，无法确认则停止。

在当前 PyTorch/模型兼容时，使用 meta 构造与 `load_state_dict(..., strict=True, assign=True)`，之后再构建优化器。若不兼容，采用有峰值监测的常规 CPU 加载；不得跳过缺失/形状不匹配参数。只在验证前缀后统一移除 `module.`，不做含糊键名匹配。

EMA 是完整的一份权重，选 EMA 不会自动把同精度参数大小减半。可以保存新的 EMA-only FP32 副本，保留原文件不动并记录双方哈希。不要求转 FP16，不保证特定显存或秒数。

### 2.3 先证明官方模型在本地真的执行

运行官方已加载模型的单次前向及至少 4 个固定噪声的完整采样，保存原始 triangle soup、原始 OBJ、统一可选后处理版本。记录真实 NFE、时间网格、返回状态、峰值显存与 CPU RSS；可视化固定种子，失败也保留。

沿用原 sampler 语义作 reference，再验证自定义 sampler。参考 `num_sampling_steps=50` 是 t_eval 数量，不能不计数就称 50 次 Euler 更新。正式协议统一显式 50 次 Euler 更新：t_k=k/50，k=0..49，dt=1/50；与 reference 对账时使用相同 51 个 t_eval 节点和相同 CFG。单位前向误差与 rollout 误差分开记。

Phase 0 可另展示官方默认面数 CFG；后面的四组主比较统一：保留真实面数条件，face_cfg=1，geometry_cfg=1，无几何 CFG 外推。关闭几何控制不等于去掉面数条件。

`y` 是面数，不是类别/原语 ID；`mask` 是有效面槽位，不是局部编辑锁。原 `forward_with_cfg` 会调用自身 forward，自定义适配器可能被绕过；原 `inference.do_sample` 可能重建并载入模型，不能传入 wrapper 后就假定采样的是 wrapper。必须通过 call counter 验证实际 callable。

无官方权重：标记 BLOCKED_CHECKPOINT_BYTES，停止 official 生成与训练；不以随机权重“演示一把椅子”。有权重但无真实数据：完成无控制/接口单测，训练标记 BLOCKED_DATA。

## 3. 固定架构：只让一个关键变量改变

以下为本地首次实现的设计合同，不是已测最优超参数。

### 3.1 共同部分

冻结官方主干所有参数，保持其 eval 模式。新模块为：
- 点坐标 Fourier（6 档 sin/cos，含 xyz）+ 原语类型 one-hot 2 维；
- 共享条件 MLP：41→128→128，SiLU；
- 适配层：官方 12 层的第 3、6、9、12 层之后（1-based）；若实际不是此结构，不强行套层号；
- Query 来自主干当前面角 hidden `[B,3N,768]`，归一化并映射到128；不是只用面质心；
- 4 个 cross-attention head，每头32；输出 projection 128→768，**仅出口零初始化**；
- 主干每层之后只加残差，不改主干 attention/FFN；scale=1，首轮无额外 gate、null token、时间强度调度或 geometry logits bias；
- 每个原语至多一个 pooled Token，或点基线的总32 Token。所有臂看到相同原始提示顶点、类型、分组与采样点，不喂语义 ID。

输入条件不能逐原语重中心化。提示、Xt 和 target 使用同一全物体坐标系；该 benchmark 默认对象已经 canonicalized，不能宣称未知位姿扫描能力。

### 3.2 四组（Phase 3 的完整比较；Phase 2 先跑 Q/G）

| 代号 | 条件 token | 消息内容 | 目的 |
|---|---|---|---|
| S_static | 每原语 pooled token | 标准静态 Value cross-attention | 简单已知基线，不假定必须差 |
| Q_center | 每原语 pooled token | Query/时间相关 Value；相对向量 x−原语质心 | 排除只是增加查询相关非线性/空间坐标 |
| G_closest | 与 Q 完全相同 | 仅把相对向量换成 x−有限原语最近点 | 核心候选；与 Q 是单变量对照 |
| P_point | 同32采样点，保留分组/类型 | 与 Q/G 同样的动态 Value；相对向量 x−采样点 | 强点条件基线，排除弱基线胜利 |

Q/G/P 的 trainable tensor shapes、参数数、初始化权重、优化器完全相同；只有条件 token 内容/数量及确定性几何变换不同。S 缺少动态 MLP，参数可以更少，必须如实报告；不要给它添加不参与计算的虚假参数凑“等参数”。主因果比较是 G 对 Q，应用比较是 G 对 P，不是 G 只赢 S。

所有臂共享点级特征 f(p,type)。S/Q/G 对每个原语内部采样特征取均值得到 e_j。P 对每个采样点使用 `e_point + mean(e_same_primitive)`，不加新的可训练参数；这是分组知情的强基线。P 的 attention logits 加 `−log(m_j)`，使同一原语内部采样数量不自动变成提示总权重。S/Q/G 每原语等权。不要给 G 更多条件或给 P 删除分组关系。

### 3.3 动态消息的确切定义

对当前面角 i、条件单位 a：

    q_i = Wq(LN(h_i))                     # 128
    alpha_ia = softmax_valid(q_i·k_a/sqrt(head_dim) + log(w_a))
    R(delta) = concat(delta, ||delta||²,
                      sin(pi*2^k*delta), cos(pi*2^k*delta), k=0..3)  # 28
    tau(t) = fixed_sincos(t)              # 32
    F = MLP(128+128+28+32 -> 128 -> 128, SiLU)
    value_ia = Wv(e_a) + F(q_i, e_a, R(delta_ia), tau(t))
    residual_i = ZeroLinear(sum_a alpha_ia * value_ia)

多头按128维拆成4×32，逐头 softmax/加权后再拼接；Q/G/P 使用同一实现，不在一个组使用不同 dropout 或不同归一化。S 去掉 F 项，其余 attention 相同。

Q: delta=x_i−c_j。
G: delta=x_i−pi_{P_j}(x_i)，pi 是**有限**线段或三角片上的最近点，不是无限直线或平面投影。
P: delta=x_i−p_a。

几何 query 使用本次 forward 的真实 Xt 面角坐标，不使用 target X1、GT 面对应、未观测完整表面或“教师终点”。不要另跑主干预测干净终点，不加额外模型前向。F 同时读取 h,t，不做硬最近原语归属，不宣称早期噪声位置已可靠。

几何项是软条件特征，不把网格顶点投影到原语上。无向边反转、无向面任意角重排须保持最近点几何不变；无带符号法向输入。退化输入的确定性 fallback 必须对称、有限并被记录。

## 4. Phase 1：强制单元与接线测试（不产生效果结论）

每项有输入、实际误差、阈值及 PASS/FAIL，禁止全零残差上的空洞对称检查。

1. **官方原模型 vs 控制关闭**：同权重、同 t/N/mask/Xt、同 backend。FP32 确定性条件下目标是逐位相同；若运算顺序改变则记录 max_abs/max_rel，超预声明 atol=1e-6、rtol=1e-5 停止。完整 rollout 另报差异，不扩大阈值来掩盖接口错误。
2. **零初始化**：开启分支但零出口时与控制关闭一致。不要重置已加载主干的 initialize_weights。
3. **梯度**：出口应先获得梯度；零出口可能导致第一次更新其他分支参数为0，这是正常的。执行5–10个探针更新后，检查 Q/G 分支关键参数梯度；主干参数/状态哈希不变、主干无grad。探针不得混入正式训练。
4. **冻结不是 no_grad**：保持梯度穿过后续冻结层返回较早适配器；不能把整个主干包进 no_grad 或 detach 所有 h。重载/恢复时同样检查。
5. **单 Token 定理探针**：无 null/gate 条件下，S 的未投影聚合消息对不同 Query 相同，logits-only 距离偏置也不改变结论；Q/G 的结构允许查询相关消息。用单独非零测试投影或检查零出口之前的消息，不能用正式零初始化后的全部0输出证明成功。只验证结构，不要求训练后一定存在某个性能差。
6. **几何正确性**：线段内/外、面片内投影/边界最近点、顶点最近点、重复点、近退化、远查询；与独立CPU点到三角片/线段实现随机对账。评测oracle不要与训练几何helper共用同一函数。
7. **置换**：面重排、面内角重排、提示重排、边端点交换、三角片6排列；条件点重排。FP64 helper atol=1e-10；FP32网络先用固定 atol=1e-5、rtol=1e-5，报告实际差。无向编码测试不等于SE(3)等变。
8. **padding**：增加虚假面/原语/条件点不改变有效输出；m=1与m=3混批等于分别前向；空条件显式走零控制，不让全masked softmax产生NaN。掩码 True/False 语义核对所用SDPA。
9. **参数/调用**：Q/G/P 参数数完全相同，G/Q 相同初始化；主干调用次数不增加；wrapper训练与采样都真正调用adapter，不能绕回原forward。
10. **恢复**：连续10步 vs 5步+保存/恢复+5步，核对模型/优化器/RNG和输出（GPU非确定算子记录容差与局限）。正式训练须保存完整可恢复状态。

## 5. 数据、随机性和提示协议

### 5.1 真实父对象划分

目标建议128个训练父对象、32个验证父对象、32个保留评价父对象；按实际数据UID/父对象身份去重。训练与验证来自作者train，保留评价优先作者test；不能确认预训练是否见过时，只称 adapter-unseen，不称整个系统unseen。

在生成/训练结果出现前，确定性选定UID并保存manifest。若样本不足，报告实际数量和原因，不复制同一对象填满，也不回退到旧合成椅子。几何筛选只按预先定义的输入合法性，所有过滤率/失败保留；不得根据候选模型结果筛对象。

保留源模型可支持的面数，训练按原真实N+padding。评价统一输出512个raw面槽位。不得为达到512而截断目标面列表、复制面片或把后处理面数当raw预算。若实际checkpoint不支持512，先报告，修改共同协议而不是单组特判。

### 5.2 三种提示层次

训练按1/3、1/3、1/3采样：单表面线段、单三角片、两线段+一片。每条提示来自真实网格的非退化有限面/边，确保位于目标表面；不要未经标注就称为坐面前缘/语义特征边。不同组始终使用相同源提示。

一条边：32个线段点，包含端点。一个面：32个点，包含三个角，其余面积采样。混合：8+8+16=32点，包含全部顶点。原语内部均值权重和P中的−log(m_j)固定，不额外学习提示重要性。

每个原语单独采样128个独立内部检查点；端点/角点不作唯一指标。辅助加密评价可用不同随机种子，不根据候选结果挑检查点。退化面不采为训练提示，但几何单测要覆盖退化输入。

Phase 3 对32个评价父对象都准备三种提示、两次新噪声；按类型分层报告，不让mixed结果掩盖K=1。正确/固定循环错配提示同时评价；错配输出相对于实际输入和原提示的误差分别记录。

### 5.3 归一化与官方耦合

直接复用并核对官方处理顺序。参考代码为：bbox居中/尺度归一化→排列/增强→除以0.3762→对该坐标做噪声OT配对→target再乘2，noise没有同步乘2。不要“修正”这条路径后还称冻结官方预训练协议。

条件从最终target坐标系提取或按完全相同变换进入；若从半尺度阶段取提示，必须显式乘2并通过点在GT表面的数值检查。元数据记录可逆变换，禁止每个局部提示独立重归一化。

独立RNG流控制模型初始化、父对象/噪声/OT、提示采样与eval噪声。相同训练seed的各组必须收到相同原始input序列；逐步滚动hash tokens/noise/N/mask/原始提示/采样点以核对。不使用GT配对噪声作为完整生成起点。沿用官方每次数据采样的噪声机制，不复用旧六噪声bank代理。

## 6. Phase 2：仅接线短训，完成后停下来交付

先用16个训练父对象，Q/G各200次optimizer更新、seed10，effect batch=8，其他设置同Phase 3。共有400步，不算正式实验。使用新的初始化，不接旧代理adapter。

报告训练损失、固定teacher-forced路径MSE、梯度/哈希、4个固定对象×3提示×2噪声的完整生成。对错条件必须进入实际采样器。不用训练MSE代替生成效果。

本阶段的目的是检查模型能更新、条件被实际读取和完整采样不绕过分支。未看到明显效果可记 NEEDS_LEARNING_DIAGNOSIS，不能直接判H1失败；不自动延长至1000/10000步。检查失败应定位错误，修复后同设置重放并保留日志，不进行架构搜索。

完成Phase0–2后给用户 PRECHECK_REPORT，明确建议是否值得授权Phase3。**当前第一条执行提示不授权Phase3。**

## 7. Phase 3：独立授权后的有限正式机制pilot

四组S/Q/G/P，每组seed10/11，各1000 optimizer updates，总8000。新初始化，Phase2权重不继承；共享参数逐tensor拷贝初始值，Q/G/P全state完全相同。每个seed四组用同一输入流。

默认：adapter lr=1e-4，AdamW betas=(.9,.95)，weight_decay=0，grad_clip=1，effect batch=8，无scheduler，无adapter EMA；主干严格冻结eval。输入/data规范继承官方，主干面数condition dropout关闭，不加几何CFG。选择BF16只在实际设备通过测试后；FP32单测与训练精度明确区分。首轮不启用torch.compile，不为某组单独更换后端。

预检资源允许时microbatch=2、accumulation=4；否则共同降microbatch、保持effectbatch8、数学正确的masked loss累计。不允许候选组batch2而基线batch8还声称等预算。OOM要记录；不能跳过困难对象。变长面数下，不能简单对每个microbatch的mean loss除以accumulation；应让每个microbatch的误差sum除以本次完整effective batch的有效坐标总数后累积梯度，以保持原masked全坐标均值。相对关系query维可固定分块实现以节省显存，块大小对所有组统一。

保存step100/300/1000；中间只看validation并记录学习曲线，主结果固定1000终点，不挑最优step。测试集只用于这次预注册评价，不根据测试反复改方法。到1000后仍明显未学会的情形标记INCONCLUSIVE_TRAINING_BUDGET，不自动搜索学习率或继续训练。

主评价每模型：32父对象×3提示×2噪声×2条件模式=384输出，8模型共3072。base无几何条件仅需32×2=64个独立输出，可复用来对不同提示评分，不能虚增独立样本数。

额外pairing/recombination、点密度扫描、cross-category、面数泛化、更多训练种子均不在本轮授权范围。获得正信号后再立新协议。

## 8. 指标、统计与结论条件

### 8.1 必报指标

1. `guide_rms_bbox_pct`：独立原语内部点到生成**三角形表面**的RMS，先每原语平均平方距离再等权汇总。另报告各原语max/q95和coverage@1% bbox；该阈值是诊断设定，不是认证。
2. 正确与错配提示的效果，按父对象平均两个噪声后计胜率；相对于实际输入提示与原提示分开，不把所有错配视为非法形状。
3. 原始几何的NaN/Inf、退化面、min-angle<5°面占比、全部内角p05；逐输出统计后聚合。自交/水密/流形另审计，工具缺失必须NOT_RUN。raw triangle-soup没有共享索引，不把它直接按索引检测的孤立面数当形状拓扑结论。
4. 全表面到参考的对称RMS是辅助补全指标；稀疏条件存在多解，不能只靠随机配对参考RMS判定无条件基座质量或多样性。
5. 固定条件不同噪声下的未指定区域变化、多样性探索性指标；不把噪声/碎片造成的距离差当合理多样性。
6. 可训练/总参数、真实NFE、encoder/几何query/主干/完整sampling耗时、峰显存/RSS。条件编码在inference整条轨迹中都可缓存，各组公平缓存；G的当前位置几何项必须每步更新。
7. 未后处理raw输出为主；官方统一post作为另表，任何面数变化、时间都报告，不为G单独加修补。几何提示满足不等于输出必须包含输入边/面索引。

所有计时预热并同步GPU，同设备、dtype、batch、timegrid、CFG、NFE、后端；随机交错顺序、报告分位波动。不把cross-attention K减少比例当端到端加速。

### 8.2 统计单位与主比较

主比较：G−Q；其次G−P；S是基础参照。按single_edge/single_face/mixed分层，并另给等权总体。每seed单独报告，按父对象配对bootstrap，不把两个噪声/三个提示/多个seed当大量独立对象。同一官方基座的两次adapter训练不代表两个独立预训练实验。

预先规定：把引导RMS均值相对减少10%作为值得继续的**pilot效应量目标**，不是普遍标准。正信号需至少两seed方向一致、父对象配对区间支持下降，并且窄长面占比相对主要baseline不劣化超过2个百分点、数值失败不增；其他质量变化同时展示。门槛是聚合筛选，不是逐对象保证。达不到不等于架构不可能有效。

若同guidance误差下还需比较quality，必须下一轮在validation上预先定义共同scale扫描；本轮固定scale1，不允许测试后为G单独挑更弱控制来改善角度。

### 8.3 允许的最终判定

- `BLOCKED_ASSETS`：权重/数据缺失，完成的单测明确列出。
- `IMPLEMENTATION_FAIL`：identity/gradient/mask/load不通过，不能谈模型效果。
- `IMPLEMENTATION_PASS_ONLY`：Phase0–2完成，尚无正式优势结论。
- `CONDITION_RESPONSE_ONLY`：能够响应条件，但没有超过强基线。
- `QUERY_DEPENDENCE_SIGNAL`：Q/G相对S改善，但G未胜Q，支持动态消息，不支持有限原语最近点的额外必要性。
- `GEOMETRY_ADVANTAGE_SIGNAL`：G对Q有预注册正信号；若未胜P，不宣称原语方案优于强点方案。
- `NO_ADDED_VALUE_AT_TESTED_BUDGET`：基线已学会、G无额外收益；不要写理论否定。
- `INCONCLUSIVE_TRAINING_BUDGET`：所有相关分支尚未学会/变化大，预算不足；不自动扩大。

单token单测只支撑分支的表达性质；测试错误、训练不足、算法无增益、有效网格质量是四件不同的事。巨大平面/细碎面片也可能贴合提示，绝不只用guideRMS给出有效性结论。

## 9. 建议的独立文件结构和产出

    experiments/mf_h1_local0/
      audit_assets.py
      official_wrapper.py
      geometry.py
      adapters.py
      prompt_dataset.py
      train_adapter.py
      sample.py
      evaluate.py
      tests/
      configs/

这些是Codex要实现的目标文件，本交接包不冒充它们已经存在或运行。

wrapper/API建议：`forward(x_t, t, y, mask, prompts, prompt_mask, control_scale=1)`；prompts保留vertices `[B,K,3,3]`（边用arity=2）、arity/type/mask、采样点/分组，不包含GT对象ID、未观测表面或目标位姿标签。空提示绕过控制。

每次运行独立目录：

    source_audit.md / environment.json / resolved_protocol.yaml
    asset_manifest.json / split_manifest.json / prompt_manifest.json
    unit_tests.json / phase2_report.md
    checkpoints/adapter_*.pt / logs/*.jsonl
    results/per_output.csv / paired_summary.json / decision.json
    raw/*.npz / raw/*.obj / figures/ / REPORT_zh.md

训练checkpoint保存adapter、optimizer、step、所有RNG、config、base SHA、data/prompt hash。报告将源码审查、单测、official strict-load、official forward、official rollout、adapter train和比较结果分别列NOT_RUN/PASS/FAIL，不能用一个PASS覆盖所有层次。

交付代码patch和轻量结果包，不把2GB基座再打包上传；只记录它的身份与哈希。失败日志、OOM、首次bug及其影响必须保留。开始训练的进程须在当前任务中监控至完成或安全checkpoint停止，最终报告不能假定离线后台会继续交付。

## 10. 第一轮结束时直接回答的五个问题

1. 真正加载的是哪一份官方EMA？有多少次原模型和wrapper实际前向/完整采样？
2. 关闭/零初始化控制能否恢复原模型，早期adapter能否穿过冻结主干获得梯度？
3. 几何消息是否正确依赖当前Xt、有限线段/三角片，且没有使用GT终点？
4. Phase2的条件响应和完整输出是什么？哪些只是单测，哪些确实来自生成？
5. 是否具备执行Phase3的条件，有何资源或数据阻塞？不得先宣布H1已经被证实。
