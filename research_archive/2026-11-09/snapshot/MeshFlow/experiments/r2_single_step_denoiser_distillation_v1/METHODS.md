# METHODS — R2_SINGLE_STEP_DENOISER_DISTILLATION_V1

本轮只有一个学生 S8_W512_KD、一条冻结训练流、一个正式4000更新终点。目标是压缩官方单次去噪器，不更新生成器、不改变控制架构、不训练新的几何损失，不按DEV挑checkpoint。

## 资产与独立性

生成器为冻结 R2_USER_MIX step6000，文件SHA256 `8c1287b13541b59fa7b1e70974a70b06fbe0d67160449c225e5bdc7fdd0897cd`。教师为原官方chair denoiser step30000 EMA，文件SHA256 `6a31f4c7dcbea7a6ac58c26075b20c897fd19aad4c33b5295cc4c93ed7078c36`，原源码提交 `db86bc32423c316c00bcc2f645a26243e8010f20`。只读加载，不使用上一轮适配教师。

学生采用同一实际identity-encoder/regression residual路径，hidden512、8层、8头、RMSNorm和实际SiLU×4 MLP、12频NeRF（保留输入，75维）、corner→face→attention→corner流程。完整N由face_bin100的AdaLN条件输入；9行embedding全部保留，包括N=800使用的第9行。实际face-count dropout和其他dropout均为0，activation checkpointing关闭。

学生38389251参数，104个state条目；教师455592963参数，296个state条目。所有学生参数按原规则随机初始化，最后残差linear零初始化。没有复制、裁切或隐藏教师参数，也没有known-role、readout、Geo新分支或时间token。已独立导出的学生只能加载自身权重；运行还需要小型原架构源码和第三方依赖，不需要大教师权重。

## 数据与冻结教师标签

沿用已审计112条件／25父白名单。UID与group-ID排除开发、确认及测试父。该白名单仅保证本轮接受的条件，不是全网格无相交认证：仍有59条Y内部非法几何记录、10条UNKNOWN；完整缺陷和来源清单随包保留。

- 224训练RAW：每条件2份冻结Gaussian，R2_USER_MIX原Euler50；每步C保持，不输入Y/M、部件标签或接口信息。
- 25内部校验RAW：每个训练父预先选定一条件、第三份Gaussian；从不反向。这是同父新噪声校验，不能称独立父泛化。
- 896合成训练输入：每条件8份，1份clean、3份free独立角噪声σ=.02、4份C/Y联合x宽度U(.9,1.1)后free角噪声σ=.02。全部扰动预先冻结，不在线抽样。
- 最终DEV：48份既有真实RAW（4生成器×6条件×2噪声）与18份既有同形合成输入。真实DEV是6个父的重复观察，不是48个独立父。

所有标签均为D0对实际输入单次前向的proposal。合成clean也模仿教师，不能偷偷以原Y作为KD答案。训练1120＋内部校验25＋真实DEV48新增1193个教师标签；18合成DEV复用已保存的精确denoiser域proposal，不从模型单位近似反推。缓存保存input_d、proposal_T_d、有效r_T_d、真实模型单位proposal、原C、mask和每个输入/权重hash。

## 数值路径与损失

令a=.95×.3762=.35739。输入变换为FP32 `x_d=a*x_model`，教师和学生均在BF16前向后用FP32计算 `proposal_d=x_d+delta`、`r_d=proposal_d-x_d`。KD比较实际有效残差，而不是用网络原始delta代替。推理FP32除以a，随后直接写回原始模型单位C字节；不把C缩放往返误差当允许的控制漂移。

每批8个不同父，4真实＋4合成，microbatch1。损失为全批free标量平方误差总和除以 `9*sum(N-K)`；C不计损失，C特征仍参与注意力和梯度。各microbatch按同一总分母累积，既不每例独立平均再平均，也不把padding或known计入分母。

FP32参数、BF16前向、FP32残差加减/loss；TF32关闭、SDPA math、严格deterministic、CUBLAS_WORKSPACE_CONFIG=:4096:8。教师eval无梯度、参数hash不变、无新RNG消耗。学生无在线增强/dropout，正式初始化与预检副本分开。

## 优化和预检

新AdamW，betas(.9,.95)，weight_decay0，foreach=False，梯度范数裁剪1（foreach=False），无学生EMA。正式4000更新、32000训练F和B。更新1的LR为1e-6、更新100为1e-4（99个线性间隔），随后cosine到更新4000的1e-5。冻结数据流的0基索引转换为1基优化器更新数，不改变样本序列。

预检教师固定2输入各回放2次；学生隔离副本2个batch8更新，LR1e-4用于非零路径和梯度检查，两次更新全部丢弃。预检与正式foreach和精度一致，但预检LR不是正式warmup。预检检验零初始化、known保持、free分母、独立FP32标量loss、完整N、非零FP32/BF16面角联合置换恢复、known特征梯度、无RNG消耗、完整AdamW显存、独立学生导出重载。FP32/BF16置换误差分别如实记录，不把浮点舍入差改称结构不变性证明。

只在0/500/1000/2000/4000做25内部校验＋8固定训练样本，共165F。每批学习曲线来自更新时参数，不冒充固定最终权重对全部训练缓存的统一评估。中间点不跑完整DEV几何、不挑best。保存500/1000学生权重、每100步闭合latest以及4000完整checkpoint/纯推理导出。

## 误差与最终质量

free有效标量E=mean((rS-rT)^2)，B=mean(rT^2)，R=E/B。残差与差值使用FP32，保存后的指标归约FP64。逐例与按free标量池化均报告；B≤1e-14为LOW_SIGNAL，R/比例标NA，绝对E保留；零学生向量的cosine为NA。报告残差norm ratio、cosine和师生preGEO差的coordinate RMS、corner p95/max，单位d、model、model/L（L=9.208138264587333）。R<1只表示比恒等更接近教师，不是几何质量保留率。

真实RAW没有唯一重建GT；只在18份同形合成DEV报告Y恢复误差，区分“模仿教师”和“接近Y”。训练固定8例、同父内部校验、真实DEV、合成DEV分别汇总。

每个既有真实输入比较RAW参考、IDENTITY_GEO、D0_GEO、STUDENT_GEO。三条POST均使用完全相同的原GEO_C合同；实际144 POST。IDENTITY和学生96份新GEO/评分，D0的48份旧POST/逐点/拓扑/质量只在身份全匹配后复用。无学生残差缩放、吸附新阈值、多轮调用、按例教师兜底或模型融合。

沿用原评分源码、原冻结查询、全部8档A-VMR/PMR、CF proper、FF非法面对/受影响free、I cross/dup/并集、角形/退化/碎片及原字段；不改变门槛以挑有利结果。另保存实际合法共边、反向绕序、从C可达free面积、非流形、原source face删除差、学生新增非法面对及GEO半径成员关系跨越诊断。小坐标差不保证相同合并拓扑；阈值跨越只是敏感性证据，不单独证明贪心合并的因果来源。

全部48请求展示四路径、四视角，并对指定五类风险各两噪声补10张局部图。AI_PROXY是有局限的图像审阅，整体形状/C参与/细件损伤/层片分别记录，允许不确定，不替代逐点几何审计。

## 时间、显存与预算

生成器、教师、学生依次单GPU进程；每阶段重测资源并保留至少2GiB或总VRAM10%的额外余量，不根据权重文件大小推训练显存。教师和学生测速使用各自独立进程、同6个预选输入、4CPU线程。

每输入神经路径5热身＋20计时；完整POST3热身＋10计时，每模型228F。神经路径使用已在GPU的固定input_d，包含实际有效proposal运算；完整POST包含CPU→GPU、坐标变换、网络、GPU→CPU、C原字节恢复和原GEO/数组合同验证。每次都从原输入开始，无反馈。CUDA同步；文件I/O、账本I/O、评分、渲染排除。完整POST收益不等于整个Euler50生成流程加速。

比较相同FP32参数字节，不拿7.29GB教师训练checkpoint作为压缩分母。记录allocated/reserved峰值及阶段范围；WDDM进程显存不可读则NA，不能用PyTorch allocated冒充全进程显存。coldload每模型仅一次，受OS缓存影响，只作描述。

总GPU占用上限14400秒；实际调用与阶段墙钟以账本为准。工程包装问题允许留证修正，未闭合模型调用不可静默重放。所有科学输入与旧结果只读，输出写入本轮OUT；固定任务结束即停止，无自动commit/push或默认模型替换。