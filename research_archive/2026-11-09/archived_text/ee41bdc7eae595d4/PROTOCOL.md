# R2_BENCH_SHIFT_RELATION_CROSSOVER_V1

用户于2026-10-22授权执行附件建议的有限后续。仅一方向SHIFT前缀→RELATION后缀，固定t=0.5；不训练、不扫描交接时刻、不增加反方向。

## 先行匿名复核

对现成32个DEV请求的FINAL，保留RELATION/SHIFT两输出、原视角/尺度/位置截面。固定seed2026102207随机左右与页序，图中仅显示匿名页码与A/B。身份映射放private_identity，不交给评阅者。使用未继承本会话的独立AI评阅线程，不能声称人工或完美盲法；旧样本可识别、AI代理可靠性仍是限制。每列沿用五轴及UNCERTAIN，逐页实际查看，一次评阅，不反复改标。root已知结论，不担当本阶段评阅者。

事前门槛：全部32对完成后，SHIFT的C_use=PLAUSIBLE总数严格高于RELATION，并且至少2个不同DEV父存在RELATION=REPLACED_OR_IGNORED而SHIFT=PLAUSIBLE的请求。若不满足，记录采用正信号未通过该有限复核，停止GPU交叉候选；不补评或改门槛。此门槛只判断是否值得继续有限检验，不认证整体质量或统计显著。

## 数据、真实前缀与新调用预算

完整48请求=历史TRAIN4父×2C×2噪声+DEV8父×2C×2噪声。全组均在当前26父适配池之外；历史TRAIN在COMMON阶段见过，DEV已多次用于开发。沿用相同C/N、Gaussian数组、OLD_S15000 C-only预测布局、角色/锚点、来源绕序、相机、查询与评分版本。禁止输入未知来源Y和owner真值。

SHIFT48个原trajectory.npz含51个FP32状态，读取真实state[25]；绝不用终点插值。接手RELATION_V1_S1500冻结live导出，绝对Euler步25..49（t=.50,.52,...,.98,dt=.02），沿原seam_forward、mask、FP32积分及硬锚。不重置或重排free，不重抽噪声、重做OT或平移推理坐标。

新候选标签SHIFT_RELATION_T050。48×25=1200新生成器前向；48个单次官方Bench D0；144个RAW/BASE_POST/CLEAN_FINAL_POST新评分单元。训练、反向、新占用头、特征模型、新噪声、新OT均0。原SHIFT/RELATION终点与评分只读复用。保存全部新RAW后才开始D0。后处理统一D0→恢复C/seam→GEO_C/来源定向→HINGE_OPP015→ALPHA→HOLE。

## 判读与停止

主问题：保持SHIFT采用收益的同时是否收回几何代价。原同轮9个DEV采用转入及8个原PLAUSIBLE固定列表均完整追踪；匿名复核重标与原列表并列，不因重标删除原坏例。候选仍按全48、TRAIN/DEV、逐父、逐部位分析。

保留全部原177数值字段、失败和NA。I主口径PASS_ONLY噪声→条件→父等权，LOW_SUPPORT独立；CF/FF比例与绝对面对、有效free面数、删除/面积、碎片、小角、支撑完整性同时保留。closed C无接口需求记NA，不补0成功率。≥30% bulk可达为既有描述指标，不是新成功定义。

有限“组合有效”候选必须同时：在本轮统一评阅中，保留至少2/3由匿名复核确认的SHIFT独有采用请求，并保留全部匿名复核两模型共同PLAUSIBLE请求；DEV的CF、FF、有效I父宏均值相对SHIFT全部下降（允许距RELATION仍有差距，明确报告），三个绝对冲突量/有效支持定义不变；DEV支撑PRESENT不减少且保留free面均值不降低。不满足则不晋级，仍记录局部正/负例。此工程决策门不是统计显著性检验，不把非盲AI判读当人评。不得通过移走主体、删除更多表面或换父/噪声取得成功叙事。

最终生成后三模型同页实际AI_PROXY图审，明确非盲且不覆盖先行匿名记录。RAW/BASE/FINAL分开，BASE包含D0、恢复和GEO，不能单归因D0。只执行一次固定候选；失败可能含接手模型状态分布不兼容，不能给早/晚时段定罪。

## 资源与恢复

每新GPU波次/worker前记录实时GPU、VRAM、利用率、GPU进程、CPU/RAM及其他任务。至少保留max(2GiB,10%总VRAM)余量，第二worker需第一worker真实输出峰测量后按1.2安全倍率计算。采样可按manifest前/后24请求两shard独立写，保留各自模型/RNG；新seed不得取workerID。D0单worker采用既有整组单写入口；CPU评分先测首件再按资源最多4worker，每进程1数值线程。

保持CUBLAS=:4096:8、strict deterministic、TF32off、SDPA math、BF16前向/FP32参数与积分。没有额外GPU探针。OOM/数值失败/部分调用立即停止扩派，保留日志；不静默重试，不变batch/精度。已完整请求可核验复用，部分调用重放需另行明确记账授权。

最接近旧路线为R2_BENCH_TRAJECTORY_CROSSOVER_V1（16请求USER/COMMON，新增800F，RAW-only，DEV无新采用）；本轮不同模型、48面板、SHIFT已有待复核DEV采用信号，且包含完整POST。该差异允许一次有边界检验，不声称新架构。

原权重、源码、数据、RAW/POST、评分及报告只读。新包装/分析均独立目录，记录实际源码hash和差分。完成后停止，不切默认、清理权重、commit或push。
