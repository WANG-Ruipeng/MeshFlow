# 窗口轨迹 RL：数学与实现合同独立审查

审查日期：2026-10-15。状态：**数学合同可实现；calibration 未运行/未观察；训练待预算授权及标定门通过**。本审查仅读取附件、现有源码、研究记忆和原始论文/官方文档，没有加载模型、调用 GPU、训练或重新评分。下述测试是待实施清单，不是已经通过的测试结果。

## 1. 来源与实验边界

完整阅读附件 `C:/Users/warpwang/.codex/attachments/1fbfea03-1949-4f5f-a60f-5ce6024668b1/Pasted text.txt`。附件没有任何实际标定结果；`W=[10,30)`、8 条 C × G=8、两三档噪声及具体 σ 数字均为建议。附件没有给出数值通过门槛、学习率、训练更新预算或实测改善。不能把这些建议写成已完成证据。

最接近的历史路线是研究记忆 R29 的终态奖励 NFT：从 FULL_S6000 出发，利用奖励重加权端点再加噪的回归目标，没有证明当前配方的独有新增成功覆盖。该路线已收口；不得续训旧 NFT 或恢复清理的中间 checkpoint。本次实质变化是对**生成时真实随机转移的轨迹 likelihood 做 score-function 梯度**，而非修改旧端点代理 loss。可检验预测是：在固定窗口/噪声和独立新初始噪声上，RL_STOCH 相对 FULL_STOCH 提升合法 CF 接口的单样本发生率。相对 FULL_DET 的实际升级还需采用、主体与 CF/FF/I 代价证据，二者不能混同。

ReinFlow 论文第 4.1–4.2 节为“高斯随机转移、轨迹联合 likelihood、策略梯度”提供原始依据；它区分轨迹联合密度和最终动作的边缘密度。这里的固定窗口、固定 σ、LoRA、二元终态奖励及 RLOO 是本实验的限定适配，并非官方完整配方或其机器人性能结论的直接复现。[ReinFlow v3](https://arxiv.org/html/2505.22094v3)

官方实现文档还允许学习噪声、PPO 以及 action/noise/logprob clipping；这些不能未经审查搬入本附件要求的未截断 Gaussian 联合 likelihood。尤其给采样噪声或 action 直接做 clamp 后，不能继续声称原 Gaussian 密度精确。[官方实现说明](https://github.com/ReinFlow/ReinFlow/blob/release/docs/Implement.md)

## 2. 附件的硬数学合同

令总步数 50，步长 Δt=1/50，t_k=k/50，k=0,…,49。对一条请求，真实 free 坐标维数 d=9(N−K)，padding 与 C 不属于随机动作空间。窗口外真实关闭 LoRA：

\[
 F_{k+1}=F_k+\Delta t\,v_0(F_k,C,t_k),\qquad k\notin W.
\]

窗口内只训练 LoRA，base 固定，且每步实际转移标准差 σ_k>0：

\[
 \mu_{\phi,k}=F_k+\Delta t\,v_\phi(F_k,C,t_k),\qquad
 F_{k+1}\sim\mathcal N(\mu_{\phi,k},\sigma_k^2 I_d),\quad k\in W.
\]

C 在所有 51 个状态逐位保留，不加噪、不计入 logprob。`valid & ~known` 是唯一 free mask。窗口外 σ=0 代表确定性转移，不计算 `log(0)`，也不额外用 epsilon 方差将其替换成随机转移。

σ 是**状态增量**标准差。`σ_k=σ0*sqrt(Δt)` 与直接常数 `σ_k=0.005/0.01/0.02` 是不同候选；前者不再额外乘 Δt。注册必须写明最终实际 σ_k 列表、模型坐标单位和时间网格，不能只写一个含糊的 “noise scale”。不新增中间投影、坐标 clamp 或依赖输出 bbox 的归一化。

rollout 时固定 φ_old、关闭 autograd，并保存真正实现的状态对 `(s_k=F_k^buf, a_k=F_{k+1}^buf)`。更新时 s_k、a_k 均为 detached 常量，重新计算可微的 μ_φ(s_k)：

\[
 \ell_{\phi,k}=-\frac{\|a_k-\mu_\phi(s_k)\|^2}{2\sigma_k^2}
 -d\log\sigma_k-\frac d2\log(2\pi).
\]

必须对 d 个坐标求和，且对窗口步求和；不除以 d 或 |W|。外层仅按原始 B×G 请求数平均。缺失状态/分数或执行失败不允许静默当作 R=0、丢组或重抽。

固定 buffer 的正确导数为：

\[
 \nabla_\phi\ell_{\phi,k}
 =\frac{\Delta t}{\sigma_k^2}\big[J_\phi v_\phi(s_k)\big]^T(a_k-\mu_\phi(s_k)).
\]

若更新中把动作重新写成 `a=mu_current+sigma*epsilon`，同一计算图内 residual 会抵消 μ 的导数；若只以旧 epsilon 的平方作 loss，它又是常量。这两种均不是附件的策略梯度。

初始噪声分布和窗口外转移都不依赖 φ；所有依赖 φ 的转移均具有正方差。因此可在窗口随机动作构成的增广轨迹空间使用 score-function 估计，detached 状态不需要经后缀或 D0 反向传播。窗口外确定性后缀仍会传播窗口造成的状态变化，故**不保证终态变化只局限于局部，也不保证远处主体自动受到保护**。这里不是在计算终态网格的边缘 log-density。

## 3. RLOO 与一次更新

唯一奖励直接读取原 POST 评分：`int(score['topology']['C_F_valid_edge_count'] > 0)`，先完整执行原单次官方 D0、C 恢复、GEO_C 和 source winding。沿用原合法边定义，不把邻近、顶点巧合或 C 逐位保留当作真实连接。旧 NFT 的 `geometry_review_reward()`、组内 std 标准化、clipping 和 `r_NFT` 转换不得复用。

对同 C 的 G≥2 条独立轨迹：

\[
 A_g=R_g-\frac1{G-1}\sum_{h\ne g}R_h
 =\frac G{G-1}(R_g-\bar R),\qquad
 L=-\frac1{BG}\sum_{c,g}A_{c,g}\sum_{k\in W}\ell_{\phi,c,g,k}.
\]

若一组有 m 个成功，成功项 A=(G−m)/(G−1)，失败项 A=−m/(G−1)，组内 A 和为 0。m=0 或 G 时整组优势为 0；保留这些组和原 B×G 分母，不按 active 组数再归一化，不筛选重抽。Active Group Fraction 为同时含 0/1 的完整 C 组数除以预登记完整组数。二项近似下活跃概率为 `1-p^G-(1-p)^G`，所以活跃率门属于梯度信号筛选，不能直接等同于质量好坏。

一个 rollout batch 对应 φ=φ_old 处的**最多一次** optimizer.step。可按轨迹/时间微批累计梯度，但参数在全部 B×G×|W| 项处理完以前不变；不能每微批一步或对同 batch 反复遍历。old 是该 batch 开始时的精确 adapter 快照，不是 EMA；rollout 后到 update 前不得有其他 writer 改参数。

主代理已采用的本轮实现选择：整 batch 全部优势为 0 时跳过 optimizer.step，保留此次 batch 机会和所有样本，不重抽。原因是已有 AdamW 一阶/二阶矩时，即使新梯度全 0，继续 step 也可能因 momentum 移动参数。附件仅保证零优势组的梯度为 0，没有定义这个 optimizer 状态边界。因此“整批 skip”应单独披露为工程合同；固定机会数、实际更新数与跳过数分开记录，不能为了凑够实际更新而新增 rollout。预算仍以最终用户授权/manifest 为准。

loss 无需 ratio exponent 或 PPO，但这不自动保证更新小：梯度受 Δt/σ、维数、窗口长度与 LoRA Jacobian 影响。记录未裁剪/裁剪梯度范数、adapter 参数位移、AdamW step 及 finite 状态；不私加 reference MSE、endpoint loss、KL penalty 或 learned noise。

同 σ 条件高斯的精确 KL 为：

\[
 D_{KL}(p_{old}(\cdot\mid s_k)\|p_\phi(\cdot\mid s_k))
 =\frac{\|\mu_{old}(s_k)-\mu_\phi(s_k)\|^2}{2\sigma_k^2}.
\]

它是固定 buffer 状态下的**条件转移 KL**。对旧轨迹窗口求和再取样本平均，是旧状态占据分布下轨迹 KL 的 Monte Carlo 估计，不是精确终态网格 KL。更新前 old=current 的 KL 为 0，不能拿它证明更新温和；有意义的更新后 KL 需要额外窗口 forwards，必须提前计入预算。保存 μ_old 可以避免重跑 old 前向；仅保存旧 epsilon 不足以在 FP32 舍入后精确还原实际动作残差。

## 4. 现有实现复用清单

| 源文件/接口 | 结论与限制 |
|---|---|
| `r2/pilots/terminal_reward_nft_v1/model.py:AdapterLinear.forward` | `reference` 分支直接返回 base 线性结果，是真旁路；`current` 和 `old` 分支各用对应 LoRA。应以非零 LoRA 验证旁路，零初始化相等不足以证明开关正确。 |
| `LoRAPolicy.activate()`、`update_old(0)` | 可复用，context 结束恢复原 adapter；old 为精确快照。窗口外逐步进入 `reference`，窗口内进入 current。仅用 `no_grad()` 不会关闭 LoRA。 |
| `nft_execution()` | 可复用 BF16 forward / FP32 参数与积分、SDPA math、deterministic algorithms、TF32 off 的合同；调用者控制 grad。`eval()` 可保留梯度；rollout 用 no_grad，更新重算必须启用 grad。不要为了采样把 current 参数永久全冻。 |
| `load_policy()`、adapter checkpoint 原语 | 可读 FULL_S6000 并新建零输出 LoRA；不隐式载入旧 NFT round15。rank4/alpha4/dropout0、optimizer 参数属于本轮注册选择，不是附件硬定义。新 checkpoint 必须有新的策略/窗口/噪声/预算标识。 |
| `sample.py` 的 Euler50、51 态 C 检查、RAW 导出与 receipt 核验 | 可复用守卫和导出；原函数全部为确定性 Euler，没有随机窗口 buffer 和真实 logprob。必须新写窗口循环/版本，不能把原 receipt 更名当作随机轨迹证据。 |
| 原 `learning.py` | endpoint 重加噪 NFT/SELF_REFLOW 科学目标不可复用；这次训练必须直接使用实际相邻状态对。 |
| 原 `evaluation.py` 评分/后处理 | 可复用冻结的 D0/GEO_C/source winding/评分函数；奖励取上述一个原字段。CF=`CF_illegal_affected_free_fraction`，FF=`FF_illegal_affected_free_fraction`，I=`ConflictCov_C`，源尺度均为 0..1。 |

运行时仍要求 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 在 CUDA 初始化前设置。每个 worker 自有模型/optimizer/RNG，种子来自 manifest，不依赖 worker 编号、完成顺序或全局 mutable RNG。预算分别记 Euler、重算 logprob、更新后诊断、D0 前向，不将额外 RL_DET 称为免费；主代理已决定本轮不启动 RL_DET。

## 5. 标定数值门：本轮工程预注册建议，非附件/论文门槛

主代理提出下表，并采用“全门通过者中选择最小实际 transition σ”。这些数字尚非实测结果，必须在看标定输出前固定到 manifest，且所有候选用同一 8 C×8 个初始噪声与登记的独立 transition 噪声流配对。若采用不同 σ schedule，需先固定其排序规则；本轮宜只比较同形常数 schedule。

| 门 | 建议明确口径 |
|---|---|
| 信号 | 至少 4/8 C 组为 mixed，Active Group Fraction≥0.5。 |
| 总体连接保留 | `mean(R_STOCH)≥0.8*mean(R_DET)`，分母均为全部 64，不能只看 active 组。 |
| DET 原成功保留 | 在相同初始噪声的 DET 成功集合，STOCH 成功数/DET 成功数≥0.75；DET 无成功则 NA，不自动 PASS，预定停止或另议。 |
| 非法几何 | 64 请求 POST 原分数宏均变化 `CF≤+0.02, FF≤+0.03, I≤+0.01`，即 +2/+3/+1 个百分点；逐输出/逐 C 值和尾部仍保存。 |
| 主体 | 64 输出的明确 bodyYES 数比 DET 少不超过 2；配对新增明确主体破坏不超过 2。rubric/视图/匿名审查协议在评分前固定；UNKNOWN 保留，不能自动算未破坏或 YES。 |
| 执行 | 所有坐标/窗口 logprob 有限，全部 51 态 C 逐位保留，评分及配对完整，无失败隐藏、重抽或超预算。 |

这些门在小样本下只是可训练性与代价筛选，没有统计显著性保证。混合组多可能来自初始噪声，不证明新注噪本身带来改善；FULL_DET 配对统计应同时保留。若没有候选完整通过，则停止在 calibration 负结果，不增候选、不延窗、不加训练。门通过后仅表示可进入已授权的后续实验，不能先行宣称 RL 有效。

## 6. 建议的最小测试与证据

下面 1–8 可用 CPU FP64 小张量/小型线性 LoRA 验证，不加载 FULL/D0；9 应嵌入首次已授权真实任务，不新增 GPU probe。

1. **联合 logprob**：手算公式对比逐坐标 Normal log_prob 求和；不同 N/K/padding 都只计 free。检查没有维数/时间平均，padding 值变化不影响结果，σ≤0 的 stochastic 调用拒绝。
2. **固定 action 梯度**：对线性 μ 的 autograd、解析 Δt/σ² 导数和 CPU FP64 有限差分比较；故意重参数化 action 的错误版本必须显示梯度抵消，旧 epsilon 常量版本不能冒充正确实现。
3. **LoRA 开关**：让 tiny LoRA B 非零后，reference 对 adapter 扰动不变、current 会变；context 嵌套后恢复原模式，base 无梯度。真实零 B 初始化时 A 第一轮梯度可为 0、B 非零，不能要求所有因子都非零。
4. **buffer/图边界**：保存状态与 action 没有 requires_grad/grad_fn；更新的 μ/logprob 有梯度。reward 和 advantage 均 detached；sample() 中无反传图积累。
5. **RLOO**：G=8，m=0…8 的成功/失败优势及 sum(A)=0；全 0/全 1 保留且不重抽。分母固定 B×G；统计 active 组与真实请求 ID，拒绝缺失/重复。
6. **单次更新等价**：固定微型 batch 全量反传与微批累积的梯度近似一致；统计 optimizer.step 调用为 1，期间 adapter hash 不变。allinactive 时 step=0、optimizer 状态和参数不变；另用已有 momentum 的零梯度示例验证不能误把 step 当 no-op。
7. **条件 KL**：同均值为 0，已知位移符合解析值，非负且按 free/窗口求和。明确区分 preupdate KL 与预算内 postupdate KL，不拿后者诊断调用免费。
8. **checkpoint/计数**：CPU 小模型保存/恢复 adapter、optimizer、batch index、已完成实际 step、RNG与 manifest 身份；完整 batch 消耗后才提交 checkpoint，部分失败保留 ledger，禁止静默重跑或重抽。
9. **实际任务内检查**：首次已授权 rollout 验证 C 全 51 态、窗口外 reference 开关、真实 σ/free-only 噪声、old/current 初始同一参数、buffer logprob 在更新前重算一致；采用相同 batch shape/runtime 以减少 BF16 GEMM 形状差异。GPU 数值 forward/backward 必须逐次入账，有限差分不要在 BF16 大模型上作无意义的小扰动判断。

连续高斯密度是数学模型；BF16 velocity、FP32 action 和有限 PRNG 是有限精度实现，不夸大为机器离散分布上的严格无偏证明。所有真实规模稳定性/效果结论仍需授权后的执行证据。

## 7. 本次结论与待冻结项

没有发现附件轨迹 score-function / RLOO 主公式的根本矛盾。成立条件是：窗口外真正无可训练参数、窗口内全部 free 维度正方差、真实 detached action、joint sum、on-policy 单步、完整组与不选择性重抽。复用旧模型/runtime 是可行的；旧 NFT loss/奖励标准化/训练循环不能直接迁移。

启动前仍须冻结实际窗口和 σ 候选、配对 seeds、calibration 完整预算、body rubric/UNKNOWN 门处理、训练机会/微批/optimizer/裁剪合同、更新后诊断前向预算、TRAIN/DEV 独立终评请求及停止条件。数值门与整批 inactive skip 是本轮工程选择。此文档不是训练许可，也不构成任何 calibration PASS。
