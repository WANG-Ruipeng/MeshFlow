# MeshFlow 条件生成研究路线记忆

更新：2026-10-15；原FULL_CONTOUR历史盘点补入NFT训练与冻结新噪声检查的收口。用途：新实验前查重，保存已经付过的试错成本。它不授权新实验、恢复退役权重、切换默认模型或 Git 提交。

## R29：NFT终态奖励路线已收口为当前配方的负结果（2026-10-15）

训练后终点评价面板36个配对请求，FULL/NFT/SELF明确严格联合均9/36，没有新增成功覆盖。补充冻结检查保留原8条奖励TRAIN C，每条4份新噪声，三模型共96输出：RAW采用均26/32，严格联合FULL/NFT/SELF为11/12/11。唯一新增a7b82d/n0也由无奖励SELF实现；NFT相对SELF的净差来自保留a7/n2，并非独有新能力。其余7条C联合数量没有变化。[S84](../../experiments/r2_terminal_reward_nft_v1/REPORT.md) [S85](../../experiments/r2_nft_training_c_frozen_v1/REPORT.md)

固定原始奖励在共同已知28对/7C上，NFT−FULL=-0.003113、NFT−SELF=-0.002472；第8条C保持UNKNOWN，全32奖励差可行区间跨零，不声称统计显著或全组真实奖励已经下降。NFT的FF均值下降，但I和局部尾部仍有代价，合法接口总长70.2352→66.6572。SELF共享NFT端点池，是off-policy无奖励对照；不是原GT普通续训。图审为实际AI_PROXY，非人类金标准。

**决定：停止扩大当前NFT配方，不追加轮次、不挑中间权重、不切默认。** 保留局部正结果，不上升为“RL无效”或“奖励函数已被证伪”。FULL_S6000继续作为方法比较参照。以后若重开，应提出实质的适配/优化或监督转化变化及可检验预测；不能只改名或加步数重跑。

按用户授权清除NFT/SELF两臂round00..14的30个中间权重，保留round15两份、FULL/官方模型及全部RAW/POST、评分、图审、日志、代码和旧报告。中间checkpoint逐轮重放已退役；旧报告与审计描述历史完成状态，不代表全部中间权重仍在磁盘。[负结果报告](../../experiments/r2_nft_negative_result_closeout_20261015/NEGATIVE_RESULT_REPORT.md) · [清理账本](../../experiments/r2_nft_negative_result_closeout_20261015/cleanup_receipt.json)。本次仅文档与清理，零模型调用。

## 先读这几条

2026-10-14盘点 **83份主实验／诊断报告**，本次补入两份NFT报告，来源索引共 **85份**，跨早期 `mf_h1_local0`、`native_t1/runs`、退役档案、独立 `MeshFlow-Control-artifacts` 和现行 `experiments/r2_*`。核对原报告的关键正文、表格、后续复验与维护记录，另读当前模型入口；没有重新训练、前向、生成、评分或重新做图审。下文数字是原记录中的有限面板结果，不是本轮新测量。完整清单及 SHA256 见 [来源索引](research_history_sources.md)。

按**机制家族**合并，登记 **29条已运行但未形成稳定升级的候选路线（R01—R29）**，另有 **2条训练前门禁受阻路线（B01—B02）**。这不是实验臂总数，也不是“31种方法已被证明永远无效”：一个家族可含多个阶段、多个loss或复验；局部收益与适用限制必须保留。原128父门槛的数据阻塞另列，不算一次训练失败。

当前问题已经收敛：**模型会读取C，能改善它的大致位置和部分用途，但没有稳定地把生成的free轮廓接到C上；局部贴近、匹配与质量收益之间经常发生取舍。** 后续不能再把“收到条件”“训练目标下降”“有一个好例子”当成完整控制成功。

最近的充分训练比较最直接：FULL与CONTOUR的6000步终点均为RAW明确采用16/24、有合法接口8/24、两者同时成立6/24；联合正例TRAIN为6/12、DEV为0/12。CONTOUR只在已有正例多出两条边，没有新增成功请求。**下一次不能只把“加局部几何读取”重新命名后再跑。** [S35](<../../experiments/r2_full_contour_readout_v1/REPORT.md>) [S36](<../../experiments/r2_full_seat_learning_curve_v1/REPORT.md>)

## 研究脉络与模型身份

| 阶段 | 已经改变过什么 | 当前应该怎样理解 |
|---|---|---|
| 冻结官方主干的Q/G适配器 | 外部条件编码、残差注入、强度与位置 | 有响应，未建立有用且稳定的控制；随后转向原生条件微调 |
| Native/T1 → Geo → 共享关系/几何loss | known time=1、free time=t、role、C几何编码、全模型FM | Native/T1、Geo被保留；额外约束有局部取舍，未普遍解决接口 |
| MVP48 / Domain → HYBRID START | 多父变长条件、宽度增强、OT后混噪 | 有可学习证据；不能把旧双对象结果外推为多父控制 |
| 新官方初始化的R2 → USER → EXP67 | fresh AdamW、全程混合条件与HYBRID，随后USER任务与父池扩展 | R2是训练配方家族；USER不再以旧C20/C40任务为主目标 |
| EXP67 → 集中200 / CREF / FULL6000 / CONTOUR6000 | 曝光、参考坐标、完整座面长程训练、局部轮廓读取 | 最新方法比较以FULL曲线为参照；研究终点不自动替换CLI默认 |
| RAW → 单次D0 → GEO_C → 来源定向 | 坐标去噪、保护C、几何整理、恢复原C角序 | 已验证并维护的流程收益；与采用/实际接续分开评价 |

`START`指HYBRID累计5000的维护默认；原R2是从官方EMA重新适配的配方，EXP67是67父USER训练6000终点，FULL_S6000是从EXP67再训6000（累计12000）。SMALL_C200是**四条件小池**续训；FULL_C200是**67父全池**续训，后者不是后来47条完整座面的FULL_S6000。不同实验的A/B/C、Full/Joint不能仅按短名对齐。[S41](<../../experiments/r2_seat_noise_transfer_v1/REPORT.md>) [S36](<../../experiments/r2_full_seat_learning_curve_v1/REPORT.md>) [S48](<../../experiments/r2_user_data_scaling_32_67_v1/REPORT.md>)

维护入口见 [训练结构](training_structure.md)；实际基线 Native/T1+role+Geo 使用Geo的P=I，不能描述成已采用邻接传播。历史READOUT/teacher代码留在依赖闭包不表示R2启用这些模块。新轮廓读取独立在 `r2/pilots/full_contour_readout_v1/`。

## 28条已运行候选：不要原样重做

“重试依据”是未来方案应说明的科学差异，不是额外审批流程，也不要求先做一轮大审计。优先引用已有证据，用最小必要验证确认新差异。

### 条件读取与表示

| ID／旧名称 | 实际证据与停止原因 | 重试至少应补什么 |
|---|---|---|
| **R01 冻结主干Q/G、几何适配器** | Q/G各200步后又做Q_correct/independent各1000步及N112小试。存在非零条件响应；Q_correct在4父上没有得到有用的完整生成改善，固定N绑定不稳定。确定性修复没有把它变成成功控制。[S20](<../../experiments/mf_h1_local0/outputs/run_20260928_01/PRECHECK_REPORT.md>) [S11](<../../experiments/mf_h1_local0/outputs/debug_repeatability_20260928_150208/DEBUG_REPORT.md>) [S15](<../../experiments/mf_h1_local0/outputs/learning_diagnosis_20260928_155505/LEARNING_DIAGNOSIS_REPORT.md>) [S19](<../../experiments/mf_h1_local0/outputs/nxC_112_sandbox_20260928_173427/NX_C_AND_112_REPORT.md>) | 说明新的条件对接或可训练范围与旧版有什么不同；与负对照比完整生成，不用“梯度非零”作主证据。不是证明冻结主干永远不可行。 |
| **R02 注入前后位置、强度及推理日程** | 试过Pre/Post同初始化各500步、条件/关闭速度组合和时间权重。新16噪声未证明Pre稳定更好；部分均值由少数噪声贡献主导，没有晋级推理设置。[S14](<../../experiments/mf_h1_local0/outputs/inference_injection_pilot_20260928_183412/INFERENCE_AND_INJECTION_REPORT.md>) | 有明确的新作用机制或诊断预测；不能继续无目标扫描层位、alpha、时段。 |
| **R06 C边界标记／boundary embedding** | 早期三臂B未建立独立接口收益；后来的START四格A/B/C/D各1000步中，C-only开放边界向量也未改善主接合结果。SeamCov均值A=.09311、B=.08876。[S23](<../../experiments/mf_h1_local0/outputs/t1_interface_3arm_20260929_181500/THREE_ARM_INTERFACE_REPORT.md>) [S53](<../../native_t1/runs/chair_boundary_attachment_v1/CHAIR_BOUNDARY_ATTACHMENT_FACTORIAL_REPORT.md>) | 说明新信息如何超出“告诉网络这是开放边界”；C开放边不等于全部必须封闭，更不等于来源真实C/Y接口。 |
| **R11 C内部邻接传播Graph** | Geo与Graph同参数、各500步，Graph对Geo原片有局部改善，新片RMS均值反而+0.4433%L，仅3/8改善。一个旧坏例贡献了某组99.51%的均值恢复。[S74](<../../native_t1/runs/geometry_context_revised_20260930_153000/GEOMETRY_CONTEXT_REPORT.md>) | 检验新增邻接信息对不同条件的独立收益，不因“用了图”就认为连接自然成立。早期纯T1置换门禁失败已另行诊断，不能冒充Graph训练负结果。 |
| **R13 MOMENT_XY** | 与H_CONT配对各1000步，真实梯度/更新正常；A20主MSE比H高2.96%，仅3/8父改善，正确条件列均9/24、三列全对均0/8；成本更高。按用户要求封存。[S62](<../../native_t1/runs/chair_moment_channel_v2/CHAIR_MOMENT_CHANNEL_V2_REPORT.md>) | 必须是实质不同的表示/任务依据，先读退役档案；不能重建同一矩通道、延长预算并称新机制。旧GT指标不是今天开放生成的唯一判据。 |
| **R16 Direct Context Readout，MIXED/CLEAN memory** | A/B/C各1000步、288评价。B/C相对FM没有一致质量收益；C相对B在7/8父降低FF，但CF/I、覆盖、尾距有反向代价。四层读取已实际训练，不是代码迁移失败。[S1](<E:/MeshFlow-Control-artifacts/EXPERIMENT_REPORT.md>) | 明确新读取提供了什么旧支路没有的信息，证明能改善采用/接续；不能单凭CLEAN名称推断known hidden污染是根因。 |
| **R17 READOUT精度路径修复** | BF16加回确有舍入；所有24个同状态开关对仍影响预测。FP32加回更忠实，但24/24分支预测增量反而减小；32份四格生成未得稳定用途收益。[S7](<E:/MeshFlow-Control-artifacts/followups/readout_effective_injection_v1/READOUT_EFFECTIVE_INJECTION_REPORT.md>) [S8](<E:/MeshFlow-Control-artifacts/followups/readout_precision_path_v1/READOUT_PRECISION_PATH_AND_ROLLOUT_REPORT.md>) | 不能把“残差被完全吞掉”作为已证实解释，也不能把推理FP32对照说成已做FP32训练。更高精度只是数值选择，不是已证实质量修复。 |
| **R28 FULL Contour Readout** | EXP67同起点、同47条件、同48000曝光/OT，block6后128维4头、17维逐查询几何、412032新增参数，训练6000步。终点采用16/24、接口8/24、联合6/24与FULL逐例相同；RAW边距3.28205→3.34608%L，CF/I略升。中途7/24不替代终点。[S35](<../../experiments/r2_full_contour_readout_v1/REPORT.md>) | 重新提议“空间读取／轮廓支路”时先指出与本版本在信息、分配或优化上的区别，以及可证伪预测。不能仅改模块名、加宽或盲目延长。 |

### 几何监督、辅助表征与训练组织

| ID／旧名称 | 实际证据与停止原因 | 重试至少应补什么 |
|---|---|---|
| **R07 FC/FF共享关系、C_gate、FC-only/FF-only、后置FC、M10** | 源顶点副本一致性和位置锚定确有减少部分相交的信号，但常伴Gamma尾距/覆盖代价。时间门控、2×2拆项及成熟A后再加FC都做过；后置FC相对等预算FM未保住独立FF收益。Geo+M10降低CF，但接口尾部仍恶化。[S10](<../../experiments/mf_h1_local0/outputs/c_relation_diagnosis_20260929_204330/C_RELATION_DIAGNOSIS_REPORT.md>) [S25](<../../experiments/mf_h1_local0/outputs/t1_time_gate_20260929_213000/TIME_GATE_RELATION_REPORT.md>) [S22](<../../experiments/mf_h1_local0/outputs/t1_gated_loss_factorial_20260929_225800/GATED_RELATION_FACTORIAL_REPORT.md>) [S9](<../../experiments/mf_h1_local0/outputs/a_then_fc_20260930_001700/A_THEN_FC_REPORT.md>) [S26](<../../experiments/mf_h1_local0/outputs/tgeo_m10_20260930_173200/T_GEO_M10_REPORT.md>) | “共同顶点应一致”“只在低噪声加”“先训FM再加”“拆成FC/FF”都不是新方案。新设计要解释怎样避免位置/离散、局部覆盖的旧取舍；L_FF不是FF相交指标。 |
| **R08 六项几何loss筛查L1—L6** | L1局部侧向、L2相邻夹角、L3边界线切向放松、L4表面Sinkhorn、L5面积—无向法向测度、L6接口薄带覆盖；六项+FM各500步。开发选中的L4在新噪声上RMS+24.21%、Gamma+35.98%；其他项只有开发局部取舍，不能补选赢家。[S16](<../../experiments/mf_h1_local0/outputs/loss_screen_20260930_232200/LOSS_SCREEN_REPORT.md>) | 新loss先查这些目标是否同义；需要不同的监督身份/生成分布处理等实质变化。不能只展示目标loss下降或重新扫lambda。未确认的其他项不写成全被独立复验否定。 |
| **R09 J35／Edge-only／JEdge5／Domain Joint／UCorr** | 已做真实源C/Y边的对应监督、有限线段距离和组合；JEdge5在旧片有真实局部收益，未在新片/多父上稳定保住。新片对FM的RMS约+11.85%、Gamma p95+11.04%；Domain Joint覆盖四格均下降。后期解析对应边UCorr四格也未形成接合信号。[S83](<../../native_t1/runs/l3_l5_interface_pilot_20261001_104600/L3_L5_INTERFACE_REPORT.md>) [S71](<../../native_t1/runs/edge_factorial_20261001_123000/EDGE_FACTORIAL_REPORT.md>) [S60](<../../native_t1/runs/chair_domain_control_v1/CHAIR_DOMAIN_CONTROL_REPORT.md>) [S53](<../../native_t1/runs/chair_boundary_attachment_v1/CHAIR_BOUNDARY_ATTACHMENT_FACTORIAL_REPORT.md>) | “发明一个边界loss”必须说明如何超出既有对应端点/有限边/解析积分监督。JEdge5仍是显式可选历史配方，不是完全无收益，也不能仅凭源码存在视为默认。 |
| **R10 Geometry-derived Partner Closure／P_Time／P_Conf** | 两臂各500步；48个teacher状态伙伴+端点正确率100%，不代表自由生成配对可信。四模型各40输出，合法闭合均0/40；160份中125锚冲突，35候选全非法；候选位移10.16%—57.22%L远超0.5%预算。置信仅拒绝1/37720对。路线已退役。[S51](<../../native_t1/maintenance/pair_cleanup_20261002_100559/evidence/GP_CLOSURE_REPORT.md>) | 显式伙伴、端点身份、全局匹配再闭合并非没试过。须处理自由生成状态上的配对可靠性、全局可行性与C锚冲突；不能用teacher配对正确率代替rollout。 |
| **R12 解析几何表征对齐Angular／AngularScale** | 三臂各500步。属性cosine可学到0.982，但完整生成未稳定改善，FF、尾部与覆盖有代价；Scale预测方差仅目标约1.35%。固定解析教师，第9层。[S77](<../../native_t1/runs/geometry_rep_alignment_20260930_205100/GEOMETRY_REP_ALIGNMENT_REPORT.md>) | 教师信息与生成目标的联系需要新证据；不因辅助头拟合好就扩大训练。不是对所有外部教师或所有表示学习的否定。 |
| **R14 真实轨迹反馈TR／CAPPED500/1000／梯度预算G** | 初轮CAPPED500相对匹配teacher旧S+6.0072pp是真实记录；同权重FRESH变−5.7789pp。新流L500−T500为OLD−7.1538/FRESH−3.8318pp；完整辅助梯度rho=1限制确实生效，仍未得到联合收益。[S58](<../../native_t1/runs/chair_control_feedback_auto_v1/CONTROL_FEEDBACK_AUTONOMOUS_REPORT.md>) [S56](<../../native_t1/runs/chair_capped500_offline_v1/CAPPED500_DEEP_DIVE.md>) [S54](<../../native_t1/runs/chair_capped500_gradbudget_v1/CAPPED500_REPLICATION_GRADIENT_BUDGET_REPORT.md>) | 先说明监督关系为何会更可信、能在新噪声兑现；“辅助梯度太大，clip一下即可”已经试过。loss值cap、梯度范数cap、AdamW实际更新大小必须区分。 |
| **R15 CAPPED权重半幅HALF** | 全194参数取START与历史CAPPED500的0.5均值，零训练，32新输出。保留一点平均S收益，但父中位数/两噪声不稳定，仍有新的重层和严重I代价；不晋级。[S55](<../../native_t1/runs/chair_capped500_half_v1/CAPPED500_HALF_REPORT.md>) | 仅检验过alpha=.5，不能宣称所有插值无效；但没有新机制证据时不沿比例盲扫。 |
| **R18 上下文教师对齐SIE：B/free与C/known** | 三臂各1000步。B相对A的ALL S约+1.100pp、NEW−.026pp；C没有跨父新噪声稳定增量，个别I尾部严重。两者均未过原冻结筛选。[S3](<E:/MeshFlow-Control-artifacts/experiments/chair_contextual_alignment_sie_v1/EXPERIMENT_REPORT.md>) | 与R12解析教师、R16的B/C读取分支严格区别。辅助loss下降不是条件用途标签，不能把已观察小增量宣称稳定迁移。 |
| **R19 B更早接入／EARLY、LATE日程** | 从官方初始化的统一6000步条件适配已比较FM_ONLY/EARLY/LATE。最初1000步B对CF均值仅约−.043pp，FF/I和其他质量有反向代价；没有额外采用价值证据。[S4](<E:/MeshFlow-Control-artifacts/experiments/chair_early_alignment_schedule40_v1/EXPERIMENT_REPORT.md>) | “更早加B，后期关掉”已经做过；若新任务值得重测，要说明数据合同/机制差异，而非把提前接入再当新想法。 |
| **R20 条件比例与HYBRID时序、课程训练** | 原配方试验累计42000更新。R2全程混合相对R1全程40%有部分FF收益；20→40→混合课程首轮优势在独立流与噪声下反向。R2保留为简单研究配方，没有独立确认其对R1普遍更优。[S2](<E:/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1/EXPERIMENT_REPORT.md>) | 不再原样重跑比例/课程网格。新的USER条件与旧C20/C40不是同一任务，不能把旧比例优劣当USER配方定律。 |

### 推理干预、求解器与去噪

| ID／旧名称 | 实际证据与停止原因 | 重试至少应补什么 |
|---|---|---|
| **R03 免训练三角片软匹配引导** | 冻结官方模型、lambda5/20，两个训练父K1/K2。距离均改善，但严格一对一全顶点误差≤1e−3 bbox两强度均0/8；其余几何也会移动。[S21](<../../experiments/mf_h1_local0/outputs/soft_matching_20260928_210321/SOFT_MATCHING_REPORT.md>) | 有限外力的吸引不是面片精确保留。若只是需要保留C，已有Native硬保持；不能重复把guide下降作为成功。 |
| **R04 顶点/采样核斥力、整面R、边界S、GA/GR/GAR** | 顶点斥力漏掉面内部穿刺；GPU核能量下降但CF/FF可能上升。整面R减少CF却有FF/轮廓/成本代价，R+S未过组合门槛；确认未触发不能算确认失败。[S13](<../../experiments/mf_h1_local0/outputs/guided_native_geometry_20260929_114004/GUIDED_NATIVE_GEOMETRY_REPORT.md>) [S12](<../../experiments/mf_h1_local0/outputs/gpu_guidance_and_protected_weld_20260929_160351/GPU_GUIDANCE_AND_PROTECTED_WELD_REPORT.md>) | 能量代理需要真实交检和整体质量验证，不能默认更多核点或叠加两项会协同。当前局部“靠近”与合法接触的区别仍要保留。 |
| **R05 保护C焊接W／有界局部锚定修正** | 早期W合并顶点却引入大量非流形顶点，32份W顶点link均失败；后期有界ENDPOINT16份只改1份，strict接缝父均约+1.035pp，未触发TRAJECTORY。后者用户已明确退役。[S12](<../../experiments/mf_h1_local0/outputs/gpu_guidance_and_protected_weld_20260929_160351/GPU_GUIDANCE_AND_PROTECTED_WELD_REPORT.md>) [S50](<../../native_t1/maintenance/local_anchor_retirement_20261006/evidence/BOUNDED_ATTACHMENT_CORRECTION_REPORT.md>) | 不能直接扩大焊接半径，也不能说轨迹干预已经做完失败。后来的GEO_C有独立实测收益，不能和旧W失败混为一条实现。 |
| **R21 Euler步数、FP32、EMA、高阶/非均匀求解器** | JEdge诊断做过50/100、FP32、后缀、局部误差、RK4/Dopri、Midpoint/Heun及等50NFE新噪声确认。个别失败确能改善；等NFE确认J的FF在17/20变差，覆盖收益不稳定。没有“全部只是Euler50误差”的证据。[S79](<../../native_t1/runs/jedge5_diagnosis_20261002_102215/JEDGE5_DIAGNOSIS_REPORT.md>) [S78](<../../native_t1/runs/jedge5_blackbox_20261002_113600/BLACKBOX_REPORT.md>) [S80](<../../native_t1/runs/jedge5_local_error_B_20261002_114200/B_LOCAL_ERROR_REPORT.md>) [S81](<../../native_t1/runs/jedge5_numeric_reference_A_20261002_115426/A_NUMERICAL_REFERENCE_REPORT.md>) [S82](<../../native_t1/runs/jedge5_solver_pilot_20261002_121000/SOLVER_PILOT_REPORT.md>) [S72](<../../native_t1/runs/equal_nfe_confirmation_20261002_152000/EQUAL_NFE_CONFIRMATION_REPORT.md>) | 给出当前模型的具体数值故障证据后再安排有限对照；不把旧四例局部修复泛化为统一求解器升级。 |
| **R22 Anchor-aware denoiser** | 官方D0普通适配与known角色适配各1000步。USER_MIX有共边输出1→2→2，均集中同一靠背父，角色没有额外连接收益；ANCHOR对D0 FF+1.392pp，合成free/interface MSE约+22.40/+21.33%。[S27](<../../experiments/r2_anchor_aware_denoiser_v1/REPORT.md>) | “增加C不可移动标记再微调”已试。旧绕序限制后来在另一批36输出修正，不能因此抹掉该实验角色无增量和去噪退化。 |
| **R23 官方D0重复3/5次** | FF12.02→8.02→6.12%，PMR未改善，I略升；33ec座面n0出现整组腿丢失，D5 FF=0也不能称更好。[S37](<../../experiments/r2_multistep_denoise_v1/REPORT.md>) | 不能以FF最低选择轮数；须有能保留薄构件/主体的新干预。不是噪声日程，也不是官方证明的多步采样。 |
| **R24 单步学生蒸馏S8_W512_KD** | 4000步，455.59M→38.39M参数；USER_MIX残差误差比恒等基线1.1649，FF教师12.02%→学生60.98%，仅GEO66.13%。整段POST只约1.11倍快。属于预算内拟合不足，非蒸馏原理不成立。[S43](<../../experiments/r2_single_step_denoiser_distillation_v1/REPORT.md>) | 先证明能保住教师去噪质量并找到真实计算瓶颈，再谈扩蒸馏；参数缩小不等于整条生成流程等比提速。 |
| **R25 学生重复去噪3/5次** | 同一final4000学生，5次对1次FF仅−.647pp、6好6坏，非法面对总数略增，C附近冲突/删面/小角代价更大。[S46](<../../experiments/r2_student_multistep_denoise_v1/REPORT.md>) | 重复一个拟合不足的学生并没有补回教师质量。需要不同误差机制证据，不自动增加轮数。 |
| **R26 D0对C的自预测FEEDBACK** | 固定F_RAW，内部C两次反馈；C预测误差12/12下降，均值.043324→.017103，但没有足够接合收益，四座面仍未接上。[S31](<../../experiments/r2_c_restore_coupling_v1/REPORT.md>) | 自己预测准C不是F会接到真实C的充分目标；新的反馈必须直接说明如何改善F关系，而非只优化内部C误差。 |
| **R27 有界非负BLEND协调恢复** | 同12份D0 proposal，压低RBF两个I尾部，但距离12/12弱于RBF、合法接口3/12→2/12、边9→3，FF更高。[S30](<../../experiments/r2_c_restore_bounded_v1/REPORT.md>) | 已试过温和的局部非负融合；降低改动幅度会同时失去连接。新限制要有选择哪些修正值得保留的证据。RBF本身见后文局部正结果。 |

## 训练前阻塞，不是假装跑过的负结果

| ID | 实际发生了什么 | 能下的结论／未来前提 |
|---|---|---|
| **B01 Localized Boundary Cancellation** | CPU解析门禁出现loss下降而裂缝变大；法向相反的重叠双面可假抵消到约1e−34。正式模型前向、训练、生成均0。[S52](<../../native_t1/runs/boundary_cancellation_20261001_161219/BOUNDARY_CANCELLATION_REPORT.md>) | 当前数学代理有明确反例；必须先修复并通过这些反例，不能直接开训。但没有模型层面的质量负结果。 |
| **B02 Finite Surface Attention** | v1日志器故障随后修复；192点求积在7504对子中1459不达.01门槛。独立高精度参考支持晚期窄核欠分辨；修复版两个40%任务forward约3.05/3.14倍，超原3倍线。正式更新仍0。[S64](<../../native_t1/runs/chair_surface_attention_v1/CHAIR_FINITE_SURFACE_ATTENTION_REPORT.md>) [S65](<../../native_t1/runs/chair_surface_attention_v1r1/CHAIR_FINITE_SURFACE_ATTENTION_R1_REPORT.md>) | 不是“implementation没有所以放弃”，也不是训完证明surface attention无效。要有精度/成本达标的新实现，不能跳过旧反例或放宽口径后称同方案通过。 |

原 `R2_USER_DATA_SCALING_12H` 是 **DATA_SCALE_BLOCKED**：实际67训练父，未达到原128门槛，两臂没有训练。后续用户授权的32/67试验已实际完成。这两件事必须分别记；本地没有做成32/256，不能把67重复采样冒充256，也不能说扩规模第一次就训练失败。[S47](<../../experiments/r2_user_data_scaling_12h_v1/REPORT.md>) [S48](<../../experiments/r2_user_data_scaling_32_67_v1/REPORT.md>)

## 不应当一起扔掉的进展

| 保留项 | 正证据 | 明确限制 |
|---|---|---|
| Native/T1、Geo与HYBRID/R2基础 | Native相对独立条件对照有可学习信号；Geo部分新片覆盖改善；HYBRID在A20对BASE的原查询MSE约降40%、6/8父改善。后来建立统一R2条件后训练入口。[S17](<../../experiments/mf_h1_local0/outputs/native_inpainting_20260928_233911/NATIVE_INPAINTING_COMPARISON_REPORT.md>) [S24](<../../experiments/mf_h1_local0/outputs/t1_native_confirmation_20260929_171056/T1_NATIVE_CONFIRMATION_REPORT.md>) [S74](<../../native_t1/runs/geometry_context_revised_20260930_153000/GEOMETRY_CONTEXT_REPORT.md>) [S61](<../../native_t1/runs/chair_hybrid_coupling_v1/CHAIR_HYBRID_COUPLING_REPORT.md>) [S2](<E:/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1/EXPERIMENT_REPORT.md>) | 来源重建与几何响应不等于开放完成可靠；Graph、Joint并未因此获认证。 |
| USER条件组织与32→67扩父 | USER从早期参与有局部座面用途收益；新32父POST的EXP67对SMALL CF−.700pp、FF−2.532pp。[S5](<E:/MeshFlow-Control-artifacts/experiments/chair_r2_user_conditions_v1/DATA_PREPARATION_REPORT.md>) [S6](<E:/MeshFlow-Control-artifacts/experiments/r2_user_condition_finetune_pilot_v1/R2_USER_CONDITION_FINETUNE_REPORT.md>) [S49](<../../experiments/r2_user_mix_from_official_v1/R2_USER_MIX_REPORT.md>) [S48](<../../experiments/r2_user_data_scaling_32_67_v1/REPORT.md>) | I+.648pp、合法接口仍7/128，明确采用59→63同时明确未采用30→33；不能说扩大数据已解决，也不能说数据完全无用。 |
| 集中200步快筛 | 有局部可学性；新噪声熟悉面板合法接口12→15/32，未丢原接口。[S42](<../../experiments/r2_seat_quick_learnability_v1/REPORT.md>) [S41](<../../experiments/r2_seat_noise_transfer_v1/REPORT.md>) | RAW明确采用仍16→16；原3f正例两模型均5/8。DEV只有弱信号；不能认定集中曝光稳定提高采用，也未证明只是记忆。 |
| C参考坐标CREF | 对SMALL出现7份明确NO→YES，分布5条件，说明无新增参数也能改变用途。[S29](<../../experiments/r2_c_reference_frame_v1/REPORT.md>) | 熟悉I1.050→8.737%，接口15→14/32；DEV1→0/16。平移参考和Gaussian起点一起改变，不是纯坐标等价变换。 |
| 完整座面FULL_S6000曲线 | 47父47条、6000新增更新；RAW采用12→16/24，有限free边距离4.762→3.282%L，18/24更近。[S36](<../../experiments/r2_full_seat_learning_curve_v1/REPORT.md>) | 接口7→8，新增采用与新增接口不在同一批；联合7→6，DEV仍无明确联合正例，FF等代价上升。保留充分训练基线，不能说FM完全学不动或继续加步数必好。 |
| 官方单次D0＋GEO_C | 48/48 FF下降；USER_MIX约89.65→12.02%，不仅是删除坏面改变分母。[S38](<../../experiments/r2_postprocess_avmr_pmr_v1/REPORT.md>) | 座面接续仍差、薄腿可能损伤；不把去噪质量当采用保证。旧START报告“缺D0”只描述当时资产状态。 |
| RBF协调恢复 | 同12份USER_MIX，合法边1→9、接口输出1→3，距离11/12改善。来源定向后9边均可一致定向，3份进入大free分量。[S31](<../../experiments/r2_c_restore_coupling_v1/REPORT.md>) [S45](<../../experiments/r2_source_winding_export_v1/REPORT.md>) | 跨父仍只有2/6，4座面0；FF约+.530pp、I+.704pp。是可选局部候选，非普遍接合路线。 |
| 来源角序恢复＋固定C定向 | 36份验证恢复正确C绕序后，C可达合法图达到无向上限；RBF方向可达free面积宏均.2643→19.9459%。[S45](<../../experiments/r2_source_winding_export_v1/REPORT.md>) | 未新增几何边；9/12仍0，纯free矛盾与非流形仍在。只修正有映射、实际验证过的输出，不给所有旧失败翻案。 |

## 已经排除的简单解释，和仍然未知的部分

1. **“模型完全没看到C”不符合当前EXP67证据。** 64组固定状态均有free速度响应，16条宽度曲线均正向变化；同面板全部48 RAW/POST仍没有合法共边。响应不等于采用，更不等于接续。[S32](<../../experiments/r2_condition_width_response_v1/REPORT.md>)
2. **“座面训练样本几乎没有”不成立。** EXP67座面约49.12%真实曝光；DEV条件质心高度都落在同类型训练实际范围内。联合几何覆盖仍可能不足，边际范围不等于IID。[S40](<../../experiments/r2_seat_failure_diagnosis_v1/REPORT.md>)
3. **“训练答案根本没示范接续”不是普遍解释。** TRAIN67+DEV32的258条来源条件均有合法C/Y接口；98/98训练座面也成立。完整来源依然可能有远端自交，不能反向认证全数据干净。[S44](<../../experiments/r2_source_interface_audit_v1/REPORT.md>) [S40](<../../experiments/r2_seat_failure_diagnosis_v1/REPORT.md>)
4. **“全是在学重复座面”没有得到支持。** 98座面中Y任意位置精确重复较多，但平行且近C≤1%L的重复仅2条、1父，约2.04%座面曝光；不能把任意位置缺陷都叫重复座面。[S40](<../../experiments/r2_seat_failure_diagnosis_v1/REPORT.md>)
5. **“只是一个三角形太难；改两张就行”不成立。** 干净K2小试、完整K≥2面板都做过；新两三角矩形仍可错层且无接口。保留人类合理输入偏好，不把K1继续当主面板，也不因此宣称应改四边形网络。[S66](<../../native_t1/runs/chair_two_triangle_control_v1/REPORT.md>) [S39](<../../experiments/r2_regular_patch_control_v1/REPORT.md>)
6. **“同一表面多细分一下就能修好”没有证据。** sub4显著改变输出但8/8仍无合法接口，且同时改变K、N和部分N嵌入档位，不能据此单因定位attention稀释。[S40](<../../experiments/r2_seat_failure_diagnosis_v1/REPORT.md>)
7. **“只是后处理没焊上”不成立。** RAW已经存在法线/切向轮廓错位；长边中间会有缺口，哪怕端点匹配。放宽焊接不替代轮廓到位。[S34](<../../experiments/r2_full_boundary_attribution_v1/REPORT.md>)
8. **“I低就是C用得好；I_dup高都违法”均不成立。** 生成离C很远也可低I；正常厚底面、重复上皮和混合情况均出现。逐点检测是技术事实，合法用途还需有限表面/剖面和整体判断。[S8](<E:/MeshFlow-Control-artifacts/followups/readout_precision_path_v1/READOUT_PRECISION_PATH_AND_ROLLOUT_REPORT.md>) [S33](<../../experiments/r2_cref_layer_audit_v1/REPORT.md>)
9. **“200步不够，补6000就行”已经检验过有限版本。** FULL6000改善采用和轮廓，但接续/代价不同时兑现；CONTOUR同预算也未扩成功覆盖。不能把每次阴性都推给预算不足。[S36](<../../experiments/r2_full_seat_learning_curve_v1/REPORT.md>) [S35](<../../experiments/r2_full_contour_readout_v1/REPORT.md>)
10. **“只有训练里的某几种pattern能成功”仍是待检验假说。** 已见C也随噪声失败，DEV有有限用途改善；一个近重复TRAIN/DEV匹配曾被识别并披露。不能凭两张像图断言坍缩，不能将已观察DEV称全新测试。[S40](<../../experiments/r2_seat_failure_diagnosis_v1/REPORT.md>) [S41](<../../experiments/r2_seat_noise_transfer_v1/REPORT.md>)

更准确的未知是：**在自由生成中，什么训练/表示关系能稳定把C的局部轮廓转成相容的F，并在新条件上兑现？** 现有记录没有唯一定位到数据、loss、attention或离散连接中的一个原因。这个问题不能靠继续解释同一批坏例自动解决，也不能靠再次叠一个同类模块自动解决。

## 防止重复的具体规则

- 新方案先写：最接近的本表ID、原失败条件、这次实际改变的一项、预计哪项观测会不同、固定停止点。**没有实质差异就不建议原样重跑。** 不把本表变成必须先做大审计或多seed才能启动的小试门槛。
- 重点交叉检索：新“边界loss”→R07/R08/R09/B01；新“伙伴/接口身份”→R10；新“读取支路”→R01/R06/R11/R13/R16/R28/B02；新“减弱辅助”→R14/R15；新“teacher/B”→R12/R18/R19；新“去噪迭代/蒸馏”→R22—R27。
- 用**等预算普通FM**区分续训和模块增量；不只与没续训起点比较。复用已完成FULL曲线时核对实际输入流/OT、初始化与优化器，不为查重重新训练基线。
- 固定主终点，不挑CAPPED500式好看的中途点替代预定终点；旧噪声回归与新噪声确认分开。新采样噪声不是新父，也不是独立训练重复。
- 当前任务是保留真实C的开放式生成。来源GT只作监督来源/诊断，不以唯一款式距离作为普遍成功标准。旧C20/C40回归不再作为USER主任务自动否决项，旧结论按当时合同保存。
- 采用、合法共边、共享长度、图可达面积、CF/FF/I、整体完整性分别报告；联合成功必须在**同一请求**成立。不能把“新增采用”和另一请求“新增接口”相加。
- A-VMR/PMR只匹配F顶点；即使端点全匹配也不认证长边中段接好。全部C开放边没有自动升级为用户必接边界，不恢复已暂停的S_join。细分改变顶点分母应披露。
- teacher插值状态、Yhat局部误差、梯度和属性头拟合不能替代rollout。Pair100% teacher配对、表征cosine.982、FEEDBACK的C误差下降已给出直接反例。
- 旧3×3单元评分正确，但“correct-column”曾误用了固定输出行；后续归因已修正117/216项优势符号/严格胜负。以后保存完整矩阵，No-response重复行正控应列优势0、严格赢0。不能用旧错轴结论讲机制故事。[S57](<../../native_t1/runs/chair_control_attribution_v1/ATTRIBUTION_REPORT.md>)
- 绕序合同、确定性、dtype、可逆C映射和渲染深度排序是应保留的工程修正。显示蓝C被错误遮挡不是生成改善；来源定向没有增加新连接。[S11](<../../experiments/mf_h1_local0/outputs/debug_repeatability_20260928_150208/DEBUG_REPORT.md>) [S75](<../../native_t1/runs/geometry_gate_code_audit_20260930_150400/CODE_PATH_DIAGNOSIS_REPORT.md>) [S76](<../../native_t1/runs/geometry_precision_audit_20260930_152000/PRECISION_DIAGNOSIS_REPORT.md>) [S39](<../../experiments/r2_regular_patch_control_v1/REPORT.md>) [S45](<../../experiments/r2_source_winding_export_v1/REPORT.md>)
- 质量改善不能来自选择有利指标或删坏样本。多次D0能把有腿椅变成FF=0的残缺体，是低相交不等于好成品的实测警告。AI_PROXY不是人评；本轮没有重新看图，全部视觉结论引用原记录。

## 退役资产与证据保存

Pair、MOMENT、局部锚定及CAPPED有明确退役/删除记录；2026-10-14训练前又删除66个旧权重文件，约92.065GiB逻辑字节。**旧报告里的“权重已保留”只代表当时，不能直接执行其中恢复命令。** 本轮没有删除或恢复任何文件，也没有加载大权重重新hash；资产可恢复性以对应清理清单和实际文件检查为准。

- [Pair清理](../../native_t1/maintenance/pair_cleanup_20261002_100559/CLEANUP_REPORT.md)
- [MOMENT清理](../../native_t1/maintenance/moment_retirement_20261005/CLEANUP_REPORT.md)
- [局部锚定清理](../../native_t1/maintenance/local_anchor_retirement_20261006/CLEANUP_REPORT.md)
- [CAPPED清理](../../native_t1/maintenance/capped_retirement_20261007/CLEANUP_REPORT.md)
- [近期训练前清理](../../experiments/r2_full_seat_learning_curve_v1/cleanup/CLEANUP_REPORT.md)

原报告、原评分、RAW、代码快照继续作为历史证据；本总结不回写它们。`source_inventory.json`登记主报告的路径/大小/SHA；`read_ledger.jsonl`记录工具提取范围，**不等于逐字人工阅读或全量原始数组复算认证**。长报告的嵌入JSON不重复展开为新证据。此次查重遗漏风险主要在未命名的小型试跑与已删除产物；83份不是对所有历史命令的穷尽计数。

本次只固化研究记忆。下一步先根据这些已知边界形成真正不同、可证伪的小实验问题；没有因此自动授权任何新训练、loss、求解器扫描或模型切换。
## 2026-10-14 Partial-edge binding CPU pilot completed

R2_PARTIAL_EDGE_BINDING_V1 executed all 24 FULL_S6000 outputs x A/B x 1/2/3.5%L = 144 cells, zero model/GPU calls. Original POST representatives and actual GEO_C clusters were frozen; no radius reclustering. A reassigns selected corners; B assigns the original free vertex cluster to exact C indices. All six policies preserve the original 6 joint positives and all 62 legal CF edges; zero face deletion and zero newly illegal CF pairs.

Joint screening counts A=8/8/9, B=7/8/8 of24. Original legal FF edge losses A=12/16/27, B=0/0/0. Newly illegal FF pairs A=0/1/4, B=0/4/4. B2's two conversions are TRAIN73 n0/n1, DEV remains0/12. Only A3.5 adds DEV2d n1 with tearing costs. All608/610 requests have zero eligible candidates even at3.5%; this is a candidate-coverage limit of whole-edge endpoint correspondence, NOT evidence that attempted binding destroyed them. Do not infer a physical upper bound, global post-hoc impossibility, or pointer learnability. Whole-edge one-to-one matching does not test edge-subsegment bindings.

34 paired images actually reviewed by nonblind AI_PROXY (6 contact sheets plus3 full-size cases), including all12 new joint arm/request entries. C's seat role remains visible; local cracks/full seams are not visually certified. Formal binding/scoring151.24s,2CPU workers; this excludes preparation/review. Independent144-cell audit and9880 cached scalar comparisons PASS,611 protected files unchanged. One before-decode import failure and hash amendments retained. Current default unchanged; no training/commit/push. Report: [REPORT.md](../../experiments/r2_partial_edge_binding_v1/REPORT.md). Resume: r2/tools/partial_edge_binding_v1/resume.py (completed runs verify only).

