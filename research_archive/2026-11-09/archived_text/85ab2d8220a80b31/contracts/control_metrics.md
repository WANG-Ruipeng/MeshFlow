# Chair Control 完整指标库与评价记忆

版本：control_metric_policy_v1；记录日期：2026-10-07。

用户授权将本轮 35 类指标建议审阅后写入仓库长期记忆。本文件保存任务定义、指标字典和选择规则；**它是评价规范，不是已完成的统一评分库，也不触发训练、生成、安装依赖或批量重评。** 当前 baseline 仍为 [Chair HYBRID START](chair_start.md)。

目标是给定真实三角面 C 及其需要保留的关联信息，生成满足明确条件的多种合理完整网格；不默认唯一来源模型就是唯一正确完成。仍可从完整训练网格抽取 C 与剩余面构成监督训练对。

<a id="meshfim-boundary-metrics"></a>

## 2026-10-10 补充记录：A-VMR／PMR【文献；待接入】

**用户决定：先记录 A-VMR 和 PMR，不推进新增 S_join 的定义、实现或实验。** 下方原35类指标和历史S定义保留，不据此删除或重写旧结果。本补充不代表已冻结全部论文主指标。

来源：[MeshFIM: Local Low-Poly Mesh Editing via Fill-in-the-Middle Autoregressive Generation，arXiv:2605.08744v1，附录A，式(8)–(11)](https://arxiv.org/html/2605.08744v1#A1)。
状态：来源类别 EXTERNAL；本地接入 NOT_IMPLEMENTED；本次运行 NOT_RUN。只记录文献定义和适用边界，没有计算新模型分数。

对样本 \(s\)，令 \(V_{\partial}^{(s)}\) 为登记的输入边界顶点集合，\(V_F^{(s)}\) 为新生成free部分的顶点集合。原文以欧氏距离 **严格小于** \(\tau=10^{-6}\) 判定匹配：

\[
V_{\mathrm{match}}^{(s)}
=\{v\in V_{\partial}^{(s)}:
\min_{u\in V_F^{(s)}}\|v-u\|_2<\tau\},
\qquad
r(s)=\frac{|V_{\mathrm{match}}^{(s)}|}{|V_{\partial}^{(s)}|}.
\]

**A-VMR（Average Vertex Matching Ratio，平均边界顶点匹配率）**：

\[
\mathrm{A\text{-}VMR}=\frac1n\sum_{s=1}^{n}r(s).
\]

先计算每个样本的匹配比例，再对样本取平均；不是将所有样本顶点混在一起算一个比例。

**PMR（Perfect Matching Ratio，完全边界顶点匹配的样本比例）**：

\[
\mathrm{PMR}=\frac1n\sum_{s=1}^{n}
\mathbf1[|V_{\mathrm{match}}^{(s)}|=|V_{\partial}^{(s)}|].
\]

两项均越高越好。公式取值0–1，若乘100以百分比展示须标明；这里的 \(n\) 为评价样本数，不是每个mesh的总面数N。

后续接入时遵守：

- **只匹配新生成F的顶点。** 不能使用完整输出 \(C\sqcup F\) 的顶点集合，否则复制的C会形成自匹配。逐例保存匹配数、边界顶点数、比例和指标版本。
- 先登记哪些输入边界要求接续、顶点集合的身份/去重规则及RAW或POST阶段，不默认所有C开放边都必须接合。上述匹配计算不需要唯一完整GT，但需要已知的待评边界；改变原论文的边界域、顶点资格或匹配规则，必须标为适配版本。
- \(10^{-6}\) 是论文坐标系中的原始欧氏距离阈值，不是平方距离阈值，也不是已确定的本地模型单位阈值。接入前核对坐标归一化；预先固定尺度和容差，不能用每份输出bbox重新归一。扩展容差须以 A-VMR@tau／PMR@tau 等独立设置标明，不能混作原版结果。
- 没有指定待评边界或边界集合为空时记NA及原因，不能按空集“全部匹配”记PMR成功。有非空待评边界而合法评分输入的F顶点为空时匹配数为0。执行失败、缺文件和未评分保持各自状态，不静默删除。
- **顶点匹配不认证拓扑接合。** 两项不验证完整共享边、有效相邻面、无穿插/重层或接入主体；不能命名为水密率或真实接合成功率。实际共边、CF/I和质量证据仍分开保留，RAW与POST不能混报。

本次不实现评分器，不启动重评分、训练、生成、后处理或新的loss；不修改原评分器、权重、RAW及历史报告，不自动commit/push。

### 2026-10-10 后续授权更新：已完成固定面板离线接入

用户随后明确要求实际估计。当前局部版本 r2_avmr_pmr_all_C_open_boundary_v1 已实现并执行；此前NOT_IMPLEMENTED／NOT_RUN及“本次不实现”描述的是最初仅记录的阶段，不代表此次执行后的状态。

代码：[固定面板CPU入口](../../r2/tools/avmr_pmr_v1/run.py)；[定义与结果](../../experiments/r2_avmr_pmr_offline_v1/REPORT.md)；[结果解读](../../experiments/r2_avmr_pmr_offline_v1/FINDINGS.md)。状态为 IMPLEMENTED / COMPLETE，仅限4模型×6既有USER条件×2噪声的48份RAW；没有接入通用评分API或改变旧评分。

本轮使用C源顶点ID拓扑全部开放边界作描述性匹配域（ALL_C_SOURCE_TOPOLOGY_OPEN_BOUNDARY），只查F角点、严格L2阈值、逐输出宏平均。需接合接口仍NA_NOT_REGISTERED；paper_scale_compatibility为UNKNOWN。记录模型单位1e-6及固定L的7档预注册阈值，不能混作论文原尺度结果，也不称控制/真实接合成功率。

8项确定性指标检查通过；独立cKDTree实现复核384评分行、496逐顶点距离、96汇总和288配对无差异；77原文件hash未变。当前固定入口的6条边界都非空，不宣称任意空边界输入已获得统一NA接口处理。S_join不推进，未训练、生成或执行POST。

### 2026-10-10 后续执行：官方去噪＋保持C的POST复评完成

同一48份RAW已执行48次官方Chair denoiser单次前向、C原字节恢复和固定GEO_C，独立目录[完整报告](../../experiments/r2_postprocess_avmr_pmr_v1/REPORT.md)，[逐例评分](../../experiments/r2_postprocess_avmr_pmr_v1/per_output.csv)。未重抽噪声或重新生成；48 C逐位不变。原评分源码、RAW和历史定义保留；POST质量分母改用实际保留free数，冻结C2048查询点不变。

同一边界域/同一8档容差对RAW与POST配对。严格模型单位1e-6 POST A-VMR依FROZEN/OLD/USER/MIX为1.85/3.90/3.51/1.39%，PMR仍全0。MIX在1%L由6.75%变5.56%；仅OLD和MIX各一份有C/free实际共享边，不能把精确匹配当水密或主体接合。论文尺度兼容性仍UNKNOWN，必要接口未登记；S_join没有推进。

FF在48/48配对中改善（MIX89.65→12.02%），对保存面对的核账证明不仅由删面造成；但MIX I仅3.77→3.69%，小角17.18→17.06%，且有细腿损伤图像证据。今后同条件模型比较保留RAW和固定POST两阶段，独立报告C匹配、实际连接、冲突、删面/位移与图像代价；不得用一个收益掩盖另一个缺口。没有因此自动改loss、切模型或追加实验。原文件终态hash通过；一份父缓存的准备误写及按原hash恢复详见本轮事故记录。

### 2026-10-10 多次去噪补充证据：低冲突不等于完整成品

固定USER_MIX12输出的D1/D3/D5复评分已完成，见[完整报告](../../experiments/r2_multistep_denoise_v1/REPORT.md)。每轮恢复C，中间不Geo；终点各Geo一次。FF12.02→8.02→6.12%，I3.69→3.71→3.77%，PMR全0。33ec座面n0的D5虽FF=0，却在两视角失去整组支撑腿、小角54.55%。这是必须同时报告完整形状、删面和薄构件损伤的直接证据，不能只取FF最低点。严格A-VMR略升及共边输出1→2也未解决四座面边界匹配，新增共边没有通过双面反向绕序检查。所有阈值、配对及AI_PROXY局限保留；不新增S_join或改旧定义，不自动升为默认。

## 长期执行要点

1. 在每次已授权实验的协议/预算内，统一登记适用指标并完整保存结果及未计算原因；不要求每次强算全部35项。额外渲染、特征网络前向或新采样仍计入对应任务授权和预算。
2. 旧源码、RAW、评分缓存和历史报告只读。新增指标及修正使用独立版本和输出目录；不得把新口径回写成旧结果。
3. C保留、局部几何、完整形状分布、同条件变化是不同证据。集合指标不能证明每个输出正确利用对应C，精确复制C也不能证明接合成功。
4. 没有来源GT不算来源重建；没有指定接口不强算S；没有外露要求不最大化E。固定尺度不得随生成输出bbox变化。
5. 主文可收敛到三个核心指标，优先采用任务适用且可复现的客观量。三个名字目前未定；人评不是默认强制项。一般任意C的“语义相容”尚无本库已验证的通用客观单分数，保留这个缺口，不拿FID或复制C代替。
6. 开发集可探索主张与指标；正式评价前冻结选择。完整结果保留，严重C丢失、主体破坏等影响结论的代价要在主结论中披露。
7. 指标越好只支持其明确定义的性质。距离近不等于共享拓扑，低自交不等于合理椅子，多样性大不等于合理变化。

## 实现与证据状态

- 【现有】仅表示某个历史入口有相应诊断证据，不能推定全部域、所有字段、无GT输入或统一API已实现。登记每个具体字段时分别标 IMPLEMENTED（已核源码）、DERIVED（现有证据可派生但未算）、PROPOSED（新定义）、EXTERNAL（文献待适配）；该状态与每次运行的 value/status 分开。
- 【文献】表示有论文定义/使用依据；接入前仍要核对实际代码、倍率、特征、采样、预处理和版本。
- 【扩展／待接入】表示本库接受为候选定义，运行状态为 NOT_IMPLEMENTED 或 NOT_RUN，不能填写模型结果。
- 原始建议中将有效输出聚合、全域角度/退化、非流形顶点等概括为“现有”的表述已收窄；S_required、E_required、统一索引身份保持仍需独立实现。
- 本轮核验了本地S/I/E、CF/FF、来源距离、宽度/矩阵的定义，以及文献TMD、UHD、MMD/COV/JSD、KID等依据；没有运行新评分或模型。

## 本地核验锚点

以下历史源码/报告链接用于复核；部分路径受Git忽略、仅原工作区可用，不构成可移植依赖或恢复旧实验的授权。

- [START采样合同](chair_start.md)
- [S/I/E实际定义](../../experiments/mf_h1_local0/chair_boundary_attachment_v1/control_metrics.py)
- [来源距离、接口和多样性](../../experiments/mf_h1_local0/chair_control_usability_quick_v1/scoring.py)
- [CF/FF历史评分](../../experiments/mf_h1_local0/control_mvp48/eval_scoring.py)
- [宽度及3×3方向审计](../runs/chair_control_attribution_v1/ATTRIBUTION_REPORT.md)
- [RAW/POST现有证据](../runs/chair_start_postprocess_v1/START_POSTPROCESS_REPORT.md)

下文完整保留35类指标和公式，含审阅修订。公式只是统一说明；旧结果的最终定义仍由其冻结源码、输入与版本决定。

---

# 一、统一符号与计算约定

设：

| 符号 | 含义 |
|---|---|
| \(C\) | 输入的known三角面及其保留身份 |
| \(F\) | 新生成的free三角面 |
| \(M=C\sqcup F\) | 完整输出，\(\sqcup\)强调保留原面片身份，不承诺已经焊接 |
| \(Y\) | 来源模型中对应的参考free补集；开放式任务可以没有 |
| \(Q_A\) | 在表面 \(A\) 上按面积采样的点 |
| \(d(q,A)\) | 点 \(q\) 到**有限三角表面** \(A\) 的最近欧氏距离 |
| \(\mathcal E(F)\) | free的有限边段集合 |
| \(\Gamma\) | 任务指定的待接合边界 |
| \(L\) | 预先固定的距离归一尺度 |
| \(m\) | 同一C的固定生成候选数 |
| \(\mathcal G,\mathcal R\) | 完整生成网格集合、真实参考网格集合 |

**尺度必须写入指标版本。** 旧实验的 \(L_0\) 来自完整来源模型；无GT任务应使用固定规范坐标尺度或事先定义的输入尺度，不能随每个输出bbox变化，否则异常变大的输出可能被归一化掩盖。旧报告已经明确区分了距离除以 \(L_0\) 与平方距离除以 \(L_0^2\)。

下列公式用于统一说明。**旧指标仍以冻结源码的采样、归一化和聚合方式为准，不能仅凭名称相同就替换实现。**

---

# 二、输入保持与执行正确性

## 01｜C精确保留率、坐标漂移【现有】

利用已保存的known面映射 \(\pi\)，逐面检查：

\[
P_C=
\frac1{|C|}
\sum_{f\in C}
\mathbf1[
f\text{在输出中按约定精确保留}
].
\]

同时报告：

\[
D_C^{\max}
=
\max_i\frac{\|c_i^{\mathrm{out}}-c_i^{\mathrm{in}}\|_2}{L},
\]

\[
D_C^{\mathrm{RMS}}
=
\sqrt{
\frac1{n_C}
\sum_i
\frac{\|c_i^{\mathrm{out}}-c_i^{\mathrm{in}}\|_2^2}{L^2}
}.
\]

**期望：** \(P_C=1\)，漂移为0。

精确保留还包括面片多重集、绕序及输入提供的关联信息；重复的两张C面不能因为去重只剩一张。逐位保持与容差内保持应分开记录。当前 START 采样有首 K 个 FP32 三角面坐标、顺序和绕序的逐步核验，部分导出流程另有重读核验；不能据此声称任意原始 vertex-ID 关联均已实现保留。若输入带共享顶点身份，必须保留并独立审计映射；坐标相同不自动等于原始索引相同。后处理只审计终点时，不继承采样的 51 态证据。上面的保留率与漂移字段需按实际入口核对，尚非统一 API。

**适用范围：所有任务。它证明输入被保留，不证明free合理利用了C。**

## 02｜有效输出率与预算保持【已有执行账本；统一聚合待接入】

定义每个登记请求的执行有效标记：

\[
v_i=
\mathbf1[
\text{成功输出、坐标有限、C保持、面数符合对应输出合同}
],
\]

\[
R_{\mathrm{valid}}=\frac{\sum_i v_i}{n_{\mathrm{requests}}}.
\]

RAW记录 \(N_{\mathrm{raw}}\)、\(N_{\mathrm{free}}\)；后处理另记录 \(N_{\mathrm{final}}\)。删面后不能继续声称最终严格保持原N。

**适用范围：所有任务。有效输出不等于好椅子，更不等于水密网格。** 该比率用于登记请求均已结案的协议；另报 requested、attempted、completed、failed、blocked、not_run 数。实际失败不能删出分母；未执行或缺资产不能伪装成模型失败。未结案时只报已知有效数/请求数的完成下界和未决数，不称为最终模型成功率。执行状态、输出合同状态、各指标评分状态分开；指标 PARTIAL/NA 不自动意味着生成失败，只剩 C 是否有效由任务合同预先决定。

---

# 三、条件接合与C附近的几何关系

## 03｜接口距离、AttachCov【现有】

在 \(\Gamma\) 上按弧长采样 \(q_j\)，权重 \(w_j\) 表示对应弧长：

\[
D_{\Gamma,\mathrm{edge}}^{\mathrm{RMS}}
=
\sqrt{
\frac{\sum_jw_jd(q_j,\mathcal E(F))^2}
{L^2\sum_jw_j}
},
\]

\[
\operatorname{AttachCov}(\tau)
=
\frac{
\sum_jw_j\mathbf1[d(q_j,\mathcal E(F))<\tau L]
}{
\sum_jw_j
}.
\]

另报告加权p95。将目标换成 \(F\) 的三角表面，可得到接口到表面的版本，**但不能与到边版本混名**。

**距离越低、覆盖越高越好；仅在存在指定接合接口时适用。**

它只检查“附近有边／表面”，不检查是否真正形成合理接缝。

## 04｜S：有效接缝覆盖率【现有；无GT接口版本需另立】

对每个接口点 \(q_j\)，令 \(\mathcal A(q_j)\) 表示存在**同一条**free候选边，累计满足：

距离小于 \(\tau L\)、无向切向夹角不大于 \(\theta\)、有限投影支撑成立、所属面满足指定的CF资格。

\[
S(\tau,\theta)
=
\frac{
\sum_jw_j\mathbf1[\mathcal A(q_j)]
}{
\sum_jw_j
}.
\]

旧主设置为 \(\tau=0.01,\theta=20^\circ\)。旧S明确包含上述四项，不等于仅距离AttachCov。

**越高越好，但只适用于明确要求接缝的任务。**

需要区分：

\[
S_{\mathrm{source}}:\Gamma=\text{来源known/free接口},
\]

\[
S_{\mathrm{required}}:\Gamma=\text{输入或任务明确指定的接口}.
\]

后者是新版本，不能直接继承旧结果。旧实现要求同一候选边同时满足全部条件，并排除退化面及 owner C-face 对应的 proper、共面非法和未决 CF；未知支持另报上界，不是仅查最近边的距离。**没有接合要求时记NA；不能默认C所有开放边都必须封闭。**

## 05｜实际C—free共享边数量／覆盖【已有后处理诊断，可统一】

基于最终索引拓扑，找出同时邻接known面和free面的边：

\[
N_{\mathrm{shared}}
=
|\{e:\ e\text{同时属于C面和free面}\}|.
\]

若有指定接口，还可计算：

\[
R_{\mathrm{shared}}
=
\frac{
\sum_{e\in\Gamma}
\ell(e)\,
\mathbf1[e\text{形成符合合同的实际共享连接}]
}{
\sum_{e\in\Gamma}\ell(e)
}.
\]

**它与S不同：S是容差几何关系；这里检查实际连接。**

必须分别记录原始索引、仅精确同坐标合并后的索引、真正后处理后的索引。不能把阈值焊接后的连接冒充RAW已有连接。

**只在要求共边连续表面时越高越好；构件装配任务不必强求。** 已有 C_open_edges_exact_attached_fraction 按所有 C 开放边的边数计比例，并非上式指定 Gamma 的长度覆盖；后者是 PROPOSED。共享边存在不等于正确连接，还需区分额外邻接面、方向与非流形情况。如果允许细分接口，覆盖应按指定边上已连接子段的弧长并集计算，不能重复累计；若要求原 C 面不变，切分 C 属于违反输入保持合同，不能静默用于增加共享率。

## 06｜I：C冲突覆盖率【现有】

在C上固定面积采样点 \(q_j\)，定义：

\[
a_j=\mathbf1[
q_j\text{落入所属C面proper交线的固定影响带}
],
\]

\[
b_j=\mathbf1[
q_j\text{位于登记的C内部区域，且附近存在近平行free面}
].
\]

则：

\[
I_C
=
\frac{\sum_jw_j\mathbf1[a_j\lor b_j]}{\sum_jw_j}.
\]

同时保存：

\[
I_{\mathrm{cross}},\qquad
I_{\mathrm{dup}},\qquad
I_{\mathrm{cross}\cap\mathrm{dup}},
\]

并检查：

\[
I_C=I_{\mathrm{cross}}+I_{\mathrm{dup}}
-I_{\mathrm{cross}\cap\mathrm{dup}}.
\]

旧实现使用穿越影响带与近重复覆盖的并集，不是“发生冲突的三角面数比例”。

**通常越低越好，原始几何量由C、F与预先冻结的尺度/阈值决定。** 旧阈值 rho_cross=rho_dup=0.005L，C内部要求距C边界至少0.01L，无向法向夹角不大于15度，分母为全C采样权重。当前旧入口 prepare_reference 仍会读取 GT_free 并使用来源完整模型的 L0；数学上可做无GT版本，不等于现有入口已经支持。 但近重复项可能标记合法薄壁，不能自动当成语义错误。

若报告：

\[
\Delta I_{\mathrm{ref}}=I(C,F)-I(C,Y),
\]

这个相对版本仍依赖来源GT；原始I与相对I必须分列。

## 07｜E：外露保持率【现有；无GT版本需另立】

旧版本先由来源完整模型确定每个C点的参考可见方向集合 \(V_Y(q_j)\)：

\[
E_{\mathrm{ref}}
=
\frac{
\sum_jw_j
\mathbf1[
V_Y(q_j)\neq\varnothing
\land
\exists d\in V_Y(q_j):q_j\text{在输出中可见}
]
}{
\sum_jw_j\mathbf1[V_Y(q_j)\neq\varnothing]
}.
\]

这与旧报告的参考外露定义一致；重合、极瘦面和射线未决需保留上下界。主 lower/upper 的分母只含确定参考外露点；ExposedRetain_global_lower/upper 还考虑参考资格未知。两套区间必须用独立字段与 domain 表示，不可混成同一对上下界。

**只有“外露确属要求”时越高越好。**

无GT任务可另定义 \(E_{\mathrm{required}}\)：由用户指定必须外露的C区域和观察方向，而不是由来源模型生成这些要求。**未指定外露要求时，不应最大化E。**

---

# 四、网格自身的几何与拓扑质量

## 08｜CF／FF非法相交对数与受影响面比例【现有分项；统一并集需核对】

设：

\[
\mathcal J_{CF}
=
\{(c,f):c\in C,f\in F,\text{满足登记的非法交定义}\},
\]

\[
\mathcal J_{FF}
=
\{(f_i,f_j):i<j,\text{满足登记的非法交定义}\}.
\]

报告：

\[
N_{CF}=|\mathcal J_{CF}|,\qquad
N_{FF}=|\mathcal J_{FF}|.
\]

free侧受影响比例为：

\[
R_{CF}^{F}
=
\frac{
|\{f\in F:\exists c,(c,f)\in\mathcal J_{CF}\}|
}{|F|},
\]

\[
R_{FF}
=
\frac{
|\{f\in F:f\text{参与至少一个FF非法对}\}|
}{|F|}.
\]

还可以补C侧受影响面比例，分母换为 \(|C|\)。

**通常越低越好，几何判断不需要来源完整GT，但须提供冻结尺度。** 对数、受影响面比例、I的面积覆盖是三个不同量，不能混用。旧 CF_proper_affected_free_fraction 只计 proper 穿越；共面非法项另列，不能直接改名为上式 CF 非法并集。旧 FF_illegal_affected_free_fraction 为 proper 或共面正面积非法重叠的受影响 free 面并集。合法接触、未决与退化分别保存；未决不能当无交。

## 09｜全网格自交比例 \(R_i\)【文献；需统一实现】

令 \(\mathcal J_M\) 为完整网格M的非法相交面对集合：

\[
R_i(M)
=
\frac{
|\{f\in M:f\text{参与至少一个非法相交对}\}|
}{|M|}.
\]

这与MeshFlow所讨论的面相交比例属于同一类指标；但要与论文数字直接比较，还必须匹配其相交分类、数据和后处理阶段。[arXiv](https://arxiv.org/html/2606.23489v2)

**需要计入CC、CF和FF，不能把两个free侧百分比简单相加。** C原本已有的缺陷应另外披露。

合法共边／共点应与非法穿越区分；几何相同而索引不同的接触，也会受检测器定义影响。CGAL的定义明确涉及交集是否为双方共同的边或顶点。[CGAL Manual](https://doc.cgal.org/latest/Polygon_mesh_processing/group__PMP__intersection__grp.html)

## 10｜零面积、近退化与重复面比例【已有部分free诊断；全域统一待接入】

对三角面 \(f=(a,b,c)\)：

\[
A_f=\frac12\|(b-a)\times(c-a)\|_2.
\]

\[
R_{\mathrm{zero}}
=
\frac{\#\{f:A_f=0\}}{|M|},
\]

\[
R_{\mathrm{near}}
=
\frac{\#\{f:A_f/L^2<\epsilon_A\}}{|M|}.
\]

重复面比例可定义为：

\[
R_{\mathrm{dupface}}
=
\frac{|M|-|\operatorname{unique\_faces}(M)|}{|M|}.
\]

**越低通常越好。** \(\epsilon_A\)、是否考虑绕序、按索引还是按坐标判重，都必须固定；反向重合面可另列，不能静默合并。

C、free和全网格分别保存，避免已有输入缺陷被归罪于生成。

## 11｜三角形角度质量【已有free诊断；全域字段需核对】

设 \(\theta_{\min}(f)\) 为一张面的最小内角：

\[
R_{<5^\circ}
=
\frac{
\#\{f:\theta_{\min}(f)<5^\circ\}
}{|M|}.
\]

同时保存全局最小角、逐面最小角的p05、中位数等。

**它是离散化质量诊断，不是完整生成质量。** 少量细长面可能是几何结构所需；“更接近来源小角比例”也不代表更好。旧报告已经将小角、退化、相交分开解释。

## 12｜连通分量、最大分量面积占比【已有分量数；最大分量面积比待接入】

构建面邻接图：两面共享登记意义下的一条边时连接。

\[
N_{\mathrm{comp}}=\text{图的连通分量数},
\]

\[
R_{\mathrm{largest}}
=
\frac{\max_k A(M_k)}{A(M)}.
\]

可另记录小分量面积占比，但阈值必须预先定义。

**不是所有任务都越少分量越好。** 合理椅子可以由多个部件构成；关键是不要把碎裂与合理装配混为一谈。MeshPad也报告分量数，而没有把单连通视为所有结果的必要条件。[arXiv](https://arxiv.org/html/2503.01425v4)

## 13｜边界边、非流形边与非流形顶点【已有部分边诊断；完整顶点检查未确认】

令 \(\deg(e)\) 为边的邻接面数：

\[
N_{\mathrm{boundary}}=\#\{e:\deg(e)=1\},
\]

\[
R_{\mathrm{NMedge}}
=
\frac{\#\{e:\deg(e)>2\}}{|\mathcal E(M)|}.
\]

边界总长度：

\[
L_{\mathrm{boundary}}^{\mathrm{norm}}
=
\frac{\sum_{\deg(e)=1}\ell(e)}{L}.
\]

非流形顶点需另外检查局部link：闭表面顶点应对应一个环，边界顶点应对应一条链；仅查边数不能排除所有非流形情况。

**开放设计的边界边不自动是缺陷。** 上式 deg(e)>2 是这里明确提出的非流形边版本，deg 按不同邻接面计；部分库在不允许边界时还将 deg(e)=1 计入非流形边，MeshPad 表中具体实现亦需核对，不能仅凭同名对比。顶点 link 的环/链判据适用于非退化、结构合法的二维三角复形；重复顶点/面、零长度边、孤立实体及索引异常须先分列。 这些是拓扑描述，不是通用越低越好的分数。

## 14｜闭边、定向一致与水密／有效表面检查【按任务启用】

首先单独记录：

\[
B_{\mathrm{closed\ edge}}
=
\mathbf1[\forall e,\deg(e)=2].
\]

再分别检查顶点流形性、定向一致、退化、自交与未决事件。

**“每边两面”“无自交”“闭合可定向表面”不能互相代替。** 检查未完成就记NOT_RUN／UNKNOWN，不用一个库的`is_watertight`字段代替全部几何认证。

仅在已经确认是闭合、可定向二维流形时，才可用：

\[
\chi=|V|-|E|+|F|,\qquad
g=\frac{2-\chi}{2}
\]

计算每个连通分量的亏格（connected component；不是要求该分量 simply connected）。**亏格不是越小越好。**

---

# 五、来源参考重建指标

这一组继续保存，但对于开放式补全，**它们评价“接近这一种参考完成的程度”，不自动评价“是否生成了合理完成”。**

## 15｜单向表面MSE、双向MSE与RMS【现有】

\[
D_{Y\rightarrow F}^{(2)}
=
\frac1{|Q_Y|}
\sum_{q\in Q_Y}\frac{d(q,F)^2}{L^2},
\]

\[
D_{F\rightarrow Y}^{(2)}
=
\frac1{|Q_F|}
\sum_{p\in Q_F}\frac{d(p,Y)^2}{L^2},
\]

\[
D_{\mathrm{sym}}
=
\frac{
D_{Y\rightarrow F}^{(2)}+
D_{F\rightarrow Y}^{(2)}
}{2}.
\]

同一采样定义下，单输出RMS为对应MSE的平方根。

**旧512点、2048点、旧RMS采样分别保留。** 不能将“平均MSE开根号”当作“逐输出RMS的平均”，也不能拿新Dsym开方替换旧的双向RMS。旧报告已明确这些采样和聚合差别。 

## 16｜距离分位数、最大距离与Hausdorff【现有／文献】

对归一距离集合：

\[
\mathcal D_{Y\rightarrow F}
=
\{d(q,F)/L:q\in Q_Y\},
\]

报告p50、p95、p99、最大值。

连续表面的双向Hausdorff为：

\[
H(A,B)=
\max\left\{
\sup_{a\in A}d(a,B),
\sup_{b\in B}d(b,A)
\right\}.
\]

若只是有限采样点上的最大值，名称应是**采样Hausdorff近似**，不是严格最大误差界。

**参考重建任务中越低越好；开放式补全不据此单独判错。**

## 17｜表面Precision／Recall／F-score【现有】

\[
P(\tau)
=
\frac1{|Q_F|}
\sum_{p\in Q_F}
\mathbf1[d(p,Y)<\tau L],
\]

\[
R(\tau)
=
\frac1{|Q_Y|}
\sum_{q\in Q_Y}
\mathbf1[d(q,F)<\tau L],
\]

\[
F_1(\tau)=\frac{2P(\tau)R(\tau)}{P(\tau)+R(\tau)}.
\]

分母为0时规定 \(F_1=0\)。

旧“GT表面coverage”主要对应这里的Recall，不是后文集合COV。**平方距离数组应与 \((\tau L)^2\) 比较，不能与线性阈值直接比较。**

所有值均针对free与参考free，不能混入C到自身的零距离来稀释误差；旧评价也明确采用free-only口径。

## 18｜点云Chamfer Distance与EMD【文献】

对于非空点集 \(P,Q\)，建议统一版本采用“两个方向均值相加”的平方CD（尚未接入；文献对照另冻结原版）：

\[
\operatorname{CD}_2(P,Q)
=
\frac1{|P|}\sum_{p\in P}\min_{q\in Q}\frac{\|p-q\|^2}{L^2}
+
\frac1{|Q|}\sum_{q\in Q}\min_{p\in P}\frac{\|q-p\|^2}{L^2}.
\]

等点数 \(n\) 的平均EMD可写为：

\[
\operatorname{EMD}_1(P,Q)
=
\frac1n\min_{\pi}
\sum_{i=1}^{n}\frac{\|p_i-q_{\pi(i)}\|_2}{L}.
\]

CD使用最近邻；EMD使用一一匹配。原文／代码可能采用求和而非均值、是否除2也可能不同，必须固定版本。[Proceedings of Machine Learning Research](https://proceedings.mlr.press/v80/achlioptas18a/achlioptas18a.pdf)

**点云CD不等于点到有限三角表面的Dsym。** 这里的 CD/EMD 归一化公式是候选版本，不声称所有引用论文使用完全相同的倍率。 EMD近似求解时需报告近似方法。

## 19｜UHD：输入到输出的单向Hausdorff【文献】

\[
\operatorname{UHD}(C,M)
=
\sup_{c\in C}\frac{d(c,M)}{L}.
\]

它在多模态形状补全文献中用于输入忠实度。[arXiv](https://arxiv.org/html/2003.07717v3)

但我们若已经满足 \(M=C\sqcup F\)，则连续几何下UHD天然为0。**因此它只适合跨方法输入保持比较，不适合作为我们自身control强弱的主要分数。** 点云采样实现出现的非零还可能只是采样误差。

## 20｜GT-oracle Best@\(m\)【已有探索协议】

固定同一C的 \(m\) 个候选：

\[
\operatorname{Best}@m(C)
=
\min_{1\le j\le m}D(F_j,Y).
\]

同时报告候选平均误差。

必须注明这是**利用参考Y的oracle选择**，不是部署时自动能选出的结果。增加候选数使最小值不升，是集合包含带来的性质，不是独立方法贡献。

## 21｜多片上下文的公共未知区域误差【现有】

对于 \(C_1\subset C_2\subset C_3\)，定义：

\[
Y_{\mathrm{common}}=M_{\mathrm{source}}\setminus C_3.
\]

从该区域固定采样：

\[
D_{\mathrm{common},l}
=
\frac1{|Q_{\mathrm{common}}|}
\sum_q\frac{d(q,F_l)^2}{L^2}.
\]

只将free作为距离目标，避免多复制C取得免费收益。原多片实验已经采用这一原则。

**它仍是来源重建诊断，不要求开放式生成随C增加逐候选单调改善。**

---

# 六、完整生成集合的质量与覆盖

以下 \(\mathcal G\) 与 \(\mathcal R\) 比较的是**形状集合**，不是给每个输出指定唯一来源模型。距离 \(d_{\mathrm{shape}}\) 可以采用固定点云CD或其他已登记的形状距离。

## 22｜Minimum Matching Distance：MMD【文献】

\[
\operatorname{MMD}_{\mathrm{NN}}(\mathcal G,\mathcal R)
=
\frac1{|\mathcal R|}
\sum_{r\in\mathcal R}
\min_{g\in\mathcal G}d_{\mathrm{shape}}(r,g).
\]

**越低越好。** 它衡量参考形状能否在生成集合中找到近似结果；许多额外坏样本可能不影响这些最小值，因此不能独自代表全部生成质量。[arXiv](https://arxiv.org/html/2405.20853v2)

这里MMD是**Minimum Matching Distance**，不是后文KID中的Maximum Mean Discrepancy。

## 23｜集合Coverage：COV【文献】

对每个生成形状找最近参考：

\[
r^\star(g)=\arg\min_{r\in\mathcal R}d_{\mathrm{shape}}(g,r),
\]

\[
\operatorname{COV}
=
\frac{
|\{r^\star(g):g\in\mathcal G\}|
}{|\mathcal R|}.
\]

**越高表示匹配覆盖更多参考形状。** 它不是表面面积覆盖，也不保证匹配距离小。生成／参考集合大小与平局规则需固定。[Proceedings of Machine Learning Research](https://proceedings.mlr.press/v80/achlioptas18a/achlioptas18a.pdf)

## 24｜1-Nearest Neighbor Accuracy：1-NNA【文献】

将生成／真实形状标记为两类。对每个样本，在**排除自身**后找最近邻：

\[
\operatorname{1NNA}
=
\frac1{|\mathcal G|+|\mathcal R|}
\sum_x
\mathbf1[\ell(x)=\ell(\operatorname{NN}_{-x}(x))].
\]

两类数量平衡时，通常以**接近50%**作为难以区分的方向，不能简单写成越低越好。MeshXL采用这一评价。[arXiv](https://arxiv.org/html/2405.20853v2)

它仍然不验证C—输出对应关系：把合理椅子错误分配给不同C，集合分布可能不变。

## 25｜Jensen–Shannon Divergence：JSD【文献】

在统一坐标、固定三维网格上统计两组表面采样点的空间直方图，归一得到 \(P,Q\)，令 \(H=(P+Q)/2\)：

\[
\operatorname{JSD}(P,Q)
=
\frac12\operatorname{KL}(P\|H)
+
\frac12\operatorname{KL}(Q\|H).
\]

**越低越好。** 固定每形状点数、网格分辨率、坐标范围和对数底数；越界点如何计数也需冻结，不能静默丢掉。论文正文是跨点云的点计数直方图；若采用每形状二值占据计数，应明确另立版本。这里是表面点位置分布，不是给开放网格擅自填充得到的实体体积。[Proceedings of Machine Learning Research](https://proceedings.mlr.press/v80/achlioptas18a/achlioptas18a.pdf)

## 26｜Fréchet Inception Distance：FID【文献】

对统一渲染图提取同一Inception特征，得到两组均值和协方差：

\[
\operatorname{FID}
=
\|\mu_G-\mu_R\|_2^2+
\operatorname{Tr}
\left(
\Sigma_G+\Sigma_R
-2(\Sigma_R^{1/2}\Sigma_G\Sigma_R^{1/2})^{1/2}
\right).
\]

**越低越好。** 换成别的特征网络，应明确命名为对应特征距离，不能假装仍可与原FID数值比较。[arXiv](https://arxiv.org/abs/1706.08500)

渲染评估时应统一中性材质、相机与预处理；不能生成结果蓝橙着色、参考全灰。有限样本FID存在模型相关偏差，32份输出不能靠多渲染几个视角变成可靠的大规模形状评价。[arXiv](https://arxiv.org/abs/1911.07023)

## 27｜Kernel Inception Distance：KID【文献】

对Inception特征 \(x_i,y_j\)，采用：

\[
k(x,y)=\left(\frac{x^\top y}{d}+1\right)^3.
\]

无偏估计为：

\[
\widehat{\operatorname{KID}}
=
\frac{\sum_{i\ne j}k(x_i,x_j)}{n(n-1)}
+
\frac{\sum_{i\ne j}k(y_i,y_j)}{r(r-1)}
-
\frac{2\sum_{i,j}k(x_i,y_j)}{nr}.
\]

**总体差异越低越好。** 独立样本假设下该 U-statistic 是无偏估计，有限样本值可能为负，不截到0；子集估计应报告重复与不确定性。同一 mesh 的多个视角、同一父的多个条件并非独立样本，不能直接沿用独立图像的无偏性或误差条结论；按独立父/条件簇设计抽样与不确定性分析，固定并说明目标分布。两组特征数必须各至少2。其核MMD与第22项的最近邻MMD不是同一概念。[arXiv](https://arxiv.org/html/1801.01401v5)

**集合指标共同限制：** 真实参考集合应与任务分布合理匹配；不能用大量与手工C根本不兼容的Chair分布，惩罚条件约束本来就导致的合法分布变化。

---

# 七、同条件多样性与编辑响应

## 28｜条件内free两两距离、TMD【现有适配＋文献】

同一C生成 \(m\) 个free，建议沿用：

\[
D_{\mathrm{free}}(C)
=
\frac{2}{m(m-1)}
\sum_{i<j}d(F_i,F_j).
\]

其中 \(d\) 的采样、距离类型、尺度固定。**不对每个候选独立ICP或重缩放。**

文献原TMD的形式是：

\[
\operatorname{TMD}(C)
=
\sum_{i=1}^m\frac1{m-1}\sum_{j\ne i}d(M_i,M_j)
=
\frac2{m-1}\sum_{i<j}d(M_i,M_j).
\]

因此，在相同对象与距离定义下，原TMD是普通两两平均的 \(m\) 倍；我们改用free而非完整M，又是另一处区别。不能直接混名比较。[arXiv](https://arxiv.org/html/2003.07717v3)

**衡量变化程度，不是无条件越大越好。** 碎片、远飞面同样会增大它。m 必须至少2；若候选失败，保留固定候选账本及缺失对，不仅对幸存者重算后冒称全候选多样性。只在合格候选上算的版本需另命名、预定义资格并同时报合格率；不能事后删坏样本。

## 29｜宽度响应斜率、基准宽度比例【现有】

旧宽度定义：

\[
W(F)=q_{95}(x_{\mathrm{free\ corners}})
-q_{05}(x_{\mathrm{free\ corners}}).
\]

\[
b_{\mathrm{self}}
=
\frac{W(F_{1.1})-W(F_{0.9})}
{0.2W(F_1)},
\]

\[
b_{\mathrm{GT}}
=
\frac{W(F_{1.1})-W(F_{0.9})}
{0.2W(Y_1)},
\qquad
r_{\mathrm{base}}=\frac{W(F_1)}{W(Y_1)}.
\]

W(F_1) 或 W(Y_1) 为0时对应比值记 NA 并注明原因，不改分母。报告 \(|b-1|\)、\(|r_{\mathrm{base}}-1|\)，先逐曲线取绝对值，再聚合，避免正负误差抵消。

**只有明确要求该宽度同比变化时，斜率1才是正确目标。** 角点分位宽度受三角划分影响；改成面积采样宽度应另立版本。

## 30｜3×3条件矩阵、正确列优势与严格胜率【现有】

保留原矩阵：

\[
\mathcal E_{\alpha,\beta}=D(F_\alpha,Y_\beta).
\]

固定目标列 \(\beta\)：

\[
A_\beta
=
\frac12\sum_{\alpha\ne\beta}\mathcal E_{\alpha,\beta}
-\mathcal E_{\beta,\beta}.
\]

严格列胜出：

\[
w_\beta=
\mathbf1[
\mathcal E_{\beta,\beta}
<
\min_{\alpha\ne\beta}\mathcal E_{\alpha,\beta}
].
\]

另记录三列全胜、同一输入通吃三列等。平局不算严格胜。必须保留完整矩阵及行列身份，并用重复相同行的 No-response 正控验证列优势为0、严格胜率为0；固定输出行比较不同目标只称行偏好，不能称正确条件列优势。

**这是有指定变化目标的受控诊断，不是开放式补全的通用准确率。**

---

# 八、用途相容性与人类评价

## 31｜条件匹配评分、完整形状评分、偏好率【文献思路；本任务需制定量表】

分别给每份输出收集：

\[
s_{\mathrm{cond}},\qquad s_{\mathrm{quality}},
\]

例如使用固定1—5等级，先对同一输出的评阅者评分求均值，再按条件聚合。

若采用成对偏好，可定义：

\[
P_{\mathrm{prefer}}
=
\frac{\#\mathrm{win}+0.5\,\#\mathrm{tie}}
{\#\mathrm{all\ rated\ comparisons}}.
\]

平局计法必须事先声明。

MeshPad将生成／编辑质量、条件匹配和未编辑区一致性分开打分；迁移到我们时，应把草图匹配改成**当前C及明确用途的相容性**，不能继承其数值。[arXiv](https://arxiv.org/html/2503.01425v4)

**主观评分不等于不规范；但AI技术看图必须标AI_PROXY，不能冒充人类研究。** 开放式任务评阅者不应被来源GT诱导成“选最像来源的”。

## 32｜可用候选比例 \(U_{\mathrm{single}}\)、\(U_{@m}\)【已提出协议，需实际标签】

令 \(u_{c,j}\in\{0,1\}\) 表示候选是否满足事先定义的用途标准：

\[
U_{\mathrm{single}}
=
\frac1{|\mathcal C|}
\sum_c\frac1m\sum_j u_{c,j},
\]

\[
U_{@m}
=
\frac1{|\mathcal C|}
\sum_c
\mathbf1\left[\sum_j u_{c,j}\ge1\right].
\]

**前者衡量单候选成功机会；后者衡量固定预算内是否至少有一个可用候选。**

可用标准不能是“比来源GT近”。它应明确包含C保持、整体相容和当前用途所需的几何质量。没有真实标签就记NOT_RUN，不由S、FID或AI观感自动代替。

\(U_{@m}\ge U_{\mathrm{single}}\) 本身是定义性质；研究价值在绝对水平、增益幅度与同预算方法差异。

## 33｜CLIP相似度、LPIPS【文献；当前C-only任务不启用】

本库当前仅在协议实际提供可比较的草图／图像条件或成对参考时启用；并非断言 CLIP/LPIPS 在其他研究任务中没有用途。当前 C-only 任务不启用。

CLIP相似度：

\[
S_{\mathrm{CLIP}}(I,J)
=
\frac{\phi(I)^\top\phi(J)}
{\|\phi(I)\|\,\|\phi(J)\|}.
\]

LPIPS在对应图像的深层特征上计算学习加权差异：

\[
D_{\mathrm{LPIPS}}(I,J)
=
\sum_l\frac1{H_lW_l}
\sum_{h,w}
\|a_l\odot(\hat\phi_l(I)-\hat\phi_l(J))_{hw}\|_2^2.
\]

前者通常越高越好，后者越低越好。特征模型、视角与图像预处理必须固定。LPIPS衡量的是学习到的感知距离。[arXiv](https://arxiv.org/abs/1801.03924)

**不能把局部C渲染与完整椅子渲染直接算CLIP／LPIPS，就宣称得到了通用C相容性分数。**

---

# 九、成本与后处理代价

## 34｜延迟、吞吐、NFE与显存【现有】

\[
T_{\mathrm{deploy}}
=
T_{\mathrm{generation}}+T_{\mathrm{postprocess}},
\]

仅在串行、同范围计时条件下相加；有流水并行则直接测端到端。

\[
\mathrm{Throughput}
=
\frac{\text{完成输出数}}{\text{实际墙钟时间}}.
\]

同时报告网络函数评价次数NFE、batch大小、峰值显存、采样步骤和输出面数。

**缓存读取不算生成速度；50步不一定等于50次网络评价；设备总显存不等于进程峰值。**

## 35｜后处理位移、删面与连接变化【已有后处理报告，可统一】

有原始角对应时：

\[
D_{\mathrm{move}}^{\max}
=
\max_i\frac{\|x_i^{\mathrm{post}}-x_i^{\mathrm{raw}}\|}{L},
\]

\[
D_{\mathrm{move}}^{\mathrm{RMS}}
=
\sqrt{\frac1n\sum_i
\frac{\|x_i^{\mathrm{post}}-x_i^{\mathrm{raw}}\|^2}{L^2}}.
\]

记录修改角／面比例、删面数和原因、C吸附数量、RAW／POST面数，以及相交事件的新增／移除集合。

**这些是收益的代价说明，不是越小就必然越好。** 若后处理改变了拓扑、无法维持一一对应，应另用表面距离及来源映射，不伪造角对应。

---

# 十、哪些是“每轮都用”，哪些只按任务启用？

| 任务类型 | 应启用的主要指标族 | 不应强行要求 |
|---|---|---|
| **任意C的开放式Chair生成** | C保持、有效输出、CF/FF及网格质量、条件相容性、完整形状质量、条件内多样性、成本 | 来源GT距离必须低、所有边必须接上、C必须完全外露 |
| **指定接口的表面补全** | 上述项目＋S／接口距离／实际共享连接 | 必须长成来源椅型 |
| **必须外露的表面或扫描观测保持** | 上述项目＋明确版本的E | 从来源模型自动继承所有可见方向 |
| **来源模型重建回归** | 双向距离、RMS、p95、F-score、参考接口等 | 将结论外推成全部开放式生成能力 |
| **宽度或其他参数编辑** | 明确的响应曲线、矩阵及生成质量 | 任意局部编辑都应整体等比例变化 |
| **完整后处理系统** | RAW／POST配对、C保持、几何质量、位移／删面／成本 | 只展示处理后结果却标RAW |

**这才是“以后统一用这些指标”的合理含义：统一字典、统一实现、按任务启用，而不是每次强行计算所有量。**

---

# 十一、所有指标共同遵守的汇总方式

对逐输出指标 \(h\)，保持：

\[
\bar h
=
\frac1{|\mathcal C|}
\sum_c
\frac1m\sum_{j=1}^m h_{c,j}.
\]

若同一父对象有多个C，应先明确“父等权”还是“条件等权”，不能临时选择有利的加权方式。现有报告已经采用先噪声平均、再父等权，并明确两份噪声不是两份独立训练。

建议每项结果固定保存：

`metric_id、version、output_id、parent_id、condition_id、seed/candidate_id、output_stage、domain、reference_type、scale及其来源、thresholds、sampling及种子/hash、value、lower、upper、status、reason、denominator、source_hash、implementation_version`。

同时遵守四条规则：

**第一，失败、NA和未决分开。** 任务不定义GT是 NA_NOT_APPLICABLE；任务需要但资产缺失是 BLOCKED_MISSING_ASSET；未实现/未计算为 NOT_IMPLEMENTED/NOT_RUN；已尝试空输出为 FAILED；检测器未决为 UNKNOWN/PARTIAL。不能都丢掉后只算幸存结果。旧失败数值哨兵保持旧合同；新集合指标不能对无特征的失败样本伪造特征，应披露有效集口径及失败率。

**第二，平均差、中位差、改善父数与尾部一起保留。** 不要求主文全放，但内部字典不能只留一个均值。

**第三，多视角不是多个独立形状。** 集合指标与置信区间需要尊重同一C、同一父和同一输出之间的相关性。

**第四，版本变化不能回写历史。** 在正式评价前冻结主/辅助指标、阈值、样本预算、集合、平局规则、失败规则和聚合方式；已多次观察的历史面板标探索，不将其追认为未见测试集。 换尺度、接口、可见域、三角检测器、渲染特征或采样方式，都需要新版本。

---

# 十二、论文最后选三个，怎样既突出优势，又真正回答任务？

**主文只放三个核心指标完全可以，但三个应覆盖不同问题，而不是从同一方向挑三个高度重复、又恰好都好看的数字。**

针对当前“给定真实面片，生成多种合理完成”的目标，以下为候选分工，尚未冻结三个具体指标：

| 主文位置 | 要证明什么 | 可选指标 |
|---|---|---|
| **条件相容性** | C不只是被复制，确实被合理使用 | 明确定义的条件匹配评分／\(U_{\mathrm{single}}\)；有接缝要求时用 \(S_{\mathrm{required}}\) |
| **完整生成质量** | 输出本身是合理Chair，而非局部刷分 | 固定的KID或FID，或规范的整体质量盲评 |
| **条件内多样性** | 同一C下有不同合理选择 | \(D_{\mathrm{free}}\)／明确定义的TMD |

C精确保留另作硬合同报告，不必占一个主要性能指标。I、CF／FF、退化和拓扑统计放质量附表；**若存在足以影响结论的严重代价，主文必须说明，不能因附表不占三列就当作不存在。**

如果最终论文明确收窄成“指定边界的表面接续”，则可以预先选择 **S、全网格自交比例、整体质量指标**，多样性放补充实验。**选择依据应该是论文实际承诺的能力，而不是哪三项涨得最多。**

尤其不要直接采用“FID、MMD、COV都好，所以控制成功”：这三个主要描述集合，**即使将生成椅子错误分配给不同C，它们也可能几乎不变。**

**持久规则：完整保存已授权评价范围内的指标结果和不适用/缺失原因，按任务冻结适用项；开发阶段确定支持贡献的主指标，正式评价前冻结。当前没有新结果证明某三个指标已经最优，也没有自动授权执行整套指标库。**

## 来源与本轮变更范围

来源：用户附件 f6cb75c9-f825-4f68-85c1-91cf62c4b29b/Pasted text.txt；原文件 SHA256：442a2942aba8fed4ff0d3e2e95bc53c7ba28cc3509048871f9f35e0a74007b6e。

本轮只新增本文件并在 AGENTS.md、WORKSPACE.md 登记入口；未运行训练、模型前向、生成或评分，未修改旧数据、权重、raw或报告，未commit/push。
