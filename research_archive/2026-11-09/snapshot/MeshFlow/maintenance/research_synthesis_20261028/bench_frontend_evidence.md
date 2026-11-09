# Bench 前端训练证据梳理（2026-10-28）

当前最值得保留为 **Bench 生成器研究候选** 的是四方向实验的 **EXPOSURE：M/global22500 恢复完整 AdamW/RNG，再训练3000步至 global25500/native13500**。它在同一46请求面板上，相对M增加了RAW合理采用，并降低总体FF/I；但固定后端的最终主体和支撑没有同步改善，因此不能称为已经证实的完整流程SOTA。最新PLAIN200/FLIP200没有取代EXPOSURE。窄池NATURAL_HIGH B仍是有独立证据价值的参照，旧Native6000/SEAM3000的维护身份也没有被自动改写。

这是一份既有本地证据综述，不是新的模型评测。只读正式REPORT、训练登记、既有汇总及保留模型索引；大权重只检查文件存在和长度，没有加载、重新hash、模型前向、训练或重评分。未修改历史文件。

## 谱系与实际配方

“B”在两轮报告里含义不同，下文写全名：曝光B=`B_75_25`；自然N窄池B=`NATURAL_HIGH`。W与M都从曝光B/global21000分叉，**没有从自然N B终点依次串行训练W→M**。

```text
COMMON12000
└─ 完整座面 Native +6000 → global18000
   ├─ 旧 SEAM +3000 → global21000（旧研究参考分支）
   └─ Coverage +1500 → global19500
      ├─ 曝光 A_50_50 +1500 → global21000
      └─ 曝光 B_75_25 +1500 → global21000（后续共同P0）
         ├─ 自然N CONTROL_OLD +1500 → global22500
         ├─ 自然N NATURAL_HIGH B +1500 → global22500
         │  └─ 四方向 原版SEAM +1500 → global24000
         ├─ WIDE_HIGH W +1500 → global22500
         └─ W_MATCHED M +1500 → global22500
            └─ EXPOSURE +3000 → global25500
               ├─ PLAIN200 +200 → global25700
               └─ FLIP200 +200 → global25700
```

| 阶段 | 起点→终点；新增更新 | 训练父/完整C条件 | 本段实际实例曝光与改变 |
|---|---|---|---|
| 完整座面Native | COMMON12000→18000；6000 | 22父22C，N128..256 | 每步4个不同父，每父CLEAN与X_WIDTH各1例；共48000实例 |
| Coverage | Native18000→19500；1500 | OLD22+a8/27两父，共24父24C | OLD6000、NEW6000；a8/27各3000，OLD每父272–274；同时扩池和续训，没有等预算OLD-only臂 |
| 曝光A_50_50 | Coverage19500→21000；1500 | 同24父24C | OLD/FOCUS=6000/6000；a8/27各3000 |
| 曝光B_75_25 | 同Coverage19500→21000；1500 | 同24父24C | OLD/FOCUS=9000/3000；a8/27各1500，OLD每父408–410；固定池曝光分配对照 |
| 自然N CONTROL_OLD | 曝光B21000→22500；1500 | 同24父24C | OLD/FOCUS=9000/3000 |
| 自然N NATURAL_HIGH B | 同曝光B21000→22500；1500 | OLD22+FOCUS2+HIGH2，共26父26C | OLD/FOCUS/HIGH=9000/1500/1500；HIGH两父各750；自然版本100%，未做多LOD混合 |
| WIDE_HIGH W | 同曝光B21000→22500；1500 | OLD22+FOCUS2+HIGH6，共30父30C | 同9000/1500/1500；HIGH六父各250；扩大父池且提高自然N范围 |
| W_MATCHED M | 同曝光B21000→22500；1500 | 同W的30父30C | 完整重放W的12000实例；只将HIGH组MSE系数匹配窄B，非从W继续训练 |
| EXPOSURE | M22500→25500；3000 | 同30父30C | OLD/FOCUS/HIGH=18000/3000/3000；HIGH六父各新增500，连同M各累计750；原父/增强序列重复两遍，噪声、时间、epsilon及排列为新登记身份 |
| 四方向SEAM1500 | 窄B22500→24000；1500 | 窄26父26C，训练最大N384 | OLD/FOCUS/HIGH=9000/1500/1500；切原source-owner SEAM，冻结历史C-only头；未配等预算Native续训反事实 |
| PLAIN200/FLIP200 | 各从EXPOSURE25500→25700；各200 | 同30父30C | 各OLD/FOCUS/HIGH=1200/200/200；FLIP28条可翻、2条原样，实际翻728实例、52计划翻转回退；两臂复用EXPOSURE旧计划前200行随机流作为新续训分支 |

共同Native配方为原Native/T1+role+Geo、**free-only FM，先free OT再HYBRID0.25**；batch8、microbatch1，每步4父，每父identity与x-width U[0.9,1.1]各一例；AdamW lr1e-5、betas(0.9,0.95)、weight_decay0、clip1，继承live参数及匹配优化器/RNG。BF16前向、FP32参数/积分、math SDPA、TF32关闭、严格确定性；不增加头调用或新辅助loss。1500步臂各12000训练前向/反向，EXPOSURE为24000，200步臂各1600。SEAM另采用原owner构造、受限bulk OT、锚点mask和第三点FM，推理用冻结833参数旧C-only头，无oracle来源布局。

自然N先新增28b9 N384/K10、abf N308/K44；W再增加945 N448/K14、af N636/K84、bcb N520/K2、77aa N776/K68。运行合同N128..799，实际训练最大776，不等于所有N桶都覆盖。新增bin22/26/31/38仍各一父；原bin15/19没有补第二父，bin18没有训练父。617/756仍留出；“最大N范围包住617”不能改写其计数桶与父身份未曝光。

自然N窄B起点为曝光B的global21000/native9000，终点global22500/native10500。其每步3个OLD父和1个交替FOCUS/HIGH父，各有CLEAN/X_WIDTH两个实例。原损失登记名称为 `whole_effective_batch_free_coordinates`：令组g的有效free标量数为m_g，`L=sum_g SSE_g / sum_g m_g = sum_g [m_g/sum_h m_h] * MSE_g`。因此OLD/FOCUS/HIGH的75%/12.5%/12.5%是实例份额，不能当固定组loss系数。

W的HIGH实例比例仍12.5%，但累计free坐标份额由窄B的21.418%升至28.869%，逐更新平均HIGH组MSE系数19.852%→24.307%。M使用登记名 `narrow_reference_group_MSE_coefficients_v1` 恢复该系数至19.852%，并未恢复每HIGH父750次曝光。750个HIGH更新采用 `rho=m_HIGH_B/m_HIGH_W`，`D_B=m_OLD+m_HIGH_B`，`L_M=(SSE_OLD+rho*SSE_HIGH_W)/D_B`；对应组MSE系数是OLD的`m_OLD/D_B`与HIGH的`m_HIGH_B/D_B`，其余750个FOCUS更新不变。19.852%含HIGH系数为零的FOCUS步，仅在HIGH存在的750步均值为39.704%。EXPOSURE继续M这一实际目标。它是当前候选的真实组成，**M的对照本身并未证明该加权方式优于普通归一化**，不能把全部组成分别宣传成有效机制。实例份额、坐标份额、MSE系数与实测loss贡献都不是梯度份额。

谱系来源：[完整座面报告](../../experiments/r2_bench_full_seat_curriculum_v1/REPORT.md)、[Coverage报告](../../experiments/r2_bench_seat_coverage_v1/REPORT.md)、[曝光报告](../../experiments/r2_bench_exposure_allocation_v1/REPORT.md)、[自然N报告](../../experiments/r2_bench_natural_n_recipe_v1/REPORT.md)、[W报告](../../experiments/r2_bench_high_joint_coverage_v1/REPORT.md)、[M报告](../../experiments/r2_bench_group_weight_match_v1/REPORT.md)、[四方向方法](../../experiments/r2_bench_four_routes_v1/METHODS.md)、[EXPOSURE训练登记](../../experiments/r2_bench_four_routes_v1/exposure/training_registration.json)、[SEAM训练登记](../../experiments/r2_bench_four_routes_v1/seam/training_registration.json)、[FLIP训练登记](../../experiments/r2_bench_c_edgeflip_v1/training_registration.json)。

## 每轮同面板收益与代价

以下只描述各轮自己的配对，不把30、38、46请求或最新6+4请求混排。FF/CF都是受非法面对影响的实际free面比例；I为C有效支持域的ConflictCov_C，三个分母不同。百分数按各输出原分母计算后父等权；面数和面积必须一起看。视觉“三轴交集”为合理主体∩合理C采用∩主要支撑存在，均是非盲AI_PROXY描述标签，不认证无相交、无碎片、水密或真实共边。

| 同面板对照 | 真实正信号 | 同时存在的代价/证据边界 |
|---|---|---|
| Coverage：Native→Coverage，30请求 | 新2父4请求RAW交集2→4/4，新增均a8；NEW FINAL FF20.87→1.79%、I2.73→0 | 同组FINAL交集3→0/4，free面57.75→30.75、面积20.52→3.80、小角57.28→71.14%；RAW CF7.16→15.04%、I6.85→10.02%。扩池学习可见，完整流程未晋级 |
| 曝光：A→B，30请求 | OLD RAW交集12→14/16、FINAL11→12/16；DEV FINAL FF21.683→11.783%，四父均降，且比A多3.875张free、面积多1.344 | OLD FF14.158→15.057%、I0.610→1.349%；FOCUS FINAL仍0/4。DEV相对冻结Coverage仍少11.875张free、面积少4.092，新增FINAL交集仅459/n0一例且有碎条 |
| 自然N：CONTROL_OLD A→NATURAL_HIGH B，38请求 | 新HIGH两父RAW交集0→4/4；HIGH RAW FF92.096→68.510%，FINAL16.026→1.584%；DEV FF21.137→11.494%，四父均降 | HIGH FINAL交集仍0/4，I是LOW_SUPPORT/NA；RAW/preGEO/FINAL free面积23.577/8.437/8.412，319原槽均值只留51。DEV I10.376→11.719%，起点仅1.044%；高N留出无新增交集 |
| W：窄B→宽W，46请求 | 新四父FINAL FF34.823→21.612%；945/n1 RAW新增合理采用，新增四父RAW交集两者6/8 | DEV FF11.494→20.139%；新四父FINAL面积28.312→10.189、交集4→1/8；bcb/n1用途退化抵消945的RAW新增。未形成留出采用增益 |
| M：W→M，46请求 | 新四父FINAL FF21.612→18.163，三父两噪声都下降；OLD CF6.723→5.928%、FF13.890→12.879%；4cd/n1 POST端撑保留改善 | OLD I1.553→5.664%，主要4cd/n1由3.369→69.629%；HIGH留出I4.822→8.508%。DEV交集仍0/8；新增四父FINAL交集仍1/8。M不晋级 |
| EXPOSURE：M→M+3000，46请求 | RAW交集29→33/46（新增5、丢1）；RAW FF88.123→87.443%、I10.833→8.852%。FINAL FF14.260→9.044%、I6.755→4.828%、CF7.602→7.350%；合法CF输出14→19/46 | FINAL交集17→15/46（无新增、丢2）；DEV FINAL CF5.694→8.013%、I11.237→12.744%。保留生成器候选，不能升级为完整成品流程 |
| 四方向SEAM：窄B→SEAM1500，46请求 | FINAL CF7.047→5.130%；42/42开放C请求都有合法CF边；全free方向可达面积8.165→24.111% | FINAL FF13.056→18.994%、I4.247→7.588%；交集21→18/46。真正seam→bulk可达面积≥30%仅5/42，DEV0/8。固定底边接C不等于接入主体 |
| 最新200步：EXPOSURE/PLAIN/FLIP，原6请求 | FLIP在459/n0对换划分更稳定；相对PLAIN小角66.699→63.600%；索引单面孤岛19.667→14.667 | FINAL FF12.414/18.808/23.099%，合法CF输出2/2/1；RAW交集3/2/1、FINAL1/0/0。FLIP的bcb/n1主体/C错置且合法边丢失；PLAIN和FLIP都不晋级 |

早期精确结果与独立登记：[Coverage](../../experiments/r2_bench_seat_coverage_v1/REPORT.md)、[曝光A/B](../../experiments/r2_bench_exposure_allocation_v1/REPORT.md)、[自然N](../../experiments/r2_bench_natural_n_recipe_v1/REPORT.md)。后段结果：[W](../../experiments/r2_bench_high_joint_coverage_v1/REPORT.md)、[M](../../experiments/r2_bench_group_weight_match_v1/REPORT.md)、[EXPOSURE](../../experiments/r2_bench_four_routes_v1/exposure/REPORT.md)、[SEAM](../../experiments/r2_bench_four_routes_v1/seam/REPORT.md)、[最新200步](../../experiments/r2_bench_c_edgeflip_v1/REPORT.md)。

窄B相对CONTROL_OLD的几何收益值得单独保留，而不只说“仍失败”：同38请求中，OLD FINAL CF6.366→6.276%、FF10.385→9.182%、I1.636→1.486%；HIGH_TRAIN CF5.615→0%、小角52.481→31.414%；HIGH_HOLDOUT FF21.181→12.481%、I4.797→4.553%。但是OLD方向可达面积11.085→9.021%，同轮明确碎片/孔洞11→13/16；DEV CF5.872→6.039%、I10.376→11.719%；HIGH_HOLDOUT CF20.518→20.748%、FINAL面积34.028→30.085，视觉2/4全部来自已有617正例，没有新增留出采用。HIGH_TRAIN的CONTROL_OLD/B FINAL面积22.331/8.412、原槽存活52.5/51；B的RAW三轴4/4没有延续到FINAL。该轮LOW_SUPPORT不能用于声称零I冲突。

EXPOSURE的RAW净增四例有明确身份：新增6b/n1、2d/n1、459/n1、945/n0、bcb/n1，丢失4cd/n1。两份DEV新增来自两个父各一噪声，DEV RAW交集0→2/8、FINAL仍0/8。NEW_WIDE_TRAIN RAW交集6→8/8，FINAL1→0/8；不能用熟悉四父的RAW8/8声称跨父泛化。最终丢失为4cd/n1、af/n0。

EXPOSURE全组FINAL/RAW free面积比57.092→58.814%，平均删除原槽219.783→216.304，说明全组FF下降不能全部归于“多删面”。但最终free面积均值20.094→19.375；新增四父FINAL/RAW比38.051→35.803%、最终面积10.788→6.969，细腿框、圆弧端撑仍损失。RAW到FINAL面积比包括D0位移、合并、删面及桥面，不是来源GT覆盖率或完整性认证。原数据：[完整177字段](../../experiments/r2_bench_four_routes_v1/exposure/results/per_output.csv)、[主口径汇总](../../experiments/r2_bench_four_routes_v1/exposure/results/delivery_v1/summary.json)、[逐槽账本](../../experiments/r2_bench_four_routes_v1/exposure/results/analysis_v1/deletion_fate_per_slot.csv)。

最新翻边4请求独立于原6：EXPOSURE/PLAIN/FLIP FINAL FF18.246/8.856/22.329%，共同C的CF5.108/0/0%、I3.455/0/0%，FINAL/RAW面积比64.195/39.658/51.708%。PLAIN低冲突伴随大量面积损失。共同物理C的额外CF/I没有覆盖原177字段；f495几何门槛BLOCKED，没有生成翻边版，也不计作模型失败。该轮只有2个可翻评价C、3父旧开发噪声，不能将其46请求以外的数值拼成EXPOSURE质量涨跌。

## 为什么优先保留EXPOSURE，而不宣称通用SOTA

EXPOSURE是这条近期主线中，正式同面板证据最明确保留的生成侧候选：原FM/数据方法下补足曝光，带来五个可列名的RAW新增交集、一个丢失，以及FF/I总体收益。它没有新增网络或辅助loss；当前证据更支持“完整条件与自然N合法数据，加上足够训练曝光”的研究方向，而非已经验证某个额外几何loss或内部翻边增强。3000步也同时增加OLD/FOCUS曝光，未单独证明HIGH曝光或数据规模因果；没有等预算W续训臂证明M归一化必要。

不能将同46请求上不同分支的视觉基线拼成EXPOSURE、窄B、SEAM的统一排行榜。四方向中相同M在EXPOSURE/HYBRID图审的RAW交集为29/30、FINAL为17/15；相同B在SEAM/BESTOF图审FINAL为21/23，几何和原177字段完全一致。早期也有同几何重审漂移：Coverage113/720轴、曝光P0 47/450轴、自然N P0 66/450轴。只能使用同次配对和具体观察，不以跨版本标签计数宣称“已经最优”。

RAW独立角点三角汤缺少索引共边，不应以RAW索引孤岛数量替代视觉碎片数。C坐标精确保留不等于C被合理采用；SEAM固定底边构成合法C侧边也不等于bulk已接入。I的LOW_SUPPORT必须保留NA：四方向46请求的主I仅共同PASS19/23父、38/46请求，其余28b9/abf/945/af不补零；闭合C不强求开放边接口。

四方向HYBRID和BESTOF4是不同成本的后端/选择路线，不能算成新的训练配方。HYBRID只处理M，未运行“EXPOSURE+HYBRID”；它恢复更多RAW支撑，但FINAL FF14.260→33.075%、I6.755→9.667%。BESTOF4的FINAL交集固定n0/n1/选择为11/12/10（23C），不晋级。两者不为EXPOSURE提供未经执行的组合收益。来源：[四方向总报告](../../experiments/r2_bench_four_routes_v1/REPORT.md)。

旧SEAM3000仍有自身确定边界：同24请求里，训练父FINAL CF COMMON/Native/SEAM=13.23/9.33/2.09%、FF34.58/14.57/10.25%；但DEV FF26.87/24.05/27.21%，SEAM真正bulk可达≥30%仅训练2/16、DEV0/8。它是保留的接续研究参考，不是已解决整体泛化的SOTA。维护索引的“当前研究参考”与10月27–28日“最新生成器候选”是不同身份；本综述不切换默认。

## 现存终点（仅stat）

以下路径相对`E:/MeshFlow/experiments/`。2026-10-28只用`Get-Item`检查存在及长度，所有列出文件现存。生成器约0.522GB、完整恢复约1.565GB；完整文件包含匹配恢复状态，不用`checkpoint_latest.pt`冒充正式终点。10月28日清理已去掉8份旧latest中间点，正式终点保持。更早保留策略见[保留模型索引](../storage_cleanup_20261025/RETAINED_MODELS.md)与[清理报告](../storage_cleanup_20261025/CLEANUP_REPORT.md)。

| 身份 | 目录 | 推理文件 | 完整恢复文件 | 长度bytes：推理/完整 |
|---|---|---|---|---:|
| Coverage | `r2_bench_seat_coverage_v1/training/NATIVE/` | `generator_step1500.pt` | `checkpoint_step1500.pt` | 521670963 / 1565060835 |
| 曝光A | `r2_bench_exposure_allocation_v1/training/A_50_50/` | `generator_step1500.pt` | `checkpoint_step1500.pt` | 521670963 / 1565061027 |
| 曝光B/P0 | `r2_bench_exposure_allocation_v1/training/B_75_25/` | `generator_step1500.pt` | `checkpoint_step1500.pt` | 521670963 / 1565061027 |
| 自然N A | `r2_bench_natural_n_recipe_v1/training/CONTROL_OLD/` | `generator_step1500.pt` | `checkpoint_step1500.pt` | 521671027 / 1565061155 |
| 窄池B | `r2_bench_natural_n_recipe_v1/training/NATURAL_HIGH/` | `generator_step1500.pt` | `checkpoint_step1500.pt` | 521671027 / 1565061155 |
| W | `r2_bench_high_joint_coverage_v1/training/WIDE_HIGH/` | `generator_step1500.pt` | `checkpoint_step1500.pt` | 521671027 / 1565061219 |
| M | `r2_bench_group_weight_match_v1/training/W_MATCHED/` | `generator_step1500.pt` | `checkpoint_step1500.pt` | 521671027 / 1565061283 |
| EXPOSURE | `r2_bench_four_routes_v1/exposure/training/M_PLUS3000/` | `generator_step3000.pt` | `checkpoint_step3000.pt` | 521671091 / 1565061283 |
| 新SEAM | `r2_bench_four_routes_v1/seam/training/SEAM1500/` | `generator_step1500.pt` | `checkpoint_step1500.pt` | 521671027 / 1565061219 |
| PLAIN200 | `r2_bench_c_edgeflip_v1/training/PLAIN200/` | `generator_step200.pt` | `checkpoint_step200.pt` | 521671027 / 1565061155 |
| FLIP200 | `r2_bench_c_edgeflip_v1/training/FLIP200/` | `generator_step200.pt` | `checkpoint_step200.pt` | 521671027 / 1565061155 |

最优先候选的直接路径：[EXPOSURE推理](../../experiments/r2_bench_four_routes_v1/exposure/training/M_PLUS3000/generator_step3000.pt)、[EXPOSURE完整恢复](../../experiments/r2_bench_four_routes_v1/exposure/training/M_PLUS3000/checkpoint_step3000.pt)。保留该终点不授权续训、切默认或运行新组合。
