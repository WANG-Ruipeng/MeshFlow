# Coverage + GEO_ONLY：完整固定面板候选验收

用户2026-10-23授权按附件590f4c9d推进。先验收冻结Coverage1500配原GEO_ONLY的完整流程。复用原Native→Coverage全部30请求：CURRENT_TRAIN 8父16份、DEV 4父8份、NEW_TRAIN a8/27共4份、NEW_PARENT_HOLDOUT df6共2份。所有请求同一种GEO_ONLY后端，不按父、面数或结果切换D0。

## 固定范围

- 30个既有Coverage RAW：4个a8/27 GEO_RAW复用上轮，新增26次原GEO_C；0训练/生成/新噪声/D0（验收阶段）。
- 完整原评分30份（4份补齐全字段时核对已有CF/FF/I完全一致，不称4个新样本），原Coverage和Native的RAW/BASE/FINAL共180个评分记录只读。原C查询、分母、尺度、半径、代表点顺序和来源定向保持。
- 30页，每页Coverage RAW、BASE_POST、CLEAN_FINAL_POST、GEO_ONLY及Native CLEAN_FINAL_POST；同两个原视角与两个原位置截面。不能将BASE与含HINGE/ALPHA/HOLE的FINAL混称同阶段。
- 原177数值字段及NA状态保存，特别记录CF/FF/I、非法面对数、实际合法CF共边/长度、连通分量/碎片、非流形/小角、删除面数/面积。Native不具有SEAM角色，SEAM专属指标NA，不能填0。

## 图审与汇总

采用原五轴：overall_shape(REASONABLE/INCOMPLETE/UNCERTAIN)、C_use(PLAUSIBLE/REPLACED_OR_IGNORED/UNCERTAIN)、support_retention(PRESENT/MISSING_OR_DAMAGED/UNCERTAIN)、visible_fragment_or_gap(YES/NO_OBVIOUS/UNCERTAIN)、layering_or_fold(YES/NO_OBVIOUS/UNCERTAIN)。必须实际打开图片后填写非盲AI_PROXY，不从指标或历史标签复制。每页五列均记录，页hash和查看者保存。原中央截面可能错过端腿，截面断点不自动是物理裂口，需同时看两表面视图。

主要视觉交集为同输出C_use=PLAUSIBLE且overall_shape=REASONABLE且support_retention=PRESENT。四个数据组分开，先噪声平均再父等权；旧DEV为重复使用开发集，df6为当前适配谱系留出，不称官方预训练未见或全项目首次出现。

## 执行前验收规则

这是开发性工程验收，不是新独立测试，也没有文献规定的可接受FF阈值。以下明确告警线仅用于决定是否值得进入后端适配准备，不能被写成普适质量标准：

1. Candidate继续资格要求四组视觉交集均不低于同Coverage原FINAL，且支撑保留收益不只来自同两NEW父；任何旧正例丢失单列，不以总体面积抵消。
2. 若至少两个非NEW父相对同Coverage原FINAL，父均FF增加>=10个百分点，或CF/I增加>=5个百分点，记录为广泛几何代价告警。逐请求原数量/分母同时报告，不能仅凭百分比定性原因。
3. 任一资格不满足，整份GEO_ONLY不作为下一独立测试的优先候选；保留局部正结果。即使全部满足，也仅登记候选，不自动替换维护默认或宣布泛化完成。
4. 数值和图审矛盾均保留；不按这些开发结果事后调半径/阈值/模型/请求，不新增选择器。Native后端表现是控制，不单独作为Coverage候选的否决条件。

## 条件性第二阶段与停止

仅若验收FAIL，按附件补两份已存来源目标(a8/27)各一次官方Bench D0及原GEO，预算2 D0前向、2 GEO，生成器保持冻结且零调用；其目的是检查identity/薄结构先验，不把来源当完美流形或真实生成唯一答案。先核验原D0 packing/scale，再记录新资源、1 GPU worker及调用账本。不得增加D0次数、扰动或第三来源。

若来源也收缩，则准备identity/薄结构校准数据合同；若来源稳定而RAW不稳定，则优先真实生成修复对。只能列出已存在、可追溯监督与缺项，不把原D0抹腿结果当教师，不将自由生成RAW逐槽配原Y，不把面坍缩当删除标签。

本轮不启动没有固定数据、预算及独立验证划分的后端训练。全面板验收、必要的两份identity检查和准备说明完成即停止。无默认切换、清理、commit或push。

资源：开工CPU24逻辑核、可用RAM61.23GiB，数学库每进程1线程。core单worker；随后评分最多3worker和渲染1worker并行。GPU阶段有独立新鲜快照与max(2GiB,10%总显存)余量，不影响用户其他进程。所有旧资产只读，新增放本目录。
