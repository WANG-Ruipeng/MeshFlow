# CHAIR_EARLY_ALIGNMENT_SCHEDULE_40_V1 — 方法与解释合同

本文件在训练进行中写入，不包含效果结论。最终结果以 EXPERIMENT_REPORT.md、完整逐输出表和实际图像审阅为准。

## 两个比较问题

- 新 FM_ONLY、EARLY、LATE：相同官方 Chair EMA、条件初始化、40%条件数据、HYBRID、AdamW与输入流，只改变 B 的时间窗口。
- 新三组对 START5000、历史 FM6000、历史 FREE_ALIGN6000：评价完整训练方案的实际输出；不能把差异单独归因于 B 提前。

历史 Geo 从最初条件适配已经存在，不应宣称本轮首次将 Geo 提前到第0步。旧流程先使用20%C，随后混合20/40%，HYBRID在4001步才启用；本轮使用现有26父203条40%任务，HYBRID从第1次更新启用。旧32父中6父没有合法40%任务，本轮没有另造训练条件。

## 固定训练合同

|路径|1–1000|1001–5000|5001–6000|B样本曝光|
|---|---|---|---|---:|
|FM_ONLY|FM|FM|FM|0|
|EARLY|FM+B|FM|FM|4000|
|LATE|FM|FM|FM+B|4000|

COMMON5000是FM_ONLY/LATE共享的完整模型、AdamW、RNG与输入游标前缀。因此逻辑更新3×6000，物理唯一更新13000。每批8个不同父、microbatch1，4个未增强样本与4个宽度增强样本；B仅用于未增强样本。λ_H=.25，λ_B=.1。辅助头初始参数一致；关闭B不重置生成器AdamW。新配方从官方EMA与fresh AdamW开始，没有拼接START动量。

6000×8份输入、实际噪声和OT映射在训练前冻结。三组相同更新编号的输入、配对、分母均核对hash，随机数不依赖worker完成顺序。官方参数及新增role/Geo初始化被单独核验，零出口不被解释为带C的T1前向与官方无条件前向相同。

严格确定性；CUBLAS_WORKSPACE_CONFIG=:4096:8；TF32关闭；SDPA math；BF16生成器前向；FP32参数与积分。复用冻结教师特征，无新增教师编码。每100步保存可恢复状态；双worker出现OOM才按授权全局降为单worker，最多一次降级且每个受影响任务一次恢复，已完成重放最多99步；失败/部分调用另记。

## 固定评价面板

8个已观察DEV父，每父一份实际Gaussian，六模型共48份Euler50 RAW。K=round(.4N)。6个条件沿用既有A40；341b…和c1b8…保留原C，再按最近free质心追加真实面片，后两者不假定可信接缝。完整来源只用于构造C，不用于推理或质量排名。主结果同时提供全部8父和6/2条件构造分层。没有新增手工稀疏C面板或多噪声确认。

RAW对RAW，无本轮后处理。C在51状态、最终NPZ和PLY中的坐标、面序与绕向逐位保持，并核对Gaussian、最终轨迹、N和有限性。该合同不证明保留原顶点ID图或实现合理接合。

## 指标实际含义

固定L=sqrt(3)*2/.3762=9.208138264587333，阈值不随输出bbox改变。原数学定义沿用，但尺度不同于部分旧父L0报告，不跨尺度直接拼数值。

- CF proper：扫描全部K×(N−K)对，报告真实横向穿越对数及受影响free面并集/(N−K)；共面正面积重叠另列。
- FF illegal：扫描全部不同free/free对，非法集合为proper与共面正面积重叠并集；受影响free面比例每面只计一次。
- 两者都保留未决/退化面对，未决不等于合法；未扫描C/C，因此不是全域Ri。主表PASS只表示执行与输出合同通过。
- 原I_C：C上2048个预先冻结面积采样点，穿越影响带和近重复覆盖取并集，分母为全部C采样权重。分别保存cross、dup、intersection、union、上下界及support状态。lower/upper为数值不确定范围，不是统计置信区间。
- I_cross：采样点距其所在C面的proper交线段小于.005L。I_dup：点距C精确开放边至少.01L，有非退化free面满足无向法向≤15°且有限表面距离<.005L。检查所有满足法向条件的free面。它可能计入合法薄壁，并非语义错误检测或全局FF近重复率。
- I复用并验证主评分CF缓存、输入、RAW和源码hash；不静默重算缺失缓存。原脚本版本与查询冻结在任何rollout之前。
- Free质量：逐面最小角<5°比例；全部3|F|角的P05；q=sum(edge_length²)/(4sqrt(3)A)，q>10比例；近退化A/L²≤1e−12及零面积A=0。q_mean只平均有限q，须与退化项并读。
- CF/I有效法向退化阈值1e−11L²不同于shape近退化1e−12L²，不混为同一计数。
- Free精确重复：(F数−无序三角坐标唯一数)/F数，忽略绕序；不等于近重复。整体形状统计与固定C自身背景另列，总面积无单调好坏方向。
- 原始三角汤及精确坐标焊接组件都是描述量，不当作肉眼碎片数量，不是水密/流形认证。

无来源GT距离、S/E评分或联合胜负门禁。每父等权，完整逐父配对保留失败、NA及分母。I的可用子集均值与完整8父均值分开；PARTIAL/LOW_SUPPORT状态另列。

## 完整图片及结论边界

全部48输出均展示。每条件一行输入C及六匿名方法，四固定视角加线框，同时中性材质与蓝C/橙F；同条件六方法使用联合完整RAW范围，包含飞片，不单独裁剪、平滑、修复或缩放。实际AI逐图检查另记，不编造人类可用率。

单训练流、单噪声、8个已观察DEV父用于快速架构探索，不能推断生成方差或广泛泛化。只有全部结果完成后再判断哪个方向值得下一步；不默认EARLY优越，不自动切换START、不增加训练、不commit/push。

## 账本路径

protocol.json；stream/train_plan.json、paired_inputs.json、precompute_receipt.json；teacher/teacher_manifest.json；evaluation/requests.json、condition_construction.json、conflict_registration.json；training/*/attempts/*/{events.jsonl,calls.jsonl,exposure.csv,training_log.csv}；audit/controller.jsonl、dispatch_state.json、final_accounting.json、input_pairing.json；source_code/、code.diff、audit/new_source_hashes.json。

final_accounting在运行中可为IN_PROGRESS。最终CPU审计与实际权重核验必须分别通过，不能仅凭任务脚本退出或图集文件存在宣布完成。
