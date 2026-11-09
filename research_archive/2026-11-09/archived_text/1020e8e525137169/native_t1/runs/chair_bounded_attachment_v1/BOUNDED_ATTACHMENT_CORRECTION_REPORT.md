# Bounded Attachment Correction Pilot V1 实际报告

本轮实际账本：训练 0 次，模型反向 0 次，优化器更新 0 次；新模型前向 0 次，新完整采样尝试 0 次（Euler NFE 0）。复用原 RAW 16 份，正式 ENDPOINT 几何修正调用 16 次，TRAJECTORY 内几何修正 0 次。CPU 固定小配置 8/8，配置内实际 project 调用 12 次另列，不算模型调用。失败调用 0，未决调用 0。

**结论：NO_PHASE1_SIGNAL；保留 RAW。** 阶段一没有通过冻结联合门禁，保留 START 原始输出；不启动轨迹干预。

本轮交付状态：COMPLETE。阶段一门禁 FAIL；阶段二 NOT_RUN_NO_PHASE1_SIGNAL。

## 固定身份与实际执行

模型始终为 OT_HYBRID pilot1000 / conditional_total5000，训练 coupling λ=0.25。原完整文件 /mnt/e/MeshFlow/native_t1/runs/chair_hybrid_coupling_v1/checkpoints/OT_HYBRID_pilot1000_total5000.pt 的 SHA256 为 d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80；本轮注册实际全文件核对，没有重新训练、替换模型或恢复 AdamW。

固定原 A20 验证 8 父对象、每父 candidate 0/1，共 16 条。C、N/K、原 Gaussian、known/valid mask、α=1、原 50 步 Euler、BF16 forward / FP32 积分身份全部复用。历史前后 runtime 的确定性、TF32 禁用、math SDPA 与 C 的 51 状态保持记录已核验。没有重新抽噪声或 GT 点；未保存完整状态的旧样本仅能核验旧逐步审计记录与现存 raw，不声称本轮重新检查不存在的状态数组。

新推理 bank 仅含 z/C/known_mask/valid_mask/N/K；GT、Gamma、源面身份与评分参考留在独立 condition/评分文件。project 只接收 C/F/remaining_budget/固定配置，16 份修正与接受/拒绝日志全部保存后才读取 GT 评分。

原 Partner Closure 曾在全局连接后产生大幅位移、退化与冲突，本轮采用一次局部双端互为最佳、严格有界锚定，并允许不修改。旧 boundary 的 NO_ATTACHMENT_SIGNAL 结论保持不变，本轮不把它改写成训练收益。

## 六个必须回答的问题

1. **16 例中实际改了多少，接受多少边？** 已保存 16/16，修改 1 例，接受 3 条约束。状态分布：NO_CANDIDATE=15、PARTIALLY_CORRECTED=1。无变化、拒绝与不适用局部候选仍留在全体分母。

2. **strict 接缝是否提高，原 1% 是否保持？** ENDPOINT−RAW 的 strict SeamCov(τ=0.001)父均差为 0.01034818，父配对中位数 0，改善父 1/8。原 τ=0.01 父均差 0.0001150791。全部 .005/.01/.02 旧口径保留，strict 是合同预登记细容差，不覆盖原定义。

3. **冲突或遮挡代价如何？** ConflictCov_C 父均差 -3.051758e-05，ExposedRetain 父均差 0。数值含确认下界；未知域的 global 上下界用于冻结门禁。局部 project 的无新非法事件检查不能替代独立 CF/FF、重复覆盖与可见性评分。

4. **主体质量与移动付出什么代价？** D_sym2048 父均差 -8.861665e-10，原固定 512 点 MSE 差 -7.031473e-10，原 p95 差 0。每输出移动、自由面积、小角面、CF/FF 与退化见质量表和逐例 CSV。

5. **TRAJECTORY 执行了吗，相比 ENDPOINT 如何？** 未执行。阶段一 FAIL，依合同不加载模型、不消耗补充前向或采样预算；所有轨迹格明确标 NOT_RUN。

6. **保留哪个方法，下一步是什么？** RAW。阶段一没有通过冻结联合门禁，保留 START 原始输出；不启动轨迹干预。

## 主指标：每父先平均两噪声，再八父等权

|指标|RAW|ENDPOINT|TRAJECTORY|
|---|---|---|---|
|strict SeamCov τ=.001|0.001875101|0.01222328|NA|
|SeamCov τ=.005|0.05023871|0.05035379|NA|
|原 SeamCov τ=.01|0.1426901|0.1428052|NA|
|SeamCov τ=.02|0.2788909|0.279006|NA|
|ConflictCov_C 确认下界|0.04251099|0.04248047|NA|
|ConflictCov_C 上界|0.04251099|0.04248047|NA|
|ExposedRetain 确认可见下界|0.8448645|0.8448645|NA|
|Exposure global 下界|0.8448645|0.8448645|NA|
|Exposure global 上界|0.8449255|0.8449255|NA|

低冲突更好；高接缝覆盖、暴露保留更好。Gamma 几何容差内的覆盖不是拓扑焊接、水密性或百分之百可用。PARTIAL 不直接作好或坏解释；门禁使用保存的保守区间。

## 质量与实际移动：与主控制指标分列

|指标|RAW|ENDPOINT|TRAJECTORY|
|---|---|---|---|
|D_sym2048|0.0016198|0.001619799|NA|
|GT→F MSE2048/L0²|0.001510906|0.001510905|NA|
|F→GT MSE2048/L0²|0.001728693|0.001728692|NA|
|原固定512 MSE|0.001531074|0.001531073|NA|
|原 p95/L0|0.06884473|0.06884473|NA|
|CF proper pairs|15.0625|15.0625|NA|
|FF illegal pairs|235.8125|235.6875|NA|
|精确零面积面|0|0|NA|
|近退化面|0|0|NA|
|自由面最小角<5°比例|0.1467566|0.1467566|NA|
|free/GTfree 面积|1.081587|1.081656|NA|
|delta/L0 仅评价|0.0066199|0.0066199|NA|
|单输出最大角移动的父均|0|0.002962166|NA|
|单输出RMS角移动的父均|0|0.0003133011|NA|
|变化角比例|0|0.001179245|NA|
|变化面比例|0|0.001768868|NA|

ENDPOINT 全 16 输出最大实际角移动 0.04739466 模型单位，最大 move/delta=0.9214524；接受方程最大残差 0。delta=0.01×C 包围盒对角线，每次还受 0.1×较短局部边长约束。原脸和角对应保持时，每角位移界给出相应三角面的 Hausdorff 上界；不推出 MSE 单调改善或拓扑连接保证。

接受检查仅覆盖实际改变的 free face 所触及的面片对，原非法交可保留；不是全局清理。修改例的实际事件：[{"uid": "86ec9c796a6c1789e792598ed58dd1cc", "candidate": 0, "scope": "Pairs touching actually changed free faces; unaffected pairs unchanged", "removed": 2, "added": 0, "remaining": {"('CF', 'proper_transverse_penetration')": 2, "('FF', 'proper_transverse_penetration')": 2}}]。

## 全八父 strict 配对与两噪声

|父对象|RAW|ENDPOINT|TRAJECTORY|ENDPOINT−RAW|
|---|---|---|---|---|
|33ec1e64a02ff718faaadcdc0221eb72|0.001069518|0.001069518|NA|0|
|341bcae1e7f366a233d93f57729688db|0|0|NA|0|
|534b0e6e2c53a8e1a7cdfd3874ed3723|0.0009308275|0.0009308275|NA|0|
|866e7248434968051cc7750d0afca796|0.001385383|0.001385383|NA|0|
|86ec9c796a6c1789e792598ed58dd1cc|0.01027754|0.09306295|NA|0.08278541|
|bdd29e651e5f6fb2b079317292bdc5d4|0.001337542|0.001337542|NA|0|
|c1b8c2f0c4f3cc43aff7c08b060f5ed6|0|0|NA|0|
|f59627a077dcc2566c63b8b48495261a|0|0|NA|0|

|对比|指标|均值差|配对中位数|改善父|最小差|最大差|
|---|---|---|---|---|---|---|
|ENDPOINT-RAW|SeamCov_strict|0.01034818|0|1/8|0|0.08278541|
|ENDPOINT-RAW|SeamCov|0.0001150791|0|1/8|0|0.0009206329|
|ENDPOINT-RAW|ConflictCov_C|-3.051758e-05|0|1/8|-0.0002441406|0|
|ENDPOINT-RAW|ExposedRetain|0|0|0/8|0|0|
|ENDPOINT-RAW|D_sym2048|-8.861665e-10|0|1/8|-7.089332e-09|0|
|ENDPOINT-RAW|fixed_query_MSE|-7.031473e-10|0|1/8|-5.625179e-09|0|
|ENDPOINT-RAW|fixed_query_p95_L|0|0|0/8|0|0|
|TRAJECTORY-RAW|SeamCov_strict|NA|NA|None/8|NA|NA|
|TRAJECTORY-RAW|SeamCov|NA|NA|None/8|NA|NA|
|TRAJECTORY-RAW|ConflictCov_C|NA|NA|None/8|NA|NA|
|TRAJECTORY-RAW|ExposedRetain|NA|NA|None/8|NA|NA|
|TRAJECTORY-RAW|D_sym2048|NA|NA|None/8|NA|NA|
|TRAJECTORY-RAW|fixed_query_MSE|NA|NA|None/8|NA|NA|
|TRAJECTORY-RAW|fixed_query_p95_L|NA|NA|None/8|NA|NA|
|TRAJECTORY-ENDPOINT|SeamCov_strict|NA|NA|None/8|NA|NA|
|TRAJECTORY-ENDPOINT|SeamCov|NA|NA|None/8|NA|NA|
|TRAJECTORY-ENDPOINT|ConflictCov_C|NA|NA|None/8|NA|NA|
|TRAJECTORY-ENDPOINT|ExposedRetain|NA|NA|None/8|NA|NA|
|TRAJECTORY-ENDPOINT|D_sym2048|NA|NA|None/8|NA|NA|
|TRAJECTORY-ENDPOINT|fixed_query_MSE|NA|NA|None/8|NA|NA|
|TRAJECTORY-ENDPOINT|fixed_query_p95_L|NA|NA|None/8|NA|NA|

|对比|噪声|strict均值差|strict中位数|改善父|
|---|---|---|---|---|
|ENDPOINT-RAW|0|0.02069635|0|1/8|
|ENDPOINT-RAW|1|0|0|0/8|
|TRAJECTORY-RAW|0|NA|NA|None/8|
|TRAJECTORY-RAW|1|NA|NA|None/8|
|TRAJECTORY-ENDPOINT|0|NA|NA|None/8|
|TRAJECTORY-ENDPOINT|1|NA|NA|None/8|

两噪声是同父重复，不是 16 个独立父对象。接受边子集只用于解释适用范围，不改变主结果八父等权。

## 冻结门禁逐项证据

### Phase 1：FAIL

|检查|状态|保守下界|保守上界|阈值|
|---|---|---|---|---|
|1 strict_SeamCov_mean_gain|FAIL|0.01034818|0.01034818|0.02|
|2 positive_parent_median_and_at_least_four_improve|FAIL|NA|NA|None|
|3 old_SeamCov_mean_drop|PASS|-0.0001150791|-0.0001150791|0.005|
|4 ConflictCov_C_mean_increase|PASS|-3.051758e-05|-3.051758e-05|0.01|
|5 ExposedRetain_mean_drop|PASS|-6.103516e-05|6.103516e-05|0.01|
|6 every_parent_conflict_or_exposure_cost|PASS|0|0.0004882812|0.05|
|7 D_sym_and_original_MSE_relative_mean_cost|PASS|NA|NA|None|
|8 original_p95_relative_mean_cost|PASS|NA|NA|None|
|9 C_N_finite_budget_accepted_equations_no_new_degeneracy|PASS|NA|NA|None|

完整门限、父极端、质量相对变化和原 RAW 评分一致性见 [phase1_gate.json](phase1_gate.json)。门禁 status 是原聚合器输出，本报告未改变阈值或丢弃失败样本。

## 成本与资源

|方法|实际评分输出|历史基础生成秒/输出|project秒/输出|部署总秒/输出|
|---|---|---|---|---|
|RAW|16|2.735413|NA|2.735413|
|ENDPOINT|16|2.735413|0.09075268|2.826166|
|TRAJECTORY|0|NA|NA|NA|

|ENDPOINT project 子项|16条实际累计秒|
|---|---|
|candidate_construction|1.32221|
|acceptance_checks|0.1261058|
|coordinate_writes|7.4671e-05|
|total|1.452043|

ENDPOINT 两个独立 CPU worker、每 worker 2 线程，wall 2.150434 秒。total 包含候选构造、接受几何检查和写坐标，不重复相加。部署 ENDPOINT 仍需原始生成成本；历史 GPU 生成与本轮并行 CPU 时刻不同，不能据此宣称训练或内核加速。评分与渲染属于独立离线成本，图库本轮渲染 wall 9.466239 秒。若阶段二执行，50 NFE 不意味着没有额外几何成本。

资源快照保存在 resources/snapshots.jsonl；WSL/WDDM 下不可获得的单进程 GPU 占用为 NA，设备占用包含桌面。未启动阶段二时，本轮没有模型 GPU 运行峰值可报告。

## 范围、工程修复与唯一下一步

这是 8 个已观察开发对象的小型配对试验，不能声称新父泛化、人类可用性或模型已学会精确贴合。局部算法保留原非法事件，不做全局清理；原始面片没有索引合并。接受检查为原浮点几何判别，不是精确谓词或连续碰撞认证。有限查询的冲突/暴露指标及不确定范围保持完整披露。

注册第一次把旧 quick 查询的 array_sha 误读成 common.array_hash。已确认旧数组和文件没有变化，按原 hash 定义修复新读取器；首次源码、失败登记和 code.diff 完整保留于 engineering/registration_attempt1。该修复涉及零模型、零 project、零评分调用，不属于算法或阈值调整。

继续保留 RAW / START，停止本轮局部修正路线。本固定任务不追加失败分类研究、训练、采样或参数扫描。

本固定任务到此停止；没有放宽预算、扫描匹配/求解器、自动启动训练、修改 loss 或切换默认模型。

## 交付索引

- [16×3 全量图库](gallery/index.html)、[逐例 CSV](per_output_metrics.csv)、[全部父均表](aggregation/per_parent_metrics.csv)。
- [父配对](aggregation/paired_parent.csv)、[噪声配对](aggregation/paired_noise.csv)、[父摘要](aggregation/paired_parent_summary.csv)。
- [协议](protocol.json)、[projection config](projection_config.json)、[模型与输入manifest](registration/model_input_manifest.json)、[RAW身份审计](registration/raw_identity_audit.json)。
- 已执行的 outputs/{request_id}/ 保存实际 raw.npz、mesh.ply、result.json；修改输出另含 intervention.json 和 geometry_arrays.npz。未执行 TRAJECTORY 仅保存明确的 NOT_RUN 状态 result.json。
- [账本](accounting.json)、ledger/calls.jsonl、tests/project/、[端点执行摘要](endpoint_summary.json)、[图库帧审计](gallery/frames.json)。
- [质量成本紧凑表](quality_cost.csv)、[实际源码hash](source_hashes.json)、[代码diff](code.diff)、[保护依赖审计](audit/protected_dependencies.json)、[全部制品清单](artifact_manifest.json)。
- 实际新源码及旧评分依赖位于 source_snapshots/new 与 source_snapshots/dependencies；[恢复入口](README_RESUME.md)。
