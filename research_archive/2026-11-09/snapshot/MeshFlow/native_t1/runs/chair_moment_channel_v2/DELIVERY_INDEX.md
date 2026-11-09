# Chair Moment Channel Learning V2 交付索引

状态：固定任务 DONE；质量结论 **NO_CLEAR_LEARNING_SIGNAL**。训练、终点评价和CPU独立审计均已实际执行。旧模型、数据和报告保留。

优先阅读 [完整报告](CHAIR_MOMENT_CHANNEL_V2_REPORT.md)、[独立执行审计](independent_audit/final_audit.md) 和 [独立质量复核](independent_audit/quality_review.md)。本目录为完整交付，不另行复制旧权重或制作重复ZIP。

|证据|路径与内容|
|---|---|
|协议与完成状态|[protocol.json](protocol.json)、[protocol.sha256](protocol.sha256)、[RUN_COMPLETE.json](RUN_COMPLETE.json)、[原任务文本](registration/task_prompt.txt)|
|逐输出与逐父|[per_output.csv](paired_results/per_output.csv) 共432登记行；[per_parent.csv](paired_results/per_parent.csv)、[paired_per_output.csv](paired_results/paired_per_output.csv)、[paired_per_parent.csv](paired_results/paired_per_parent.csv)|
|主指标三组配对|[primary_summary.csv](paired_results/primary_summary.csv)、[primary_paired_parents.csv](paired_results/primary_paired_parents.csv)、[primary_noise_pairs.csv](paired_results/primary_noise_pairs.csv)、[primary_noise_summary.csv](paired_results/primary_noise_summary.csv)|
|逐输出/逐曲线宽度|[width_per_output.csv](paired_results/width_per_output.csv)、[width_per_curve.csv](paired_results/width_per_curve.csv)、[width_paired_parents.csv](paired_results/width_paired_parents.csv)、[width_noise_summary.csv](paired_results/width_noise_summary.csv)、[noise_pairs_per_parent.csv](paired_results/noise_pairs_per_parent.csv)|
|完整3×3与方向|[cross_matrix.csv](paired_results/cross_matrix.csv) 共1296单元；[target_column_audit.csv](paired_results/target_column_audit.csv)、[target_column_noise_pairs.csv](paired_results/target_column_noise_pairs.csv)、[control_curve_counts.csv](paired_results/control_curve_counts.csv)、[方向正控](tests/scoring_cpu_tests.json)|
|质量与覆盖分区|[quality_paired_summary.csv](paired_results/quality_paired_summary.csv)、[coverage_partition.csv](paired_results/coverage_partition.csv)；距离/接口/相交/角度与GT参照均保留原字段|
|实际逐例评分|[START](evaluation/START/scoring/)、[H_CONT_END](evaluation/H_CONT_END/scoring/)、[MOMENT_XY_END](evaluation/MOMENT_XY_END/scoring/) 的 scores/及结果表|
|终点raw与生成身份|[H_CONT_END](evaluation/H_CONT_END/)、[MOMENT_XY_END](evaluation/MOMENT_XY_END/)；[START缓存身份](evaluation/START/)；[初始回放](initial_replay/)|
|检查点|[checkpoint_manifest.json](checkpoint_manifest.json)；[H_CONT终点](checkpoints/H_CONT_moment1000_total6000.pt)、[MOMENT终点](checkpoints/MOMENT_XY_moment1000_total6000.pt)；checkpoints/另保留200/400/600/800恢复点及每个检查点的JSON身份|
|配对训练日志|[training/](training/)、[training_selection.json](registration/training_selection.json)、[train_plan.json](train_plan.json)、[独立首更新检查](independent_audit/training_first_updates_MOMENT_XY.json)|
|padding与数值合同|[precheck/](precheck/)、[numerical_comparisons.csv](paired_results/numerical_comparisons.csv)、[CPU数学对照](tests/attention/results.json)、[数值独立复核](independent_audit/precheck_review.md)|
|成本与资源|[cost_summary.csv](paired_results/cost_summary.csv)、[sampling_cost_per_output.csv](paired_results/sampling_cost_per_output.csv)、[precheck_cost_summary.csv](paired_results/precheck_cost_summary.csv)、[phase_summary.csv](resources/phase_summary.csv)；resources/保留每阶段实际资源采样|
|调用账本与恢复入口|[accounting.json](accounting.json)、[ledger/](ledger/)、[RUN_AND_RESUME.md](RUN_AND_RESUME.md)；完成后不再启动实验|
|实际源码|[source_code/](source_code/)、[本轮差异](code.diff)；[评分实际依赖](source_code/scorer_dependencies/)、[评分依赖hash](source_code/scorer_dependencies_manifest.json)，包含真实评分实现而非只有包装器|
|源码与旧资产保护|[source_hashes_before.json](registration/source_hashes_before.json)、[source_verification_after.json](registration/source_verification_after.json)、[training_implementation_hashes.json](registration/training_implementation_hashes.json)、[protected_hashes_after.json](independent_audit/protected_hashes_after.json)|
|工程修订|[预训练恢复修复](revisions/pretraining_recovery_fix/)、[独立审计修复](independent_audit/final_audit_metadata_interface_fix.json)、[计时展示修订](presentation/sampling_timing_correction/)、[报告编辑记录](presentation/editorial_review/)|
|科研图与制图数据|[主指标逐父PNG](presentation/figures/primary_parent_pairs.png)、[PDF](presentation/figures/primary_parent_pairs.pdf)、[条件选择PNG](presentation/figures/control_columns_and_curve_errors.png)、[PDF](presentation/figures/control_columns_and_curve_errors.pdf)；[视觉QA](presentation/visual_qa.json)|
|交付校验|[delivery_check.json](delivery_check.json)、[delivery_hashes.json](delivery_hashes.json)；后者覆盖目录内非.pt证据，检查点hash单列于manifest及独立审计|

CSV有同名JSON供保留数值精度与NA。START raw引用历史只读位置；最终独立审计直接核验了该位置的实际文件。图内P1–P8与完整父UID的对应关系保留在 `presentation/figures/` 制图数据。

唯一下一步建议是使用已有raw定位主退化父在noise1的共同局部错误，检验局部缺面/错位能否解释远端误差；本次未自动启动该后续诊断，未追加训练、改loss或切换默认模型。
