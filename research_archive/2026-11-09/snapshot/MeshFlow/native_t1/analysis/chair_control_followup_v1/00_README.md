# 给云端的分析补充包

请先读 `FOLLOWUP_ANALYSIS_REPORT.md` 和 `ERRATA_MATRIX_DIRECTION.md`。这是根据上一轮云端反馈，利用现有实验结果实际完成的 CPU 分析；新增训练、生成、模型前向、OT、几何评分均为零。

最重要的两项更新：

1. 旧“correct-column advantage”实际取固定输出的行，方向解释有误；已从原 648 个矩阵单元独立重算固定目标的列对照，原结果保留。
2. 终点验证三臂各 8 条响应曲线的基准自由宽度都小于 GT 的 90%。基准偏差、相对响应与几何代价已经逐例拆开，不再只用均值斜率评价控制。

`source/` 中补上此前精简包没有的 `B_width_response.csv`、`B_cross_condition_scores.csv` 与 `teacher_per_case.csv`；`source_code/` 是对应的原宽度/矩阵统计和报告代码，供审阅，**不要执行原实验入口**。新列对照请读 `matrix_audit/fixed_target_beta_per_case.csv`；`axis_summary_parent_equal.csv` 应筛选 `direction=fixed_target_beta`。

其余目录提供选定的逐例/逐父新结果、精确 teacher 变化、噪声图与四份详细审计说明。`EXECUTION_LEDGER.json` 和 `RUN_COMPLETE.json` 说明实际执行范围；`FILES_SHA256.json` 校验本包内容。原 raw/GT、权重、完整日志和分析过程的全部中间表未装入这个精简包；详细报告中提到的其他明细在原工作站保留，缺少文件不等于未执行。

请核对而非预设模型赢家。区分行/列问题、基准尺寸/相对响应、逐父/逐噪声、均值/尾部，以及共变/因果。阶段 PASS 表示核查完成，不表示模型质量合格。对于未随包提供的 raw，只能核对本次给出的审计记录，不能声称已在云端独立重算。不要根据这些表自行启动训练、修改 loss、门控或求解器。
