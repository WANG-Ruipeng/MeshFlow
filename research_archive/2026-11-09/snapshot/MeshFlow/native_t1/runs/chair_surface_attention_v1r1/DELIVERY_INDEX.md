# 修订交付索引

- `CHAIR_FINITE_SURFACE_ATTENTION_R1_REPORT.md`：修复后真实三臂检查、独立参考、成本和数值限制。
- `protocol.json`、`protocol.sha256`：用户授权后的独立修订协议；原预算累计96前向/24反向不变。
- `registration/budget_audit.json`：原10/1已耗调用与本次76/20分配；MOMENT每任务单次成本的限制。
- `precheck/result.json`、`cost_rows.csv`、`real_comparisons.csv`：76次真实前向、20次反向的实际测量。
- `precheck/H_CONT/`、`MOMENT_XY/`、`SURFACE_XY/`：逐输出NPZ、原/新参数优化器映射、遥测与状态恢复证据。
- `results/cost_per_fixture_arm.csv`：每任务/每臂完整前向、F+B、关系及四层修正耗时。
- `reference/REFERENCE_REPORT.md`、`reference_result.json`、`reference_comparisons.csv`：固定8例独立积分参考与192/12288/49152点差异。
- `reference/`下协议、源码hash、synthetic结果、积分及旧规则CPU账本：34次积分、1472625核点、24次CPU关系调用，无重试。
- `tests/monitor/`：重复日志字段修复的CPU回归和真实只读显卡信息测试。
- `independent_audit/permutation_padding_attribution.csv`：H已有置换误差和两关系臂新增padding差异的逐例分解。
- `registration/real_precheck_independent_audit.json`：调用、输出、恢复、资源和旧资产只读独立审核。
- `accounting.json`、`ledger/`、`controller/`、`stages/`、`resources/`：本次与历史累计计数、耗时和资源。
- `source_code/`、`code.diff`：实际修订源码、差异与SHA256；生产关系/attention仍引用未改的旧实现。
- `RUN_AND_RESUME.md`：可恢复总入口，已完成GPU阶段不会重跑。
- `figures/repair_evidence.png`：实测成本和独立参考差异。

旧v1全部结果、138个START raw引用与既有评分源码保留只读，没有复制大权重或制作额外ZIP。
