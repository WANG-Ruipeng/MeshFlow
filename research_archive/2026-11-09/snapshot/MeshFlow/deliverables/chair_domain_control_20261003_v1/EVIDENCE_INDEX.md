# 证据索引

以下本轮文件均位于 `native_t1/runs/chair_domain_control_v1/`。阅读统计时先看摘要与逐父表，再看逐例、raw 与 score；不要将重复展示的值当作独立样本。

| 主题 | 文件或目录 |
|---|---|
| A：新上下文补全 | `A_context_summary.csv`、`A_context_per_parent.csv`、`A_context_per_case.csv`、`A_context_area_strata.csv` |
| B：配对宽度控制 | `B_width_summary.csv`、`B_width_per_parent.csv`、`B_width_response.csv`、`B_cross_condition_scores.csv` |
| C：原留出回归 | `C_holdout_regression.csv`、`C_holdout_paired_summary.csv`、`C_holdout_paired_per_parent.csv` |
| Geo 与 Joint 匹配预算增量 | `matched_budget_summary.csv`、`matched_budget_per_parent.csv`、`matched_budget_per_case.csv`；宽度另见 `B_width_matched_budget_summary.csv` |
| 不响应 / Post-scale | `B_analytic_baselines.csv`、`B_analytic_summary.csv`、`B_model_minus_analytic_summary.csv` 及对应 per_parent/per_case |
| 起终点共有输入变化 | `shared_start_endpoint_summary.csv`、`shared_start_endpoint_per_parent.csv`、`shared_start_endpoint_per_case.csv` |
| 宽度起终点配对 | `B_width_start_endpoint_summary.csv`、`B_width_start_endpoint_pairs.csv` |
| 训练、生成与 teacher 曲线 | `training_learning_curves.csv`、`shared_input_generation_curves.csv`、`teacher_summary.csv`、`teacher_per_case.csv` |
| 固定视图与曲线图 | `figures/manifest.json`；A/C 优先看 `figures/layout_v2/manifest.json` 指向的排版副本，B 和曲线看原图 |
| 冻结协议 / 任务 / 输入流 | `registration/requested_protocol.md`、`protocol.json`、`panels.json`、`task_manifest.json`、`train_plan.json`、`inputs/` |
| C 可用性与失败分类 | `C_availability_overlap.csv`、`failure_classification.csv`、`registration/` 内准备/验证记录 |
| 原始模型输出与评分 | `evaluation/`，解析与跨条件证据在 `analysis/` |
| 训练批次与中间保存 | `training/`、`checkpoints/*.json`、`checkpoint_manifest.json` |
| 真实执行与资源 | `execution_accounting.json`、`execution_ledger.jsonl`、`execution_attempts.csv`、`resource_summary.csv`、`run_state.json`、`logs/` |
| I/O 中断与零重放恢复 | `incidents/001_atomic_replace_permission/` |
| 完成后的幂等重入 | `registration/completed_reentry_check.json`、`logs/completed_reentry_check.log` |
| 代码差异与身份 | `code.diff`、`registration/code_hashes.json`、`protected_assets_after.json` |
| 显示排版修订 | `delivery_only_code.diff`、`delivery_revision_001/`；与冻结科学实现分开 |
| 原机恢复命令 | `restart.txt`；无需为阅读包执行 |

原数据和源 step1000 证据位于 `native_t1/runs/control_mvp48_v1/`。先看 `DATA_READY.json`、`TRAIN_READY.json`，再看 `data/manifest.json`、`data/splits.json`（如清单引用）、`registration/` 与 checkpoint JSON。48 个修复前原始 NPZ 保存在其原仓库路径 `experiments/mf_h1_local0/assets/` 下；原始与修复后的 SHA256 沿用既有清单。

历史5000仅作不同训练预算的背景，位于 `native_t1/runs/control_mvp48_budget_single_v1/`。本包仅附本轮 `C_historical_background.csv` 精确引用的 raw/score、三个 holdout ledger 和必要身份/报告，不包含旧实验所有曲线和拟合探针。源1000与历史5000的 192 条复用明细见本轮 `C_historical_background.csv`；身份检查与计数汇总见 `history_reuse.json`。

代码位于 `experiments/mf_h1_local0/chair_domain_control_v1/`，并附原 MVP48、旧适配器、已登记模型/采样/评分依赖。动态评分依赖 `edge_factorial_20261001_123000/source_snapshot/.../edge_metric.py` 也已包含。第三方软件和模型权重未打包。

精确文件范围以顶层 `PACKAGE_MANIFEST.json` 为准；冻结哈希检查以 `PACKAGE_CHECKS.json` 为准。`OMITTED_WEIGHTS.json` 说明未附的 checkpoint 文件及其登记身份，`EXCLUSIONS.json` 说明权重、运行锁和缓存等排除规则。

终点模型以 checkpoint_manifest.json 中 phase3000_total4000 的固定终点为准；latest 文件是途中恢复快照，名称不表示最终步数。
