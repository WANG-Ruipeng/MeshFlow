# 本轮必须更正的旧指标解释

原 `CHAIR_DOMAIN_CONTROL_REPORT.md` 中的 **correct-column advantage** 实际由固定输出 alpha 的一行计算。此名称及其“固定目标比较正确/错误 C”的解释不成立。本次保留旧代码、报告和结果，不静默覆盖已分发文件。

对矩阵 `E[alpha,beta] = distance(Y_beta, F_alpha)`：

- 旧字段固定 alpha，改变 beta：同一个生成输出更接近哪个目标。
- 正确条件对照应固定 beta，改变 alpha：面对同一个目标，正确 C 所生成的输出是否优于错误 C 的输出。

原始 648 个矩阵单元及对角值仍有效。新统计只是从已保存数字取列，无需训练、生成或重新评分。全部 72 个矩阵、216 项每目标比较均已纳入。

受影响的旧字段为 `wrong_mean_minus_diagonal_RMS_L0`、`diagonal_no_worse_than_both_wrong`、`diagonal_strictly_better_than_both_wrong` 及其派生汇总和增量。精确文件列表见 `matrix_audit/affected_old_artifacts.csv`。原始宽度、响应斜率、几何评分、teacher 数值不受这一方向错误影响。

修正后，117/216 项优势符号和对角胜负改变。旧验证组的 100%/0%/0% 胜率不能继续当成正确条件的收益。固定目标后的终点严格胜率（beta=.9/1/1.1）是：

| 模型 | .9 | 1 | 1.1 |
|---|---:|---:|---:|
| Native | 37.5% | 37.5% | 50% |
| Native+Geo | 37.5% | 25% | 50% |
| Native+Geo+Joint | 25% | 50% | 50% |

每格 4 个父对象、每父 2 份噪声。修正改变了逐目标解释，但未证明可靠控制：验证 beta=1.1 的三臂平均列优势仍为负，Geo 的全 beta 严格胜率也没有优于 Native。

三个 alpha/beta 全平均后的 margin 恰好保持不变，是完整矩阵的代数恒等式，不是两种问题等价。详见 `matrix_audit/MATRIX_DIRECTION_AUDIT.md`；使用 `fixed_target_beta_per_case.csv` 和 `axis_summary_parent_equal.csv` 中 `direction=fixed_target_beta` 的结果替代旧“正确条件”解释。新表具体字段取值以其表头为准。
