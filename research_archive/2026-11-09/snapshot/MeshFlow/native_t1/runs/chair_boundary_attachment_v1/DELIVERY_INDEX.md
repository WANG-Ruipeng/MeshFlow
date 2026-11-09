# 交付索引

| 内容 | 路径 | 实际存在 |
| --- | --- | --- |
| 主报告 | [CHAIR_BOUNDARY_ATTACHMENT_FACTORIAL_REPORT.md](CHAIR_BOUNDARY_ATTACHMENT_FACTORIAL_REPORT.md) | 是 |
| 完整逐输出 | [per_output_metrics.csv](per_output_metrics.csv) | 是 |
| 逐父均值 | [aggregation/parent_means.csv](aggregation/parent_means.csv) | 是 |
| 面板均值 | [aggregation/panel_means.csv](aggregation/panel_means.csv) | 是 |
| 逐父配对 | [aggregation/paired_parent_differences.csv](aggregation/paired_parent_differences.csv) | 是 |
| 逐输出交互 | [aggregation/interaction_per_output.csv](aggregation/interaction_per_output.csv) | 是 |
| 逐父交互 | [aggregation/interaction_per_parent.csv](aggregation/interaction_per_parent.csv) | 是 |
| 密度敏感性 | [aggregation/sensitivity.csv](aggregation/sensitivity.csv) | 是 |
| 请求 | [requests.json](requests.json) | 是 |
| raw索引 | [raw_index.json](raw_index.json) | 是 |
| 冻结协议 | [protocol.json](protocol.json) | 是 |
| 协议hash | [protocol/protocol_identity.json](protocol/protocol_identity.json) | 是 |
| 权重清单 | [checkpoint_manifest.json](checkpoint_manifest.json) | 是 |
| 一次校准 | [lambda_calibration.json](lambda_calibration.json) | 是 |
| 真实预检 | [precheck/results.json](precheck/results.json) | 是 |
| 两条回放 | [replays/summary.json](replays/summary.json) | 是 |
| 方法CPU测试 | [tests/method/results.json](tests/method/results.json) | 是 |
| 原指标测试 | [metric_tests_final/metric_test_results.json](metric_tests_final/metric_test_results.json) | 是 |
| 新版指标测试 | [metric_tests_v1r1/metric_test_results.json](metric_tests_v1r1/metric_test_results.json) | 是 |
| 训练 | [training/](training/) | 是 |
| 调用 | [ledger/](ledger/) | 是 |
| 资源 | [resources/](resources/) | 是 |
| 新版控制逐例 | [scoring/control_v1r1/](scoring/control_v1r1/) | 是 |
| 新版固定参考 | [scoring/control_references_v1r1/](scoring/control_references_v1r1/) | 是 |
| 旧版保留 | [scoring/control/](scoring/control/) | 是 |
| 工程修订 | [engineering/](engineering/) | 是 |
| 图库 | [gallery/index.html](gallery/index.html) | 是 |
| 恢复说明 | [RUN_AND_RESUME.md](RUN_AND_RESUME.md) | 是 |
| 源码快照 | [code_snapshot/](code_snapshot/) | 是 |
| 源码hash | [source_hashes.json](source_hashes.json) | 否 |
| 代码diff | [code.diff](code.diff) | 否 |

数据和账本是结论依据。缺失目录或文件显示为否，不把链接当成已完成证据。仅一套当前报告；工程原件另留证，不自动制作多重ZIP。

最终独立交付：

- [独立核算收据](independent_audit.json)
- [逐噪声配对差](paired_noise_differences.csv)
- [受保护源文件核验](protected_after_audit.json)
- [实际源码](actual_source/)
- [源文件hash](source_hashes.json)
- [code.diff](code.diff)
- [完整raw身份索引](raw_integrity_index.csv)

- [完整图库与训练诊断最终复核](final_review.json)
- [训练诊断逐项收据](review_training_diagnostics.json)

Final CPU-only audit recovery and recorded engineering correction: [FINAL_RECOVERY_NOTE.md](FINAL_RECOVERY_NOTE.md).
Technical image checks: [technical_visual_review.json](technical_visual_review.json).
