# Bench 跨类别试验交付索引

先读 [REPORT.md](REPORT.md)，然后看 [全部24组配对表面与截面](gallery/paired/index.html)。主结论是局部 C/F 连接可迁移，但没有接入 bulk，且整体质量有代价；不晋级维护默认模型。

| 内容 | 文件/目录 |
|---|---|
| 完整训练/推理合同 | [TRAINING_CONTRACT.md](TRAINING_CONTRACT.md) |
| 数据72父及输入审阅 | [数据卡](data/DATASET_CARD.md)、`data/manifest.json`、`data/input_visual_review.csv` |
| 两臂完整终点 | `training/PLAIN/step3000.pt`、`training/SEAM/step3000.pt` |
| 全部原始更新/曝光 | `training/`、[训练审计](training_audit/FINAL_TRAINING_AUDIT.json)、`training_audit/paired_exposure_full.csv` |
| 学习曲线 | [learning_curves.png](learning_curves/learning_curves.png)、`learning_curves/per_update_full.csv` |
| 96逐输出×124字段 | [per_output.csv](evaluation/per_output.csv) |
| 完整逐父/逐噪声配对 | `evaluation/paired_per_noise.csv`、`evaluation/paired_complete/` |
| 全部合法图/非法面对 | `evaluation/scores/`、`evaluation/illegal_pairs_by_role.csv` |
| 实际AI_PROXY图审 | [REVIEW_SUMMARY.md](visual_review/REVIEW_SUMMARY.md)、`visual_review/per_output.csv`、原`part_a/b.json` |
| 独立数值/删面复核 | [NUMERICAL_REVIEW.md](numerical_review/NUMERICAL_REVIEW.md) |
| 最终身份核验 | [final_integrity.json](final_integrity.json) |
| 实际评分源码 | `scoring_source_snapshot/` |
| 实验源码归档 | `source_snapshot/`、`delivery/code.diff`、`delivery/source_hashes.json` |
| 资源/调用与最终收据 | `dispatch_*.json`、`resource_monitor.jsonl`、[FINAL_RECEIPT.json](FINAL_RECEIPT.json) |
| 旧权重清理 | [cleanup_receipt.json](cleanup/cleanup_receipt.json)、`cleanup/final_rolling_retirement/` |
| 只读状态/恢复说明 | [RUN_AND_RESUME.md](RUN_AND_RESUME.md) |

`BENCH_CONTROL_TRANSFER_REVIEW.zip` 是小审阅包：报告、逐例/逐父CSV、实际图审、24全组PNG、代码与哈希；不含大权重、RAW/POST网格数组或全量轨迹。完整本地证据均保留。ZIP 内清单及排除范围见 `delivery/review_package.json`。

没有 commit/push，没有切换 Chair 默认。Table/Lamp 本轮没有训练。
