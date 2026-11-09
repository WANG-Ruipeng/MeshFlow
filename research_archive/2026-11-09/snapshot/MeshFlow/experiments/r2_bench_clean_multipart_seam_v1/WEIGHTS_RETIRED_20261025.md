# 2026-10-25 资产退役补记

原实验/资产目录：`E:/MeshFlow/experiments/r2_bench_clean_multipart_seam_v1`。

本文件记录用户本轮授权清理之后的本地可用性，不修改该实验在运行当时的事实、指标、结论或历史完成回执。

实际移除 2 份权重及 0 份已有校验 ZIP 的展开审阅副本，逻辑大小 1.944 GiB。

以下权重已退役，依赖它们的历史完整训练重放、对应旧检查点续训或重新模型采样在本地不再可用。代码或旧报告仍可能列出其原路径；不得自动下载、恢复、重训或将缓存输出冒充重新生成。

若本目录仍有保留终点，保留终点的推理/恢复能力与已退役中间状态分开；清理没有启动任何续训。

原 RAW、POST、固定噪声、查询、逐例表、图审、源代码、原报告、原 `COMPLETE/completion` 和历史 hash 不被本次权重退役覆盖。离线复算仍可使用保存的输出；要求已删权重存在的历史全链校验入口不再作为当前可执行入口。

| 移除路径 | 类型 | Bytes | SHA256 | 原因 |
| --- | --- | ---: | --- | --- |
| `E:/MeshFlow/experiments/r2_bench_clean_multipart_seam_v1/inference/BENCH_CLEAN_MULTIPART_S1500_generator.pt` | RETIRED_WEIGHT | 521665915 | `474d06f8225fbffeab099807a1d32557b85a724d2481171d9e75a1e7d3a97011` | Historical multipart candidate no DEV body join; superseded research branch. |
| `E:/MeshFlow/experiments/r2_bench_clean_multipart_seam_v1/training/checkpoint_step1500.pt` | RETIRED_WEIGHT | 1565305991 | `0a96b123cc8e5cd4e7c1fc64588a1baaa8de93d83debe4f941b3c5f866a2db42` | Historical multipart candidate no DEV body join; superseded research branch. |

完整清理事实见 [清理报告](<../../maintenance/storage_cleanup_20261025/CLEANUP_REPORT.md>) 与 [实际完成回执](<../../maintenance/storage_cleanup_20261025/cleanup_receipt.json>)。

本次不切换默认模型，不训练、不生成，不 commit/push。
