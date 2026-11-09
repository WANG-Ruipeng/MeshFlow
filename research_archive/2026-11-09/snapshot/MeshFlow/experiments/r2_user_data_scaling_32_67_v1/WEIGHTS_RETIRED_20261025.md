# 2026-10-25 资产退役补记

原实验/资产目录：`E:/MeshFlow/experiments/r2_user_data_scaling_32_67_v1`。

本文件记录用户本轮授权清理之后的本地可用性，不修改该实验在运行当时的事实、指标、结论或历史完成回执。

实际移除 3 份权重及 0 份已有校验 ZIP 的展开审阅副本，逻辑大小 2.429 GiB。

以下权重已退役，依赖它们的历史完整训练重放、对应旧检查点续训或重新模型采样在本地不再可用。代码或旧报告仍可能列出其原路径；不得自动下载、恢复、重训或将缓存输出冒充重新生成。

若本目录仍有保留终点，保留终点的推理/恢复能力与已退役中间状态分开；清理没有启动任何续训。

原 RAW、POST、固定噪声、查询、逐例表、图审、源代码、原报告、原 `COMPLETE/completion` 和历史 hash 不被本次权重退役覆盖。离线复算仍可使用保存的输出；要求已删权重存在的历史全链校验入口不再作为当前可执行入口。

| 移除路径 | 类型 | Bytes | SHA256 | 原因 |
| --- | --- | ---: | --- | --- |
| `E:/MeshFlow/experiments/r2_user_data_scaling_32_67_v1/checkpoints/USER_EXPANDED_67_step6000.pt` | RETIRED_WEIGHT | 1565063467 | `cfb18cfa0dd82dbe55c1e7dfb7c78e862f01647d8f044bd9002a46a8d3d1ee27` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/r2_user_data_scaling_32_67_v1/inference/USER_EXPANDED_67_generator_step6000.pt` | RETIRED_WEIGHT | 521672251 | `2e2a40bcacda7e5ee04a5754018f71986ef5d378345058075ba412bec808f925` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/r2_user_data_scaling_32_67_v1/inference/USER_SMALL_generator_step6000.pt` | RETIRED_WEIGHT | 521672251 | `b99e530200ad11de707f73b434a811e5dcba1304e8dd8aad50df5bd7a817efbf` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |

完整清理事实见 [清理报告](<../../maintenance/storage_cleanup_20261025/CLEANUP_REPORT.md>) 与 [实际完成回执](<../../maintenance/storage_cleanup_20261025/cleanup_receipt.json>)。

本次不切换默认模型，不训练、不生成，不 commit/push。
