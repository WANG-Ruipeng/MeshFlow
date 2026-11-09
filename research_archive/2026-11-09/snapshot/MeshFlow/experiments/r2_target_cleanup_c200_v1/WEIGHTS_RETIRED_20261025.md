# 2026-10-25 资产退役补记

原实验/资产目录：`E:/MeshFlow/experiments/r2_target_cleanup_c200_v1`。

本文件记录用户本轮授权清理之后的本地可用性，不修改该实验在运行当时的事实、指标、结论或历史完成回执。

实际移除 4 份权重及 0 份已有校验 ZIP 的展开审阅副本，逻辑大小 3.887 GiB。

以下权重已退役，依赖它们的历史完整训练重放、对应旧检查点续训或重新模型采样在本地不再可用。代码或旧报告仍可能列出其原路径；不得自动下载、恢复、重训或将缓存输出冒充重新生成。

若本目录仍有保留终点，保留终点的推理/恢复能力与已退役中间状态分开；清理没有启动任何续训。

原 RAW、POST、固定噪声、查询、逐例表、图审、源代码、原报告、原 `COMPLETE/completion` 和历史 hash 不被本次权重退役覆盖。离线复算仍可使用保存的输出；要求已删权重存在的历史全链校验入口不再作为当前可执行入口。

| 移除路径 | 类型 | Bytes | SHA256 | 原因 |
| --- | --- | ---: | --- | --- |
| `E:/MeshFlow/experiments/r2_target_cleanup_c200_v1/checkpoints/DEDUP_C200_step200.pt` | RETIRED_WEIGHT | 1565217707 | `08a5598d9c6e08f872ced689984487bd2f6dd8896bf5b8a3529f37bba2afa0c4` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/r2_target_cleanup_c200_v1/checkpoints/ORIGINAL_C200_step200.pt` | RETIRED_WEIGHT | 1565217771 | `a8022fcb8a5de4dc26cd5efc2f798cdba197bfc1a3b27cc993f51afd09bf0819` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/r2_target_cleanup_c200_v1/inference/DEDUP_C200_generator.pt` | RETIRED_WEIGHT | 521665787 | `2d87a3f5669a8c658ced0459daaaafdede93563fde8b56fccbd1386c6a356d72` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/r2_target_cleanup_c200_v1/inference/ORIGINAL_C200_generator.pt` | RETIRED_WEIGHT | 521665851 | `86b51788a4212b47931fd81d76e423fc343c24c6d1465185e0d23b5db827b245` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |

完整清理事实见 [清理报告](<../../maintenance/storage_cleanup_20261025/CLEANUP_REPORT.md>) 与 [实际完成回执](<../../maintenance/storage_cleanup_20261025/cleanup_receipt.json>)。

本次不切换默认模型，不训练、不生成，不 commit/push。
