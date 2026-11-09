# 2026-10-25 资产退役补记

原实验/资产目录：`E:/MeshFlow/experiments/r2_seam_quality_wave1_wave2_v1`。

本文件记录用户本轮授权清理之后的本地可用性，不修改该实验在运行当时的事实、指标、结论或历史完成回执。

实际移除 6 份权重及 0 份已有校验 ZIP 的展开审阅副本，逻辑大小 4.225 GiB。

以下权重已退役，依赖它们的历史完整训练重放、对应旧检查点续训或重新模型采样在本地不再可用。代码或旧报告仍可能列出其原路径；不得自动下载、恢复、重训或将缓存输出冒充重新生成。

若本目录仍有保留终点，保留终点的推理/恢复能力与已退役中间状态分开；清理没有启动任何续训。

原 RAW、POST、固定噪声、查询、逐例表、图审、源代码、原报告、原 `COMPLETE/completion` 和历史 hash 不被本次权重退役覆盖。离线复算仍可使用保存的输出；要求已删权重存在的历史全链校验入口不再作为当前可执行入口。

| 移除路径 | 类型 | Bytes | SHA256 | 原因 |
| --- | --- | ---: | --- | --- |
| `E:/MeshFlow/experiments/r2_seam_quality_wave1_wave2_v1/checkpoints/CLEAN_BULK_FM_step1000.pt` | RETIRED_WEIGHT | 1565079083 | `3d07de861ffcf64afb9c53c47eaa815cf040e9abad7673eda597ae8ed4ad2bb2` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/r2_seam_quality_wave1_wave2_v1/checkpoints/SURGICAL_LATE4_step1000.pt` | RETIRED_WEIGHT | 884661387 | `9d8f862b7883cd98068f9721709c6f9ff0203f01e5c7a1fbcf4fb28e7c8d08ed` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/r2_seam_quality_wave1_wave2_v1/inference/CLEAN_BULK_FM_generator.pt` | RETIRED_WEIGHT | 521665851 | `14ae54a0020885ef1641448a4fa2c2df5076b8d5c13a3ffd5104ff737126e3be` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/r2_seam_quality_wave1_wave2_v1/inference/PLAIN_S1000_generator.pt` | RETIRED_WEIGHT | 521665979 | `cf64464d2ced56d56dc50fe587607dcf4fca4d35671148323183092fd1f85334` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/r2_seam_quality_wave1_wave2_v1/inference/SURGICAL_LATE4_generator.pt` | RETIRED_WEIGHT | 521665851 | `638fab93e02276ba843f7038f7d9d054fb18edd8426a6c5a0b2eea9abb8a1d4b` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/r2_seam_quality_wave1_wave2_v1/inference/WISE_B_PLAIN_050_generator.pt` | RETIRED_WEIGHT | 521666171 | `0c69e5b1ea20466b0cd67e21e62d0a061774f8053b970a2f476fbbee95b4ce7d` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |

本目录继续保留的权重：

- [PLAIN_S2000_generator.pt](<inference/PLAIN_S2000_generator.pt>)：0.486 GiB，SHA256 `46be9a29d6d9e150205cc46cf8c952ae1cd32b2aa58bff24cb879eb451932f43`。

完整清理事实见 [清理报告](<../../maintenance/storage_cleanup_20261025/CLEANUP_REPORT.md>) 与 [实际完成回执](<../../maintenance/storage_cleanup_20261025/cleanup_receipt.json>)。

本次不切换默认模型，不训练、不生成，不 commit/push。
