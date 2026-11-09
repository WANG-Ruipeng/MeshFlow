# 2026-10-25 资产退役补记

原实验/资产目录：`E:/MeshFlow/experiments/r2_seam_quality_four_v1`。

本文件记录用户本轮授权清理之后的本地可用性，不修改该实验在运行当时的事实、指标、结论或历史完成回执。

实际移除 6 份权重及 0 份已有校验 ZIP 的展开审阅副本，逻辑大小 2.915 GiB。

以下权重已退役，依赖它们的历史完整训练重放、对应旧检查点续训或重新模型采样在本地不再可用。代码或旧报告仍可能列出其原路径；不得自动下载、恢复、重训或将缓存输出冒充重新生成。

若本目录仍有保留终点，保留终点的推理/恢复能力与已退役中间状态分开；清理没有启动任何续训。

原 RAW、POST、固定噪声、查询、逐例表、图审、源代码、原报告、原 `COMPLETE/completion` 和历史 hash 不被本次权重退役覆盖。离线复算仍可使用保存的输出；要求已删权重存在的历史全链校验入口不再作为当前可执行入口。

| 移除路径 | 类型 | Bytes | SHA256 | 原因 |
| --- | --- | ---: | --- | --- |
| `E:/MeshFlow/experiments/r2_seam_quality_four_v1/checkpoints/LONG_step1000.pt` | RETIRED_WEIGHT | 1565072395 | `d72772cb24eb5aed7f9a0960963b0fb742f749547dbc090cb3b47aaeb5d210d2` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/r2_seam_quality_four_v1/inference/ANTI_COVER_generator.pt` | RETIRED_WEIGHT | 521665851 | `0f16d012a9594a03c3cc5d2698b6fe9bf9104de4f29923c50682e72f577b05a7` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/r2_seam_quality_four_v1/inference/ANTI_COVER_occupancy.pt` | RETIRED_WEIGHT | 6045 | `f81d0ce926e82cf3f0504a4d8e32e32ff06fae0cd1ec0b653545297207f4575a` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/r2_seam_quality_four_v1/inference/LOCAL_NOISE_generator.pt` | RETIRED_WEIGHT | 521665851 | `034a0e54fbf523b06a238745469c817e4e875b645e023912da87e86a4b0e9af8` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/r2_seam_quality_four_v1/inference/LOCAL_NOISE_occupancy.pt` | RETIRED_WEIGHT | 6045 | `72b69fb029a96a222bd98aeeac1c89dace1bc4a0aaa8e3a450055fe16f394b06` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/r2_seam_quality_four_v1/inference/LONG_generator.pt` | RETIRED_WEIGHT | 521665851 | `8308ff24dc51f341a0db5dcfa505cbc77db2f2a9f2ffd7bf85888a6515ddc025` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |

本目录继续保留的权重：

- [LONG_occupancy.pt](<inference/LONG_occupancy.pt>)：0.000 GiB，SHA256 `6e8f9ab606403c9c2b96d7f6bf6a2c67dacd11d4c45179369414e420a75982df`。

完整清理事实见 [清理报告](<../../maintenance/storage_cleanup_20261025/CLEANUP_REPORT.md>) 与 [实际完成回执](<../../maintenance/storage_cleanup_20261025/cleanup_receipt.json>)。

本次不切换默认模型，不训练、不生成，不 commit/push。
