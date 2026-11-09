# 2026-10-25 资产退役补记

原实验/资产目录：`E:/MeshFlow/experiments/r2_seam_pool47_couple_v1`。

本文件记录用户本轮授权清理之后的本地可用性，不修改该实验在运行当时的事实、指标、结论或历史完成回执。

实际移除 4 份权重及 0 份已有校验 ZIP 的展开审阅副本，逻辑大小 3.887 GiB。

以下权重已退役，依赖它们的历史完整训练重放、对应旧检查点续训或重新模型采样在本地不再可用。代码或旧报告仍可能列出其原路径；不得自动下载、恢复、重训或将缓存输出冒充重新生成。

若本目录仍有保留终点，保留终点的推理/恢复能力与已退役中间状态分开；清理没有启动任何续训。

原 RAW、POST、固定噪声、查询、逐例表、图审、源代码、原报告、原 `COMPLETE/completion` 和历史 hash 不被本次权重退役覆盖。离线复算仍可使用保存的输出；要求已删权重存在的历史全链校验入口不再作为当前可执行入口。

| 移除路径 | 类型 | Bytes | SHA256 | 原因 |
| --- | --- | ---: | --- | --- |
| `E:/MeshFlow/experiments/r2_seam_pool47_couple_v1/checkpoints/PLAIN3000_step1000.pt` | RETIRED_WEIGHT | 1565057323 | `51a0a60a498c7c1f4f747c0c7c993c9cf2f32cf69558bd90c4ab4383dda8e85c` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/r2_seam_pool47_couple_v1/checkpoints/PLAIN3000_step3000.pt` | RETIRED_WEIGHT | 1565057323 | `82425bbf6517b292c54f1b941f4c083dd019045f6bdb0175b6396f0f752630f6` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/r2_seam_pool47_couple_v1/inference/COUPLE3000_generator.pt` | RETIRED_WEIGHT | 521665851 | `621c55ef91c41985ef68cbde1415f56a2a9729251495a3ba8a39f5dad1ff766f` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/r2_seam_pool47_couple_v1/inference/PLAIN3000_generator.pt` | RETIRED_WEIGHT | 521665851 | `89614faf3877c20fd99eb346f39f4f762302ebe06d565be64f39d5c741ba286d` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |

本目录继续保留的权重：

- [PLAIN3000_step2000.pt](<checkpoints/PLAIN3000_step2000.pt>)：1.458 GiB，SHA256 `ea6675a8d77364936fb13443ea0378011423404ee28a8f3074b7f8bd4ef3f56a`。

完整清理事实见 [清理报告](<../../maintenance/storage_cleanup_20261025/CLEANUP_REPORT.md>) 与 [实际完成回执](<../../maintenance/storage_cleanup_20261025/cleanup_receipt.json>)。

本次不切换默认模型，不训练、不生成，不 commit/push。
