# 2026-10-25 资产退役补记

原实验/资产目录：`E:/MeshFlow/experiments/r2_seam_abc_v1`。

本文件记录用户本轮授权清理之后的本地可用性，不修改该实验在运行当时的事实、指标、结论或历史完成回执。

实际移除 8 份权重及 0 份已有校验 ZIP 的展开审阅副本，逻辑大小 2.915 GiB。

以下权重已退役，依赖它们的历史完整训练重放、对应旧检查点续训或重新模型采样在本地不再可用。代码或旧报告仍可能列出其原路径；不得自动下载、恢复、重训或将缓存输出冒充重新生成。

若本目录仍有保留终点，保留终点的推理/恢复能力与已退役中间状态分开；清理没有启动任何续训。

原 RAW、POST、固定噪声、查询、逐例表、图审、源代码、原报告、原 `COMPLETE/completion` 和历史 hash 不被本次权重退役覆盖。离线复算仍可使用保存的输出；要求已删权重存在的历史全链校验入口不再作为当前可执行入口。

| 移除路径 | 类型 | Bytes | SHA256 | 原因 |
| --- | --- | ---: | --- | --- |
| `E:/MeshFlow/experiments/r2_seam_abc_v1/checkpoints/B_POOL47_step1000.pt` | RETIRED_WEIGHT | 1565072395 | `c6411c6a686624e7af1dba445129f3fdbe2c6918d96a6115417548cb48e8e3ba` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/r2_seam_abc_v1/inference/A_COUPLE_generator.pt` | RETIRED_WEIGHT | 521665851 | `e10bd7439be86af034ce57f7100843d0141a5f54349a33786283f54a49cd09a2` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/r2_seam_abc_v1/inference/A_COUPLE_occupancy.pt` | RETIRED_WEIGHT | 5981 | `f460dd5bc96e076c35fbbdde9b7cc142d8c636d428f88fd940889dd035e95125` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/r2_seam_abc_v1/inference/B_POOL47_generator.pt` | RETIRED_WEIGHT | 521665851 | `d625c7ed9d407d936943a6cccde7fd7f230d2c7ef3b68a48c3cfa43a4a067121` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/r2_seam_abc_v1/inference/B_POOL47_occupancy.pt` | RETIRED_WEIGHT | 5981 | `c315baff306cfd222d5dbc1970a5e6f3b25c6e2ef379dee878b6b5f737da7440` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/r2_seam_abc_v1/inference/C_SHARED_TIP_generator.pt` | RETIRED_WEIGHT | 521665851 | `b717e8de6ee7054b578f75ef8819e1fed8e1b8c85e1d46a7de93034191cc6470` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/r2_seam_abc_v1/inference/C_SHARED_TIP_occupancy.pt` | RETIRED_WEIGHT | 6045 | `db280842f06507dc31305599011b797d27981056908a45cbf12123ed7874c4bc` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/r2_seam_abc_v1/inference/C_SHARED_TIP_pointer.pt` | RETIRED_WEIGHT | 19405 | `5b70d495e44ea23de5a3076a0931316b07cad66e98de2fc463cc24946eab3292` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |

完整清理事实见 [清理报告](<../../maintenance/storage_cleanup_20261025/CLEANUP_REPORT.md>) 与 [实际完成回执](<../../maintenance/storage_cleanup_20261025/cleanup_receipt.json>)。

本次不切换默认模型，不训练、不生成，不 commit/push。
