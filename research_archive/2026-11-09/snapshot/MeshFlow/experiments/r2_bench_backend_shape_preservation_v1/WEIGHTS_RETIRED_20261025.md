# 2026-10-25 资产退役补记

原实验/资产目录：`E:/MeshFlow/experiments/r2_bench_backend_shape_preservation_v1`。

本文件记录用户本轮授权清理之后的本地可用性，不修改该实验在运行当时的事实、指标、结论或历史完成回执。

实际移除 4 份权重及 0 份已有校验 ZIP 的展开审阅副本，逻辑大小 20.368 GiB。

以下权重已退役，依赖它们的历史完整训练重放、对应旧检查点续训或重新模型采样在本地不再可用。代码或旧报告仍可能列出其原路径；不得自动下载、恢复、重训或将缓存输出冒充重新生成。

若本目录仍有保留终点，保留终点的推理/恢复能力与已退役中间状态分开；清理没有启动任何续训。

原 RAW、POST、固定噪声、查询、逐例表、图审、源代码、原报告、原 `COMPLETE/completion` 和历史 hash 不被本次权重退役覆盖。离线复算仍可使用保存的输出；要求已删权重存在的历史全链校验入口不再作为当前可执行入口。

| 移除路径 | 类型 | Bytes | SHA256 | 原因 |
| --- | --- | ---: | --- | --- |
| `E:/MeshFlow/experiments/r2_bench_backend_shape_preservation_v1/training/checkpoints/step0100.pt` | RETIRED_WEIGHT | 5467513531 | `cbbbf7c80037a3a51b73b642d254c738a1adc8a7493644caaf5d1dca1629a56a` | Intermediate SHAPE recovery snapshot; final complete step500 and inference export retained. Old step100 boundary replay retires. |
| `E:/MeshFlow/experiments/r2_bench_backend_shape_preservation_v1/training/checkpoints/step0200.pt` | RETIRED_WEIGHT | 5467514747 | `21f50afb9dc8cd208db58f3918466e0c106af501fa8c35fd560d6d3a99933aa2` | Intermediate SHAPE recovery snapshot; final complete step500 and inference export retained. Old step100 boundary replay retires. |
| `E:/MeshFlow/experiments/r2_bench_backend_shape_preservation_v1/training/checkpoints/step0300.pt` | RETIRED_WEIGHT | 5467514747 | `7cc48573e8ed51d75453d436ddee26a8ae10483b829902bd426396d8bd781e61` | Intermediate SHAPE recovery snapshot; final complete step500 and inference export retained. Old step100 boundary replay retires. |
| `E:/MeshFlow/experiments/r2_bench_backend_shape_preservation_v1/training/checkpoints/step0400.pt` | RETIRED_WEIGHT | 5467514747 | `a26f8a3c85f639eaa7e7da515cdea9157d98c3c2d6d9b6d5e24eb012f14a7e77` | Intermediate SHAPE recovery snapshot; final complete step500 and inference export retained. Old step100 boundary replay retires. |

本目录继续保留的权重：

- [candidate_denoiser.pt](<training/candidate_denoiser.pt>)：1.697 GiB，SHA256 `60f0aaf0003907913ed06d7d40f516fa88cf912517d37e8b155056acd298a5cc`。
- [step0500.pt](<training/checkpoints/step0500.pt>)：5.092 GiB，SHA256 `ee34b1605f1a37437556407940c537154c611bc310c20f1b0dd48ce2c9ee94a1`。

完整清理事实见 [清理报告](<../../maintenance/storage_cleanup_20261025/CLEANUP_REPORT.md>) 与 [实际完成回执](<../../maintenance/storage_cleanup_20261025/cleanup_receipt.json>)。

本次不切换默认模型，不训练、不生成，不 commit/push。
