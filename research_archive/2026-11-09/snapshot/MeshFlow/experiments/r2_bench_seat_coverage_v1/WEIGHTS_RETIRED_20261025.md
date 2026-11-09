# 2026-10-25 资产退役补记

原实验/资产目录：`E:/MeshFlow/experiments/r2_bench_seat_coverage_v1`。

本文件记录用户本轮授权清理之后的本地可用性，不修改该实验在运行当时的事实、指标、结论或历史完成回执。

实际移除 2 份权重及 0 份已有校验 ZIP 的展开审阅副本，逻辑大小 2.915 GiB。

以下权重已退役，依赖它们的历史完整训练重放、对应旧检查点续训或重新模型采样在本地不再可用。代码或旧报告仍可能列出其原路径；不得自动下载、恢复、重训或将缓存输出冒充重新生成。

若本目录仍有保留终点，保留终点的推理/恢复能力与已退役中间状态分开；清理没有启动任何续训。

原 RAW、POST、固定噪声、查询、逐例表、图审、源代码、原报告、原 `COMPLETE/completion` 和历史 hash 不被本次权重退役覆盖。离线复算仍可使用保存的输出；要求已删权重存在的历史全链校验入口不再作为当前可执行入口。

| 移除路径 | 类型 | Bytes | SHA256 | 原因 |
| --- | --- | ---: | --- | --- |
| `E:/MeshFlow/experiments/r2_bench_seat_coverage_v1/training/NATIVE/checkpoint_latest.pt` | RETIRED_WEIGHT | 1565060835 | `f2aaae02649ca4fe3e0b552af7dadf18fb14c8b4c4318a89dc19790a8d5f8e24` | Initial/rolling Coverage snapshot; final step1500 checkpoint plus export and original Native6000 source retained. |
| `E:/MeshFlow/experiments/r2_bench_seat_coverage_v1/training/NATIVE/checkpoint_step0.pt` | RETIRED_WEIGHT | 1565060771 | `cdcd935d2cbb467857cbeb35f91da305f744a27deace738e285b09c16bde97da` | Initial/rolling Coverage snapshot; final step1500 checkpoint plus export and original Native6000 source retained. |

本目录继续保留的权重：

- [checkpoint_step1500.pt](<training/NATIVE/checkpoint_step1500.pt>)：1.458 GiB，SHA256 `68703028dc20d84f210daa5b8c6eb4468299bf5aa2a5bace8881e8bd500fcf10`。
- [generator_step1500.pt](<training/NATIVE/generator_step1500.pt>)：0.486 GiB，SHA256 `30a50ab69ef52c8b4549b5c33671252ab6c1816a32578e6611f9c7b258e4acc3`。

完整清理事实见 [清理报告](<../../maintenance/storage_cleanup_20261025/CLEANUP_REPORT.md>) 与 [实际完成回执](<../../maintenance/storage_cleanup_20261025/cleanup_receipt.json>)。

本次不切换默认模型，不训练、不生成，不 commit/push。
