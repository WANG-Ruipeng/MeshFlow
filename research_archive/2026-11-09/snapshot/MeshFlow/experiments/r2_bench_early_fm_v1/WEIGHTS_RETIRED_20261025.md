# 2026-10-25 资产退役补记

原实验/资产目录：`E:/MeshFlow/experiments/r2_bench_early_fm_v1`。

本文件记录用户本轮授权清理之后的本地可用性，不修改该实验在运行当时的事实、指标、结论或历史完成回执。

实际移除 3 份权重及 0 份已有校验 ZIP 的展开审阅副本，逻辑大小 2.429 GiB。

以下权重已退役，依赖它们的历史完整训练重放、对应旧检查点续训或重新模型采样在本地不再可用。代码或旧报告仍可能列出其原路径；不得自动下载、恢复、重训或将缓存输出冒充重新生成。

若本目录仍有保留终点，保留终点的推理/恢复能力与已退役中间状态分开；清理没有启动任何续训。

原 RAW、POST、固定噪声、查询、逐例表、图审、源代码、原报告、原 `COMPLETE/completion` 和历史 hash 不被本次权重退役覆盖。离线复算仍可使用保存的输出；要求已删权重存在的历史全链校验入口不再作为当前可执行入口。

| 移除路径 | 类型 | Bytes | SHA256 | 原因 |
| --- | --- | ---: | --- | --- |
| `E:/MeshFlow/experiments/r2_bench_early_fm_v1/training/EARLY_FM/checkpoint_step0.pt` | RETIRED_WEIGHT | 521690679 | `529ed21ab0d4caa2f809f077b205c078b3ab4e084f2ad1caad6396e36e281c2b` | No new adoption identities or DEV upgrade; closed time-weighting candidate. |
| `E:/MeshFlow/experiments/r2_bench_early_fm_v1/training/EARLY_FM/checkpoint_step1500.pt` | RETIRED_WEIGHT | 1565056363 | `f838cc8c341991f5a11c696b8d77198c712177c7330a073fd92e4711a9ee54aa` | No new adoption identities or DEV upgrade; closed time-weighting candidate. |
| `E:/MeshFlow/experiments/r2_bench_early_fm_v1/training/EARLY_FM/generator_step1500.pt` | RETIRED_WEIGHT | 521665851 | `69d698a143455852a1aadd17d42c72f6dae1c95ba76d573ead50280479f1c0a5` | No new adoption identities or DEV upgrade; closed time-weighting candidate. |

完整清理事实见 [清理报告](<../../maintenance/storage_cleanup_20261025/CLEANUP_REPORT.md>) 与 [实际完成回执](<../../maintenance/storage_cleanup_20261025/cleanup_receipt.json>)。

本次不切换默认模型，不训练、不生成，不 commit/push。
