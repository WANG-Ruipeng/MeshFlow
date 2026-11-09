# 2026-10-25 资产退役补记

原实验/资产目录：`E:/MeshFlow/experiments/r2_bench_relation_training_v1`。

本文件记录用户本轮授权清理之后的本地可用性，不修改该实验在运行当时的事实、指标、结论或历史完成回执。

实际移除 5 份权重及 0 份已有校验 ZIP 的展开审阅副本，逻辑大小 5.344 GiB。

以下权重已退役，依赖它们的历史完整训练重放、对应旧检查点续训或重新模型采样在本地不再可用。代码或旧报告仍可能列出其原路径；不得自动下载、恢复、重训或将缓存输出冒充重新生成。

若本目录仍有保留终点，保留终点的推理/恢复能力与已退役中间状态分开；清理没有启动任何续训。

原 RAW、POST、固定噪声、查询、逐例表、图审、源代码、原报告、原 `COMPLETE/completion` 和历史 hash 不被本次权重退役覆盖。离线复算仍可使用保存的输出；要求已删权重存在的历史全链校验入口不再作为当前可执行入口。

| 移除路径 | 类型 | Bytes | SHA256 | 原因 |
| --- | --- | ---: | --- | --- |
| `E:/MeshFlow/experiments/r2_bench_relation_training_v1/training/RELATION_V1_SEAM/checkpoint_step0.pt` | RETIRED_WEIGHT | 521690679 | `27c9b2f73ce9e9dbe29a33c5c9295229f387e67b98fbea45dfbd0c8cfc55c282` | Historical relation-data candidate not promoted; preserve original data/evidence, retire old generator branch. |
| `E:/MeshFlow/experiments/r2_bench_relation_training_v1/training/RELATION_V1_SEAM/checkpoint_step1500.pt` | RETIRED_WEIGHT | 1565056363 | `77602fada9d22adaf9f398a7760b16c77e4a9f73f9c42cba40a1d7205e1863dd` | Historical relation-data candidate not promoted; preserve original data/evidence, retire old generator branch. |
| `E:/MeshFlow/experiments/r2_bench_relation_training_v1/training/RELATION_V1_SEAM/checkpoint_step500.pt` | RETIRED_WEIGHT | 1565056363 | `2a1bb7e9d78f9c0e9f3f48f7a9bbe721d0af62e64ccd5c442110786e6b1ffbbc` | Historical relation-data candidate not promoted; preserve original data/evidence, retire old generator branch. |
| `E:/MeshFlow/experiments/r2_bench_relation_training_v1/training/RELATION_V1_SEAM/generator_step1500.pt` | RETIRED_WEIGHT | 521665915 | `9c2063053d3aa6f1987b7e61aabeaf4852264e6d72326d40afe413059ab3ad15` | Historical relation-data candidate not promoted; preserve original data/evidence, retire old generator branch. |
| `E:/MeshFlow/experiments/r2_bench_relation_training_v1/training/RELATION_V1_SEAM/rolling.pt` | RETIRED_WEIGHT | 1565056363 | `8b5fc2c3024ccc90709ab1fb0190a31cb71b44308b7e41d7385dcd4a82567e07` | Historical relation-data candidate not promoted; preserve original data/evidence, retire old generator branch. |

完整清理事实见 [清理报告](<../../maintenance/storage_cleanup_20261025/CLEANUP_REPORT.md>) 与 [实际完成回执](<../../maintenance/storage_cleanup_20261025/cleanup_receipt.json>)。

本次不切换默认模型，不训练、不生成，不 commit/push。
