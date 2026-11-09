# 2026-10-25 资产退役补记

原实验/资产目录：`E:/MeshFlow/experiments/r2_bench_adaptation_strength_v1`。

本文件记录用户本轮授权清理之后的本地可用性，不修改该实验在运行当时的事实、指标、结论或历史完成回执。

实际移除 9 份权重及 0 份已有校验 ZIP 的展开审阅副本，逻辑大小 7.288 GiB。

以下权重已退役，依赖它们的历史完整训练重放、对应旧检查点续训或重新模型采样在本地不再可用。代码或旧报告仍可能列出其原路径；不得自动下载、恢复、重训或将缓存输出冒充重新生成。

若本目录仍有保留终点，保留终点的推理/恢复能力与已退役中间状态分开；清理没有启动任何续训。

原 RAW、POST、固定噪声、查询、逐例表、图审、源代码、原报告、原 `COMPLETE/completion` 和历史 hash 不被本次权重退役覆盖。离线复算仍可使用保存的输出；要求已删权重存在的历史全链校验入口不再作为当前可执行入口。

| 移除路径 | 类型 | Bytes | SHA256 | 原因 |
| --- | --- | ---: | --- | --- |
| `E:/MeshFlow/experiments/r2_bench_adaptation_strength_v1/inference/A_COMMON_LR1E5_S1500_generator.pt` | RETIRED_WEIGHT | 521665915 | `b8b703cb02ae4f2638b8b0fe73ae1a7663b9d53f9ce3b0da8ae1c79a6e3391fc` | A/B/C1500 historical comparisons complete, no accepted overall upgrade; later current references kept. |
| `E:/MeshFlow/experiments/r2_bench_adaptation_strength_v1/inference/B_COMMON_LR2E6_S1500_generator.pt` | RETIRED_WEIGHT | 521665915 | `3817d4bcc818d92bafe3891df36d16f7201d69cce955afebf772649bd3a4ffa4` | A/B/C1500 historical comparisons complete, no accepted overall upgrade; later current references kept. |
| `E:/MeshFlow/experiments/r2_bench_adaptation_strength_v1/inference/C_OFFICIAL_LR2E6_S1500_generator.pt` | RETIRED_WEIGHT | 521665915 | `a272f0b5b1a469f682fafeb3dc4a4d3820784d1af473e9f179a16a4e128683ec` | A/B/C1500 historical comparisons complete, no accepted overall upgrade; later current references kept. |
| `E:/MeshFlow/experiments/r2_bench_adaptation_strength_v1/training/A_COMMON_LR1E5/checkpoint_step0.pt` | RETIRED_WEIGHT | 521699511 | `75d54c15f3ff2ee1328dddca1269352db48f5ecbca4d459eaff8c603c2297398` | A/B/C1500 historical comparisons complete, no accepted overall upgrade; later current references kept. |
| `E:/MeshFlow/experiments/r2_bench_adaptation_strength_v1/training/A_COMMON_LR1E5/checkpoint_step1500.pt` | RETIRED_WEIGHT | 1565065195 | `3a0a88c7a842b9131987f2dae94229646de87e36f3bb6b6fa5f0b6230f0602f2` | A/B/C1500 historical comparisons complete, no accepted overall upgrade; later current references kept. |
| `E:/MeshFlow/experiments/r2_bench_adaptation_strength_v1/training/B_COMMON_LR2E6/checkpoint_step0.pt` | RETIRED_WEIGHT | 521699511 | `eb1d27ca9150d01320859ad5a3ab14d1ba25e8969c2fb1d1535a28d96e96a0b2` | A/B/C1500 historical comparisons complete, no accepted overall upgrade; later current references kept. |
| `E:/MeshFlow/experiments/r2_bench_adaptation_strength_v1/training/B_COMMON_LR2E6/checkpoint_step1500.pt` | RETIRED_WEIGHT | 1565065195 | `fda7d4eeaa75396c87516423187e77f05e425f9209ab0521a8f95dfa8acce07c` | A/B/C1500 historical comparisons complete, no accepted overall upgrade; later current references kept. |
| `E:/MeshFlow/experiments/r2_bench_adaptation_strength_v1/training/C_OFFICIAL_LR2E6/checkpoint_step0.pt` | RETIRED_WEIGHT | 521699575 | `c75a7623ac22fcc6c20fc7acda02aff0eab5093cb86062234792998ffb2ba6b8` | A/B/C1500 historical comparisons complete, no accepted overall upgrade; later current references kept. |
| `E:/MeshFlow/experiments/r2_bench_adaptation_strength_v1/training/C_OFFICIAL_LR2E6/checkpoint_step1500.pt` | RETIRED_WEIGHT | 1565065259 | `9207bc1b8359877e448459e625825c77f4f430c609edb361692c4b6de0364100` | A/B/C1500 historical comparisons complete, no accepted overall upgrade; later current references kept. |

完整清理事实见 [清理报告](<../../maintenance/storage_cleanup_20261025/CLEANUP_REPORT.md>) 与 [实际完成回执](<../../maintenance/storage_cleanup_20261025/cleanup_receipt.json>)。

本次不切换默认模型，不训练、不生成，不 commit/push。
