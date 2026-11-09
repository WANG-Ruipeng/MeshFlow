# 2026-10-25 资产退役补记

原实验/资产目录：`E:/MeshFlow/experiments/r2_bench_parts_convergence_v1`。

本文件记录用户本轮授权清理之后的本地可用性，不修改该实验在运行当时的事实、指标、结论或历史完成回执。

实际移除 9 份权重及 0 份已有校验 ZIP 的展开审阅副本，逻辑大小 9.231 GiB。

以下权重已退役，依赖它们的历史完整训练重放、对应旧检查点续训或重新模型采样在本地不再可用。代码或旧报告仍可能列出其原路径；不得自动下载、恢复、重训或将缓存输出冒充重新生成。

若本目录仍有保留终点，保留终点的推理/恢复能力与已退役中间状态分开；清理没有启动任何续训。

原 RAW、POST、固定噪声、查询、逐例表、图审、源代码、原报告、原 `COMPLETE/completion` 和历史 hash 不被本次权重退役覆盖。离线复算仍可使用保存的输出；要求已删权重存在的历史全链校验入口不再作为当前可执行入口。

| 移除路径 | 类型 | Bytes | SHA256 | 原因 |
| --- | --- | ---: | --- | --- |
| `E:/MeshFlow/experiments/r2_bench_parts_convergence_v1/audits/worker_write_race_20261019/interrupted.rolling.pt` | RETIRED_WEIGHT | 1565045931 | `2c37e0cd7c0c7054f1bea215e9e6634e96be1f52838f8cf9ab11cad1f33a936c` | Historical interrupted checkpoint from completed recovery engineering audit; incident receipt/source evidence remains, COMMON12000 final retained. |
| `E:/MeshFlow/experiments/r2_bench_parts_convergence_v1/training/PLAIN/generator_step15000.pt` | RETIRED_WEIGHT | 521665723 | `1b98541e31759fe7cc64968b4413633b1f6a8de0f5697cc6bea91de7e905facc` | Historical multipart Plain/SEAM15000/21000 endpoints superseded by explicitly retained complete-seat reference; head exported independently and kept. |
| `E:/MeshFlow/experiments/r2_bench_parts_convergence_v1/training/PLAIN/generator_step21000.pt` | RETIRED_WEIGHT | 521665723 | `36a8d84a6ecc97b90c0deda625e77534a43e6dbc1fe45c47c7acada74660f0fa` | Historical multipart Plain/SEAM15000/21000 endpoints superseded by explicitly retained complete-seat reference; head exported independently and kept. |
| `E:/MeshFlow/experiments/r2_bench_parts_convergence_v1/training/PLAIN/step15000.pt` | RETIRED_WEIGHT | 1565046315 | `fa22a30c08e49723418417246832cddef9e9d5715156e18af5192274b61c4239` | Historical multipart Plain/SEAM15000/21000 endpoints superseded by explicitly retained complete-seat reference; head exported independently and kept. |
| `E:/MeshFlow/experiments/r2_bench_parts_convergence_v1/training/PLAIN/step21000.pt` | RETIRED_WEIGHT | 1565046315 | `c078b20e01b1afa7dde272971f4aff4c04fa5e746c3b6f212694a0b22f96bdd3` | Historical multipart Plain/SEAM15000/21000 endpoints superseded by explicitly retained complete-seat reference; head exported independently and kept. |
| `E:/MeshFlow/experiments/r2_bench_parts_convergence_v1/training/SEAM/generator_step15000.pt` | RETIRED_WEIGHT | 521670195 | `01e2e1834eb79834cfc47f88bd7080ff8cf166c6f243f890d36522be9c4d8023` | Historical multipart Plain/SEAM15000/21000 endpoints superseded by explicitly retained complete-seat reference; head exported independently and kept. |
| `E:/MeshFlow/experiments/r2_bench_parts_convergence_v1/training/SEAM/generator_step21000.pt` | RETIRED_WEIGHT | 521670195 | `98b5fb8e562c59e18a40ef40f0cbc7497a5e4088d3bb36cfb35a4b2bcc70acc9` | Historical multipart Plain/SEAM15000/21000 endpoints superseded by explicitly retained complete-seat reference; head exported independently and kept. |
| `E:/MeshFlow/experiments/r2_bench_parts_convergence_v1/training/SEAM/step15000.pt` | RETIRED_WEIGHT | 1565061003 | `53984dead4e0505d5dfe7b3b98623ca5a5d512017eb3bc6ea0a2ec59a2248cbe` | Historical multipart Plain/SEAM15000/21000 endpoints superseded by explicitly retained complete-seat reference; head exported independently and kept. |
| `E:/MeshFlow/experiments/r2_bench_parts_convergence_v1/training/SEAM/step21000.pt` | RETIRED_WEIGHT | 1565061003 | `7c32b114724e94a08a7a9ddbc20ee26e7d92e71e62639103b0ca4462724f24bc` | Historical multipart Plain/SEAM15000/21000 endpoints superseded by explicitly retained complete-seat reference; head exported independently and kept. |

本目录继续保留的权重：

- [generator_step12000.pt](<training/COMMON/generator_step12000.pt>)：0.486 GiB，SHA256 `a8eb21c00a05ef5965ce1b7783d773593efc7013665d784ee7cb1dd3096abcc1`。
- [step12000.pt](<training/COMMON/step12000.pt>)：1.458 GiB，SHA256 `40aecccd1838a5b04dadeac13e72b9cf7315f3721c16e180b97844a04c2f8032`。

完整清理事实见 [清理报告](<../../maintenance/storage_cleanup_20261025/CLEANUP_REPORT.md>) 与 [实际完成回执](<../../maintenance/storage_cleanup_20261025/cleanup_receipt.json>)。

本次不切换默认模型，不训练、不生成，不 commit/push。
