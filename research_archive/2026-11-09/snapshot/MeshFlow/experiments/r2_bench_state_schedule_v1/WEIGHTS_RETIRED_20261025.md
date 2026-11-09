# 2026-10-25 资产退役补记

原实验/资产目录：`E:/MeshFlow/experiments/r2_bench_state_schedule_v1`。

本文件记录用户本轮授权清理之后的本地可用性，不修改该实验在运行当时的事实、指标、结论或历史完成回执。

实际移除 6 份权重及 0 份已有校验 ZIP 的展开审阅副本，逻辑大小 4.859 GiB。

以下权重已退役，依赖它们的历史完整训练重放、对应旧检查点续训或重新模型采样在本地不再可用。代码或旧报告仍可能列出其原路径；不得自动下载、恢复、重训或将缓存输出冒充重新生成。

若本目录仍有保留终点，保留终点的推理/恢复能力与已退役中间状态分开；清理没有启动任何续训。

原 RAW、POST、固定噪声、查询、逐例表、图审、源代码、原报告、原 `COMPLETE/completion` 和历史 hash 不被本次权重退役覆盖。离线复算仍可使用保存的输出；要求已删权重存在的历史全链校验入口不再作为当前可执行入口。

| 移除路径 | 类型 | Bytes | SHA256 | 原因 |
| --- | --- | ---: | --- | --- |
| `E:/MeshFlow/experiments/r2_bench_state_schedule_v1/training/ASYNC_LEAD_SEAM/checkpoint_step0.pt` | RETIRED_WEIGHT | 521690679 | `064a9855ae42adfd321d33986d5ea8e3113e4b4c2cc7f7a489e458b3a81f453b` | PERT/ASYNC both completed without overall upgrade; retain fixed outputs and full reports. |
| `E:/MeshFlow/experiments/r2_bench_state_schedule_v1/training/ASYNC_LEAD_SEAM/checkpoint_step1500.pt` | RETIRED_WEIGHT | 1565056363 | `f3a973c3ffd59daf1d5d5a86fbbc133570c9093e570ab513655a1f323e917341` | PERT/ASYNC both completed without overall upgrade; retain fixed outputs and full reports. |
| `E:/MeshFlow/experiments/r2_bench_state_schedule_v1/training/ASYNC_LEAD_SEAM/generator_step1500.pt` | RETIRED_WEIGHT | 521665851 | `e26b2ea39f0fb23a6d368b3a958e9c68654b45eb8260b0f48acba3064561556c` | PERT/ASYNC both completed without overall upgrade; retain fixed outputs and full reports. |
| `E:/MeshFlow/experiments/r2_bench_state_schedule_v1/training/PERTURBED_STATE/checkpoint_step0.pt` | RETIRED_WEIGHT | 521690679 | `9fea0066edaf1703647a90219ff08af82a638f5b4e3ccf4bf61f4465e9bbf36b` | PERT/ASYNC both completed without overall upgrade; retain fixed outputs and full reports. |
| `E:/MeshFlow/experiments/r2_bench_state_schedule_v1/training/PERTURBED_STATE/checkpoint_step1500.pt` | RETIRED_WEIGHT | 1565056363 | `e0c772f11200ef25607ddd76c3092cc5998cb865eff008f7e459f7e641d1ae7b` | PERT/ASYNC both completed without overall upgrade; retain fixed outputs and full reports. |
| `E:/MeshFlow/experiments/r2_bench_state_schedule_v1/training/PERTURBED_STATE/generator_step1500.pt` | RETIRED_WEIGHT | 521665851 | `c0d7d5d0f8df51024cb99064489ff3d9f3959797f80d30624fcaf5f3312a64fc` | PERT/ASYNC both completed without overall upgrade; retain fixed outputs and full reports. |

完整清理事实见 [清理报告](<../../maintenance/storage_cleanup_20261025/CLEANUP_REPORT.md>) 与 [实际完成回执](<../../maintenance/storage_cleanup_20261025/cleanup_receipt.json>)。

本次不切换默认模型，不训练、不生成，不 commit/push。
