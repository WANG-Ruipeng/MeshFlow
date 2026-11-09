# 2026-10-25 资产退役补记

原实验/资产目录：`E:/MeshFlow/experiments/r2_bench_joint_shift_v1`。

本文件记录用户本轮授权清理之后的本地可用性，不修改该实验在运行当时的事实、指标、结论或历史完成回执。

实际移除 5 份权重及 0 份已有校验 ZIP 的展开审阅副本，逻辑大小 5.344 GiB。

以下权重已退役，依赖它们的历史完整训练重放、对应旧检查点续训或重新模型采样在本地不再可用。代码或旧报告仍可能列出其原路径；不得自动下载、恢复、重训或将缓存输出冒充重新生成。

若本目录仍有保留终点，保留终点的推理/恢复能力与已退役中间状态分开；清理没有启动任何续训。

原 RAW、POST、固定噪声、查询、逐例表、图审、源代码、原报告、原 `COMPLETE/completion` 和历史 hash 不被本次权重退役覆盖。离线复算仍可使用保存的输出；要求已删权重存在的历史全链校验入口不再作为当前可执行入口。

| 移除路径 | 类型 | Bytes | SHA256 | 原因 |
| --- | --- | ---: | --- | --- |
| `E:/MeshFlow/experiments/r2_bench_joint_shift_v1/training/JOINT_SHIFT_SEAM/checkpoint_step0.pt` | RETIRED_WEIGHT | 521690679 | `c36df3120824a5a20160875fb080e2139801463007d3df770e98b658ec9e2677` | SHIFT follow-up gate closed; adoption evidence insufficient across DEV parents and geometry costs remain. |
| `E:/MeshFlow/experiments/r2_bench_joint_shift_v1/training/JOINT_SHIFT_SEAM/checkpoint_step1500.pt` | RETIRED_WEIGHT | 1565056363 | `536c6adf11736261ce9e31274f13f9ded86b3fb62d830024d2301f745d63ff82` | SHIFT follow-up gate closed; adoption evidence insufficient across DEV parents and geometry costs remain. |
| `E:/MeshFlow/experiments/r2_bench_joint_shift_v1/training/JOINT_SHIFT_SEAM/checkpoint_step500.pt` | RETIRED_WEIGHT | 1565056363 | `364964df42df3a4301951efd362636cbff6d46a3177d3390b52772886b8ec480` | SHIFT follow-up gate closed; adoption evidence insufficient across DEV parents and geometry costs remain. |
| `E:/MeshFlow/experiments/r2_bench_joint_shift_v1/training/JOINT_SHIFT_SEAM/generator_step1500.pt` | RETIRED_WEIGHT | 521665851 | `b78fb5bd67e682387154b815b1393a6096e3bc434f9ae4cfb64029e09840a440` | SHIFT follow-up gate closed; adoption evidence insufficient across DEV parents and geometry costs remain. |
| `E:/MeshFlow/experiments/r2_bench_joint_shift_v1/training/JOINT_SHIFT_SEAM/rolling.pt` | RETIRED_WEIGHT | 1565056363 | `bcbb010aff3a2981f04cf0f9cc10edaf3c8454b92e9e7cb7ccaf3b1d28c76afa` | SHIFT follow-up gate closed; adoption evidence insufficient across DEV parents and geometry costs remain. |

完整清理事实见 [清理报告](<../../maintenance/storage_cleanup_20261025/CLEANUP_REPORT.md>) 与 [实际完成回执](<../../maintenance/storage_cleanup_20261025/cleanup_receipt.json>)。

本次不切换默认模型，不训练、不生成，不 commit/push。
