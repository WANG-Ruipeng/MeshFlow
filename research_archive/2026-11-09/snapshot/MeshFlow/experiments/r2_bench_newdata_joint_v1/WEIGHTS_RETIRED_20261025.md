# 2026-10-25 资产退役补记

原实验/资产目录：`E:/MeshFlow/experiments/r2_bench_newdata_joint_v1`。

本文件记录用户本轮授权清理之后的本地可用性，不修改该实验在运行当时的事实、指标、结论或历史完成回执。

实际移除 5 份权重及 0 份已有校验 ZIP 的展开审阅副本，逻辑大小 5.344 GiB。

以下权重已退役，依赖它们的历史完整训练重放、对应旧检查点续训或重新模型采样在本地不再可用。代码或旧报告仍可能列出其原路径；不得自动下载、恢复、重训或将缓存输出冒充重新生成。

若本目录仍有保留终点，保留终点的推理/恢复能力与已退役中间状态分开；清理没有启动任何续训。

原 RAW、POST、固定噪声、查询、逐例表、图审、源代码、原报告、原 `COMPLETE/completion` 和历史 hash 不被本次权重退役覆盖。离线复算仍可使用保存的输出；要求已删权重存在的历史全链校验入口不再作为当前可执行入口。

| 移除路径 | 类型 | Bytes | SHA256 | 原因 |
| --- | --- | ---: | --- | --- |
| `E:/MeshFlow/experiments/r2_bench_newdata_joint_v1/training/NEW_DATA_JOINT/checkpoint_step0.pt` | RETIRED_WEIGHT | 521696751 | `938b833eabb1f2b0c2efbff4d7ee9ed00f9f44b8e1a6b0def1593bb742026501` | No promotion: old layout recall and DEV FF/I costs; latest complete-seat and Coverage endpoints supersede continued use. |
| `E:/MeshFlow/experiments/r2_bench_newdata_joint_v1/training/NEW_DATA_JOINT/checkpoint_step1500.pt` | RETIRED_WEIGHT | 1565072395 | `eabb01e879813f280b56a028bec59c81e7a2e3a6c7e2b94633d788c134bd49e4` | No promotion: old layout recall and DEV FF/I costs; latest complete-seat and Coverage endpoints supersede continued use. |
| `E:/MeshFlow/experiments/r2_bench_newdata_joint_v1/training/NEW_DATA_JOINT/checkpoint_step500.pt` | RETIRED_WEIGHT | 1565072395 | `c7eec3fa4510dbf62c4749fb1c7bf7ae82c5959a3401d7ce792d61ec0fa21cd4` | No promotion: old layout recall and DEV FF/I costs; latest complete-seat and Coverage endpoints supersede continued use. |
| `E:/MeshFlow/experiments/r2_bench_newdata_joint_v1/training/NEW_DATA_JOINT/generator_step1500.pt` | RETIRED_WEIGHT | 521670835 | `7d28f2f9ba0db42e66a389a1f7c014f56630a5f04e130e4274a927d356fabad6` | No promotion: old layout recall and DEV FF/I costs; latest complete-seat and Coverage endpoints supersede continued use. |
| `E:/MeshFlow/experiments/r2_bench_newdata_joint_v1/training/NEW_DATA_JOINT/rolling.pt` | RETIRED_WEIGHT | 1565072395 | `defdbff5e869aaba113b01e32f2b0323abe74bb446d0e77c0e973de840f63114` | No promotion: old layout recall and DEV FF/I costs; latest complete-seat and Coverage endpoints supersede continued use. |

完整清理事实见 [清理报告](<../../maintenance/storage_cleanup_20261025/CLEANUP_REPORT.md>) 与 [实际完成回执](<../../maintenance/storage_cleanup_20261025/cleanup_receipt.json>)。

本次不切换默认模型，不训练、不生成，不 commit/push。
