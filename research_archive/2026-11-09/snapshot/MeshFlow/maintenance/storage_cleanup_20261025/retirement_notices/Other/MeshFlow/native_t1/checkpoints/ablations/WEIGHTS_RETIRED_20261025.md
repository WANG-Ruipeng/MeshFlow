# 2026-10-25 资产退役补记

原实验/资产目录：`E:/MeshFlow/native_t1/checkpoints/ablations`。

本文件记录用户本轮授权清理之后的本地可用性，不修改该实验在运行当时的事实、指标、结论或历史完成回执。

实际移除 3 份权重及 0 份已有校验 ZIP 的展开审阅副本，逻辑大小 1.458 GiB。

这是中央镜像告示；未向上述外部目录或旧工作树写入新文件。

以下权重已退役，依赖它们的历史完整训练重放、对应旧检查点续训或重新模型采样在本地不再可用。代码或旧报告仍可能列出其原路径；不得自动下载、恢复、重训或将缓存输出冒充重新生成。

若本目录仍有保留终点，保留终点的推理/恢复能力与已退役中间状态分开；清理没有启动任何续训。

原 RAW、POST、固定噪声、查询、逐例表、图审、源代码、原报告、原 `COMPLETE/completion` 和历史 hash 不被本次权重退役覆盖。离线复算仍可使用保存的输出；要求已删权重存在的历史全链校验入口不再作为当前可执行入口。

| 移除路径 | 类型 | Bytes | SHA256 | 原因 |
| --- | --- | ---: | --- | --- |
| `E:/MeshFlow/native_t1/checkpoints/ablations/edge_only_cumulative3000.pt` | RETIRED_WEIGHT | 521710363 | `61d09ce272d2b677ed0ae045e81e8a0c6229cbf77b08de0a612e4a76ff10015f` | Explicit legacy N112 profile, superseded by maintained START; retire weight and document profile replay unavailable |
| `E:/MeshFlow/native_t1/checkpoints/ablations/fm_cumulative3000.pt` | RETIRED_WEIGHT | 521689571 | `0f22ce498ca2c27dc74b43890fb8aad20f0f3e7153aa74ee6d3c14ea445a2660` | Explicit legacy N112 profile, superseded by maintained START; retire weight and document profile replay unavailable |
| `E:/MeshFlow/native_t1/checkpoints/ablations/surface_cumulative3000.pt` | RETIRED_WEIGHT | 521691211 | `93546de419582302e70cf140dae3aa44aad9ab913bd11b394482a8c347203999` | Explicit legacy N112 profile, superseded by maintained START; retire weight and document profile replay unavailable |

完整清理事实见 [清理报告](<../../../../../../CLEANUP_REPORT.md>) 与 [实际完成回执](<../../../../../../cleanup_receipt.json>)。

本次不切换默认模型，不训练、不生成，不 commit/push。
