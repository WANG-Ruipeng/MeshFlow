# 2026-10-25 资产退役补记

原实验/资产目录：`E:/MeshFlow/native_t1/checkpoints`。

本文件记录用户本轮授权清理之后的本地可用性，不修改该实验在运行当时的事实、指标、结论或历史完成回执。

实际移除 2 份权重及 0 份已有校验 ZIP 的展开审阅副本，逻辑大小 0.972 GiB。

这是中央镜像告示；未向上述外部目录或旧工作树写入新文件。

以下权重已退役，依赖它们的历史完整训练重放、对应旧检查点续训或重新模型采样在本地不再可用。代码或旧报告仍可能列出其原路径；不得自动下载、恢复、重训或将缓存输出冒充重新生成。

若本目录仍有保留终点，保留终点的推理/恢复能力与已退役中间状态分开；清理没有启动任何续训。

原 RAW、POST、固定噪声、查询、逐例表、图审、源代码、原报告、原 `COMPLETE/completion` 和历史 hash 不被本次权重退役覆盖。离线复算仍可使用保存的输出；要求已删权重存在的历史全链校验入口不再作为当前可执行入口。

| 移除路径 | 类型 | Bytes | SHA256 | 原因 |
| --- | --- | ---: | --- | --- |
| `E:/MeshFlow/native_t1/checkpoints/jedge5_cumulative3000.pt` | RETIRED_WEIGHT | 521691011 | `cc7a248fa9a8570059f225cd1c313f21f08585925407a79418df6d63c9cae0e9` | Explicit legacy N112 profile, superseded by maintained START; retire weight and document profile replay unavailable |
| `E:/MeshFlow/native_t1/checkpoints/tgeo_g0_cumulative2500.pt` | RETIRED_WEIGHT | 521693171 | `3a66659e86586f54d2ec30ed3d4a18275ab69ba0f8cba2866ef67b340c481c0e` | Explicit legacy N112 profile, superseded by maintained START; retire weight and document profile replay unavailable |

本目录继续保留的权重：

- [chair_hybrid_start_total5000.pt](<../../../../../../../native_t1/checkpoints/chair_hybrid_start_total5000.pt>)：1.458 GiB，SHA256 `d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80`。

完整清理事实见 [清理报告](<../../../../../CLEANUP_REPORT.md>) 与 [实际完成回执](<../../../../../cleanup_receipt.json>)。

本次不切换默认模型，不训练、不生成，不 commit/push。
