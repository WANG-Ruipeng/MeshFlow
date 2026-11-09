# 2026-10-25 资产退役补记

原实验/资产目录：`E:/MeshFlow-Control-artifacts/experiments/r2_user_condition_finetune_pilot_v1`。

本文件记录用户本轮授权清理之后的本地可用性，不修改该实验在运行当时的事实、指标、结论或历史完成回执。

实际移除 1 份权重及 0 份已有校验 ZIP 的展开审阅副本，逻辑大小 0.486 GiB。

这是中央镜像告示；未向上述外部目录或旧工作树写入新文件。

以下权重已退役，依赖它们的历史完整训练重放、对应旧检查点续训或重新模型采样在本地不再可用。代码或旧报告仍可能列出其原路径；不得自动下载、恢复、重训或将缓存输出冒充重新生成。

若本目录仍有保留终点，保留终点的推理/恢复能力与已退役中间状态分开；清理没有启动任何续训。

原 RAW、POST、固定噪声、查询、逐例表、图审、源代码、原报告、原 `COMPLETE/completion` 和历史 hash 不被本次权重退役覆盖。离线复算仍可使用保存的输出；要求已删权重存在的历史全链校验入口不再作为当前可执行入口。

| 移除路径 | 类型 | Bytes | SHA256 | 原因 |
| --- | --- | ---: | --- | --- |
| `E:/MeshFlow-Control-artifacts/experiments/r2_user_condition_finetune_pilot_v1/inference/R2_CONT_USER_generator_step7000.pt` | RETIRED_WEIGHT | 521687739 | `61e531992ad5201f55fa829a9e6cfee318309e865a3800a1484cc7f6a7fd547b` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |

完整清理事实见 [清理报告](<../../../../../CLEANUP_REPORT.md>) 与 [实际完成回执](<../../../../../cleanup_receipt.json>)。

本次不切换默认模型，不训练、不生成，不 commit/push。
