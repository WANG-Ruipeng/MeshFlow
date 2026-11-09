# 2026-10-25 资产退役补记

原实验/资产目录：`E:/MeshFlow/experiments/r2_full_seat_learning_curve_v1`。

本文件记录用户本轮授权清理之后的本地可用性，不修改该实验在运行当时的事实、指标、结论或历史完成回执。

实际移除 4 份权重及 0 份已有校验 ZIP 的展开审阅副本，逻辑大小 3.887 GiB。

以下权重已退役，依赖它们的历史完整训练重放、对应旧检查点续训或重新模型采样在本地不再可用。代码或旧报告仍可能列出其原路径；不得自动下载、恢复、重训或将缓存输出冒充重新生成。

若本目录仍有保留终点，保留终点的推理/恢复能力与已退役中间状态分开；清理没有启动任何续训。

原 RAW、POST、固定噪声、查询、逐例表、图审、源代码、原报告、原 `COMPLETE/completion` 和历史 hash 不被本次权重退役覆盖。离线复算仍可使用保存的输出；要求已删权重存在的历史全链校验入口不再作为当前可执行入口。

| 移除路径 | 类型 | Bytes | SHA256 | 原因 |
| --- | --- | ---: | --- | --- |
| `E:/MeshFlow/experiments/r2_full_seat_learning_curve_v1/checkpoints/FULL_SEAT_step1000.pt` | RETIRED_WEIGHT | 1565284587 | `b6f65aa93435f3d9429f6b125bd0b042c26b447fb8849d40be4e40f17bf5a554` | Intermediate learning-curve checkpoint; final FULL_S6000 retained with all saved curve outputs |
| `E:/MeshFlow/experiments/r2_full_seat_learning_curve_v1/checkpoints/FULL_SEAT_step3000.pt` | RETIRED_WEIGHT | 1565284779 | `297781d1c740fcce690cc1b73f92e67a232b3d27178d1eb78a1de21bd474fccb` | Intermediate learning-curve checkpoint; final FULL_S6000 retained with all saved curve outputs |
| `E:/MeshFlow/experiments/r2_full_seat_learning_curve_v1/inference/FULL_S1000_generator.pt` | RETIRED_WEIGHT | 521666683 | `ac6af8ecbeff657fbf0f7666ed170d2805e356690472c3d52a3663be499b40c7` | Intermediate learning-curve checkpoint; final FULL_S6000 retained with all saved curve outputs |
| `E:/MeshFlow/experiments/r2_full_seat_learning_curve_v1/inference/FULL_S3000_generator.pt` | RETIRED_WEIGHT | 521666683 | `4a95e1765f7deaf849b99be0d9b539e77895cf24ec267c937980f5ac99604eab` | Intermediate learning-curve checkpoint; final FULL_S6000 retained with all saved curve outputs |

本目录继续保留的权重：

- [FULL_SEAT_step6000.pt](<checkpoints/FULL_SEAT_step6000.pt>)：1.458 GiB，SHA256 `5f41df3d3c11a88fdeb9ad70682a4b18020a1d5838ee08a1956e5ba2c927af33`。
- [FULL_S6000_generator.pt](<inference/FULL_S6000_generator.pt>)：0.486 GiB，SHA256 `86b38d35c44d3a6fd83a5b8144b57813f5238efab557e0cf46a7872bbde5c5cc`。

完整清理事实见 [清理报告](<../../maintenance/storage_cleanup_20261025/CLEANUP_REPORT.md>) 与 [实际完成回执](<../../maintenance/storage_cleanup_20261025/cleanup_receipt.json>)。

本次不切换默认模型，不训练、不生成，不 commit/push。
