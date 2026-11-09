# 2026-10-25 资产退役补记

原实验/资产目录：`E:/MeshFlow/experiments/r2_seam_slot_c200_v1`。

本文件记录用户本轮授权清理之后的本地可用性，不修改该实验在运行当时的事实、指标、结论或历史完成回执。

实际移除 5 份权重及 0 份已有校验 ZIP 的展开审阅副本，逻辑大小 3.887 GiB。

以下权重已退役，依赖它们的历史完整训练重放、对应旧检查点续训或重新模型采样在本地不再可用。代码或旧报告仍可能列出其原路径；不得自动下载、恢复、重训或将缓存输出冒充重新生成。

若本目录仍有保留终点，保留终点的推理/恢复能力与已退役中间状态分开；清理没有启动任何续训。

原 RAW、POST、固定噪声、查询、逐例表、图审、源代码、原报告、原 `COMPLETE/completion` 和历史 hash 不被本次权重退役覆盖。离线复算仍可使用保存的输出；要求已删权重存在的历史全链校验入口不再作为当前可执行入口。

| 移除路径 | 类型 | Bytes | SHA256 | 原因 |
| --- | --- | ---: | --- | --- |
| `E:/MeshFlow/experiments/r2_seam_slot_c200_v1/checkpoints/FULL4_C200_step200.pt` | RETIRED_WEIGHT | 1565057323 | `dd78752c3d63248243a24af0cf0941f3ce3a03da1dc937c445d496509e96e419` | Historical R31 positive proof-of-concept and paired control superseded operationally by S2000; preserve code/reports/RAW/POST and per-output evidence, retire exact neural replay |
| `E:/MeshFlow/experiments/r2_seam_slot_c200_v1/checkpoints/SEAM_SLOT_C200_step200.pt` | RETIRED_WEIGHT | 1565072395 | `c0c26ad26d2123062d3fd38039227d6cd05e269b5ca16e312602c45266fb3d65` | Historical R31 positive proof-of-concept and paired control superseded operationally by S2000; preserve code/reports/RAW/POST and per-output evidence, retire exact neural replay |
| `E:/MeshFlow/experiments/r2_seam_slot_c200_v1/inference/FULL4_C200_generator.pt` | RETIRED_WEIGHT | 521665787 | `5620851b5859380e3dfd183dcdb3d205ec81629bf90dc4ad423d83e766dc40e4` | Historical R31 positive proof-of-concept and paired control superseded operationally by S2000; preserve code/reports/RAW/POST and per-output evidence, retire exact neural replay |
| `E:/MeshFlow/experiments/r2_seam_slot_c200_v1/inference/SEAM_SLOT_C200_generator.pt` | RETIRED_WEIGHT | 521665787 | `faa615dbba945d2d8092cd82976010292e16f313238f9d673f0bd56573d5f2dd` | Historical R31 positive proof-of-concept and paired control superseded operationally by S2000; preserve code/reports/RAW/POST and per-output evidence, retire exact neural replay |
| `E:/MeshFlow/experiments/r2_seam_slot_c200_v1/inference/SEAM_SLOT_C200_occupancy.pt` | RETIRED_WEIGHT | 5981 | `a09523765920f939c3ebd9a200e649f8eb9a0be8c189c1f444eda24e79ac0eba` | Historical R31 positive proof-of-concept and paired control superseded operationally by S2000; preserve code/reports/RAW/POST and per-output evidence, retire exact neural replay |

完整清理事实见 [清理报告](<../../maintenance/storage_cleanup_20261025/CLEANUP_REPORT.md>) 与 [实际完成回执](<../../maintenance/storage_cleanup_20261025/cleanup_receipt.json>)。

本次不切换默认模型，不训练、不生成，不 commit/push。
