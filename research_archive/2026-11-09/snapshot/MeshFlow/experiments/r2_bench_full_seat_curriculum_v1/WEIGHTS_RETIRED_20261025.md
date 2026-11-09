# 2026-10-25 资产退役补记

原实验/资产目录：`E:/MeshFlow/experiments/r2_bench_full_seat_curriculum_v1`。

本文件记录用户本轮授权清理之后的本地可用性，不修改该实验在运行当时的事实、指标、结论或历史完成回执。

实际移除 8 份权重及 0 份已有校验 ZIP 的展开审阅副本，逻辑大小 10.689 GiB。

以下权重已退役，依赖它们的历史完整训练重放、对应旧检查点续训或重新模型采样在本地不再可用。代码或旧报告仍可能列出其原路径；不得自动下载、恢复、重训或将缓存输出冒充重新生成。

若本目录仍有保留终点，保留终点的推理/恢复能力与已退役中间状态分开；清理没有启动任何续训。

原 RAW、POST、固定噪声、查询、逐例表、图审、源代码、原报告、原 `COMPLETE/completion` 和历史 hash 不被本次权重退役覆盖。离线复算仍可使用保存的输出；要求已删权重存在的历史全链校验入口不再作为当前可执行入口。

| 移除路径 | 类型 | Bytes | SHA256 | 原因 |
| --- | --- | ---: | --- | --- |
| `E:/MeshFlow/experiments/r2_bench_full_seat_curriculum_v1/training/NATIVE/checkpoint_latest.pt` | RETIRED_WEIGHT | 1565061091 | `56c3701bba57bf4ce72be938cefd3996b7c594fed4e06f53d5b1745f1be41367` | Intermediate/initial/rolling curriculum snapshot; both final Native6000 and SEAM3000 full checkpoints plus exports retained. COMMON12000 source retained. |
| `E:/MeshFlow/experiments/r2_bench_full_seat_curriculum_v1/training/NATIVE/checkpoint_step0.pt` | RETIRED_WEIGHT | 1565061027 | `49e9826dc39e514c3ad4d428d99f99391b47bc9789f06d0ba848773494d32c97` | Intermediate/initial/rolling curriculum snapshot; both final Native6000 and SEAM3000 full checkpoints plus exports retained. COMMON12000 source retained. |
| `E:/MeshFlow/experiments/r2_bench_full_seat_curriculum_v1/training/NATIVE/checkpoint_step1000.pt` | RETIRED_WEIGHT | 1565061091 | `781904c9f901e9378e51fb371d4f74800d07d2a408f40bf430f925b63cb161d6` | Intermediate/initial/rolling curriculum snapshot; both final Native6000 and SEAM3000 full checkpoints plus exports retained. COMMON12000 source retained. |
| `E:/MeshFlow/experiments/r2_bench_full_seat_curriculum_v1/training/NATIVE/checkpoint_step3000.pt` | RETIRED_WEIGHT | 1565061091 | `73ac32a28f89c0d05935bb4a09d796d8fc4f7d9356eb2b0b55742882cf762be1` | Intermediate/initial/rolling curriculum snapshot; both final Native6000 and SEAM3000 full checkpoints plus exports retained. COMMON12000 source retained. |
| `E:/MeshFlow/experiments/r2_bench_full_seat_curriculum_v1/training/NATIVE/generator_step0.pt` | RETIRED_WEIGHT | 521670963 | `13cffc5586e640270e18d86970db6602fb14032753cd2f2a143f5dacd6cb8b9d` | Intermediate/initial/rolling curriculum snapshot; both final Native6000 and SEAM3000 full checkpoints plus exports retained. COMMON12000 source retained. |
| `E:/MeshFlow/experiments/r2_bench_full_seat_curriculum_v1/training/SEAM/checkpoint_latest.pt` | RETIRED_WEIGHT | 1565061795 | `42bc41aaedfd96a4544ae75e16607a73ca5943f7cbaf398bac8d567491e23065` | Intermediate/initial/rolling curriculum snapshot; both final Native6000 and SEAM3000 full checkpoints plus exports retained. COMMON12000 source retained. |
| `E:/MeshFlow/experiments/r2_bench_full_seat_curriculum_v1/training/SEAM/checkpoint_step0.pt` | RETIRED_WEIGHT | 1565061731 | `c701d75d9d933d3d9380dfc411dbdb58672864c12de25edb1525322b00b0d30d` | Intermediate/initial/rolling curriculum snapshot; both final Native6000 and SEAM3000 full checkpoints plus exports retained. COMMON12000 source retained. |
| `E:/MeshFlow/experiments/r2_bench_full_seat_curriculum_v1/training/SEAM/checkpoint_step1000.pt` | RETIRED_WEIGHT | 1565061795 | `5e0b00a153150a7b754368d7e07255a67a1805f0c842f8089b677dd0ac633934` | Intermediate/initial/rolling curriculum snapshot; both final Native6000 and SEAM3000 full checkpoints plus exports retained. COMMON12000 source retained. |

本目录继续保留的权重：

- [checkpoint_step6000.pt](<training/NATIVE/checkpoint_step6000.pt>)：1.458 GiB，SHA256 `31aef34d1126c4ecb7a22201ec3b874288417d7b57dda8e56f7dfe7eb49a8bfe`。
- [generator_step6000.pt](<training/NATIVE/generator_step6000.pt>)：0.486 GiB，SHA256 `3d296ab1067c6edd1ddbafad759e75090970739aaac1a9ccc23ac9ecc9467ede`。
- [checkpoint_step3000.pt](<training/SEAM/checkpoint_step3000.pt>)：1.458 GiB，SHA256 `db85c70b5d1014387f741003b19347be29a077246a1cc9c1c497c1e5012e39f9`。
- [generator_step3000.pt](<training/SEAM/generator_step3000.pt>)：0.486 GiB，SHA256 `3b4da0254584d3140b430eee50183eb50d04fff6dde83a237dd163efbf2d712d`。

完整清理事实见 [清理报告](<../../maintenance/storage_cleanup_20261025/CLEANUP_REPORT.md>) 与 [实际完成回执](<../../maintenance/storage_cleanup_20261025/cleanup_receipt.json>)。

本次不切换默认模型，不训练、不生成，不 commit/push。
