# 本轮保留模型身份

均为Bench Native/T1＋role＋Geo，无SEAM/占用头调用。两臂固定终点均local1500、global22500、native10500、curriculum4500。所有hash为完整文件SHA256；生成器state hash、AdamW和RNG另见同名JSON。

| 身份 | 文件（相对本目录） | SHA256 |
|---|---|---|
| A 普通续训推理 | `training/CONTROL_OLD/generator_step1500.pt` | `b7c8178b84ae70d6eef47cf75387fcc172b32b4785a23cf67550365510c9e839` |
| A 完整恢复 | `training/CONTROL_OLD/checkpoint_step1500.pt` | `0da7492297f8dd307ce204c57cda41c78d37c35fce8235b2c535272117678f2c` |
| B 自然面数新配方推理 | `training/NATURAL_HIGH/generator_step1500.pt` | `b9a3979c84bb851489c49f8b1539618c6908d9f64b4b579cad4e51043264f2f8` |
| B 完整恢复 | `training/NATURAL_HIGH/checkpoint_step1500.pt` | `111d2142780ddb21c9aa94cef9174571fbb22f0602d561c663c4af92b4a9e17d` |

共同来源完整恢复点：`../r2_bench_exposure_allocation_v1/training/B_75_25/checkpoint_step1500.pt`，SHA256 `e22c72b26ef6648336a0f14f98f371db9f5d4a86696a9f197680593baf303443`。P0推理导出在同目录 `generator_step1500.pt`，SHA256 `f427d5115ed86fb4f32482c4237afdb63fbec135a9b24b9780b9f324d348d09e`。

每份新完整恢复文件1,565,061,155字节，每份新推理文件521,671,027字节。滚动安全恢复点仍保留；本轮没有清理权重。官方权重、现有Chair/Bench研究参考和维护CLI默认不改变。

保留B作为本轮已见新结构学习的研究候选，不据此晋级为通用Bench最佳流程。完整适用范围、矛盾代价与图审见REPORT.md。
