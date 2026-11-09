# 当前工作区：JEdge5 主线

当前维护入口是 `native_t1/`。JEdge5 = Native T1 + C-only Geo + FM + 有限接口边 D_edge + 表面测度 L5。D_edge/L5 仅用于训练；网络和原50步采样器不增加推理模块。训练固定时间门控、系数、ramp与分母，详见 [方法与完整训练流程](native_t1/docs/jedge5.md)。

`python -B -m native_t1 train-edge5 ...` 是主训练入口；`sample` 默认使用 edge5 profile。旧纯T1兼容入口保留；FM与surface仅在消融附录中显式使用。主代码无历史实验目录依赖。

| 维护模块 | 职责 |
| --- | --- |
| `model.py` / `context_geometry.py` | 原生时间非对称T1与已知单面Geo编码 |
| `data.py` / `prepare.py` | 真实输入、原变换、free-only nested OT、源标签与面角排列 |
| `training.py` / `train_cli.py` | 显式预算、fresh AdamW、同一原FM路径及几何目标 |
| `objectives.py` / `losses/` | 固定JEdge5配方与两项训练损失，无七臂扫描器 |
| `geometry_checkpoint.py` / `working_model.py` | 严格区分纯FM与几何监督checkpoint身份 |
| `sampling.py` / `runtime.py` | 原50步Euler、51状态C保持、严格确定性BF16/FP32路径 |
| `tests/` | CPU数值、标签、身份、入口与导入边界测试 |

## 本地权重

| 角色 | 路径 | 累计步数 |
| --- | --- | --- |
| 默认工作模型 | `native_t1/checkpoints/jedge5_cumulative3000.pt` | 3000 |
| 共同起点G0 | `native_t1/checkpoints/tgeo_g0_cumulative2500.pt` | 2500 |
| L0消融 | `native_t1/checkpoints/ablations/fm_cumulative3000.pt` | 3000 |
| L5消融 | `native_t1/checkpoints/ablations/surface_cumulative3000.pt` | 3000 |

三个3000步终点为精简导出，每个约522 MB：完整生成器、配方身份和输入RNG保留，优化器动量已剥离；继续训练统一fresh AdamW。所有参数逐位匹配原终点，文件SHA与模型state SHA详见 [迁移记录](native_t1/maintenance/jedge5_cleanup_20261001_115000/checkpoint_migration.json)。G0未修改。原纯T1/A与Geo2000等更早资产仍保留。

JEdge5模型state SHA256：`b9c1df5326ad4815ec3c32ed20e08dc34ff6248ff05be2c412e051fcb728aae3`。
精简文件SHA256：`cc7a248fa9a8570059f225cd1c313f21f08585925407a79418df6d63c9cae0e9`。

## 本次清理边界

已删除七臂筛选的14个step250/500权重，以及J35/JEdge5试验的4个原完整训练权重；保留上述三个无损精简终点。移除的权重共28.172 GB，扣除三个新副本后净释放约26.607 GB。L1/L2/L3/L4/L6/J35及中间权重已经退役，历史报告里的旧权重路径不再可加载。

两个旧活动脚本包 `experiments/mf_h1_local0/loss_screen/` 与 `l3_l5_interface_pilot/` 的33个文件已移出活动代码。可恢复源码ZIP仅126333 bytes，保存在maintenance；完整删除条目、原hash、替代路径与证据校验见 [清理清单](native_t1/maintenance/jedge5_cleanup_20261001_115000/cleanup_manifest.json)。

历史报告、results、逐输出表、图片、原始数组、teacher状态、源码快照全部保留。官方权重、原数据、两棵worktree不在删除范围。权重、数据、runs和maintenance均Git忽略，本轮未commit/push。

原输入仍为 `output/t1_publish_20260930/chair_n112.npz`；公开代码可依据官方数据和固定recipe重新准备，无需私有历史输出。当前证据范围仍是两个观察过的训练父对象，不表示新对象泛化、水密或全指标优势。

本次没有新训练、完整生成或生成器前向；仅迁移损失、CPU验证、严格加载和清理。实际记录见maintenance清理报告。
