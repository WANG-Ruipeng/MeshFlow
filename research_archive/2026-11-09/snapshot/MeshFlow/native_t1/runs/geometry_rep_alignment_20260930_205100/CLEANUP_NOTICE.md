# 本轮文件清理说明

用户在实验完成后授权清理代码和临时权重。

- 报告、results.json、code.diff、指标、图和原始输出均保留，不改实验结论。
- G0 / G_Angular / G_AngularScale 的新增 step250、step500 共六个训练 checkpoint 已删除。历史报告中的这些路径不再可用。
- G0 的完整模型与输入 RNG 已无算术修改地转存为公共 Geo 格式：`../../checkpoints/tgeo_g0_cumulative2500.pt`。模型 state SHA256：`6fa39d1c8c1deade17467075160281dcf01948c45e90ca4a9ad111a7dd97c392`。
- 工作权重文件 SHA256：`3a66659e86586f54d2ec30ed3d4a18275ab69ba0f8cba2866ef67b340c481c0e`；优化器动量被剥离，后续训练使用 fresh AdamW。
- `source/` 历史快照移到仓库根目录的 `output/tgeo_loss_cleanup_20260930_222130/retired_representation_source/`，文件内容哈希保持。
- 实验 Python 模块与 H9 出口退出维护包，主代码恢复已发布基线。恢复实验实现需参考 code.diff / 包外快照，而不是当前默认代码。
- 详细清理与回归记录见仓库根目录 `output/tgeo_loss_cleanup_20260930_222130/`。

本次没有新增训练或完整采样。
