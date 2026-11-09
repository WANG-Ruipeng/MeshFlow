# 后续维护：资产位置变更

2026-10-01 用户授权临时文件清理。原报告、results.json、逐输出数据、最终_v2对照图保持原样，避免改写实验记录。

- 旧 Edge_only step250 中间检查点已退役。
- 旧 step500 完整检查点的模型与输入 RNG 已逐位迁移到 `native_t1/checkpoints/ablations/edge_only_cumulative3000.pt`，只移除了 optimizer 动量。
- 新文件 SHA256：`61d09ce272d2b677ed0ae045e81e8a0c6229cbf77b08de0a612e4a76ff10015f`。
- 生成器 state SHA256 未变：`b9ed4bfd5916d9731daec5afd9732b8bdfff5dbd81749dfda4c7ec3e12f74c7a`。
- Edge-only schema 保持 `native_t1_geo_edge_only_factorial_v1`，不伪装成 FM/JEdge5；公开 CLI 不接受该实验 schema。
- 活动 `edge_factorial_pilot` 源包已存入 `native_t1/maintenance/baseline_cleanup_20261001_163503/retired_recent_experiment_source.zip`，逐成员 SHA 核验通过。需要复现旧加载命令时，先恢复该包，再将 checkpoint 路径改为精简文件；当前公开 FM/JEdge5 运行不依赖该包。
- 五张不带_v2的首版图已删除；五张_v2最终图保留。首版因集合级绘制深度排序而被替代；原哈希及历史引用保留在报告元数据和清理清单中。

完整迁移、删除与保护记录位于上述 maintenance 目录。当前主线改为纯 FM T1+Geo，JEdge5 显式可选；此次维护没有训练或采样。
