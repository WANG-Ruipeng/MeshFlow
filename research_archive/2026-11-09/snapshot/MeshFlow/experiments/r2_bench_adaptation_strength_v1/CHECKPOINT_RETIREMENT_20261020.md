# 2026-10-20 中间权重退役

用户要求清理冗余权重。三臂各自 `checkpoint_step500.pt`、`checkpoint_step1000.pt`、`rolling.pt`（实际 local1250）及 `inference/*_S0_generator.pt`、`*_S500_generator.pt` 已删除，共 15 文件。

各臂 `checkpoint_step0.pt`、`checkpoint_step1500.pt` 和 `*_S1500_generator.pt` 保留。step0 是恢复时的初始参数核验依赖；最终完整权重包含匹配的优化器及 RNG。当前三臂均 COMPLETE，不回退状态或补跑。

原有曲线、过程面板、RAW／POST、逐例评分、图审、注册、源码、旁车 JSON 和完成收据保持。已删除点无法直接重新加载；S0 完整初始化仍在，但不自动重建已删推理导出。原全权重库存审计依赖 S0／S500 导出，清理后不得再称其库存完整；清理前 PASS 与清理后的资产可用性分开记录。

[清理说明](../../maintenance/bench_weight_cleanup_20261020_2241/CLEANUP_REPORT.md) · [逐文件清单](../../maintenance/bench_weight_cleanup_20261020_2241/prepared_manifest.json) · [完成回执](../../maintenance/bench_weight_cleanup_20261020_2241/cleanup_receipt.json)
