# 2026-10-20 旧 rolling 退役

用户要求清理冗余权重。本次仅删除 `training/rolling.pt`，其实际状态为 local1250／global13250，已被 local1500／global13500 完整终点取代。

保留 `training/checkpoint_step1500.pt`、最终生成器及 occupancy 导出；当前状态 COMPLETE，指向最终完整检查点。旧 rolling 不再可加载，不自动恢复或补跑。

原数据、噪声、RAW／POST、评分、图审、训练曲线、源码、报告、所有 JSON 旁车与完成收据保留。历史库存证据仍反映当时状态，当前资产可用性以本次清理回执补充。

[清理说明](../../maintenance/bench_weight_cleanup_20261020_2241/CLEANUP_REPORT.md) · [完成回执](../../maintenance/bench_weight_cleanup_20261020_2241/cleanup_receipt.json)
