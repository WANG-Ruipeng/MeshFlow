
## 2026-10-07 当前决策：回到原始 START，结束 CAPPED 探索

按用户明确要求，以原始 Chair HYBRID START / total5000 继续作为 baseline。正式默认 profile 一直为 chair-hybrid，已实际核验解析及一次 CPU strict load；零模型前向、零训练、零生成。START 两条硬链接路径及完整 AdamW/RNG/输入流状态均保留。

已删除 chair_control_feedback_auto_v1、chair_capped500_gradbudget_v1 和 chair_capped500_half_v1 的22份实验权重，以及两处正式 OT 派生缓存的16,000个NPZ，共33,426,755,776逻辑字节（约33.43GB）。OT尝试日志、训练计划、小型测试缓存、原始RAW、冻结条件/噪声/查询、反馈池、全部评分/报告/CSV/图集/源码和审阅ZIP保留，43,405个保留文件前后hash一致。

相关历史代码留作评分依赖和证据，不自动运行旧恢复入口，也不因权重/缓存缺失而重训或重生成。原报告中的旧文件存在状态以本轮删除清单为准。官方权重、Geo4000、START、旧显式FM/JEdge5及其它既定保留checkpoint不在本轮删除范围。没有修改正式代码或commit/push。

详见 [清理报告与完整账本](native_t1/maintenance/capped_retirement_20261007/CLEANUP_REPORT.md)。
