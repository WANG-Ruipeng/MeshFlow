# 2026-10-07 权重与派生缓存退役

用户已要求结束本轮反馈/CAPPED探索，并统一回到原始 Chair HYBRID START baseline。

本目录全部实验 PT 已清理，JSON 身份记录保留原样。原始 RAW、条件/Gaussian/查询、反馈池、报告、逐例 CSV、评分及源代码、图集和最终审阅 ZIP 均保留。

反馈探索与复现实验中 streams 下的正式派生 OT NPZ 已清理；*.attempt.json 和训练计划保留，避免旧入口静默重跑。小型首批门禁测试缓存保留。HALF 本身无 OT 缓存。

历史报告中的“权重保留”及全文件审计 PASS 是执行当时的状态；当前删除清单以 [清理报告](../../../maintenance/capped_retirement_20261007/CLEANUP_REPORT.md) 为准。旧恢复/训练/半幅构建命令不再是可直接恢复的承诺。不得因缺文件而自动重训、重生成或恢复该路线。

正式 baseline 为 native_t1/checkpoints/chair_hybrid_start_total5000.pt，SHA256 d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80；正式 CLI 默认 chair-hybrid 已验证。旧原件、官方权重及其它既定保留模型不在本轮删除范围。