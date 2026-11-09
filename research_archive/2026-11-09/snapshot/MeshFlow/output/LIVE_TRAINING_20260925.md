# 训练已按用户要求停止

2026-09-25 10:15，北京时间。

**当前没有继续运行的官方训练队列。** 用户在白天研究总结后选择立即停止，保留20000步完整检查点。controller、driver、trainer与此页原自动监视器均已退出；此页不再自动刷新。

- 官方seed10最后完成23050步；完整可恢复检查点为20000步。
- 保存后的3050次更新没有完整恢复权重；已有训练日志、探针和未保存部分的trace仍保留。
- 20000步检查点在停止前后SHA256和大小一致，含模型、优化器、调度器、EMA、RNG及抽样状态。
- seed11未启动，最终自动采样/评测未启动；未完成预定2×50000预算。
- 所有已完成A侧实验及封存包保留，没有启动新训练。

原机器队列状态为FAILED，原因是本次用户请求的SIGINT所导致的非零退出；原始中断记录保留，不能解读成数值发散或程序自行崩溃。

[白天研究总结与取舍](DAYTIME_REVIEW_20260925.md) · [夜间历史结果](OVERNIGHT_20260925.md)

停止证据：`E:\MeshFlow\.worktrees\a-layout-value-20260924\output\a_official_user_stop_20260925_01\STOP_CONFIRMED.json`

恢复检查点：`E:\MeshFlow\.worktrees\a-layout-value-20260924\output\a_system_official_closed16_20260925_01\training\seed_10\invocations\0001\step00020000.pt`

SHA256：`059d2708a4620310bd5730d0c3a9de4e0341a6c9967c688d1afca41212babf5a`