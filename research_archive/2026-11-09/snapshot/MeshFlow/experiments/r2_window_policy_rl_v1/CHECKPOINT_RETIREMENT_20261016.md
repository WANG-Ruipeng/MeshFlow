# 当前检查点生命周期（2026-10-16）

R2_WINDOW_POLICY_RL_V1 已完成并收口为当前配置的负结果。用户随后授权清理中间权重。

- 已删除：training/batch_000..062/update/checkpoint.pt，共63份、605,818,757字节。
- 保留：batch_063终点（SHA256 bbda0ccaa1c86160b0574a6071a95d5eacc4e38d5db1051b9a26cae85ca26603）、FULL_S6000基座、官方模型及所有输入、buffer、RAW/POST、评分/图审、日志、代码与回执。
- 可复核：保留结果与逐例配对；终点推理依赖完整，但本次维护不授权新生成。
- 已退役：中间模型逐批重放/优化器恢复，以及原训练控制器全链校验/恢复。不要运行 `resume.ps1 -Phase train` 期待恢复；不要因旧路径缺文件自动补训。
- 原报告、README_RESUME.md、final.json与审阅ZIP未改，描述的是清理前的历史完成状态；本文件补充当前存储事实。固定64批合同结束，终点含优化器状态不授权续训。

[收口报告](../r2_window_policy_rl_closeout_20261016/NEGATIVE_RESULT_REPORT.md) · [逐文件清理回执](../r2_window_policy_rl_closeout_20261016/cleanup_receipt.json)
