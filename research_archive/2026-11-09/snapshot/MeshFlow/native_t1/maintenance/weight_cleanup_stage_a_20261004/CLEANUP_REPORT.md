# 第一档旧恢复点清理完成

2026-10-04，经用户批准，仅执行前次评估中建议优先清理的 A_STALE_RECOVERY。实际删除 7 份旧恢复 checkpoint，共 10,950,114,685 bytes（10.95 GB）。没有递归删除，没有清理第二、第三档。

|目录（相对工作区）|删除的文件|登记步数|
|---|---|---|
|native_t1/runs/chair_domain_control_v1/checkpoints/|Full_latest.pt、Geo_latest.pt、T1_latest.pt|phase2800 / total3800|
|native_t1/runs/control_mvp48_budget_single_v1/checkpoints/|Full_latest.pt、Geo_latest.pt、T1_latest.pt|total4800|
|native_t1/runs/control_mvp48_budget_single_v1/checkpoints/|T1_fit2_probe_latest.pt|probe800|

这些文件是较早的不同训练状态，不是字节相同的重复副本。清理后无法从这些旧步数恢复或重新采样。三臂固定终点、固定中间评价状态及 fit2 固定权重均保留。

下一步指定起点 native_t1/runs/chair_domain_control_v1/checkpoints/Geo_phase3000_total4000.pt 原样保留，包括完整优化器状态。完整文件 SHA256 仍为 190a16e2aec96a1f84f63460b5f97b00e23499338102f82e7530218ecb185ebf。

删除前后完成 10,773 个受保护文件的完整 SHA256 校验，变化 0；清理范围外 65,630 个原有文件的大小、修改时间和类型一致，意外缺失 0。保护范围包括其余主工作区权重、原 Chair 数据、冻结训练/评价输入、raw、评分与分析代码、逐例数据和报告。归因交付包继续保留。

原 checkpoint_manifest.json、latest.json、协议和历史报告未修改；其中指向本次已删文件的路径是历史记录，应结合本目录独立删除账本解释，不应将原清单改写为从未存在这些状态。部分历史全文件存在性检查会遇到已明确退役的路径。

实际删除采用 PowerShell Remove-Item -LiteralPath，逐项限定在工作区内的七个确定文件；删除前核对类型、文件长度与完整 SHA256。两份运行状态均已完成，执行前未发现活动实验进程。

账本：deletion_manifest.json 保存路径、大小、SHA256 和原身份；deletion_receipts.jsonl 保存逐文件实际删除回执；verification.json 保存核验结果。inventory_before.json.gz 和 protected_before.json.gz 是压缩的清理前核验记录，不包含大权重副本。

清理后仅向 WORKSPACE.md 追加本次清理记录，文档前后哈希另存 workspace_note_update.json。上述零变化核验在追加维护记录之前完成；原实验文档未修改。

本轮模型加载、前向、训练更新、生成均为 0。没有启动 NATIVE_CHAIR_HYBRID_COUPLING_PILOT_V1，没有改默认模型，没有 Git reset/clean/commit/push。
