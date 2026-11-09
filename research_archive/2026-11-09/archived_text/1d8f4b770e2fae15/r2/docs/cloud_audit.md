# GitHub 训练结构核查（2026-10-09）

核查远端：<https://github.com/WANG-Ruipeng/MeshFlow>，分支 `master`。

通过实际执行 `git ls-remote origin refs/heads/master` 核实，远端与本地 HEAD 同为：

```text
8961b7a41fc153b5b45f0c8903370c6bfd1cf04f
```

该版本已经包含根目录的原版训练、`native_t1/` 条件扩展、START 采样和保持 C 的 CPU 后处理；不包含 R2。

R2 原工作目录为 `E:\MeshFlow-Control`。检查时该目录虽然有 Git 初始化信息，但没有提交、分支或 remote；此前 R2 通过本地专用实验工具运行，旧主 README/CLI 仍围绕早期 A/B/C。这是云端看不到 R2 训练结构的原因。

根据用户选择，R2 已整理到现有仓库的 `r2/`，目录结构见 [训练结构](training_structure.md)。首次整理只完成本地代码、文档和 CPU 验证；随后用户于 2026-10-09 明确要求提交并推送 R2。本次发布包含 R2 源码、测试、来源与验证文档及仓库首页入口。上述 `8961b7a` 记录保留为迁移前的核查快照，发布版本以包含本目录的 Git 提交为准。

权重、冻结数据/OT、RAW 和报告继续留在外部资产目录，不随源代码入 Git。
