# 已完成交付

入口：[完整报告](R2_USER_MIX_REPORT.md) · [全部图片](gallery/index.html) · [逐输出CSV](evaluation/per_output.csv)

实验完成：6000训练更新、12新增RAW、48评分、24PNG、48逐输出与36配对AI_PROXY审阅。10部分执行审计PASS，1059保护文件未变。实际结果和限制见报告，当前保留为候选，未切换默认模型。

- 主代码：E:/MeshFlow/r2/pilots/user_mix_from_official_v1/；[可恢复分阶段入口说明](RUN_AND_RESUME.md)。
- 新最终完整权重：checkpoints/R2_USER_MIX_step6000.pt；推理：inference/R2_USER_MIX_generator_step6000.pt。
- 新RAW：evaluation/outputs/；历史RAW、冻结评价C/Gaussian及查询按evaluation/registration.json原路径只读引用。
- 全部定量CSV：evaluation/；独立数值审阅和AI_PROXY逐例审阅：review/。
- 实际执行源码快照：source_snapshot/；评分代码小文件：source_external_snapshot/；辅助源码：source_addendum_snapshot/。
- code.diff、code_addendum.diff、code_manifest.json、source_addendum_manifest.json记录代码身份。
- [执行审计](audit/execution_audit.json)、[最终回执](completion.json)、[最终文件hash](FINAL_ARTIFACT_HASHES.csv)。
- 原audit/delivery_hashes.csv保留审计时快照；最终manifest追加覆盖此后完成的报告/审阅/回执，无需重跑模型或评分。

最终hash清单覆盖本目录文本、PNG和小代码证据，排除权重/NPZ二进制和manifest本身。二进制身份由训练、RAW、输入和评分原回执及执行审计核验。所有列表文件在落盘后重新hash验证。

当前任务已完成并停止；不要从5500恢复点重放本轮已完成更新。没有自动Git提交/推送。
