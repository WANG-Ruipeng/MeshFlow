# 本轮交付索引

R2_USER_CONDITION_FINETUNE_PILOT_V1；固定任务已完成，停止。最终机器状态见 [completion.json](completion.json)。

首先阅读 [结果报告](R2_USER_CONDITION_FINETUNE_REPORT.md) 和 [完整图集](gallery/index.html)。本轮不晋级USER模型；后续训练路线仍是R2。没有自动Git提交/push、删除权重或增加实验。

## 权威结果与证据

| 内容 | 位置/数量 |
|---|---|
| 完整可恢复训练终点 | checkpoints/ 内两个 local1000/global7000.pt，含AdamW/RNG/输入流时钟 |
| 纯生成器 | inference/ 内两个 generator_step7000.pt |
| 仅用于恢复的中间点 | checkpoints/ 内两个 local0500/global6500.pt；未评价，不择优 |
| 训练日志与调用账本 | training/R2_CONT_OLD、training/R2_CONT_USER，各1000更新/8000F/8000B |
| 预检证据 | preflight/，8F/8B/0更新 |
| 冻结训练配对 | data/paired_plan.json、两臂物化缓存和索引；OT每臂8000次 |
| 冻结评价 | evaluation/registration.json、inputs/、queries/、checkpoints/绑定 |
| 全部RAW索引 | [raw_index.csv](evaluation/raw_index.csv)，84行（68新生成+16历史复用） |
| 每次输出与轨迹 | evaluation/outputs/，JSON收据指向对应RAW/轨迹/PLY |
| 完整数值 | [per_output.csv](evaluation/per_output.csv)84行、[per_condition.csv](evaluation/per_condition.csv)42行 |
| 数值配对 | [paired_noise.csv](evaluation/paired_noise.csv)84行、[paired_conditions.csv](evaluation/paired_conditions.csv)42行 |
| 新旧分开汇总 | [summary_equal_parent.csv](evaluation/summary_equal_parent.csv)6行 |
| 多样性 | [diversity.csv](evaluation/diversity.csv)42行，每条件一个噪声对 |
| 逐面对与逐点证据 | evaluation/score_details/，每输出pairs.json及vectors.npz |
| 完整几何分布/方向 | evaluation/geometry_distributions.csv、paired_direction_counts.csv、descriptive_summary.json |
| 完整PNG | gallery/colored、gallery/neutral，共56图，每图3模型×2视角 |
| 全量图审 | review/per_output_ai_proxy.json/.csv 84行，paired_ai_proxy.json/.csv 28行 |
| 图审覆盖/不确定性 | review/review_coverage.json、summary_ai_proxy.json/.csv；AI_PROXY、未盲审、非人评 |
| 最终执行审计 | audit/execution_audit.json（579检查PASS）、execution_checks.csv |
| 真实预算 | audit/ledger_summary.json、execution_ledger.csv |
| 终点/RAW/保护文件身份 | audit/execution_endpoints.csv、execution_outputs.csv、execution_protected_files.csv |
| 实际调度 | resources/、dispatch/、controller_events.jsonl、audit/runtime_summary.json/.csv |
| 数据分布与源风险 | audit/DATA_PAIRING_AUDIT.md及逐父/逐条件曝光、split_identity_audit |
| 数值独立复核 | audit/numeric_interpretation.md/.json（12642单元格0差异） |
| 实际执行源码 | source_snapshot/（12个冻结py及README）、code_manifest.json、code.diff |
| 实际旧评分源码 | source_external_snapshot/（11个小文件，含CF/FF/I与几何实现） |
| 后续CPU交付脚本 | audit/merge_visual_review.py、summarize_runtime.py、summarize_geometry.py、build_gallery_index.py、finalize_delivery.py |
| 新文档/源码/图像hash | [delivery_hashes.csv](delivery_hashes.csv)，大二进制hash由既有执行审计提供 |

原本的gallery/AI_PROXY_review_template.csv和manifest内PENDING是渲染时创建的空白模板状态；没有回写原渲染manifest，**实际完成状态以review/review_coverage.json和completion.json为准**。原数据准备目录中的training_authorized=false是上一阶段历史状态；本轮授权与实际执行见本目录registration.json及账本。

代码的新入口位于 `E:/MeshFlow/r2/pilots/user_condition_finetune_v1/`。旧R2正式训练入口仍保留历史数据/6000步约束。未自动改旧START采样默认入口。

## 安全回读与恢复

查看已完成阶段（0模型调用）：

```powershell
wsl.exe -e /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B /mnt/e/MeshFlow/r2/pilots/user_condition_finetune_v1/run.py --stage status
```

打开图集：`gallery/index.html`，无需联网，无外部资源。可按面板/部位过滤；全部28请求均嵌入，不隐藏坏例。

本轮所有授权更新/生成已经消费完成，**不要重新派发train/sample来“继续”这个已完成任务**。可恢复终点指参数、AdamW、RNG、输入游标和身份合同齐备；未来追加更新需独立新预算/注册。500步恢复点不能用来重放已执行501—1000步。

训练噪声、OT缓存、输入与RAW留在当前独立目录，未新复制旧大权重，也未制作重复ZIP。6份本轮权重约6.80GiB，其中4份full各1,565,078,571字节、2份generator各521,687,739字节。历史保护文件3846项hash未变。以后清理应另有明确范围；本轮没有删除这些证据。

## 报告脚本修正记录

- audit/summarize_geometry.pre_serialization_fix.py、summarize_geometry_serialization.diff/修复receipt：仅将NumPy计数转为JSON支持的原生int，没有重评分。
- 数值解读的.encoding_damaged及encoding修复receipt：保留原中文管道编码损坏证据，当前MD已用UTF-8修复。
- review合并保留原前8OLD审阅输入，使用另存的UTF-8文字恢复表修复中文notes，判断码/ID/PNG哈希不变；恢复说明与输入哈希见review_coverage。

冻结的模型执行源码和原评分数值未因此改变。报告内容与派生脚本是独立CPU结果整理，不是额外实验。
