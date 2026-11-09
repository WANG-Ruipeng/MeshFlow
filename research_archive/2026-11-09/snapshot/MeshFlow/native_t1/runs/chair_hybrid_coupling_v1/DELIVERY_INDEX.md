# Chair Hybrid Coupling Pilot V1 交付索引

实验、固定评价和独立审计均已实际完成。两臂各1000更新；432条登记请求包括18条原始NA，414条适用请求由292次实际完整采样和122次历史raw缓存组成。总真实前向33082次，失败/未决均为0。结论为 **GEOMETRY_SIGNAL_WITH_TRADEOFF**；未切换默认模型，未追加实验。

|内容|文件|
|---|---|
|总报告与具体下一步假说|[CHAIR_HYBRID_COUPLING_REPORT.md](CHAIR_HYBRID_COUPLING_REPORT.md)|
|机器可读最终决策|[decision_summary.json](decision_summary.json)|
|实际调用总账|[accounting.json](accounting.json)|
|运行和恢复说明|[RUN_AND_RESUME.md](RUN_AND_RESUME.md)|
|原始用户任务|[task_prompt.txt](registration/task_prompt.txt)|
|冻结协议|[protocol.json](protocol.json)、[protocol.sha256](protocol.sha256)|
|输入与起点身份|[evaluation_records.json](registration/evaluation_records.json)、[start_identity.json](registration/start_identity.json)|
|两个固定终点身份|[checkpoint_manifest.json](checkpoint_manifest.json)|
|实际实现与hash|[code.diff](code.diff)、[implementation_hashes.json](registration/implementation_hashes.json)|
|实际评分代码|[scoring.py](../../../experiments/mf_h1_local0/chair_hybrid_coupling_v1/scoring.py)、[reporting.py](../../../experiments/mf_h1_local0/chair_hybrid_coupling_v1/reporting.py)|
|可恢复总入口|[run_all.py](../../../experiments/mf_h1_local0/chair_hybrid_coupling_v1/run_all.py)|
|受保护源文件前后核验|[source_hashes_before.json](registration/source_hashes_before.json)、[source_verification_after.json](registration/source_verification_after.json)|
|历史缓存源码身份|[cache_code_audit.json](registration/cache_code_audit.json)|

实际表格完整保留失败/NA及各父、噪声，不只提供汇总：

|表格|规模与文件|
|---|---|
|A/B/C逐输出|432行：[per_output.csv](paired_results/per_output.csv)|
|父等权及配对|[per_parent.csv](paired_results/per_parent.csv)、[paired_per_parent.csv](paired_results/paired_per_parent.csv)|
|验证主终点逐父/逐噪声|[primary_paired_parents.csv](paired_results/primary_paired_parents.csv)、[primary_noise_pairs.csv](paired_results/primary_noise_pairs.csv)|
|逐输出宽度|[width_per_output.csv](paired_results/width_per_output.csv)|
|逐曲线宽度|144条模型及解析参照曲线：[width_per_curve.csv](paired_results/width_per_curve.csv)|
|逐父双噪声对账|[noise_pairs_per_parent.csv](paired_results/noise_pairs_per_parent.csv)、[width_paired_curves.csv](paired_results/width_paired_curves.csv)|
|完整3×3矩阵|144个矩阵、1296个单元：[cross_matrix.csv](paired_results/cross_matrix.csv)|
|固定目标列方向审计|432个目标列：[target_column_audit.csv](paired_results/target_column_audit.csv)|
|覆盖分区及质量代价|[coverage_partition.csv](paired_results/coverage_partition.csv)、[quality_paired_summary.csv](paired_results/quality_paired_summary.csv)|
|真实响应逐步分解|24曲线、1200步：[response_per_step.csv](diagnostic_results/response_per_step.csv)|
|响应逐曲线/逐父|[response_per_curve.csv](diagnostic_results/response_per_curve.csv)、[response_per_parent.csv](diagnostic_results/response_per_parent.csv)|
|真实FP32诊断及dtype证据|[fp32_precision_records.csv](diagnostic_results/fp32_precision_records.csv)|

三个检查点的完整raw、51状态/50实际速度、交叉速度和原向量在 `evaluation/START/`、`evaluation/OT_BASE/`、`evaluation/OT_HYBRID/`。原评分全字段在各目录的 `scoring/scores/`。每检查点8曲线图：

- [START范数](diagnostic_results/START_norms.png) / [余弦](diagnostic_results/START_cosine.png)
- [BASE范数](diagnostic_results/OT_BASE_norms.png) / [余弦](diagnostic_results/OT_BASE_cosine.png)
- [HYBRID范数](diagnostic_results/OT_HYBRID_norms.png) / [余弦](diagnostic_results/OT_HYBRID_cosine.png)

独立核验：

- [账本与训练曝光审计](independent_audit/INDEPENDENT_LEDGER_EXPOSURE_AUDIT.md)、[audit_result.json](independent_audit/audit_result.json)
- [8000个实际配对输入](independent_audit/paired_input_audit.csv)、[每批归一核验](independent_audit/batch_normalization_audit.csv)、[父对象曝光](independent_audit/parent_exposure_audit.csv)
- [主终点/矩阵/距离向量独立重算](independent_audit/results_arithmetic_audit.json)
- [START轨迹独立审计](registration/start_diagnostic_audit.json)、[BASE轨迹独立审计](registration/base_diagnostic_audit.json)、[HYBRID轨迹独立审计](registration/hybrid_diagnostic_audit.json)

本轮所有真正模型调用、失败和缓存记录均保留。纯CPU独立审计读取器及报告文案曾作局部修正，原版本、修正diff及哈希分别保留在 `independent_audit/` 和 `reporting_corrections/`；正式实验代码和协议没有在冻结后改变。补充脚本diff与哈希另列，不覆盖冻结的 `code.diff`。没有额外交接压缩包或起点大权重副本。
