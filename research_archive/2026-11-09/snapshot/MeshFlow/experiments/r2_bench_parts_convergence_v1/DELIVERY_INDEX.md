# Bench多部位条件训练：最终交付与恢复索引

共同终点global21000，执行完成；停止原因为预算封顶而非SEAM平台。完整结论见[REPORT.md](REPORT.md)。

|交付|位置|
|---|---|
|冻结协议|[registration.json](registration.json)|
|数据与训练日程|[manifest](data/manifest.json)、[input_schedule](data/input_schedule.json)|
|48请求及固定噪声身份|[evaluation_inputs/manifest.json](evaluation_inputs/manifest.json)|
|固定loss探针|[fixed_loss_probe/manifest.json](fixed_loss_probe/manifest.json)|
|完整学习曲线|[loss.png](learning_curves_final/loss.png)、[loss.csv](learning_curves_final/loss.csv)、[部位/t日志](learning_curves_final/training_by_part_time.csv)|
|共同停止判定|[step21000](dispatch/decisions/step21000.json)|
|COMMON12000完整图集|[打开](gallery/NATIVE_S12000_full/index.html)|
|同预算15000双臂图集|[打开](gallery/NATIVE_S15000__SEAM_S15000_full/index.html)|
|最终21000双臂表面／位置截面图集|[打开](gallery/NATIVE_S21000__SEAM_S21000_full/index.html)|
|终点192条真实阶段观察|[review.jsonl](visual_review/NATIVE_S21000__SEAM_S21000_full/review.jsonl)、[图审说明](visual_review/NATIVE_S21000__SEAM_S21000_full/REVIEW.md)|
|终点逐输出124字段|[per_output.csv](evaluation/tables/NATIVE_S21000__SEAM_S21000_full/per_output.csv)|
|终点逐父|[per_parent.csv](evaluation/tables/NATIVE_S21000__SEAM_S21000_full/per_parent.csv)|
|终点完整配对|[paired_per_output.csv](evaluation/tables/NATIVE_S21000__SEAM_S21000_full/paired_per_output.csv)|
|面对角色与删面|[illegal_pairs_by_role.csv](evaluation/tables/NATIVE_S21000__SEAM_S21000_full/illegal_pairs_by_role.csv)、[deleted_faces.csv](evaluation/tables/NATIVE_S21000__SEAM_S21000_full/deleted_faces.csv)|
|12000／15000／21000完整数字、分母和来源hash|[摘要](report_support/numeric_snapshots/20261019T105638_250735Z/NUMERIC_SUMMARY.md)、[manifest](report_support/numeric_snapshots/20261019T105638_250735Z/manifest.json)|
|全部312份RAW/POST计算账本|[账本](audits/execution_accounting_v1/snapshots/20261019T105654_981140Z/EXECUTION_COUNTS.md)、[完整JSON](audits/execution_accounting_v1/snapshots/20261019T105654_981140Z/execution_counts.json)|
|资源／墙钟原回执汇总|[summary](report_support/runtime/summary.json)、[逐子进程](report_support/runtime/completed_children.csv)|
|三次恢复与重放核验|[recovery_compute_accounting.json](audits/worker_write_race_20261019/recovery_compute_accounting.json)|
|原评分源码快照|[scoring_source_snapshot](scoring_source_snapshot)|
|实际实验代码|[r2/pilots/bench_parts_convergence_v1](../../r2/pilots/bench_parts_convergence_v1)|

## 权重

这些是本地实际文件，没有上传Git。完整恢复文件包含生成器、匹配AdamW、RNG及SEAM小头／小头优化器。纯生成器导出不应被误当作可独立恢复训练或替代小头的完整包。

|文件|SHA256|
|---|---|
|[PLAIN完整step21000.pt](training/PLAIN/step21000.pt)|c078b20e01b1afa7dde272971f4aff4c04fa5e746c3b6f212694a0b22f96bdd3|
|[SEAM完整step21000.pt](training/SEAM/step21000.pt)|7c32b114724e94a08a7a9ddbc20ee26e7d92e71e62639103b0ca4462724f24bc|
|[PLAIN纯生成器](training/PLAIN/generator_step21000.pt)|36a8d84a6ecc97b90c0deda625e77534a43e6dbc1fe45c47c7acada74660f0fa|
|[SEAM纯生成器](training/SEAM/generator_step21000.pt)|98b5fb8e562c59e18a40ef40f0cbc7497a5e4088d3bb36cfb35a4b2bcc70acc9|

[PLAIN完整状态核验](dispatch/final_checkpoint_PLAIN.json)与[SEAM完整状态核验](dispatch/final_checkpoint_SEAM.json)均PASS。不要把纯生成器和另一个阶段的优化器、小头或RNG拼接。

## 可恢复入口与完成后的行为

原总入口为`python -m r2.pilots.bench_parts_convergence_v1.run_all`。不加`--execute`只打印计划；已完成任务的`--execute --resume-controller`读取现有完成回执，不自动延长到21000以外。不要删除回执以强迫重跑。

本地WSL只读计划示例：

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- env PYTHONPATH=/mnt/e/MeshFlow:/mnt/e/MeshFlow/r2/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_parts_convergence_v1.run_all --root /mnt/e/MeshFlow/experiments/r2_bench_parts_convergence_v1
```

数字摘要和实际计算账本可用独立CPU入口重新读取现有记录：`report_support/build_numeric_summary.py`、`audits/execution_accounting_v1/build.py --final`。它们不训练、不采样、不重评分。图审合并保留原分片，禁止用未实际查看的新图片复用旧标签。

原`REPORT_DRAFT.md`和早期交付草稿保留为历史，不是当前状态。调度器的`report_ready.json`记录数值阶段完成；根目录`completion.json`记录实际图审与最终报告收口。此索引不授权新实验、额外训练、删除保留资产、切默认、commit或push。
