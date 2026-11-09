# 固定实验入口与恢复边界

实验：CHAIR_EARLY_ALIGNMENT_SCHEDULE_40_V1。源码 E:/MeshFlow-Control，结果为本目录。原 E:/MeshFlow、历史权重、数据与结果只读；START不自动切换。当前最终状态：COMPLETE；训练、48份采样/评分、逐图审阅和报告均完成。以下保留实际入口和恢复合同用于复核，不授权再次运行或扩展实验。

## 入口

WSL Ubuntu 的 Python：`/mnt/e/MeshFlow-Control-artifacts/venv/bin/python`。
总调度器：`/mnt/e/MeshFlow-Control/tools/run_schedule40.py`。
默认 --out 即本目录；支持 --stage prepare / train / sample / evaluate / all。

本轮 prepare/train/sample/evaluate 和独立补充分析均已完成，最终注册实验进程为0（audit/final_idle_check.json）。控制器运行期间持有 `/tmp/wrp_meshflow_jedge5_diag_gpu0.lock`；禁止另起训练副本。以 `audit/active_workers.json`、各 `training/<segment>/status.json` 及 `audit/controller.jsonl` 查看真实状态。

本轮按顺序实际执行了既定 sample、evaluate。每个阶段将已完成的任务依据完整receipt跳过，不增加arms、更新、噪声或输出数量。`all` 不包含下述I补充、最终CPU审计、实际视觉审阅和报告，这些本轮也均已完成，收据见最终报告。

```bash
python -B /mnt/e/MeshFlow-Control/tools/run_schedule40_dispatch_r1.py --stage sample
python -B /mnt/e/MeshFlow-Control/tools/run_schedule40.py --stage evaluate
python -B /mnt/e/MeshFlow-Control/tools/score_schedule40_conflict.py --stage score --workers 2 --out /mnt/e/MeshFlow-Control-artifacts/experiments/chair_early_alignment_schedule40_v1
python -B /mnt/e/MeshFlow-Control/tools/analyze_schedule40.py --final
python -B /mnt/e/MeshFlow-Control-artifacts/experiments/chair_early_alignment_schedule40_v1/audit/verify_final_checkpoints.py
python -B /mnt/e/MeshFlow-Control-artifacts/experiments/chair_early_alignment_schedule40_v1/audit/summarize_fixed_panel.py
```

实际执行使用上述venv完整路径，且PYTHONDONTWRITEBYTECODE=1、OMP/MKL线程按各阶段登记设置。CUDA初始化前CUBLAS_WORKSPACE_CONFIG=:4096:8；确定性、精度、batch与采样定义由冻结实现固定。

## 恢复合同

- 上限2个GPU worker，分别拥有模型、优化器、RNG和输出目录，逐个加载并测量资源。
- 训练每100步原子保存完整 `latest.pt` 与receipt；不可变终点在 `checkpoints/`，纯生成器导出在 `inference/`。
- 若双worker中的任务实际发生OOM，调度器按用户授权全局降为1；同一受影响波次的每任务至多一次串行恢复，已完成重放最多99步。局部失败调用与完整重放另记，不隐藏成本。
- 单worker OOM、数值失败、未知中断、缺完成receipt或哈希不符不授权静默重跑。保留证据，审核具体失败；不得改batch、精度、loss或多训练以绕过失败。
- 文件存在不是完成证明：查completed.json、dispatch receipt、最终账本和真实权重审计。缺失是PENDING/BLOCKED，不从报告名称推断成功。

## 交付检查

要求13,000个唯一物理更新、三条6000逻辑路径，EARLY/LATE各4000个B clean样本，无新增教师前向；实际输入配对与完整fork状态通过。

要求48个采样与逐输出主评分记录、48个I记录（保留不确定性）、全部8父两套多视角图集；实际AI视觉审阅另记，不用图集存在冒充已审阅，也不称人类评分。

数值表包括 evaluation/per_output.csv、paired_parent_deltas.csv、I补充表及独立分层汇总。新增分析源码和hash见 source_code/experiment_audit/、audit/postfreeze_analysis_hashes.json；训练/评分源码见source_code及code.diff。最终结论只在全部执行和审阅后写入EXPERIMENT_REPORT.md。

## 已执行的调度边界修正

训练正常运行、无OOM时，发现原控制器的parallel_exposed保留了历史并发资格：一个后来独跑的任务OOM可能仍被允许重试。原科学源码与原控制器不改写。

本轮在只剩LATE后，由独立guard_serial_tail.py暂停经过PID/starttime/父子关系校验的控制器318；GPU worker继续原预算，守卫每5秒单独记录资源。LATE成功退出后恢复控制器正常收尾；若串行失败则禁止恢复派发并记BLOCKED。实际信号和结果见audit/serial_tail_guard。

后续采样改用独立run_schedule40_dispatch_r1.py。它先固定本轮退出集合，再撤销独跑幸存者的并发恢复资格；同轮双OOM仍最多各恢复一次，之后串行OOM不得重试。6项CPU mock回归PASS；训练、采样、评分的科学函数保持原定义，原63文件hash核验通过。独立源码、精确diff、hash与回归证据见audit/dispatch_r1。

这不是新增训练、额外采样或已发生OOM的恢复；最终实际调用与重放数量仍以完整账本为准。
