# Bench Natural N：执行与恢复

科学合同见 PROTOCOL.md、training_registration.json 及 data/eval_manifest.json。用户已授权两臂各1500次更新；完成固定评价后停止。状态不是研究结论，最终以 COMPLETE.json 和 REPORT.md 为准。

所有命令在 `/mnt/e/MeshFlow` 下使用 `/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B` 执行。模块前缀为 `r2.pilots.bench_natural_n_recipe_v1.`。Windows 外壳用 `wsl.exe --cd /mnt/e/MeshFlow` 启动；后台监督器自身没有自动加训、重试或阶段派发逻辑。

- `workflow`：只读打印所有阶段状态，检查 PID 与进程创建时间，不能据历史 PID 杀进程。
- `workflow --launch train-b`：NATURAL_HIGH；先完成资源检查与根调度 GO，观察真实高N前后向峰值。
- `workflow --launch train-a`：CONTROL_OLD；仅在重新测量资源并留下余量后独立派发。
- `workflow --launch sample-p0/sample-a/sample-b`：固定 P0 高N8条，以及两终点各38条。P0旧30只读复用。
- `workflow --launch d0-p0/d0-a/d0-b`：全部84份新RAW完成后才能各执行一次官方D0及原GEO_C。
- `workflow --launch prepare-scores/score-raw/score-post/collect`：原评分器及CPU固定处理链，原逐输出不覆盖。

每个命名任务目录是一次尝试的独占凭据，重复launch仅返回状态，不自动重新执行。GPU工作前后留下资源观测；并发不得改变batch、噪声身份、精度或预算。

如进程停止，先读 `workflow/*/status.json`、`training/*/status.json`、attempt失败记录与calls.jsonl。只有已保存的完整安全边界且调用账本与检查点一致，才能明确恢复；未完成更新、已有未存更新或部分生成不得静默重放。工程问题先记录修复与版本，再决定是否有无需额外科学计算的恢复路径。不能把已登记失败擦掉。

旧模型、数据、RAW/POST、源码和评分结果只读。本轮只保留新终点及滚动恢复，不删除历史资产，不改维护默认，不commit/push。

## 已完成后的核验与查看

训练与固定推理已完成，不需要再启动训练。以下入口仅验证已有结果或整理已保存表，不运行模型：

```text
python -B -m r2.pilots.bench_natural_n_recipe_v1.workflow
python -B -m r2.pilots.bench_natural_n_recipe_v1.evaluation_closeout complete
python -B -m r2.pilots.bench_natural_n_recipe_v1.report_results tables
python -B -m r2.pilots.bench_natural_n_recipe_v1.report_results visual
```

查看 `gallery/index.html` 可逐页对比P0/A/B，含RAW、BASE_POST、CLEAN_FINAL_POST以及固定位置的正/侧截面。完整177字段在 `results/per_output.csv`；五轴实际图审在 `visual_review/summary_v1/actual_observations.csv`，标签由实际查看图片得到，不从分数推断。

两个新终点的 `checkpoint_step1500.pt` 含live模型、匹配AdamW和RNG；`generator_step1500.pt` 为推理导出。原起点仍在曝光实验B目录。精确路径与hash见 `MODELS.md`，禁止把另一权重与这些优化器状态拼接。N128..799是本轮独立Native包装合同，生产默认仍未更改；真实训练最高384，724仅被留出评价测试，不宣称全范围训练充分。

`source_archive/final/` 保存本轮实际源码及依赖的原评分实现；`code.diff` 仅列独立pilot的新增文件。小ZIP不含大权重、RAW/POST数组和全部原始调用日志；它用于审阅与CSV复算，完整本地目录才是恢复资产。已完成任务的状态文件与原始证据不得删除来触发重跑；任何后续训练/采样需要新的明确预算。
