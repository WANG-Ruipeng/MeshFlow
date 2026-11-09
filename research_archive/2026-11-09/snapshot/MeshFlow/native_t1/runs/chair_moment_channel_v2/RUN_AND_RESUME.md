# V2 实际执行与恢复

总入口：`experiments/mf_h1_local0/chair_moment_channel_v2/run_all.py`。

在 WSL 中执行：

```sh
cd /mnt/e/MeshFlow
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home/warpwang/miniforge3/envs/meshflow/bin/python -B -m \
experiments.mf_h1_local0.chair_moment_channel_v2.run_all --resume
```

入口复用已完成阶段，核验被冻结的协议、旧资产和实际 raw 身份。GPU worker 先获取 `/tmp/wrp_meshflow_jedge5_diag_gpu0.lock`，再初始化 CUDA；不得删除锁或终止无关进程。

两臂各 1000 更新，每 200 步保存独立完整恢复点。恢复需同时匹配参数、AdamW、RNG、训练游标与调用账本。安全中断只能从已保存且覆盖所有已执行更新的恢复点继续；不得重放未被恢复点覆盖的更新来凑数。`controller/`、`stages/`、`training/` 和 `ledger/` 分别保留阶段状态、工作进程状态、逐更新证据和每次实际调用的 BEGIN/RETURN/FAILED。

日志、文件或监控故障的恢复必须保留原故障、回执、源码差异与已消耗预算。未决调用不会自动重试。数学、数值或科学配置故障不得串接为同一训练版本。

`--through prepare`、`--through precheck`、`--through replay` 是依赖完整的分阶段入口；不会自行改变预算。完整固定任务完成后停止，默认模型保持原状。

旧实验目录只读，本目录不包含复制的旧大权重。本轮源代码和实际评分依赖保存在 `source_code/`，冻结版本与保护资产哈希在 `registration/`，完整调用汇总在 `accounting.json`。
