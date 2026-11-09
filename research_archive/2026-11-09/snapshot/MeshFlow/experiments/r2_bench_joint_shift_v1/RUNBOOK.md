# Bench共同平移1500步实验

主入口：`python -B -m r2.pilots.bench_joint_shift_v1.workflow --execute`。只能在CPU门禁与注册冻结后启动一次；不要重复启动已有workflow状态。

准备顺序：`prepare` → `cpu_checks` → `register` → `training --check-cpu` → `eval_plan`。所有模块前缀均为 `r2.pilots.bench_joint_shift_v1`。CPU检查不调用模型。

进度见 `workflow/state.json`、`training/JOINT_SHIFT_SEAM/status.json`，逐步loss见同目录 `training_log.jsonl`。监督进程应持续持有前台exec会话，不能切换WSL进程造成中断。

只读恢复检查：`python -B -m r2.pilots.bench_joint_shift_v1.resume`。只有记录为INTERRUPTED_SAFE且完整检查点游标一致时，才允许同模块加`--execute`精确继续；FAILED/OOM/部分更新不允许自动回滚。完成后只读检查，不会重复训练。

自动流程结束于数值和图集完成；仍需实际图审、报告和独立完成收据。固定预算结束后不自动扩训练，不切默认，不commit/push。
