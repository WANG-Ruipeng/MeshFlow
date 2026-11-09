# Session C 无人值守接续

已启动一次性的本地 WSL 守候进程，PID 2759。启动命令退出后进程仍独立运行；不是依靠聊天窗口再次被唤醒。

- 状态与心跳：`evaluation/automation_status.json`（等待期间每 30 秒更新）。
- 日志：`evaluation/automation.log`；正式 GPU 阶段另有 `evaluation/gpu_worker.log`。
- 冻结执行计划：`registration/eval_automation_plan.json`，绑定 80 个源码、协议与输入文件。
- 当前只等待 B 的 TRAIN_READY；等待期间无模型、无 CUDA、无前向。
- 三个 step1000 终点全部核验并通过 B 统一入口严格加载后，取得共同 GPU 锁，使用一个 GPU worker 执行 96 条 Euler50（4800 NFE）和 24 次 teacher。CPU 单进程评分可与后续推理重叠。
- 正式完成后生成 MVP48_REPORT.md、results.json、per_output.csv、per_parent.csv、teacher_validation.csv（及 teacher.csv 同步副本）、固定视图与账本，随后停止。
- 身份不符、数值失败、资源不足或评分异常时保留证据并停止；不自动重试、补 seed、训练或切换模型。中断请求不能伪记为 0 次前向。

必须让电脑保持开机且不休眠，并保持 WSL 运行。关机、重启或关闭 WSL 会中断本地任务；此程序不设置重启恢复或系统级自动唤醒，也不更改电源设置。

如需主动停止，可在评价目录创建 automation_stop.json，例如内容 {"reason":"user requested stop"}；进程在任务边界停止，不终止正在进行的 CUDA 调用。停止后不会自动重跑。不要重新启动第二份守候程序。

自动报告保留描述性结果和失败分母，不把 8 个父对象解释为统计定论，不自动判定充分收敛，也不构造偏向 Full 的总分。
