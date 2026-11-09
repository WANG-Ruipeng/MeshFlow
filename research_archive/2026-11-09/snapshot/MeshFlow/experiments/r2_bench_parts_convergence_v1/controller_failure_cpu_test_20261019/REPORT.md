# 调度元数据与安全退出 CPU 验证

本次只验证工程修复，不训练、不生成、不调用模型，不重试任何科学任务。未修改正式训练器。已直接测试修复后的生产 `controller.read`、`recovery_precheck`、`Driver.progress_peak` 与 `Driver.drain_after_failure`，18 项检查通过。

## 两个故障与对应修复

1. 原 `exists()` 后立刻读取 `status.json` 存在检查与读取之间的竞态；持续更新的状态文件短暂不可见时，不能直接判定训练不存在或产生孤儿状态。现在生产读取函数在 2 秒内仅对指定临时文件访问错误、JSON 解码错误及字符解码错误重读；读到后的 arm、step 等完整性检查保持严格。此重读没有重启任何科学任务。
2. 原异常处理发送 SIGTERM 后立即退出父进程，发送信号并不代表子进程已保存。现在先停止新派发，训练子进程只接收一次原有更新边界停止信号，已运行采样自然结束；父进程保活直到所有自有子进程实际退出，再返回原失败。初始/末尾状态记录或完成记录失败不会提前中断这一等待。

## 直接生产方法实测

最终证据：`production_1792346165051859428/results.json`。生产源码测试前后 SHA 相同，精确 hash 已写入结果文件。

- FileNotFound、PermissionError 和短暂坏 JSON 读取后恢复；EIO 立即抛出，无盲目重试。
- 持续缺失状态等待约 2.09 秒后，已有训练账本的目录判为不可安全续跑；没有把缺失状态当作新任务。
- 错误 arm、负数 step、缺失 step 立即被拒绝，未通过重读绕过完整性检查。
- 真实 `progress_peak` 在模拟状态不可见窗口后正常返回。
- 两个普通 CPU 子进程中，模拟训练进程收到一次 SIGTERM，在延迟完成一个模拟更新、写入小检查点和安全状态后再次延迟退出；模拟采样进程未收到停止信号，自然结束。生产 drain 等待二者实际退出，记录正确安全游标。
- 第二组两个 CPU 子进程同时注入调度侧 `write` 的 PermissionError、`update` 的 EBUSY，以及一次等待中的 KeyboardInterrupt；生产 drain 仍等待两个子进程退出，保留记录异常清单，没有重复发送信号或再次派发。

## 解释边界

这些是控制流和生命周期测试，使用几十字节的模拟检查点；不宣称测试了真实大权重写入性能或 CUDA 故障恢复。真实训练器仍负责在更新边界完成模型、AdamW、RNG 的检查点保存与恢复校验。持续故障会阻止新派发；元数据重读不会授权任何已失败科学任务重跑。

测试入口：`test_production.py`；另有早期独立机制 fixture `check.py`（9 项通过）。所有 fixture 与报告均位于独立实验子目录。

## 非整千步恢复入口集成复验

直接调用最终生产 `Driver.__init__`、`train_target`、`cmd`，以状态 fixture 替代真实训练进度，并将 `wave` 替换为只捕获命令；`Popen` 被禁止。6 个案例通过：COMMON 5250 恢复使用 `resume_train`；COMMON 4000 恢复使用原 `train`；ABSENT COMMON 使用原 `train` 且不带恢复/分叉参数；ABSENT PLAIN/SEAM 均使用原 `train` 并从同一 COMMON12000 分叉；SEAM 14250 恢复使用 `resume_train`；已到达目标的任务不产生新派发。真实生成的 identity 包含新包装器 hash。

路由证据：`dispatch_routing_1792346438406704825/results.json`。

在相同最终源码上重跑全部 18 项元数据与 drain 检查，仍全部通过：`production_1792346445963952966/results.json`。

最终生产 hash：

- `run_all.py`: `66519cb5a4ef256c36b13df06a4744e2a7e864b0930133e63b4d926434965ffb`
- `controller.py`: `de67cb3c9b3bd0a67033a9a93f67df18cfe2009da7555d15fd4d422253451d2d`
- 原 `train.py`: `4eddcfbd7da51860a58033656210b58274248227560017d45a23b0f80f979fd3`
- `resume_train.py`: `ae932fa1cec731c0ddfd52e04b4939e6053a76474706cd287ad96fcccad1a2a1`

本测试仅核查调度集成，不执行包装器或恢复训练，也不构成第二次科学恢复授权。
