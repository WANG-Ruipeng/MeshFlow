# 有界训练验收建议

本次只读核查范围为 T1 的 MVP48 条件微调 step1001–2000。未运行模型、GPU、OT 或 optimizer 更新，未修改冻结 .py。这不是整轮实验完成结论。

## 当前已实查结果

T1_to2000 为 SEGMENT_COMPLETE：CSV 有1000条连续且无重复的更新；append-only events 中 UPDATE_BEGIN 与 UPDATE_VALIDATED 各1000条。三者的 step 与 batch SHA 逐条一致，且等于共享输入登记。每步 optimizer 计数增加1，forward/backward attempted/returned 均增加8；终点为1000次新增更新和8000个微批前向/反向。loss 均有限，无失败事件。CSV 的训练和输入准备时间求和与 summary 相同。T1_step2000.pt 实际文件 SHA 与 sidecar 匹配，variant、step、counts、protocol 哈希与完成 summary 相同。详细数值见 training_T1_to2000_readonly.json。

当前 T1_to2000 训练段：worker_train_seconds=1084.645121858；worker_OT_seconds=24.033393381；worker_save_seconds=85.427719596；load_seconds=16.874360029；worker_wall_seconds=1238.258084212。计时只代表这个实际 worker，不是独立、无竞争的设备基准。

## 最终固定预算的交叉验收

1. 对 T1、Geo、Full 分别要求 logical_step=5000 且新增4000更新；probe 要求独立 logical_step=1000 且新增1000更新。不要将源历史1000步或官方预训练混入本轮更新。四臂最终 counts.optimizer_updates 相加必须13000；forward_attempts/returns、backward_attempts/returns 各104000；optimizer_step_attempts=13000。不得累加各阶段 summary 里的累计 counts，否则会重复计账。
2. 以每臂 append-only 事件与 CSV 按 (arm,logical_step) 核对唯一、连续的已验证更新，再与最终 summary 和 checkpoint.metadata.counts 相等。检查 UPDATE_BEGIN 是否存在未被可靠 checkpoint 覆盖的尝试、FAILURE、重复 step 或非有限参数；不允许重放未可靠保存的更新。若 optimizer 返回后参数验证失败，应如实保留 counts.optimizer_updates，不仅数 validated 行。
3. 对固定2000/3500/5000与 probe500/1000 核对文件实际 SHA、sidecar 路径/variant/logical_step/protocol/source_checkpoint/model SHA。最终主5000/probe1000必须 full_recovery_state=true 并有 AdamW/RNG/stream；中间固定状态允许 model-only，继续训练必须来自已验证 full latest。主三臂恢复模式 RESTORE_ALL，probe 单独 FRESH_PROBE。独立 CPU 严格 load 验收不得运行额外 forward/更新。
4. 运行主控 execution_budget_audit 并检查 errors 为空；再确认28个评价 stage 的真实分母、生成 attempt/return、50NFE 与 teacher 一次 forward 对应，缺失历史列只标 NOT_AVAILABLE。检查全部文件哈希，不能凭文件名或 rc0 认为完成。
5. 最后同时验收 run_state、execution_budget_audit、reporting_status 和实际交付文件。当前代码在所有计算 stage 完成后先写 COMPLETE，再运行 reporting；报告失败仅写 reporting_status=STOPPED，因此不能仅凭 run_state.status 声称全部交付完成。

## 时间字段的准确范围

- 每行 train_seconds 由原 optimizer_update 内部计时：包含 zero_grad、Full 目标准备、8次 forward/loss/backward、clip、AdamW、CUDA synchronize 及参数有限性检查。它不是纯 GPU kernel 时间；函数前置参数/optimizer 断言及外层日志不包含在内。
- OT_seconds 实际覆盖整个 next_effective_batch：8样本的 Gaussian/t/排列、原 CPU OT、GT/接口标签检查与样本哈希；不能解释为纯 OT 算法时间。worker_OT_seconds 是当前 worker 这些行的和。
- worker_save_seconds 为 save_checkpoint 返回的 save_and_verify_seconds 求和：包含 CPU 状态拷贝、模型/optimizer/RNG 哈希、序列化、文件哈希与 CPU 恢复验证，固定点还包含参数变化统计；不包含该函数末尾 gc、外层 latest 指针/事件写入等全部尾部成本。
- load_seconds 覆盖当前 worker 模型/AdamW/RNG/stream 恢复及源 checkpoint 文件哈希。worker_wall_seconds 从预备 stream、初始资源检查之后开始，含锁等待、加载、训练/准备/保存、期间日志/资源采样等，但不覆盖进程导入、最初准备及函数 finally 的最终 monitor 停止/清理/资源快照。
- 主控 STAGE_END.elapsed_seconds 覆盖子进程启动到退出，范围更宽；不能与 worker_wall_seconds 重复相加。剩余差值是不同范围的开销，不应伪造分项归因。
- allocated/reserved 峰值是每 worker 加载完成后重置的训练阶段峰值，包含恢复驻留模型/AdamW和后续 workspace；二者有包含关系，不相加。nvidia-smi global 峰值含其他进程；process 为 null 表示 WSL 不可得，不填0。

## 安全恢复后的分段成本

latest 保存的是累计训练计数；新 worker 恢复后计数继续增长，但 OT_counts 和 worker_*seconds 从该 worker 重新开始。同一 target 安全中断再恢复会覆盖同名 _to{target}_summary.json，因此 finalize 仅扫这些 summary 会漏掉前一次 worker 的 OT/耗时。应按 append-only SEGMENT_END 事件的独立 pid/时间/target 汇总 actual_updates_this_worker、OT_counts 与 worker_*seconds，FAILURE 记录另列；CSV 行级训练/输入准备时间可交叉核对。不要把累计 counts 当 worker 增量，也不要给未记录的 load/save 开销补造数值。保留原汇总，另写校正表及来源 event 定位；未发生恢复时按现有完整 summary 聚合即可。
