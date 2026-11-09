## 执行合同与工程修正（方法事实片段）

本轮是新授权的 **USER_SMALL32 对 USER_EXPANDED_67** 实验。两臂采用同一官方 Chair 生成器 EMA、同一份 role／Geo 条件初始化，从优化器累计步数0开始使用 fresh AdamW；不继承历史 R2、USER_MIX 或预检的优化器状态。`initialization/audit.json` 记录全部参数为 FP32、可训练，role 和 Geo 出口均为零初始化，没有构造辅助头或加载教师。共同初始化文件的已记录 SHA256 为 `6fdd6d666f38ca1292091ce47438c0e77456aa8b8d0a7d08ceec66d7c6568352`。这些事实验证继承与初始化合同，不声称带 C 的条件前向与原无条件输出等价。

训练固定 USER-only、Native/T1＋role＋Geo、FM 和全程 HYBRID=0.25，batch8／microbatch1，每批8个不同父、4个clean与4个width增强，沿用有效 free 变量归一化与已冻结数据流。单 GPU worker，每2000步轮换两臂，分段保存和恢复各自模型、优化器与随机状态。训练阶段不引入 B、READOUT、新 loss、蒸馏或去噪器适配。

正式训练前执行了16次隔离的真实优化器更新：128次训练前向、128次反向，以及32次一致性检查前向，共160次前向；预检收据没有失败或待结算调用。16个一致性fixture全部记录为exact。预检权重与优化器状态被丢弃，正式两臂重新从上述共同step0开始。资源和计时来自这些已授权更新，没有另增GPU探针任务。

`training_budget_lock.json` 于 **2026-10-12 03:27:42 UTC** 一次性锁定 **每臂6000步**，`downgrade=false`。预检估算包含OT等待，计时统计排除最初2步暖机，剩余更新均值约1.460秒，保守估计2.045秒／更新；加上保存、重载和固定诊断余量，两臂6000步估计约26213秒，当时至训练截止尚余31672秒。4000步仅为正式训练前的共同预算备选，本轮没有因生成质量或中途排名改用4000，也没有挑选best checkpoint。预检160F／128B／16更新与两臂正式预算分开记账，不能把它们隐去或算成正式端点的前16步。

### 三项已留痕的工程修正

| 事项 | 实际问题与处理 | 对科学调用和合同的影响 |
| --- | --- | --- |
| 冻结条件文件读取接口，03:24:10 UTC | 旧条件NPZ按推理安全原则只存 C 与来源ID，没有独立 Y 数组。新读取器原先误要求存在 Y；修正后核对 C、父ID、N/K、条件类型与 C/free 面ID，再从同一冻结父 M 的已登记补集读取训练 Y。若文件含 Y，仍逐值验证。 | 发生在CPU读取核验阶段，F=0、B=0、OT=0、更新=0；未改 C、目标几何、分区或旧NPZ。前后源码和diff已保存。 |
| 控制器事件参数重名，03:26:41 UTC | `event(name, **fields)` 与 `event(..., name=...)` 冲突，异常位于 `subprocess.Popen` 之前。修正事件形参，保留原 `evaluation_prepare` 半记录，以明确的新调度名 `evaluation_prepare_v2` 接续。 | 该次子进程=0、模型调用=0、新噪声=0；不是GPU预检失败，也没有模型重跑。 |
| EXPANDED最后2000步启动前的资源阻塞，07:37:13 UTC至07:44:58 UTC | 原控制器因可用显存不能保留余量而拒绝启动 `USER_EXPANDED_67_segment_6000`。该原调度目录为空，没有该次模型调用；已完成的EXPANDED step4000与SMALL step6000保持原状。依据已完成阶段遥测，记录修订的增量显存估计，再用新调度名 `USER_EXPANDED_67_segment_6000_resource_recovered` 启动原定4000→6000段。 | 记录 `model_retries=0`、额外模型探针=0、终止无关进程=0；batch、精度、优化器、数据、随机流和6000步预算均未改变。恢复的是尚未启动的调度，不是重做失败更新。 |

资源修订不能描述成“放宽门槛强行开跑”。原增量估计6948 MiB来自预检**整卡**峰值5790 MiB乘1.2，包含约2.2 GiB背景占用；修订依据已完成阶段的整卡前后差和PyTorch reserved峰值，取保守代理再留余量，得到5120 MiB。WDDM下进程显存峰值不可用；3904 MiB设备差值只是代理，1024 MiB上下文／采样余量是人为保守余量，均不冒称实测单进程峰值。保留的额外显存余量仍为 `max(2048 MiB, 15%总显存)`，即2445.45 MiB。

**实际恢复启动时**，新鲜资源检查记录可用显存13402 MiB，原门槛 `6948＋2445.45=9393.45 MiB` 同样能够通过。`launch_context.json` 明确记录 `original_gate_would_also_pass_at_actual_launch=true`。因此本次启动既有资源回落的事实，也有估计方法修订的记录；不能把恢复完全归因于修订估计，更不能将启动前阻塞称为OOM、数值失败或一次模型重试。

本片段只说明已执行的初始化、预算锁定和工程事件，不据此宣布后续全部训练、生成、评分或图审完成，也不提供模型效果结论。所有时间为UTC。原12小时阻塞实验仍只读，本轮截止时间仍为15:15:34 UTC，训练派发截止12:15:34 UTC。

证据位置：`initialization/audit.json`、`initialization/common_step0000.json`、`preflight/report.json`、`training_budget_lock.json`、`engineering/reader_schema_fix/{receipt.json,code.diff}`、`engineering/controller_dispatch_fix/receipt.json`、`engineering/resource_dispatch_recovery/{blocked_controller_status.json,memory_basis.json,decision.json,launch_context.json}`，以及新恢复调度的`started.json`。本片段复核只读取小收据及旧调度目录的空目录状态，没有读取本轮输出质量、FM排名或重哈希权重。
