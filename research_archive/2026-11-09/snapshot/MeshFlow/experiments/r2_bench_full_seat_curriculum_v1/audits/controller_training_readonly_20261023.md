# 运行中只读工程审计：训练继承与有限流水线

观察时间：2026-10-23 02:59:21 UTC。注册 SHA256：`7fb69aa203d01b2de027d978adabc88169037774359cd7ce918077e2eca20113`。

审查只读取源码、小型 JSON 状态、回执和已有日志；没有重新加载或哈希大权重，没有模型前向、评分、进程干预或冻结文件修改。本文件是新建审计记录，不改变注册、调度或正在运行进程的行为。

观察时 Native 为 `TRAINING`、stage466/global12466；controller 为 `RUNNING`。COMMON 的24份RAW工作流已完成且exit0，其余采样与D0尚未开始。未发现当前正在阻塞的科学阶段。

## 已发现的外层完成状态误报路径

- `r2/pilots/bench_full_seat_curriculum_v1/controller.py:41` 检测任一子任务为 `STOPPED_REQUIRES_INSPECTION`、`INTERRUPTED_SAFE`、`FAILED` 或 `OOM`。
- 同文件43–44行写入 `controller/status.json` 的 `STOPPED_REQUIRES_INSPECTION`，随后正常 `return`，因此控制器子进程可以exit0。
- `r2/pilots/bench_full_seat_curriculum_v1/run.py:64` 对非训练任务（包括pipeline）把exit0一律记录为外层 `COMPLETE`，没有读取控制器终态。

因此，若本轮以后发生安全中断或子任务失败，可能同时存在：内层controller为 `STOPPED_REQUIRES_INSPECTION`、外层workflow/pipeline为 `COMPLETE`。这个风险影响状态解释，不表示控制器会继续派发失败后的科学任务；控制器在上述路径已停止。观察时没有触发此路径。

**本轮成功判定必须以 `controller/status.json` 的 `READY_REVIEW`，以及各科学阶段实际完成状态、逐调用账本和产物回执共同确认。外层pipeline的 `COMPLETE` 或exit0单独均不是成功证据。** `READY_REVIEW` 也只表示训练、采样、评分和图集准备完成，不等于实际图审或最终报告完成。

审查时磁盘代码的版本标识：

- `run.py` SHA256：`8d7cf68fcf8e11431bec5b045180cf20e33f98e999cd5af01b43022ada68f537`
- `controller.py` SHA256：`3ec296a768d037680a5cef8bee8abaa8e005c0ca331339aad2d036f7087ccc8a`

两文件不在训练注册的 `training_code_hashes` 中。未来可对外层校验作独立工程修正，但不能声称磁盘新代码已改变本次已启动进程；当前控制器和训练未停止、重跑或替换。

后续工程记录：root已为未来调用修正run.py的pipeline分支，要求控制器实际READY_REVIEW才报告COMPLETE；详见 `engineering/pipeline_status_guard.json`。该记录明确本次已运行supervisor已加载旧代码，不声称受到修正保护。本审计的最终成功判据与结束核对项仍然适用。

## 阶段依赖与预算核对

- controller先要求Native工作流 `COMPLETE`，才派SEAM；Native采样须等待SEAM实际达到25步或已完成。SEAM采样须等待SEAM工作流完成。见controller.py:50–55。
- 三组RAW工作流全部完成后才进入评分与D0；D0入口还独立核验72份RAW逐请求回执、数组哈希和50步调用数。见controller.py:56、eval_runtime.py:55–82、post_d0.py:19–21。
- D0三组按模型加载串行，CPU评分可与其重叠。所有D0完成后才派POST评分，两种评分完成后collect，随后gallery。见controller.py:57–75。
- 注册与控制器固定Native6000、SEAM3000；trainer按注册切片连续9000条日程，每阶段调用账本分别上限48000/24000次前向和反向、6000/3000次更新。头与probe调用预算为0。没有自适应延长或追加臂入口。见train.py:343、training_state.py:31–84。
- 已有launch记录不自动重派，失败/安全中断要求显式检查。这可防止隐式科学重放；也意味着发生启动器崩溃或中断时，需要人工检查状态与进程，而不是指望控制器自行修复。

## Native→SEAM继承边界

真实COMMON→Native的CPU检查及stage0回执已通过：单组194份AdamW状态在12000步，live生成器、动量和RNG哈希与源COMMON完全一致。首25步有200次F、200次B、25次更新，全部200曝光虽有来源owner，但激活SEAM槽为0，tip标量为0，自由标量为9(N−K)。这些是已执行证据。

**Native→SEAM尚未发生，本次只能确认实现与注册约束，不能提前宣称实际交接已经通过。** 交接代码要求Native实际COMPLETE/6000步、相同注册和完整checkpoint hash；核其payload后严格载入生成器、单组AdamW与同一参数顺序，恢复RNG并比较内容哈希。SEAM stage0会再次保存这些身份供终点审计。见train.py:142–156、263–284。

两阶段共用同一22父/22C清单；切换仅改变PLAIN/SEAM打包和前向分支。原旧parts adapter保留预OT半尺度、OT后HYBRID0.25与x-width增强；损失均为同一free-corner FM，以整批有效自由标量为分母，无附加生成器损失。SEAM采用来源owner监督；旧833头仅附于checkpoint和导出，训练无头实例/前向/优化器。见train.py:358–396、training_state.py的 `verify_sample` 样本角色检查。

## 资源放行口径与限制

GPU放行均逐次测量当前资源，要求 `free >= max(2048 MiB, 0.1×total) + 1.2×planning_peak`，RAM至少8GiB；控制器每次只尝试一个新的GPU任务。训练、采样、D0估计分别为4500、2800、6000MiB，未将其伪称为实测进程峰。16303MiB设备对应放行最低free分别为7448、5408、9248MiB。见dispatch.py:12–25、controller.py:10–18。

首25步训练实测allocator峰为allocated2795.22MiB、reserved3154MiB；系统used从dispatch1715MiB到step25的5252MiB，step25 free10728MiB。WDDM进程显存为N/A，系统增量和allocator峰均不能冒充隔离的进程峰。

两个需在最终描述中保留的工程边界：

- GPU利用率虽被记录，但未进入controller的准入判断；并行吞吐是否获益需要实际计时，不可仅凭显存余量宣布加速。
- CPU评分实际先执行一个已授权任务，再按其RSS、可用RAM和CPU负载选择最多4进程、每进程1线程。不过复用的 `chair_back_transfer_v1/evaluate.py:139–145` 以 `max(1, ...)` 兜底，即计算出的内存槽位为0时仍返回1；它不是严格的零槽位等待器。观察时RAM充足，尚不构成当前阻塞；若后期内存紧张应先检查实际资源，不把这一公式描述成无条件安全保证。

## 结束时必须显式补核

1. controller为READY_REVIEW；每个训练/RAW/D0/评分工作流与实际产物均完成，不能只读外层pipeline COMPLETE。
2. Native6000/SEAM3000准确完成，总72000曝光与9000更新；所有调用BEGIN/DONE成对，无失败、未闭合调用、额外头/probe或未披露尝试。
3. Native终点与SEAM0的生成器、AdamW、RNG、冻结头四组内容哈希相等；global18000→21000与连续输入游标一致。
4. 各阶段不可变checkpoint回执、终点导出及status所指回执一致；保留JSON回执核验和大文件/张量重验的证据等级差别。
5. 保存各实际并行阶段资源峰、worker数及墙钟时间；不能把竞争期间的墙钟时间与串行时间当作同条件速度比较。
6. READY_REVIEW后仍需实际图审与报告；未把图集存在等同于逐图已查看。
