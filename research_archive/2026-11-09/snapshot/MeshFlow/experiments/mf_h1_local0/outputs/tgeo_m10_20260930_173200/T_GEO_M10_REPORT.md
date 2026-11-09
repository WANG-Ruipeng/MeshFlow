# T_GEO_M10_REPORT

本轮新增训练：既有T_geo累计2000步起点，两臂各新增500次更新，总1000次，终点累计2500步。评估保留原始终点，不挑最佳。

Geo_M10−Geo_continue / 原片patch0：P0 自由面RMS均差0.3436个百分点（改善2/8），Gamma RMS均差0.6604，CF比例均差-0.0525，FF对数均差0.0000；P1 自由面RMS均差-0.0312个百分点（改善5/8），Gamma RMS均差0.5139，CF比例均差-0.0275，FF对数均差-0.3750。
Geo_M10−Geo_continue / 额外片patch1：P0 自由面RMS均差1.3912个百分点（改善0/2），Gamma RMS均差1.5947，CF比例均差-0.0350，FF对数均差15.0000；P1 自由面RMS均差-0.0292个百分点（改善1/2），Gamma RMS均差-0.0004，CF比例均差-0.0100，FF对数均差-16.0000。
Geo_M10−Geo_continue / 额外片patch2：P0 自由面RMS均差0.0366个百分点（改善1/2），Gamma RMS均差0.0272，CF比例均差0.0000，FF对数均差16.0000；P1 自由面RMS均差0.5663个百分点（改善1/2），Gamma RMS均差0.1673，CF比例均差0.0050，FF对数均差-4.0000。
Geo_continue−Geo_initial / 原片patch0：P0 自由面RMS均差0.0921个百分点（改善4/8），Gamma RMS均差0.3450，CF比例均差-0.0338，FF对数均差-9.1250；P1 自由面RMS均差0.0738个百分点（改善5/8），Gamma RMS均差-0.0933，CF比例均差-0.0025，FF对数均差-24.1250。
Geo_continue−Geo_initial / 额外片patch1：P0 自由面RMS均差-1.0844个百分点（改善1/2），Gamma RMS均差-1.1832，CF比例均差0.0000，FF对数均差-17.0000；P1 自由面RMS均差-0.6409个百分点（改善1/2），Gamma RMS均差-0.2966，CF比例均差-0.0050，FF对数均差10.5000。
Geo_continue−Geo_initial / 额外片patch2：P0 自由面RMS均差1.1492个百分点（改善1/2），Gamma RMS均差2.5589，CF比例均差-0.0300，FF对数均差-8.0000；P1 自由面RMS均差-0.8075个百分点（改善2/2），Gamma RMS均差-0.0497，CF比例均差-0.0200，FF对数均差0.0000。
取舍：P1/原片patch0 FF均数下降，同时Gamma RMS (% L)均值上升；P1/额外片patch1 FF均数下降，所列精度/CF均值未同时上升；P1/额外片patch2 FF均数下降，同时自由面对称RMS (% L)、Gamma RMS (% L)、CF穿刺free比例均值上升。不能将FF减少单独称为全面提升。
以下分组未显示M10相对继续纯FM的FF均数下降：P0/原片patch0、P0/额外片patch1、P0/额外片patch2。
新增损失的独立效果以Geo_M10−Geo_continue为准；Geo_continue−Geo_initial只描述额外500步纯FM。没有复合总分或择优终点。
两个父对象与完整目标都已参与训练，额外patch也已在前轮实验观察过；本轮独立的是9801–9808噪声。单一训练流、小样本结果不是新对象泛化或统计显著性证明。

## 本轮判决

实现与执行通过，但没有建立M10在既有T_geo上的稳定增量收益。建议保留现有T1+T_geo工作基线，不将M10改为默认；M10终点仅保留作CF穿刺与覆盖/接口精度取舍的研究分支。

原patch、每对象8个新噪声的均值如下。误差单位为百分比bbox对角线；CF为受proper穿刺影响的free面百分比，FF为非法面片对数。

| 指标 | P0 Geo_continue | P0 Geo_M10 | P1 Geo_continue | P1 Geo_M10 |
| --- | --- | --- | --- | --- |
| 排除C的对称RMS (% L) | 0.7935 | 1.1371 | 0.5566 | 0.5255 |
| 目标→free RMS (% L) | 0.8396 | 1.3338 | 0.4730 | 0.6048 |
| Gamma p95 (% L) | 2.7481 | 4.2468 | 2.6870 | 4.0638 |
| CF受影响free面 (%) | 12.375 | 7.125 | 6.375 | 3.625 |
| FF非法对均数 | 74.000 | 74.000 | 66.250 | 65.875 |

P0自由面RMS配对差均值+0.3436个百分点，但配对中位仅+0.0045；9802是主要均值退化来源，仍有6/8变差。P1自由面RMS有5/8改善，Gamma p95有6/8变差。P0边界尾部主要受9802/9805影响，P1主要受9802/9807影响。保留全部异常样本，不将均值变化说成每个噪声均匀变化。

CF下降进入了完整生成，但FF大幅减少没有复现。原patch两对象的FF均差分别为0与−0.375对；部分额外patch存在局部FF收益，同时有对象/patch退化，不能合并为普遍收益。既有额外patch每对象每片仅2个噪声，分别报告。

纯FM继续训练相对Geo_initial也并非全面改善：原patch自由RMS均值P0 0.7014→0.7935、P1 0.4828→0.5566，而FF均数83.125→74、90.375→66.25。这里不能把“继续训练”或“叠加M10”任何一项自动当成更好的默认模型。

独立标量复核补充：patch1/P1虽然CF与FF均数下降，但两个噪声的Gamma覆盖都下降，p95均值略升；这不是无代价改善。原patch/P0的阈值内Gamma覆盖增加与Gamma RMS/p95恶化可以并存，表示局部贴近和远端缝隙分布的不同变化，不能互相抵消。原patch/P1的Gamma覆盖7/8下降；其RMS均值改善也受9805影响，不能仅报均值。

本轮仅两个已训练父对象与一个训练流；新噪声通过不等于新对象泛化。没有在看到结果后调整FC权重、门槛、seed、patch或训练长度。

## 对照与实际执行

Geo_continue保留纯masked FM；Geo_M10仅添加0.1×FC，激活条件为保存的FP32 t≥0.5。FC按effective batch全部有效组归一化，包括未激活样本；没有按激活比例补偿、FF损失、bbox归一化或新模块。
FC监督先构造free预测终点Xt+(1−t)v；源顶点ID只构建监督分组，不作模型条件或最近邻匹配。C锚点detach，已训练geo编码器仍全参数可学习。
两臂逐步输入一致：True，500批；恢复同一seed1010输入RNG，从batch500/sample4000继续。每臂fresh AdamW，lr=1e-05，betas=[0.9, 0.95]，weight_decay=0.0，clip=1.0；effective8/micro1。
所有主干、role及geo参数FP32且可训练；模块eval禁dropout。生产前向BF16/math SDPA、TF32关闭、严格确定性；geo内部FP32。不缓存跨更新条件编码，不自动重试数值失败。

| 模型 | 新增更新 | 累计更新 | 全参数数 | 可训练数 | encoder参数 | 训练秒 | 峰值显存bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Geo_continue | 500 | 2500 | 130401155 | 130401155 | 149888 | 564.9578 | 2706070016 |
| Geo_M10 | 500 | 2500 | 130401155 | 130401155 | 149888 | 571.9926 | 2707162112 |

实际调用按记录统计；预算不作为已执行证据，CPU数学fixture梯度单独保留于gates记录。

| 调用/更新项 | 实际次数 |
| --- | --- |
| model_forward_attempts | 11652 |
| model_forward_returns | 11652 |
| backward_attempts | 8002 |
| backward_returns | 8002 |
| optimizer_step_attempts | 1000 |
| optimizer_updates | 1000 |
| training_OT_calls | 8000 |
| training_OT_returns | 8000 |
| gate_OT_calls | 8 |
| gate_OT_returns | 8 |
| sampling_forward_attempts | 3650 |
| sampling_forward_returns | 3650 |
| rollout_attempts | 74 |
| complete_rollouts | 73 |
| formal_rollouts | 72 |
| replay_rollouts | 1 |
| euler_updates | 3650 |

分阶段前向账本：`{"gates": 2, "Geo_continue_training": 4000, "Geo_M10_training": 4000, "sampling_Geo_initial": 1200, "sampling_Geo_continue": 1200, "sampling_Geo_M10": 1250}`。
执行时间记录（秒）：`{"preflight_seconds": 9.361703216999999, "sampling_resumption_wall_seconds": 165.069487384, "model_sampling_seconds": 159.86788248399975, "sampling_total_seconds": null, "sampling_total_status": "NOT_RECORDED: initial wall timer interrupted; per-rollout model timing complete", "offline_evaluation_seconds": 63.940605990999984}`。
本报告仅汇总已有分数和绘图；新增模型前向/反向/优化器更新/OT/几何评分均为0。报告渲染耗时另记于results.json的report字段。

## 门禁与已知数值边界

CPU M10公式：PASS；真实GPU门禁：PASS，2前向/2反向/0更新。
同输入velocity逐位一致=True；参数不变=True；C不变=True；C锚点无梯度=True。
门禁比较FM与FM+FC的已训练encoder梯度增量；并非重新验证整个主干的面序置换数值等价。既有BF16主干顺序敏感性仍存在，本轮不切全FP32、不更改旧容差、不据此声称修复该边界。

| encoder组 | FM梯度L2 | M10总梯度L2 | 梯度增量L2 |
| --- | --- | --- | --- |
| mlp | 4.196e-07 | 4.272e-07 | 8.86e-09 |
| W_self | 1.121e-07 | 1.141e-07 | 2.323e-09 |
| W_neighbor | 1.121e-07 | 1.141e-07 | 2.323e-09 |
| norm | 3.394e-08 | 3.467e-08 | 7.576e-10 |
| W_out | 6.489e-05 | 6.557e-05 | 1.085e-06 |

## 采样I/O中断与独立恢复

采样调度曾因results.json的跨进程读锁与atomic replace冲突发生PermissionError(13)。失败位于新增调度attempt记录写回、进入模型之前；这次调度没有对应模型前向或生成结果。
恢复只补尚未完成的固定采样项与预定重放，保留已完成31条正式输出，不重训、不改损失/核心采样、不重新选择噪声。下表使用恢复原记录，缺失字段不从计划推算；最终实际调用以上方总账本为准。

| 恢复证据字段 | 登记值 |
| --- | --- |
| io_recovery.status | PASS |
| io_recovery.error | PermissionError(13, 'Permission denied') |
| io_recovery.interruption_snapshot.path | /mnt/e/MeshFlow/experiments/mf_h1_local0/outputs/tgeo_m10_20260930_173200/sampling_io_interruption.json |
| io_recovery.interruption_snapshot.bytes | 24413249 |
| io_recovery.interruption_snapshot.sha256 | 22e9b2a1ebedd09e780cdd791b822395cb02d8d313a138a8a9e741449132b077 |
| io_recovery.source.path | /mnt/e/MeshFlow/experiments/mf_h1_local0/tgeo_m10_resume_io.py |
| io_recovery.source.bytes | 5730 |
| io_recovery.source.sha256 | 663df23702c4a7ca08fa231897aa63e1bb97a507a32e0e9775c2c812ce193029 |
| io_recovery.stage | Before model call: sampling dispatcher JSON atomic rename |
| io_recovery.cause | Windows read handle blocked WSL atomic replace; subsequent observation uses WSL reads only |
| io_recovery.preserved_formal_outputs | 31 |
| io_recovery.unstarted_dispatch_attempts | 1 |
| io_recovery.interrupted_model_forwards | 0 |
| io_recovery.interrupted_complete_rollouts | 0 |
| io_recovery.resumed_missing_formal | 41 |
| io_recovery.planned_replays | 1 |
| io_recovery.training_restarted | False |
| io_recovery.original_execution_sources_unchanged | True |
| io_recovery.no_sampler_or_loss_changes | True |
| io_recovery.resumption_wall_seconds | 165.0695 |
| io_recovery.initial_sampling_wall_seconds | NOT_RECORDED_BEFORE_IO_EXIT |
| io_recovery.original_raw_files_preserved | True |
| io_recovery.final_model_forward_count | 11652 |

失败原trace（io_recovery.original_traceback）：

```text
Traceback (most recent call last):
  File "/mnt/e/MeshFlow/experiments/mf_h1_local0/tgeo_m10_run.py", line 450, in main
    sample_all(r,out)
  File "/mnt/e/MeshFlow/experiments/mf_h1_local0/tgeo_m10_run.py", line 375, in sample_all
    row=_sample_one(r,out,model,name,parent,patch,seed,noises[seed],C)
  File "/mnt/e/MeshFlow/experiments/mf_h1_local0/tgeo_m10_run.py", line 395, in _sample_one
    counts['rollout_attempts']+=1;write(r,out)
  File "/mnt/e/MeshFlow/experiments/mf_h1_local0/tgeo_m10_run.py", line 52, in write
    atomic_json(Path(out)/'results.json',r)
  File "/mnt/e/MeshFlow/native_t1/artifacts.py", line 42, in atomic_json
    temporary.replace(path)
  File "/home/warpwang/miniforge3/envs/meshflow/lib/python3.10/pathlib.py", line 1247, in replace
    self._accessor.replace(self, target)
PermissionError: [Errno 13] Permission denied: '/mnt/e/MeshFlow/experiments/mf_h1_local0/outputs/tgeo_m10_20260930_173200/results.tmp' -> '/mnt/e/MeshFlow/experiments/mf_h1_local0/outputs/tgeo_m10_20260930_173200/results.json'

```

未进入模型的I/O调度attempt与实际完整采样分别记账；不能把调度次数当成完整生成数，也不能抹掉该次失败。恢复源码路径/hash与恢复前后计数以上述原记录为准；没有把预计前向数写成实际结果。
中断前的采样进程wall计时未保存，因此完整采样总wall为NOT_RECORDED；恢复进程wall单列。model_sampling_seconds是每条实际完成rollout的采样计时之和，包含原采样中的同步/记录，不是纯GPU内核耗时，也不能替代缺失的总wall。


## 固定评价口径

Geo_initial累计2000、Geo_continue/Geo_M10累计2500；三个模型各24条，正式共72条。原patch0每父对象8噪声9801–9808，额外patch1/2每父对象各2噪声9801/9802。重放不计入分数分布。额外patch1、patch2分别报告，每对象每片只有2个噪声；不把两片合成4个独立噪声样本。
全部N112/K12/free100，50步Euler、51状态，C逐位clamp保持；同seed跨模型/对象/上下文共享原始自由高斯起点，不使用OT匹配训练噪声。额外patch原样复用前轮bank，没有重新选片。
误差单位为百分比bbox对角线L；Gamma覆盖与质量比例是0–1。下表每格“均值 / 输出中位 / 最差”；覆盖率最差取最小，其余误差/非法数量最差取最大。


### 原片patch0：按父对象的表面与接口


| 父对象 | 指标 | Geo_initial | Geo_continue | Geo_M10 |
| --- | --- | --- | --- | --- |
| 0 | 自由面对称RMS (% L) | 0.7014 / 0.1932 / 3.4578 | 0.7935 / 0.2811 / 4.0807 | 1.1371 / 0.2916 / 7.0496 |
| 0 | free→目标RMS (% L) | 0.6984 / 0.1755 / 3.3380 | 0.6604 / 0.0913 / 3.6416 | 0.8478 / 0.1014 / 6.0116 |
| 0 | 目标→free RMS (% L) | 0.6784 / 0.2179 / 3.5736 | 0.8396 / 0.3037 / 4.4769 | 1.3338 / 0.3888 / 7.9532 |
| 0 | Gamma RMS (% L) | 0.9152 / 0.2447 / 3.3121 | 1.2603 / 0.2498 / 4.1406 | 1.9206 / 1.2833 / 6.8873 |
| 0 | Gamma p95 (% L) | 1.8421 / 0.3002 / 7.8608 | 2.7481 / 0.3602 / 7.8997 | 4.2468 / 3.2084 / 10.9516 |
| 0 | Gamma覆盖率 | 0.3977 / 0.4129 / 0.0039 | 0.3449 / 0.3410 / 0.0058 | 0.3890 / 0.3952 / 0.0273 |
| 1 | 自由面对称RMS (% L) | 0.4828 / 0.3882 / 1.1723 | 0.5566 / 0.4747 / 0.9209 | 0.5255 / 0.4454 / 1.0761 |
| 1 | free→目标RMS (% L) | 0.4027 / 0.1798 / 1.2084 | 0.5142 / 0.4068 / 1.2883 | 0.3412 / 0.2102 / 0.9977 |
| 1 | 目标→free RMS (% L) | 0.4822 / 0.3539 / 1.1350 | 0.4730 / 0.3912 / 1.0698 | 0.6048 / 0.4973 / 1.2415 |
| 1 | Gamma RMS (% L) | 1.1507 / 0.4714 / 2.7768 | 1.0574 / 0.5033 / 2.7163 | 1.5714 / 1.4656 / 3.2346 |
| 1 | Gamma p95 (% L) | 2.7820 / 0.7305 / 7.7959 | 2.6870 / 0.6353 / 7.6131 | 4.0638 / 3.9589 / 7.8609 |
| 1 | Gamma覆盖率 | 0.3181 / 0.2293 / 0.1171 | 0.3759 / 0.2905 / 0.2093 | 0.2292 / 0.2387 / 0.0418 |

### 额外片patch1：按父对象的表面与接口


| 父对象 | 指标 | Geo_initial | Geo_continue | Geo_M10 |
| --- | --- | --- | --- | --- |
| 0 | 自由面对称RMS (% L) | 3.1869 / 3.1869 / 6.2035 | 2.1024 / 2.1024 / 3.9314 | 3.4937 / 3.4937 / 6.5645 |
| 0 | free→目标RMS (% L) | 2.4294 / 2.4294 / 4.7454 | 1.6301 / 1.6301 / 3.0176 | 2.8528 / 2.8528 / 5.4080 |
| 0 | 目标→free RMS (% L) | 3.7956 / 3.7956 / 7.3790 | 2.4855 / 2.4855 / 4.6697 | 4.0323 / 4.0323 / 7.5458 |
| 0 | Gamma RMS (% L) | 4.6685 / 4.6685 / 9.2591 | 3.4853 / 3.4853 / 6.3385 | 5.0800 / 5.0800 / 9.5080 |
| 0 | Gamma p95 (% L) | 6.4016 / 6.4016 / 12.6647 | 5.1526 / 5.1526 / 8.4601 | 7.4250 / 7.4250 / 12.9473 |
| 0 | Gamma覆盖率 | 0.3836 / 0.3836 / 0.0000 | 0.2785 / 0.2785 / 0.0058 | 0.2571 / 0.2571 / 0.0000 |
| 1 | 自由面对称RMS (% L) | 1.4090 / 1.4090 / 1.9657 | 0.7681 / 0.7681 / 1.1253 | 0.7389 / 0.7389 / 1.2074 |
| 1 | free→目标RMS (% L) | 1.6286 / 1.6286 / 2.1375 | 0.8176 / 0.8176 / 1.3238 | 0.8445 / 0.8445 / 1.4540 |
| 1 | 目标→free RMS (% L) | 1.1118 / 1.1118 / 1.7775 | 0.6869 / 0.6869 / 0.8832 | 0.5985 / 0.5985 / 0.8954 |
| 1 | Gamma RMS (% L) | 0.8351 / 0.8351 / 1.3060 | 0.5385 / 0.5385 / 0.5964 | 0.5381 / 0.5381 / 0.6107 |
| 1 | Gamma p95 (% L) | 2.0038 / 2.0038 / 3.1322 | 0.8658 / 0.8658 / 1.1740 | 0.8851 / 0.8851 / 1.2877 |
| 1 | Gamma覆盖率 | 0.2889 / 0.2889 / 0.0694 | 0.5713 / 0.5713 / 0.5415 | 0.5520 / 0.5520 / 0.5104 |

### 额外片patch2：按父对象的表面与接口


| 父对象 | 指标 | Geo_initial | Geo_continue | Geo_M10 |
| --- | --- | --- | --- | --- |
| 0 | 自由面对称RMS (% L) | 2.6004 / 2.6004 / 4.2468 | 3.7496 / 3.7496 / 6.5550 | 3.7862 / 3.7862 / 6.5012 |
| 0 | free→目标RMS (% L) | 2.1590 / 2.1590 / 2.9934 | 3.0751 / 3.0751 / 5.2284 | 3.0995 / 3.0995 / 5.0795 |
| 0 | 目标→free RMS (% L) | 2.7314 / 2.7314 / 5.2068 | 4.3106 / 4.3106 / 7.6550 | 4.3422 / 4.3422 / 7.6635 |
| 0 | Gamma RMS (% L) | 3.4904 / 3.4904 / 6.4662 | 6.0493 / 6.0493 / 11.4131 | 6.0765 / 6.0765 / 11.3908 |
| 0 | Gamma p95 (% L) | 4.8552 / 4.8552 / 8.7065 | 7.4700 / 7.4700 / 13.3797 | 7.4754 / 7.4754 / 13.2970 |
| 0 | Gamma覆盖率 | 0.1996 / 0.1996 / 0.0000 | 0.1763 / 0.1763 / 0.0000 | 0.0974 / 0.0974 / 0.0000 |
| 1 | 自由面对称RMS (% L) | 2.2275 / 2.2275 / 2.5992 | 1.4200 / 1.4200 / 1.7937 | 1.9862 / 1.9862 / 2.2639 |
| 1 | free→目标RMS (% L) | 2.9078 / 2.9078 / 3.2111 | 1.8558 / 1.8558 / 2.3471 | 2.4467 / 2.4467 / 2.6586 |
| 1 | 目标→free RMS (% L) | 1.0566 / 1.0566 / 1.7889 | 0.7671 / 0.7671 / 0.9622 | 1.3513 / 1.3513 / 1.7839 |
| 1 | Gamma RMS (% L) | 0.3711 / 0.3711 / 0.6431 | 0.3214 / 0.3214 / 0.3656 | 0.4887 / 0.4887 / 0.6236 |
| 1 | Gamma p95 (% L) | 0.7179 / 0.7179 / 1.2900 | 0.6875 / 0.6875 / 0.7427 | 0.9362 / 0.9362 / 1.1616 |
| 1 | Gamma覆盖率 | 0.3369 / 0.3369 / 0.0019 | 0.3077 / 0.3077 / 0.1639 | 0.0946 / 0.0946 / 0.0019 |

## 配对效果

差=候选−参考；每格“差均值 / 差中位 / 最差配对差；改善数/n”。配对差中位直接对同parent/patch/seed的差取中位，不是两个输出中位相减。所有样本保留。


### Geo_M10 − Geo_continue

| 片组 | 父对象 | 自由面RMS | Gamma RMS | Gamma覆盖 | CF穿刺比例 | FF非法对 |
| --- | --- | --- | --- | --- | --- | --- |
| 原片patch0 | 0 | 0.3436 / 0.0045 / 2.9689; 2/8 | 0.6604 / 0.0090 / 2.7467; 2/8 | 0.0441 / 0.0207 / -0.1292; 5/8 | -0.0525 / -0.0050 / 0.0400; 4/8 | 0.0000 / 5.5000 / 19.0000; 3/8 |
| 原片patch0 | 1 | -0.0312 / -0.0159 / 0.1552; 5/8 | 0.5139 / 0.0110 / 2.8310; 3/8 | -0.1467 / -0.0859 / -0.5762; 1/8 | -0.0275 / -0.0100 / 0.0100; 6/8 | -0.3750 / 0.0000 / 25.0000; 4/8 |
| 额外片patch1 | 0 | 1.3912 / 1.3912 / 2.6331; 0/2 | 1.5947 / 1.5947 / 3.1695; 0/2 | -0.0214 / -0.0214 / -0.0370; 0/2 | -0.0350 / -0.0350 / -0.0200; 2/2 | 15.0000 / 15.0000 / 29.0000; 0/2 |
| 额外片patch1 | 1 | -0.0292 / -0.0292 / 0.0821; 1/2 | -0.0004 / -0.0004 / 0.0143; 1/2 | -0.0193 / -0.0193 / -0.0311; 0/2 | -0.0100 / -0.0100 / -0.0100; 2/2 | -16.0000 / -16.0000 / -4.0000; 2/2 |
| 额外片patch2 | 0 | 0.0366 / 0.0366 / 0.1270; 1/2 | 0.0272 / 0.0272 / 0.0767; 1/2 | -0.0789 / -0.0789 / -0.1578; 0/2 | 0.0000 / 0.0000 / 0.0000; 0/2 | 16.0000 / 16.0000 / 22.0000; 0/2 |
| 额外片patch2 | 1 | 0.5663 / 0.5663 / 1.2177; 1/2 | 0.1673 / 0.1673 / 0.3465; 1/2 | -0.2131 / -0.2131 / -0.4495; 1/2 | 0.0050 / 0.0050 / 0.0100; 0/2 | -4.0000 / -4.0000 / 7.0000; 1/2 |

### Geo_continue − Geo_initial

| 片组 | 父对象 | 自由面RMS | Gamma RMS | Gamma覆盖 | CF穿刺比例 | FF非法对 |
| --- | --- | --- | --- | --- | --- | --- |
| 原片patch0 | 0 | 0.0921 / 0.0094 / 0.6231; 4/8 | 0.3450 / 0.0139 / 2.5796; 4/8 | -0.0528 / -0.0971 / -0.3167; 4/8 | -0.0338 / -0.0500 / 0.1900; 6/8 | -9.1250 / -14.0000 / 14.0000; 6/8 |
| 原片patch0 | 1 | 0.0738 / -0.0104 / 0.6098; 5/8 | -0.0933 / -0.0411 / 1.5385; 5/8 | 0.0579 / 0.0494 / -0.3644; 5/8 | -0.0025 / -0.0100 / 0.0700; 5/8 | -24.1250 / -14.5000 / 7.0000; 6/8 |
| 额外片patch1 | 0 | -1.0844 / -1.0844 / 0.1033; 1/2 | -1.1832 / -1.1832 / 0.5542; 1/2 | -0.1051 / -0.1051 / -0.2160; 1/2 | 0.0000 / 0.0000 / 0.0500; 1/2 | -17.0000 / -17.0000 / -6.0000; 2/2 |
| 额外片patch1 | 1 | -0.6409 / -0.6409 / 0.2730; 1/2 | -0.2966 / -0.2966 / 0.2321; 1/2 | 0.2824 / 0.2824 / 0.0330; 2/2 | -0.0050 / -0.0050 / 0.0100; 1/2 | 10.5000 / 10.5000 / 15.0000; 0/2 |
| 额外片patch2 | 0 | 1.1492 / 1.1492 / 2.3081; 1/2 | 2.5589 / 2.5589 / 4.9469; 0/2 | -0.0233 / -0.0233 / -0.0465; 0/2 | -0.0300 / -0.0300 / -0.0100; 2/2 | -8.0000 / -8.0000 / -1.0000; 2/2 |
| 额外片patch2 | 1 | -0.8075 / -0.8075 / -0.0621; 2/2 | -0.0497 / -0.0497 / 0.2664; 1/2 | -0.0292 / -0.0292 / -0.5080; 1/2 | -0.0200 / -0.0200 / -0.0100; 2/2 | 0.0000 / 0.0000 / 2.0000; 1/2 |

## 几何质量与检查范围

分数均来自既有离线评分；本报告不重新测试相交。每格均值/中位/最差；free面积只诊断，无好坏方向。

| 片组 | 父对象 | 指标 | Geo_initial | Geo_continue | Geo_M10 |
| --- | --- | --- | --- | --- | --- |
| 原片patch0 | 0 | CF穿刺free比例 | 0.1575 / 0.1650 / 0.2900 | 0.1237 / 0.1100 / 0.3100 | 0.0713 / 0.0450 / 0.1800 |
| 原片patch0 | 0 | CF穿刺对数 | 22.8750 / 19.5000 / 44.0000 | 15.3750 / 14.0000 / 37.0000 | 9.6250 / 7.0000 / 25.0000 |
| 原片patch0 | 0 | FF非法对数 | 83.1250 / 78.0000 / 136.0000 | 74.0000 / 62.5000 / 148.0000 | 74.0000 / 66.5000 / 156.0000 |
| 原片patch0 | 0 | 最小角<5°比例 | 0.1412 / 0.1400 / 0.1900 | 0.1650 / 0.1650 / 0.2200 | 0.1613 / 0.1650 / 0.2000 |
| 原片patch0 | 0 | q>10比例 | 0.0087 / 0.0000 / 0.0600 | 0.0100 / 0.0000 / 0.0400 | 0.0088 / 0.0000 / 0.0500 |
| 原片patch0 | 0 | free总面积（无好坏方向） | 50.3954 / 50.6895 / N/A | 48.7234 / 48.7239 / N/A | 47.5844 / 48.8388 / N/A |
| 原片patch0 | 1 | CF穿刺free比例 | 0.0663 / 0.0700 / 0.1100 | 0.0638 / 0.0650 / 0.1500 | 0.0363 / 0.0300 / 0.0800 |
| 原片patch0 | 1 | CF穿刺对数 | 11.1250 / 11.0000 / 24.0000 | 9.0000 / 8.0000 / 23.0000 | 6.1250 / 6.5000 / 12.0000 |
| 原片patch0 | 1 | FF非法对数 | 90.3750 / 84.5000 / 147.0000 | 66.2500 / 63.5000 / 88.0000 | 65.8750 / 70.5000 / 79.0000 |
| 原片patch0 | 1 | 最小角<5°比例 | 0.1900 / 0.1800 / 0.3100 | 0.1787 / 0.1800 / 0.2000 | 0.1975 / 0.1950 / 0.2500 |
| 原片patch0 | 1 | q>10比例 | 0.0675 / 0.0600 / 0.1200 | 0.0612 / 0.0650 / 0.0900 | 0.0838 / 0.0700 / 0.2000 |
| 原片patch0 | 1 | free总面积（无好坏方向） | 31.3005 / 30.9113 / N/A | 31.7095 / 31.7217 / N/A | 31.1415 / 31.3796 / N/A |
| 额外片patch1 | 0 | CF穿刺free比例 | 0.0400 / 0.0400 / 0.0800 | 0.0400 / 0.0400 / 0.0500 | 0.0050 / 0.0050 / 0.0100 |
| 额外片patch1 | 0 | CF穿刺对数 | 4.0000 / 4.0000 / 8.0000 | 5.5000 / 5.5000 / 6.0000 | 1.0000 / 1.0000 / 2.0000 |
| 额外片patch1 | 0 | FF非法对数 | 102.0000 / 102.0000 / 134.0000 | 85.0000 / 85.0000 / 128.0000 | 100.0000 / 100.0000 / 129.0000 |
| 额外片patch1 | 0 | 最小角<5°比例 | 0.1200 / 0.1200 / 0.1300 | 0.1100 / 0.1100 / 0.1200 | 0.1100 / 0.1100 / 0.1100 |
| 额外片patch1 | 0 | q>10比例 | 0.0350 / 0.0350 / 0.0700 | 0.0250 / 0.0250 / 0.0500 | 0.0450 / 0.0450 / 0.0800 |
| 额外片patch1 | 0 | free总面积（无好坏方向） | 53.6403 / 53.6403 / N/A | 55.9922 / 55.9922 / N/A | 52.1047 / 52.1047 / N/A |
| 额外片patch1 | 1 | CF穿刺free比例 | 0.0450 / 0.0450 / 0.0700 | 0.0400 / 0.0400 / 0.0500 | 0.0300 / 0.0300 / 0.0400 |
| 额外片patch1 | 1 | CF穿刺对数 | 7.5000 / 7.5000 / 9.0000 | 7.5000 / 7.5000 / 8.0000 | 6.0000 / 6.0000 / 7.0000 |
| 额外片patch1 | 1 | FF非法对数 | 78.5000 / 78.5000 / 84.0000 | 89.0000 / 89.0000 / 90.0000 | 73.0000 / 73.0000 / 86.0000 |
| 额外片patch1 | 1 | 最小角<5°比例 | 0.2050 / 0.2050 / 0.2100 | 0.1900 / 0.1900 / 0.2100 | 0.1750 / 0.1750 / 0.1800 |
| 额外片patch1 | 1 | q>10比例 | 0.0800 / 0.0800 / 0.0900 | 0.0700 / 0.0700 / 0.0900 | 0.0450 / 0.0450 / 0.0600 |
| 额外片patch1 | 1 | free总面积（无好坏方向） | 38.4018 / 38.4018 / N/A | 35.7421 / 35.7421 / N/A | 35.4595 / 35.4595 / N/A |
| 额外片patch2 | 0 | CF穿刺free比例 | 0.0700 / 0.0700 / 0.1300 | 0.0400 / 0.0400 / 0.0800 | 0.0400 / 0.0400 / 0.0800 |
| 额外片patch2 | 0 | CF穿刺对数 | 8.5000 / 8.5000 / 15.0000 | 4.5000 / 4.5000 / 9.0000 | 4.5000 / 4.5000 / 9.0000 |
| 额外片patch2 | 0 | FF非法对数 | 98.5000 / 98.5000 / 140.0000 | 90.5000 / 90.5000 / 139.0000 | 106.5000 / 106.5000 / 149.0000 |
| 额外片patch2 | 0 | 最小角<5°比例 | 0.1100 / 0.1100 / 0.1500 | 0.1050 / 0.1050 / 0.1100 | 0.1000 / 0.1000 / 0.1100 |
| 额外片patch2 | 0 | q>10比例 | 0.0450 / 0.0450 / 0.0900 | 0.0250 / 0.0250 / 0.0500 | 0.0350 / 0.0350 / 0.0700 |
| 额外片patch2 | 0 | free总面积（无好坏方向） | 58.9346 / 58.9346 / N/A | 52.6288 / 52.6288 / N/A | 50.5952 / 50.5952 / N/A |
| 额外片patch2 | 1 | CF穿刺free比例 | 0.0400 / 0.0400 / 0.0500 | 0.0200 / 0.0200 / 0.0200 | 0.0250 / 0.0250 / 0.0300 |
| 额外片patch2 | 1 | CF穿刺对数 | 7.0000 / 7.0000 / 8.0000 | 2.5000 / 2.5000 / 3.0000 | 3.5000 / 3.5000 / 4.0000 |
| 额外片patch2 | 1 | FF非法对数 | 70.0000 / 70.0000 / 81.0000 | 70.0000 / 70.0000 / 83.0000 | 66.0000 / 66.0000 / 68.0000 |
| 额外片patch2 | 1 | 最小角<5°比例 | 0.1950 / 0.1950 / 0.2000 | 0.1750 / 0.1750 / 0.1900 | 0.2000 / 0.2000 / 0.2100 |
| 额外片patch2 | 1 | q>10比例 | 0.0700 / 0.0700 / 0.0700 | 0.0700 / 0.0700 / 0.0700 | 0.0750 / 0.0750 / 0.0900 |
| 额外片patch2 | 1 | free总面积（无好坏方向） | 34.8185 / 34.8185 / N/A | 32.7396 / 32.7396 / N/A | 33.7798 / 33.7798 / N/A |

离线评价账本（与模型调用分开）：

| 记录项 | 实际值 |
| --- | --- |
| counts | {"model_calls": 0, "geometry_autograd_calls": 0, "optimizer_updates": 0, "OT_calls": 0, "repairs": 0, "input_parent_patch_audits": 6, "input_indexed_mesh_audits": 18, "input_exact_index_calls": 6, "verified_pair_tests": 443196, "primary_pair_tests": 529596, "independent_pair_tests": 443196, "independent_AABB_disjoint": 412224, "static_context_audits": 6, "historical_seam_evaluations": 72, "endpoint_metric_evaluations": 72, "boundary_distance_evaluations": 72, "static_context_audit_reuses": 72, "near_C_fixed_area_sample_diagnostics": 72, "indexed_mesh_audits": 144, "exact_coordinate_indexings": 144, "geometry_repairs": 0, "generated_topology_attributions": 72, "t1_native_confirmation_endpoint_scores": 72} |
| seconds | 63.9148 |
| complete_pairs_per_output | {"CF": 1200, "FF": 4950, "CC_cached": 66} |
Geo_initial：在24份同时有CF/FF记录的输出中，24份检测到所列非法相交。未检出不等于完整合法性认证。
Geo_continue：在24份同时有CF/FF记录的输出中，24份检测到所列非法相交。未检出不等于完整合法性认证。
Geo_M10：在24份同时有CF/FF记录的输出中，24份检测到所列非法相交。未检出不等于完整合法性认证。

以下为已记录诊断的输出级汇总，计数之和不是去重的源缺陷数。

| 诊断字段 | Geo_initial | Geo_continue | Geo_M10 |
| --- | --- | --- | --- |
| CF_coplanar_pair_count | 0.0000 | 0.0000 | 0.0000 |
| CF_unresolved_or_degenerate_pair_count | 0.0000 | 0.0000 | 0.0000 |
| C_current_winding_conflict_count | 208.0000 | 208.0000 | 208.0000 |
| C_inherited_nonmanifold_vertex_count | 16.0000 | 16.0000 | 16.0000 |
| FF_coplanar_pair_count | 0.0000 | 0.0000 | 0.0000 |
| FF_unresolved_or_degenerate_pair_count | 0.0000 | 0.0000 | 0.0000 |
| exact_C_free_shared_edge_count | 0.0000 | 0.0000 | 0.0000 |
| exact_C_free_shared_vertex_count | 0.0000 | 0.0000 | 0.0000 |
| free_near_degenerate_count | 0.0000 | 0.0000 | 0.0000 |
| free_related_current_winding_conflict_count | 0.0000 | 0.0000 | 0.0000 |
| inf_coordinate_count | 0.0000 | 0.0000 | 0.0000 |
| nan_coordinate_count | 0.0000 | 0.0000 | 0.0000 |
| new_free_related_nonmanifold_vertex_count | 0.0000 | 0.0000 | 0.0000 |
| shared_edge_component_count | 2424.0000 | 2424.0000 | 2424.0000 |

C继承缺陷与新增free相关缺陷需按上述独立字段解释；当前绕序冲突不直接等于不可重定向。CF比例只指proper穿刺，FF非法对口径以评价记录为准。执行PASS不是合法网格认证，C恒等来自clamping而非学习成绩；若精确C/free共边为0，不能据vertex-link计数推断水密。
精确谓词认证与连续碰撞安全：NOT_RUN。未记录的其他检查明确保留NOT_RECORDED，不根据轻量距离推断通过。

## 同噪声跨上下文绑定

仅在同模型、同seed、同patch index配对P0/P1上下文；四格分数与逐seed记录保存在results.json的summary.patches。不同patch不混成同一配对。下表Delta0/Delta1/B为均值/中位/最差（越大越好）；单独正向不代表新增损失有益。

| patch | 模型 | n | Delta0 | Delta1 | B | 双向均正次数 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Geo_initial | 8 | 7.2486 / 7.7058 / 4.5955 | 8.2054 / 8.6478 / 4.8051 | 7.7270 / 8.1096 / 4.7003 | 8 |
| 0 | Geo_continue | 8 | 7.2410 / 7.7337 / 3.9961 | 8.1311 / 8.8155 / 3.2108 | 7.6861 / 8.2121 / 3.6034 | 8 |
| 0 | Geo_M10 | 8 | 6.8146 / 7.6606 / 0.9543 | 7.9077 / 8.9998 / 0.4601 | 7.3612 / 8.3463 / 0.7072 | 8 |
| 1 | Geo_initial | 2 | 3.8522 / 3.8522 / 0.7914 | 3.1453 / 3.1453 / 0.1958 | 3.4987 / 3.4987 / 0.4936 | 2 |
| 1 | Geo_continue | 2 | 4.8776 / 4.8776 / 3.0919 | 5.0161 / 5.0161 / 2.3263 | 4.9468 / 4.9468 / 2.7091 | 2 |
| 1 | Geo_M10 | 2 | 3.5448 / 3.5448 / 0.5953 | 3.8425 / 3.8425 / -0.1329 | 3.6936 / 3.6936 / 0.2312 | 1 |
| 2 | Geo_initial | 2 | 4.5670 / 4.5670 / 2.9928 | 3.5335 / 3.5335 / 1.2814 | 4.0502 / 4.0502 / 2.1371 | 2 |
| 2 | Geo_continue | 2 | 3.2997 / 3.2997 / 0.6659 | 3.3104 / 3.3104 / -0.5783 | 3.3050 / 3.3050 / 0.0438 | 1 |
| 2 | Geo_M10 | 2 | 3.3586 / 3.3586 / 0.6628 | 2.8428 / 2.8428 / -0.3605 | 3.1007 / 3.1007 / 0.1511 | 1 |

四格均值：Eij的i为目标父对象、j为提供的上下文；逐噪声完整四格保留在JSON。

| patch | 模型 | E00 | E01 | E10 | E11 |
| --- | --- | --- | --- | --- | --- |
| 0 | Geo_initial | 0.7014 | 7.9500 | 8.6882 | 0.4828 |
| 0 | Geo_continue | 0.7935 | 8.0345 | 8.6877 | 0.5566 |
| 0 | Geo_M10 | 1.1371 | 7.9517 | 8.4332 | 0.5255 |
| 1 | Geo_initial | 3.1869 | 7.0390 | 4.5543 | 1.4090 |
| 1 | Geo_continue | 2.1024 | 6.9800 | 5.7842 | 0.7681 |
| 1 | Geo_M10 | 3.4937 | 7.0385 | 4.5814 | 0.7389 |
| 2 | Geo_initial | 2.6004 | 7.1674 | 5.7609 | 2.2275 |
| 2 | Geo_continue | 3.7496 | 7.0493 | 4.7303 | 1.4200 |
| 2 | Geo_M10 | 3.7862 | 7.1448 | 4.8291 | 1.9862 |

绑定比较沿用同patch/seed四格记录，以下B变化越大越好：

| patch | 比较 | 配对B差均值 | 配对B差中位 | B提高次数 |
| --- | --- | --- | --- | --- |
| 0 | Geo_M10−Geo_continue | -0.3249 | 0.0229 | 5 |
| 0 | Geo_continue−Geo_initial | -0.0410 | 0.1198 | 6 |
| 1 | Geo_M10−Geo_continue | -1.2532 | -1.2532 | 0 |
| 1 | Geo_continue−Geo_initial | 1.4481 | 1.4481 | 2 |
| 2 | Geo_M10−Geo_continue | -0.2043 | -0.2043 | 1 |
| 2 | Geo_continue−Geo_initial | -0.7452 | -0.7452 | 1 |

## 训练损失与参数确实更新

以下均为预定训练步原日志，没有额外teacher或模型前向。FC未应用以NOT_APPLIED标记，不能将其写成一个实测零损失；FF始终NOT_APPLIED。

| 臂 | 全部FC组分母累计 | 符合t门槛的FC组累计 | 实际应用FC组累计 | 全inactive批数 |
| --- | --- | --- | --- | --- |
| Geo_continue | 35998 | 17622 | NOT_APPLIED | 2 |
| Geo_M10 | 35998 | 17622 | 17622 | 2 |
| 臂 | 新增步 | FM | FC | 0.1FC | 总loss | FC全组分母 | 激活FC组 | 裁剪前总梯度L2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Geo_continue | 1 | 0.0005 | NOT_APPLIED | NOT_APPLIED | 0.0005 | 72 | NOT_RECORDED | 0.0803 |
| Geo_continue | 100 | 0.0002 | NOT_APPLIED | NOT_APPLIED | 0.0002 | 72 | NOT_RECORDED | 0.0680 |
| Geo_continue | 250 | 0.0001 | NOT_APPLIED | NOT_APPLIED | 0.0001 | 72 | NOT_RECORDED | 0.0469 |
| Geo_continue | 500 | 0.0004 | NOT_APPLIED | NOT_APPLIED | 0.0004 | 70 | NOT_RECORDED | 0.1356 |
| Geo_M10 | 1 | 0.0005 | 2.651e-06 | 2.651e-07 | 0.0005 | 72 | 18 | 0.0803 |
| Geo_M10 | 100 | 0.0002 | 1.65e-05 | 1.65e-06 | 0.0002 | 72 | 42 | 0.1361 |
| Geo_M10 | 250 | 0.0001 | 1.561e-05 | 1.561e-06 | 0.0001 | 72 | 64 | 0.0335 |
| Geo_M10 | 500 | 0.0004 | 3.842e-06 | 3.842e-07 | 0.0004 | 70 | 22 | 0.1486 |

参数组梯度处于累积完成、global clip之前；更新量为optimizer.step前后真实差。

| 臂 | 步 | 参数组 | 梯度RMS | 实际更新RMS | 梯度非零数 | 更新非零数 | 有限 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Geo_continue | 1 | role_embedding | 7.301e-06 | 9.504e-06 | 1536 | 1536 | True |
| Geo_continue | 1 | backbone.x_embedder.mlp.0.weight | 7.47e-06 | 9.899e-06 | 94456 | 94456 | True |
| Geo_continue | 1 | backbone.final_layer.linear.weight | 0.0015 | 9.82e-06 | 2304 | 2304 | True |
| Geo_continue | 1 | context_encoder.mlp | 1.171e-08 | 3.427e-06 | 17920 | 17906 | True |
| Geo_continue | 1 | context_encoder.W_self | 3.12e-09 | 1.764e-06 | 16511 | 16495 | True |
| Geo_continue | 1 | context_encoder.W_neighbor | 3.12e-09 | 1.764e-06 | 16511 | 16490 | True |
| Geo_continue | 1 | context_encoder.norm | 9.39e-09 | 4.318e-06 | 256 | 256 | True |
| Geo_continue | 1 | context_encoder.W_out | 5.804e-07 | 8.791e-06 | 98299 | 98299 | True |
| Geo_continue | 100 | role_embedding | 7.775e-06 | 1.791e-06 | 1536 | 1536 | True |
| Geo_continue | 100 | backbone.x_embedder.mlp.0.weight | 7.784e-06 | 1.809e-06 | 94458 | 94427 | True |
| Geo_continue | 100 | backbone.final_layer.linear.weight | 0.0012 | 2.48e-06 | 2304 | 2303 | True |
| Geo_continue | 100 | context_encoder.mlp | 1.47e-08 | 1.458e-06 | 17920 | 17906 | True |
| Geo_continue | 100 | context_encoder.W_self | 5.832e-09 | 1.065e-06 | 16512 | 16485 | True |
| Geo_continue | 100 | context_encoder.W_neighbor | 5.832e-09 | 1.065e-06 | 16512 | 16486 | True |
| Geo_continue | 100 | context_encoder.norm | 1.552e-08 | 1.067e-06 | 256 | 256 | True |
| Geo_continue | 100 | context_encoder.W_out | 5.293e-07 | 1.849e-06 | 98304 | 98304 | True |
| Geo_continue | 250 | role_embedding | 3.947e-06 | 2.466e-06 | 1536 | 1536 | True |
| Geo_continue | 250 | backbone.x_embedder.mlp.0.weight | 3.995e-06 | 2.397e-06 | 94455 | 94442 | True |
| Geo_continue | 250 | backbone.final_layer.linear.weight | 0.0009 | 1.803e-06 | 2304 | 2304 | True |
| Geo_continue | 250 | context_encoder.mlp | 4.145e-09 | 2.307e-06 | 17919 | 17905 | True |
| Geo_continue | 250 | context_encoder.W_self | 1.422e-09 | 1.66e-06 | 16512 | 16494 | True |
| Geo_continue | 250 | context_encoder.W_neighbor | 1.422e-09 | 1.66e-06 | 16512 | 16493 | True |
| Geo_continue | 250 | context_encoder.norm | 2.027e-09 | 2.866e-06 | 256 | 255 | True |
| Geo_continue | 250 | context_encoder.W_out | 2.206e-07 | 2.469e-06 | 98302 | 98303 | True |
| Geo_continue | 500 | role_embedding | 7.02e-06 | 1.013e-06 | 1536 | 1536 | True |
| Geo_continue | 500 | backbone.x_embedder.mlp.0.weight | 7.646e-06 | 1.167e-06 | 94462 | 94416 | True |
| Geo_continue | 500 | backbone.final_layer.linear.weight | 0.0024 | 2.657e-06 | 2304 | 2304 | True |
| Geo_continue | 500 | context_encoder.mlp | 1.079e-08 | 6.412e-07 | 17919 | 17883 | True |
| Geo_continue | 500 | context_encoder.W_self | 4.406e-09 | 4.402e-07 | 16512 | 16439 | True |
| Geo_continue | 500 | context_encoder.W_neighbor | 4.406e-09 | 4.402e-07 | 16512 | 16434 | True |
| Geo_continue | 500 | context_encoder.norm | 1.265e-08 | 6.073e-07 | 256 | 253 | True |
| Geo_continue | 500 | context_encoder.W_out | 3.283e-07 | 9.681e-07 | 98295 | 98304 | True |
| Geo_M10 | 1 | role_embedding | 7.314e-06 | 9.504e-06 | 1536 | 1536 | True |
| Geo_M10 | 1 | backbone.x_embedder.mlp.0.weight | 7.473e-06 | 9.899e-06 | 94461 | 94461 | True |
| Geo_M10 | 1 | backbone.final_layer.linear.weight | 0.0015 | 9.82e-06 | 2304 | 2304 | True |
| Geo_M10 | 1 | context_encoder.mlp | 1.171e-08 | 3.429e-06 | 17920 | 17915 | True |
| Geo_M10 | 1 | context_encoder.W_self | 3.124e-09 | 1.765e-06 | 16512 | 16488 | True |
| Geo_M10 | 1 | context_encoder.W_neighbor | 3.124e-09 | 1.765e-06 | 16512 | 16488 | True |
| Geo_M10 | 1 | context_encoder.norm | 9.419e-09 | 4.324e-06 | 256 | 256 | True |
| Geo_M10 | 1 | context_encoder.W_out | 5.813e-07 | 8.792e-06 | 98299 | 98299 | True |
| Geo_M10 | 100 | role_embedding | 9.601e-06 | 1.793e-06 | 1536 | 1536 | True |
| Geo_M10 | 100 | backbone.x_embedder.mlp.0.weight | 9.65e-06 | 1.833e-06 | 94461 | 94431 | True |
| Geo_M10 | 100 | backbone.final_layer.linear.weight | 0.0023 | 2.663e-06 | 2304 | 2304 | True |
| Geo_M10 | 100 | context_encoder.mlp | 1.574e-08 | 1.527e-06 | 17920 | 17911 | True |
| Geo_M10 | 100 | context_encoder.W_self | 6.207e-09 | 1.157e-06 | 16512 | 16499 | True |
| Geo_M10 | 100 | context_encoder.W_neighbor | 6.207e-09 | 1.157e-06 | 16512 | 16498 | True |
| Geo_M10 | 100 | context_encoder.norm | 1.559e-08 | 1.257e-06 | 256 | 254 | True |
| Geo_M10 | 100 | context_encoder.W_out | 5.6e-07 | 1.826e-06 | 98304 | 98302 | True |
| Geo_M10 | 250 | role_embedding | 3.35e-06 | 2.532e-06 | 1536 | 1536 | True |
| Geo_M10 | 250 | backbone.x_embedder.mlp.0.weight | 3.594e-06 | 2.445e-06 | 94450 | 94448 | True |
| Geo_M10 | 250 | backbone.final_layer.linear.weight | 0.0006 | 1.973e-06 | 2304 | 2304 | True |
| Geo_M10 | 250 | context_encoder.mlp | 5.169e-09 | 2.386e-06 | 17920 | 17911 | True |
| Geo_M10 | 250 | context_encoder.W_self | 1.909e-09 | 1.771e-06 | 16512 | 16494 | True |
| Geo_M10 | 250 | context_encoder.W_neighbor | 1.909e-09 | 1.771e-06 | 16512 | 16500 | True |
| Geo_M10 | 250 | context_encoder.norm | 5.27e-09 | 2.846e-06 | 256 | 255 | True |
| Geo_M10 | 250 | context_encoder.W_out | 2.151e-07 | 2.516e-06 | 98299 | 98304 | True |
| Geo_M10 | 500 | role_embedding | 8.617e-06 | 1.09e-06 | 1536 | 1536 | True |
| Geo_M10 | 500 | backbone.x_embedder.mlp.0.weight | 8.871e-06 | 1.164e-06 | 94460 | 94406 | True |
| Geo_M10 | 500 | backbone.final_layer.linear.weight | 0.0025 | 2.201e-06 | 2304 | 2304 | True |
| Geo_M10 | 500 | context_encoder.mlp | 1.338e-08 | 7.054e-07 | 17920 | 17861 | True |
| Geo_M10 | 500 | context_encoder.W_self | 4.881e-09 | 5.059e-07 | 16512 | 16461 | True |
| Geo_M10 | 500 | context_encoder.W_neighbor | 4.881e-09 | 5.059e-07 | 16512 | 16459 | True |
| Geo_M10 | 500 | context_encoder.norm | 1.286e-08 | 8.18e-07 | 256 | 252 | True |
| Geo_M10 | 500 | context_encoder.W_out | 4.677e-07 | 1.004e-06 | 98302 | 98304 | True |

本轮没有重新记录逐层有效残差舍入比例；该项NOT_RUN，不沿用上一轮数值冒充本轮观测。

## 固定视角原始输出

固定首噪声9801；patch0和patch1各一图，P0/P1×三模型，每模型full/free两列。同父对象跨两图使用统一范围和视角。橙色C12、蓝色free100；无投影、焊接或后处理。图用于检查，不新增或替代数值评分。

![固定seed9801 patch0 full/free对比](comparison_seed9801_patch0_full_free.png)

![固定seed9801 patch1 full/free对比](comparison_seed9801_patch1_full_free.png)

两张PNG已由主代理实际查看：full/free标签、已知C着色及统一视角/范围可读，无面板裁切；固定图不能替代全部样本统计。

## 资产、源码与交付

本轮起点与新终点仅引用其路径和hash，没有复制官方EMA或旧数据包。

| 角色 | 路径 | 文件SHA256 | state SHA256 |
| --- | --- | --- | --- |
| Geo_initial | /mnt/e/MeshFlow/native_t1/runs/geometry_context_revised_20260930_153000/T_geo/additional_step500_cumulative_step2000.pt | c52e102530fe8d50d9f90e5d9df7a5dadff89655b792a019305d373cf83a0571 | 1742a8aad06ed082dd7d21f91829f88d3ef118e0cfcf1d3c4cd8a6c853b3bf4e |
| Geo_continue | /mnt/e/MeshFlow/experiments/mf_h1_local0/outputs/tgeo_m10_20260930_173200/Geo_continue/additional_step500_cumulative_step2500.pt | 33cb1331f4c1bad9eb676dd552f0b5fafa0644cdec2b674647c5c6728d96aef3 | 6fa39d1c8c1deade17467075160281dcf01948c45e90ca4a9ad111a7dd97c392 |
| Geo_M10 | /mnt/e/MeshFlow/experiments/mf_h1_local0/outputs/tgeo_m10_20260930_173200/Geo_M10/additional_step500_cumulative_step2500.pt | b91eb389a234050287d32d19a5278c27254fed7709d9bc78ea390086a9dbf62c | e8b66564faaf18225a4998d84f2020a5e12097edba83a5621466cd66c6fed772 |

协议SHA256：`ee17f0550a8eaf0710cca1d0c0a66746cee8766de48fb6879c3348804fa13cdd`；公开代码基准：`5ef35b421ad00ccd3a97867ce0e33e5f77f65b1c`。
完整输入/噪声/源码hash、原始72输出、逐样本指标、训练流和门禁均保存在results.json及其引用文件。代码diff若提供见同目录code.diff；未提供时保持NOT_RECORDED，不伪造交付。
已达到授权预算后停止。不自动扩大数据、增加更新、搜索FC权重/时间门槛、训练新分支或启动Phase3。


最终完整性核对：1000新增更新；11652前向、8002反向；72正式完整生成+1逐位一致重放。74次调度包括1次未进入模型的I/O失败，原始失败快照与恢复源码均保留。文件锁原因是基于同时读取与原子替换失败的推断，未做OS句柄追踪。5份实验/恢复源码快照、4个本轮检查点、72原始数组、72评分文件及全部已登记原资产hash核对通过；公开Git工作树仍干净，HEAD未变，没有自动提交或推送。
