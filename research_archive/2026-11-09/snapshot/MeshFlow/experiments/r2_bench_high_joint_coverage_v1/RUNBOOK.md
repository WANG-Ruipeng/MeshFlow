# 执行与恢复入口

实验独立目录：`experiments/r2_bench_high_joint_coverage_v1`。
代码：`r2/pilots/bench_high_joint_coverage_v1`。
使用现有WSL环境 `/mnt/e/MeshFlow-Control-artifacts/venv/bin/python`，工作目录 `/mnt/e/MeshFlow`，命令均加 `-B`。

只读状态：

```text
python -B -m r2.pilots.bench_high_joint_coverage_v1.workflow
python -B -m r2.pilots.bench_high_joint_coverage_v1.monitor
```

训练前条件是数据、流、模型恢复、运行合同、评测缓存与workflow CPU门槛PASS，并冻结`training_registration.json`。根进程在实际测量资源后创建独立`dispatch/WIDE_HIGH_go.json`，再启动唯一1500步正式训练。`dispatch_train.py`不可当作自动重试入口。

手动单阶段入口：

```text
python -B -m r2.pilots.bench_high_joint_coverage_v1.workflow --launch train-wide
python -B -m r2.pilots.bench_high_joint_coverage_v1.workflow --launch sample-b
python -B -m r2.pilots.bench_high_joint_coverage_v1.workflow --launch sample-w
python -B -m r2.pilots.bench_high_joint_coverage_v1.workflow --launch d0-b
python -B -m r2.pilots.bench_high_joint_coverage_v1.workflow --launch d0-w
python -B -m r2.pilots.bench_high_joint_coverage_v1.workflow --launch prepare-scores
python -B -m r2.pilots.bench_high_joint_coverage_v1.workflow --launch score-raw
python -B -m r2.pilots.bench_high_joint_coverage_v1.workflow --launch collect
```

这些命令不是无需检查的批处理脚本。每次GPU派发重新测资源，所有54份新RAW完成才放行D0。已占用任务目录不会自动重启；进程PID及创建时间匹配后才判断仍存活。失败、残留或中断必须先核对调用账本、已提交更新与完整恢复点，不能删除状态重新跑或把未完成更新算完成。本轮没有自动科学回放额度。

本轮POST CPU评分实际采用独立CLI，B与W各有自己的输出写入者，没有使用工作流中的`score-post`任务：

```text
python -B -m r2.pilots.bench_high_joint_coverage_v1.evaluation post --models B --workers 2
python -B -m r2.pilots.bench_high_joint_coverage_v1.evaluation post --models W --workers 3
```

它们是本轮已完成的执行历史，不是要求再次评分。工作流中未派发的`score-post`不能被误解为评价缺失，最终以`post_complete_full.json`和完整评分/核验收据为准。

训练每250步更新滚动完整恢复状态并保存hash/收据，不承诺永久保留全部中间大权重；固定1500为主终点，最终完整及推理文件保留。恢复应明确记录checkpoint、随机状态、AdamW、计划游标、实际已调用的forward/backward/update，不修改原预算与数据。已完成任务再次进入仅验证现有产物，不追加更新。

报告代码的`prepare/render/tables/visual`分别负责冻结显示、CPU渲染、完整数值表及显式实际图审汇总；只有真正打开图后才能通过`review_record`提交标签。评分与图审不可由程序互相推断。完整后验核验使用`evaluation_closeout complete`，代码归档与包由`delivery`完成。

原窄池B38份输出、评分、相机及噪声只读复用。新B8与W46在本目录保存。四份GEO_ONLY对照在`backend_geo_only`单独记录，入口`geo_only`，不计作新生成器或D0调用。具体命令、状态及恢复细节以该子任务记录为准。

最终报告落盘后停止；不切换默认，不清理权重/数据，不commit/push。

最终只读核验（不调用模型、不生成或重新评分）：

```text
python -B -m r2.pilots.bench_high_joint_coverage_v1.evaluation_closeout complete
python -B -m r2.pilots.bench_high_joint_coverage_v1.geo_only verify
```

`REPORT.md`为结果判断，`COMPLETE.json`为固定任务封口；`model_index.json`列出真实模型和恢复权重。`package_receipt.json`保存审阅包及包内逐文件hash。重建数据/训练/采样不是收口后的自动动作。
