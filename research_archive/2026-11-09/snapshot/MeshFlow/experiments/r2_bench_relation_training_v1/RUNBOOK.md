# Bench 关系覆盖补充训练

独立代码：`r2/pilots/bench_relation_training_v1`。本文件是执行说明，不能单独证明任务已完成；实际状态以 `workflow/state.json` 及 `training/RELATION_V1_SEAM/status.json` 为准。

先完成 CPU 输入/收据核对和注册：

```text
python -B -m r2.pilots.bench_relation_training_v1.prepare
python -B -m r2.pilots.bench_relation_training_v1.register
python -B -m r2.pilots.bench_relation_training_v1.training --check-cpu
python -B -m r2.pilots.bench_relation_training_v1.eval_plan
```

确认冻结合同与当前资源后，只启动一次：

```text
python -B -m r2.pilots.bench_relation_training_v1.workflow --execute
```

推荐通过持续持有的 exec 会话前台运行上述命令；不要关闭所属WSL监督进程。可选后台 `--launch` 必须核实监督进程持久性，不能与前台重复派发。流水线依次执行资源检查、1500步、48 RAW、48单D0与RAW CPU评分、统一后处理评分、逐例完整配对表、144页三模型同位置截面。自动终点 `NUMERICS_AND_GALLERY_COMPLETE_PENDING_ACTUAL_REVIEW` 仍要求实际图审和报告。

loss每步写 `training/RELATION_V1_SEAM/training_log.jsonl`；实例曝光 `exposure.jsonl`，实际调用 `calls.jsonl`。`data/inventory.json` 保存规模/大小，`data/exposure_plan.json` 保存预定曝光。主对照USER_V2_S1500，次对照A_S1500均只读复用。

只读检查恢复状态：`python -B -m r2.pilots.bench_relation_training_v1.resume`。只有 INTERRUPTED_SAFE 且完整检查点/游标一致才允许同入口 `--execute`；失败、OOM、部分更新不能自动回滚。不要第二次启动workflow、删除state或停止用户其它进程。
