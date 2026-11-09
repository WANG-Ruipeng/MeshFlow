# 运行、核验与恢复

实验已完成。默认仅做无模型调用的完整性核验：

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B r2/pilots/seam_occupancy_rollout_v1/closeout.py --verify
```

原阶段入口如下。每阶段会核验冻结输入、保护hash和已有receipt；已完成请求跳过。不要删除receipt来触发重跑，不要改manifest。部分失败有attempt而无完成receipt时，入口拒绝隐式重试，须先检查失败日志与剩余授权预算。

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/mnt/e/MeshFlow/r2/src:/mnt/e/MeshFlow
PY=/mnt/e/MeshFlow-Control-artifacts/venv/bin/python
$PY -B -m r2.pilots.seam_occupancy_rollout_v1.run --stage status
# 以下为原执行入口，不代表授权额外采样；本轮全部完成。
$PY -B -m r2.pilots.seam_occupancy_rollout_v1.run --stage generate --arm ALL_ON
$PY -B -m r2.pilots.seam_occupancy_rollout_v1.run --stage generate --arm SOURCE_ORACLE
# post入口强制全部32份RAW屏障
$PY -B -m r2.pilots.seam_occupancy_rollout_v1.run --stage post --arm ALL_ON
$PY -B -m r2.pilots.seam_occupancy_rollout_v1.run --stage post --arm SOURCE_ORACLE
$PY -B -m r2.pilots.seam_occupancy_rollout_v1.run --stage capacity
$PY -B -m r2.pilots.seam_occupancy_rollout_v1.evaluate --stage score --workers 2
$PY -B -m r2.pilots.seam_occupancy_rollout_v1.evaluate --stage aggregate
```

任何需要恢复的GPU波次仍须先按AGENTS检查实际资源。旧PREDICTED来自只读旧实验，不能另跑替换缓存。SOURCE_ORACLE依赖来源占用标签，仅供诊断。

`closeout.py` 只用Python标准库，整理已有CSV/JSON、实际图审记录及代码快照，不导入模型、不评分、不重新渲染。`--verify` 只读核验最终文件hash与预算。评阅记录是已实际看图后的人工式转录，不是自动指标分类。

小ZIP包含报告、逐输出/逐父/逐噪声/逐槽CSV、16张联系图、实际评分源码、hash、审计和账本；不含权重及完整NPZ/PLY。完整图集见本地 `gallery/index.html`，完整资产仍在 `evaluation/` 与 `capacity/`。
