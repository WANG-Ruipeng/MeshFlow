# 恢复入口

已完成，不重复计算。使用WSL既有环境，PYTHONPATH=/mnt/e/MeshFlow/r2/src:/mnt/e/MeshFlow。

1. python -B -m r2.tools.d0_restore_quality_v1.run --stage run --workers 2 （命中完整receipt只核验）
2. python -B -m r2.tools.d0_restore_quality_v1.finish （独立汇总恢复；不要用旧run --stage aggregate/all）
3. python -B -m r2.tools.d0_restore_quality_v1.render （读取既有图）
4. python -B -m r2.tools.d0_restore_quality_v1.finalize （核验已记录图审/报告）

发现未完成cell保留attempt后停止，不隐式重评分；缺缓存BLOCKED，不模型重生成。
