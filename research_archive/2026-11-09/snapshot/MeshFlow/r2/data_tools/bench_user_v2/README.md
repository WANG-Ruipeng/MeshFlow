# Bench USER v2 数据制作入口

本模块只制作和验证数据，不导入生成器、不训练，也不选择默认模型。

当前冻结批次位于 `experiments/r2_bench_user_data_v2/`。先读 `REPORT.md` 和 `DATASET_READY.json`；仅 `manifest.train` 加其匹配的 PASS 验收收据可作为已通过的样本。`candidate_manifest.json` 中有未通过目标，不能整体作为训练池。

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.data_tools.bench_user_v2.run verify
```

分阶段恢复：`build`（已有冻结32父）、`render`（现存图只核验）、`finalize`（必须已有实际图审）、`validate`（已有成功收据只验hash，未完成项一次CPU OT）、`closeout`。运行验证器会追加审计日志；封存后日常只运行 `verify`，避免改变已封存的日志快照。

同父所有条件共享一个目标版本。原始坐标、来源面ID、角序逆映射、去零/归并/细分多对多血缘、owner/alias/bulk分区均保留。`input.npz` 不含来源补集或owner标签；`case.npz` 是监督文件，不能作为C-only推理输入。

副本单层化和低N细分只在未通过的提案中发生，本批正式通过者不构成这两项修复有效性的证据。没有全局水密要求，但已发现非法面对、非流形边或未决关系的完整目标不得进入当前通过池。闭合/无owner输入的适配代码保留，当前批次没有此类验收实例。

新父、修改规则或新的人工区域选择应放入新版本，不能覆盖本批。制作采样政策不授权训练；需另行冻结训练预算和评测。
