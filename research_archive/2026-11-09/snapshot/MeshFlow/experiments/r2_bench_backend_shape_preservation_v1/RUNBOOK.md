# 执行与恢复

总入口（从Windows工作区执行）：

```powershell
wsl.exe -d Ubuntu -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_backend_shape_preservation_v1.run resume
```

训练和各后端前向只有一次固定预算。已经有complete receipt的阶段不重复调用；未完成的模型attempt明确停止，不能删除failure后重跑。训练每100更新保存完整AdamW/RNG；如真实中断，须先核对ledger与检查点、明确记录需重算的更新，才允许单独恢复，不自动回滚。正常运行期间不要另开总入口或更换监督进程。

GPU训练只有一臂500更新；训练实测后余量不满足保守的额外推理slot，故不同时加载第二个GPU模型。CPU评分最多3 worker，每进程1线程；source保形CPU分析和图集可与独立CPU阶段重叠。评分/几何模块会隐藏CUDA，必须单独进程，不能在模型进程导入。

完成后只读核验：

```powershell
wsl.exe -d Ubuntu -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_backend_shape_preservation_v1.run verify
```

计算完成不等于图审完成。最终COMPLETE由实际图审、全表/账本核验、报告和哈希封存之后写入。原数据、原报告、官方模型、生成器和默认CLI完全保留；本轮无自动commit/push。

## 第100步检查点后的明确边界恢复

原训练完成100次更新、400次完整模型前向及400次反向，并成功写出完整 `training/checkpoints/step0100.pt` 及其SHA收据后，在写 `CHECKPOINT_SAVED` 日志时退出：`event` 的首参数 `path` 与保存收据中的 `path` 关键字重名。这是保存完成之后的日志异常，账本中没有第101步或未完成的科学计算。

旧源码原样保存在 `code_before_boundary_fix.py`（SHA256 `fd011dbb01070c9c1bfebb06727d746fd17dc0f975b834d6a90956230227ccd7`），原 `failure.json`、`attempt.json`、初始 `PRETRAIN_READY.json` 和任务协议均保留。修正只将日志形参改为 `ledger_path`，并加入限定本次错误的显式边界恢复；新代码SHA256为 `d034d24a7125af6ff8e65db5e7d8c1582fa756d2ac6b79a122e706247b7c81f9`。完整工程核验在 `engineering_amendment.json`。

已在CPU上以 `weights_only=True` 核验保存快照：296个FP32模型张量、296项AdamW状态，全部优化器步数为100，完整Python／NumPy／Torch CPU／CUDA RNG状态均在；此检查未实例化模型或初始化CUDA。

本次唯一允许的恢复命令已单独派发，**运行期间不要再次执行**：

```powershell
wsl.exe -d Ubuntu -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_backend_shape_preservation_v1.runtime train --resume-boundary
```

恢复入口先验证原失败类型、400/400/100逐微批账本、快照SHA及科学合同，只恢复一次，从第101步至预定第500步，剩余1600次前向、1600次反向、400次更新，重算为0。模型、AdamW与四类RNG恢复一致性记录在 `training/resume_boundary_loaded.json`；恢复资源和实际账本另外保留。若恢复本身失败，写入独立 `resume_boundary_failure.json`，不覆盖原失败，也不再自动恢复。

总入口只接受上述精确旧／新SHA与对应工程记录；它不会自动调用边界恢复或放宽到其他代码更改。恢复完成后继续原来的90次固定后端验证与CPU评分预算。
