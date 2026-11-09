# 执行与恢复

本轮总入口为 `experiments/mf_h1_local0/chair_hybrid_coupling_v1/run_all.py`。实验固定为 OT_BASE / OT_HYBRID 各 1000 次更新，不根据评价结果延长。

在 Windows PowerShell、工作目录 `E:\MeshFlow` 中运行：

```powershell
wsl -e sh -c 'cd /mnt/e/MeshFlow && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.chair_hybrid_coupling_v1.run_all --resume --through all'
```

入口核对协议、源码和输入哈希，跳过已完成阶段；控制器锁和既有 GPU 锁防止重复启动。完整起点、AdamW 状态、训练游标和 RNG 必须一起恢复。实际调用以 `ledger/*.jsonl` 为准，汇总见 `accounting.json`。

安全中断在完成当前更新并保存完整状态后记录 `INTERRUPTED_SAFE`。真实错误或无法对应完整检查点的中断记录为 `BLOCKED`；入口拒绝自动重放失败调用，不能手工删除状态文件来凑满预算。正常完成后再次进入会核验和复用结果，不新增模型调用。

`registration/task_prompt.txt` 保存用户原始任务。`protocol.json`、`protocol.sha256`、`registration/implementation_hashes.json` 和 `code.diff` 保留冻结定义。`checkpoint_manifest.json` 在全部阶段完成时列出原起点引用和两个固定终点；新 pilot1000/total5000 与历史其他实验的 total5000 不同源。

本轮总报告为 `CHAIR_HYBRID_COUPLING_REPORT.md`；逐输出、逐父、逐噪声和完整矩阵在 `paired_results/`，响应逐步数据在 `diagnostic_results/`，真实轨迹和速度在 `evaluation/<label>/`。解析 No-response/Post-scale 仅是 CPU 参照，未替换 raw 模型输出。

冻结报告代码有局部中文编码显示损坏，另设 `reporting_corrections/render_corrected_report.py` 只恢复文案，并保留原报告及前后哈希。此修正不改变科学定义、数据表、判定逻辑或任何训练/评价代码。
