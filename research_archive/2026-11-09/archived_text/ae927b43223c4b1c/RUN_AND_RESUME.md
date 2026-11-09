# 已完成实验的检查与恢复入口

本轮已经完成固定预算。默认入口只读状态，不自动训练、生成、重新评分或延长。旧 rolling.pt 已在固定终点核验后退役；保留 step1000/step3000 完整模型、AdamW、RNG 和登记身份。step1000 只是历史恢复证据，不能在已消费到3000步的原目录静默回退。

在 Windows PowerShell，以下命令只查看状态：

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- env PYTHONPATH=/mnt/e/MeshFlow/r2/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_control_transfer_v1.resume --stage status
```

实现目录：`E:\MeshFlow\r2\pilots\bench_control_transfer_v1`。

入口按显式 `--stage` 分为 `train/sample/post/score/gallery`。训练、采样和后处理阶段必须提供 `--arm PLAIN` 或 `--arm SEAM`，并受冻结登记与资源余量检查约束。现有训练/RAW/POST完成收据会返回已完成结果，不追加预算。不应为了“确认能跑”重启已完成阶段。

若未来一个**新授权任务**使用这些权重，应另建目录和登记；不得修改本次 `registration.json`、噪声、原评分/图审或原终点。未完成更新/模型调用的恢复必须先查调用账本，禁止静默重放。此说明不授权继续训练。

关键恢复资产：

- `training/{PLAIN,SEAM}/step3000.pt`：主终点及匹配优化器/RNG。
- `training/{PLAIN,SEAM}/step1000.pt`：固定中间证据，未用于挑最好结果。
- `data/manifest.json`、`input_schedule.json`、`registration.json`：冻结训练身份与流。
- `evaluation_inputs/manifest.json` 及其中指定的每请求噪声数组：完整固定推理输入；准确实际文件名以 manifest 为准。
- `evaluation/{arm}/{request}/raw/`：RAW、51个Euler状态、调用与锚点检查。
- `evaluation/{arm}/{request}/post/`：原D0提议、pre_geo、索引POST、原角序恢复和逐面清理原因。
- `evaluation/scores/`：原124字段、合法图及非法面对。
- `source_snapshot/`、`scoring_source_snapshot/`：交付时源码及实际评分源码；`delivery/source_hashes.json` 记录身份。

CPU汇总入口分别是 `paired_summary.py`、`numerical_review.py`、`closeout.py`。它们读取保存结果，不调用模型。不要把重建汇总描述为独立重新训练或重新生成。

环境与依赖使用当前已安装的 WSL venv。`source_snapshot` 是本轮实现/登记依赖快照，不是完整环境镜像；官方模型不放入小审阅包。大权重位置与SHA见主报告。
