# 固定任务与恢复入口

本实验为500步候选，既有500步SHAPE_D0只读对照；不用从该权重续训。所有科学输入沿上一轮manifest，新增参数固定在loss_config.json。

WSL环境：`/mnt/e/MeshFlow-Control-artifacts/venv/bin/python`，工作目录`/mnt/e/MeshFlow`。

统一入口：`python -B -m r2.pilots.bench_backend_relative_deformation_v1.run resume`。最终完成后`run verify`只校验，不自动训练/重新评分。

训练中发生失败须先检查ledger和完整快照；入口拒绝静默回滚/重算。只有完整边界且证明零未记账更新才可记录工程修复后显式恢复。无此证据即停住。完成receipt存在时校验复用，不因查看进度启动第二任务。

预算：500×4训练前向/反向；RAW30与SOURCE30候选单次D0；无生成器调用/新噪声。原官方与上轮适配提议/评分直接复用。CPU新增真实30GEO/60评分、来源15identity GEO、30六列图与15来源对照图。实际AI_PROXY图审和独立审计后封存，不延长、不切默认、不清理旧资产、不commit/push。

原数据和评分定义不修改。代码、统计和执行错误分别记录；训练曲线和来源27只作说明，正式判定用全组来源及原30RAW配对。
