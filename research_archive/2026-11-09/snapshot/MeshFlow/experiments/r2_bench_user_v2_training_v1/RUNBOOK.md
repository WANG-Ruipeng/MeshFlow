# Bench USER v2 训练与终点评测

**完成更新（2026-10-21）：** 1500步、固定48请求三阶段评价及实际AI_PROXY图审已经结束。最终状态以 [CLOSEOUT.json](CLOSEOUT.json) 为准，结果见 [REPORT.md](REPORT.md)。以下保留原启动/恢复说明；旧workflow的等待图审状态不表示仍需启动任务。没有后续自动训练。

本轮已于2026-10-21启动。当前状态以 [workflow/state.json](workflow/state.json) 和 [training/USER_V2_SEAM/status.json](training/USER_V2_SEAM/status.json) 为准；本说明不代表训练或评价已经完成。

用户授权使用新封存的25父／81条数据，单臂1500更新，COMMON12000共同起点、fresh AdamW、batch8/micro1。完整合同见 [PROTOCOL.md](PROTOCOL.md)，注册见 [training_registration.json](training_registration.json)。同一日程中每父480次实例曝光，真实曝光以日志为准。

后台监督进程独立于聊天工具调用运行。阶段顺序为：资源释放检查 → 训练1500步 → 固定48请求RAW → 单次官方D0（与独立RAW CPU评分重叠）→ 统一POST级联与评分 → 逐输出/逐父完整配对表 → 同尺度两视角及固定front/side位置截面图。

每步loss与有效分母写入 [training_log.jsonl](training/USER_V2_SEAM/training_log.jsonl)，逐实例写入 [exposure.jsonl](training/USER_V2_SEAM/exposure.jsonl)，每次真实模型调用和更新另有 [calls.jsonl](training/USER_V2_SEAM/calls.jsonl)。检查点位于同目录；正式终点只用step1500，另保存0/500和250步滚动安全点。运行时不删除旧权重。

自动流水线最后状态为 `NUMERICS_AND_GALLERY_COMPLETE_PENDING_ACTUAL_REVIEW`；这表示数值与图已生成，**不表示图片已被实际审阅**。真实图审与研究报告需在图像产生后完成，不能用指标自动写视觉标签。

## 查看进度

```powershell
Get-Content experiments/r2_bench_user_v2_training_v1/workflow/state.json -Encoding UTF8
Get-Content experiments/r2_bench_user_v2_training_v1/training/USER_V2_SEAM/status.json -Encoding UTF8
Get-Content experiments/r2_bench_user_v2_training_v1/workflow/training.log -Tail 3 -Encoding UTF8
```

只读恢复状态检查：

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_user_v2_training_v1.resume
```

只有 `INTERRUPTED_SAFE` 且完整检查点／账本游标一致时，可用同入口加 `--execute` 从精确游标恢复。FAILED/OOM/未完成更新不能自动回滚；检查失败证据后另记恢复决定。不要第二次调用 workflow --launch 或删除已有state绕过单写者保护。

历史A对照和COVERAGE结果保持只读。48请求中原标TRAIN的4父不属于本轮25父池，原标签仅作历史比较分组；实际成员关系见 [data/evaluation_membership.json](data/evaluation_membership.json)。本次DEV8父没有进入新训练，但该面板已经反复用于开发研究。

最后数值将写入 `results/`，RAW/D0/BASE_POST写入`evaluation/`，几何级联写入`cpu/`，图集写入`gallery/index.html`。不存在这些产物时不能提前报告效果。没有自动改默认、commit或push。
