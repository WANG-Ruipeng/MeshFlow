# 恢复与复算入口

工作目录 `E:\MeshFlow`；WSL Ubuntu，Python `/mnt/e/MeshFlow-Control-artifacts/venv/bin/python`。所有产物隔离于本目录，旧资产只读。

完成后只核验，不产生模型调用：

```powershell
wsl.exe -d Ubuntu -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_geo_only_acceptance_v1.run verify
```

若在完成前发生中断，运行受预算约束的入口：

```powershell
wsl.exe -d Ubuntu -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_geo_only_acceptance_v1.run resume
```

已完成阶段核验receipt并复用，不重抽随机数组；存在已开始但未完成的D0事件会停止，不能静默再前向。两份D0提议须都落盘后再做几何。完整评分最多3CPU worker（首个实际任务测量后扩展），渲染1worker；GPU仅条件性2来源D0共享1worker，实际派发前重新测资源。所有科学参数原样保留。

阶段入口分别为 `core run`、`evaluate --manifest <manifest.json> --workers 3`、`render --manifest <manifest.json>`、`review_collect`、`identity_prepare run`、`identity_render`。几何告警审计和实际图片评阅完成前不会打开identity门；人工/AI_PROXY观察不能由resume脚本合成。

最终报告与独立源归档完成后执行 `closeout` 封存；重新执行封存只验证。`code.diff` 只含本次独立新增模块，不混入工作区原有修改。任何已完成输入/代码hash改变均应新建版本，不能覆盖旧receipt。

本实验不含新训练；后端训练数据与预算尚未冻结，恢复入口不会自动继续训练、调参数、切模型或上传。
