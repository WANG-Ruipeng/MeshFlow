# R2_CREF_LAYER_AUDIT_V1 恢复入口

这是已保存 SMALL/CREF 几何的 CPU 机制诊断，不会调用模型、旧评分器、训练、生成或去噪。

完成后使用以下总入口验证，已完成阶段不会自动重放：

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.tools.cref_layer_audit_v1.resume --verify
```

输入身份和新定义分别保存在 registration.json / science_freeze.json。44 个模型阶段单元与 6 个来源参考各有独立 diagnostics 目录和 receipts 收据。主进程独占 ledger.jsonl，两个 CPU worker 不共写结果。完整逐查询 CSV、全部有限法线命中、候选面与精确共边记录都在本地，旧产物保持只读。

阶段入口为 `run --stage prepare|freeze|geometry|render|aggregate|audit`，仅供定位未完成阶段。不要在已完成目录盲目重跑 `prepare/freeze/geometry`：科学输入冻结、资源快照和账本是不可覆盖的。若中断，先检查已有 receipts；已有单元通过文件 hash 后可复用。没有完成收据的局部输出必须先保留和核对，不能删除后静默重做。对发生过阶段派发的中断需要另存资源快照并登记恢复工程处理，不能当作第一次派发。

`independent_audit/audit.py` 进行独立数值验证，首次审计器自身假设修复已记录；此验证不替代人工或 AI 图审。图审为明确署名的非盲 AI_PROXY，不是人类打分。

小包压缩图片仅用于云端快速审阅；原分辨率28张PNG、完整近点坐标/射线重心坐标和所有逐查询列均保留本地。压缩不是重新渲染或新实验。包内 SMALL_DATA_SCOPE.json 明确包含和省略的数据。