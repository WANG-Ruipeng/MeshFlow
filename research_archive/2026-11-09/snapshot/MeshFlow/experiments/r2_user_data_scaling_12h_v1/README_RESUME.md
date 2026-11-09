# 恢复说明

状态：DATA_SCALE_BLOCKED。当前67训练父低于任务书128门槛，不能强行继续训练。

在 PowerShell / E:\MeshFlow 中：

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B r2/pilots/user_data_scaling_12h_v1/resume_all.py --stage status
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B r2/pilots/user_data_scaling_12h_v1/resume_all.py --stage verify
```

`start` 已实际验证在数据门返回2，不导入Torch、不执行GPU。没有可恢复的本轮训练检查点。

源数据清单、286份条件与181父输入图审已经保存；不应重新下载权重或重新扫描相同来源。GPU实现未验证，不能把入口存在当作已完成实验。现有窗口、停止状态和冻结文件不能通过改时间戳、强行扩噪声或改父名单绕过。

若未来用户另行授权足量数据的新任务，先解决来源/区域缺口及先划父后派条件顺序，再登记新窗口与新冻结数据；不得静默将它写回本轮。
