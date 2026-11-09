# CPU数据入口与恢复

已有完成结果优先校验，不重抽样、不调用模型。

在E:/MeshFlow工作区，用已有WSL环境：
```powershell
wsl.exe --cd /mnt/e/MeshFlow /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.data_tools.bench_multilod_v1.run status
wsl.exe --cd /mnt/e/MeshFlow /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.data_tools.bench_multilod_v1.run_v2 status
wsl.exe --cd /mnt/e/MeshFlow /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.data_tools.bench_multilod_v1.verify
```

从完整原输入构建的已执行顺序是inspect → run freeze → run propose/build → render → 工程v2 build/render → 实际view_image图审 → finalize → closeout。
`build`遇到已完成receipt只核hash并复用；遇到不完整proposal_attempt停止，不静默重算。工程v2只限原20个减面不足父，不对质量失败再次减面。

`review_root.py`仅封存本会话实际已看的三份观察，不是自动图审程序；不得用于为新图伪造人工或AI看图。其余六份由实际审阅代理保存。

`audit_sampler`仅以历史日程做CPU映射/身份核验，3000步是被读取的历史日程长度，不是新增训练；它不会写新训练日程或运行模型。

正式训练入口尚未修改。新manifest不能平铺传给旧trainer。见sampler.py及其接入说明：同父曝光不增加，CLEAN/X_WIDTH对齐，N/OT/FM分母重算。没有此阶段GPU预算或预测头权重。
