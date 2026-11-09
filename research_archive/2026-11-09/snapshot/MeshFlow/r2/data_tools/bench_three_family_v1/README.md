# Bench 三家族目标/生产输入小样

数据目录：`experiments/r2_bench_three_family_v1/`。这是一个已固定3父、最多6条新C/Y的数据制作实验，不是新训练器。

完成后恢复入口只核验hash，不重放模型调用：

```bash
python -B -m r2.data_tools.bench_three_family_v1.run --resume
```

Windows本地环境：

```powershell
wsl.exe -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.data_tools.bench_three_family_v1.run --resume
```

首次执行依次完成的阶段：`common`冻结协议；`n_budget`静态核对；`targets --stage boundary`固定严格边界构造；`part585`一次12部件方盒提案；`part27`冻结后以`--execute`执行一次短边提案、`fallback_checks`检查；`select_targets`固定最终三份；`materialize_tasks`构造六C/Y；`render parents`和`render conditions`；实际查看后`record_review`；`production_contract run --manifest .../tasks/reviewed_manifest.json`完成1次小头CPU前向和1次闭C空布局；最后`run --closeout`保存汇总与来源hash。

不能盲目重跑前述阶段。未完成模型尝试会阻塞自动恢复，不能删除attempt文件再次调用；已完成的布局从缓存读取。`run --resume`故意不导入占用头/生成器，也不调度上述阶段。

权威状态为`tasks/final_manifest.json`与根`COMPLETE.json`。中间manifest里的pending字段和早期候选训练用途属于历史阶段，不能作为准入。R1仅为拟训练，R2固定预留；不能因R1失败而将R2挪进训练池。

保留目标NPZ/OBJ、C-only输入、真实C/Y、来源多对多区域谱系、冻结布局、逐边失败原因和逐面往返。保真数值来自固定有限采样点，不是连续Hausdorff认证。图审是实际非盲AI_PROXY，不是人类独立评分。无生成器、D0、训练、新噪声或高N运行。

CPU验证：7项共享边界测试、8项输入/布局/往返测试；最后另核对3目标和6条件实际数组。原评分代码和原数据只读。
