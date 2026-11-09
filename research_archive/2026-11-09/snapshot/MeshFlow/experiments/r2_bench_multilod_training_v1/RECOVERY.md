# 固定双臂实验恢复说明

本文件说明既有入口与恢复边界，不替代冻结注册、资源调度记录或失败账本。当前工作目录为`E:\MeshFlow`，实际计算使用WSL Python：

```powershell
wsl.exe --cd /mnt/e/MeshFlow /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -m r2.pilots.bench_multilod_training_v1.training --check-cpu
```

数据与注册由WSL路径空间生成（`/mnt/e/MeshFlow/...`）。在Windows直接重建相同JSON会产生不同绝对路径字符串，触发冻结身份检查；这不是科学流差异。正式检查和恢复沿用WSL入口，不修改已经冻结的manifest、schedule、注册或源码来绕过检查。

## 首先确定完成状态

读取`training_registration.json`、两臂`training/<arm>/status.json`、`calls.jsonl`、`training_log.jsonl`、`exposure.jsonl`和检查点receipt。`COMPLETE`需验证终点文件hash、完整调用账本与注册身份，不能仅根据权重文件存在判断完成。训练入口对已验证完成的臂直接返回，不重复训练。

每臂只允许1500个optimizer更新、12000次训练前向及12000次反向；没有head/probe预算。分支局部步数`s`对应：

| 游标 | 必须值 |
| --- | --- |
| stage | NATIVE |
| stage_step / local_step | s |
| curriculum_step | 7500+s |
| global_step | 25500+s |
| native_total_step | 13500+s |
| seam_total_step | 0 |
| stream batch_index / next_sample_index | s / 8s |

模型、匹配AdamW、RNG、游标、数据/schedule/系数hash和调用日志必须同时一致。不能只恢复generator，也不能把新版本数据冒充原EXPOSURE历史流继续运行。

## 训练恢复入口

每次新worker前按AGENTS测量GPU空闲/利用率/其他进程及CPU/RAM，保留至少max(2 GiB,总VRAM10%)的初始安全余量。训练需要本目录`dispatch/<arm>_go.json`中与注册hash匹配的资源GO。不要为了并行改变batch、精度、优化器或预算。

只有`INTERRUPTED_SAFE`且检查点、日志和调用账本正好处在同一完整更新边界时，使用：

```powershell
wsl.exe --cd /mnt/e/MeshFlow /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -m r2.pilots.bench_multilod_training_v1.training --arm PLAIN1500 --resume
wsl.exe --cd /mnt/e/MeshFlow /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -m r2.pilots.bench_multilod_training_v1.training --arm MULTILOD1500 --resume
```

以上命令是各臂独立恢复方式，不表示应同时启动。正常新臂首次启动不带`--resume`。SIGINT/SIGTERM仅请求当前更新后安全保存；不要强杀当前更新或无关进程。`FAILED`、`OOM`、挂起调用或检查点落后于账本不能伪装为安全恢复；保留实际尝试记录，明确工程修复与额外重算范围后再决定，不自动退回250步旧检查点重算。

中间滚动检查点每250步保存；固定正式终点为`checkpoint_step1500.pt`与`generator_step1500.pt`。本轮没有best-checkpoint选择。

## CPU数据准备恢复

`data/manifest.json`、两份schedule、两份coefficients、`stream_gate.json`和`runtime_cpu_gate.json`均已形成。39版本×两种增强的78次实际CPU OT已完成，不需重做。入口可验证并复用完成receipt：

```powershell
wsl.exe --cd /mnt/e/MeshFlow /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -m r2.pilots.bench_multilod_training_v1.prepare_data
```

`data/cpu_ot_checks/`每项先写attempt后写receipt。存在attempt但没有PASS receipt时，入口会停止，不静默重算。此入口显式隐藏CUDA，只用于独立CPU进程，不能在训练进程里调用。恢复后78是累计实际OT数，不是每次恢复都新增78次。

## 评价严格分阶段

终点身份核验后，以固定面板分别采样两臂：

```powershell
wsl.exe --cd /mnt/e/MeshFlow /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -m r2.pilots.bench_multilod_training_v1.evaluation sample --label PLAIN1500
wsl.exe --cd /mnt/e/MeshFlow /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -m r2.pilots.bench_multilod_training_v1.evaluation sample --label MULTILOD1500
wsl.exe --cd /mnt/e/MeshFlow /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -m r2.pilots.bench_multilod_training_v1.post verify
```

`post verify`必须通过全92份新增RAW barrier；在此之前不执行任何新D0。采样与D0分别依据当前资源决定并行度及`--estimated-peak-mib`，不能把默认估计当实测。固定噪声来自manifest，与worker编号或完成顺序无关。

```powershell
wsl.exe --cd /mnt/e/MeshFlow /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -m r2.pilots.bench_multilod_training_v1.post d0 --label PLAIN1500
wsl.exe --cd /mnt/e/MeshFlow /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -m r2.pilots.bench_multilod_training_v1.post d0 --label MULTILOD1500
wsl.exe --cd /mnt/e/MeshFlow /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -m r2.pilots.bench_multilod_training_v1.post raw --workers 2
wsl.exe --cd /mnt/e/MeshFlow /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -m r2.pilots.bench_multilod_training_v1.post post --workers 2
```

CPU worker数按资源在已支持的1或2中选择。原EXPOSURE只验证和复用缓存，不重新生成、D0或评分。所有原输入、查询、评分器与结果只读；本轮完成receipt优先核验复用，失败/残缺尝试不能删掉后假装首次执行。

首次CPU评分前核对`evaluation/score_registration.json`及CPU合同receipt；如尚未建立，使用`evaluation prepare-scores`与`evaluation cpu-contract`的既有入口冻结原评分定义。它们不授权额外模型采样或改指标。

## 汇总与结束

在固定分数齐全后运行：

```powershell
wsl.exe --cd /mnt/e/MeshFlow /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -m r2.pilots.bench_multilod_training_v1.analysis collect
wsl.exe --cd /mnt/e/MeshFlow /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -m r2.pilots.bench_multilod_training_v1.gallery --workers 2
```

完整汇总应含414行逐输出结果及逐条件、逐父配对；图集138页保留原位置截面。图集生成不会产生评阅标签；必须实际查看并保存AI_PROXY记录，再由最终报告回答MULTILOD相对PLAIN是否值得继续。中途状态不当成最终结果。未完成项标明实际状态/原因，完成固定范围后停止，不自动扩训练或切换默认模型。
