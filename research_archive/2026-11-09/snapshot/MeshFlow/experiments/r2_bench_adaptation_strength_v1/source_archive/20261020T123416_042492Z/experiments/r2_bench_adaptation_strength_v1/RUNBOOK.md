# Bench 适配强度三臂实验：执行与恢复

本目录对应 `R2_BENCH_ADAPTATION_STRENGTH_V1`。最终科学结论见 `REPORT.md`；只有 `completion.json` 存在且核验通过才代表交付完成。工作流的计算完成状态不代表已经完成实际图审。

## 已冻结的范围

| 臂 | 起点 | 主干学习率 | role/Geo 学习率 | 新增更新数 |
|---|---|---:|---:|---:|
| A | COMMON12000 live | 1e-5 | 1e-5 | 1500 |
| B | 同一 COMMON12000 live | 2e-6 | 1e-5 | 1500 |
| C | 官方 Bench EMA＋原条件初始化 | 2e-6 | 1e-5 | 1500 |

三臂均使用 fresh AdamW、batch 8 / microbatch 1、同一 58 父 / 195 条条件及原 CLEAN1500 数据流。训练使用来源 owner 监督；推理使用冻结的旧 SEAM_S15000 C-only 预测布局。零占用头训练、零占用头前向。

主比较为 B−A（COMMON 起点下的主干学习率）和 C−B（相同后续预算下的完整初始化包）。主结果固定为新增第 1500 步，不根据图片或 loss 选择中间权重。

## 入口

在 WSL Ubuntu、`/mnt/e/MeshFlow` 下使用既有环境：

```bash
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_adaptation_strength_v1.resume
```

默认只读查看状态，不启动训练。

总工作流：

```bash
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -u -m r2.pilots.bench_adaptation_strength_v1.workflow --execute --max-cpu-workers 6
```

该入口会核验既有产物并按资源测量增量调度；运行中不要同时手动派发同一任务。科学工作进程拥有各自的输出目录、模型、优化器和随机流。完成项复用已有身份匹配的结果，不重复生成。

本轮另有独立 CPU 扩展派发器，仅用于训练完成后、尚未完成且没有其他进程拥有的 POST 标签：

```bash
python -B -m r2.pilots.bench_adaptation_strength_v1.parallel_cpu_dispatch --label A_S1500 --workers 6 --pool-limit 2
```

这是入口示例，不是再次执行已完成标签的要求。它读取当前资源、所有权和完成收据，保留原控制器及原作业；本轮按复测曾从2池扩展到3池、每池6 worker，再回到2池。不得跳过资源检查把18 worker当成固定配置。实际派发及显存统计范围见 `diagnostics/execution_summary.md/json`。

若出现失败或未完整保存的更新，不自动重试。保存日志与调用账本，按失败合同处理；不能通过删状态文件让工作流从头重算。只有完整安全保存状态才适用训练入口的 `--resume-training`。本任务不授权任意恢复已删除历史权重。

## 固定执行顺序与预算

1. 三臂各 1500 次优化器更新，共 4500 次；训练实例前向 / 反向各 36000 次。
2. 固定监督探针 0 / 500 / 1000 / 1500，A/B 共用相同 step0，共 1056 次前向。
3. 五个过程模型各 16 请求，三个终点各 48 请求，共 224 份 Euler50 RAW，即 11200 次生成器前向。
4. 全部 224 份 RAW 保存后，才执行 224 次单次官方 Bench D0。
5. 同一 GEO_C、来源定向和 CPU HINGE_OPP015 → ALPHA → HOLE；三阶段保留全部原指标。
6. 672 条评分记录；144 张同相机、同尺度、同位置截面对照图；672 条实际非盲 AI_PROXY 图审记录。

过程面板固定为 16 请求，不含 SEAT_SURFACE，但有 2 个 SEAT_SUPPORT 请求，不能与终点 48 请求均值混成学习曲线。部位以冻结 manifest 的真实 part_type 为准，历史条件 ID 字样不重新定义部位。终点包含 TRAIN 16 / DEV 32；开放输入 14 / 26，闭合输入 2 / 6 的 seam 指标为 NA。

## 收口

所有训练完成后：

```bash
python -B -m r2.pilots.bench_adaptation_strength_v1.summary_training --final
```

实际查看图库中的两视角及两截面，并按 `visual_review/protocol.json` 写独立观察记录后：

```bash
python -B -m r2.pilots.bench_adaptation_strength_v1.collect_v2
python -B experiments/r2_bench_adaptation_strength_v1/diagnostics/audit_results_v2_readonly.py
python -B -m r2.pilots.bench_adaptation_strength_v1.visual_review collect
python -B -m r2.pilots.bench_adaptation_strength_v1.result_analysis
python -B -m r2.pilots.bench_adaptation_strength_v1.closeout_audit_v2 --final
```

正式结果路径为 **`results_v2/`**，不是旧 `results/`。旧汇总器曾把全网格 `boundary_edges` 覆盖为输入C边数；V2保留原字段，并把C边数另存为 `C_boundary_edges`。原评分JSON、旧汇总与冻结源码未改。修订记录见 `diagnostics/collector_v2_revision.json`。

`closeout_audit_v2`保留旧审计入口不动，输出独立 `closeout_audits_v2/<timestamp>/`；开始大文件hash与源码归档前，要求训练、672条原评分、144张图页及672条实际图审全部完成，并要求 `diagnostics/results_v2_independent_audit.json` 为PASS且绑定当前V2表与完成收据的hash。该值级核验核对原字段、条件/父/宏汇总与配对；缺失项只记INCOMPLETE，FAIL不能标完成。大hash只在最终运行，不用于循环轮询。最终报告及项目级完成标记仍由主会话审结论后写入。

小源码归档包括本轮入口、注册依赖、`diagnostics/*.py`、`visual_review/*.py`与`*.ps1`（含实际图审整理器），并保留独立值审计初版收据和RUNBOOK；不复制权重或历史结果。

最终审计 PASS、主会话完成科学结论并将报告状态改为 COMPLETE 后，使用实际审计路径封存：

```bash
python -B -m r2.pilots.bench_adaptation_strength_v1.finalize --audit experiments/r2_bench_adaptation_strength_v1/closeout_audits_v2/<timestamp>/closeout_audit.json
```

该命令只认证并写入 `completion.json`；已有封存且内容变化时拒绝覆盖，不启动训练、生成或重评分。

这些收口命令读取已保存结果，不新增模型前向。渲染图、读取指标、生成表格均不冒充实际图审。`UNCERTAIN` 单列，来源款式不是唯一正确答案；数值拓扑与视觉判断分开报告。

Chair、默认采样模型、官方资产、旧结果保持原状态。本轮不自动延长、扩父、清理权重、commit 或 push。
