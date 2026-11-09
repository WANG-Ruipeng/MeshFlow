# Bench 完整座面扩覆盖：恢复与复查

本轮为一个已固定的 Native 续训分支：从旧 Native6000/global18000，新增1500步，到 global19500。原22父回放占半数曝光，新增a8/27两父占半数；df6留出，旧24评价请求不变。当前Bench参考仍为原完整座面课程的SEAM3000；不自动启动SEAM、不挑最好checkpoint。

## 环境

在 WSL Ubuntu 执行，工作目录 `/mnt/e/MeshFlow`，Python为 `/mnt/e/MeshFlow-Control-artifacts/venv/bin/python`，设置 `PYTHONPATH=/mnt/e/MeshFlow:/mnt/e/MeshFlow/r2/src`。所有GPU工作由已冻结入口记录，不另起探针。不要同时运行两个pipeline。

```bash
python -B -m r2.pilots.bench_seat_coverage_v1.run
python -B -m r2.pilots.bench_seat_coverage_v1.controller
```

状态文件：`controller/status.json`、`workflow/*/status.json`、`training/NATIVE/status.json`。训练逐更新日志、逐例曝光及前向/反向/优化器账本均在`training/NATIVE/`。图像页在`gallery/index.html`，保留相同视角和截面。

首次运行入口为 `python -B -m r2.pilots.bench_seat_coverage_v1.run --launch pipeline`；本轮已经启动，重复调用只读取既存启动状态，不会重跑。启动不完整、死PID、失败与不安全中断均显式停在需检查状态。

## 恢复边界

只有训练状态为`INTERRUPTED_SAFE`且完整checkpoint、执行账本、随机流和输入游标完全一致，才支持 `python -B -m r2.pilots.bench_seat_coverage_v1.train --resume`。这不授权丢失更新重算；其他失败先检查实际已执行调用。不得删除launch或失败收据来伪装首次运行。调度层的恢复应另写明确工程记录，不静默覆盖原尝试。

训练权重保留local0、local1500完整快照，以及每250步覆盖的rolling恢复快照。推理使用最终`training/NATIVE/generator_step1500.pt`；推理导出不能替代含AdamW/RNG的完整恢复快照。

## 阶段合同

1. 完成固定1500步；只生成预登记终点。
2. 新Native参考6份与候选30份RAW全部完成；连同只读复用旧48份，验证84份逻辑输出。
3. 新36份统一执行单次官方Bench D0、GEO_C/来源定向；CPU后处理仍为HINGE_OPP015、ALPHA、HOLE。
4. 新108条评分加只读复用144条，共252条；旧评分器与原结果不变。
5. 渲染90张完整对照页，实际查看252模型单元；AI_PROXY非盲图审与数值评分分开保存。
6. 汇总逐父/逐请求变化、完整指标、损失曲线及报告，最后核验账本。`READY_REVIEW`只表示数值和渲染完成，不是整个任务完成。

所有指标与NA、T-junction准入限制、历史父曝光边界见`PROTOCOL.md`和`DATA_REPORT.md`。本轮不删旧权重/数据，不改默认，不commit或push。
