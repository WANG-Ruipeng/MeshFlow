# 本轮恢复与核验

解释器沿用 `/mnt/e/MeshFlow-Control-artifacts/venv/bin/python`，仓库工作目录 `/mnt/e/MeshFlow`。

冻结 jobs.json：`7865f436d746c60703cc7038f21f01109acc9ce29cdeaf40c9687e1991f4b060`。
纯读取合同核验已通过：30逻辑候选、23新物理后端、最多46新评分。六份全RAW参照复用，另有一份自动候选精确等于原D0。

## 只读状态

```
python -B -m r2.pilots.bench_regional_mix_dual_v1.runtime status --jobs experiments/r2_bench_regional_mix_dual_v1/jobs.json
python -B -m r2.pilots.bench_regional_mix_dual_v1.runtime contract --jobs experiments/r2_bench_regional_mix_dual_v1/jobs.json
```

## 有限的逐任务恢复

主调度器首次执行测量一个真实物理任务后增至最多两CPU worker。若进程中断，不盲目重跑主调度入口：其 `measured_dispatch.json` 是不可覆盖收据，自动续排路径仍有收据碰撞限制。

先以 status 与每个 runs/J*/attempt、events、complete、failure 检查已完成、尚未开始和部分执行三种状态。完整任务执行下列命令只核验旧结果；只有尚未开始的登记任务才允许运行。部分执行/失败必须单独记录工程恢复方案，不自动从头重放。

```
python -B -m r2.pilots.bench_regional_mix_dual_v1.runtime run-one --jobs experiments/r2_bench_regional_mix_dual_v1/jobs.json --job-id J00_ASSIST10
```

此处job-id仅为语法示例，应从冻结清单选择。新增worker仍须测资源；同轮alias等待其owner完成。不得增加请求、区域、阈值、噪声或模型调用。

## 收表与显示

全部30逻辑任务完成后，`runtime collect --jobs ...` 收集原评分，不重评分。
`runtime render-one --jobs ... --job-id ... --stage CLEAN_FINAL_POST` 输出四列：原RAW、原D0整链、全RAW整链、本候选，使用继承的两视角及两位置截面。对24个ASSIST/AUTO逻辑候选逐页记录真实AI_PROXY观察；不从评分自动填视觉标签。完整结果、原文件hash和实际调用数在最终报告与完成收据中核验。

原资产、默认模型和评分器只读；本轮零训练、零模型前向、零新生成。
