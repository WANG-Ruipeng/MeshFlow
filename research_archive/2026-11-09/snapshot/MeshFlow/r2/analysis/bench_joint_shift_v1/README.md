# Bench JOINT_SHIFT：独立 CPU 分析入口

这是准备好的分析代码，不是实验完成声明。默认实验目录为
`experiments/r2_bench_joint_shift_v1`，输出仅写到该实验的
`analysis_joint_shift_v1` 与明确的 `analysis_notes`。不导入 pilot、模型或评分器，
不修改旧 RELATION 分析、科学代码、数据或评分结果。

比较身份固定：`JOINT_SHIFT_S1500` 对主对照 `RELATION_V1_S1500`，
以及历史次对照 `USER_V2_S1500`。同一旧 48 请求、三阶段、三模型。

## 使用时机与命令

正式训练完成后，只核对训练账本，不触发 collector：

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.analysis.bench_joint_shift_v1.training_audit
```

正式 collector 完成后，聚合原结果并绘制学习曲线：

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.analysis.bench_joint_shift_v1.run --check-inputs
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.analysis.bench_joint_shift_v1.run
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.analysis.bench_joint_shift_v1.plot
```

尚未完成时返回 WAITING 或明确报错，不创建完成结果。已有产物只允许完全一致重入。
所有缺失、支持域与异常保留；不会自动重训或重评分。

全部数值、训练曲线和三组实际图审完成后，独立收口：

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.analysis.bench_joint_shift_v1.closeout --check-only
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.analysis.bench_joint_shift_v1.closeout
```

`closeout` 读取新 `visual_review/protocol.md` 和 `assignments.json`，核对每组48页/144条、
总144页/432条、图片hash、实际查看声明、五轴枚举及固定列序
RELATION / USER / SHIFT。必须本轮重新看三模型；旧参考标签只在新记录通过后用于
披露跨轮主观评阅变化，绝不作为新标签来源。当前证据不全时即返回 WAITING，
不写 `REPORT.md` 或 `COMPLETE.json`；不会改写原workflow自动状态。

## 训练核对

- 1500 更新、12000 实际曝光、12000 forward/backward、1500 optimizer，调用成对闭合。
- 对每次曝光核对新 schedule、旧 RELATION schedule 和两份实际 exposure。
  除 `translation_seed`、`translation_xyz` 两键外，所有旧 schedule 字段都必须相等。
  另核对 N/K/t、更新与 micro 身份、各角色有效坐标数；时间戳、PID、机制文字和
  后续实际 loss/SSE 不是必须相等的输入字段。
- CLEAN 前四 micro 恰好 6000 份，零平移与单位缩放；后四 micro 恰好 6000 份，
  与原 ANISOTROPIC 同缩放，平移每轴限于 ±0.04L。保存逐轴与向量范数范围。
  种子独立身份与冻结向量均核对，不重新抽随机数。
- 检查登记的同一初始化模型、优化器与 RNG hash，不额外加载权重。
- 第一次更新前四个 identity micro 的 loss、三类速度 SSE/分母和 endpoint SSE
  逐项精确比较，共 40 项；不使用容差把差异抹掉，也不扩大为全训练逐位相同承诺。
- 学习曲线固定 100 步分箱，按原 SSE 总和 / 有效标量分母总和计算。
  endpoint 使用实际记录的 endpoint SSE，不用速度误差乘平均 t 推算。
  平移组的目标状态变化，所以即便源流相同，loss 也不是同目标的严格配对质量指标。

## 评价核对

保留 collector 实际 177 个数值字段，432 行原始结果；不为了旧文档的 179 填造字段。
ALL、历史 TRAIN、DEV、各 part、逐父与逐请求配对均保留。
主宏平均先噪声→条件→父对象等权；同时保存请求均值、绝对总数。
I 的全记录支持域、PASS、LOW_SUPPORT 分开，配对另保留 BOTH_PASS。
FF 比例、绝对非法面对数、受影响面数、有效面数、删面数和表面积分别报告，
避免用删面或分母变化直接宣称质量提升。图审必须实际看图，不能从这些数值推断。

CPU 合成账本测试（不运行任何实验）：

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m unittest r2.analysis.bench_joint_shift_v1.test_aggregation r2.analysis.bench_joint_shift_v1.test_stream_audit
```
