# W_MATCHED：执行与恢复入口

本轮只有一条新增训练臂，固定 1500 步。已完成的 B/W 数据、权重、输出、评分均只读。以下命令在 `/mnt/e/MeshFlow` 使用现有 `/mnt/e/MeshFlow-Control-artifacts/venv/bin/python` 执行；不需要安装依赖。

## 身份对照

|简称|完整身份|本轮活动|
|---|---|---|
|P0|exposure_allocation / B_75_25 / step1500，global21000|共同起点；恢复 live 生成器及匹配 AdamW/RNG|
|B|natural_n_recipe / NATURAL_HIGH / step1500，global22500|窄 HIGH 历史参照，46 输出复用|
|W|high_joint_coverage / WIDE_HIGH / step1500，global22500|宽 HIGH 直接对照，46 输出复用|
|M|group_weight_match / W_MATCHED / step1500，global22500|唯一新增训练与46请求评价|

M 不从 B 或 W 的终点续训。它重放原 W 的全部训练实例，仅按冻结的窄 B 计数调整 HIGH 更新的组 MSE 系数。生成器结构与数据没有改变。

## 状态入口

```powershell
wsl.exe --cd /mnt/e/MeshFlow /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_group_weight_match_v1.workflow
wsl.exe --cd /mnt/e/MeshFlow /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_group_weight_match_v1.monitor
```

`workflow` 仅报告状态，不自动启动后续阶段。训练写入 `training/W_MATCHED/`，每250步保存匹配恢复状态；固定终点同时保存完整状态与推理导出。

## 已登记阶段

1. `register`：核对并冻结原 W 数据流、系数、源模型、优化器、输入及代码。
2. `experiments.r2_bench_group_weight_match_v1.dispatch_train`：测量资源并启动唯一正式训练。
3. `experiments.r2_bench_group_weight_match_v1.dispatch_eval sample-m`：训练完成后测量资源，只生成登记的46份 RAW。
4. `...dispatch_eval d0-m`：全部46份 RAW 完成并保存后，执行46次单次官方 D0 和同一 GEO_C。
5. `workflow --launch prepare-scores`、`score-raw`、`score-post`、`collect`：原评分和固定 HINGE_OPP015 / ALPHA / HOLE；只计算 M 的138个新增阶段单元。
6. `report_results tables`、`render`：完整数值与同框 W/M 图集。实际图审另写逐页记录，不能由指标代填。
7. 调用、曝光、评测收口审计及最终报告/归档。

每个 GPU 阶段前必须重新记录当前资源；RAW 与 POST 的 CPU 评分可在资源允许时并行。科学参数与请求数不能为了并发而改变。

## 中断边界

没有自动重试或科学重算额度。先读取 `workflow/*/status.json`、`training/W_MATCHED/status.json` 与 `calls.jsonl`。已完成阶段按收据验证后复用；失败或 pending 科学调用不能直接重放。只有完整保存、调用账本精确一致的安全边界，才可能按原随机状态继续；不能因磁盘里有一个旧检查点就回退重算。

纯路径、表格、图集和归档问题可以在零新增科学调用下修复，须记录 `ENGINEERING_EVENTS.md`。不得修改冻结训练代码后冒充同一运行。

## 结果口径

完整177字段及 NA 原因保留。主比较为 M−W；B只作数值参照。图审采用采样前冻结的 V2 定义，逐页实际查看 W/M 的两视角与两截面，标为非盲 AI_PROXY；历史图审标签不覆盖、不跨版本拼接成功率。

完成后停止；不自动延长训练、扫系数、切默认模型、清理资产或 commit/push。
