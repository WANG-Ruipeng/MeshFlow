# 固定实验执行与恢复

本目录是 `CHAIR_CAPPED500_REPLICATION_AND_GRADIENT_BUDGET_V1` 的独立结果。原START、历史teacher/CAPPED500、原反馈池和评分器均只读。默认模型不切换。本文件提供最小入口，不授权固定任务完成后再增加训练、噪声或检查点。

环境：已有WSL `/home/warpwang/miniforge3/envs/meshflow/bin/python`；工作目录 `/mnt/e/MeshFlow`；不安装依赖。所有GPU命令在现有单GPU锁下执行；OMP/MKL/OpenBLAS各2线程，`CUBLAS_WORKSPACE_CONFIG=:4096:8`，严格确定性、SDPA math、TF32关闭。

```bash
export PYTHONPATH=/mnt/e/MeshFlow
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/home/warpwang/miniforge3/envs/meshflow/bin/python

# 只在输入准备阶段使用；已冻结目录仅验证缓存，不重抽Gaussian
$PY -B -m experiments.mf_h1_local0.chair_capped500_gradbudget_v1.registration
$PY -B -m experiments.mf_h1_local0.chair_capped500_gradbudget_v1.protocol

# 已实际使用的可恢复固定GPU总入口
$PY -B -u -m experiments.mf_h1_local0.chair_capped500_gradbudget_v1.run

# 独立单臂/单终点入口；从新完整恢复权重继续，不能误用旧训练器
$PY -B -u -m experiments.mf_h1_local0.chair_capped500_gradbudget_v1.worker train \
  --arm L --until 1000 \
  --checkpoint native_t1/runs/chair_capped500_gradbudget_v1/checkpoints/repeat_L_500.pt

# 已保存终点评价；冻结manifest提供固定C/N/Gaussian
$PY -B -u -m experiments.mf_h1_local0.chair_capped500_gradbudget_v1.worker sample \
  --label L500 --panel BOTH \
  --checkpoint native_t1/runs/chair_capped500_gradbudget_v1/checkpoints/repeat_L_500.pt

# 纯CPU账本/训练日志核验，训练中加 --live，结束后省略
$PY -B -m experiments.mf_h1_local0.chair_capped500_gradbudget_v1.audit_runtime --live
$PY -B -m experiments.mf_h1_local0.chair_capped500_gradbudget_v1.audit_training --live
```

GPU计划是：有限真实预检 → 历史三模型各FRESH16 → T/L/G各500步与OLD+FRESH32 → T/L/G各恢复到1000步与OLD+FRESH32。两个终点均已预登记，不按中途分数增删分支。每个新臂从同一原START完整AdamW恢复；全部共享新主流seed3594377816，固定辅助teacher-line使用原池seed20261008。辅助队列位置由已保存更新步与冻结队列函数确定，未刷新池。

新checkpoint使用 `chair_capped500_gradbudget_checkpoint_v1` schema，含参数、AdamW、RNG、主流游标、辅助next step、recipe与协议hash。正式维护START入口仅接受原START，不应拿它加载本轮研究权重。旧历史权重通过其原严格加载器只做推理。

CPU评分接口为 `evaluation.score_set(out,label,panel,workers=2)`，标签与面板见protocol。`evaluation_watch.py` 等完整PASS回执后增量评分，48历史OLD单元直接校验复用；所有288单元完成后调用 `analysis.run(out)`，再冻结共同相机、渲染OLD/FRESH全图。评分用原固定参考和原源码阈值。视觉状态只有实际查看全组后才能由AI_VISUAL_REVIEW记录补齐，图片生成本身不等于已审阅。

调用计数在 `ledger/calls_*.jsonl`；GPU持锁时间在 `ledger/gpu_occupancy.jsonl`，包括加载、保存及持锁期间CPU等待，不能等同kernel时间。恢复不会清空旧计数；未闭合模型调用/占用、未检查点化更新必须先明确审计。允许的路径/日志等工程修正保留在 `engineering/`，不改变冻结科学定义。预检隔离optimizer副本的1次更新单列，不计作正式训练成果。

硬预算：正式3000更新/27000F/27000B，正常240新Euler50；完整采样尝试≤290、采样NFE≤14500；预检≤24F/12B；顶层前向≤41524；GPU累计持锁≤14400秒。旧CONFIRM不评价。预算触顶或科学定义需改变时停止并登记实际未完成项。

以下为对应的最小CPU评分、配对分析与最终审计入口。已完成的固定单元读取其身份核验缓存；这些入口不授权追加评价请求。

```bash
$PY -B -m experiments.mf_h1_local0.chair_capped500_gradbudget_v1.evaluation --label G500 --panel ALL --workers 2
$PY -B -m experiments.mf_h1_local0.chair_capped500_gradbudget_v1.analysis
$PY -B -m experiments.mf_h1_local0.chair_capped500_gradbudget_v1.audit_training
$PY -B -m experiments.mf_h1_local0.chair_capped500_gradbudget_v1.summarize_training
$PY -B -m experiments.mf_h1_local0.chair_capped500_gradbudget_v1.audit_runtime
$PY -B -m experiments.mf_h1_local0.chair_capped500_gradbudget_v1.report_figures
```

`review_sources/` 保存实际新实现、历史训练分支及实际评分依赖小源码，`SOURCE_HASHES.json` 和 `code.diff` 标明身份与差异。审阅ZIP不包含权重或RAW，完整生成重现依赖本地登记资产；不能把审阅包误称无需数据即可独立运行的完整环境。