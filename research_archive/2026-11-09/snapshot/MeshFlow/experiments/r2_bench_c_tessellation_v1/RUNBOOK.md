# 可恢复入口

使用现有WSL环境，仓库目录 `/mnt/e/MeshFlow`，Python为 `/mnt/e/MeshFlow-Control-artifacts/venv/bin/python`。

安全查看当前状态：

```text
python -B -m r2.pilots.bench_c_tessellation_v1.run status
python -B -m r2.pilots.bench_c_tessellation_v1.run prepare
```

prepare在registration存在时只核验保护资产和输入hash，不启动模型。新实验的550前向已经完成；不要重跑旧采样来“恢复”完成状态。各完整RAW由complete.json保护，部分开始但未完成的模型调用禁止自动重算。

实际历史执行：worker w0 jobs0..6，worker w1 jobs7..10。job0为原输出复现，jobs1..10为新增固定条件对照。两个GPU进程先后运行；随后CPU score首份单worker，余九份双worker，并行双worker渲染。相关入口在run.py、cpu.py、evaluate.py；它们不下载权重、不训练、不运行D0。

当前曲线与指标已经完成，不再调用 `analyze curves/collect/all` 重新汇总。历史曲线收据记录了当时源码hash；随后metrics_v2的纯元数据适配改变了analyze.py的hash。旧curves入口没有完成态早退，重跑会先重写派生表/图，再因不可变收据不一致而失败。这是恢复说明中的工程问题，不是模型或评分失败；本轮发现后未重跑，也未覆盖旧曲线。

当前指标以analysis/metrics_v2为准，v1及原score仍保留。已完成状态只执行上方的 `run status/prepare`。完整结果/图片hash见 `COMPLETE_v2.json`；旧 `COMPLETE.json` 与首版ZIP保留用于工程变更追溯。完成后的归档入口不是新增科学任务入口，不要将 `closeout` 当作任意代码版本下通用重建命令。

图审是实际查看图片后写入的AI_PROXY记录，不由汇总入口自动推导；不要把PENDING_ACTUAL_VIEW渲染收据误当成缺少图审，独立实审记录在visual_review/。

正式采样时原 y 始终等于真实有效N。459两个诊断格仅改变本实例y_embedder lookup，不能用于默认生产入口。COMMON_PHYSICAL_C表仅用于同表面几何比较，不是新C的实际拓扑认证。所有结果是RAW，POST明确未执行。
