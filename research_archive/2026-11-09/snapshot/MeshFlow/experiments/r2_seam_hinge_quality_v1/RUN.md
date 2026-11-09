# 本轮执行与恢复入口

本轮入口是 `r2/pilots/seam_hinge_quality_v1/resume.py`。在仓库根目录运行。默认只读状态，不会启动训练或生成：

```powershell
wsl -d Ubuntu -- env PYTHONPATH=/mnt/e/MeshFlow/r2/src /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.seam_hinge_quality_v1.resume
```

必须显式指定当前仓库的 `PYTHONPATH`。该虚拟环境历史上的 editable 安装仍指向 `E:\MeshFlow-Control`；本轮不改全局安装。一次采样启动在旧包缺少 `geo_io` 时、模型加载和任何前向之前失败，环境修正记录在 `generation_engineering/`，没有消耗生成预算。

阶段按以下依赖执行；这份清单不是重新运行已完成阶段的授权：

1. `protocol`、`training_prepare`：固定输入、源码、预算和CPU搬运检查。
2. `dispatch training` 后，`training --arm CLEAN_EDGE_COH` 与 `training --arm SMOOTH_PRIOR`：各1000更新。额外worker依据实际资源渐进启动。
3. `training_summary`：只读训练状态、曝光、调用和终点权重审计。
4. `generation prepare`、`dispatch generation` 后，两个 `generation raw --arm ...`：各28份固定请求。
5. 全部56份RAW完成后，两个 `generation post --arm ...`：每份仅单次官方D0及原GEO_C/来源定向。
6. `scoring score --wave main`、`scoring summarize --wave main`；`binding run --wave main`。
7. 基线及两新臂分别 `hinge run --arm ... --workers 2`，每臂两阈值56格；`hinge_collect collect` 汇总168格。
8. `gallery`、`section_gallery`、`case_sections` 保存固定视角与四阶段同位置截面。随后实际打开图片图审，不能用“渲染成功”代替图审。
9. `aggregation --include-hinge`、`evaluation_audit`、`audit`，最后生成报告、源码快照及 `review_index`。

阶段入口依据完成回执跳过或拒绝重复执行。遇到部分前向、未完成产物或hash变化时停止并保留现场，不能删除目录后当作首次运行。训练恢复同样不能额外突破1000次更新或8000次前向预算。全部收口后 `completion.json` 会阻止总入口自动执行更多阶段。

`checkpoints/` 保存可恢复训练终点，`inference/` 保存相同活生成器参数的推理导出；本轮不是EMA导出。SMOOTH_PRIOR必须配合本轮冻结的bulk噪声变换，不能把该权重与标准Gaussian起点混用后声称复现本结果。

原始资产、既有报告及评分定义保持只读。无默认模型切换、权重清理、commit或push。

最终收口依次为 `report`、实际图审记录、`source_snapshot --finalize`、`review_index`、`closeout`（均为同一模块前缀，最后几项直接调用，不会启动模型）。源码快照之后不得继续改源码；如需修正，必须显式保留前一版证据。`closeout` 只在完整数值审计、图审覆盖、报告及源码均具备时写入停止回执。日后只查看状态，使用上面的总入口即可；不要为了查看结果重新执行训练或采样。
