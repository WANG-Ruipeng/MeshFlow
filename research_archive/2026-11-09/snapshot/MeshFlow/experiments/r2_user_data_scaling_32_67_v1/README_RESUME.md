# 运行状态与恢复入口

本轮为独立的 SMALL32 / EXPANDED67 实验；两臂均已完成共同6000步，不能再追加更新或重选终点。历史12h轮仍为DATA_SCALE_BLOCKED，本轮不是256父实验。训练、完整实际图审和父级聚合已经完成：控制器状态为SCORES_COMPLETE_REVIEW_COMPLETE，report_tables、review_tables及training_analysis/final均已完成。报告已经独立复核，配对统计图及QA收据齐全；scientific_completion.json记录科学阶段在12小时窗口内完成。本说明随最终审阅包一起交付；ZIP实际完成状态、时间与hash以delivery_receipt.json为准。

## 安全只读检查

以下入口只读取状态，不加载模型或派发任务：

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B r2/pilots/user_data_scaling_32_67_v1/resume_all.py --stage status
```

主要状态文件：`training/USER_SMALL/status.json`、`training/USER_EXPANDED_67/status.json`、`controller/status.json`。正式图审以 `evaluation/gallery/output_review_receipt.json` 及其绑定CSV为准，分块记录和交接保留在 `review_work/`；图集渲染完成不代表已经审阅。最终交付状态与本地大资产清单见 `delivery/`（仅在最终打包运行成功后成立）。

`--stage verify` 是有限CPU身份检查，会新写核验收据，并非纯只读状态命令。`--stage aggregate` 会重新调用已注册汇总，不是查看结果的快捷方式。完成后直接读报告与已有CSV，不需重新执行这些步骤。

## 恢复边界

总入口保留 `--stage run`，但本轮完成后不再用于追加实验。已完成dispatch仅在身份匹配时跳过；失败或中断目录不会自动重放。若未来遇到损坏或不完整状态，先核对最后完整检查点、持久账本和输入游标，按原任务失败合同处理；不能跳坏batch、重新抽噪声或从旧点重复已经确认的更新。不可启动第二个控制器或并发写同一结果。

两臂从同一官方生成器EMA和相同条件初始化开始，fresh AdamW，按每2000物理段轮换至6000。每500步完整latest、固定4000及最终6000权重和推理导出保留。选择终点不依赖图像或评分。截止条件仍在registration.json：2026-10-12 20:15:34北京时间停止新训练更新，23:15:34停止新科学任务，不延期续跑。

所有RAW完成后才统一执行单次官方D0、GEO_C和来源角序恢复/固定C的free定向。没有RBF、B、READOUT、新loss、蒸馏或多次去噪。历史USER_MIX只作冻结现状参照，不能替代新训练臂；不切默认模型，不commit/push。

## 保留的工程历史

- 首次控制器日志参数错误发生在Popen前，0子进程/模型调用，证据保留在 `engineering/controller_dispatch_fix/`；显式 `evaluation_prepare_v2` 仅校验已冻结输入，没有重抽噪声。
- C-only条件NPZ读取器接线修复保留在 `engineering/reader_schema_fix/`，没有修改旧NPZ、数据分区或科学定义。
- 2026-10-12 15:37北京时间，SMALL已经6000而EXPANDED为4000。EXPANDED末段在创建空派发目录后、启动子进程前被资源余量门槛拦截，0模型调用/更新。旧空目录与阻塞记录保留在 `engineering/resource_dispatch_recovery/`。明确恢复后，同一个末段从完整AdamW/RNG/输入游标继续到6000，现已完成。
- 上述资源恢复入口 `recover_resource_dispatch.py --execute` 是已处理事件的专用命令，**现在不要再次执行**。启动时资源已经恢复，原资源门槛也可通过；更新保守进程估计没有形成模型重试。WDDM独立进程峰仍为NA，未把Torch显存当作完整进程峰。
- 实际图审stdin的UTF-8 BOM解码修复发生在写入观察之前，保留 `review_work/stdin_bom_fix.json`，没有模型或评分调用。图审串行交接保留在 `review_work/chunk_0_handoff.json`。

本次更新前的恢复说明原文与hash保留在 `engineering/readme_resume_completion/`。其他工程修复及其版本以 `engineering/` 和账本为准，不能覆盖或删除失败记录。
## 最终交付包

最终执行命令为 `build_delivery.py --build --training-analysis final`，仅打包已有证据，不触发模型、生成、重评分或训练。

本地目标为 `delivery/R2_USER_DATA_SCALING_32_67_REVIEW.zip`；解压后从 `REPORT.md` 与 `evaluation/gallery/index.html` 查看结果。包内保留完整逐输出/逐条件/逐父CSV、840行实际AI_PROXY图审、280张RAW/POST页面、最终训练分析和两张统计图、实际评分源码、新脚本快照、code.diff及源文件hash。区域构造与筛选证据在 `inherited/`，所有数据筛选顺序限制仍保留。

大权重、RAW/POST网格数组、轨迹、冻结噪声/查询与来源完整网格仍在本地，包内 `local_artifacts.csv` 和 `local_artifacts.json` 保存实际路径、大小、已有hash来源及NA说明；打包不重新hash大权重。这个审阅小包不能独立训练或重新生成。`PACKAGE_MANIFEST.json` 保存ZIP成员的来源及内容hash，实际打包后逐项从ZIP读取验证。

完成后停止。不自动改变默认模型、不commit或push，也不再运行科学任务。若仅查状态，使用前文只读 `--stage status` 命令。