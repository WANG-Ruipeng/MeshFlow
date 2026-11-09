# Bench轨迹诊断交付与恢复入口

最终状态以[COMPLETE.json](COMPLETE.json)及[完整报告](REPORT.md)为准。旧PARTIAL_STATUS、历史阻塞报告及早期权重画像的BLOCKED保留作事故历史，不代表尚待GPU执行。

| 产物 | 路径 |
|---|---|
| 完整结论、方法、代价、限制 | [REPORT.md](REPORT.md) |
| 全64页表面与固定位置截面 | [gallery_full/index.html](gallery_full/index.html) |
| 96行全指标及分母/状态 | [per_output_all_metrics.csv](analysis_full/per_output_all_metrics.csv) |
| 48组模型配对差值 | [per_request_metric_deltas.csv](analysis_full/per_request_metric_deltas.csv) |
| 逐父逐阶段与全部汇总 | [summaries](analysis_full/summaries/) |
| 固定态逐例与120行角色配对 | [fixed_state_full.csv](analysis_full/fixed_state_full.csv)、[fixed_state_paired.csv](analysis_full/fixed_state_paired.csv) |
| 24条续跑起终点48行 | [continuation_start_end.csv](analysis_full/continuation_start_end.csv) |
| 72行条件响应配对 | [response_paired.csv](analysis_full/response_paired.csv) |
| 来源结构及投影数值验证 | [source_reference](source_reference/) |
| C-only/来源布局差异 | [comparison.csv](layouts/comparison.csv) |
| 连续阶段面对及原槽位删面账本 | [stage_audit](stage_audit/) |
| 实际非盲AI_PROXY图审 | [main.csv](visual_review/main.csv)、[source_layout.csv](visual_review/source_layout.csv)、[continuation.csv](visual_review/continuation.csv)、[targeted_stages.csv](visual_review/targeted_stages.csv) |
| 逐参数更新、精确SVD、更新余弦 | [weights](weights/) |
| 40实际激活库上的1320行ΔWh | [delta_W_h.csv](weights/functional/delta_W_h.csv) |
| 本轮代码最终快照 | [source_snapshot_final](source_snapshot_final/) |
| 实际原评分源码快照 | [scoring_source_snapshot](scoring_source_snapshot/) |
| 工程改动diff、哈希与完整性 | [code.diff](code.diff)、[delivery_hashes.json](delivery_hashes.json)、[final_integrity.json](final_integrity.json) |
| <10MB网页GPT审阅包，无权重/大RAW | [BENCH_TRAJECTORY_DIAGNOSTIC_REVIEW.zip](BENCH_TRAJECTORY_DIAGNOSTIC_REVIEW.zip) |

本地大资产：`evaluation/{COMMON,USER}/{request}/raw/`含32份RAW与51态/50速度；`post/`含实际D0提议、PRE_GEO、来源定向POST及收据；`cpu/{label}/{request}/CLEAN_FINAL_POST/`含最终网格。`execution/USER/fixed/`含来源布局RAW和中途续跑NPZ；`execution/COMMON/fixed/`含实际激活。原检查点路径/hash由`registration.json`锁定，不复制新权重。

审阅包包含全部16张主配对页、三张分析曲线、全部CSV、实际评分/诊断代码、协议及图审。完整五阶段、oracle、continuation图像、大数组及冗长面对检测JSON缓存仅保留本地；小包中的全图库索引对这些文件的链接需要本地目录，不能当成缺失实验。全量delivery_hashes也含本地证据；小包不含对应大文件并非哈希失配。

只读完成复核：

```powershell
powershell -NoProfile -File E:\MeshFlow\r2\pilots\bench_trajectory_diagnostic_v1\finish.ps1
```

等价WSL命令：

```text
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_trajectory_diagnostic_v1.closeout --verify-only
```

该入口只检查冻结文件和输入哈希，不训练、不前向、不重采样。旧`resume.py`也增加完成状态守卫；不再用故障恢复参数重放已完成模型任务。原GPU失败和2次重算均保留账本，不抹去事故。
