# R2_BENCH_CLEAN_MULTIPART_SEAM_V1 交付索引

本目录保留本次真实训练、生成、CPU后处理、完整评分与图审。没有新增授权外训练、阈值扫描、best选择或默认切换。

## 先看结果

- [完整报告](REPORT.md)
- [完整图集，含两视角及正侧截面](gallery/BENCH_CLEAN_FIXED48_MAIN_AUX/index.html)
- [实际AI_PROXY逐单元记录](visual_review/per_cell.csv)
- [训练曲线图](training_summary/20261020T035233_197593Z/loss_and_fixed_probes.png)
- [训练、探针、曝光、时间与显存摘要](training_summary/20261020T035233_197593Z/compact_training_report.json)
- [正式TRAIN/DEV关键表](audits/final_key_summary.json)

## 权重与恢复状态

| 文件 | 用途 | SHA256 |
|---|---|---|
| [checkpoint_step1500.pt](training/checkpoint_step1500.pt) | global13500完整训练状态，含生成器/头/AdamW/RNG | `0a96b123cc8e5cd4e7c1fc64588a1baaa8de93d83debe4f941b3c5f866a2db42` |
| [BENCH_CLEAN_MULTIPART_S1500_generator.pt](inference/BENCH_CLEAN_MULTIPART_S1500_generator.pt) | 实际预定终点生成器导出 | `474d06f8225fbffeab099807a1d32557b85a724d2481171d9e75a1e7d3a97011` |
| [BENCH_CLEAN_MULTIPART_S1500_occupancy.pt](inference/BENCH_CLEAN_MULTIPART_S1500_occupancy.pt) | 新头、结构和TRAIN归一化元数据 | `4bc2c1e471707b693e45c7eb9564e6e34d222c97d5c2369b27679dcb86eceb5f` |

完整训练检查点1565305991字节，生成器521665915字节，头81993字节。权重不重复复制进源码归档。

## 冻结身份与逐例证据

- [训练注册](training_registration.json)、[推理注册](inference_registration.json)、[数据清单](data/manifest.json)、[固定评测请求](eval_manifest.json)。
- `training/calls.jsonl`、`training/exposure.jsonl`、训练loss与四次固定probe、`training_audit/`。
- `evaluation/NEW_HEAD/`、`evaluation/OLD_S15000_LAYOUT/`：每请求RAW及51状态轨迹、单次D0、PRE_GEO/BASE、调用账本与锚身份。
- `historical/`：144份历史缓存的独立CPU级联；`cpu/`：96份新输出的级联。
- [逐输出179指标及附加字段](comparison/per_output.csv)、[逐输出配对](comparison/paired_per_output.csv)、[逐条件](comparison/per_condition.csv)、[逐父](comparison/per_parent.csv)、[逐父配对](comparison/paired_per_parent.csv)。
- [全部父等权汇总](comparison/summary_parent_equal.csv)、[关键指标含部位汇总](comparison/key_summary.csv)、[拓扑分层](comparison/topology_strata_summary.csv)、[DEV官方划分](comparison/dev_official_split_summary.csv)。
- [固定BASE分母/点跳诊断](post_extras/all/per_output_fixed_BASE_and_pointjump.csv)、[安全账本](post_extras/all/safety_ledger.csv)、[移动旧面形状检查](post_extras/all/moved_old_faces_shape_audit.csv)、[指标口径](post_extras/all/metric_policy.json)。
- `topology_audit/`：316活动条件与24评测条件的拓扑、窄缝、槽身份与去向。
- [头逐边预测](layout_diagnostic/per_edge_with_slit.csv)、[头完整汇总](layout_diagnostic/summary.csv)；不把排除的全负条件从全组汇总删掉。

## 实际代码、账本与完成凭据

- 实现：[r2/pilots/bench_clean_multipart_seam_v1](../../r2/pilots/bench_clean_multipart_seam_v1/README.md)。
- `source_archive/`：68份本次实际代码及复用训练、采样、评分和后处理小源码/政策副本；外部MeshFlow-Control评分代码另列external。
- [源文件hash](source_hashes.json)、[code.diff](code.diff)。diff仅为新增独立pilot文件对/dev/null；不把其他未提交修改纳入本次。
- [独立执行审计](independent_execution_audit.json)、[CPU执行账本](cpu_execution_accounting.json)、[最终完成凭据](completion.json)。
- `engineering_revision_01/`与`audits/*bom*`保留工程失败/修正；`dispatch/`保留实际并发、资源、阶段时间与进程身份。

在现有WSL Ubuntu环境中，仅检查状态、不启动计算：

```bash
cd /mnt/e/MeshFlow
PYTHONPATH=/mnt/e/MeshFlow:/mnt/e/MeshFlow/r2/src \
  /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B \
  -m r2.pilots.bench_clean_multipart_seam_v1.resume
```

恢复入口不会自动追加步数或重启失败任务；显式执行某阶段前仍须遵守原预算与状态检查。本轮已完成后不要重跑采样或选择其他检查点。`completion.json`核对权重、报告、图审、源码归档和评分hash后写入，随后停止。
