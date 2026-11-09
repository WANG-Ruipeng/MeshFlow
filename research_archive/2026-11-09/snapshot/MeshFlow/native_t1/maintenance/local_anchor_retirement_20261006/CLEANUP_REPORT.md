# 局部锚定路线退役与权重清理

已按用户要求保留 RAW / Chair HYBRID START，停止当前局部锚定路线。清理和最终核验均完成。

| 清理内容 | 文件数 | 删除的逻辑字节 |
|---|---:|---:|
| 边界接合实验未使用权重 | 19 | 29,736,512,413 |
| 该实验派生 OT 缓存（NPZ 与对应 JSON） | 16,000 | 27,538,573 |
| 已完整归档的局部锚定代码与运行目录 | 636 | 39,757,447 |
| 合计 | 16,655 | 29,803,808,433 |

合计约 29.80 GB 为删除文件的逻辑大小，不把它冒称精确物理磁盘释放量。另保留 17,657,410 bytes 的完整封存 ZIP，以及少量清理账本和可直接读取的结果副本。

## 已删除

- `chair_boundary_attachment_v1/checkpoints/` 下 A_FM 的 attachment200/400/600/800 恢复点；B_BOUNDARY、C_SEAM、D_BOUNDARY_SEAM 的 attachment200/400/600/800/1000 检查点，共 19 份。删除前均实际核对 SHA256，与同名 JSON sidecar 一致；权重未复制进 ZIP。
- `native_t1/runs/chair_boundary_attachment_v1/ot_cache/`：结束实验的派生训练匹配缓存。
- `experiments/mf_h1_local0/chair_bounded_attachment_v1/`：当前局部锚定实验的活动代码入口。
- `native_t1/runs/chair_bounded_attachment_v1/`：完整封存后移除其活动运行目录及重复 RAW 副本。

局部锚定实验本身没有训练、没有产生权重。清理没有扩展到其它历史模型分支。

## 保留

- 正式 START：`native_t1/checkpoints/chair_hybrid_start_total5000.pt`，以及原 `chair_hybrid_coupling_v1/checkpoints/OT_HYBRID_pilot1000_total5000.pt`。两条路径是同一个 NTFS 文件的硬链接，SHA256 均为 `d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80`。
- 原始 HYBRID/START RAW、原 C / Gaussian / 查询点、官方数据、官方权重、仍保留的维护模型和正式生成代码。
- `A_FM_attachment1000_total6000.pt`：旧报告明确保留的等预算对照终点，SHA256 为 `27e5076eda8763a99b7973c9a637ff01f9f466ae94fbae123d2b945c3307f7eb`。它不是默认模型；其余 A 中间恢复点已清理。
- 原四格实验的 RAW、评分、报告、源码证据、checkpoint manifest 与小型 sidecar JSON 保留原样。历史 manifest 内已删除的权重路径是执行时记录，当前状态见新增 `checkpoints/RETIREMENT.md`。
- 当前局部锚定的全部实际源码、32 份网格/NPZ、逐角位移与候选日志、逐例和配对 CSV、评分源码、图集、原报告、code.diff 与哈希均完整保存在 [封存 ZIP](LOCAL_ANCHOR_RETIRED_SOURCE_AND_EVIDENCE.zip)。归档按工作区相对路径组织，共 636 个条目，逐条解压流 SHA256 核验通过。

原始 16 个 RAW 的当前位置见 [保留 RAW 索引](retained_RAW_index.csv)。快速结果副本见 [原实验报告](evidence/BOUNDED_ATTACHMENT_CORRECTION_REPORT.md)、[逐例指标](evidence/per_output_metrics.csv)、[质量与成本](evidence/quality_cost.csv)。历史报告的运行目录和相对链接以 ZIP 中的原目录结构为准；归档内旧恢复说明不代表该路线仍处于活动状态。

## 验证与账本

- 578 个受保护文件前后 SHA256 全部一致，清单外保留文件变化为 0；登记删除文件残留为 0。
- 删除后实际执行 `python -B -m native_t1 sample --help` 成功，默认仍为 `chair-hybrid` / START；没有加载模型、前向、训练或生成。
- 正式 Git 跟踪代码无差异，没有 commit 或 push。维护状态记录追加至 WORKSPACE.md。
- 清理脚本最终核验阶段曾遇到 PowerShell 5.1 根 JSON 数组被额外包装的问题。独立逐文件复核先确认 578 文件未变、删除目标无残留，再修正读取循环；只重跑了核验，没有重复删除。原脚本、差异与错误细节保存在 `engineering/verification_array_reader/` 和 `verification_failure_details.json`。

完整记录：[删除清单](deletion_manifest.json)、[逐项操作账本](operations.jsonl)、[归档清单](archive_manifest.json)、[最终核验](verification.json)、[保护文件前后哈希](protected_after.json)、[入口检查](cli_smoke.json)。本轮清理到此结束，不恢复局部锚定或启动其它实验。
