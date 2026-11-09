# Loop＋motif 收口与本地存储清理

2026-10-29，用户授权后实际执行完成。状态 `COMPLETE_VERIFIED`。**本轮冻结loop＋motif不晋级，到此停止；EXPOSURE和Chair保留流程不变。**

L0/L3的RAW、FINAL联合正例均0/8；同面板G1为1/8，EXPOSURE为8/8 RAW、4/8 FINAL。L3相对G3的FINAL FF下降1.835pp，但CF上升1.297pp、I上升8.162pp。完整结论和边界见[实际实验报告](../../experiments/r2_bench_loop_motif_mini_v1/REPORT.md)，不是对所有循环架构的否定。已补入WORKSPACE及研究路线记忆，不启动新训练或扫描。

## 已执行的清理

| 操作 | 数量 | 存储变化 |
|---|---:|---:|
| 删除已由200步正式终点替代的PLAIN200/FLIP200旧150步滚动检查点 | 2份 | 3,130,122,310字节 |
| 删除r2/native_t1/models下可重建的Python字节码 | 185份 | 1,854,990字节 |
| 两轮motif的封存评分JSON进行NTFS无损压缩 | 8,328份 | 文件存储量减少1,167,694,774字节，约1.088GiB |

删除量合计约2.917GiB；加压缩后按文件统计节省约4.004GiB。**E盘可用空间实测增加4,303,163,392字节，约4.008GiB。** 卷空间变化可能包含其他进程活动；文件存储量来自Windows `GetCompressedFileSizeW`，与卷剩余空间分开记录。

两份旧checkpoint不是字节重复副本，而是明确被正式终点替代的local150/global25650中间恢复点。local200/global25700正式完整检查点和推理导出都保留。原latest元数据、历史报告和完成hash不改写；训练目录附独立退役说明，不能再从被删latest恢复。

压缩范围仅为以下两个已完成实验内的 `base_pair_caches`、`cpu`、`raw_scores` 下JSON：

- `experiments/r2_bench_full_prior_motif_guidance_v1`
- `experiments/r2_bench_loop_motif_mini_v1`

压缩不删除文件、不改变逻辑内容、路径或引用，正常读取透明解压，无需修改评分和恢复代码。压缩后8,328项逐文件SHA256全部一致，原复核路径仍可使用。未压缩模型权重或活动训练数据。

## 保留与核验

- 正式终点、官方模型、当前EXPOSURE/窄B及Chair参考资产保留；11项关键保护资产SHA256前后核验通过，其他100份权重大小/修改时间未变。
- 训练原数据、条件、固定噪声/查询、逐步轨迹、RAW/BASE/FINAL、原评分、图审、图集、实际源码及审阅ZIP全部保留。
- 源数据删除0、RAW/POST删除0、评分证据删除0、模型调用0、训练更新0；没有commit、push或默认切换。
- 清理前确认无项目训练/采样进程；逐文件白名单删除，检查绝对路径仍在工作区并拒绝重解析点；未使用递归删除。
- 准备脚本首次因PowerShell目录对象类型判断停止，发生在生成清单和任何删除/压缩之前；修正后执行通过。该准备失败未造成数据变动。

## 实际记录

[完成回执](cleanup_receipt.json) · [白名单及原hash](prepared_manifest.json) · [删除账本](deletion_ledger.jsonl) · [压缩前后hash与大小](compression_ledger.jsonl) · [实际清理脚本](cleanup.ps1)

[实验报告](../../experiments/r2_bench_loop_motif_mini_v1/REPORT.md) · [逐例177字段](../../experiments/r2_bench_loop_motif_mini_v1/analysis/per_output_original177.csv) · [完整图与截面](../../experiments/r2_bench_loop_motif_mini_v1/gallery/index.html) · [审阅包](../../experiments/r2_bench_loop_motif_mini_v1/R2_BENCH_LOOP_MOTIF_MINI_REVIEW.zip)
