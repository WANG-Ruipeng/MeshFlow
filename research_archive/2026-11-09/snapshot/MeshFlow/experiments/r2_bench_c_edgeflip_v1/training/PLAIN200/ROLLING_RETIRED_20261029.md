# 旧滚动权重清理

用户于2026-10-29授权收口与清理。`checkpoint_latest.pt` 是local150/global25650的旧中间恢复点，已由local200/global25700的正式终点替代；它不是正式终点的重复副本。

本次删除旧滚动文件，继续保留 `checkpoint_step200.pt`、`generator_step200.pt`、全部元数据、日志和评价。原 `checkpoint_latest.json` 与 `rolling_checkpoint_retirement.json` 是历史状态，保持原样；物理删除事实以本告示和中央账本为准。不能再按旧latest路径恢复，不自动重训或伪造该文件。

实际删除hash与保护核验见 [清理账本](../../../../maintenance/loop_motif_closeout_20261029/deletion_ledger.jsonl) 和 [收口记录](../../../../maintenance/loop_motif_closeout_20261029/CLOSEOUT.md)。
