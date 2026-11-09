# 2026-10-06 权重清理后的当前状态

用户保留 RAW / START，并要求清理不用的权重。

本目录现在仅保留 `A_FM_attachment1000_total6000.pt`，作为旧报告指定的等预算对照终点。A 的 200/400/600/800 恢复点及 B_BOUNDARY、C_SEAM、D_BOUNDARY_SEAM 的全部五档权重已删除，共 19 份、29,736,512,413 bytes。训练 OT 缓存已移除。

原同名 JSON、checkpoint_manifest、训练 pointer、RAW、评分及历史报告仍是当时执行的证据，未改写；其中 B/C/D 等已删除路径不再支持恢复或重新采样。正式默认仍为原 HYBRID START total5000，两条原有权重路径均保留。

逐文件删除清单及核验见 [清理报告](../../../maintenance/local_anchor_retirement_20261006/CLEANUP_REPORT.md)。未重新训练或生成，不自动恢复失败路线。
