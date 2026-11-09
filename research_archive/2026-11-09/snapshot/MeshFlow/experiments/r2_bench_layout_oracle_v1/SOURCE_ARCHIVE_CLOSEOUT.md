# 收口源码补充快照

本补充快照在原 `source_archive/` 之后独立保存，不重写原源码归档、`code.diff` 或 `source_hashes.json`。

- `source_archive_closeout/control/` 保存本次现有全部 7 个收口 Python 脚本，包括 `write_report.py`、`report_tables.py`、完整性核验和训练汇总工具。
- 另保存本归档工具，合计 8 文件、67,565 字节。
- `supplement_hashes.json` 逐项记录原路径、归档路径、SHA-256 和字节数；所有复制均核对原件与副本相同。
- `audits/report_read_only_check.json` 核对报告 16 行数值主表、4 行图审计数、硬编码修正后的差值及 HOLDOUT FF，以及 12 个相对链接。

最初报告正文少量差值及 HOLDOUT 舍入数与 JSON 不完全相符；主控已将这些段落改为从 `results/report_numbers.json` 动态生成，之后才封存此补充快照。差异不改变实验结论，原数值表未修改。

报告与汇总表的此次审阅 SHA-256 保存在补充清单中。图审仍为非盲 AI_PROXY，补充归档不构成额外评分或独立人类认证。

本操作没有模型前向、训练、重评分、权重/数据复制或 Git 操作。原快照清单 SHA-256 为 `3a5c9fe50ae53d7640f0ae745795e7b040b8c184514d661b357fbed48af0ac93`，与原封存值相同。
