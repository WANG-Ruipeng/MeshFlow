# Chair 域内三角几何控制：云端交付包

本包对应已完成的 `chair_domain_control_v1`，研究主体是 Native+Geo，保留 Native 与 Native+Geo+Joint 匹配对照。它与旧的 5000 步 MVP48 实验不是同一轮。旧结果仅作为明确标记的历史回归背景。

**实验已实际结束：三臂各新增 3000 次更新、累计 4000 步；合计 9000 次更新、582 条生成，重复生成 0 条。** 本次打包仅复制与核验既有文件，没有训练、生成、评分或改动原结果。

建议先读：

1. [结果摘要](RESULT_SUMMARY.md)：预算、主要发现与适用边界。
2. [交付验收补记](native_t1/runs/chair_domain_control_v1/DELIVERY_NOTES.md)：分母、配对口径、中断与恢复说明。
3. [正式主报告](native_t1/runs/chair_domain_control_v1/CHAIR_DOMAIN_CONTROL_REPORT.md)：完整协议和结果。
4. [证据索引](EVIDENCE_INDEX.md)：三张正式表、增量、图件、逐例证据、代码和账本的入口。
5. [云端审阅提示](REVIEW_PROMPT.md)：可复制给 GPT，要求基于证据审阅。

包内保留原仓库相对目录。原文件中的 `/mnt/e/MeshFlow/` 或 `E:\MeshFlow\` 都映射到解压后的本包根目录；分隔符相应转换。没有改写原 JSON/CSV 内的绝对路径，以保留冻结 SHA256。本轮正式报告和交付补记的相对链接可直接在解压目录中打开。精选的历史报告可能链接未随包提供的旧曲线或拟合探针。

本包包含本轮全部非权重实验文件、冻结科学代码及已审计依赖、48 个原始父对象与 repair_v2 资产、冻结外部留出输入、历史回归精确引用的 192 个 raw 和 192 个 score，以及必要的源/历史身份和执行记录。历史目录是精选依赖，不能理解为旧实验的全量备份。

大型权重按原协议不重复打包。checkpoint 清单、JSON sidecar、来源路径、逻辑步数及已登记 SHA256 均保留，另见 `OMITTED_WEIGHTS.json`。**云端能够审阅证据，不能凭本包重新校验未包含的权重字节，也不能把本包视为带权重的独立运行环境。** 恢复命令只供原工作站按原环境恢复；实验已完成，不应据此启动新训练。

`PACKAGE_MANIFEST.json` 列出每个交付文件的大小、SHA256、来源和类别；`PACKAGE_CHECKS.json` 记录本次对原有冻结清单的检查。清单本身不自含其 SHA256，整个 ZIP 的 SHA256 在 ZIP 旁的 `.sha256` 文件中。可运行包内的 `verify_package.py` 只读核验已解压文件，或传入 ZIP 路径直接核验归档；仅需 Python 标准库。

`packaging/build_package.py` 是新增的纯 CPU 打包工具。它不属于冻结科学实现，不触发模型，不修改实验账本。原科学 `code.diff`、单独的交付排版修订 diff、以及本次打包代码分开保存。
