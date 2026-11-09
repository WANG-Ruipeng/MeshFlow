# 下一次从哪里继续

这是一份已有证据的交接索引，没有新增训练、生成、评分或图审。原报告保持各自的时间、条件和证据范围。

## 先恢复研究上下文

1. 读[研究综述](snapshot/MeshFlow/maintenance/research_synthesis_20261028/RESEARCH_SYNTHESIS.md)、[Chair 谱系](snapshot/MeshFlow/maintenance/research_synthesis_20261028/chair_evidence.md)和[Bench 谱系](snapshot/MeshFlow/maintenance/research_synthesis_20261028/bench_frontend_evidence.md)。这三份文件是 10 月 28 日的整理，后续结果另看下面的原报告。
2. 读[指标政策](snapshot/MeshFlow/native_t1/docs/control_metrics.md)。保留真实三角形 C 的多样完成是任务，匹配唯一来源父只是重建诊断；RAW/POST、指标版本、NA/UNKNOWN 和实际评阅范围不能混在一起比较。
3. 新机制先查[路线记忆](snapshot/MeshFlow/r2/docs/research_route_memory.md)、[路线登记](snapshot/MeshFlow/r2/docs/research_route_registry.json)及[历史来源索引](snapshot/MeshFlow/r2/docs/research_history_sources.md)，再查工作记录中的后续。旧负结果只关闭其已测配置，不能泛化为整个机制家族无效；旧建议也可能已经完成后续测试。

## 已有研究位置

- Chair 的 `PLAIN_S2000 + LONG` 与归档 CPU 后处理有有限正证据。`PLAIN` 仍属于 SEAM 表示；LONG 与 B_POOL47 是并行分叉。BACK MULTIPART 是另一分支，不能把不同证据拼成统一最优模型。
- 10 月 28 日综述把 Bench `EXPOSURE / M_PLUS3000` 记录为后续生成器研究的优先候选；保留窄池自然 N 的 B、Native6000/SEAM3000 等参照。它并未成为完整流程的全面升级，也不能将未实测的前端/后端组合称作新最佳方案。
- 维护 CLI 默认仍是 Chair HYBRID START。研究参考与维护默认是两个不同身份。这份文字归档没有修改它们。

以上是原报告的交接摘要，详见链接中的训练谱系、面板、分母和代价。

## 10 月 28 日整理之后的记录

| 后续路线 | 原始报告 |
|---|---|
| 冻结 motif / loop 小试与收口 | [报告](snapshot/MeshFlow/experiments/r2_bench_loop_motif_mini_v1/REPORT.md) |
| 固定 C 多 LOD 数据准备 | [报告](snapshot/MeshFlow/experiments/r2_bench_multilod_data_v1/REPORT.md) |
| PLAIN1500 / MULTILOD1500 对照 | [报告](snapshot/MeshFlow/experiments/r2_bench_multilod_training_v1/REPORT.md) |
| PATCH、训练式 LOOP、局部 RL 三机制快筛 | [报告](snapshot/MeshFlow/experiments/r2_bench_three_mechanisms_v1/REPORT.md) |

这些后续没有形成统一的整体升级。有限连接或局部质量收益与几何/泛化代价须一起保留，具体以每轮报告为准。不要把短预算、开发面板上的结论扩大成普遍不可能。

## 开始新的研究之前

从原方案找出条件构造、面数/批量、模型起点、loss、优化器、精度、种子、样本面板、预算和停止条件；实现位于 `snapshot/MeshFlow/r2/`、`snapshot/MeshFlow/native_t1/` 及其他来源的快照中。关键结果表按来源路径保存，可以用 DOCUMENT_INDEX 和 MANIFEST 定位。

本地权重和数据清理后，需要另行取得允许使用的数据/官方模型，并重新准备当前方案所需资产。报告中的资产哈希和文件名是身份线索，不能替代文件本身，也不保证还能重现历史训练。不要为了通过旧核验而自动恢复或重训已退役路线。

本档案足以延续研究思路、查阅原实现与结果；不承诺从任一历史 step 续训或逐位重放。
