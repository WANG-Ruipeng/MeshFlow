# 匿名门判补充复核

**门判 FAIL 正确，本轮在 GPU 派发前停止；SHIFT→RELATION 候选未采样，不能称为候选的负生成结果。** 本补充只重算已封存的 32 个 DEV FINAL 配对、64 条 AI_PROXY 匿名记录，不重新看图、改标或改变门槛。

采用 PLAUSIBLE：RELATION 17/32，SHIFT 21/32。总量门通过；严格 REPLACED_OR_IGNORED→PLAUSIBLE 只有 2 个请求、1 个父，不满足至少 2 父。UNCERTAIN→PLAUSIBLE 不得替代严格改善。

| 采用配对变化 | 请求数 | 不同父数 |
| --- | ---: | ---: |
| 明确未采用→采用 | 2 | 1 |
| 不确定→采用 | 4 | 4 |
| SHIFT 独有采用（前两项合计） | 6 | 4 |
| 原采用丢失 | 2 | 2 |
| 采用→不确定 | 1 | 1 |
| 采用→明确未采用 | 1 | 1 |
| 共同采用 | 15 | 7 |

各集合的精确请求、父 ID、condition_id 和实际 part_type 已写入 summary.json；父数不可跨集合相加。

| 轴 / 枚举 | RELATION /32 | SHIFT /32 |
| --- | ---: | ---: |
| overall_shape: REASONABLE | 24 | 24 |
| overall_shape: INCOMPLETE | 4 | 6 |
| overall_shape: UNCERTAIN | 4 | 2 |
| C_use: PLAUSIBLE | 17 | 21 |
| C_use: REPLACED_OR_IGNORED | 3 | 2 |
| C_use: UNCERTAIN | 12 | 9 |
| support_retention: PRESENT | 26 | 25 |
| support_retention: MISSING_OR_DAMAGED | 2 | 4 |
| support_retention: UNCERTAIN | 4 | 3 |
| visible_fragment_or_gap: YES | 15 | 14 |
| visible_fragment_or_gap: NO_OBVIOUS | 7 | 7 |
| visible_fragment_or_gap: UNCERTAIN | 10 | 11 |
| layering_or_fold: YES | 13 | 9 |
| layering_or_fold: NO_OBVIOUS | 6 | 8 |
| layering_or_fold: UNCERTAIN | 13 | 15 |

**匿名重评并不稳定复现旧非盲标签。** 下表比较同一组旧网格、同一 DEV FINAL 请求；变化来自评阅者、匿名呈现和比较上下文等因素，不能解释为模型或网格改变，也不能据此单独证明哪一次标签是真值。

| 轴 | RELATION 标签变化 /32 | SHIFT 标签变化 /32 |
| --- | ---: | ---: |
| overall_shape | 14 | 16 |
| C_use | 19 | 11 |
| support_retention | 16 | 13 |
| visible_fragment_or_gap | 20 | 21 |
| layering_or_fold | 9 | 12 |

共 64×5=320 项标签比较，151 项不同。逐枚举转移和旧/新计数见 summary.json；逐条证据仍使用已封存的 historical_label_differences.csv。当前配对与旧非盲成功率必须分别陈述，不能跨轮拼成统一成功率。该匿名评阅仍是 AI_PROXY，非独立人类盲测。

账本核对：新训练更新、生成器前向、D0 前向、RAW、POST、新评分均为 0。无 workflow/state、采样 shard、真实调用/轨迹或候选评分/图集完成产物；24 份冻结 CPU layout receipt 属于准备缓存，不能算生成。已验证原 review、图片/映射、gate、REPORT 与 COMPLETE 哈希及身份一致。

正确的有限结论是：SHIFT 在本次匿名采用总量上仍有增量，但严格跨父采用改善未达到预设继续门。因此不启动交叉采样；这既不否定所有 SHIFT 局部收益，也不回答尚未执行的交叉候选质量。本补充没有训练、重评分、生成、补评、改门、commit 或 push。
