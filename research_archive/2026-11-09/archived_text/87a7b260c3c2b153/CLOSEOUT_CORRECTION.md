# 完成态恢复说明更正（v2）

独立只读复核通过：16比较格、6缓存＋10新RAW＋1复现、550前向；12曲线/612逐步行；10完整评分及8额外COMMON_C CF扫描；每父两噪声均值、RAW阶段、资源并行记录和科学结论均一致。

唯一工程更正：RUNBOOK不再建议完成状态运行analyze curves。曲线计算完成后，metrics_v2元数据修正改变了analyze.py hash；旧curves函数没有完成态早退，重跑会先重写派生表/图再因旧不可变收据不一致而失败。此次只修改使用说明，没有实际触发该重跑，没有重写科学表/图/收据。

已完成状态仅用run status/prepare检查；不要将closeout当任意代码版本下的通用重建命令。实际看图记录均为AI_PROXY，非人类评阅。

原COMPLETE.json和首版ZIP原样保留，包含更正前RUNBOOK的历史hash。当前权威完成核验为COMPLETE_v2.json，当前交付包为R2_BENCH_C_TESSELLATION_REVIEW_V2.zip。所有科学结果完全相同。
