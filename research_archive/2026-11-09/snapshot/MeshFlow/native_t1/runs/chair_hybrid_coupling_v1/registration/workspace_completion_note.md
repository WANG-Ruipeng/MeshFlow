
## 2026-10-04 Chair Hybrid Coupling 有限试验完成

`native_t1/runs/chair_hybrid_coupling_v1/` 已实际完成同源 Geo4000 两臂各1000更新、固定A/B/C与响应诊断。两新终点为本实验 pilot1000/total5000，身份见 checkpoint_manifest.json；不要与历史预算实验5000步混称。未切换FM默认模型或JEdge5可选模型。

实际总计2000更新、432登记请求（18原始NA）、292完整采样、122历史raw缓存、2400 BF16交叉诊断、72 FP32诊断及10预检前向；总前向33082，模型失败/未决0。866个保护依赖前后hash一致；冻结新实现未改变，训练曝光、主指标算术及三个检查点轨迹独立审计PASS。

结果为 GEOMETRY_SIGNAL_WITH_TRADEOFF：HYBRID相对BASE验证A20固定查询MSE下降约40%，6/8父改善，覆盖与p95改善；但小角面增加、跨噪声控制分歧、40%上下文部分指标及已观察留出退化。优先下一步是现有raw的小角面收益归因，检验误差收益是否主要由新增细长面贡献；该后续未执行，没有追加训练、loss或求解器扫描。

详见本轮 CHAIR_HYBRID_COUPLING_REPORT.md 与 DELIVERY_INDEX.md。纯CPU审计读取器/报告文案修正单独保留原版、diff和哈希；正式实验没有静默重跑。当前固定任务已停止。
