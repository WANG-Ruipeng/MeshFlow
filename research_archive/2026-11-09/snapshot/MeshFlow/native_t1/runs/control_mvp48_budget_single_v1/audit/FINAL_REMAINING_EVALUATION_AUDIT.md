# 拟合探针与留出回归：最终只读审计

结论：**PASS**。本次执行 15,082 项断言，0 项错误；仅读取现有 JSON、CSV、NPZ、代码和 checkpoint 字节。审计自身没有 GPU 任务、模型调用、OT、重新评分、表面重采样、训练更新或 checkpoint 张量反序列化。

| 本次范围 | 已核验数量 |
| --- | ---: |
| 拟合探针生成：源 T1@1000、probe@500、probe@1000，各 8 条 | 24 |
| 拟合探针 teacher：3 个状态，各 6 条 | 18 |
| 留出生成：T1、Geo、Full@5000，各 32 条 | 96 |
| 新增生成合计 | 120 |
| 直接复用的源 Session C@1000 历史 raw/score | 96 |
| 留出配对回归行 | 96 |
| 直接核对的旧值、新值及差值字段 | 3,648 |

前一份 [FINAL_DATA_PROTOCOL_AUDIT.md](FINAL_DATA_PROTOCOL_AUDIT.md) 已核验 1,072 条生成与 720 条 teacher；两份审计覆盖固定矩阵 **1,192 条新生成、738 条 teacher**。直接复用的 96 条旧 Session C 输出不计入新增生成。

## 输入与模型身份

双对象拟合探针使用训练 UID 排序的前两个对象 `103b75dfd146976563ed57e35c972b4b`、`12c2b7080b07146a9d4d5dcd2cc33826`，均为 p0，8 条生成对应每对象 4 个固定噪声；6 条 teacher 对应每对象 t=0.2、0.5、0.8。核验了冻结 fit plan、panels、实际数组和全部 teacher 输入身份；未重建随机输入。

留出的 32 条实际输入逐条核对源 Session C preparation ledger 的路径、文件 SHA256、数组身份和 16 份参考。新阶段直接读取原输入与原参考；原 96 条历史输出逐条绑定原 gpu_run、scoring_run、三臂 step1000 权重、实际 raw/score 文件与评分器身份。旧评分全部直接复用，未产生 fallback 重新评分文件。

6 个新评价 ledger 均为 PASS，checkpoint 字节 SHA256 与 sidecar、ledger 和模型身份一致；两个 probe checkpoint 的 FRESH_PROBE 身份及源 T1@1000 关联一致。冻结协议、面板、计划、执行代码和源评分器文件身份通过检查。

## 输出与评分口径

120 对新 raw/score 的文件哈希、数组哈希、N/K、C、Z、mask、固定 Euler 50 NFE、BF16 forward / FP32 integration 和严格确定性运行记录均匹配。每条输出记录了 51 个状态的 C 位级夹持；其中 6 条保留完整轨迹，已直接逐状态核验，另 114 条依据执行时记录的 51 状态审计及实际最终 raw 核验，不声称其完整轨迹仍已保存。

18 条 teacher 的冻结输入、输出原始文件、单次 forward 计数、t、free 分母、有限数值与 Yhat 的 C 保持均通过检查。120 条新评分的预测 free 几何哈希与实际 raw 的 free 部分一致，且评分未调用模型、OT 或几何梯度。96 条历史评分同样只提取真实 bool known mask 之外的 free 部分；原完整 51 状态轨迹和原始噪声起点均已核验。

`holdout_regression.csv` 的 96 行全部 comparison_status=PASS；旧值逐项等于原 score，新值逐项等于实际 step5000 score，差值逐项等于 5000−1000，共核验 3,648 个配对指标字段。CSV SHA256：`8f653336d25aedc3d4f259283709ac0ffe0a9a3d9c6e09505a2e33fdbf5de96a`。

## 范围与限制

本审计证明输入、权重、执行记录和分数复用的身份一致；PASS 不代表几何质量通过了额外阈值。8 个留出父对象此前已被观察，不是新的确认集。总训练预算、全局调用总账、最终报告和顶层 manifest 由总入口另行验收。

完整证据、逐条历史复用绑定、ledger 摘要与限制见 [FINAL_REMAINING_EVALUATION_AUDIT.json](FINAL_REMAINING_EVALUATION_AUDIT.json)。