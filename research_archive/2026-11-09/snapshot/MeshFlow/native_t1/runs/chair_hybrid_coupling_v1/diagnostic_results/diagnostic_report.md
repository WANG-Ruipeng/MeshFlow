## 响应诊断：逐父、逐噪声与 FP32 复核

BF16/FP32 在 1 个预选冻结状态的 I/P 相反方向判断不一致。目前不能据 BF16 单独判定注入不足或传播抵消为主要原因；逐曲线代数分解仍有效。

诊断对象固定为表B按UID排序前2训练父、前2验证父，各原2份噪声；每检查点8条曲线。I/P/S均在有效free标量坐标上计算RMS；零范数余弦留空且状态为UNDEFINED。

|检查点|完成曲线/8|曲线平均相反步比例|积分I/P相反曲线|最大闭合RMS|FP32方向不一致/可比状态|
|---|---|---|---|---|---|
|START|8/8|0.52551|4|2.02996e-06|0/4|
|OT_BASE|8/8|0.433673|3|2.07106e-06|1/4|
|OT_HYBRID|8/8|0.492347|5|2.22095e-06|0/4|

FP32只复核冻结的首训练/首验证父noise0、k=0/25/49四组合。CSV保留实际autocast关闭、Linear输入/输出/参数dtype、Geo实际重算与输入hash；不是把BF16输出转成FP32。其余状态仍是正式BF16 Euler50轨迹。

逐曲线表按相同检查点、UID和noise关联真实几何fixed_query_MSE/p95、宽度r_base/b_self/b_GT及各目标列优势。缺失评分关联标BLOCKED，不用另一噪声或另一父替代。父表先等权平均两份噪声，同时保留逐噪声方向，不能把两噪声当独立父对象。

较大I/P范数不等于控制更正确；相反方向证明代数抵消存在，不自动证明其为错误主要原因。没有额外阈值把“注入接近零”或“Jacobian爆炸”写成已证实。闭合残差用于核对实际Euler数值关系，不作质量容许退化额度。

数据：`diagnostic_results/response_per_step.csv`、`response_per_curve.csv`、`response_per_parent.csv`、`fp32_precision_records.csv`。每检查点各有一张8曲线I/P/S图和一张8曲线余弦图；原向量与全部dtype审计保留在evaluation/<label>/diagnostics/。
