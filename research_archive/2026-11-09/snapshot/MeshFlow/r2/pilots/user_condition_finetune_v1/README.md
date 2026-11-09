# R2 用户式条件续训小实验

从**主实验 R2 full step6000**（包括 AdamW、逐参数步数及 RNG）独立分叉两条1000步续训；不使用确认实验权重，不修改历史6000步训练器。

- R2_FROZEN：主实验R2，0更新。
- R2_CONT_OLD：相同25父的200条C20、195条C40；保留每批条件比例/增强配平。
- R2_CONT_USER：同25父、117条已核对活动条件；先父后条件抽样，HOLD不进入训练。
- Native/T1 + role + Geo，free-only FM，全程HYBRID0.25；BF16前向、FP32参数与积分，原确定性设置。
- 每臂effective batch8/microbatch1、8不同父、4clean/4width；AdamW状态接续，local1..1000/global6001..7000。FM分母仍为整batch的9×free面数。
- 两臂的父、N、来源M、alpha和t配对；新旧K、free集合及实际noise数组/OT映射不同，不声称仅改变单一语义因素。
- 预检最多16F/8B、0更新；500只用于恢复，固定1000评价。
- 新6条件×2噪声、旧8条件×2原噪声，三模型84结果。16旧RAW核验复用，正常68新增Euler50。所有结果同一CPU评分配置重算，无POST、教师、S/E伪值或唯一GT重建排名。

完整产物路径：

`E:/MeshFlow-Control-artifacts/experiments/r2_user_condition_finetune_pilot_v1/`

原准备数据保持只读：

`E:/MeshFlow-Control-artifacts/experiments/chair_r2_user_conditions_v1/active_conditions.json`

## 入口与恢复

在既有WSL虚拟环境中使用 `run.py --stage <stage>`。支持 status、prepare-data、prepare-eval、preflight、frozen、train-old、train-user、sample-old、sample-user、score、render、verify-protected。根目录的 registration.json 是预算/任务合同；code_manifest.json 锁定实现，阶段账本保留真实尝试。

GPU阶段还须给出控制器记录的 `--dispatch-receipt <json>`。这份记录来自当前资源读数与已授权任务的实测峰值，用于按阶段分配并行度，不额外启动探测训练。每臂独立状态与日志。发生OOM/数值失败，停止新增GPU派发并保留失败；不自动换种子、减batch、回滚已执行更新或额外续训。

CPU样本已完成时只能按注册缓存读回。训练正常中断仅从完全一致的新分支安全检查点及下一个输入游标恢复；不得从500步重放已经执行的501+步。已完成采样单元核验后复用；已消耗的失败调用不能当作未发生。

`trainer.py` 的纯FM算术来自原 `training/schedule40.py` COMMON 路径，必要新分支时钟和schema独立实现；`data.py` 直接调用原HYBRID/free-only OT函数；`evaluation.py` 导入原CPU几何评分数学并在输出中保留源码快照。

## 解释边界

当前来源M并非无缺陷目标；14个已知缺陷父仍保留，不从生成分数扣除来源问题。新DEV仅6父（2座面/4靠背），属重复使用的开发面板。完整图集的整体质量与C用途审阅为AI_PROXY，不是人类可用率。新旧任务分开汇总，完整披露逐条件和两噪声变化。

固定任务完成后停止；不自动晋级默认模型，不commit/push。最终结论以本次实际账本和结果报告为准。
