# 本轮文献依据及适用边界

2026-10-07（本地时间）核验了任务指定的五份 arXiv 主来源身份与摘要。它们是研究动机，不是本轮算法身份、预训练权重或理论保证的来源；未安装或运行这些论文的模型。

- [DHO: Flow Matching Reinforcement for 3D Mesh Generation via Dynamic Homing Optimization](https://arxiv.org/abs/2610.01233)：其摘要描述目标吸引、匹配与考虑剩余时间的速度纠偏。本轮保留既有真实START状态及已匹配训练GT的剩余速度目标，不引入其在线强化学习、异步池或时间加权，也不声称完整复现DHO。
- [MARBLE: Multi-Aspect Reward Balance for Diffusion RL](https://arxiv.org/abs/2605.06507)：该文在梯度空间组合多奖励更新，使用二次规划等机制。本轮只区分loss尺度与梯度尺度，使用预先固定rho=1的标量范数预算；没有多奖励优势估计、QP、MGDA、梯度投影或EMA。
- [Self Forcing](https://arxiv.org/abs/2506.08009)：讨论自回归视频生成中训练上下文与实际生成上下文的差异。本轮重放固定START状态，没有在线滚动更新状态或自回归视频机制，因此不能借用其泛化或速度结论。
- [DiffusionNFT](https://arxiv.org/abs/2509.16117)：在前向过程上学习正负生成样本引导的更新。本轮没有正负策略目标或在线奖励学习；固定反馈池和训练强度的取舍只作为待检验动机。
- [DRaFT: Directly Fine-Tuning Diffusion Models on Differentiable Rewards](https://arxiv.org/abs/2309.17400)：作为任务指定的干预强度/训练终点消融背景。本轮不通过完整Euler50轨迹反传，也不采用可微奖励优化；500与1000在执行前固定，不能根据事后最优值选择终点。

本轮实际定义完全由 `task_contract.txt`、冻结 `protocol.json`、`recipes/` 和实际源码决定。尤其同时对辅助项乘时间权重并以其数值重新计算loss-ratio cap，可能在双限幅区抵消时间权；CPU代数检查已验证该性质，所以没有新增时间权重。梯度预算只约束当前合成前的辅助梯度范数，不保证AdamW参数更新同样受限，更不保证控制或几何指标改善。
