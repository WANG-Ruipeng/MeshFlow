# 有限假说登记

本轮从原 OT_HYBRID pilot1000 / conditional_total5000 开始。冻结定义见 protocol.json；输入、噪声和门槛在第一次真实模型采样之前冻结。

1. **H1：真实晚期生成状态与终态 GT 对应的重放。** 使用 START 在 8 个训练父、每父 2 噪声上的完整 Euler50 轨迹；S、I、E 三个独立困难队列决定重放噪声。对 t=0.6/0.8 的真实状态，以同最终坐标尺度的 free 面/角双射匹配 GT，监督剩余位移速度。原 batch8 HYBRID-FM 不变，额外一条监督权重 0.1。检验是否通过固定 DEV 的 A/B 控制门槛，同时满足相对 START 和匹配对照的形状约束。
2. **H2：额外 teacher FM 是必要匹配对照。** 相同队列、父、C、初始噪声、时间、额外前向/反向数量，改用原 pre-OT、HYBRID teacher 直线及速度。用于检验 H1 收益是否超出额外 GT 监督本身。原 FM 续训另作为无额外损失的对照。H1 是状态及对应方式的复合干预，本轮不能归因为单一机制。
3. **H3：已冻结并启动的辅助损失限幅修订。** 完整 H1/H2 DEV 与真实训练反馈显示训练内收益未迁移，且 H1 辅助梯度经常占优。保持同一 START、原主 FM、队列、时间和 GT 对应，只令 lambda_eff=min(0.1, detached_primary_FM/max(detached_aux_MSE,1e-30))，不改变原 FM 正权重。它限制损失数值，不保证限制梯度范数。教师完整 1000 步反事实证明该规则始终不活跃，故可复用这一精确匹配路径；独立重复需另证。预算 1000 更新、DEV 32 和原反馈 16 条；只用预选 1000 步终点判晋级。风险包括 stale 轨迹、有限父过拟合和丢失 H1 已有局部形状收益。未通过原质量及 A/B 门槛即停止，不新增第四假说。冻结方案及全部证据见 decisions/H3_recipe.json。

所有模型真实更新数、失败及采样次数以 ledger 为准。独立重复和封存 CONFIRM 仅在正面 DEV 信号及质量门槛通过后执行。Human usability 为 NOT_RUN；默认 START 不变。


H3 FROZEN: TRAJECTORY_REPLAY_CAPPED, same START and original TR inputs/targets, lambda_eff=min(0.1, detached_primary_FM / max(detached_aux_MSE,1e-30)). It caps loss value, not gradient norm. Existing full teacher path is exactly cap-inactive at all1000 steps; frozen proof and risks in decisions/H3_recipe.json. Only1000 is eligible for promotion, 500 intermediate. Budget1000updates + DEV32 + original TRAIN_FEEDBACK16. No fourth hypothesis.
