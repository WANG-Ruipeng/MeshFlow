# R2_NFT_TRAINING_C_FROZEN_V1

固定八条已经获得NFT奖励监督的原C，各四份预登记新噪声；FULL_S6000、NFT最终256更新、SELF_REFLOW最终256更新各32输出。全部96为新采样，零训练、零新增loss、零checkpoint选择。新噪声与旧512训练和36评估请求按种子及数组hash核验无重复。

同Euler50，BF16前向/FP32参数与积分，单次官方D0、GEO_C、来源C绕序恢复与free定向。原始查询和奖励定义复用；跨模型比较原始R，不使用组标准化奖励。UNKNOWN独立保留。不是新父泛化。

恢复入口：`r2/pilots/nft_training_c_frozen_v1/resume.ps1` 默认只读status；`-Stage all`只推进已注册未完成阶段。完成阶段核验后复用；不完整attempt会阻塞，不静默重试。没有训练入口。

最终查看REPORT.md、evaluation/analysis/逐例及逐C表、evaluation/blind_gallery/index.html、audit/和completion.json。图审是实际模型匿名AI_PROXY，不能替代人类评估或几何认证。原权重和旧结果只读保留。
