# R2_WINDOW_POLICY_RL_V1 执行合同与恢复入口

本轮从冻结 FULL_S6000 出发，实际标定通过后才允许执行用户选择的 64 个训练批次，每批同一 C 的八条新轨迹。附件提供的是微实验方案，没有实测结果；本轮没有把建议当作已有正结果。

## 固定范围

- 八条既有奖励训练 C，每条八份全新配对初始噪声；W=[10,30)，Euler50，实际每转移 σ=0、0.005、0.01、0.02。合计256 RAW、256次官方D0、256 POST。
- C 在51个状态逐位不动；σ只加在全部有效free坐标。窗口外物理旁路LoRA；不把 no_grad 当作关闭参数依赖。
- 标定门、缺失值政策和主体图审定义分别在 registration.json、gate_contract.json；门是本轮预登记工程选择，不是论文保证。
- 通过者中只选最小σ；无候选通过则停止，不扩大窗口、噪声范围或重抽数据。
- 训练最多64次优化器更新，共512条轨迹；整批奖励相同跳过AdamW（包括动量），不补齐更新。八条C循环等曝光。
- 真实 detached 窗口状态/动作重算联合Gaussian logprob，按free维和20步求和；RLOO只按8条轨迹平均；一次fresh batch最多一步。
- LoRA rank4/alpha4，fresh AdamW lr=1e-5、betas=(0.9,0.999)、eps=1e-8、weight_decay=0，梯度范数上限1；不加参考MSE、KL penalty或其他奖励项。
- 每批160次训练重算/反向+160次更新后条件KL诊断；首批额外2次真实状态零LoRA检查。所有前向独立记账。
- 主奖励仅为原 POST 合法C/F共边数>0。采用、共享长度、主体、CF/FF/I独立汇报，不用二元奖励替代几何可用性。
- 若训练执行，终评固定八个TRAIN父和三个DEV父，每父四份独立噪声；三模型FULL_DET/FULL_STOCH/RL_STOCH共132 RAW/POST。RL_STOCH对FULL_STOCH为主要配对比较。不挑checkpoint。

## 恢复

在PowerShell运行 `./experiments/r2_window_policy_rl_v1/resume.ps1 -Phase <阶段>`。
阶段顺序为 prepare、calibration-sample、calibration-d0、calibration-cpu、calibration-gallery、calibration-analyze。
真实模型盲图审须填入 calibration/blind_gallery/reviews_actual.json；自动模板UNKNOWN不是已审记录。缺审只能产生PENDING，不能PASS。
只有 analysis/decode_gate.json 为PASS且 training_config.json已按选定σ锁定，才运行 train；之后依次 final-sample、final-d0、final-cpu、final-gallery。

已完成阶段先核验缓存，不新增模型调用。未完成/失败attempt拒绝自动重跑；如发生工程失败须先记清已消耗调用、定位并记录恢复边界。不会覆盖旧模型、旧结果或评分定义。

原模型、D0、条件与查询只读。完成后停止，不自动切默认、commit或push。
