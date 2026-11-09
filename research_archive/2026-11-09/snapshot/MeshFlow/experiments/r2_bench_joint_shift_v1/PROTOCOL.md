# R2_BENCH_JOINT_SHIFT_V1 — 冻结的共同平移训练检验

2026-10-22。用户授权：本轮仍未解决整体质量时，检索改进方法并再派发一次实验。当前 RELATION_V1_S1500 的 DEV 仍无 ≥30% bulk 可达，最终 FF 29.044→29.457%，实际图审仍见 C/灰主体错层。因此新增一条1500更新臂，不自动扩方向、扫幅度或延长。

## 问题、文献与历史边界

检验假说：训练几何始终围绕统一原点，会让条件生成继续依赖绝对位置先验。对完整 C/Y 同时做温和随机平移，保持 Gaussian 起点不动，可能促使生成器更多读取 C 的位置来组织主体。**这是假说，不是已确定的失败根因；既有训练座面高度本身有变化，当前坏例也不全在高度范围外。**

PDR（ICLR2022）附录B.3明确给 partial/complete 同加随机三维平移，且报告增强可能损害测试性能。CPD（TMLR2025）设计条件先验、Simba（AAAI2026）学习几何变换分布也相关，但本轮不复制它们的模型或宣称论文保证本任务有效。详见 LITERATURE_AND_ROUTE_CHECK.md。

最接近的本地路线：Chair CREF 同时改变参考坐标与 world Gaussian 起点，出现用途收益及 CF/FF/I 代价；Bench PERTURBED_STATE 只扰动 free 插值态并纠偏回原 Y；它们与共同移动 C/Y 终点不同。`r2_phantom_feature_cpu_v1` 提过平移但未训练，旧附件的0.08L/0.15L未锁定。新幅度0.04L为一次保守工程选择，不称已有最优值。早期FM加权、异步SEAM和占用头改进已运行，本轮不叠加。

## 唯一训练变化

直接复制 RELATION_V1 的26 TRAIN父、91条件及1500×8实例日程，所有父/C、scale、noise/epsilon/time/permutation seeds保持相同。仍每更新4个不同父各两实例：前半 identity；后半原 XYZ U[.92,1.08] 缩放后，再加共同平移。

后半实例取三轴独立 `tau_j/L ~ U[-0.04,0.04]`；L是项目固定模型单位尺度，绝不使用输出 bbox 或逐父归一。单轴最大4%L，向量范数最多√3×4%L≈6.93%L；不能混称总位移上限4%L。前半 tau=0。独立固定平移 RNG 不扰动既有随机流。每实例完整记录 tau/seed。

模型单位下：`M'=S*M+tau`，`C'=S*C+tau`，`Y'=S*Y+tau`；源面/角点/顶点/owner身份不变。预 OT 数组采用一半单位，因此 `pre_ot_full'=full_target'/2`，不能给两种单位加同幅 tau。所有依赖坐标的别名字段须同步，来源标签/布局仅训练监督。没有移动C而保持Y不动，没有事后重新中心化消除增强。

原始 z 和 epsilon 不移动，原时间分布不改。每实例重新执行原生产受限欧氏 OT（不是平方 OT，不假设旧排列平移不变），之后原 HYBRID0.25。自由变量 `x_t=(1-t)z_H+t*y_OT'`、`u=y_OT'-z_H`；固定 C/seam 端点来自 C'，固定角点目标零且不入分母。owner-only SEAM 和闭合/无owner普通bulk路径不改。

## 相同训练预算与超参数

与 RELATION 相同 COMMON12000 live/fresh AdamW 起点：`r2_bench_adaptation_strength_v1/training/A_COMMON_LR1E5/checkpoint_step0.pt`，SHA256 `75d54c15f3ff2ee1328dddca1269352db48f5ecbca4d459eaff8c603c2297398`。不从已完成 RELATION 终点继续。

Native/T1+role+Geo+同步SEAM，原FM、batch8/micro1、AdamW lr1e-5、betas(.9,.95)、eps1e-8、wd0、clip1。BF16前向，FP32参数/积分，SDPA math、TF32off、strict deterministic、CUBLAS=:4096:8。

固定1500更新（local1500/global13500），12000前向/12000反向；0额外神经探针、0新参数、0占用头训练。每步loss/分母/bulk/seam-tip误差、增强与曝光、梯度、资源和调用账本；预定1500终点，保留0/500/1500与安全滚动点，不挑best。损失下降不等于收敛或生成改善。

## 固定评价与决策

同48个原位置请求、C/N、缓存C-only OLD_S15000布局、噪声、查询点和来源角序。不平移评价输入，不新抽噪声。新增48 Euler50 RAW（2400生成前向），全部落盘后48单次官方Bench D0；C/seam恢复→GEO_C/来源定向→BASE_POST→HINGE_OPP015→ALPHA→HOLE→CLEAN_FINAL_POST。

主对照 RELATION_V1_S1500 与新臂同池同预算，次对照 USER_V2_S1500仅历史定位。旧96输出只读复用，合计432评分行。保存原完整字段（当前collector识别177数值字段，其余身份/NA字段照留），不更换评分器/阈值/分母。分别报告 RAW、BASE、FINAL、历史TRAIN16、DEV32、部位、每父和同噪声配对；I的LOW_SUPPORT单列，不靠删面获得的低冲突冒充成品质量。

144页三模型同视角/同位置截面须实际AI_PROXY图审，明确非盲与不确定。全部48请求不在本26父适配池；历史TRAIN4父在COMMON见过，DEV8父COMMON未见，但官方预训练成员身份未审计，且该面板已多次开发使用。

主要判断：多个DEV条件的C采用/整体形状改善、接续变化与FF/I/小角/删面/碎片代价一起看。未改善则关闭本固定幅度与预算，不宣称所有增强无效；有局部改善也不自动晋级。当前实验只测原位开发输入的迁移，不能单独证明严格平移等变；本轮不增加平移评价面板。

## CPU门禁、资源与恢复

冻结前核对91条件的共同变换/坐标单位/源身份/分区/几何不变量与典型owner、closed、no-owner路径的真实OT/FM几何构造，不调用神经网络。复制日程逐字段校验，tau=0输入应与原版逐位一致。

派发前测量实际GPU/CPU/RAM。参考刚完成同配置训练设备约5–5.4GiB含后台、allocator reserved3.14GiB（非完整进程峰）；采用4500MiB保守进程规划、1.2倍系数和≥2GiB额外空闲。仅一条授权训练臂，单GPU worker；首批正式更新测峰，不增加探针。CPU分析可独立并行。采样沿用单写者入口，不冒险共享布局缓存；CPU RAW评分2、POST评分4、渲染4，须按实际资源上限调整纯CPU并发，不改科学设置。

无自动科学重试。仅安全完整游标可精确继续；OOM、部分更新或模型错误保留日志并停止派发。完整训练/评价/实际图审/报告后停止，不自动再开实验、不切默认、不commit/push。
