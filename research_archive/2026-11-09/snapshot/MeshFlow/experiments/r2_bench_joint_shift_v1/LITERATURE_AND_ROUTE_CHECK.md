# 本轮检索与选择依据

检索日期：2026-10-22。只采用作者论文与官方实现作为技术来源。以下是方法启发，不是Bench生成质量保证，也不是严格同任务baseline。

|来源|核对内容|本轮选择|
|---|---|---|
|[PDR, ICLR2022, 附录B.3](https://arxiv.org/html/2112.03530v4#A2.SS3) / [官方实现](https://github.com/ZhaoyangLyu/Point_Diffusion_Refinement)|对不完整和完整点云添加同一个随机三维平移；同做缩放。作者也说明增强可能降低测试性能。其任务和评价是点云重建，不是保留真实三角面的开放生成。|借用共同变换训练对的做法；不借用其预算、模型、GT评分或具体最优幅度。|
|[Designing a Conditional Prior Distribution for Flow-Based Generative Models, TMLR2025](https://openreview.net/forum?id=Teh9Bq4giF) / [论文](https://arxiv.org/abs/2502.09611)|用条件组织初始分布，以缩短到目标条件模态的路径；主要实验不是本任务。|不作为这轮改动，避免与已有CREF起点改变重复且混入第二变量。|
|[Simba, AAAI2026](https://arxiv.org/abs/2511.16161)|从点级变换回归改为变换分布学习，并使用级联细化。|更大表示/模型变化，不能当作简单开关塞入当前配对实验。|

与现存路线的实质区别：

- `experiments/r2_phantom_feature_cpu_v1/REPORT.md` §5.1：共同平移曾建议，未启动1500步；旧幅度0.15L/0.08L并未冻结。现在仅选择每轴0.04L一个新范围，不扫描。
- `experiments/r2_c_reference_frame_v1/REPORT.md`：Chair CREF已实跑200步，C参考坐标和世界Gaussian起点同时改变；用途有局部收益、I/FF与接口有代价。当前保留世界坐标接口和标准Gaussian，只改变训练(C,Y)分布。
- `experiments/r2_bench_state_schedule_v1/REPORT.md`：PERTURBED_STATE扰动free插值态但C/Y终点不动；ASYNC改变SEAM时序。两者已运行，不能叫未试。
- `experiments/r2_bench_early_fm_v1/REPORT.md`：低t加权1500步未扩大采用范围；本轮FM权重不改。
- `experiments/r2_bench_clean_multipart_seam_v1/REPORT.md`：新布局头改进召回并未解决DEV主体连接；本轮使用同一缓存布局隔离生成器数据增强。
- `experiments/r2_bench_relation_training_v1/REPORT.md`：本轮主数值/视觉对照。同26父91条件，完全相同初始化/步数/已登记原随机流；上一轮增加10条件但只有1新父，不能说大规模扩数据无效。

待检验推断：共同随机平移可以减弱绝对原点/高度与完整目标布局的固定联系，使C的位置更有信息量。但不增加独立款式，不保证解决未覆盖拓扑、薄构件或布局错误；本实验评价仍在原位，不能由改善反推已经证明严格平移等变。

不新增软件依赖，不下载模型或数据，不运行论文代码。研究文本含全部候选及不采用理由，避免只保留成功叙事。
