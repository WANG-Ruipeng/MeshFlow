> 编写过程草稿，保留占位和过程记录；最终结论以实验根目录 REPORT.md 为准。

# R2_FULL_CONTOUR_READOUT_V1：固定6000步配对训练报告

2026-10-14。本轮实际完成一个候选臂的6000次更新及固定0/1000/3000/6000曲线。主比较为 CONTOUR_S6000 对 FULL_S6000；不选择中途最好结果。

{{FINAL_CONCLUSION}}

## 1. 终点回答

{{FINAL_REVIEW_TABLE}}

客观连接状态完全一致：原有8份有合法C/F共边的输出全部保留，没有新增从零到非零的输出。总共边从62增至64，两条增量都在85b169的已有连接样例中。总合法长度46.26627→46.56211模型单位，即5.02450→5.05663 L，增加约0.6394%；不能解释成更多输入接好了。

| 终点24请求等权均值 | FULL_S6000 | CONTOUR_S6000 | 候选−基线 |
| --- | ---: | ---: | ---: |
| RAW轮廓→有限F边均距 | 3.28205%L | 3.34608%L | +0.06403%L |
| RAW轮廓→有限F表面均距 | 2.85392%L | 2.91351%L | +0.05959%L |
| POST CF受影响free面 | 4.45378% | 4.75748% | +0.30371 pp |
| POST FF受影响free面 | 19.97185% | 19.72589% | −0.24596 pp |
| POST I：C冲突覆盖 | 2.19320% | 2.34578% | +0.15259 pp |
| POST保留free面总数 | 3521 | 3497 | −24 |
| POST删除free面总数 | 907 | 931 | +24 |

RAW的CF/FF/I分别为5.46308/86.82024/2.47396%→5.98456/87.33070/2.75675%，也没有整体改善。RAW边均距在13/24请求降低、11/24升高，但按24请求等权的平均距离仍略升。完整逐输出及逐父结果均保留，不只提供总体均值。

CF与FF使用原评分器的 `CF_illegal_affected_free_fraction` 和 `FF_illegal_affected_free_fraction`，分母是各自阶段实际free面数；I为 `ConflictCov_C`，分母为冻结C查询权重，不是全物体相交率。POST改变几何和面数，不能把RAW/POST混池。距离来自C真实轮廓上的确定性长度加权查询，对有限表面/有限边求距；L固定为9.208138264587333。整个C轮廓是观察域，并不意味着每段都应封闭。合法边沿用原过滤定义；一条合法边也不保证整圈接好、水密或整体无交。

## 2. 完整曲线及逐父得失

{{CURVE_TABLE}}

S1000没有新增采用或联合正例，2d/n0的下方支撑还曾缺失。S3000相对同预算FULL多保留73b369/n1一个联合正例；它在共同起点已成功，因此属于保留/恢复原能力，不是新输入模式的突破。这个中途结果不会替换预定S6000终点。

{{FINAL_PARENT_REVIEW}}

每父下表均为四噪声等权；Δ为候选减FULL，边数为四份输出之和。

| 父/分组 | POST合法边 | ΔCF(pp) | ΔFF(pp) | ΔI(pp) | ΔRAW边均距(%L) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3f7417 TRAIN | 37→37 | +0.01173 | −0.28058 | 0 | −0.01855 |
| 73b369 TRAIN | 0→0 | −0.60624 | −0.45688 | −0.12207 | +0.07855 |
| 85b169 TRAIN | 21→23 | +0.33764 | −0.53330 | +0.01221 | −0.00492 |
| 2d08a6 DEV | 4→4 | −0.13127 | +1.46325 | −0.07324 | +0.10475 |
| 608b46 DEV | 0→0 | −0.01191 | +1.50322 | −0.10986 | +0.01165 |
| 610b16 DEV | 0→0 | +2.22229 | −3.17145 | +1.20850 | +0.21271 |

TRAIN的POST CF/FF/I平均差为−0.08562/−0.42358/−0.03662 pp，DEV为+0.69304/−0.06833/+0.34180 pp。DEV的RAW轮廓边距4.94890→5.05860%L，TRAIN为1.61520→1.63356%L。既有的TRAIN/DEV差距没有被新模块压小。

两条边增量分别为85/n0的5→6、85/n3的8→9。n0的CF/I保持零、FF降低4.57516pp；n3则CF从0升至1.32450%、I从0升至0.04883%，FF降低1.98675pp。局部增量真实存在，但其代价不相同，也没有扩大接口请求覆盖。

必须披露的尾部是610/n0：POST CF 1.15607→8.87097%、I 2.68555→7.32422%；FF却14.45087→4.83871%。同时保留free面173→124，多删除49面（53→102），仍无合法接口。因此不能以FF下降单独称整体改善，也不能仅凭这份计数断言删除是全部原因。608/n2及2d/n1的FF分别升6.39350及5.13962pp；608/n0的FF下降9.28359pp。全部24请求的正负尾部排名已保存。

独立数值复算：[NOTES.md](audits/final_numeric_readout/NOTES.md)、[逐输出配对](audits/final_numeric_readout/endpoint_per_output_pairs.csv)、[逐父表](audits/final_numeric_readout/endpoint_per_parent_paired_metrics.csv)、[全部请求尾部](audits/final_numeric_readout/all_requests_tail_ranks.csv)。

## 3. 新通路确实训练了，但没有明显额外误差收益

两臂都使用48,000次实际曝光，47父各1021或1022次。来源合法接口标量坐标共2,224,188，其余81,004,482，总free为83,228,670；接口约2.67%。两臂窗口和五个t区间的样本/坐标分母全部相同。这个比例本身不证明应增加局部loss权重。

| 更新窗口 | 接口速度MSE FULL→CONTOUR | 其余速度MSE FULL→CONTOUR |
| --- | --- | --- |
| 1—1000 | 0.13737210→0.13741606 | 0.11249425→0.11249838 |
| 1001—3000 | 0.13793963→0.13791858 | 0.11305360→0.11305423 |
| 3001—6000 | 0.13196045→0.13192787 | 0.11018763→0.11019075 |

后三千步接口MSE相对基线只低0.02469%；总free MSE为0.11076932→0.11077149。t分为[0,.2)、[.2,.4)、[.4,.6)、[.6,.8)、[.8,1]：后三千步接口MSE依次为0.660269→0.660105、0.218797→0.218725、0.045734→0.045743、0.017644→0.017640、0.011850→0.011823，没有某个时间域出现明显改善。端点缩放接口误差0.07547557→0.07545870也只略变。

这些是同一曝光流中变化着的普通训练输入，MSE按sum(SSE)/sum(标量坐标数)计算；不是固定probe，也不是rollout。端点缩放误差仅是同次前向的(1−t)²速度SSE，不算第二次预测。不能据此断言唯一根因、已经收敛或所有读取结构都无效。

首批8次正式前向的零残差、hidden不变及FM逐项匹配全部通过，FM均值0.18918075459077954。首步旧/新梯度范数为0.4408757091/0.0032997436，只有零出口首先获得非零梯度；第二步内部QKV/MLP已有非零梯度。全部6000步新增组梯度范数非零；实际全模型裁剪发生85次，最小系数0.08789938。说明支路进入了优化，不能将本次弱结果归为“模块根本没训到”，也不能凭有梯度便声称学到了有效轮廓机制。

训练表：[完整诊断指针](training_analysis/latest.json)、[窗口配对](audits/final_numeric_readout/training_fixed_window_diagnostics_paired.csv)、[五时间段配对](audits/final_numeric_readout/training_fixed_window_time_bins_paired.csv)。

## 4. 实际实现及配对合同

候选从原EXP67 global6000的live非EMA权重、匹配AdamW与RNG开始，沿用世界坐标Native/T1＋role＋Geo、原free-only FM、free-only OT后HYBRID0.25。复用FULL的47条合格TRAIN SEAT_FULL（K≥2）、逐字节一致的48,000条计划及6,000批只读OT缓存；没有新OT、新噪声、DEV混入或来源修补。仍保留31条有既有来源Y内部FF风险的条件（其中3条还含未决项），不因本轮结果挑数据。

有效batch8、micro1，4 clean＋4 x-width U(.9,1.1)，lr1e−5，AdamW(.9,.95)、eps1e−8、wd0，统一clip1。先匹配194个旧参数的optimizer身份，再附加8个新参数；旧状态时钟6000+k、新组从空状态真实增长至k。恢复不拼接EMA/非EMA，不重置旧动量。新增模块的初始化不消耗原数据随机流。

模块仅接在第6个完整block后：128维、4头；Q/K/V为768→128无bias，无affine LayerNorm eps1e−5。每C面的memory是本次forward中坐标＋role＋Geo后的三角点H0的FP32均值，查询和memory均不detach。对全部有效C面做QK softmax，message为Σ_j a_ij[V(s_j)+φ(r_ij)]，保留i/j双索引；无几何logit bias、top-k、吸附或新loss。

17维为有限表面相对向量3+距离1、当前C面所属真实开放边的相对向量3+距离1、无向切线dyad6、开放边存在1、边数/3一维、t一维。边按实际FULL来源顶点连接incidence=1确定，内部对角线排除；原ID数值不进网络，平局对称处理不依赖排列或ID。无边界时边界字段有限零，仍保留表面memory。2d/610八请求的旧HALF描述没有用于拓扑，实际FULL sidecar独立冻结。

φ为17→128→SiLU→128，带bias；出口128→768无bias且精确零初始化，无额外零门。新增**412,032参数**。分支显式关闭autocast、全FP32运算，先FP32相加再cast回原hidden dtype，只写有效free。主干BF16前向、FP32参数/积分、禁TF32、SDPA math、严格确定性算法。没有硬共享顶点或拓扑连接保证。历史G_closest已有几何pair-value，本轮不宣称首次引入相对几何；整个模块单臂比较也不能独立归因轮廓、层位或容量。

实现细节：[README](../../r2/pilots/full_contour_readout_v1/README.md)、[模型](../../r2/pilots/full_contour_readout_v1/model.py)、[17维几何](../../r2/pilots/full_contour_readout_v1/relations.py)、[拓扑](../../r2/pilots/full_contour_readout_v1/topology.py)。13项CPU模块测试和192项拓扑检查通过；检查没有增加完整模型GPU探针。

## 5. 评价、实际图审及适用范围

六父FULL，每父4份既有固定噪声；TRAIN为73/3f/85，DEV为2d/610/608。四个固定时点，全部Euler50、同一单次官方D0＋GEO_C＋恢复来源C角序/固定C的free定向。一个时点全部RAW保存后才进入D0。没有RBF、B、CREF、蒸馏、多步去噪或额外采样。

新增72次rollout=3,600次生成器前向，新增72次D0；候选S0的24RAW/24POST为明确缓存引用，不称重新采样复现。旧FULL四时点96RAW/96POST全部复用。新增的144条实际AI_PROXY图审来自候选三个新时点；旧FULL192条图审继承，候选S0另48条标签映射继承，不能把384条逻辑标签算成384次新图审。

每个新时点实际查看24张两模型depth页和24张有限剖面页，共144页；RAW/POST分开填写8项YES/NO/UNCERTAIN观察。图审为非盲AI_PROXY，不是人类审稿评分；旧标签及观察口径保留，不按数值调采用标签。用途采用、漂浮/错层、主体完整、折层和可见接触分别记录，不能以视觉贴近冒充索引共边。

实际评阅的配对页在各自时点、同C的两模型/四噪声/RAWPOST之间共享范围；各时点可有不同范围。最终全8列图集为另一个统一展示集合，不能说它是先前实际查看的同一图页。CSV保存实际图页与剖面路径和SHA，归档保存原图集manifest及关联索引。

[实际图审CSV](evaluation/review/AI_PROXY_reviews.csv)、[全曲线depth图集](evaluation/gallery/index.html)、[全曲线剖面](evaluation/sections_gallery/index.html)、[终点配对图集](evaluation/gallery_FULL_S6000_CONTOUR_S6000/index.html)、[完整配对结果指针](evaluation/comparison/latest.json)。这批DEV已经反复观察；它们与TRAIN分别汇总，但不能称新的独立泛化测试。没有统计显著性声明，没有用唯一来源GT款式判开放生成成败。

## 6. 执行账本、资源和工程记录

实际成功6000次更新、48,000次训练前向、48,000次反向；无失败或挂起模型调用、无超预算重放。训练段累计墙钟约157.58分钟，其中日志update_seconds合计约145.78分钟；加载/保存、评价、图审与用户GPU共享的开销范围另列，不能将分段时间当隔离性能基准。

RTX5080总16303MiB、24逻辑CPU。遵守用户约6GiB训练可用空间和动态余量，GPU一次一个任务；每个实际CPU评分阶段使用2个worker、每worker 2线程，与独立GPU阶段重叠；控制器允许最多两个CPU流水线，本轮实际评分流水线之间没有重叠。训练PyTorch reserved峰值3321888768 bytes（约3.094GiB），不是进程总显存；WDDM进程VRAM不可得记NA，device峰值含用户图形任务，未冒充模型占用。全过程未停止用户其他进程。

local1114因可用显存低于2048MiB安全存盘暂停，非OOM；核对生成器/AdamW/RNG及账本后从1115继续，无重复更新。控制器在local3000前发生一次替换status.json期间的短暂读取失败；已派发训练子进程继续到3000并exit0，控制器安全排空后exit1。仅修控制器读取竞态，保留原错误、源码和diff，七项CPU fixture通过、85份冻结科学源码不变；核对3000状态后从3001继续。最终控制器exit0。

正式调用前的集成变量/观察列表/S0缓存路由修正，冻结后但首次评价前恢复不在注册内的CPU缓存编辑，汇总键顺序修正及审计器schema修正均有独立收据，未修改科学定义。旧FAIL进度审计不删除；它的错误是合法rollout状态/尚未完成表的schema处理，不能作为最终审计结果。打包页绑定修正只完善实际图审来源身份。原资产、旧评分器与旧结果只读。

[资源恢复记录](audits/resource_resume_1114.json)、[控制器恢复](audits/controller_resume_3000.json)、[竞态修正收据](audits/orchestrator_status_fix/0bd1874d2b432ae6/receipt.json)、[零初始化](audits/zero_initialization.json)、[最终执行审计指针](audits/execution_final_latest.json)、[控制器状态](orchestration/controller_state.json)。最终审计绑定本报告、注册、模型/优化器时钟、源hash、逐次调用、OT缓存和图审，实际状态以其收据为准。

## 7. 交付、恢复与研究决定

完整恢复点位于checkpoints，主终点[CONTOUR_step6000.pt](checkpoints/CONTOUR_step6000.pt)；推理导出为[inference/CONTOUR_S6000_generator.pt](inference/CONTOUR_S6000_generator.pt)。保留0/1000/3000/6000曲线及1114安全恢复点；不自动清理本次证据。

| 身份 | SHA256 |
| --- | --- |
| 注册 | d9c9aae17bc6df9d108fcbdaf0b072f1aefa627ddfef069520a80f205c13c41e |
| 原曝光计划 | 52a48be0987c11511dea149829935e6af9baceb4ac70317052653b95c557f4c9 |
| 终点完整checkpoint | ccdc5b67ad8ee5c2c2e455f808d79748ca8f6276e78d2fe82f3b1abfc073f29e |
| 终点推理权重 | 0be212ab5d58011621f59fe7428e722afad872dd4e23115ce21f3a6bcd67185e |

只读状态入口：`r2/pilots/full_contour_readout_v1/resume.ps1 -Stage status`。`-Stage all`用于按冻结身份恢复未完成阶段，完整科学任务已完成后不应追加训练/采样。图审、汇总、报告和终态审计各自有回执，不将控制器的“等待图审报告”状态冒称全部交付完成。

小型审阅包收据见[archive/latest.json](archive/latest.json)，包含逐输出/逐父CSV、实际评分源码、code.diff、源hash、完整48张全曲线压缩预览及真实图审；大权重、RAW/POST数组和OT缓存留在本地，以路径/SHA登记。压缩图不是新增图审，无法代替本地原分辨率图。包内完整结果保留负向指标与NA。

{{NEXT_DECISION}}

本轮不切换默认模型，不新增实验，不commit或push。交付完成后停止。