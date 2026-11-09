## 2026-10-15 NFT负结果收口与中间权重清理

当前NFT配置停止扩大。训练后终点评价面板36个配对请求三模型明确联合均9/36；八条奖励TRAIN C×四份新噪声的冻结检查，FULL/NFT/SELF严格联合11/12/11，RAW采用均26/32。唯一新增请求也由无奖励SELF实现，NFT相对SELF的净差是保住另一旧正例。共同已知28对/7C的NFT原始奖励相对FULL -0.003113；第8条C奖励UNKNOWN，不填零。FF均值局部改善与I/尾部代价同时记录，不否定全部RL，也不认为改奖励已被证明是首要答案。

用户授权删除30个NFT/SELF round00..14中间.pt，逻辑空间287,949,354字节（274.61 MiB）；保留两份round15、FULL_S6000、官方模型及全部RAW/POST/图审/评分/源码/日志/旧报告。历史逐轮权重重放已退役；不自动补回、不续训。实际磁盘变化、完整路径与SHA见[清理回执](experiments/r2_nft_negative_result_closeout_20261015/cleanup_receipt.json)。[负结果报告](experiments/r2_nft_negative_result_closeout_20261015/NEGATIVE_RESULT_REPORT.md)；研究记忆新增R29，来源索引83+2=85份。FULL仍为方法比较参照，维护CLI默认不变；零新模型调用，无commit/push。

## 2026-10-14 Partial-edge binding CPU pilot completed

R2_PARTIAL_EDGE_BINDING_V1 executed all 24 FULL_S6000 outputs x A/B x 1/2/3.5%L = 144 cells, zero model/GPU calls. Original POST representatives and actual GEO_C clusters were frozen; no radius reclustering. A reassigns selected corners; B assigns the original free vertex cluster to exact C indices. All six policies preserve the original 6 joint positives and all 62 legal CF edges; zero face deletion and zero newly illegal CF pairs.

Joint screening counts A=8/8/9, B=7/8/8 of24. Original legal FF edge losses A=12/16/27, B=0/0/0. Newly illegal FF pairs A=0/1/4, B=0/4/4. B2's two conversions are TRAIN73 n0/n1, DEV remains0/12. Only A3.5 adds DEV2d n1 with tearing costs. All608/610 requests have zero eligible candidates even at3.5%; this is a candidate-coverage limit of whole-edge endpoint correspondence, NOT evidence that attempted binding destroyed them. Do not infer a physical upper bound, global post-hoc impossibility, or pointer learnability. Whole-edge one-to-one matching does not test edge-subsegment bindings.

34 paired images actually reviewed by nonblind AI_PROXY (6 contact sheets plus3 full-size cases), including all12 new joint arm/request entries. C's seat role remains visible; local cracks/full seams are not visually certified. Formal binding/scoring151.24s,2CPU workers; this excludes preparation/review. Independent144-cell audit and9880 cached scalar comparisons PASS,611 protected files unchanged. One before-decode import failure and hash amendments retained. Current default unchanged; no training/commit/push. Report: [REPORT.md](experiments/r2_partial_edge_binding_v1/REPORT.md). Resume: r2/tools/partial_edge_binding_v1/resume.py (completed runs verify only).

## 2026-10-14 全历史研究路线查重记忆

按用户要求回看本地全部主要实验阶段，建立[r2研究路线记忆](r2/docs/research_route_memory.md)、[机制登记JSON](r2/docs/research_route_registry.json)和[83份主报告及SHA256索引](r2/docs/research_history_sources.md)。按机制合并28条已运行但未形成稳定升级的候选、2条训练前门禁阻塞；另列128父数据门槛阻塞和应保留的局部正结果，不把报告数当失败方法数。

特别记录：FC/FF与时间门控、六loss筛查、JEdge/UCorr、Pair伙伴闭合、边界标记、MOMENT、READOUT及FP32复验、B早晚日程、CAPPED复现/梯度预算/半幅、课程、求解器、锚点去噪、蒸馏及多轮去噪、反馈/BLEND，以及最新FULL轮廓读取。新实验先查最接近的旧路线、实际失败边界和本次实质变化，避免改名重跑；无需因此先做一轮大审计。

FULL6000确有采用和轮廓改善，不能称FM完全学不动；同预算CONTOUR终点仍16/24采用、8/24合法接口、6/24联合正例，新增成功0，DEV联合0/12。Native/Geo/HYBRID/R2、USER构造、单次D0/GEO_C及来源方向修正的有限正证据保留。历史GT唯一款式评价、旧C20/C40回归否决规则不作为当前USER主任务的默认合同。

本轮仅报告/源码/维护记录阅读与文档整理；没有重评网格、重新图审、模型加载、训练、前向、生成或权重清理，没有commit/push或切默认。独立目录experiments/research_history_audit_20261014保存来源清单、读取范围、文档变更和完成核验；原83份报告hash保持。读取范围日志不冒充全量原始数组复算。
## 2026-10-14 FULL座面6000步训练曲线完成

R2_FULL_SEAT_LEARNING_CURVE_V1已实际完成。从EXP67 step6000的live生成器、匹配AdamW及RNG恢复，47个合格TRAIN SEAT_FULL父/条件，原世界坐标Native/T1+role+Geo、free-only FM、OT后HYBRID0.25，新增6000步至global12000。三条C不连贯输入在冻结前排除，其余31条来源Y-only风险保留记录；没有训练DEV、修原几何或按生成失败删条件。48000曝光，每父1021/1022次；零额外诊断前向或loss。

预定0/1000/3000/6000四点完整评价六父×四固定噪声，6000为主终点，不挑best。96逻辑RAW/96POST，其中88新增rollout=4400生成前向、88单次官方D0；另两冻结参照48RAW/48POST。全288原质量与边界单元、192主图审完成，AI_PROXY非盲，参照图审另存。RAW采用YES为12/12/13/16；最终新增4请求来自3父，无原YES丢失。RAW有限边长度加权距离4.762→3.282%L，18/24下降；POST合法接口输出7/6/6/8，合法边53/40/53/62，总合法边长47.955→46.266模型单位。不能将1000/3000全部共享边41/55冒称合法边。

最终采用改善与新增合法接口不重合。新增接口是2d n0/n2（采用UNCERTAIN），73 n1丢失；610虽新增两份采用仍无合法接口，608全部采用但仍无合法接口。POST CF/FF/I宏均增加1.217/3.864/0.765个百分点；DEV FF17.939→25.855%，不能认定代价可控或直接升级默认。接口训练vMSE在固定日志窗口间小幅下降，属于变化输入上的训练诊断，不是固定探针或自由生成误差。结论是原FM在FULL任务上仍有可兑现改善，但稳定接续与几何代价未同时解决；不自动延长训练、加loss、改模块或放宽焊接。

独立最终执行审计PASS；保留2d/610共8请求继承HALF描述字段的元数据sidecar，实际FULL数组、K/N、查询及评分身份核验正确，原冻结记录不重写。早期训练与0点采样阶段包络重叠69.24秒、逐调用区间交集47.14秒；这不是CUDA kernel并行证据。三训练段总墙钟153.37分钟，包含加载保存。GPU科学任务全部结束。

报告 experiments/r2_full_seat_learning_curve_v1/REPORT.md；图集evaluation/gallery/index.html与sections_gallery/index.html；最终checkpoint FULL_SEAT_step6000.pt（5f41df3d…af33）、推理FULL_S6000_generator.pt（86b38d35…c5cc），1000/3000恢复点保留。实际逐例CSV、曲线、图审及源码在同实验目录；小审阅包9,827,092字节，SHA256 d3cd93bedd7c652455ee60a7dba47015c5297c728847773ed1d0a5f261489a40；archive/latest.json为COMPLETE，独立schema_v2归档核验PASS（1143成员SHA、192主图审、118源码、26原评分定义、48预览）。原验证器对可选N/K空列的解析错误与修正另存；包未修改。三个冗余原日志/旧审计账本留本地，包中列原路径/hash，全部CSV与失败案例保留。恢复入口r2/pilots/full_seat_learning_curve_v1/resume.ps1默认只读；控制器阶段状态保留图审前的COMPLETE_PENDING_ACTUAL_IMAGE_REVIEW_AND_REPORT，整轮封存状态另见completion.json。未切默认、commit或push；固定实验结束，不自动启动下一轮。

## 2026-10-14 FULL训练前清理旧权重

用户要求在训练前清空不再用的旧检查点。已按逐文件白名单删除66文件，逻辑92.065 GiB、实际可用空间增加约87.7 GiB；23个保留大权重哈希复核。EXP67 step6000匹配恢复点、SMALL/CREF终点、官方Chair/D0、维护START/R2等保留。旧anchor/student/ABC、中间点、重复latest及退休续训对照按清单删除；对应旧分支不再可恢复。RAW、原数据、评分、报告和日志未动。清单与账本：experiments/r2_full_seat_learning_curve_v1/cleanup/。原报告只读；不回写旧结论。后续继续已授权的FULL座面6000步训练，无自动commit/push。
## 2026-10-13 FULL边界离线归因完成

R2_FULL_BOUNDARY_ATTRIBUTION_V1已完成：固定上轮11请求（3f n0–n3、73 n2、2d n2/n3、85 n0–n3）×SMALL/CREF×RAW/POST=44既有几何；源边界等弧段中点与单独顶点共27112查询。两个CPU worker，零训练/模型前向/新噪声/生成/几何修改/原质量重评分。独立逐点和聚合审计PASS，593原文件hash不变；11页实际非盲AI_PROXY、44行图审完成，旧采用标签不变。

主要发现：3f n0/n1/n3 CREF的轮廓错位已存在RAW，n1/n3以切向错位为主；2d n2/n3用途改善同时仍有显著轮廓间隙，不是单纯导出焊接问题。SMALL 3f n0 POST虽16/16边界顶点精确匹配，但仍有长边中段缺口；83.9%轮廓在.005L内不等于整圈接好。73 n2是近表面/局部点匹配与冲突混合情形。85 n0/n2/n3保留流程能形成局部接续的正例，n1单列。来源法线侧别、最近面CF/FF注释及多次法线命中均不可直接认证层语义或非法点。

优先建议后续研究RAW局部边界对应，不以高度接近替代接续，不自动放宽后处理；本轮未新增loss或训练。全边界距离为描述性中点近似，另存Lipschitz采样界限，S仍NA_NO_REQUIRED_INTERFACE，不恢复SJoin。此11请求按观察结果选例，不能替代全24请求成功率或新父泛化。

报告 experiments/r2_full_boundary_attribution_v1/REPORT.md；图集gallery_v2/index.html；逐点/边/面、配对表和法线命中均在同目录。小包R2_FULL_BOUNDARY_ATTRIBUTION_V1_REVIEW_SMALL.zip为8591321字节，含44几何、全部27112逐点CSV行、实际分析/原评分源码、hash/diff及11图预览；可解压纯NumPy复算全部44单元且已验证。恢复 r2/tools/full_boundary_attribution_v1/resume.ps1 已实际验证命中缓存，不重新生成。未切默认、commit或push；固定任务停止。

## 2026-10-13 完整规整面片冻结模型检验完成

R2_REGULAR_PATCH_CONTROL_V1已实际完成。六条既有SEAT_FULL，K=20/14/12/14/2/2；610b16与608b46为原生两三角矩形，3f7417后缘有凹口需单列。四份固定原噪声n0..3，冻结SMALL_C200/CREF_C200；32新RAW、1600生成前向、32单次官方D0，另严格复用16RAW/16POST；48RAW+48POST、96逐输出分数和实际AI_PROXY已交付。零训练，无新高斯、RBF、loss、蒸馏或多次D0。

最终V2实际图审：SMALL采用11/24、CREF14/24；POST合法接口输出7/24与3/24。真正改变FULL的16请求：SMALL采用5、合法接口3；CREF采用8、合法接口0。3f SMALL共边15→35，但有接口仍3/4；三个其他新FULL父均0/12合法接口，610两三角矩形未获接合。CREF的用途/布局改善仍与接续及CF/FF/I代价分离，未晋级或切换默认。HALF→FULL同时改变K/面积/边界/free/中心，不能称纯K消融；两原FULL是恒等缓存，不能记新增条件收益。

本轮发现Matplotlib分立C/F Poly3DCollection的平均深度排序会错遮蓝C。V1图及96原审阅保留；另以同几何逐像素CPU深度输出全部24页，统一实际重看，形成完整V2并逐项记录变化。显示纠正本身不算模型收益。后续图审不要仅因蓝色未显示便判未采用，优先深度正确的图并结合有限剖面；共面重层、次像素间隙和实际共边仍需独立数据。

报告 experiments/r2_regular_patch_control_v1/REPORT.md；优先看 evaluation/depth_gallery/index.html，原剖面在evaluation/gallery/index.html。恢复 python -m r2.pilots.regular_patch_control_v1.run --stage status/all；科学入口拒绝静默重跑未完成cell。全部新代码独立放r2/pilots/regular_patch_control_v1。输入/执行/数值审计通过；GPU调度进程曾重叠，但实际前向无重叠，不能宣称GPU并行加速；CPU评分2worker。无commit/push，固定任务后停止。

## 2026-10-13 CREF层结构离线诊断完成

R2_CREF_LAYER_AUDIT_V1已实际完成：预选11请求×SMALL/CREF×RAW/POST=44缓存几何，另6来源结构参照；两个CPU worker，零训练/模型前向/生成/新噪声/旧质量评分。102400逐点记录、全部有限法线命中与侧壁邻接、28张原图和44非盲AI_PROXY已交付。44旧dup逐点一致，210原文件hash不变；独立数值、聚合、报告审计PASS。

结论分情况：73b/n3 CREF支持正常厚座；3f/n2有近重合额外上皮；73b/n4、2d/n2、610b/n0混合；73b/n6薄垫/厚框/冗余解释仍不唯一。I_dup不能全部当重复面，也不能全部豁免；FF和合法接口代价保持。85b/n6有局部正例，n7接口5→0且采用仍YES。此为按历史结果选例的机制诊断，不是新泛化成功率。CREF不默认晋级，不添加dup惩罚loss。

报告 experiments/r2_cref_layer_audit_v1/REPORT.md；恢复 r2/tools/cref_layer_audit_v1/resume.py --verify；小包R2_CREF_LAYER_AUDIT_V1_REVIEW_SMALL.zip共9873148字节，含全部例/图/逐查询CSV/实际代码及精简几何，完整派生表本地保留。SHA256 0f407ffafda6f9f808a9a049eee4b36ea0379a0a324b7fc19f2eb4a0b8529c09。固定实验完成停止，无默认切换、commit/push。后续仍优先规整连贯K≥2的用户式输入；未自动构造或运行下一轮。

## 2026-10-13 后续 USER 输入优先规整、连贯的人类式局部表面

用户明确提醒：后续新构造的条件 C 尽量采用至少两个三角形组成、形状规整、连贯且真实用户可能提供的局部表面。例如两个相邻三角形组成完整四边形座面，或多个相邻三角形组成有清楚轮廓的座面／靠背表面。重点是有意义的完整局部区域，不是机械地凑 K≥2；不为满足面数而把孤立面、狭窄边条或无关部位拼在一起，也不预设所有合理输入都必须是平面矩形。

这是一项后续数据构造及主评价的默认偏好，不是宣布 K=1 不可生成或模型必须改为四边形 token。未来单三角、残缺角落等输入可作为单独登记的压力测试，不再默认以这些极简条件主导当前架构选择。现有冻结实验中的 K=1 输入、正负例、图审和评分继续完整保留；不回头删行、换父、改分母或将历史收益重新筛选。新主面板应在生成前按输入可用性确定。

此次同时审阅 CREF 反馈：它提出的是“布局更靠近 C，但可能仍生成了额外近平行表面”的待检验机制；已有 I_dup 不能区分正常薄底面与多余座层。后续若获授权核层，应比较已保存 SMALL/CREF 的 RAW/POST，保留原 I、查询和阈值，结合多层候选、局部延续和侧壁，不能只凭最近面或法向正负认证薄壁合法。7个明确收益＋2个YES→UNCERTAIN共9请求的36份缓存几何是可用诊断范围，本次尚未执行该新诊断。

本次只记录用户偏好及分析边界，不构造新数据、不训练／生成／重评分、不改loss或默认模型，不commit/push。


## 2026-10-11 来源方向恢复正式接入 denoise，已提交并推送 GitHub

用户明确授权将来源 C 角序恢复＋固定 C 的 free 定向固化进 denoise，并整理正结果上传。正式入口位于 r2/src/meshflow_control/，包版本0.2.0；新增 prepare-condition / denoise / orient。新 denoise 默认单次官方 D0 → 恢复C → GEO_C → 来源C恢复 → free定向；RBF只通过 --coordination rbf 显式启用，使用固定 --reference-length；旧 sample 仍输出RAW，旧START/native_t1入口没有静默切换。后续需要此合同的去噪使用 r2 正式入口，不再依赖旧实验tools导入。

prepare-condition 在规范化前保存用户C原角序/顶点ID及可逆排列，不需要完整GT；历史保存映射标 SAVED_MAPPING_ONLY。坐标相等不替代来源连接身份：GEO引入合并/拆分需报 CHANGED_BY_GEO_C，无来源ID报 UNKNOWN；未知属性拒绝静默丢弃。方向矛盾/非流形状态保留，不承诺水密、全局朝外或新增接缝。保留 proposal / canonical / restored_C / final 和逐阶段hash账本；不自动重试模型。

验证：115 CPU tests PASS；36仅定向＋24缓存D0/RBF链，4320旧数组逐字节一致，199旧文件hash不变；单次真实D0完整proposal与旧缓存maxabs=0，canonical18及final46数组一致；独立wheel安装验证128数组另PRE_GEO2一致，无旧工程导入。维护验证合计仅1次真实D0，0训练/生成器/新噪声/新科学评分；CPU迁移额外60定向24GEO12RBF，wheel额外2定向1GEO，GPU全链额外1定向1GEO，单元测试使用合成输入另计。GPU资源及峰值scope有记录，不把allocated/reserved说成进程显存。1次D0有readonly NumPy提示，未写输入/未重试。

公共研究综述 r2/docs/research_results.md 汇总R2、USER、单次D0、RBF、来源方向修正及学生/多轮的有限收益与代价，含30 CSV、来源hash与验证小收据；CF proper/illegal按原字段区分，完整选定面板不按输赢删行。RBF九条已有连接可一致定向，座面仍0/4、FF/I代价仍在；RBF非默认升级。完整入口文档 r2/docs/denoising.md，验证 r2/docs/denoise_validation.md。本地工程账本 experiments/r2_denoise_publish_v1/。

提交及origin/master已实核同一SHA：1c6d15b0e6abe5c7053590b7c80d292315e2b346。共68文件约1.21MB，权重/RAW/原始网格/旧完整实验和无关未跟踪代码未上传，旧报告只读；GitHub https://github.com/WANG-Ruipeng/MeshFlow/commit/1c6d15b0e6abe5c7053590b7c80d292315e2b346 。当前任务完成并停止，未自动启动后续实验。

# 当前工作区：R2 条件研究（保留历史 START 采样入口）

## 2026-10-11 来源C角序恢复与free定向导出完成

已完成R2_SOURCE_WINDING_EXPORT_V1：只读同36份D0/RBF/BLEND POST，按保存的来源排列取逆恢复C，再固定来源C整理free；保存36恢复及36最终NPZ/PLY，原坐标、无序索引面、源面ID/连接、面数及逐角追溯保持。六条件原45条双面C边均反向，规范化后28同向；恢复后36实例的168条同向C/C边全部消失。新合同恢复原输入角序，不再锁死内部规范化角序；没有改模型输入。

D0/RBF/BLEND原合法共边1/9/3条不变，方向一致共边1/3/2→1/9/3；RBF方向可达free面积宏均0.2643%→19.9459%，BLEND0.1584%→6.8409%，均达到各自无方向上限。RBF86ec n0/n1及bdd2 n0分别85.333/77.262/76.756%。这是已有连接的导出表示修正，没有新接缝；每组四座面仍无合法共边。CF/FF/I、A-VMR/PMR、距离及小角值在几何等价核验后继承，未重新评分。

每组各一份f596 n0仍有177面纯free奇偶环冲突；分量未改，不能归因于C。非流形边80/84/80保留，每组10/12涉及；没有水密/全局外向或流形认证。独立来源逆映射、XOR并查集、72 PLY、108图阶段及864VM行审计PASS；独立审计首次ASCII读取器遇二进制PLY，工程修正/失败版保留，无科学重放。2314保护文件hash不变。

执行36来源恢复/36定向，2 CPU worker；模型加载/前向/训练/生成/新噪声/GEO/完整质量评分/新图全部0。科学阶段墙钟37.469秒含末尾hash，不含初始准备核验。[报告](experiments/r2_source_winding_export_v1/REPORT.md)、[逐父噪声](experiments/r2_source_winding_export_v1/paired_per_parent_noise.csv)、[独立审计](experiments/r2_source_winding_export_v1/independent_audit/completion.json)。源码r2/tools/source_winding_export_v1，实际代码/diff/hash和小审阅包保留。仅独立导出实验，D0比较基准不变，未改默认/训练/loss、未commit/push；固定任务已完成停止。


## 2026-10-11 RBF有界对照与固定C方向审计完成

完成R2_C_RESTORE_BOUNDED_AND_ORIENTATION_V1：同12 USER_MIX缓存D0，12纯CPU非负局部BLEND、同GEO_C和原质量评分；D0/RBF质量24复用，36 POST均执行固定C绕序的free-only方向审计。零训练/模型加载/前向/新生成/新噪声；24三组两视角图、36非盲AI_PROXY和独立审计完成。1408保护文件/原权重hash保持，独立来源审计另96文件不变。

D0/RBF/BLEND严格AVMR为1.389/13.492/7.077%，FF12.019/12.550/13.410%，I3.687/4.390/3.988%，小角17.062/17.455/16.930%，有效共边输出1/3/2（各12）。BLEND压低866e n0与bdd29 n0的I尾部，但12/12边界距离均比RBF差，86ec两噪声代价增加；不替代RBF或D0。四座面仍无有效共边，PMR全部0。

方向解释需修正：6 DEV条件原C所选面的45条双面共享边原本相反，保存的逐面角点规范化使28条变同向（官方sort_vertices_and_faces逐面sorted）。5/6条件、10/12输出的固定C方向合同已矛盾；并非原始来源面本身有方向缺陷，也非RBF或GEO才制造。全冲突分量保持导致三组接口反向可达均无新增；不等于所有可达差异都是物理缝隙，也未否定局部free翻面可能。原无方向共边/相交/错层证据保留。未自动改变C合同或旧评分。

[报告](experiments/r2_c_restore_bounded_v1/REPORT.md)、[图集](experiments/r2_c_restore_bounded_v1/gallery/index.html)、[逐例](experiments/r2_c_restore_bounded_v1/decision_per_output.csv)、[六条件方向来源](experiments/r2_c_restore_bounded_v1/review/source_C_orientation_conditions.csv)。代码r2/tools/c_restore_bounded_v1；实际源码/diff/hash及小审阅包保留。优先建议澄清保持C几何是否允许重排角点，再独立验证全网格方向整理；该新合同未执行。没有默认切换、commit/push；固定任务完成停止。


## 2026-10-11 C协调恢复与有限反馈实验完成：保留局部信号，不晋级

已完成R2_C_RESTORE_COUPLING_V1：同12 USER_MIX RAW，官方D0与GEO_C固定；12 RBF离线协调恢复、24真实D0反馈查询（两更新，F_RAW恒定），零训练/新生成/新噪声。24新POST评分、D0/D3缓存、24两视角四组图和48非盲AI_PROXY审阅完成，独立链/数值审计PASS。

RBF严格A-VMR1.389→13.492%，有效共边输出1→3/12，但相反绕序仅1→2/12、完整路径可达面积仍小；FF12.019→12.550%、I3.687→4.390%，两主要I尾部为866e座面n0与bdd29背板n0。FEEDBACK的C误差12/12下降（宏均值0.043324→0.017103），F平均仅变0.001772，接合改进很小；四座面均无新增有效共边。RBF新增边在原GEO之后出现，不能称为自己生成了精确接口。维持单次D0参照，两个候选均不默认采用，不自动追加求解器/训练/蒸馏。

[完整报告](experiments/r2_c_restore_coupling_v1/REPORT.md)、[图集](experiments/r2_c_restore_coupling_v1/gallery/index.html)、[逐父噪声](experiments/r2_c_restore_coupling_v1/decision_per_parent_noise.csv)、[恢复说明](experiments/r2_c_restore_coupling_v1/README_RESUME.md)。代码r2/tools/c_restore_coupling_v1，原数据/模型/评分器/旧报告hash保持，零commit/push；固定任务已停止。


## 2026-10-10 锚点感知去噪器适配完成：未证明role增量，暂不晋级

已完成R2_ANCHOR_AWARE_DENOISER_PILOT_V1。官方Chair去噪EMA同起点，112条通过局部C/CY筛查条件/25父（59条仍有Y内部非法、10条Y内部UNKNOWN，不是全网格干净集），clean C＋逐角点独立sigma0.02噪声free，唯一free MSE；D_CONT_NO_ROLE和D_ANCHOR各1000更新、batch8/micro1。ANCHOR额外零初始化1024D known向量加在PE投影后；梯度与BF16实际加法均已核验，不能归因于角色完全没参与计算。生成器、既有48RAW、GEO_C和原评分器固定。

实际完成16000F/16000B/2000正式更新；另32F/16B/2丢弃预检更新；96真实＋54合成最终推理。累计GPU阶段3939.52秒（65.66分钟），单GPU worker顺序，无GPU重试。144 POST、全8阈值、逐边界距离、实际共边图、FF源面事件、56张原PNG及144 AI_PROXY审阅均完成。独立最终审计PASS，927保护文件/hash不变。

USER_MIX的D0/CONT/ANCHOR有效共边输出1/2/2（各12份），仍只在86ec一父；新增n0可达free面积约78.46%，但新增边均非相反绕序，要求相反绕序的覆盖仍1/12。四座面在2%L仍零匹配/无有效共边。FF12.019/13.534/13.411%，ANCHOR比D0+1.392pp、9/12更差；I3.6865/3.6580/3.6580%，严格A-VMR1.3889/2.3148/2.3148%，PMR全0。ANCHOR合成noisy free/interface MSE相对D0+22.40%/+21.33%，clean不必要移动也增加。两个适配器有共有保腿/去局部突刺正例，但无重复明确的role独有接续改善。

全48有效共边输出2→4→4，新增OLD/bdd29父；相反绕序有效边数仍2→2→2。CONT_USER的FF有改善，FROZEN有大退化尾部，完整四来源/父/噪声结果均保存。一次CPU缓存身份int/string断言在任何新quality调用前触发，只修元数据规范化；无GEO/模型/质量重放，原始错误与diff/hash保留。

[完整报告](experiments/r2_anchor_aware_denoiser_v1/REPORT.md)、[图集](experiments/r2_anchor_aware_denoiser_v1/gallery/index.html)、[逐输出表](experiments/r2_anchor_aware_denoiser_v1/report_tables/real_per_output.csv)、[恢复说明](experiments/r2_anchor_aware_denoiser_v1/README_RESUME.md)。[审阅包](experiments/r2_anchor_aware_denoiser_v1/R2_ANCHOR_AWARE_DENOISER_REVIEW.zip)9,540,620字节，小于10MB；全部CSV/56JPEG/55份实际源码/diff及账本保留，CRC/相对链接PASS，CSV采用ZIP-LZMA、其余DEFLATE。重复大JSON留本地并列OMITTED_JSON.csv路径/hash。代码位于r2/pilots/anchor_aware_denoiser_v1；两份最终完整权重/推理导出和500恢复点留本地。维持D0单次为比较基准，不自动晋级、蒸馏、放大role、改loss或默认模型；无commit/push。固定任务完成后停止。


## 2026-10-10 USER_MIX 多次去噪对照完成：暂不采用3/5次默认

同一12 USER_MIX RAW复用D1，实际新增48官方去噪前向得到D2..D5，每轮C原字节恢复、N不变，无中途Geo；D3/D5终点各一次同配置GEO_C、24 CPU评分与24两视角图全部完成。零训练/蒸馏/新生成/新噪声。ALL D1/3/5：FF12.02/8.02/6.12%，CF4.88/4.49/4.32%，I3.69/3.71/3.77%，小角17.06/16.28/16.06%；严格A-VMR1.39/2.31/3.37%，PMR全0，共边输出1/2/2，但新增共边未通过双面反向绕序检查。四座面在2%L仍0。

D5 FF改善10/12、退化2；I改善4/12、持平3、退化5。33ec seat n0在D3/D5两视角整组腿消失，D1/3/5 free数136/85/66，D5 FF=0却小角54.55%，C错层未解决。其他11请求未见同等级支撑消失，但图审非盲AI_PROXY，不能认证无损。实际FF改善不仅靠删面：140个D1非法面对两面保留后变合法接触/分离，也有新非法对。保留一次为比较基准，未自动改默认或蒸馏目标。

[结论](experiments/r2_multistep_denoise_v1/FINDINGS.md)、[完整报告](experiments/r2_multistep_denoise_v1/REPORT.md)、[全部图集](experiments/r2_multistep_denoise_v1/gallery/index.html)、384指标行/496顶点距离/48质量与拓扑/逐父噪声/24新PLY及实际源码diff/hash均保存。独立数值、网格和48前向链审计PASS，895受保护文件/官方权重hash一致。固定任务完成后停止；没有commit/push、新loss、模型切换或后续训练。


## 2026-10-10 完整后处理复评：官方Chair去噪＋C保护GEO_C

用户要求按完整生成流程处理既有mesh并复评。已对同一4模型×6USER条件×2噪声的48份RAW，执行48次官方Chair去噪前向、原C写回、固定GEO_C及48次CPU复评分；无新生成器前向/训练/噪声。48份C逐位保持，48PLY/NPZ一致。真实官方0030000.pt（约6.79GiB，SHA256 6a31f4c7…）及匹配db86bc3源码已找到并保存在本轮独立目录；不再把历史“缺denoiser资产”当作当前状态。固定缩放0.95×0.3762；最终C保护几何导出替代原无约束量化导出，属于官方发布权重的C保持适配，不是原脚本逐位复现。

48/48 FF均下降，FROZEN/OLD/USER/MIX从90.36/91.97/88.96/89.65%降到16.11/15.30/15.36/12.02%。MIX CF5.60→4.88%、I3.77→3.69%、小角17.18→17.06%；座面小角31.93→33.28%。FF改善包含1886对MIX原非法面对两面均保留后变为合法接触/分离，不能只归为删面，也不能当体积或完整形状质量。

POST严格模型1e-6 A-VMR四组为1.85/3.90/3.51/1.39%，PMR仍全0；1%L的MIX从6.75降到5.56%。实际共享C边仅OLD与MIX各1/12；MIX四份seat在2%L仍0。后处理有很大free内部几何整理收益，仍未普遍接好C；今后评价保留RAW/同一POST并列，不自动采用新默认。24张两视角图全部AI_PROXY审阅，记录33ec细腿损伤；不能只看较低FF。

[结论](experiments/r2_postprocess_avmr_pmr_v1/FINDINGS.md)、[完整报告](experiments/r2_postprocess_avmr_pmr_v1/REPORT.md)、[图集](experiments/r2_postprocess_avmr_pmr_v1/gallery/index.html)、逐例CSV、最终48PLY/NPZ、原评分代码快照/diff/hash和资源账本均已保存。独立复算768指标/496距离/192汇总/96拓扑一致。准备阶段误写1份父NPZ，已按历史ZIP原SHA256恢复，77原输入终态hash均一致；事故和恢复有独立记录，正式48前向无重试。未改loss/默认模型/原评分器，无commit/push；任务完成并停止。


## 2026-10-10 A-VMR／PMR 首轮离线估计完成

用户后续授权实际估计；已在独立目录 experiments/r2_avmr_pmr_offline_v1 读取4模型×6USER条件×2噪声的48份既有RAW，零模型加载/前向/训练/新生成/POST/S_join。代码 r2/tools/avmr_pmr_v1/run.py 为固定面板CPU入口，公式和8档阈值先冻结；77原文件hash未变，8项指标正反例通过，独立实现核对384评分行、496顶点距离、96汇总与288配对均0差异。

全部C原始顶点ID开放边界作为描述性评分域，尚无用户必接接口；论文评分尺度兼容性UNKNOWN，因此不是论文表数值复现或接合成功率。严格模型单位1e-6下四组A-VMR/PMR均0；固定L的1%档FROZEN、CONT_OLD、CONT_USER、MIX分别为22.72/8.33、17.76/0、17.03/0、6.75/0（%）；2%档分别41.27/25.00、37.00/8.33、33.99/16.67、17.13/0。MIX4份座面在2%档边界顶点匹配仍0，但不等于整片表面都远离F。低I不代表顶点匹配改善，匹配高也不保证无重层或真实共边。

[结果解读](experiments/r2_avmr_pmr_offline_v1/FINDINGS.md)与[定义及全阈值表](experiments/r2_avmr_pmr_offline_v1/REPORT.md)；逐输出、逐顶点、逐父/噪声CSV、实际代码/diff/hash与CPU账本均保留。固定任务完成后停止；未改原评分器/RAW/旧报告，未改变loss或默认模型，未commit/push。下方NOT_IMPLEMENTED/NOT_RUN仅描述此前“先记下来”的历史阶段。

## 2026-10-10 评价选择：记录 A-VMR／PMR，不推进 S_join

用户明确要求先记下 MeshFIM 的 A-VMR（Average Vertex Matching Ratio，平均边界顶点匹配率）和 PMR（Perfect Matching Ratio，全部边界顶点匹配的样本比例），不继续推进新增 S_join 指标。优先保留论文原定义与来源，后续采用时明确边界集合、坐标尺度和容差；这不是已冻结的完整论文主表。

定义及适用限制见 [指标库补充记录](native_t1/docs/control_metrics.md#meshfim-boundary-metrics)。匹配对象只能是新生成 F 的顶点，不能包含复制的 C；A-VMR／PMR 衡量边界顶点匹配，不认证实际共边、无穿插或完整主体接合。原论文容差为 1e-6，不能不核对坐标尺度就直接套入本地模型单位。

本次仅更新持久化文档：两项本地接入状态 NOT_IMPLEMENTED，运行状态 NOT_RUN；未新增评分、训练、模型前向、生成或后处理。旧 S、I、评分源码、RAW和历史报告保留；不自动改loss、切换默认模型或commit/push。

## 2026-10-10 USER主导混合从官方初始化实验完成，保留候选

已完成R2_USER_MIX_FROM_OFFICIAL_V1：单臂从官方EMA/fresh AdamW step0训练6000步，Native/T1+role+Geo、FM和全程HYBRID0.25；25父117USER+395OLD，每批6USER/2OLD、父曝光平衡。实际48,000F/48,000B/6000更新全部成功，无OOM/重跑；训练8556.64秒。12新增USER Euler50/600采样F，核验复用36历史同请求结果，共48评分、24PNG和48逐输出/36配对AI_PROXY审阅。

ALL结果MIX CF5.60%、FF89.65%、I3.77%、小角<5°17.18%；相对FROZEN为−2.72pp、−0.71pp、−6.06pp、+6.81pp。座面小角31.93%，不可称全面更好。866e完整座面相对FROZEN/OLD双噪声C采用改善，33ec座面一好一坏，四靠背无明确C用途提升。对CONT_USER，I−8.65pp但FF+0.70pp、小角+1.37pp。未以旧随机任务退化作门槛，未做唯一GT排名。

24项CPU测试PASS，执行审计10部分PASS，2236数值单元独立回算一致，1059份保护文件未变。正式全量权重、推理导出和5500恢复点保留；没有默认切换/自动commit/push。当前候选用于继续研究，不认定为整体更优基线；六父开发面板与历史26/25父、6000/7000步差异限制因果及泛化结论。后续优先候选是固定权重在未看过的USER条件上验证采用收益和小角代价，本轮没有追加执行。

报告：[R2_USER_MIX_REPORT.md](experiments/r2_user_mix_from_official_v1/R2_USER_MIX_REPORT.md)；图集：[gallery/index.html](experiments/r2_user_mix_from_official_v1/gallery/index.html)；完整CSV、原始评分代码小文件、diff/hash、资源与调用账本均在同目录。代码仍为r2/pilots/user_mix_from_official_v1/独立实验入口。固定任务已完成并停止。


## 2026-10-10 USER 主任务单版从官方初始化实验已授权

用户明确要求按 USER 为主的方向实际训练一版。冻结 R2_USER_MIX_FROM_OFFICIAL_V1：单臂 R2_USER_MIX，从同一官方 Chair EMA 初始化 Native/T1+role+Geo、fresh AdamW step0，FM和HYBRID0.25全程6000更新。25共同父，117 USER条件＋395旧条件；每批8不同父、6USER/2OLD、4clean/4width，父曝光严格平衡。75%/25%为本次探索选择，不称最优，也不是梯度或loss贡献的固定权重。

仅新增现有6 USER DEV×2冻结噪声的12份Euler50 RAW；此前R2_FROZEN、CONT_OLD、CONT_USER各12份同请求RAW/评分核验后复用，共48结果。旧随机裁剪回归不新增生成、不作为门槛。无新loss、B、教师、POST或唯一GT排名。历史参照父数和总步数不同，只回答新配方是否值得继续，不作混合比例的独立因果证明。

独立代码 r2/pilots/user_mix_from_official_v1/；独立产物 experiments/r2_user_mix_from_official_v1/。run.py提供可恢复分阶段入口，按账本计实际调用；失败不自动重放或降低设置。固定6000终点评价，不按中间图选点。任务完成后停止，不自动更换默认模型或commit/push。当前执行状态以该目录status/账本和最终报告为准。

## 2026-10-10 后续主任务以 USER 型部件条件完成为准

用户明确调整后续研究目标：主要研究有实际用途的 USER 型输入，当前为座面／靠背等局部面片，后续可扩展不同靠背、椅腿等部件。后续主评价围绕这些输入展开；旧随机裁剪 C20/C40 任务的保持能力不再作为晋级或否决门槛，不为保留旧任务而强制安排旧面板生成预算。

此前用户提出把 USER 条件混入原 R2 数据作为增强，这与评价目标的调整分别处理。旧条件可以作为可选训练辅助，但是否保留、混入多少，只根据其对 USER 条件采用和生成质量的帮助决定。此前讨论的 75% 旧条件／25% USER 只是未执行建议，不是已确定配方，不因维持旧任务而锁定该比例。继续采用 R2 训练路线；具体混合比例、更新预算及实验注册另行确定。

主判断仍包括：C 是否合理参与完整形状、整体是否完整合理，以及相交／重叠候选、退化面与明显碎片等代价；不恢复唯一 GT 重建排名。既有 pilot 的原始结果和报告保留，旧面板退化仍是历史观察，但不能单独否决面向新任务的方法；新 USER 面板自身的质量问题仍需如实评估。该条是研究方向记录，没有启动训练、生成或评分，没有切换默认模型或自动 commit/push。

## 2026-10-10 R2 用户式条件续训 pilot 完成，暂不晋级 USER

已按用户授权实际执行 R2_USER_CONDITION_FINETUNE_PILOT_V1：同主R2 full step6000/完整AdamW，25共同父、OLD395旧C与USER117活动C，各1000更新（global7000）；Native/T1+role+Geo、FM、全程HYBRID0.25。正式2000更新/16000F/16000B；小K预检8F/8B/0更新；68新增Euler50+16核验复用RAW，84统一CPU评分、56PNG、84逐输出及28配对AI_PROXY审阅均完成。训练双worker波次1818.62秒，设备VRAM观测峰9202MiB，无OOM/模型调用失败/重跑。579执行核验PASS，3846保护文件未变。

结果不支持用USER替换主R2：新DEV只有2条座面各一个噪声C用途更相容，无双噪声一致收益，4靠背未见清楚C用途改善。USER相对OLD：新DEV CF −0.30pp、FF −3.02pp，但I +4.77pp、free小角<5° +5.35pp；旧回归CF +1.12pp、I +0.60pp、小角 +1.83pp。图审旧面板整体1改善/13接近/2退化，AI_PROXY非人评。I_dup为近并行覆盖候选，可能包含合法薄壳；不能直接当非法重复或采用率。

两份full终点、两份generator、两份仅恢复500检查点均保留。结果报告：[R2_USER_CONDITION_FINETUNE_REPORT.md](E:/MeshFlow-Control-artifacts/experiments/r2_user_condition_finetune_pilot_v1/R2_USER_CONDITION_FINETUNE_REPORT.md)；图集：[gallery/index.html](E:/MeshFlow-Control-artifacts/experiments/r2_user_condition_finetune_pilot_v1/gallery/index.html)。完整逐例CSV、来源源码/diff/hash、账本及DELIVERY在同独立目录；新代码r2/pilots/user_condition_finetune_v1/。没有自动commit/push或默认模型切换。

后续训练继续R2路线，当前117池是保留的局部适配证据，未定为新默认数据。若继续，优先候选为新旧条件混合回放：检验完全替换C20/C40的分布偏移是否造成部分回归，同时能否保留局部C采用信号；此假说未启动、未追加比例/预算。固定任务已经结束，不从500点重放已消费更新。


## 2026-10-09 R2 用户式条件续训 pilot 已获授权并开始执行

用户明确批准执行 R2_USER_CONDITION_FINETUNE_PILOT_V1；此前数据准备交接中的 training_authorized=false 仅描述旧准备阶段，不再是本次执行的授权状态，原交接文件保持只读。
新代码位于 r2/pilots/user_condition_finetune_v1/，独立输出 E:/MeshFlow-Control-artifacts/experiments/r2_user_condition_finetune_pilot_v1/。冻结 registration.json（SHA256 e0ad17e15915efba7dd5e20c668e99b27ed40e7630fcdaf8bd8690325abd5961）。
固定主实验 R2 full step6000（SHA256 89f49e014615eb4a41e052ed90719a2580ead46418997c268e35e09ff37a7541）及完整 AdamW/RNG；25共同父，OLD395旧条件、USER117活动条件，各1000更新/8000曝光，Native/T1+role+Geo、纯FM、全程HYBRID0.25，新增辅助头/教师/后处理均0。
评价固定6新DEV×2噪声与8旧请求×2噪声，三模型共84输出，正常新增68条Euler50、旧冻结R2的16RAW核验后统一CPU重评分。预检上限16F/8次零更新B，500步只作恢复，不按图片选终点。新旧面板分别汇总，完整定性审阅，不做唯一GT重建排名。
当前为执行中，实际完成以该输出目录的各阶段账本/最终报告为准。固定预算完成后停止；不自动增加训练、切换旧采样默认模型或commit/push。

## 2026-10-09 R2 用户式条件数据准备完成（后续训练路线）

后续训练统一使用 R2：Native/T1 + role + Geo、FM、全程 HYBRID lambda=0.25。新条件池替换旧 C20/C40 构造，不将区域面积25/50/100%混同于父面数比例。未来分支起点固定主实验完整 R2 step6000 及对应 AdamW；本轮没有加载模型、训练、OT、抽噪声或生成mesh。

实际数据位于 E:/MeshFlow-Control-artifacts/experiments/chair_r2_user_conditions_v1/。权威活动清单为 active_conditions.json，交接为 dataset_handoff_active.json，查看 DATA_PREPARATION_REPORT.md 和 gallery/index.html。25共同训练父、47区域、117训练C/Y（56座面/61靠背），6新DEV条件；8旧DEV条件×原2噪声继续保留。原122train+7DEV候选已完整保存；6条极窄边条在实际看图后标HOLD（5train/1DEV），不进入主池。400a两类区域未能可靠分区，因此两条未来续训须同步排除该父；其余已知14个源缺陷父没有自动删除。语义为AI_PROXY，源相交/重叠/UNKNOWN仍逐例记录，不能称作无交干净数据。

129份C/Y都只由原FP32面与严格补集构成；6154项原导出验证、669项活动子集验证均通过，289个保护文件hash未变。图集包含每候选两视角M/R/C/Y/C+Y，原CSV、面ID、源映射、完整面对诊断和代码快照均保留。准备代码在 r2/data_tools/user_conditions_v1/；run.py --stage check 可回读核验并记账。

新DEV仅6条件，未来三模型新旧合计84逻辑输出；冻结R2旧16 RAW的身份和评分合同满足复用条件时新增68。新噪声只是未物化槽位。公共R2入口仍绑定旧26父/C20-C40/6000步，下一次需独立续训合同与入口，不能绕过恢复检查直接塞入此清单。此次停止在数据准备，等待数据审阅后再启动两条1000步；不自动改旧START采样默认，不commit/push。

## 2026-10-09 用户式条件数据准备与后续训练路线

用户明确要求后续训练统一走 R2。这里指以 R2 的 Native/T1 + role + Geo、FM、全程 HYBRID lambda=0.25 为配方基础；当前任务仅准备新的条件池，不把新座面/靠背条件强行凑成旧20%/40%面数。未来续训候选起点固定主实验 R2 完整 step6000，不自动选择复验权重，不恢复 B 或其他辅助loss。

本次已授权构造：现有26训练父，每父座面/靠背各至多一个核验区域、区域内25/50/100%面积参考的嵌套真实面子集，最多156条，去重/NA后如实计数；8DEV每父至多1新C，另登记8旧C及既有两噪声作回归。数据和审阅图独立放 E:/MeshFlow-Control-artifacts/experiments/chair_r2_user_conditions_v1/，源码在 r2/data_tools/user_conditions_v1/。零模型加载、零前向、零训练、零OT、零新Gaussian、零完整网格生成；只从原父网格取原面及严格补集，不修复或重三角化。先交付区域/条件清单、质量证据和检查图，待数据清单确认后才启动两条1000步续训。旧默认START采样入口与保留权重不自动切换；本轮不commit/push。


## 2026-10-09 R2 代码独立整理（已按用户要求提交并推送）

用户确定下一步研究配方为 R2 全程混合 C20/C40 + 全程 HYBRID，并选择现有 GitHub 仓库内独立的 r2/ 目录。使用说明见 [r2/README.md](r2/README.md)，三条训练/采样路径见 [训练结构](r2/docs/training_structure.md)。

实际 GitHub master 核查为 8961b7a41fc153b5b45f0c8903370c6bfd1cf04f，与本地 HEAD 一致；此前 R2 位于无提交、无 remote 的 E:/MeshFlow-Control。此次复制计算依赖并保留其源码字节，新增只公开 R2 的入口；旧项目及外部权重/数据/RAW 不修改。主实验和确认实验两套 R2 均保留，没有自动选定默认终点，native_t1 的 START 采样默认继续保留。

当前 R2 训练接口仍绑定历史 26 父条件池、6000 步、20%/40% 面数占比；新 C 构造、洗数据与追加训练尚未执行。此轮只做 CPU 测试、权重严格加载和冻结输入抽查，不代表重训或生成等价性已经验证。证据位于 native_t1/maintenance/r2_port_20261009/，公开验证说明见 r2/docs/validation.md。

用户随后明确要求推送 R2，已将 72 个 R2 源码/测试/文档及首页入口文件提交并推送到 origin/master：573ad4dc960de26b1569595a4bfa736abe6d1e39。远端 SHA 已再次核实一致。入库 blob 保留 51 个源文件的原始 SHA256；未提交权重、数据、其他未跟踪文件，未运行新训练或生成。推送记录见 native_t1/maintenance/r2_port_20261009/push_receipt.json。

## 2026-10-08 GEO_C 可选后处理正式入库

保持 C 的固定 GEO_C 几何整理已迁入正式模块 native_t1/postprocessing.py，入口为 `python -B -m native_t1 postprocess --input RAW.npz --known-faces K --out NEW_DIR`。NPZ默认键output，也可显式--input-key raw；输入FP32、前K面为C、使用原模型坐标。仅需NumPy，CPU执行，不读取权重；另存post.npz、indexed post.ply和postprocess.json，保留来源映射、删面原因和C位模式回读。默认START采样仍输出RAW，不自动执行后处理。

本轮完成并推送13个源码/测试/文档文件到origin/master：8961b7a41fc153b5b45f0c8903370c6bfd1cf04f。公共CPU测试130/130通过；32个原后处理案例416项数组逐字节一致，统计和逐面记录一致（耗时除外），97个旧输入/结果hash未变。真实模型加载/前向、训练实验、新生成和重评分均0；原实验报告和评分器未改。C保持不意味着原vertex-ID图、水密、无相交或稳定接缝。

用法和固定尺度见 [正式后处理文档](native_t1/docs/postprocessing.md)，本地迁移账本位于 native_t1/maintenance/postprocess_port_20261008/。此次入库未包含任何权重、网格、实验数据或此前无关的未跟踪文件。独立MeshFlow-Control仓库没有在本轮提交或设置remote。

## 2026-10-07 持久评价记忆：完整指标库

按用户要求审阅并保存 [Chair Control 完整指标库](native_t1/docs/control_metrics.md)，共35类指标，包含定义、适用任务、GT依赖、尺度、失败/未决和汇总规则。AGENTS.md 已加入后续评价前的阅读入口。

任务目标是给定真实面片 C 的多样化合理补全；来源重建是辅助诊断。按已授权预算冻结适用项并保存完整结果，主文可选择三个有贡献证据的指标，但目前未确定具体三项。无GT、无指定接口、无外露要求的任务分别不强算来源重建、S、E；优先客观量，人评另列。

已修订附件中的实现状态与关键口径：C坐标/绕序保持不等于原索引图保持；旧CF比例只计proper，FF非法比例另有并集定义；已有共享边数覆盖不等于新Gamma弧长覆盖；S角度为≤20°；E确定参考域与全局未知域上下界分开。新定义和文献项不冒称已实现，旧评分源码/结果不回写。

本轮仅写文档记忆，未运行训练、模型或特征前向、生成、评分，未改默认START、loss、权重、RAW及历史报告，未commit/push。

## 2026-10-07 当前决策：回到原始 START，结束 CAPPED 探索

按用户明确要求，以原始 Chair HYBRID START / total5000 继续作为 baseline。正式默认 profile 一直为 chair-hybrid，已实际核验解析及一次 CPU strict load；零模型前向、零训练、零生成。START 两条硬链接路径及完整 AdamW/RNG/输入流状态均保留。

已删除 chair_control_feedback_auto_v1、chair_capped500_gradbudget_v1 和 chair_capped500_half_v1 的22份实验权重，以及两处正式 OT 派生缓存的16,000个NPZ，共33,426,755,776逻辑字节（约33.43GB）。OT尝试日志、训练计划、小型测试缓存、原始RAW、冻结条件/噪声/查询、反馈池、全部评分/报告/CSV/图集/源码和审阅ZIP保留，43,405个保留文件前后hash一致。

相关历史代码留作评分依赖和证据，不自动运行旧恢复入口，也不因权重/缓存缺失而重训或重生成。原报告中的旧文件存在状态以本轮删除清单为准。官方权重、Geo4000、START、旧显式FM/JEdge5及其它既定保留checkpoint不在本轮删除范围。没有修改正式代码或commit/push。

详见 [清理报告与完整账本](native_t1/maintenance/capped_retirement_20261007/CLEANUP_REPORT.md)。

维护入口为 `native_t1/`。2026-10-05 按用户明确要求退回并正式接入 START：原 OT_HYBRID pilot1000 / conditional_total5000，Native T1 + C-only Geo + FM，训练 coupling lambda=0.25。默认采样 profile 现为 `chair-hybrid`；MOMENT V2 及其 finite-surface 前序活动包已封存并移除。

## 当前使用方式与边界

```bash
python -B -m native_t1 sample --condition /path/to/C.npy --num-faces 163 --seed 2452526625 --out native_t1/runs/chair_start_sample
```

总 N 必须显式指定 128..256；C 为原坐标尺度的有限 FP32 数据，0<K<N。使用原 MT19937 free-face 噪声、50 步 Euler、BF16 forward / FP32 参数与积分，已知 C 在 51 个状态均精确保留。详见 [START 说明](native_t1/docs/chair_start.md)。

正式本地权重为 `native_t1/checkpoints/chair_hybrid_start_total5000.pt`，是原 `chair_hybrid_coupling_v1/checkpoints/OT_HYBRID_pilot1000_total5000.pt` 的硬链接。文件 SHA256：`d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80`。原完整 AdamW、RNG 和输入流状态保留；推理不恢复优化器。Git 只纳入正式源码、测试、文档及身份 manifest，权重继续本地保存。

旧 `--profile fm-geo`、`edge5` 等保持显式可用的 N112 路径。`train` / `train-edge5` 仍是显式预算、fresh AdamW 的旧 N112 训练入口，不是 START 续训入口。任何新训练需要独立的数据、coupling 和预算合同，不因改变默认模型自动开始。

## 保留的正式和历史权重

|角色|路径|累计步数|
|---|---|---:|
|START 默认|native_t1/checkpoints/chair_hybrid_start_total5000.pt|5000|
|旧 FM 显式 profile|native_t1/checkpoints/ablations/fm_cumulative3000.pt|3000|
|JEdge5 显式可选|native_t1/checkpoints/jedge5_cumulative3000.pt|3000|
|旧共同起点 G0|native_t1/checkpoints/tgeo_g0_cumulative2500.pt|2500|
|surface 旧参照|native_t1/checkpoints/ablations/surface_cumulative3000.pt|3000|
|Edge-only 旧参照|native_t1/checkpoints/ablations/edge_only_cumulative3000.pt|3000|

## 2026-10-05 MOMENT 封存完成

删除 MOMENT/H_CONT 正式 V2 权重各 5 份、8000 个派生 OT NPZ 和三个退役活动源码包，共 8061 文件、15,677,075,950 bytes（约 15.68 GB 逻辑字节）。源码先行归档；原始结果、报告、逐例 CSV、日志、评分源码快照、决策 ZIP 和小型合成 CPU 测试 fixture 保留。

11,488 个保留文件前后 SHA256 一致。START CPU strict load 通过，公共 CPU 测试 109/109 通过，删除后 CLI 导入通过；真实模型前向、训练更新和生成均 0。根据本轮用户明确授权完成本地 Git 提交，没有 push。详见 [清理报告](native_t1/maintenance/moment_retirement_20261005/CLEANUP_REPORT.md) 和其中的账本。

后续仍遵守 [AGENTS.md](AGENTS.md) 的资源调度与保护边界。以下是按日期保留的历史记录；其中旧的默认模型、HEAD、实验状态及已删除路径以其记录日期为准。

## 2026-10-02 Pair清理

删除两处隔离目录：`experiments/mf_h1_local0/geometric_pair_closure_pilot/`、`native_t1/runs/geometric_pair_closure_20261001_171356/`。四份Pair step250/500 checkpoint、诊断及正式原始数组、连接候选、活动实验源码均退役。

报告、results、逐输出表、训练记录、固定图、协议和code.diff保存在 [精简档案](native_t1/maintenance/pair_cleanup_20261002_100559/CLEANUP_REPORT.md)。档案内历史报告的checkpoint/raw路径是执行时记录，现已删除；code.diff可恢复实验代码，但不恢复已删除的权重/数组。

官方权重、原数据、G0/FM/JEdge5和其他历史实验不在删除范围。本轮没有模型前向、训练更新、完整生成、reset/clean、commit或push；公共代码保持HEAD `5c11eb2`。权重、数据、runs、maintenance及本地WORKSPACE说明仍Git忽略。

本地已准备输入为 `output/t1_publish_20260930/chair_n112.npz`；公开代码可依据官方数据和固定recipe重新准备。当前证据仍来自两个观察过的父对象，不代表新对象泛化、水密或全指标优势。

## GPU任务调度约定（2026-10-02）

后续每次实验及新增GPU任务前按 [AGENTS.md](AGENTS.md) 检查显存、GPU忙碌率、已有进程及CPU/RAM；独立任务优先尝试有资源约束的并行，不再默认逐臂串行。并发数来自同类任务实际峰值和吞吐，保留显存余量，逐个启动并动态复核。

未知峰值时先观察一个已授权任务，不额外新增探针训练。训练、评价、采样分开估算；不改变microbatch、精度、随机流、确定性或实验预算来塞入更多任务。CPU匹配/OT/离线审计可与独立GPU工作重叠，独立输出和固定seed不随完成顺序变化。

本次只读取GPU状态和记录执行偏好，没有启动训练、采样或新的调度服务。检查时RTX5080总显存16303 MiB、可用13080 MiB、GPU利用率0%；这是空闲快照，不是运行峰值或固定并发数依据。

已有显存参考：`experiments/mf_h1_local0/outputs/tgeo_m10_20260930_173200/results.json` 中两臂 `arms.*.peak_cuda_allocated` 为约2.520/2.521 GiB；其72条 `generated[].sampling.peak_cuda_allocated` 为约0.536 GiB。这只是历史PyTorch allocated峰值，不含完整CUDA进程占用，也不是当前JEdge5所有loss的峰值；不能按16GB直接相除定并发。当前采样入口记录allocated峰值，公共训练入口尚无完整reserved/进程峰值记录，后续须随首个已授权任务补齐观测。allocated属于reserved的一部分，不把两者相加重复计算。

## 2026-10-02 正式多对象实验前的数据盘点

本地 `experiments/mf_h1_local0/assets/shapenet-03001627` 实际已有 3135 个官方 UID、3125 组精确几何；官方 train/test 为 2822/313，未发现 UID 或精确几何跨划分重复。近重复与同源家族尚未排查。当前两对象 N112 是维护训练入口的限制，不代表本地只有两个对象。正式实验应区分条件微调未见与官方预训练未见，不直接把旧双对象模型作为公平的纯 T1 / Geo 比较起点。

面数为 28–800，中位 470；原始索引拓扑/退化面诊断及逐对象记录见 [数据扫描](native_t1/maintenance/temp_cleanup_20261002_160601/dataset_scan.json)。先准备现有椅子数据即可推进；跨类别可补下载 MeshFlow 的 `shapenet-cls.tar.gz`（25.48 MB，历史归档清单确认包含 bench/chair/lamp/table），语义部件控制可申请 PartNet。具体链接、体积与适用边界见 [数据下载清单](native_t1/maintenance/temp_cleanup_20261002_160601/DATASET_DOWNLOADS.md)。本轮未下载数据、未启动训练或采样。

历史 checkpoint 已按用户明确批准的固定清单完成清理：删除 27 份退役权重及 30 个临时文件，共 35,971,879,861 bytes（约 35.97 GB）。official、无 Geo T1 500/1000/1500、canonical 五份及关键 Base/Geo 和 EMA 终点继续保留。3,711 个关键文件 SHA256 校验一致，原盘点中清单外缺失为 0；实验报告、指标和原始采样证据均保留。已删历史终点不能直接重新采样，相关分支若重新研究需重训；历史报告内旧 checkpoint 路径以 [清理报告](native_t1/maintenance/temp_cleanup_20261002_160601/CLEANUP_REPORT.md) 的删除记录为准。
## 2026-10-02 已接收 ShapeNet class split 归档

用户下载的 `shapenet-cls.tar.gz` 已从 Downloads 移至 `downloaded_data/archives/shapenet-cls.tar.gz`（25,483,419 bytes）；SHA256 `06707caff4ce465f402e6b69f81056204c579921d371eb2c9e4e9a3929f7aef2` 与官方元数据一致。来源及校验记录在同目录 `shapenet-cls.manifest.json`。原始归档已保留，尚未解压；既有 chair 数据不变，后续可从归档提取缺失的 bench/lamp/table 类别。
## 2026-10-04 Chair 第一档旧恢复点清理

经用户批准，删除 chair_domain_control_v1 的 T1/Geo/Full_latest.pt（phase2800、total3800），以及 control_mvp48_budget_single_v1 的 T1/Geo/Full_latest.pt（total4800）和 T1_fit2_probe_latest.pt（probe800），共 7 份、10,950,114,685 bytes（约10.95 GB）。只执行已评估的第一档；第二、第三档未删除。

全部固定中间评价状态和终点继续保留。下一步 Hybrid Coupling Pilot 的指定起点仍为 native_t1/runs/chair_domain_control_v1/checkpoints/Geo_phase3000_total4000.pt，模型与完整优化器状态未改变，文件 SHA256 为 190a16e2aec96a1f84f63460b5f97b00e23499338102f82e7530218ecb185ebf。

10,773 个受保护文件前后 SHA256 一致，清单外意外缺失为0；原数据、冻结C/噪声/GT查询点、raw、源码、报告和归因包保留。旧清单与同名JSON保留为历史证据，其中上述七个PT路径已明确退役；不能再用它们恢复旧步数。详见[清理报告与账本](native_t1/maintenance/weight_cleanup_stage_a_20261004/CLEANUP_REPORT.md)。本轮没有模型加载、前向、训练或生成，没有启动下一步试验。


## 2026-10-04 Chair Hybrid Coupling 有限试验完成

`native_t1/runs/chair_hybrid_coupling_v1/` 已实际完成同源 Geo4000 两臂各1000更新、固定A/B/C与响应诊断。两新终点为本实验 pilot1000/total5000，身份见 checkpoint_manifest.json；不要与历史预算实验5000步混称。未切换FM默认模型或JEdge5可选模型。

实际总计2000更新、432登记请求（18原始NA）、292完整采样、122历史raw缓存、2400 BF16交叉诊断、72 FP32诊断及10预检前向；总前向33082，模型失败/未决0。866个保护依赖前后hash一致；冻结新实现未改变，训练曝光、主指标算术及三个检查点轨迹独立审计PASS。

结果为 GEOMETRY_SIGNAL_WITH_TRADEOFF：HYBRID相对BASE验证A20固定查询MSE下降约40%，6/8父改善，覆盖与p95改善；但小角面增加、跨噪声控制分歧、40%上下文部分指标及已观察留出退化。优先下一步是现有raw的小角面收益归因，检验误差收益是否主要由新增细长面贡献；该后续未执行，没有追加训练、loss或求解器扫描。

详见本轮 CHAIR_HYBRID_COUPLING_REPORT.md 与 DELIVERY_INDEX.md。纯CPU审计读取器/报告文案修正单独保留原版、diff和哈希；正式实验没有静默重跑。当前固定任务已停止。


## 2026-10-04 Chair Finite Surface Attention 门禁停止

`native_t1/runs/chair_surface_attention_v1/` 已完成资产登记、独立实现、CPU门禁及受限真实预检，结论为 IMPLEMENTATION_BLOCKED。共同起点固定为上一轮 OT_HYBRID pilot1000/conditional_total5000（SHA256 d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80）。没有新增正式训练权重，没有切换默认模型。

192点求积在4个真实训练任务均未通过0.01精度门禁，7504个普通query/known对中1459个失败；误差集中在t=0.9/1。96个密集参考对子集中另有1个未达到0.001收敛要求，故保留求积规则差异的表述，不声称真积分误差界。

实际真实预检仅10前向、1次零更新反向；新增显存日志器重复传入time_unix/pid导致停止，模型调用失败0。没有修正后重跑，真实三臂成本比较与MOMENT/SURFACE接线检查均未完成。H_CONT参数、AdamW与RNG恢复通过。三臂正式更新均0，所有新质量比较NA。登记576请求，START只读复用138个完整raw，零新采样；隔离CPU小模块1次测试更新另记。

固定任务已停止，不应自动恢复训练或把控制器DONE解释为真实预检通过。优先后续假说是晚期窄核相对长三角边的n8子网格欠分辨，先在同一冻结失败输入上建立误差受控参考；本轮未执行该后续。日志接口缺陷是未来重新审计的工程前置修正。

详见本轮 CHAIR_FINITE_SURFACE_ATTENTION_REPORT.md、DELIVERY_INDEX.md 和 RUN_AND_RESUME.md。原失败源码、自动报告、完整CSV、实际评分源码与hash均保留，最终呈现修正位于postrun_review，未改旧结果。


## 2026-10-04 Chair Finite Surface Attention 明确授权修复续跑完成

用户明确要求继续下一步、不要因实现缺失停止后，在独立 `chair_surface_attention_v1r1` 代码与结果目录修复了显存日志保留字段重复问题。CPU回归、真实只读nvidia-smi及实际三臂遥测均通过，旧v1所有文件未改。

本次实际新增76前向/20次零更新反向，与旧轮累计86/21，仍在原96/24硬上限内。三臂全部完成预定检查，无调用/遥测失败、无未决调用；原参数、AdamW与RNG恢复。H/S各任务2次成本测量，M每任务1次限制已登记；未用余量不授权追加任务。MOMENT/SURFACE真实FP32及BF16零初始化均与H输出逐位一致。

独立CPU自适应积分完成固定8例，34次积分、1472625核点、24次旧求积CPU调用，无重试。两档精度自洽通过；192点最坏均值差约0.111808，确认晚期窄核欠分辨。12288点参考在部分难例也有偏差；49152点在本8例低于0.001，但不是全任务或严格认证。

真实成本已完成：SURFACE/H完整forward在普通40%/最大40%为3.05084/3.13547，略超冻结3倍门槛；F+B四任务均低于3倍。严格置换容差H本身已失败，M/S在最大N另有padding数值差异，全部保留、未放宽阈值。实现已实际跑通，当前阻塞由数值精度、成本及数值不变性合同构成。

正式训练和新采样仍为0，没有改变默认模型。优先后续是保持数学定义、在冻结难例上验证能解析边/角窄核的有限三角积分实现；该新生产算法与训练本次未启动。修订固定任务已完成，详见 `native_t1/runs/chair_surface_attention_v1r1/CHAIR_FINITE_SURFACE_ATTENTION_R1_REPORT.md` 及 `DELIVERY_INDEX.md`。

## 2026-10-05 Chair Moment Channel Learning V2 实际训练对照完成

`native_t1/runs/chair_moment_channel_v2/` 已实际完成从同一 HYBRID total5000 起点恢复的 H_CONT/MOMENT_XY 两臂各1000更新，终点均 moment1000/total6000。20个冻结新实现文件和2464个保护依赖hash不变；旧AdamW延续、新通道参数与状态单列。未切换FM默认模型或JEdge5可选模型，未覆盖旧START。

实际16000训练forward/backward、2000 optimizer更新、120预检forward/44零更新backward、276新终点Euler50与2初始回放、138 START raw缓存、18原NA；采样NFE13900，总forward30020，模型失败/未决0。两臂8000训练样本与1000批次逐项配对，完整检查点/raw/账本独立审计PASS。孤立CPU模块仅1次测试更新另记。padding压缩修复在四真实fixture的变长padding逐层与输出均bitwise通过；原严格逐元素置换FAIL仍保留，新归一化FP32合同通过，BF16非零置换仅作披露。

质量结论 NO_CLEAR_LEARNING_SIGNAL：验证A20主MSE START/H/M分别0.0015310739/0.0015961439/0.0016433661，M比H高2.96%、仅3/8父改善，比START高7.33%。验证正确条件列H/M均9/24、三目标全对均0/8；通吃稍减与平均斜率误差稍降未形成跨噪声一致控制收益。M训练循环约43.8分钟、H约25.3分钟，采样均值4.6352秒/2.7163秒；不宣称已证明可靠内核加速。

暂留简单HYBRID/START，不扩大本通道训练。唯一后续建议是用既有raw与冻结查询定位父c1b8c2f0c4f3cc43aff7c08b060f5ed6的noise1两臂共同退化区域，检验是否局部缺面/错位而非整体宽度偏差；该后续本轮未启动。科学协议偏离0；训练前恢复状态修复、1次独立审计元数据工程错误修复、报告采样计时字段与编码修订均保留原件/diff。固定任务已停止。详见本轮 `CHAIR_MOMENT_CHANNEL_V2_REPORT.md`、`DELIVERY_INDEX.md` 和 `delivery_hashes.json`。


## 2026-10-06 保留 RAW / START，局部锚定路线退役

用户明确要求停止当前局部锚定路线并清理不用的代码、权重和缓存。`chair_bounded_attachment_v1` 的活动源码与运行目录已完整归档后移除；其 16 次 CPU 修正只改变 1 例、strict 接缝父均提高约 1.035 个百分点，未过冻结推进门槛，TRAJECTORY 未执行。继续使用原始 RAW / START，不启用局部后处理。

删除此前边界接合实验 19 份不用的权重（A 的 4 个中间恢复点、B/C/D 各 5 份），共 29,736,512,413 bytes；另删除 16,000 个派生 OT 缓存文件。A_FM attachment1000/total6000 按旧报告继续保留为等预算对照证据，不作为默认模型。START 两条硬链接路径、官方及维护权重、原始 RAW、C/噪声/查询点、四格实验报告和评分证据保持不变。

合计删除 16,655 文件、29,803,808,433 逻辑字节（约 29.80 GB）；局部锚定的全部代码与逐例结果封存在 17.66 MB ZIP，未保存已删除的大权重。578 个受保护文件前后哈希一致，删除后正式采样 CLI 帮助检查通过。模型加载、前向、训练和新生成均为 0，没有 commit/push。旧实验记录里的已删除路径以 [清理报告](native_t1/maintenance/local_anchor_retirement_20261006/CLEANUP_REPORT.md) 为准；不要自动恢复该路线。

## 2026-10-06 两三角形干净面片 control 小试

按用户要求清理本轮论文图目录 `deliverables/chair_paper_figures_v1/`，删除394文件、32,675,398逻辑字节（包含该轮ZIP/预览/面板缓存）。509个旧来源依赖hash不变，原RAW/数据/权重保留。清理账本见 `native_t1/maintenance/paper_figures_cleanup_20261006/`；旧论文图链接已失效。

随后固定START做两三角形共面座片小试：K2/N160，座面宽2.4与3.0、深2.4、y=-0.55，两配对种子20261006/20261007，共4次Euler50、200前向，零训练/反向/额外重采样。全部51态C精确保留，checkpoint、正式代码及模型state不变。输入增宽25%时，自由部分X宽分别增约1.47%与8.02%，边界接近比例均下降；图中可见生成座面位于已知面上方，不能称可靠接合。此为两噪声的局部观察，不代表总体性能。

完整新输入、RAW、51态、逐例/配对CSV、清晰输入与control/侧视PNG和独立审计保存在 `native_t1/runs/chair_two_triangle_control_v1/`，说明为 `REPORT.md`。不改默认模型/loss，没有自动训练、重放、修补或Git提交；该固定小试已完成停止。

## 2026-10-14 FULL Contour Readout 固定6000步完成

用户批准局部几何读取候选后，已实际完成 `r2/pilots/full_contour_readout_v1/` 的单臂6000步配对训练，起点为EXP67 global6000 live+匹配AdamW/RNG；复用FULL的47条完整座面条件、48,000曝光及6,000原OT，只有block6后的128维4头17D来源真实轮廓读取为方法变化（412,032参数）。原FM/R2、单次D0/GEO_C/来源定向合同不变。

同预算终点CONTOUR vs FULL：RAW明确采用均16/24，有合法接口均8/24，“采用且合法接口”均6/24，保留6、新增0、丢失0，DEV均0/12联合正例。合法边62→64，仅85既有接口增长；RAW轮廓边距3.28205→3.34608%L，POST CF4.45378→4.75748%、FF19.97185→19.72589%、I2.19320→2.34578%。本候选不晋级；FULL_S6000及其曲线继续作为充分训练的方法比较基线，维护默认模型未切换。S3000的7/24不替代预定终点。

实际6000更新、48000F/B、72新Euler50输出（3600F）、72单次D0、144新增AI_PROXY图审，全部完成；S0及原FULL结果明确缓存/继承。一次显存余量安全暂停在1114，另一次控制器状态读取竞态均按已保存状态恢复，无重复科学更新。最终审计PASS，0错误/缺失；旧194参数optimizer=12000、新8参数=6000，官方/历史资产及冻结源hash通过。归档器筛选尝试日志的工程错误已修复并留痕，无缺失OT。

完整报告、权重、逐例/逐父配对、图集、调用和恢复入口在[交付索引](experiments/r2_full_contour_readout_v1/DELIVERY_INDEX.md)。9,602,480-byte小包包含CSV、实际评分代码、code.diff、hash与48全组压缩预览；原分辨率图和大权重本地保留。固定任务完成停止，没有自动追加实验、默认切换或Git commit/push。