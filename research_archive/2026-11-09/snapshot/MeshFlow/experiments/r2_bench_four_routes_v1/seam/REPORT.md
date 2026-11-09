# Bench 窄池 B → 原版 SEAM1500：接续增加，但未成为整体质量升级

本分支已完成预定 1,500 次更新、46 请求三阶段评价及 138 页／276 格真实图审。**原版 SEAM 明显增加了 C 侧合法共边，并降低了平均 CF；同时 FINAL FF 和 I 升高，实际图审的主体—采用—支撑交集从 21/46 降为 18/46。** 本轮保留其接续正证据和全部产物，不将它升级为 Bench 整体质量基线。

这里的 B 是冻结的窄池 `NATURAL_HIGH` Native 终点，不是宽池 W 或系数匹配 M。本轮 S 同时增加了 SEAM 表示和 1,500 步训练，未配等预算 Native 续训反事实，因此比较的是完整续训方案，不能把所有变化单独归因于表示。

## 1. 实际训练与推理合同

起点为 `r2_bench_natural_n_recipe_v1/training/NATURAL_HIGH/checkpoint_step1500.pt`，global22500。恢复其 live generator、匹配 AdamW 和 RNG，新增固定 1,500 更新，到 global24000；没有选择中间最好 checkpoint。完整起点 SHA256 为 `111d2142780ddb21c9aa94cef9174571fbb22f0602d561c663c4af92b4a9e17d`。

- 训练池沿用窄池 B 的 26 父：OLD22、FOCUS 两父 a8/27、HIGH 两父 28b9/abf，完整目标与 C 均未换版。每更新抽取三个 OLD 加一个交替的 FOCUS/HIGH，每个条件使用原 clean/x-width 两实例，batch8、microbatch1。
- 实际 12,000 实例中 OLD9,000、FOCUS1,500、HIGH1,500。所有父保留；CPU 门检验证 26/26 来源目标可由原 owner-only SEAM 表示完整表达，未删面、复制 seam 或跳过不兼容父。
- 使用原 source-owner 训练构造、受限 bulk OT 和 OT 后 HYBRID0.25。活动 seam 的两个底边角点固定引用 C，原 FM 只监督可动角点；第三点与 bulk 仍由原生成器预测。没有增加损失、共享初始噪声、couple、RL 或新的网络支路。
- AdamW 沿用 lr1e-5、betas(0.9,0.95)、weight_decay0、eps1e-8、clip1，FP32 参数与积分、BF16 前向、math SDPA 和原确定性运行设置。
- 依据保留的 Bench Native→SEAM 范式，推理使用冻结的 833 参数历史 `OLD_S15000` C-only 头，logit≥0；本轮没有训练或校准该头。头 tensor SHA256 为 `21229acce25e3e070ae3adfa580798e1534e704cc6512e006de1654d2fefb9c4`。这是历史研究头，不是来源布局 oracle。
- 生产推理只读 C 与部署元数据，不读取 Y。闭合 C 无开放边，活动 seam 为零，保持原 Native free 槽推理，并将 seam 接口指标记为不适用。

本地独立 wrapper 将既有算子合同支持到 N128..799；固定面板最高 N776，训练池最高 N384。高 N 留出和宽池新增父包含面数外推边界，不能称为已经训练覆盖。原公共模型、旧评分器和旧结果均未改动。

训练门检与恢复证据见 [注册](registration.json)、[完整训练收口](closeout/training.json)、[真实曝光](closeout/exposure.json)；源代码入口在 `r2/pilots/bench_four_routes_v1/seam/`。

## 2. 固定评价与指标口径

完全复用当前 23 父／23 C、每 C 两份固定噪声，共 46 请求。组分别为 OLD16、FOCUS4、DEV8、HOLDOUT2、HIGH_TRAIN4、HIGH_HOLDOUT4、NEW_WIDE_TRAIN8 请求。最后一组继承历史命名，四父对本轮 B 与 S 都未见，不是 S 的训练父。

RAW 为 Euler50 终点；BASE_POST 为单次官方 Bench D0、恢复 C 与预测 seam 锚点、原 GEO_C 和来源定向；CLEAN_FINAL_POST 在 BASE 上执行同一 `HINGE_OPP015 → ALPHA → HOLE`。全部新 RAW 保存完毕后才开始 D0。B 的 46 份三阶段结果经身份校验复用，S 新增 46 份；完整结果 276 行，每行保留原 177 字段。

CF、FF 为非法面对影响的实际 free 面比例；I 是原 `ConflictCov_C`，三者不是相同的分母。主 I 只纳入 PASS：两模型、三阶段共同 19/23 父、38/46 请求；28b9、abf、945、af 的 LOW_SUPPORT 原值仍在完整表中，主表记 NA，不当零。本轮所有结果的 C 坐标精确保留检查均通过，但这不等同主体采用或正确接续。

## 3. 完整数值比较

以下先在每父内平均两噪声，再跨父等权。百分比已经乘100；共边条数与非流形边是每输出均值。

| 阶段 | 模型 | CF ↓ % | FF ↓ % | I ↓ %，19/23 父 | 有合法 C/F 共边的请求 | 方向合法路径可达 free 面积 % |
|---|---|---:|---:|---:|---:|---:|
| RAW | B | 9.794 | 86.231 | 8.168 | 0/46 | 0.000 |
| RAW | S | 7.657 | 88.264 | 10.985 | 42/46 | 11.464 |
| BASE | B | 7.179 | 15.709 | 4.215 | 18/46 | 6.515 |
| BASE | S | 5.245 | 22.506 | 7.513 | 42/46 | 21.450 |
| FINAL | B | 7.047 | 13.056 | 4.247 | 18/46 | 8.165 |
| FINAL | S | 5.130 | 18.994 | 7.588 | 42/46 | 24.111 |

FINAL 的 S−B：CF **−1.917 pp**，FF **+5.938 pp**，I **+3.341 pp**。合法 C/F 边均值从 0.739 增至 12.848；小角 `<5°` 从 45.406% 到 45.732%，非流形边均值从 1.478 到 1.717。全部字段仍需共同看，不能只用接口增加抵消 FF/I 代价。

| FINAL 分组 | CF B→S % | FF B→S % | I B→S % | 三轴交集 B→S |
|---|---:|---:|---:|---:|
| OLD，8 父 | 6.276→1.488 | 9.182→12.449 | 1.486→3.635 | 10/16→7/16 |
| FOCUS，2 父 | 0→0 | 2.174→5.424 | 0→0 | 0/4→0/4 |
| DEV，4 父 | 6.039→7.268 | 11.494→23.800 | 11.719→11.127 | 1/8→1/8 |
| HOLDOUT，1 父 | 19.719→9.210 | 9.078→26.959 | 10.107→38.159 | 0/2→0/2 |
| HIGH_TRAIN，2 父 | 0→0.697 | 1.584→3.995 | NA→NA | 4/4→4/4 |
| HIGH_HOLDOUT，2 父 | 20.748→20.789 | 12.481→20.124 | 4.553→14.014 | 2/4→2/4 |
| NEW_WIDE_TRAIN，4 父 | 6.623→6.207 | 34.823→39.006 | 1.355→2.197（2/4 父） | 4/8→4/8 |

新增 C/F 边是实测正结果，但有一部分由 seam 的固定底边直接建立，必须进一步看能否进入 bulk。S 的 42 个开放 C 请求全部有合法 C/F 接口；另外四份闭合 C 不要求接口，不能当作失败。FINAL 的 `seam→bulk ≥30%` 描述计数只有 **5/42 适用请求**：OLD3/16、FOCUS2/4，其余组为0。DEV8份的合法 seam→bulk 可达面积均为0。S 的适用请求 bulk 可达面积均值为10.706%，这与包含 seam 自身面积的全 free 可达24.111%是不同量，不能互换。

主表及共同支持：[逐组原指标](results/analysis_v1/primary_parent_macro.csv)、[完整177字段](results/analysis_v1/full177_parent_macro.csv)、[逐父配对177字段](results/analysis_v1/paired177_parent_macro.csv)、[逐输出全部结果](results/per_output.csv)、[派生汇总](results/delivery_v1/summary.json)。

## 4. 真实图审与原有能力损失

三位审阅者分别逐张打开 138 页，每页两模型、双视角、两个原位置截面，共276条实际观察，记录时间、工具事件和图片身份。五轴独立，保留 UNCERTAIN；没有依据数值推标签，也没有要求恢复唯一来源款式。这是非盲 AI_PROXY，不是独立人类认证。

“三轴交集”仅指 `REASONABLE ∧ PLAUSIBLE ∧ PRESENT`，是已有标签的描述性交集，不是新增正式指标，也不要求碎片、相交或重层为零。

| 阶段 | B | S | 新增交集 | 丢失交集 |
|---|---:|---:|---:|---:|
| RAW | 32/46 | 29/46 | 0 | 3 |
| BASE | 21/46 | 18/46 | 0 | 3 |
| FINAL | 21/46 | 18/46 | 0 | 3 |

三阶段失去的身份均为 `4cd23a/n1`、`6b894c/n1`、`91e169/n0`。对应可见现象分别是：灰色高座层替代原蓝 C；L 形输入旁长出不匹配的高背椅和大孔灰座；大块橙色侧壁留下三角形缺口。不能因 C/F 接口存在就忽略这些主体问题。

FINAL 单轴计数 B→S：主体不完整19→21，合理采用31→29，支撑缺失或损伤18→17，可见碎片／缝隙40→42，重层／折叠16→19；C 使用不确定5→6、重层不确定1→5。单轴局部收益与交集退步同时存在。

仍保留以下有限正例与共同失败：

- 838、943 的两噪声主体组织和蓝座面用途保持；abf 的完整主支撑也保住。它们支持原 SEAM 可以在部分条件上工作，但没有新增三轴正例。
- 459 的 S 生成了更宽的端板，POST 比 B 的细腿损失有局部改善；灰色替代座层与蓝 C 分离仍在，不算完整控制成功。
- 7b 的薄条侧面及端部连接、9e 的曲面端撑、bcb 的圆形端部在两模型 POST 中均有明显损失；FOCUS a8/27 在 RAW 的细腿框完整，POST 仍严重丢失。SEAM 没有解决这项共同后端问题。
- 756 仍有高于 C 的灰座层，945 仍可见灰条座层和橙色片；77 的闭合厚垫内灰条性质不确定，保留 UNCERTAIN，未按无接口判失败。

图审原件：[逐格记录](visual_review/summary_v1/actual_observations.csv)、[逐请求五轴转移](visual_review/summary_v1/paired_transitions.csv)、[三轴转移](results/delivery_v1/visual_joint_transitions.csv)、[图册](gallery/index.html)。

## 5. 来源布局诊断的解释边界

23 条 C 的生产头只调用21次，另外两条为闭合 C；每 C 两噪声共享同一预测布局。生产缓存共有553条开放边，预测335条活动边。仅在生成完成后的来源对照中，526条来源owner内有319条被激活、207条未激活；27条来源Null中有16条被激活。全部当前样本的来源alias计数为0。

这说明历史头与来源布局仍有明显差异，但来源布局只是参考完成的分配，不是开放生成唯一正确布局。本轮未校准阈值，也未做来源oracle采样。不能把本轮所有质量代价直接归因为漏判，更不能声称已经排除了历史头的限制。

典型生产活动数：a8 38/39开放边、27 10/31、28b9 10/20、abf 78/88、af 14/100；617/77均为闭合 C 的0活动。每边身份与来源参考分开保存于 [逐条件](layout_diagnostics/per_condition.csv)、[逐边](layout_diagnostics/per_edge.csv)、[只读诊断收据](layout_diagnostics/complete.json)。

## 6. 删面、面积与训练曲线

FINAL/RAW free面积比先按请求求比、父内平均，再跨父平均：B58.773%、S65.219%。删除原生成槽对应的 RAW 参考面积占比从31.145%降到25.237%；原生成槽保留比例33.608%→37.981%，平均删除原槽216.413→207.304。实际 FINAL free面积均值21.339→25.154模型单位平方。

S保留的面积和面数更多是有限的几何事实，不能直接认证主体更完整。上述面积比混合了 D0 位移、GEO合并删面和新增桥面影响；相同删除槽在 pre-GEO 坐标下的面积另记，不能把全部面积变化归因于删面，更不是来源 GT 表面保留率。实际图审已给出“全局留存更多但某些腿框仍损伤”的反例。

[逐槽去向](results/analysis_v1/deletion_fate_per_slot.csv)、[逐输出面积](results/analysis_v1/deletion_area_per_output.csv)、[分组面积汇总](results/delivery_v1/area.csv)保留全部阶段和身份。

训练记录整体FM、bulk角点速度MSE、seam第三点速度MSE及同次前向一步外推终点MSE。早期50步分箱的tip速度MSE为0.435，后期约0.23且有波动，说明新表示在学习，但不能据此证明充分收敛或泛化。损失受自由坐标组成和采样混合影响，绝对值不直接与Native FM排名。实际曲线与CSV见 [loss图](closeout/loss_curve.png)、[分箱CSV](closeout/loss_curve.csv)、[原分箱记录](closeout/loss_bins.json)。

## 7. 实际预算、资源与交付

- 训练1,500更新、12,000前向、12,000反向；无头训练、额外probe、失败、pending或重放。
- 活动墙钟2,399.23秒（39分59秒），逐更新耗时合计2,197.09秒。与另一条授权训练及用户进程并行，不是隔离吞吐基准。
- 整卡观测峰值12,157MiB，最低余量3,823MiB，平均利用率55.75%；这些含并发任务。PyTorch本worker峰值allocated3.170GB、reserved3.358GB，不冒充完整进程显存。
- 采样46RAW／2,300generator前向＋21头前向，183.38秒；单次D0共46前向／46GEO，55.45秒。RAW评分46份，POST评分92份；与B缓存138行合成276行。CPU评分raw124.08秒、post585.61秒，最多3workers；图册3workers、22.56秒。没有增加噪声或重新评分科学输出。
- 收口期间一次命令传入不支持的 `--require-visual` 参数，argparse在执行前退出，模型／评分调用均为0；随后使用正确 `closeout evaluation` 完成核验。这是工具调用修正，不是科学任务重放。

最终 [generator_step1500.pt](training/SEAM1500/generator_step1500.pt) SHA256：`3161771ace3ba5a1ae28aecca3cee5accb5a47e5a62127da91f9134d69af99e1`。匹配 [完整恢复checkpoint](training/SEAM1500/checkpoint_step1500.pt) SHA256：`dfb6a6e831cad0694f6f17188c4a7eff5f6b6ae407cafbbb135c0ca3be788097`。官方资产与源 B 身份未变。

[恢复入口](RUNBOOK.md)只核验已完成阶段，不重新派发训练或采样；实际全预算收口见 [evaluation.json](closeout/evaluation.json)、[training.json](closeout/training.json)、[资源收据](closeout/resources.json)、[276格图审完成](visual_review/summary_v1/completion.json)。本轮没有切默认、清理旧资产、commit或push。

**结论：原版 SEAM 在新的窄池 Native 起点上重现了 C 侧接续收益，但尚未把它转换成稳定的 bulk 接续与整体质量提升。** 保留这一机制正证据；本版1500步终点不替代B。新的训练、占用头调整或后端改动都不由本报告自动启动。
