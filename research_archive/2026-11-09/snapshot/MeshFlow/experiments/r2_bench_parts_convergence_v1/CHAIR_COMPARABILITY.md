# Bench 多部位训练与 Chair SEAM 的可比性合同

核查日期：2026-10-18。本文只读取历史报告、登记、训练源码及检查点旁车元数据；没有重新训练、模型前向、生成或重评分，也没有据此加载大权重。下述历史结果不是本轮 Bench 的结果。

本轮可以沿用 Chair 的架构、优化定义和评测方法，形成同方法的跨类别适配证据。历史 Chair 与新 Bench 的类别先验、数据分布、条件类型和训练谱系不同，因此不能称为“其余变量完全相同、只改变类别”的单变量实验。若本轮有同起点的 Bench PLAIN/SEAM 对照，该内部配对与历史 Chair 的外部参照应分开解释。

用户随后明确要求从官方Bench权重从零开始条件训练。当前执行设计据此为：官方Bench EMA＋fresh AdamW，先完成普通Native条件训练12000步；从同一完整终点分叉，普通Native续训与SEAM适配共享数据流和相同阶段更新预算，SEAM阶段最少3000步，每1000步固定probe，预登记最多新增9000步。平台判断要求至少四个检查点跨三个间隔；共同停止规则不得让两臂各挑不同best。以上是新Bench控制变量设计，**不会复制Chair曾经先67父混合、再47父完整座面、再四C小池的历史数据课程**。具体最终数字仍由本轮正式registration确认；此文不自行派发任何GPU工作。

## 1. 不要把 PLAIN_S2000 误读为从官方只训练了 2000 步

PLAIN_S2000 中的 PLAIN 是“不加 couple 辅助项的 SEAM”，不是无 SEAM 的普通 Native 模型。

| 真实谱系 | 数据与本阶段更新 | 生成器累计更新 |
|---|---|---:|
| 官方 Chair EMA → EXP67 | USER67，fresh AdamW，6000步 | 6000 |
| EXP67 → FULL_S6000 | TRAIN47 完整座面，恢复匹配 AdamW，新增6000步 | 12000 |
| FULL_S6000 → SEAM_SLOT_C200 | 四条熟悉 TRAIN 完整座面，新增200步 | 12200 |
| SEAM200 → B_POOL47 | TRAIN47，新增800步 | 13000 |
| B_POOL47 → PLAIN_S2000 | TRAIN47普通SEAM续训，新增2000步 | **15000** |

PLAIN_S2000 的 SEAM 表示累计3000步，之前还有12000步条件训练。LONG 是另一条分支：同 SEAM200 起点、四条 C 再训练800步，终点 global13000、SEAM累计1000；LONG 不是 PLAIN_S2000 的直接生成器父检查点。后者推理使用的却是 LONG 已缓存的 C-only 预测布局，这两种谱系身份要分开。

PLAIN_S2000 的实际完整恢复点：

`experiments/r2_seam_pool47_couple_v1/checkpoints/PLAIN3000_step2000.pt`

文件SHA256：`ea6675a8d77364936fb13443ea0378011423404ee28a8f3074b7f8bd4ef3f56a`。

对应推理导出：

`experiments/r2_seam_quality_wave1_wave2_v1/inference/PLAIN_S2000_generator.pt`

文件SHA256：`46be9a29d6d9e150205cc46cf8c952ae1cd32b2aa58bff24cb879eb451932f43`。

两者旁车记录相同 generator_state_sha256：`6f27fc5828fe4de295d102dea258616d6a7059a1a2dddc87bb747b42d03ca5a8`，local2000/global15000。本核查引用已保存SHA及旁车，不冒称重新计算了大权重SHA。

## 2. 可复用的训练与后处理定义

- Native/T1＋role＋Geo；SEAM使用已有角点锚定适配，不加入 couple、局部噪声、反覆盖或新奖励。
- 生成器130,401,155参数；所有生成器参数/AdamW状态FP32，BF16前向，模型保持既有eval-mode训练合同。
- AdamW：lr `1e-5`，betas `(.9,.95)`，eps `1e-8`，weight decay `0`，梯度裁剪范数 `1`。
- batch8、microbatch1；历史每批4个父，每父1个原宽度实例＋1个x宽度U(.9,1.1)增强实例。多部位池应先抽父、再选条件，记录父与part曝光，不能让可构造条件更多的父自然获得更多训练权重。
- `t=sigmoid(N(0,1))`；来源角色约束OT后使用HYBRID0.25。未改变原pre-OT与模型坐标倍率。
- C全部角点固定；激活seam前两个角点共享C端点，只监督第三点；bulk监督全部自由角。FM分母为**整个有效batch的自由标量总数**，不是每实例MSE的等权平均。
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` 在CUDA初始化前设置；严格确定性、TF32关闭、SDPA math。固定条件/噪声身份与可恢复AdamW/RNG逐阶段保存。
- PLAIN_S2000 的额外2000步中占用头冻结且未调用；训练使用来源owner监督，推理使用LONG保存的C-only预测布局。Bench新增部位不能把这些Chair缓存布局当成可迁移标签，更不能使用评测来源Y产生推理owner。
- 本轮Bench明确沿用旧Bench的833参数C-only占用头及同一balanced BCE，在SEAM适配阶段共同持续训练，不新增头结构。它与Chair后期冻结LONG头/布局的政策不同，必须单独披露；SEAM与Native的内部对照会包含SEAM所需离散布局机制，不能把head更新假装成完全相同的训练组件。
- 推理Euler50、标准Gaussian；单次类别匹配官方D0，恢复C和seam锚点，原GEO_C，恢复来源C角序并固定C整理free方向。原POST与新增CPU后处理分别保存和评分。
- Bench必须用官方Bench D0：其记录的fixed_std=.05，Chair为.02。不得把不同类别去噪器当相同权重；相同的是处理流程和显式合同。当前Bench解码并未据fixed_std再抽一份噪声。
- 若比较HINGE，应两边统一相同版本、1.5%L和反向绕序守卫；RAW、原POST和HINGE后指标均保留，不以最终后处理列替代整个过程。

直接源码定位：

| 文件 | 相关位置 |
|---|---|
| `r2/pilots/seam_pool47_couple_v1/protocol.py` | 第28—40行：来源、训练、保存与评测合同 |
| `r2/pilots/seam_pool47_couple_v1/training.py` | `update`：SEAM前向、全batch自由标量分母、clip与AdamW |
| `r2/src/meshflow_control/training/schedule40.py` | `make_optimizer` / `validate_optimizer`：继承的优化器定义 |
| `r2/src/meshflow_control/seam/slot_data.py` | `supervised_sample`：OT、混噪、角点掩码和分母 |
| `r2/pilots/seam_abc_v1/slot_data_extension.py` | 多边owner/alias的实际监督扩展 |
| `r2/src/meshflow_control/runtime.py` | `configure_stable_runtime` / `stable_execution`：精度与确定性 |
| `r2/pilots/seam_slot_c200_v1/prepare.py` | 第64行起：FULL起点及四C原始SEAM合同 |

## 3. Chair 与旧 Bench 都没有被认证已经收敛

历史训练以预定固定预算结束，不按平台期停止。训练loss下降与生成质量/接续改善是不同证据，不能把任何一个MSE阈值写成模型已经充分收敛。

| 训练曲线 | 前100 → 后100更新 |
|---|---:|
| Chair PLAIN3000 总FM | .121060 → .104003 |
| Chair PLAIN3000 seam第三点速度MSE | .173507 → .032102 |
| Chair PLAIN3000 bulk速度MSE | .119316 → .105050 |
| 旧Bench SEAM3000 总FM | .142843 → .100600 |
| 旧Bench SEAM3000 seam第三点速度MSE | .684796 → .213787 |
| 旧Bench SEAM3000 bulk速度MSE | .137997 → .099846 |

这些是不同正常训练实例窗口上的统计，不能证明固定输入的单调收敛。Chair LONG 的 `.07088→.01633` 只来自 `t∈[.4,.6)` 的首/末50更新，不得与上表全t统计直接比较。

在Chair中，更低的训练接口MSE也没有自动形成DEV接续升级：PLAIN3000与COUPLE3000原POST DEV bulk≥30%均0/12；训练DEV FF还高于同布局B起点。因此“训练到平台”只回答拟合状态，最终仍需完整RAW/POST质量和实际图审。

## 4. 本轮有限平台判断的独立实现

`r2/pilots/bench_parts_convergence_v1/convergence.py` 是纯CPU统计模块。最少阶段更新数、最大阶段预算和阶段起始step均由正式登记传入，不在代码中暗设训练长度。

- 最近四个**同一固定probe manifest**的检查点构成三个连续间隔。
- `probe_interval`也由正式登记传入；检查点间隔不匹配时拒绝把它们拼成所登记的学习曲线。
- TRAIN `ALL_free`、`bulk`、`seam_tip` 各有定义项，在三个间隔内的相对变动绝对值必须严格小于2%。下降超过2%也说明还在学习，不能判平台。
- 完全没有对应自由坐标的可选分区记NA并跳过。部分检查点缺失分区不能证明三个间隔稳定；ALL_free必须有定义。固定probe有效标量数改变会报错，不能悄悄换分母。
- 预先指定的大part检查ALL MSE。默认连续三个间隔每次恶化严格超过5%时阻止平台；该连续间隔数是显式可登记参数，不在看完结果后修改。
- DEV只输出恶化警示，不用于选best、停止、改阈值或自动晋级。平台状态即使伴随DEV恶化也必须完整披露。
- 输出仅为 `PLATEAU`、`KEEP_TRAINING` 或 `BUDGET_EXHAUSTED_NOT_CONVERGED`。到预算上限却没平台，明确报告未观察收敛，不自动追加预算。
- `PLATEAU` 是工程上的有限loss平台，不是数学收敛证明，不等于视觉/接续质量改善。

训练日志按SSE与有效标量数聚合：`sum(SSE)/sum(count)`；按part、t分桶和split保留分母。绝不将样本MSE平均冒充标量加权值。平台probe与变化训练窗口统计必须独立标识。

## 5. 本轮必须保留的跨类别边界

新池是277 TRAIN条件、39 DEV条件，仍为64/8个独立父；Chair主线47条条件均为完整座面。类别专用官方先验、父数量、条件类型、K/面积/闭合拓扑、训练谱系和D0类别均不同。可以称“同方法的跨类别、多部位条件适配”，不能称严格类别单变量消融或Chair权重零样本泛化。

316条数据中49条闭合C是合法输入；不能为了现有入口而人为切口或默默删除，再称训练了完整数据池。其余alias/第三点引用C的输入应按真实实现支持范围登记。来源标签仅用于训练/诊断；部位名字只用于抽样和分组，不能悄悄作为新的语义模型输入。

若要独立识别多部位数据的作用，需要同一Bench起点、相同更新及父曝光政策的旧/新条件对照；历史Chair不能代替它。本文不额外授权或新增实验臂，实际预算和终点以本轮正式登记为准。

主要只读来源：`experiments/r2_seam_slot_c200_v1/REPORT.md`、`experiments/r2_seam_quality_four_v1/REPORT.md`、`experiments/r2_seam_abc_v1/REPORT.md`、`experiments/r2_seam_pool47_couple_v1/REPORT.md`、`experiments/r2_seam_quality_wave1_wave2_v1/REPORT.md`、`experiments/r2_bench_control_transfer_v1/REPORT.md`、`experiments/r2_bench_parts_v1/REPORT.md`、`WORKSPACE.md`、`r2/docs/research_route_memory.md`及`research_route_registry.json`。

实现辅助：`controller.py` 默认只读status，给出共同里程碑决策和资源派发计划，不启动GPU；只有两臂在同预算点同时达到有限loss平台才允许提前终止该阶段。`curves.py` 读取真实逐更新日志、曝光SSE/count及固定probe，分别导出loss.csv、按part/t统计、loss.png和progress.json。新快照目录写入，不静默覆盖历史曲线。
