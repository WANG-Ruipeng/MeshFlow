# 任务验收缺口审计（训练进行中）

审计范围：`TASK_REQUEST.txt` 全文、`protocol.json`、实际教师/预检/训练收据，以及当前训练、独立导出、采样、评分、汇总和图库源码。只读，不加载权重、不执行模型、教师、评分或渲染。未查看任何本轮候选 DEV 结果。以下是阶段性验收，不能替代最终完成证明。

快照时间为 2026-10-08 05:30 UTC 附近；C 持续训练，逐文件读取不是全局原子快照。该时点 A/B 各 COMPLETE 1000；C 的持久 status 到 110 步且仍运行。已完成教师与预检。`analysis/results.json` 仅 32 个 START 缓存 PASS、160 个待完成，状态 INCOMPLETE，正确保留固定分母。

## 已实现且有实际证据

| 合同要求 | 当前证据 | 验收边界 |
|---|---|---|
| 相同 START、保留旧 AdamW | `training/{A,B,C}/start_import_audit.json`；预检记录 194 张量、130401155 参数、原模型相同、AdamW step5000 完整恢复 | 本审计读取收据，不重复加载大权重；终点独立训练审计由另一审计器负责 |
| B/C 同头、同容量、同初始状态，仅 token 域不同 | `tests/context_alignment_cpu.json` 9 测试 PASS；`preflight/preflight.json` PASS，两个样本各 B/C message hash 相同、头初始化相同、3934720 参数 | 已有真实预检；不能用梯度存在代替控制收益 |
| 第4层原 attention 输出只读 capture、真实梯度 | 预检 `capture_site`、`generator_prediction_bitwise_equal`、`FM_bitwise_equal`、`direct_unselected_gradient_zero` 和早层非零辅助梯度；`context_trainer.py:89–140` | 辅助梯度不到第5–12层符合设计，不是断图 |
| 仅 alpha1 对齐；原 batch8、4+4、原 FM 分母 | `context_trainer.py:65–75,94–109,141–142,252–255`，实际日志 A 每步0、B 每步4个 aligned；全 batch free scalar 分母显式核验 | C 全程及三臂8000逐例配对待最终训练审计 |
| 独立 clip 与优化器状态 | `context_trainer.py:150–169` 分开 generator/head clip；A/B 末行旧步6000，B 头步1000；真实 generator update 已记录在1/100/500/1000，头每步记录 | 四个采样步的更新量不能称1000步逐步全覆盖 |
| 固定可信教师与训练目标 | `teacher_manifest.json` COMPLETE：32 unique meshes、128 images、8 batches；真实官方 vision pooler；资产、revision、processor、几何和源文件 hash 绑定 | 教师全局语义、常量均值基准不是 SIE 或可用性评估 |
| 固定训练相机、实际接触表检查 | `teacher/render_manifest.json`、`render_review.json`、`teacher_render_contact_sheet.png`；32个实际训练几何，4固定视角、灰材质、训练总体画幅×1.10 | root 已实际看接触表；本审计未重复声称新看图 |
| 新噪声与缓存身份 | `data/evaluation/preparation_receipt.json` PASS；48固定请求=P1 OLD16+NEW16+P2 OPEN16；`noise_identity_audit.json` 保存 n0…n5 对照和新6/7无碰撞 | NEW 为已见 DEV 父的新噪声，不是新父确认 |
| 旧 C 与 Gaussian、51状态、NPZ/PLY 保持 | `baseline_cache_manifest.json` 32/32 条所有7个 contracts 为 true：C output、C all51、N、finite、Gaussian、NPZ C、PLY C | 仅证明32缓存；160新输出必须最终逐项验收 |
| 新版严格训练/纯生成器 schema | `context_checkpoint.py:111–134,155–194,202–233` 保存 generator/head/optimizer/RNG/cursor，strict=True 导入；generator export 明确新 schema，无 head/teacher | 已实现；实际三份 export、capture on/off 及独立 Euler50 仍待执行 |
| 不自动改默认模型/择500/后处理/读CONFIRM | 协议、总入口和方法文件一致；`run_contextual_alignment.py:58–81` 只生成1000端点；采样从注册bank取 C,z | 原500恢复权重存在不代表允许500生成 |

## 图库和参考相机：静态通过，实际图像验收待完成

`tools/contextual_alignment_gallery.py:13–18` 要求全部192个固定输出齐备且 RAW hash 正确，缺失时拒绝冻结完整图库。第31–39行对同一 condition 的四标签、四噪声和 P1 来源参考共同求 bounds；第45与49行分别画无来源页和来源页，并传入相同 bounds。参考列明确使用 `C + GT_free`，与模型列同蓝 C / 橙 free。P2不造GT列。

`evaluation/render.py:19–35` 固定四视角+wire、公共中心/跨度（×1.08余量）、同材质和轴变换 `(x,z,y)`；无坐标修补、删面、焊接或平滑。异常远片进入公共 bounds，不会被有意裁走。方法排列由固定命名空间 SHA 决定，独立于结果；匿名映射单独保存。

当前仅有 `gallery/method_mapping.json`，还没有完整相机manifest与实际48无来源页/32参考页。因此当前结论只能是“实现符合”，不能写“实际所有图已核验”。最终必须检查：192模型输出全覆盖；48页先无来源观察、P1再看32参考页；图片SHA与最终相机绑定；每输出192行 AI_VISUAL_REVIEW；不确定细节如实写不确定，human usability=NOT_RUN。四视角和wire是不同几何观察，不是额外独立样本。

## 指标定义、资格与分母

- `evaluation/evaluate.py:12–20` 独立验证 inference bank 只含推理输入，拒绝 GT/Gamma/L0/source IDs 进入该bank。来源参考只在评分路径读取。
- P1 S沿用 `control.py:194–236`：原可信 Gamma、长度权重、有限边/切向/投影及原CF资格；输出 `SeamCov_gamma_total_length`、不确定比例和上下界。不能解释为逐边焊接或一般任意C成功率。
- I沿用 `control.py:239–279` 的C采样面积加权并集；cross/dup/intersection均保留。分母为原C全部采样权重，不是只用内部C或受影响free面数。GT差值另存，不替代主I。
- E沿用 `control.py:361–382`：主值分母是确定参考外露域，同时保留可能外露域和全域 conservative lower/upper；未知法向扩展到最宽界，不当无遮挡。
- `evaluate.py:85–101` 原C/GT查询从冻结参考加载；每输出 Q_F2048 用同 condition/candidate RNG 重映射到新预测面。`evaluate.py:106–121` 对P2保留I/几何，S/E/来源距离为NA。
- `contextual_alignment_analysis.py:17,20–29,97–107,113–134` 缺失噪声不缩分母；先父内均值、再父等权；ALL8父×4噪声，OLD/NEW各8×2；差区间是 `[cand_low-ref_high,cand_high-ref_low]`。E筛选使用global资格界，PARTIAL不是自动FAIL。
- `analysis.py:54–76` 六项预登记筛选方向与阈值一致；第135–151行保留超过10pp的父代价和逐输出极值，第132–134行LOPO仅诊断。16项语义测试和不完整图库拒绝收据PASS；三个适配源码与测试记录SHA当前相同。
- `sie_summary.csv` 当前保留fraction单位。最终主报告应明确×100显示百分数、差值以百分点表述；不能把检测上下界写成置信区间。主表S/I/E和形状/成本/多样性应分开。

## 调用与资源：当前合法，最终闭合未完成

05:30附近账本前缀：teacher image128/forward8全部RETURN；真实预检6F/10B，0更新；A/B各8000F/8000B/1000更新，B头4000F/4000参与combined backward，额外梯度观察6次；C运行中。已读前缀没有ERROR。一个训练microbatch可能在F和B之间，不能以活跃前缀要求F=B，也不能把正在运行的owner当泄漏。

独立导出前向开/关capture检查还计划每臂2F，即总额外6F；和当前预检相加12F/10B，仍在24F/16B内。正式新采样预期160×50=8000 F；32个START缓存不计新rollout。头反传是同一次combined backward的嵌套记账，不能与生成器反传相加成32000次独立autograd。

`runtime.py:137–161` 单共享锁，启动前实测资源；累计deadline在每次Budget.reserve前检查，6小时包含加载/保存/CPU等待。已完成owner时长约3072.58秒，另有活动C owner。设备总显存16303MiB，要求余量2445.45MiB。已读5秒监控中最小free：teacher12890、preflight11868、A9803、B9724、C当前9843MiB，均满足；这只是离散设备采样，不是瞬时峰值保证。A/B allocator reserved最大分别3990880256/4093640704bytes，不等同整进程显存。原日志/资源读数必须保留测量scope。A含CPU审计并行，臂wall不是隔离性能基准。

发现并已通知root的汇总工程问题：最初 `audit/final_accounting.py` 用 `{phase:seconds}` 会覆盖同名恢复阶段。root已确认改成累加，并核 START/END id唯一性。当前实际phase无重复，未影响本次已读时长。另一注意：rollout RESERVE(kind=rollout_attempt) 的完成事件是无kind `ROLLOUT_COMPLETE`，不能按普通RETURN误报160个未闭合；teacher_image则有普通RETURN。最终按实际context逐项闭合；独立纯CPU检查器正在准备。

当前采样源码本身验证C逐51状态字节相等并导出；评分 `evaluate.py:43–47` 对缺失trajectory可能产生None，而INVALID_OUTPUT只拒绝False。由正式采样器产生的本轮输出应字段齐全，但最终验收必须要求7项contracts每项恰为True，不能只看评分状态PASS。该项为最终证据要求，不是已观察到的输出失败。

## 尚未完成的最终交付

| 交付/工作 | 当前状态 | 最终完成条件 |
|---|---|---|
| C训练1000、三臂完整同流审核 | PENDING，正常运行 | 3000 updates；每臂8000真实F/B；共8000 head F/B参与、12稀疏反传；输入配对、旧/新AdamW、真实更新记录一致 |
| 纯生成器推理导出 | IMPLEMENTED / NOT_RUN | 3份严格新schema导出、独立进程加载、不引入头/教师，capture on/off实际逐位一致 |
| 全192输出 | PENDING | 新160完整Euler50+旧32可信缓存；逐例51状态C、Gaussian、NPZ/PLY身份和实际50NFE |
| 全量SIE/质量/多样性 | PARTIAL | 当前32缓存+160待定；192评分完成及288固定free候选配对距离，不因缺失删行 |
| 全图库/实际视觉审阅 | NOT_RUN | 48无来源页+32来源页，192逐输出真实AI观察；不生成human可用率 |
| `EXPERIMENT_REPORT.md` | MISSING | 首先真实账本，OLD/NEW/ALL SIE、全部六对比、父尾部与噪声、独立形状/多样性/成本，任务8问题逐答 |
| `checkpoint_manifest.json` | MISSING | START与3臂500/1000恢复权重、3纯推理导出的schema/用途/路径/hash，不混原START身份 |
| `visual_review.csv` / `OPEN_GALLERY.html` | MISSING | 完整真实视觉记录与最终图库入口 |
| `protocol.json`, teacher manifest/contact,方法、恢复说明、code.diff | PRESENT | 已有；最终报告引用真实执行结果，核冻结源/输入前后hash |
| training日志与四张CSV | PRESENT / PARTIAL | 实际在 `training/{A,B,C}/`，最终索引说明其对应合同 `training_logs/`；CSV完整行集不等于192已完成，须按status解释 |

本审计未发现需要改变科学配方或停止当前合法阶段的硬问题。后续应继续既定1000终点和固定评价，不因辅助loss下降提前下控制结论，不补新候选/噪声/阈值。完成固定任务后停止；默认仍不自动改变。