**交付验收补记**

总入口已实际完成 23/23 个阶段并以退出码 0 结束。三臂各新增 3000 次 optimizer 更新，累计 4000 步；合计 9000 次更新、72000 次训练微批前向与反向。正式终点实际生成 414 条（A=174、B=144、C=96），另有 120 条起点和 48 条中间验证，总计 582 条；18 个几何不适用格保留，重复生成 0 条。固定 teacher 288 次、有限预检 27 次前向，真实模型前向总数 101415/102420。432 个解析候选及 648 个跨条件评分单元均不是新模型生成。更新和模型调用已停止，没有启动下一轮或更换默认模型。

主要发现按父等权，数字与下方正式表及配对 CSV 一致：

- **补全与 Geo 增量：有局部收益，未证明整体最优。** Geo 相对 Native 在 A 的 train20/train40/validation20/validation40 覆盖分别增加 1.28/6.14/2.92/2.12 个百分点。RMS 在 train40 改善 0.608%L（6/7 有效父），其余三格均值变差。40% 条件改变了 C、free 补集和有效父集合，不能单独归因于信息量。
- **几何响应：训练父有响应，验证父很弱。** Native/Geo/Joint 的验证宽度斜率均值为 0.121/0.081/0.061，理想值为 1。Geo 的真正第一噪声配对宽度误差在训练父从 0.05774 降到 0.04886 L0，在验证父却从 0.13946 增到 0.15642 L0；斜率变化不等于绝对控制质量改善。没有整体超越 Post-scale 的证据，不能宣传复杂结构推理或任意局部编辑。
- **Joint 增量：没有保住覆盖。** 相对 Geo，A 四格覆盖均下降；验证40%条件的接口边 RMS 改善约 0.158%L，但 FF 受影响自由面增加约 1.224 个百分点。覆盖、接口、相交和角质量仍有取舍。
- **原留出回归：距离与覆盖略有收益，相交存在新增退化。** 新4000相对源1000的 FF 受影响自由面比例，Native/Geo/Joint 分别增加约 1.57/1.39/1.68 个百分点；不能把平均 RMS 下降写成没有退化。严格水密仍为 NOT_RUN。

分母说明：A40 训练父有效 7/8、验证父有效 6/8；A20 两划分均为 8/8。B 每划分仅 4 个父，终点 8 条噪声响应曲线；斜率在三个 alpha 行重复展示，不是 24 条独立响应曲线。真正起终点配对仅第一噪声。中间生成只测 8 个验证父的 A20 第一噪声，训练父及 B 的中间生成列是未测/NA；图中的两端连线不代表中间有实测。各指标有效分母以 CSV 的 metric_valid_* 字段为准。评分 PASS 只表示完成评分，不代表几何质量达标。

交付导航：

- 正式表：[A逐case](A_context_per_case.csv)、[A逐父](A_context_per_parent.csv)、[B宽度响应](B_width_response.csv)、[B跨条件](B_cross_condition_scores.csv)、[B解析参照](B_analytic_baselines.csv)、[C留出回归](C_holdout_regression.csv)。
- 配对增量：[Geo/Joint匹配预算](matched_budget_summary.csv)、[起终点共有输入](shared_start_endpoint_summary.csv)、[宽度起终点配对](B_width_start_endpoint_summary.csv)、[模型减解析参照](B_model_minus_analytic_summary.csv)。
- 曲线与固定视图：[9000步训练记录](training_learning_curves.csv)、[共有输入生成曲线](shared_input_generation_curves.csv)、[固定teacher](teacher_per_case.csv)、[58个原始图件清单](figures/manifest.json)、[A/C排版修订清单](figures/layout_v2/manifest.json)。
- 身份与执行：[checkpoint清单](checkpoint_manifest.json)、[冻结科学代码diff](code.diff)、[交付修订代码diff](delivery_only_code.diff)、[真实调用总账](execution_accounting.json)、[追加式执行账本](execution_ledger.jsonl)、[分次执行耗时](execution_attempts.csv)、[资源分账](resource_summary.csv)、[恢复命令](restart.txt)。

执行期间发生一次已恢复的文件 I/O 中断：Full 起点评价完成 40 条生成后，原子替换 ledger 文件报 PermissionError。原 40 个 raw 哈希和 2000 次采样前向计数完整保留；同一入口恢复时跳过全部已完成生成，只补 24 次 teacher 和 40 次 CPU 评分，训练/生成均未重放。证据见 [事件归档](incidents/001_atomic_replace_permission/README.md) 与 [恢复审计](incidents/001_atomic_replace_permission/recovery_audit.json)。Windows 侧读句柄冲突只是推测，未确证；后续活跃账本统一用 WSL 读取。

该阶段两次父进程外层耗时为 124.871468266 秒和 103.947449250 秒，同一范围合计 228.818917516 秒。首次 worker 内部总耗时缺失，标 NA；两次模型加载分别为 10.924566783 秒和 11.699632304 秒。第二次 worker 的 98.518799867 秒不能当作整个阶段耗时或纯恢复额外开销。sampling/teacher/加载等嵌套计时不相加。恢复时零前向 sampling 循环用时 0.079747192 秒，其记录继承原请求的 PyTorch 峰值，不能解释为新增采样显存峰值；该尝试在分次表中标注 SKIPPED_COMPLETED。

视觉验收发现原 A/C 图总标题与首行标题重叠，因此仅将子图上边界从 0.94 调到 0.88，生成 48 张排版副本；原图完整保留，56 个固定视图的登记对象、噪声、模型、相机和坐标范围均未改变。修订脚本逐项断言原始依赖与视角范围一致，模型/OT/评分调用均为 0。主科学代码的 19 个冻结文件及 code.diff 不变，额外脚本和差异单列在 delivery_revision_001/ 与 delivery_only_code.diff。已人工查看 A/C 修订样例、B 宽度图和两张曲线，标题重叠已消除。

本轮实际读取的协议文件是唯一存在的 `C:\Users\warpwang\Downloads\Chair_Domain_Triangle_Control_Codex_Prompt.md`，归档于 [requested_protocol.md](registration/requested_protocol.md)，SHA256 为 `135c5dc2fb44f64dae76def4cbd5b40c06906229d9c3ae9065c2a888e6bf3bfc`。

公共代码、原数据、旧模型和既有结果的保护核验见 [protected_assets_after.json](protected_assets_after.json)。旧 BF16 置换 FAIL 保持原结论，未重新跑历史矩阵，也未补造历史曲线。此次为单训练 seed、已观察域内开发性验证。
