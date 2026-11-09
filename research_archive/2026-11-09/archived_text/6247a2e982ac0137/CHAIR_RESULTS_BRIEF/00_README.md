# Chair 几何控制实验：精简结果包

给 GPT 阅读时，先读本文件和 `CHAIR_DOMAIN_CONTROL_REPORT.md`，再查下面的结果表。**本包只保留核心结果，不包含原始数据、逐例 raw/score、源码、权重或完整日志。** 主报告保留原文，其中部分指向完整证据的链接未随本精简包提供。

实验已实际完成并停止：Native、Native+Geo、Native+Geo+Joint 三臂分别从 step1000 新增 3000 次更新，累计 4000 步；合计 9000 次更新。完成 120 条起点、48 条中间验证、414 条终点生成，共 582 条，重复 0 条。终点有 18 个几何不适用格，保留 NA。teacher 288 次，总模型前向 101415 次；解析参照不计作模型生成。

主要发现：

- **Geo 有局部覆盖收益，但未证明整体最优。** 相对 Native，训练20%/训练40%/验证20%/验证40%的覆盖分别增加约 1.28/6.14/2.92/2.12 个百分点；RMS 仅训练40%均值改善，其余三格变差。
- **验证父的宽度控制弱。** Native/Geo/Joint 的验证斜率均值约 0.121/0.081/0.061，理想值为 1。Geo 第一噪声配对宽度误差在训练父从 0.05774 降至 0.04886 L0，在验证父从 0.13946 增至 0.15642 L0。没有整体优于 Post-scale 的证据。
- **Joint 没有保住覆盖。** 相对 Geo，A 四格覆盖均下降；验证40%的接口边 RMS 改善伴随相交恶化。
- **留出回归有相交退化。** 新4000相对源1000，Native/Geo/Joint 的 FF 受影响自由面比例分别增加约 1.57/1.39/1.68 个百分点，不能因 RMS 改善就写成“无回归”。

| 文件 | 看什么 |
|---|---|
| `CHAIR_DOMAIN_CONTROL_REPORT.md` | 完整实验说明、正式结果和限制 |
| `A_context_summary.csv` | 新上下文补全 |
| `B_width_summary.csv` | 宽度控制 |
| `C_holdout_paired_summary.csv` | 原留出起终点回归 |
| `matched_budget_summary.csv` | Geo−Native 与 Joint−Geo 匹配预算增量 |
| `B_model_minus_analytic_summary.csv` | 模型对不响应、直接后拉伸解析参照 |
| `B_width_start_endpoint_summary.csv` | 宽度的真实起终点配对变化 |
| `shared_start_endpoint_summary.csv` | 共有输入的起终点变化 |
| `figures/shared_input_generation_curves.png` | 实测生成曲线 |
| `figures/fixed_teacher_curves.png` | 固定 teacher 曲线 |
| `checkpoint_manifest.json` | checkpoint 身份、步数与登记 SHA256 |
| `execution_accounting.json` | 实际更新、生成和调用总账 |
| `FILES_SHA256.json` | 本精简包文件清单及校验值 |

注意：代码名 T1/Geo/Full 对应 Native/Native+Geo/Native+Geo+Joint。终点是 `phase3000_total4000`，`latest` 是途中恢复快照。权重未附，登记哈希不等于云端已独立核验权重字节。

统计以父等权。A40 有效训练父为 7/8、验证父为 6/8；A20 均为 8/8。B 每划分仅 4 个父×2 个噪声，斜率在三个 alpha 行重复不是独立样本。起终点真实配对仅第一噪声。中间生成只测验证 A20，其他中间项为 NA。40%条件改变了 C、free 补集和有效父集合，不是纯信息量比较。单训练 seed、已观察域内验证，不支持强泛化结论。

执行中一次 I/O 中断已零重放恢复；旧 BF16 置换 FAIL 原样保留，严格水密为 NOT_RUN。阶段 PASS 只代表程序完成，不代表模型质量达标。打包没有新增任何实验调用。

可给 GPT 的指令：**请基于这些结果，独立评价控制、覆盖、接口、Geo 增量和 Joint 增量；不要预设赢家，不只比较 JEdge5。区分训练与验证、均值与配对、NA 与失败；指出证据支持与不支持的主张。未包含的原始证据请标“无法独立复核”，不要补造结果或执行新实验。**
