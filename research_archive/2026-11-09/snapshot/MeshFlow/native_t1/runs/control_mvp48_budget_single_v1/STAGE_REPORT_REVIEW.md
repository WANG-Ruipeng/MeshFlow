# 阶段性报告只读审查

审查范围：读取现存冻结 panels、代码、已完成 ledger 与 score JSON；未运行 GPU、未改变科学输入或已冻结代码、未改写现有结果/CSV。快照时间为 Unix 1790964782.737914。此文是阶段审查，不是最终实验结论。

## 已完成事实与结果新鲜度

本次读取确认 7 个曲线状态完成：Native0、T1_500、Geo_shared_500、T1_1000、Geo_1000、Full_1000、T1_2000。每状态完整生成 64/64、teacher 48/48；训练与验证各 8 父对象、32 请求、32 有效评分。4 个替换状态 Native0/T1_1000/Geo_1000/Full_1000 各 16/16 完成且有有效评分。共 512 条新增完整生成、336 次 teacher、25,600 次生成前向；这不包含之后继续执行的工作。当前调度可继续，审查未发现要求停止的实现错误。

根目录 CSV 仍是正式运行前 CPU 报告检查快照，时间约 Unix 1790959716；ALL_IN_ONE_REPORT.md 为总入口 RUNNING 占位。最终必须在全部阶段结束后重新调用冻结 reporting，再进行最终六问分析。当前事实应读取 evaluation/{curve,swap}/{model_id}/ledger.json 与其 scores/，不能把旧 CSV 的 NA 解读为本次任务未执行。

## 聚合及分母

- 每个主曲线状态：16 父对象 × 2C × 2 噪声 = 64；训练/验证分别 8 父对象、32 case。先父对象内平均 4 case，再父对象等权。per_case.csv 和 per_parent.csv 提供全部请求及每指标有效数。
- 每个条件替换状态：8 验证父对象 × 第一块 C × 2 噪声 = 16。应先对象内平均两份噪声，再平均 8 父对象；正确对照只取其第一块 C 的同两份噪声，不能使用主曲线含两块 C 的验证总均值替代。
- 每个 teacher 状态：16 父对象 × 1C × 1 噪声 × 3t = 48。teacher_curve_summary.csv 按 split 与相同 t 分开，每行应为 8 父对象。FM、D_edge、L5 与 Yhat 面积不应跨 t 混合后用来断言过拟合。
- 每个 fit 状态：2 父对象 × 1C × 4 噪声 = 8 生成，2 父对象 × 3t = 6 teacher；先每对象平均4噪声，再2对象等权。
- 每个留出状态：8 父对象 × 2C × 2噪声 = 32。96 旧 raw 是只读复用，不计本轮新增生成。
- completed_cases/complete_rollouts 是执行完成；score_valid_cases、每指标 *_valid_cases / *_valid_parents 才是评分有效分母；均不等于补全质量成功。当前已审查阶段两类分母均齐全。
- matched_budget_increments.csv 的差在同对象、同预算内计算，matched_budget_increment_summary.csv 再父对象等权。所有三臂主结论仍固定5000，2000目前仅T1完成，不能进行同预算三模型排名。

## 指标方向与条件优势

remainder、两个方向RMS、固定coverage RMS、Gamma表面/有限边RMS与p95、退化/非法相交比例均越低越好；coverage fraction 越高越好；角p05越高越好；面积及面积比仅描述，不能自动视为接近1即成功。

control_swap.csv 固定保存 wrong−correct：距离指标正数表示正确条件较好；coverage 指标负数表示正确条件较好。相对Native0的优势变化在距离上正向更好、coverage上负向更好。不要把全列统一解释为“越大越好”。

替换条件的 CF 字段是“接收请求原 C 与生成 free 的相交/接合诊断”，并非实际 donor C 的 CF；evaluation 详细结果明确记录此口径。不得把它称为有效 donor 补全几何质量。free 输出变化只证明响应，与正确绝对质量和相对错误条件优势合并解读。

## 已观察到的阶段数值，仅作报告核对

| 状态 | 训练剩余RMS (%L) | 验证剩余RMS (%L) | 训练coverage | 验证coverage | 训练Gamma边RMS (%L) | 验证Gamma边RMS (%L) |
|---|---:|---:|---:|---:|---:|---:|
| Native0 | 7.323271 | 5.630522 | 0.183533 | 0.290894 | 5.625115 | 3.854370 |
| T1_1000 | 7.083779 | 5.106050 | 0.208740 | 0.303040 | 6.531560 | 4.044890 |
| Geo_1000 | 7.034171 | 5.136924 | 0.202393 | 0.309998 | 6.528213 | 4.141930 |
| Full_1000 | 7.075184 | 5.149337 | 0.207642 | 0.316772 | 6.671754 | 4.105682 |
| T1_2000 | 7.058665 | 4.568923 | 0.259277 | 0.314392 | 6.005015 | 3.945652 |

T1_1000→2000 的训练/验证生成剩余RMS均下降，但FF非法受影响free比例分别0.859452→0.917463和0.924789→0.939346；不能只看单项宣布整体质量已改善。相对Native0，T1_2000的Gamma有限边RMS仍稍高，体现覆盖、接口、相交证据需要分别报告。

相同t=.5的teacher FM：训练T1_1000 0.00186987→T1_2000 0.00148678；验证0.02228425→0.04121195。这是固定同定义的teacher走势，和验证完整生成RMS下降同时存在；不能用teacher一项覆盖生成结论，也不能拿混合训练日志比例代替此比较。

| 替换状态 | 有效父/请求 | 剩余RMS wrong−correct (%L) | coverage wrong−correct | Gamma边RMS wrong−correct (%L) |
|---|---:|---:|---:|---:|
| Native0 | 8/16 | 1.713712 | -0.136963 | 1.533830 |
| T1_1000 | 8/16 | 2.772098 | -0.164307 | 2.312464 |
| Geo_1000 | 8/16 | 2.722356 | -0.170654 | 2.201503 |
| Full_1000 | 8/16 | 2.641576 | -0.162720 | 2.191261 |

Native起点已具有条件响应和正确条件相对优势，因此不能把全部1000/5000的优势归因于新增训练。上述优势必须与正确C绝对质量和相对Native增量一起呈现。5000条件对照尚不属于本次快照。

## 六个最终问题的数据路径

| 问题 | 主要产物 | 最终读取约束 |
|---|---|---|
| T1相对Native新增多少条件补全能力 | relative_initialization.csv；train_validation_learning_curves.csv；control_swap.csv | 看相同父对象/输入的5000−Native、正确C绝对质量、条件优势变化，不能仅teacher |
| 1000→5000生成及训练/验证差异 | train_validation_learning_curves.csv；per_parent.csv；teacher_curve_summary.csv | 主结论固定5000，分别列训练/验证，同t teacher辅助解释 |
| 正确C是否有用及优势是否增强 | control_swap.csv | 逐父对象平均；距离/coverage方向不同；与Native正确绝对质量并列 |
| Geo/Full同预算收益代价 | matched_budget_increments.csv；matched_budget_increment_summary.csv；resource_summary.csv | 同5000下Geo−T1、Full−Geo、Full−T1，几何多指标及时间/显存分别报告 |
| fit2是否学习及与主任务差别 | fit2_probe.csv；teacher_curves.csv中phase=fit；对应raw/score | probe新增0/500/1000，独立两对象，不纳入主排名 |
| 更支持继续训练、增加覆盖还是检查基础学习 | 上述完整曲线与固定5000留出、参数变化审计 | 证据解释而非新实验授权，不能自动延长；单seed、小对象数不支持唯一原因 |

holdout_regression.csv 是未参与本轮条件微调、此前已观察的留出回归，原1000 raw严格复用；不能称为新确认集。

## fit基线步数必须单独标注

冻结 reporting 的通用 step 列来自 model_id：phase=fit 的 T1_1000 行 step=1000，而 T1_fit2_probe_500/1000 行 step=500/1000。这些是不同命名空间，不能直接画在同一个“新增probe步数”横轴，也不能把 T1_1000 当成 probe已经训练1000步。

最终报告应显式使用以下映射：

| model_id | probe新增步数 | MVP48条件微调累计步数 | 角色 |
|---|---:|---:|---|
| T1_1000，phase=fit | 0 | 1000 | 从主T1历史1000复制的探针基线 |
| T1_fit2_probe_500 | 500 | 1500 | 独立双对象fresh AdamW探针 |
| T1_fit2_probe_1000 | 1000 | 2000 | 独立探针终点，不反馈主臂 |

可以在最终分析文案或另一个派生展示表中补该映射，不必修改冻结执行代码/输入。

## 最终报告前的有限展示检查

1. 原始curve/swap/fit/holdout每个ledger请求数、complete_rollouts、teacher NFE和score状态逐项核对；若后续出现score invalid，须保留请求并按score有效性解释。当前paired(control)状态字段基于生成status；最终分析应同时核对其对应score_status，不能只读paired的PASS。
2. 图件目录为 figures/，manifest记录主曲线与各noise0 case三行full/free/GT coverage图。Native train/validation参考线样式相同，图例/文案应明确两条参考的分组，数值表作权威来源。
3. 控制CSV保存逐case差异，没有自动写父等权控制总表；最终分析按上述两噪声→父对象聚合即可，不需新增生成。
4. C面积比例保留在per_case.C_area_fraction，约20%面数不等于20%表面积；分组分析宜报告既有固定面板面积差异而不事后挑case。
5. 以上所有阶段观察都不改变后续固定预算、停止条件或默认模型。
