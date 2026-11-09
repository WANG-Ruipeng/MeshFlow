# 3×3 条件评分方向审计

审计结论：原报告的 `correct-column advantage` 标注与实现不一致。旧实现固定生成条件 alpha，比较该输出对不同目标 beta 的距离，即取矩阵的一行；“在同一个目标上，正确输入条件是否比错误输入更有帮助”需要固定目标 beta，取矩阵的一列。此次仅从已有 CSV 重算统计，原 648 个距离/覆盖单元、生成输出和评分计算未被修改或重跑。

全部 72 个完整矩阵均纳入：3 臂 × 8 个父对象 ×（起点 noise0、终点 noise0、终点 noise1）。每个矩阵 9 个已有单元；每个方向各产生 216 条按 alpha 或 beta 的记录。每个模型、划分内为 4 个父对象。72 个矩阵不是 72 个独立父对象。

## 定义与源码证据

令 E[alpha,beta] = d(Y_beta,F_alpha)，其中 F_alpha 是用条件 C_alpha 生成的自由部分，Y_beta 是目标 beta 的固定对应 GT 点。d 为这些点到生成自由三角面的单向 RMS / L0；L0 是未变形尺度。它不是原评分器的面积加权双向 RMS。

原源码 `width_analytics.py:187–198` 外层遍历生成 alpha，内层遍历目标 beta，调用 `point_diagnostic(item['free'], points, L0, counts)`，因此矩阵行列含义没有歧义。该文件第 203 行按 `r['alpha']==alpha` 筛选，第 205 行从这一行取非对角元素，第 214–216 行输出旧差值和两个胜负字段。`reporting.py:496–500` 将该差值显示为 `correct-column advantage`。

本审计并列保留两种问题：

- 固定输出行：R_alpha = mean(E[alpha,beta≠alpha]) − E[alpha,alpha]。它检验同一个输出是否更接近对应目标。
- 固定目标列：C_beta = mean(E[alpha≠beta,beta]) − E[beta,beta]。它检验对同一个目标，正确条件生成的自由部分是否优于错误条件输出。

差值大于零表示对角比两个错误项的平均距离更小；严格胜出要求对角小于两个错误项中的较小者。两者不能互换。表中差值单位为 %L0，等于 CSV 内 fraction L0 乘 100。胜率没有设置新的近似相等阈值；直接沿用严格小于。数值复算一致性的 1e−12 容差仅用于核对，不是质量门槛。

## 216 条旧字段复核

`old_field_consistency_216.csv` 保留每条旧值、行口径复算、列口径复算及差值。

|核对项|一致数/请求数|
|---|---:|
|对角 RMS 原值与矩阵|216/216|
|旧 mean margin 与固定输出行|216/216|
|旧 no-worse 与固定输出行|216/216|
|旧 strict-win 与固定输出行|216/216|
|旧 mean margin 与固定目标列|0/216|
|旧 no-worse 与固定目标列|99/216|
|旧 strict-win 与固定目标列|99/216|

旧 margin 与行复算最大绝对差为 0。转换方向后，117/216 条的优势正负改变，117/216 条的严格胜负改变。原 CSV 的数字本身能精确复算；问题是取行实现不满足“固定目标比较条件”的解释，以及正文把行统计标成列优势。

一个重要恒等式：对完整 3×3 矩阵，mean_alpha(R_alpha) = mean_beta(C_beta)。两边都将六个非对角单元各计一次，并减去相同三个对角单元；逐矩阵最大数值差仅 9.27e−18。因此，三种 alpha 全平均后的 margin 不变是代数性质，不能证明两种逐目标解释等价，也不能证明旧标签正确。对角胜率、每个 alpha/beta 的 margin 不服从这个恒等式。

## 终点：固定目标列改变了哪些解释

下表每行先对同一父对象的两份噪声平均，再对 4 个父对象等权。每行均为 4/4 有效父、8/8 有效目标请求。胜率表示这些目标请求中对角严格胜过同列两个非对角；表里的 8 个请求不能作为 8 个独立父对象。

|模型|划分|alpha/beta|旧行差 %L0|新列差 %L0|旧行胜率|新列胜率|
|---|---|---:|---:|---:|---:|---:|
|Native|train|0.9|1.113174|0.128564|100%|50%|
|Native|train|1.0|0.385415|0.391568|37.5%|12.5%|
|Native|train|1.1|−0.414404|0.564053|12.5%|87.5%|
|Native|validation|0.9|1.368891|0.294911|100%|37.5%|
|Native|validation|1.0|0.103286|0.226833|0%|37.5%|
|Native|validation|1.1|−1.274338|−0.323906|0%|50%|
|Native+Geo|train|0.9|1.260695|0.027002|100%|50%|
|Native+Geo|train|1.0|0.381953|0.214706|37.5%|12.5%|
|Native+Geo|train|1.1|−0.435966|0.964974|12.5%|75%|
|Native+Geo|validation|0.9|1.375445|0.319636|100%|37.5%|
|Native+Geo|validation|1.0|0.101765|0.157462|0%|25%|
|Native+Geo|validation|1.1|−1.306320|−0.306207|0%|50%|
|Native+Geo+Joint|train|0.9|1.118587|0.694970|100%|50%|
|Native+Geo+Joint|train|1.0|0.384614|−0.213034|37.5%|12.5%|
|Native+Geo+Joint|train|1.1|−0.405119|0.616145|12.5%|87.5%|
|Native+Geo+Joint|validation|0.9|1.383276|0.323383|100%|25%|
|Native+Geo+Joint|validation|1.0|0.100943|0.276029|0%|50%|
|Native+Geo+Joint|validation|1.1|−1.304570|−0.419762|0%|50%|

云端指出的验证组 100%/0%/0% 现象确实来自行口径：它说明那些输出对目标尺度有相同偏好，不能直接说正确收窄条件总能战胜错误条件。换成固定目标列后，三臂各 beta 的严格胜率为 Native 37.5%/37.5%/50%，Geo 37.5%/25%/50%，Joint 25%/50%/50%。

这纠正了目标方向的解释，但没有把证据改判为可靠控制。验证 beta=1.1 的平均列差三臂仍为负，意味着正确条件输出的平均距离比错误条件平均值更差。与此同时有一半请求严格胜出，说明平均值与逐请求胜负揭示不同的非均匀表现；不能从任一汇总推出每个父对象都好或都坏。训练 beta=1 的严格胜率三臂均只有 12.5%；Joint 该目标平均列差也为负。

终点所有 beta 合并后的固定目标严格胜率为训练 Native 50%、Geo 45.83%、Joint 50%；验证 Native 41.67%、Geo 37.5%、Joint 41.67%。这些数字没有提供 Geo 控制更稳定的证据。原始宽度、相对响应、覆盖、接口及相交指标不受本次方向修正影响；本审计没有重新评价它们。

## 真正共有第一噪声的起终点配对

起点为源条件 step1000，终点为新增 phase3000、累计 step4000。每个模型/划分为 4 个父对象，每父 1 份相同噪声、3 个目标 beta。先在父内平均三个 beta，再汇总父间配对差；“改善父”以该父的列 margin 平均差为正计。已有 `all_generation_per_case.csv` 的 input SHA 和 case_id 逐项一致，终点第二噪声不参与这里。

|模型|划分|列差起点 %L0|列差终点 %L0|配对均值差 %L0|配对中位差 %L0|margin 改善父|列严格胜率起→终|
|---|---|---:|---:|---:|---:|---:|---|
|Native|train|−0.004123|0.390600|0.394723|0.331429|3/4|33.33%→58.33%|
|Native+Geo|train|0.010724|0.395980|0.385256|0.362587|2/4|50%→50%|
|Native+Geo+Joint|train|−0.018885|0.408983|0.427867|0.350459|2/4|33.33%→58.33%|
|Native|validation|0.020043|0.097591|0.077548|0.031157|3/4|41.67%→41.67%|
|Native+Geo|validation|−0.016770|0.088595|0.105365|0.032417|4/4|33.33%→33.33%|
|Native+Geo+Joint|validation|0.003717|0.083446|0.079729|0.029527|3/4|33.33%→41.67%|

三臂两个划分的平均列 margin 均增加，保留了部分学习证据。Geo 验证组 margin 为 4/4 父改善，但严格胜率并未提高；Geo 训练组平均 margin 增加但只有 2/4 父改善，严格胜率仍为 50%。所以“平均差变好”“更多目标由正确条件胜出”和“各父都变好”仍是不同结论。

这组列差跨三个 beta 平均后与旧行差相同，原因是上述代数恒等式；不能将它作为本次修正新增的独立学习证据。固定目标列胜率和逐 beta 数值才提供方向修正后的新解释。全部按噪声、目标、父对象的明细均保留，没有选择样本、删除不利矩阵或把三份重复斜率当作三次独立观察。

## 受影响范围与未改动范围

确认受方向命名或解释影响的字段是 `wrong_mean_minus_diagonal_RMS_L0`、`diagonal_no_worse_than_both_wrong`、`diagonal_strictly_better_than_both_wrong`，以及这些字段的派生均值/差值。文件清单及命中行数在 `affected_old_artifacts.csv`：B_width_response、B_width_per_parent、B_width_summary、B_width_start_endpoint_pairs/summary、B_width_matched_budget_per_case/summary，以及主报告的 correct-column 表头。旧文件与分发副本不应被无说明覆盖；引用这些数字时需注明旧为行口径，或改引本目录固定目标列结果。

`B_cross_condition_scores.csv` 的 648 个基础单元不受影响，原对角距离也不受影响。宽度、斜率、单调性和原始几何评分并未因方向命名而数值失效。仍需区分评分执行 PASS 和生成质量良好；本审计未重算几何质量，也没有设定质量通过门槛。

## 可复核产物与执行边界

- `matrix_audit.py`：独立标准库脚本，固定 1 个 CPU worker、线程环境均为 1，不导入原实验执行模块。
- `fixed_output_alpha_per_case.csv` 与 `fixed_target_beta_per_case.csv`：各 216 条，清楚标注固定轴。
- `axis_summary_by_noise.csv`：分别按 arm/split/phase/noise/alpha-or-beta 汇总；`axis_summary_parent_equal.csv` 为父内噪声平均后的汇总。
- `matrix_level_72.csv`、`all_axes_parent_equal_summary.csv`：整矩阵统计与跨轴父等权统计。
- `matched_noise0_start_endpoint_*` 与 `matched_noise0_matrix_pairs_24.csv`：全部共有第一噪声配对，包括每目标和父内目标平均口径、均值/中位差/最坏父与有效分母。
- `slope_deduplicated_72.csv`：验证每矩阵三条 alpha 行的 slope 等字段完全相同，仅保留一次；不做伪独立统计或新的显著性推断。
- `old_field_consistency_216.csv`、`facts.json`、`audit.json`：旧值差异证据、计数与 SHA256 清单。输入 CSV 先与原 reporting_status 的文件哈希核对，分析前后旧输入/代码/报告哈希一致。

新增训练、生成、模型前向、反向、优化器更新、OT、几何评分器、点到三角距离调用均为 0。全部计算只在既有矩阵数字上做减法、比较与分组统计。没有修改任何旧源码、raw、评分或报告。
