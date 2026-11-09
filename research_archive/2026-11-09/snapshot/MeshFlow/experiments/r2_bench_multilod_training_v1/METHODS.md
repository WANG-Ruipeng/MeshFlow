# Bench 固定 C 的多面数版本配对训练

实验：`R2_BENCH_MULTILOD_TRAINING_V1`。本文记录已经冻结的方法与预算，不预写模型效果；训练完成、评分和实际图审后再形成结果报告。权威身份以本目录 `training_registration.json`、`data/evaluation_plan.json` 及其引用的 SHA256 为准。

## 本轮问题与对照

检验同一真实条件 C 在不同合格 free 三角划分/面数版本上的训练，是否能改善生成质量与条件使用。比较 `MULTILOD1500` 与 `PLAIN1500`；二者相对冻结 `EXPOSURE` 的变化作为辅助比较。本轮没有关系预测头，没有额外几何损失或新的 SEAM 结构。

两臂均从保留的 EXPOSURE 完整检查点分叉：global step25500、Native累计13500；继承同一 live 生成器、匹配的 AdamW 和恢复状态。每臂新增1500次更新，预定终点均为 global27000、Native累计15000、SEAM累计0。不根据中间图片或loss挑选checkpoint。

| 项目 | PLAIN1500 | MULTILOD1500 |
| --- | --- | --- |
| 独立父 / 物理 C | 同30父 / 30条C | 同30父 / 30条C |
| 目标版本 | 30份NATURAL | 同30份NATURAL＋9份已准入LOW |
| 每臂更新 / 实例曝光 | 1500 / 12000 | 1500 / 12000 |
| NATURAL / LOW实际曝光 | 12000 / 0 | 10404 / 1596（LOW占13.30%） |
| batch / microbatch | 8 / 1 | 8 / 1 |

## 数据与输入合同

增加的是同父同C的9个几何版本，不是9个新父或新条件。LOW由受约束free简化构造，保留C坐标、原顶点身份、来源绕序以及受保护的来源接口邻域；C与Y均从同一新完整目标分割，来源面域使用新版本身份，旧面槽沿袭另存。准入证据与图审保存在 `../r2_bench_multilod_data_v1/`。原NATURAL沿用旧准入边界，不能把这批资产统称为全无交数据。

模型合同允许总N128..799；训练按选中case的真实N计数。`case_id`包含物理条件与版本，缓存不能只使用`condition_id`。来源完整体、owner/alias、面沿袭仅作监督或构造元数据，不能变成推理条件；生成器输入仍为C几何、真实N与状态/时间。

父选择及CLEAN/X_WIDTH顺序使用EXPOSURE已完成曝光流的前1500步。原OLD/FOCUS/HIGH组配额保留（实例分别9000/1500/1500），组内沿用原平衡父流；这不等于30父全局均匀曝光。选中父后才决定版本：同父第1/3/...次逻辑曝光用NATURAL，第2/4/...次用LOW；没有LOW则继续NATURAL。一次逻辑曝光包含CLEAN与X_WIDTH，两条必须使用同一版本。LOW不能被平铺成额外父样本。

两臂共享新登记的noise、epsilon、time、permutation种子，按 `SHA256(experiment|step|micro|field)`构造。已核对它们与旧EXPOSURE全部3000步的同字段种子没有重合。X_WIDTH的原alpha及共同父流不变。N变化时重新生成正确形状的噪声并执行新的OT；共享种子不等于不同形状的完整噪声数组或匹配相同。

## 模型、目标及组权保持

保持既有Native/T1＋role＋Geo、free-only FM、free OT后HYBRID0.25；不引入可训练参数。BF16前向，FP32参数/积分，确定性算子，禁用TF32，SDPA math。AdamW恢复而不重置：lr1e-5、betas(.9,.95)、weight decay0、梯度裁剪1；不启用EMA。

原EXPOSURE使用窄池参考的组MSE系数。本轮按实际N重算微批系数，以免LOW让HIGH等组的有效权重发生额外变化。对每个更新，设原有效free标量数为`c_i`、旧微批gain为`q_i`、原分母为`D_old`，冻结

`a_g = sum_{i in g}(c_i * q_i) / D_old`。

本臂实际计数为`c'_i = 9*(N_i-K_i)`，`m'_g = sum_{i in g} c'_i`，`D_arm = sum_i c'_i`。采用

`gain_i = a_g * D_arm / m'_g`，

`loss = sum_i(gain_i * SSE_i) / D_arm = sum_g a_g * MSE_g`。

两臂共享同一组权`a_g`，组内仍按实际free坐标数平均。并未改成每父MSE或宣称每父梯度相同。PLAIN与旧M微批gain的最大误差为4.44e-16；各步组权和为1，缺席组权为0。记录原计数、参考组权、实际计数、gain及有效组权，不能复用旧N下的微批gain或分母。

## 固定评价与证据边界

评价复用EXPOSURE原46请求、23父的C/N、两份噪声、查询、相机、显示范围和位置截面，不因训练LOW改变评价面数。分组为OLD16、DEV8、FOCUS4、HOLDOUT2、HIGH_TRAIN4、HIGH_HOLDOUT4、NEW_WIDE_TRAIN8个请求；保留原分组和预训练重合边界，不重新命名为独立测试。

两新终点各生成46份Euler50 RAW，共92份；所有92份RAW及receipt齐全后才允许第一份D0。统一单次官方Bench D0 → GEO_C及来源定向 → HINGE_OPP015 → ALPHA → HOLE。保留RAW、BASE_POST、CLEAN_FINAL_POST三个阶段。EXPOSURE原138行分数及原产物只读复用；新增276行，合计414行、177个原指标字段。聚合顺序为噪声→条件→父→组，父等权，NA及原分母保留。

生成、接续、CF/FF/I、碎片与处理损失必须同时解释；匹配唯一来源款式不作为开放完成的成功标准。图集固定138页，每页比较三个模型的两斜视图及原前/侧截面；用途、主体、支撑等观察由实际查看后单独记为AI_PROXY，不能由指标生成标签。

CPU准备已通过39实际版本×CLEAN/X_WIDTH的78次OT检查，核对固定C、selected Y的面/角序传输、实际N、mask与9(N−K)分母。该通过只认证数据与计算合同，不是训练或生成的正结果。

## 停止条件

固定两臂终点、46请求完整配对评价和实际图审完成后收口。资源并行只影响调度，不增加更新、候选、样本或重试。结果尚待填写；不自动晋级默认模型、不额外延训、不commit/push。
