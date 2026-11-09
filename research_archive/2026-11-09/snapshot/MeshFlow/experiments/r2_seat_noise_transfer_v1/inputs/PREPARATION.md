# R2_SEAT_NOISE_TRANSFER_V1 数据准备

冻结8个条件、48份新噪声输入。熟悉4C各8噪声（32请求），DEV新父4C各4噪声（16请求）；三模型共用同一请求数组，共144次RAW及144次单次D0 POST。此准备实际模型调用、训练、几何评分均为0。

|面板|父ID|类型|K/N|噪声数|来源C/Y共边|来源FF illegal/unknown|
|---|---|---|---|---|---|---|
|TRAIN_FAMILIAR|73b369b9808bc2d0f6ba522682be8978|SEAT_HALF|11/231|8|13|0/0|
|TRAIN_FAMILIAR|3f7417590f1bcfded5c89ecb06d1099b|SEAT_HALF|6/168|8|7|0/0|
|TRAIN_FAMILIAR|8c9b2023ecdeb7b049d998127ea6baf9|SEAT_FULL|16/183|8|17|54/0|
|TRAIN_FAMILIAR|85b16941984902f8facfa12c7d71c89f|SEAT_FULL|12/166|8|10|18/0|
|DEV_HELDOUT|5b1d0dad82acd6e5893104fdda835c64|SEAT_FULL|32/168|4|34|2/0|
|DEV_HELDOUT|608b46b0f8f9eba81b1a986f753a489b|SEAT_FULL|2/140|4|3|12/0|
|DEV_HELDOUT|2d08a64e4a257e007135fc51795b4038|SEAT_HALF|1/238|4|3|41/0|
|DEV_HELDOUT|610b1621d1997ee1daf3c002be658861|SEAT_HALF|1/228|4|3|0/0|

熟悉4C直接沿用上一快速学习实验，过去是有意纳入成功/失败的训练内诊断，不能当作无偏测试。DEV先固定FULL再HALF、父ID词典序、每类前2条合格且父不重复；只按已有来源资格挑选，没有读取生成数组或评分择父。DEV虽未进入67父条件训练，曾作为开发数据被评估，不称全新未触碰测试集。

DEV排除了原67父ID、group及相同源几何hash，另显式排除PAIR04的64871…：它与训练85b169…跨group近重复。已有记录未穷尽全部近重复，因此独立形状身份仅在这些已核对范围内成立。官方生成器及D0预训练是否见过均UNKNOWN。

所有座面标签继承已有AI_PROXY审阅的upper sitting surface；C/C及C/Y未检出illegal或unknown，源ID跨C/Y双面接口存在。FF源风险完整记录，不等于全局无交，也不要求生成复制唯一来源椅子。

固定namespace为 `R2_SEAT_NOISE_TRANSFER_V1/NEW_EVAL_FULL_N_MT19937`。每request只抽一次全N标准Gaussian float32并单独保存；旧推理合同要求known槽为0，因此z仅把[:K]置零，free保持全N抽样中的[K:]。三模型不再抽噪声。完整数组、实际z、free切片及文件SHA全部保存。种子与可用原32/67训练（含preflight）、快速续训和旧评估注册核对无碰撞；不可用的官方预训练内部随机流保持UNKNOWN。

8条C的2048点查询全部复用已有完整数组，坐标与权重字节不变；仅将metadata condition_id绑定到相同C的__base身份。保存原查询身份与文件hash，没有新增查询采样、坐标归一化、修复或重三角化。

兼容合同：split=TRAIN/DEV，panel=TRAIN_FAMILIAR/DEV_HELDOUT，variant=base，alpha=1；request保留原评分META全部字段、来源角序与顶点ID映射。`old.read_input`已逐条核验48份输入。

恢复入口：`python -B -m r2.pilots.seat_noise_transfer_v1.prepare`。已冻结manifest存在时仅核对hash，不重新抽噪声或覆盖。选择/排除见dev_selection.csv，种子核对见seed_audit.json，来源映射与完整查询hash见source_verification.json。
