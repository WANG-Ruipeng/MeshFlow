# 四条件快速可学性实验：输入准备

本阶段已冻结4个独立训练父、2个SEAT_HALF及2个SEAT_FULL，并复制8条现有评测输入。没有模型调用、训练、生成、新噪声抽取或训练流构造。

|原始训练task_id|N|K|来源合法C/Y共边|来源Y精确重复|来源FF非法面对|原baseline POST有共边输出|
|---|---:|---:|---:|---:|---:|---:|
|73b369b9808bc2d0f6ba522682be8978__SEAT_HALF|231|11|13|0|0|1/2|
|3f7417590f1bcfded5c89ecb06d1099b__SEAT_HALF|168|6|7|0|0|0/2|
|8c9b2023ecdeb7b049d998127ea6baf9__SEAT_FULL|183|16|17|0|54|2/2|
|85b16941984902f8facfa12c7d71c89f__SEAT_FULL|166|12|10|0|18|0/2|

四条条件的C/C与C/Y均没有已检出非法或UNKNOWN面对，原源接口存在，Y也没有精确重复。两个FULL仍有Y内部FF风险，不能称为全网格干净。此轮不修来源几何。特意选择已有接续正例和负例，用于快速可学性诊断，不是无偏泛化评测。

排除上轮另外两条座面：e1e7…HALF含2个Y重复组；6c36…FULL含27重复组及93个非法FF对。没有为凑齐数量加入新父；上轮已有6个独立座面父，本轮选其中4个。

## 冻结身份

- `manifest.json`是本阶段唯一总入口；顶层`focused_task_ids`为训练数据原task_id，不带`__base`后缀。
- `training_data_json`指向原32/67的`data/train_manifest.json`，`all_pool_name=USER_EXPANDED_67`，原67父列表保存于`all_pool_parent_ids`。
- `train_conditions`保存4条源task、源parent、condition/parent NPZ路径及hash、源面ID、原坐标变换和质量限制。训练器应读取原Dataset并按这些task_id筛选；本准备不重建训练流。
- `requests`含8条原TRAIN/base评测请求，request_id保持原值。`input_path`指向本目录逐字节复制的NPZ；C、N、K、整个z数组及free噪声hash均与上轮相同。
- 每条request的`baseline_raw_record`、`baseline_post_record`保存原索引记录；平铺字段为`baseline_raw_path/sha256`、`baseline_post_path/sha256`、`baseline_score_RAW_path/sha256`、`baseline_score_POST_path/sha256`。基线全部复用，不调用生成器。
- `queries`包含4条原查询记录，路径指向上一轮已冻结NPZ。2048查询点不重新抽样。
- `baseline_checkpoint_identity`与`baseline_D0_identity`沿用上一轮已核验模型身份；本CPU准备未读取模型参数或重复hash大权重。
- `source_hashes`为实际读取的旧数据/结果/评分证据hash；准备前后核验一致。`verification.json`保存8请求逐位核验与manifest SHA。

## 坐标和批次约定

使用历史完整父对象的MeshFlow模型坐标与原角序映射，不重新center/scale C，也不再次对模型空间数据归一化。原父`transform_json`完整保存在每条`historical_parent_coordinate_transform`中；它记录的完整对象历史预处理不可理解成这次又要执行一次。

集中臂batch8允许重复父：四父每批各2条实例，总共8实例，独立训练噪声与时间；不能称8个不同父。两组共享每批4 clean＋4 width增强日程，普通臂保留原67父采样，由主控制器冻结。评测的8条噪声只从现成输入复制，训练的随机流由控制器独立构造，不复用评测噪声作为训练数据。

恢复命令：`python -B r2/pilots/seat_quick_learnability_v1/prepare.py`。已有manifest时仅核对受保护文件与8份复制输入，不重抽噪声，不覆盖冻结定义。