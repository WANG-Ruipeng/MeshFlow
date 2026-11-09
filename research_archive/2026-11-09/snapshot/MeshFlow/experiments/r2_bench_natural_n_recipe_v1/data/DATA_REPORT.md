# Bench 自然面数 Native 数据准备

本轮实际可用训练池为 **26父、26条完整座面条件**：原22父、既有关注父a8/27、新增28b9/abf250各一条。训练目标自然面数128–384，不再要求两新增父简化至256。另有两个既定预留高面数父作为评测，未进入训练。

这改变了新训练数据的面数范围和结构覆盖；不是仅扩大父数、或仅改变面数的单因素实验。没有修改旧24条件、完整来源、既有噪声、评分定义或模型参数。

## 实际新增及预留输入

| 角色 | 父ID前缀 | 完整目标N | 输入K | 条件含义 | 源结构与边界 |
|---|---|---:|---:|---|---|
| HIGH训练 | 28b9 | 384 | 10 | 五根并列宽座条的全部宽上表面 | 两块箱式端墩；C有20条开放边，5个座条表面分量 |
| HIGH训练 | abf250 | 308 | 44 | 22根并列弧形座条的宽上表面 | 两墩承托，座条前缘弯下；C有88条开放边，22个表面分量 |
| HIGH留出 | 617 | 724 | 44 | 完整厚座垫，含侧面和底面 | 下承框及四圆腿在Y；C是闭合部件，没有开放边 |
| HIGH留出 | 756 | 375 | 10 | 完整长座面上表面 | Y保留格背、扶手、腿及座面厚度；C有12条开放边 |

四条均复用已保存的明确部位选择，不随机抽面，不根据生成结果挑选。28b9与abf250的图中能看到真实座条间隔；多个输入分量是完整座条组，不是任意漂浮碎面。abf250是弧条装配，不能宣称它直接补齐27的连续座背碗壳家族。617的闭合厚座垫不同于开放上表面，单独记录闭合状态，不能给它强加开放边接续要求。

28b9/abf250此前仅因N128–256合同不能晋级，并不重跑已经失败的低面简化。此次完整目标、C/Y及其坐标与ID全部原样复用，只建立新的Native自然面数准入记录。旧报告中的PENDING状态保留历史含义，不覆盖。

## 数据与来源核验

- 26训练条件和2留出条件逐项核验：C/Y源面ID互斥并完整覆盖M；C/Y等于对应full_target子集；`pre_ot_full * float32(2)`逐位等于full_target；原顶点ID及角序映射正确。全部PASS。
- 新增和留出的四份完整来源复用已认证有限面审计；原记录的非法面对、未判定面对、精确零面积面均为0。此次没有重新运行完整源几何检测，亦不据图认证隐藏结构无交。
- 两新增训练case直接引用旧已冻结的完整NPZ，避免无意义复制；两留出将同一版本来源与原R1分区整合成新的可读case。未修改任一顶点、面或角序。
- 模型输入严格为C几何与总N（及Native任务的已知/自由角色）。source-owner/alias保留为历史监督/分析数据，不用于配置生产输入；没有SEAM或占用头。
- 原24条件与新增2父不包含617、756及df6。617/756沿用此前冻结的独立适配留出角色；它们属于官方train划分，官方检查点是否实际见过UNKNOWN。视觉家族区分不等于穷尽式近重复或来源谱系认证。

| 父ID前缀 | source clean.npz SHA256 | 本轮case SHA256 |
|---|---|---|
| 28b9 | `a6b3da82da85bd285126d1ebe1bf66cb46a02038da19844bd0cb1c7b07931b95` | `2c50223ea083c3d06987edb9de47ec95db0c798c32a1eb6e7f4e1cde8f5b24bf` |
| abf250 | `c9c47f3bc26cd5aba54cb9c3a82fbe988bdb49a0f27c4af462235410f5694107` | `3cda7bea3d2a55a8a47160517d523b0bf4b259b17f4232ffaa9fffd04762800d` |
| 617 | `9cf46bc9194170a251bf039930613fc90ba75488682a121f9b17c26bf94faaec` | `1e44892c80be415f926ab4f210862cfa9493afa086f0074aa376b370fe0d2c7a` |
| 756 | `e9d03ba457f8166461ce82cb5c299ae823d84f7c69bda41c0e4958ca31f3c992` | `437bd5dcb9717c34dd5c4c3e61db809e80abf2cbd91784ad69271c298c965482` |

## 评测输入与图审

原曝光实验30个请求的所有字段、数组和噪声逐字段继承，新增四条条件各两份固定噪声，合计38请求：OLD16、FOCUS4、DEV8、原HOLDOUT2、HIGH_TRAIN4、HIGH_HOLDOUT4。新增8份FP32高斯数组及4组2048点C查询；所有方法共享同一数组。新种子由固定命名空间、条件ID和噪声索引决定，与worker/完成顺序无关。

输入NPZ仅包含`C_vertices/C_faces_original/C_faces_model/N/K`；原角序导出信息另存metadata/source_condition。显示的相机和前/侧截面位置在生成前由来源固定，来源完整体仅作显示范围，不作为唯一款式成功标准。617 N724明显超出本轮训练最大N384，需披露面数外推和闭合输入变化，不能只把该请求表现解释成父级泛化。

本代理本轮使用view_image实际查看四父M/C/Y/C+Y、各两视角，非盲AI_PROXY；旧图hash已认证，未用分数代替观察。训练两图保留旧PENDING标题，最新准入以此目录清单为准。图像路径、hash、具体观察见[source_visual_review.json](source_visual_review.json)。

## 交付与执行边界

- [manifest.json](manifest.json)：训练26行及高N留出2行，父、条件、角色、来源、完整路径/hash。
- [cpu_checks.json](cpu_checks.json)：28条件数组与身份检查。
- [eval_candidates.json](eval_candidates.json)：38固定请求；由评测代理另行认证旧P0结果并冻结正式eval_manifest，不是自动执行采样。
- [eval_cpu_checks.json](eval_cpu_checks.json)：38请求噪声、查询、C-only输入与来源绕序核验PASS。
- [COMPLETE.json](COMPLETE.json)：本阶段回执；[resources_before.json](resources_before.json)：实际资源。
- 代码：`r2/pilots/bench_natural_n_recipe_v1/data_prepare.py`。恢复核验入口：`python -B -m r2.pilots.bench_natural_n_recipe_v1.data_prepare verify`；评测输入检查复用`panel`入口，不会重复抽噪声。

资源记录为RTX5080 16,303MiB，数据准备前已用2,830MiB、空闲13,150MiB、GPU利用率1%；WSL24CPU、可用约65.8GB RAM。实际一个CPU进程、数值线程1，无模型/GPU/训练/完整新几何评分调用。首次Windows CIM资源读取权限不足，随后用WSL psutil记录真实CPU/RAM；GPU进程查询以WSL记录为准，未操作其他进程。

数据准备不决定训练是否获益，不扩到额外父，不修改维护默认、commit或push。
