# Bench 跨类别准备与来源摘要

本轮冻结 **64 个 TRAIN 父对象、8 个 DEV 父对象**，每父使用一个现成网格版本和一条连贯局部表面条件。两臂从同一官方 **Bench EMA** 建立条件模型，使用 fresh AdamW；这是将 MeshflowCrtl 方法适配到另一个类别的试验，**不是 Chair 权重的零样本跨类别生成**。本文仅记录准备事实，不解释生成结果。

## 为什么先做 Bench

已有 `shapenet-cls.tar.gz` 归档直接读取所得：

| 官方类别 | NPZ 父对象数 | N128–256 数量 |
|---|---:|---:|
| Bench | 560 | 88 |
| Chair | 3,135 | 512 |
| Lamp | 627 | 58 |
| Table | 5,155 | 1,182 |

开工盘点已找到本地官方 Bench 生成器；Table/Lamp 对应生成器未在已检查的本地项目资产中找到。Bench 的 88 个面数合格父中，84 个具有符合规则的连贯条件区域，且84份精确归一化无序三角几何哈希各不相同。四小时窗口优先完成这一类的真实配对训练和评测，没有为增加类别数而展开新的多类长训。

## 固定父对象与条件

先按确定性父 ID 哈希排序，再按输入资格选择；没有查看生成结果挑父、换噪声或补挑好例。DEV 优先采用全部 **7 个合格官方 test 父**，另取1个官方 train 父 `5dda4ef4ab9add9411ee8243f54292d6` 作为**仅从条件适配中留出**的对象；64个 TRAIN 均来自其他官方 train 父。两类 DEV 暴露状态须分开解释，近重复及来源家族没有得到穷尽认证。

| 数据属性 | TRAIN | DEV |
|---|---:|---:|
| 父对象 | 64 | 8 |
| N：最小／中位／最大 | 128／190／256 | 154／179／244 |
| K：最小／中位／最大 | 2／2／51 | 2／2.5／16 |
| K/N 中位数 | 1.19% | 1.41% |
| C／完整来源计面积中位数 | 10.83% | 13.24% |
| C 开放边 | 429 | 41 |
| 来源 owner／alias／unsupported | 311／0／118 | 31／0／10 |
| 无已验证来源 owner 的父 | 11 | 1 |

面积分母是**全部保留来源面的面积总和，包含重复面**，不是去重后的外表面积。来源 owner 要求唯一 C/Y 邻面、相反绕序及经有限面对检测确认的合法共边。unsupported 包含无邻面、入射数或绕序冲突等，不能全部解释为语义上的“这条边不应连接”。

C 从原索引拓扑的连贯近共面区域取得，优先较宽的水平面，K≥2；不生成新三角形、不重三角化、不焊接或修源网格。区域包含座板、板条、底板、带孔框面，另有3个 TRAIN 非水平区域。统一名称为 **`PLANAR_PATCH`**，不称“全部都是核验过的座面上表面”。

完整父网格只作一次官方 FP32 归一化，pre-OT 除以0.3762、模型坐标再乘2；C不单独归一化。所有case保存源顶点／面ID、原角序、可逆规范化角序及完整free补集。

## 输入图审实际确认了什么

两位 AI_PROXY 审阅者实际检查了72份来源与C的9张整物体检查图，以及5张C-only主平面图。后者避免来源灰面遮挡蓝色C造成“缺片”的误读。输入均是连贯区域；长条、带孔面和底／侧面均保留，未因此改父或换区。

另对72份C执行冻结双算法有限三角面对核验：**非法C面对0，UNRESOLVED 0**。这不认证完整来源Y或未来生成网格的质量。

原完整来源仍有缺陷：TRAIN 中45/64父合计1,334张相同无序源顶点ID的重复副本，33/64父合计190张退化面；DEV 分别为5/8父、85张副本，以及3/8父、12张退化面。退化口径为两倍面积≤1e-12模型单位²。未清洗这些目标，完整来源FF本轮准备阶段为 **NOT_RUN**。

固定评测为4个 TRAIN＋8个 DEV，每父一条C、两份登记高斯噪声，共24请求／模型；source Y与owner标签不提供给推理模型。唯一来源款式不是开放生成成功标准。

## 可复核入口

- [完整数据卡](data/DATASET_CARD.md)、[逐父数值](data/dataset_card_statistics.json)、[分布概况图](data/dataset_overview.png)。
- [固定父／条件清单](data/manifest.json)、[24请求清单](data/evaluation_requests.json)、[来源接口账本](data/source_interface_audit.json)。
- [实际图审收据](data/input_visual_review.json)、[72逐父图审记录](data/input_visual_review.csv)、[C有限面对核验](data/condition_geometry_audit.json)。
- C-only图：[TRAIN 1](data/gallery_C_only/TRAIN_00.png)、[TRAIN 2](data/gallery_C_only/TRAIN_01.png)、[TRAIN 3](data/gallery_C_only/TRAIN_02.png)、[TRAIN 4](data/gallery_C_only/TRAIN_03.png)、[DEV](data/gallery_C_only/DEV_00.png)。
- 整物体图：[TRAIN首张](data/gallery/TRAIN_00.png)、[DEV](data/gallery/DEV_00.png)；全部9张路径列在图审收据中。

| 固定资产 | SHA256 |
|---|---|
| 已有原始归档 | `06707caff4ce465f402e6b69f81056204c579921d371eb2c9e4e9a3929f7aef2` |
| 官方 Bench 生成器 | `00fc5ddb5bc5690cd53fe4f092ee3320e23437a287464fa5639120c2bb82a505` |
| 数据 manifest | `a2d6ed3e3afe608d4e7ef5cb6b596ebcee5be1665946bd3438c1ee14d16f3894` |
| 评测请求清单 | `96c9e0515b091ce64b139c6fc88906e342b9290e2fbac9fdf71bdaa013825024` |
| 来源接口账本 | `c1a6bbc81d731e9cdf514f16b78ea1ecc9847ad3a7f902dabd39ab83eca9490b` |
| C有限面对核验 | `54ca4a4b707dd489ecf6709f159522b81e0a586e5df7240b10925a31124199fc` |
| 输入图审收据 | `c72e5d5cb2df4995a1a2834eefac2f0e621781ae9e6d0f065dd4d7bdd1c0ebec` |

每个case的独立SHA保存在固定manifest中。本摘要使用已保存数据与账本，没有新增模型调用、修改冻结输入或扩展实验。
