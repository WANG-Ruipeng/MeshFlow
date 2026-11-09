# 数据与派生缓存清理审计（2026-10-25）

结论：当前没有确认可以无损删除的主训练/评测数据池。确认了一份完整保存在 ZIP 中的重复审阅包展开副本，可按逐文件白名单清理；本审计没有执行删除。

- 精确副本：`experiments/r2_single_step_denoiser_distillation_v1/review_package/` 中 **1022 个文件、301,071,160 bytes（287.124 MiB）**。
- 全部 1,023 个归档成员已经逐字节 SHA256 对照通过。保留 21,964,835-byte ZIP；其 SHA 与原 completion 一致。
- 保留展开目录中的 `PACKAGE_MANIFEST.csv`，因为旧完成核验入口仍读取它。删除其他白名单文件后，原完整包可从 ZIP 无损恢复，展开版网页浏览退役。不得整删原实验目录。

## 当前数据依赖

读取 7 份当前冻结清单，解析 4,301 条非权重引用（包括目录引用），全部存在。清单：`dataset_active_dependencies.json`。

Native22 完整座面数据直接引用旧 USER v2 / expansion 的 C/Y/case；Coverage24 又继承这些，并加入自身来源。943/9e 还依赖 RELATION 重建版本。不能因为这些数据目录日期较早就整包删除。

SHAPE 后端的16训练父、2,000固定实例、15父来源验证被 RELDEF 逐字节复用。最新RAW局部修复还使用 Coverage30 RAW、官方/SHAPE/RELDEF proposals 和已有几何输出。上述数据均列保护。

## 大目录并非重复缓存

对 53 个选定 data/assets/input/cache 根做了实际尺寸统计。合计12,461,927,720 bytes，其中7,289,979,546 bytes是 assets 中受保护的官方Chair D0权重；实际非权重为5,171,948,174 bytes。此统计不包含所有源数据根，也不代表都能删除。详见 `dataset_root_sizes.csv/json`。

最大的非权重目录为旧 USER_MIX/data（2,677,819,450B）和 CONT/data（842,777,406B）。源码确认它们保存精确训练Gaussian/epsilon、OT映射及输入，不是下载临时文件。删除会导致既有只读训练回放与输入核验失效；本轮未把它们列安全白名单。对应统计/计划/逐批attempt记录也不能冒充保留了完整数组。

官方ShapeNet原归档仅25.48MB，且包含后续类别来源；原Chair源几何约47MB。保留原件的价值明显高于这点节省。MOMENT大debug NPZ也是独有数值证据，未核到完整备份，不列安全删除。

20261022已删除ZIP校验临时展开树与r2/build，本轮不重复计算它们的释放量。所有RAW/POST、固定噪声、query、C/Y、source IDs、图审和原评分继续保留。

## 执行边界

仅CPU目录、源码、清单、归档SHA核查；没有模型调用、训练、生成、重新评分、移动或删除。父会话负责决定并执行精确清理白名单。
