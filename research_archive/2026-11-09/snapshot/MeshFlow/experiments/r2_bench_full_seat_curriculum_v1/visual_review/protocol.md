# Bench 完整座面课程：实际图审协议

本次为非盲 AI_PROXY 图审，不是独立人类评测。固定 24 请求（12 父、每父
1 条完整座面 C、每 C 两份噪声）× 3 阶段 = 72 张比较页。每页按固定列序
COMMON_S12000 / NATIVE_S6000 / SEAM_S3000 展示，共 216 个模型单元。

必须实际打开每张完整 PNG，逐列查看两个固定视角（azimuth -60/120、
elevation 23）和两个固定位置截面（front/side 的预登记中间位置）。
三模型同页、同噪声、同尺度、同切面；同页三列在同一次 view_event 中记录。
缩略图只作导航，不能代替逐页查看。蓝色 C，橙色 SEAM，灰色 BULK，绿色
后处理 BRIDGE。保留相同父的两噪声及三阶段为一个评阅分片。

沿用既有五轴，分别判断：

1. overall_shape：REASONABLE / INCOMPLETE / UNCERTAIN。
   看完整且合理的 Bench 主体，不要求复刻来源款式；无靠背 Bench 本身可以合理。
2. C_use：PLAUSIBLE / REPLACED_OR_IGNORED / UNCERTAIN。
   看蓝色完整座面是否合理参与生成主体。仅 C 存在、位置大致正确、或橙色面
   与它相接均不能替代用途判断；主体在另一高度另造座面需单独识别。
3. support_retention：PRESENT / MISSING_OR_DAMAGED / UNCERTAIN。
   看最终生成物主要支撑是否可辨、合理、完整；不是要求腿数或构造与来源相同。
4. visible_fragment_or_gap：YES / NO_OBVIOUS / UNCERTAIN。
   记录明显漂片、断裂、缺口；截面线段端点本身不能证明裂缝或孤岛。
5. layering_or_fold：YES / NO_OBVIOUS / UNCERTAIN。
   记录可见的明显重层、折叠或相互穿过迹象；不把它冒充精确相交检测结果。

每个轴独立记录。完整主体可能忽略 C；采用 C 也可能缺少支撑。遮挡、切面
遗漏、投影重合、越出固定视框或用途歧义时用 UNCERTAIN 并说明。
NO_OBVIOUS 只表示这些视图中不明显，不认证无自交、无裂口或拓扑连通。
所有五轴对三个模型都适用。Native 的 SEAM 数值指标为 NA_NO_SEAM_NATIVE，
不是视觉用途、完整性或支撑指标的 NA。SEAM 表述适用只表示模型合同；不能
据此预填实际激活数、可达面积或任何视觉标签。

不能用 CF/FF/I、共边数量、loss、占用头标签、历史评语或模型名称代填视觉
判断。来源 M 不是唯一正确完成，不按款式相似评分。不要把数据构造时的
ACCEPT 图审当成生成质量图审。每个单元都需自己的简短图像依据说明。

模板与实际证据严格分开：templates/*.json 的轴、视阅时间、实际视阅声明、
图像 hash、reviewer 和 note 全部为空，status=PENDING_TEMPLATE_NOT_OBSERVATIONS。
这些模板不能计作已评阅或导入汇总。review_*.csv 初始只有表头；看完一页
后才一次加入该页三列的真实记录，未看页保持缺失，不填假标签或 UNCERTAIN。
实际记录必须含 reviewer_kind=AI_PROXY、blind=false、review_round、UTC 时间、
实际查看工具/来源、view_event_id、actually_viewed=true、
views_and_sections_checked=true，以及 PNG、渲染 receipt 和 definition 的 SHA256。
hash 只能认证文件，不能单独证明已经看图。保留原观察；禁止静默覆盖、择优
重评或合并重复单元。若需要复评，另设轮次并保留原记录。

CURRENT_TRAIN 为本轮 8 个训练父/16 请求，DEV 为 4 个未进本轮池的父/8 请求。
旧 DEV 已反复用于开发，不改称全新独立测试；官方预训练是否见过仍按原边界。
三分片各 4 父、8 请求、24 页、72 单元，可并行实际查看；不会产生独立复现。
结果需分组、分阶段报告，保留全部类别及 UNCERTAIN 分母。两噪声先在 C 内
平均，再在父内平均、最后父等权；本轮恰好每父一 C。未来可汇总三对模型
逐请求转移，但不自动创造联合成功指标或用一个轴抵消另一轴损失。

prepare 只读冻结注册/面板并生成空模板。validate 只认证实际提交记录及真实
PNG receipt；缺失仍报告 PENDING。完整 216 单元之前不得公布全组视觉汇总。
本工具不调用模型、不评分、不生成图片、不改变训练和评测合同。
