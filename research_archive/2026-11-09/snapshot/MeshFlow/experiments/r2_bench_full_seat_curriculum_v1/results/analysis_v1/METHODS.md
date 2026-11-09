# 只读结果分析口径 v1

此工具只读取本轮已完成 collect 的 216 行原评分，不构造网格、不调用模型、
不运行检测器、不增加固定查询，也不填写任何视觉评价。当前训练完成与否、
controller 是否 READY_REVIEW、训练调用账本和逐图真实审阅需要分别核验；
本工具的 COMPLETE 仅表示数值分析产物完成。

面板为 CURRENT_TRAIN 8 父 / 8 C / 16 请求、DEV 4 父 / 4 C / 8 请求。
对每个模型、处理阶段和分组，先把同 C 的有效噪声数值取均值，再把同父
的 C 均值取均值，最后父等权。全部实际分母、有效请求 ID 和逐父数值保留。
两份噪声不等于两个独立父；DEV 是重复使用的开发证据，不是独立测试。

primary 表的 I/交叉/重叠/交集仅收 ConflictCov_C_status=PASS。
每条配对增量先限制双方共同有效（I 必须双方 PASS）的相同请求，再按
同样层级聚合候选减参考。参考/候选的共同支持均值与各自边际均值可能不同，
不能相减不同支持集的边际均值来替代配对。I 支持量、未决量、上界和状态
计数另列；非 PASS 不是零。all_numeric 表原样聚合所有有限原值，包括
非 PASS I，仅用于完整证据保留，不应当作已认证 I 主结论。

COMMON/NATIVE 没有 SEAM 表示，专属 bulk reach 等字段为
NA_NOT_SEAM_REPRESENTATION，不能按零比较，更不能宣称 SEAM 相对此 NA
获得了数值增益。闭合 C 相应字段为 NA_CLOSED_C_NO_SEAM_INTERFACE。
开放 C 上的全 Null 布局仍适用，真实零可达不能改成 NA。合法 CF 边长
属于实际有效图，Native 仍适用；它、bulk reach>=0.30 都是几何描述量，
不是自动的主体完整、接合需求满足或语义成功判定。
原 scorer 若记 NA_EMPTY_OR_ZERO_AREA_BULK，则是保留 bulk 面积为空/零的
分母边界，和全 Null 布局不同；依原值保留，缺失可达比例不填零。

CF/FF 非法受影响自由面比例分母为当阶段实际 free 面数；固定父版本的
分母是该处理链原始父面数（RAW 的基准为自身，BASE/FINAL 为 BASE_POST），
二者不能混用。I 是固定 C 查询面积权重覆盖，
不是全物体自交率。面数/小角比例同时保留实际 free 分母、测试面对分母、
固定父分母；面积/长度保持原模型单位，未重标定百分数或尺度。
deleted_free_faces 按原始 RAW 父 ID 缺失计数，不能用 N 减输出总面数代替。
retained_original_RAW_free_faces=N-K-deleted_free_faces；桥接新面另列，
actual_free_faces=retained_original_RAW_free_faces+new_bridge_faces。
原 POST 父删面、桥接面积、小角、孤岛与非流形均保留，不能用删面或新增
桥面掩盖分母变化。所有原数值、状态和文字字段完整复制在 source_per_output.csv。

这是 COMMON 历史起点上同池 Native 6000→SEAM 3000 的整体课程比较。
任务池、训练量、布局表示和后处理阶段各有边界，不将结果单归因于数据量、
完整座面、AdamW、表示或某项几何操作；不与 Chair 作类别单变量因果比较。
无来源唯一正确形状的假设，也无自动视觉成功标签。
