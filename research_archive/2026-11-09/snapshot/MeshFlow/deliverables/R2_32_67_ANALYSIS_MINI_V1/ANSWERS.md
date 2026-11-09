本页为 DERIVED：仅复算既有表。S=USER_SMALL，E=USER_EXPANDED_67；Δ=E−S；pp=百分点。下列来源均相对于原实验 `experiments/r2_user_data_scaling_32_67_v1/`；完整路径和哈希见 `provenance.json`。原值未改，完整 ID 在相应逐例表中。

1. **主表复现。** NEW32/OLD6、RAW/POST、三模型的 96 项指标共 1152 个父等权均值全部一致（容差 1e−12），最大绝对差 `1.4210854715202004e-14`，为浮点累加顺序差异。按噪声→条件→父复算，HIST 单独保留。见 `derived/aggregate_recomputed.csv`，键 `(panel,stage,label,metric)`；原来源 `evaluation/per_output.csv`、`report_tables/absolute_parent_equal.csv` 的 ALL 行；未覆盖旧表。

2. **接口 7→7，实际为共同3、丢失4、新增4、均无117。** 全部128请求的边数、长度、方向可达面积和原 C 采用标签在 `derived/interface_pairs_NEW32_POST.csv`（键 `request_id`，内含完整 parent/condition/noise）。共同：3c1dc5… BACK_FULL n1、610b16… BACK_FULL n0、c7c707… SEAT_FULL n0；丢失：64871d… SEAT_FULL n0/n1、9183a9… BACK_HALF n0、ef463d… BACK_FULL n1；新增：804c35… BACK_HALF n0、9183a9… BACK_FULL n1、9d81bc… BACK_HALF n1、ef463d… BACK_FULL n0。64871dc28a21843ad504e40666187f4e 的两份 SEAT_FULL：S 边数5/6、长度7.465030840787905/7.619469136013793、方向可达面积比例0.469767048210502/0.4791001948360448；E 三项均0；两模型两噪声 C采用均YES。因此采用标签没有表达这次连接丢失。来源：`evaluation/per_output.csv` 与 `evaluation/AI_PROXY_visual_review.csv`，按同一请求、POST 配对，未用报告文字回填。

3. **边际数量掩盖双向变化。** 下表矩阵行=S、列=E，各按 YES/NO/UNCERTAIN 排列，分号分隔三行；不把不确定并入成败。

| 观察 | 全128矩阵 | 全30份 SEAT_HALF 矩阵 |
|---|---|---|
| C采用 | 53,1,5; 4,22,4; 6,10,23 | 1,0,1; 0,10,1; 1,4,12 |
| 整体完整 | 106,3,4; 3,2,0; 4,0,6 | 22,2,1; 0,0,0; 0,0,5 |
| 支撑缺失 | 1,2,0; 2,116,2; 0,2,3 | 0,0,0; 1,25,2; 0,0,2 |
| 碎片/折叠层 | 17,19,2; 9,62,6; 0,4,9 | 8,3,0; 3,14,1; 0,0,1 |

总体 C采用59→63：保留53、丢失6个YES、新增10个YES；完整YES均113但丢失/新增各7。SEAT_HALF采用YES均2，只有1个共同YES；完整YES25→22。见 `derived/visual_transitions.csv`（键 `condition_type,field,SMALL_state,EXP_state`）、对应 request 列表 JSON，以及 `SEAT_HALF_all30_visual_pairs.csv`（键 request_id）。全8项、四类型均保留，无法配对/非三态记录0；来源为原840行 AI_PROXY 图审，未重新看图。

4. **I反转主要对应dup分量增加，不能归因到单独D0/GEO。** 父等权 ΔI：RAW −0.4398345947265625pp → POST +0.6481170654296875pp，变化 +1.08795166015625pp。由 `I=cross+dup−intersection`，变化分别 −0.5748748779296875、+1.5399932861328125、−0.122833251953125pp；减去交集变化后精确闭合，逐例残差0。18/32父的差值向正方向移动；最大三父为 c7c7079b7555bf075e994da5ba753c34、672e20cc6ffa29d41c6aa36e5af1449、ef463d298bb401b3ce1493daf6a835b0。仅第一个父的自身ΔI从负转正。最大请求为 c7c707… SEAT_FULL n1：变化48.33984375pp，其中cross −1.220703125、dup +50.439453125、交集 +0.87890625pp；其次672e20… BACK_FULL n0，变化45.556640625pp。所有正负贡献见 `derived/I_stage_shift_per_request.csv`（128请求）及 `I_stage_shift_per_parent.csv`（32父）；来源为原逐输出表四个对应模型/阶段记录。未保存子阶段评分，不能分配责任。

5. **DEV分组互斥完备，未新增“最坏条件决定父”规则。** 原条件标签按 FF 已确认非法对优先、再UNKNOWN、否则NO_DETECTED_PAIR_PROBLEM；本组每父两条件恰好标签一致，故11/20/1父对应22/40/2条件。UNKNOWN完整ID为 `ef463d298bb401b3ce1493daf6a835b0`，实测 POST ΔI=`0.113525390625`，即 **+11.3525390625pp**，不是四舍五入反推的+11.359pp。见 `derived/dev_quality_parent_check.csv`（parent_uid）、`conditions.csv`（condition_id）；来源 `data/condition_manifest.json`、原逐输出表，实际函数见 `protocol_and_code/original/source_quality_scan.py::partition`、`source_data_freeze.py::emit` 及函数索引。

6. **两池均有目标风险，不能据此确认主要原因。** COMMON32与新增35：ILLEGAL父22/32 vs24/35、条件64/91 vs61/103；按已有面对ID去重的 FF 问题723 vs843，原Y精确重复涉及父9 vs15；父等权Y小角<10°比例49.283131287627646% vs46.14772798064534%（不同于输出<5°口径）。真实曝光中，ILLEGAL条件占S共同32的68.75%、E共同32的约68.7519%、E新增35的约68.5688%；文件比例不能代替曝光。原几何检测器的shared_edge_contact面对均值12 vs11.941747572815533，不是索引共边或合法唯一接口。来源实际索引/合并后/合法唯一接口与边界匹配均NOT_MEASURED。因此可支持检验训练目标质量，尚不足以说新增目标更差或是主要原因。见 `source_quality_and_interfaces.csv`（parent_uid,condition_id）、`source_events.json`（源面对事件）、`derived/training_quality_exposure.csv`（arm,parent_role,source_quality）；来源为既有 source quality JSON、条件manifest和逐样本训练日志，原路径逐行可追溯。
