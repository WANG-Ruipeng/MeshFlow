# CHAIR_DOMAIN_CONTROL_REPORT

实际新增训练更新：{'T1': 3000, 'Geo': 3000, 'Full': 3000}；实际完整生成 582/582（登记 600，几何不适用 18）；固定 teacher 288/288。所有计数来自实际账本。正式终点累计条件步数 4000，新任务 phase_step=3000。

Native=T1；Native+Geo=Geo；Native+Geo+Joint=Full。各臂从自己的旧 step1000 模型及完整优化器出发。研究主干选用 Native+Geo 不等于预先认定它整体最优。

补全结果按父对象等权，并分训练父/已观察验证父、20%/40%面数条件及实际条件面积。A 表保留 3 个不可构造的40%条件，每臂6个生成格标为 NOT_APPLICABLE；不删除父对象或换对象凑数。

| 模型 | 划分 | K档 | 有效父/请求父 | 有效case/请求 | remainder RMS (%L) | GT coverage | Gamma edge RMS (%L) | strict Gamma coverage | FF affected | small-angle fraction |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native+Geo+Joint | train | 0.2 | 8/8 | 16/16 | 5.8331 | 0.40564 | 3.87962 | 0.0135284 | 0.931611 | 0.215894 |
| Native+Geo+Joint | train | 0.4 | 7/8 | 14/16 | 3.34884 | 0.67034 | 1.87693 | 0.0546987 | 0.888668 | 0.354868 |
| Native+Geo+Joint | validation | 0.2 | 8/8 | 16/16 | 4.23319 | 0.318359 | 4.09741 | 0.00601917 | 0.935657 | 0.105404 |
| Native+Geo+Joint | validation | 0.4 | 6/8 | 12/16 | 2.57153 | 0.505208 | 3.85156 | 0.00790023 | 0.852883 | 0.0699329 |
| Native+Geo | train | 0.2 | 8/8 | 16/16 | 5.63938 | 0.411133 | 3.80769 | 0.00663173 | 0.931568 | 0.210615 |
| Native+Geo | train | 0.4 | 7/8 | 14/16 | 3.29512 | 0.709682 | 1.57876 | 0.0550418 | 0.883665 | 0.35511 |
| Native+Geo | validation | 0.2 | 8/8 | 16/16 | 4.20751 | 0.327148 | 4.21296 | 0.0078015 | 0.929906 | 0.107385 |
| Native+Geo | validation | 0.4 | 6/8 | 12/16 | 2.60873 | 0.511882 | 4.0091 | 0.00847367 | 0.840643 | 0.0793172 |
| Native | train | 0.2 | 8/8 | 16/16 | 5.38322 | 0.398315 | 3.54498 | 0.0108638 | 0.926093 | 0.250143 |
| Native | train | 0.4 | 7/8 | 14/16 | 3.90346 | 0.648298 | 2.08506 | 0.0740714 | 0.891105 | 0.350427 |
| Native | validation | 0.2 | 8/8 | 16/16 | 4.17062 | 0.297974 | 3.99196 | 0.00541261 | 0.930067 | 0.0961967 |
| Native | validation | 0.4 | 6/8 | 12/16 | 2.57693 | 0.490723 | 3.93657 | 0.00943484 | 0.840303 | 0.061412 |

几何响应只测自由部分三角角坐标的稳健跨度 q95−q05。B_width_response.csv 分列绝对宽度误差、相对自身alpha=1的残差、斜率、单调、Y/Z跨度和三轴中位坐标误差。极小基准宽度的比值/斜率为 UNDEFINED。C精确保留是采样构造性质，不计为学得的控制精度。

B_cross_condition_scores.csv 是固定对应GT点到生成自由三角面的单向诊断，距离使用共同L0；与原面积加权双向RMS严格分列。原评分器仍使用对应alpha目标的L_alpha，新增L0列只转换距离单位，没有缩放输出坐标。

起点已有多少响应、训练新增多少，只能查看 shared_start_endpoint_* 和 B_width_start_endpoint_* 的真正共有第一噪声配对。终点额外噪声与40%条件不混入起点配对；均值与中位数均在先算配对父差后汇总，不能拿两组各自中位数相减。

No-response和Post-scale是零模型前向解析参照。Post-scale的alpha_hat只由编辑前后C对应X坐标拟合；它额外持有编辑前C与旧输出，输入合同不等同于只见当前C的生成器。两类候选使用当前C附接并对当前目标评分，CPU候选/评分/缓存命中单独记账，不算新增独立生成。即使模型响应接近理想斜率，也仅能称域内几何条件响应；是否优于Post-scale要同时看绝对几何、覆盖、接口与相交。

Geo增量与Joint增量见 matched_budget_summary.csv：按相同phase、相同输入、相同父对象配对，报告均值、中位配对差、改善父数、最坏父与请求分母。Joint是否保住覆盖，必须同时读取 coverage、严格接口、CF/FF和角质量，不能仅用JEdge5或低RMS判断。

20%/40%效果见 A_context_summary.csv 和 A_context_area_strata.csv；不可用40%格保留NA原因，不把缺失填零或解释为模型失败。训练父只表示新C组合未用于已知微调，完整物体和许多面已见；验证父与原留出均已被研究观察。

C_holdout_regression.csv 是已观察留出回归。旧1000/5000只在checkpoint/raw/实际输入/评分器身份全匹配时读取原分数，缺失为NA。新累计4000与旧5000属于不同预算和训练任务的历史描述，不能当作同预算数据消融。

本轮同时改变C任务分布和宽度训练，没有“仅更多C”或“仅宽度”独立训练臂；不能分离两者因果贡献。宽度alpha=.9/1/1.1为训练范围内部及边界，不是范围外推。无可信语义标签时只称规范X向、free整体几何响应，不声称部件姿态/连接重构或任意局部编辑。训练seed只有1，结果是开发性域内验证。

CPU控制分析：{'cross_condition_group_attempts': 72, 'corresponding_point_distance_attempts': 1080, 'corresponding_point_distance_returns': 1080, 'cross_condition_group_returns': 72, 'analytic_candidate_attempts': 432, 'analytic_score_attempts': 288, 'endpoint_scores_attempted': 288, 'prediction_surface_sample_calls': 288, 'remainder_surface_distance_calls': 576, 'remainder_point_triangle_pairs': 45047808, 'fixed_GT_coverage_distance_calls': 288, 'fixed_GT_coverage_point_triangle_pairs': 22523904, 'Gamma_surface_calls_attempted': 288, 'Gamma_surface_point_triangle_pairs': 23018040, 'Gamma_surface_calls_returned': 288, 'gamma_to_free_edges_calls_attempted': 288, 'gamma_to_free_edges_point_segment_pairs': 69054120, 'gamma_to_free_edges_zero_length_occurrences': 0, 'gamma_to_free_edges_calls_returned': 288, 'free_shape_quality_calls': 288, 'GT_free_area_calculations': 288, 'verified_pair_tests': 5182020, 'primary_pair_tests': 5182020, 'independent_pair_tests': 5182020, 'independent_AABB_disjoint': 4860780, 'endpoint_scores_returned': 288, 'analytic_score_returns': 288, 'analytic_candidate_returns': 432, 'analytic_score_cache_hits': 144}。所有解析候选和3×3评分均为复用输出的CPU工作，零新增模型前向。图件见 figures/manifest.json；B的alpha之间使用同一原坐标镜头与共同范围，GT覆盖颜色来自实际评分。

全局水密：NOT_RUN。评分有效不表示质量通过；严格接合、非法相交和角质量问题按原值保留。预算结束停止，不自动更换公共默认模型或启动下一轮。

数据、代码、检查点与真实执行总账以 protocol.json、task_manifest.json、checkpoint_manifest.json、analysis_ledger.json、evaluation/ 与 training/ 目录为准。

以下为实际结果的直接回答。差值均先在父内对相同输入配对，再对父等权；“改善父”按原指标方向计数，没有临时成功阈值。

新上下文上的Geo增量与Joint增量：
train、K=20%、Geo−Native：RMS差 0.256167（改善3/8有效父，请求8父） %L；GT覆盖差 1.28174（改善6/8有效父，请求8父） 点；Gamma边RMS差 0.262711（改善4/8有效父，请求8父） %L；CF面比例差 -0.699695（改善3/8有效父，请求8父） 点；FF面比例差 0.547477（改善3/8有效父，请求8父） 点；小角面差 -3.9528（改善6/8有效父，请求8父） 点。所列指标收益与代价并存。
train、K=20%、Joint−Geo：RMS差 0.193717（改善1/8有效父，请求8父） %L；GT覆盖差 -0.549316（改善2/8有效父，请求8父） 点；Gamma边RMS差 0.0719264（改善4/8有效父，请求8父） %L；CF面比例差 0.0806714（改善2/8有效父，请求8父） 点；FF面比例差 0.00435465（改善4/8有效父，请求8父） 点；小角面差 0.527947（改善2/8有效父，请求8父） 点。所列指标没有同向均值增益。
train、K=40%、Geo−Native：RMS差 -0.608336（改善6/7有效父，请求8父） %L；GT覆盖差 6.13839（改善4/7有效父，请求8父） 点；Gamma边RMS差 -0.5063（改善6/7有效父，请求8父） %L；CF面比例差 -4.52296（改善5/7有效父，请求8父） 点；FF面比例差 -0.74409（改善4/7有效父，请求8父） 点；小角面差 0.468312（改善3/7有效父，请求8父） 点。所列指标收益与代价并存。
train、K=40%、Joint−Geo：RMS差 0.053719（改善5/7有效父，请求8父） %L；GT覆盖差 -3.93415（改善2/7有效父，请求8父） 点；Gamma边RMS差 0.298176（改善0/7有效父，请求8父） %L；CF面比例差 2.28125（改善1/7有效父，请求8父） 点；FF面比例差 0.500303（改善2/7有效父，请求8父） 点；小角面差 -0.0242032（改善3/7有效父，请求8父） 点。所列指标收益与代价并存。
validation、K=20%、Geo−Native：RMS差 0.036888（改善3/8有效父，请求8父） %L；GT覆盖差 2.91748（改善7/8有效父，请求8父） 点；Gamma边RMS差 0.221006（改善2/8有效父，请求8父） %L；CF面比例差 0.19023（改善1/8有效父，请求8父） 点；FF面比例差 -0.0161196（改善4/8有效父，请求8父） 点；小角面差 1.11886（改善2/8有效父，请求8父） 点。所列指标收益与代价并存。
validation、K=20%、Joint−Geo：RMS差 0.0256819（改善4/8有效父，请求8父） %L；GT覆盖差 -0.878906（改善2/8有效父，请求8父） 点；Gamma边RMS差 -0.115555（改善4/8有效父，请求8父） %L；CF面比例差 -0.13618（改善4/8有效父，请求8父） 点；FF面比例差 0.575098（改善2/8有效父，请求8父） 点；小角面差 -0.19809（改善4/8有效父，请求8父） 点。所列指标收益与代价并存。
validation、K=40%、Geo−Native：RMS差 0.0318015（改善3/6有效父，请求8父） %L；GT覆盖差 2.11589（改善5/6有效父，请求8父） 点；Gamma边RMS差 0.0725315（改善3/6有效父，请求8父） %L；CF面比例差 -0.440406（改善4/6有效父，请求8父） 点；FF面比例差 0.034024（改善3/6有效父，请求8父） 点；小角面差 1.79052（改善0/6有效父，请求8父） 点。所列指标收益与代价并存。
validation、K=40%、Joint−Geo：RMS差 -0.0371987（改善4/6有效父，请求8父） %L；GT覆盖差 -0.667318（改善1/6有效父，请求8父） 点；Gamma边RMS差 -0.157536（改善5/6有效父，请求8父） %L；CF面比例差 -0.941097（改善4/6有效父，请求8父） 点；FF面比例差 1.22395（改善0/6有效父，请求8父） 点；小角面差 -0.938436（改善4/6有效父，请求8父） 点。所列指标收益与代价并存。

free是否随当前C宽度响应：以下斜率与单调统计只来自free，复制的C不进入宽度。斜率1为理想全局缩放响应；绝对误差与几何覆盖仍须共同解释。
Native train终点：斜率均值 0.740303，|斜率−1| 0.543342，单调比例 1；alpha=.9/1.1的绝对宽度误差/L0分别 0.0326203/0.0701595。斜率有效case 8/8，有效父 4/4；Y/Z尺度与坐标中位数副作用另列，不把未知项填成理想响应。
Native validation终点：斜率均值 0.121278，|斜率−1| 0.878722，单调比例 0.625；alpha=.9/1.1的绝对宽度误差/L0分别 0.120635/0.207528。斜率有效case 8/8，有效父 4/4；Y/Z尺度与坐标中位数副作用另列，不把未知项填成理想响应。
Native+Geo train终点：斜率均值 0.837884，|斜率−1| 0.566769，单调比例 0.875；alpha=.9/1.1的绝对宽度误差/L0分别 0.0480683/0.0732552。斜率有效case 8/8，有效父 4/4；Y/Z尺度与坐标中位数副作用另列，不把未知项填成理想响应。
Native+Geo validation终点：斜率均值 0.080739，|斜率−1| 0.919261，单调比例 0.25；alpha=.9/1.1的绝对宽度误差/L0分别 0.12131/0.210832。斜率有效case 8/8，有效父 4/4；Y/Z尺度与坐标中位数副作用另列，不把未知项填成理想响应。
Native+Geo+Joint train终点：斜率均值 0.647605，|斜率−1| 0.568211，单调比例 0.875；alpha=.9/1.1的绝对宽度误差/L0分别 0.0334413/0.0746911。斜率有效case 8/8，有效父 4/4；Y/Z尺度与坐标中位数副作用另列，不把未知项填成理想响应。
Native+Geo+Joint validation终点：斜率均值 0.0613561，|斜率−1| 0.938644，单调比例 0.375；alpha=.9/1.1的绝对宽度误差/L0分别 0.12071/0.211676。斜率有效case 8/8，有效父 4/4；Y/Z尺度与坐标中位数副作用另列，不把未知项填成理想响应。

响应是否在起点存在及新增训练贡献：下列仅使用真正共有的第一噪声、三个alpha，每父等权；不混入终点第二噪声。
Native train共享输入：宽度绝对误差/L0 0.0659107→0.0467136；|响应残差| 0.0792043→0.0357815；|斜率−1| 1.18806→0.536722。宽度误差配对变化 -0.019197（改善2/4有效父，请求4父）；这里有 4 个宽度误差可比父，起点已有的响应不归功于本轮训练。
Native validation共享输入：宽度绝对误差/L0 0.140871→0.154334；|响应残差| 0.0585134→0.0510558；|斜率−1| 0.784432→0.765837。宽度误差配对变化 0.0134632（改善2/4有效父，请求4父）；这里有 4 个宽度误差可比父，起点已有的响应不归功于本轮训练。
Native+Geo train共享输入：宽度绝对误差/L0 0.0577357→0.0488595；|响应残差| 0.0762104→0.0378909；|斜率−1| 1.14316→0.568364。宽度误差配对变化 -0.00887619（改善2/4有效父，请求4父）；这里有 4 个宽度误差可比父，起点已有的响应不归功于本轮训练。
Native+Geo validation共享输入：宽度绝对误差/L0 0.13946→0.156422；|响应残差| 0.0671634→0.0532416；|斜率−1| 1.00745→0.798624。宽度误差配对变化 0.0169622（改善2/4有效父，请求4父）；这里有 4 个宽度误差可比父，起点已有的响应不归功于本轮训练。
Native+Geo+Joint train共享输入：宽度绝对误差/L0 0.0609992→0.049933；|响应残差| 0.0749065→0.0408129；|斜率−1| 1.1236→0.612193。宽度误差配对变化 -0.0110662（改善2/4有效父，请求4父）；这里有 4 个宽度误差可比父，起点已有的响应不归功于本轮训练。
Native+Geo+Joint validation共享输入：宽度绝对误差/L0 0.13588→0.157938；|响应残差| 0.0629885→0.0564735；|斜率−1| 0.944828→0.847103。宽度误差配对变化 0.0220583（改善2/4有效父，请求4父）；这里有 4 个宽度误差可比父，起点已有的响应不归功于本轮训练。

相对零前向解析参照：差值为当前模型减参照，在同父、同噪声、同当前alpha上配对，三alpha平均。误差负值、覆盖正值才有利于模型。Post-scale额外知道编辑前C与旧输出，因此只能比较配对编辑场景的实效。
Native train − No-response：宽度绝对误差/L0差 -0.010539（改善4/4有效父，请求4父）；原双向RMS差 -0.0094951（改善3/4有效父，请求4父） %L；GT覆盖差 3.80859（改善4/4有效父，请求4父） 点；严格Gamma覆盖差 -0.00631843（改善2/4有效父，请求4父） 点；FF面比例差 -0.602218（改善3/4有效父，请求4父） 点。
Native train − Post-scale：宽度绝对误差/L0差 -0.000105433（改善2/4有效父，请求4父）；原双向RMS差 0.223303（改善1/4有效父，请求4父） %L；GT覆盖差 -3.36914（改善1/4有效父，请求4父） 点；严格Gamma覆盖差 0.276287（改善2/4有效父，请求4父） 点；FF面比例差 -0.602218（改善3/4有效父，请求4父） 点。
Native validation − No-response：宽度绝对误差/L0差 7.69268e-05（改善2/4有效父，请求4父）；原双向RMS差 0.0586376（改善2/4有效父，请求4父） %L；GT覆盖差 -0.00813802（改善2/4有效父，请求4父） 点；严格Gamma覆盖差 -0.0317025（改善1/4有效父，请求4父） 点；FF面比例差 -0.0306976（改善2/4有效父，请求4父） 点。
Native validation − Post-scale：宽度绝对误差/L0差 7.69266e-05（改善2/4有效父，请求4父）；原双向RMS差 0.0893453（改善1/4有效父，请求4父） %L；GT覆盖差 -0.61849（改善1/4有效父，请求4父） 点；严格Gamma覆盖差 0.0724361（改善2/4有效父，请求4父） 点；FF面比例差 -0.0306976（改善2/4有效父，请求4父） 点。
Native+Geo train − No-response：宽度绝对误差/L0差 8.73787e-05（改善2/4有效父，请求4父）；原双向RMS差 -0.111794（改善2/4有效父，请求4父） %L；GT覆盖差 4.6875（改善3/4有效父，请求4父） 点；严格Gamma覆盖差 0.432726（改善4/4有效父，请求4父） 点；FF面比例差 -0.854649（改善2/4有效父，请求4父） 点。
Native+Geo train − Post-scale：宽度绝对误差/L0差 0.0106069（改善0/4有效父，请求4父）；原双向RMS差 0.130309（改善0/4有效父，请求4父） %L；GT覆盖差 -3.34473（改善1/4有效父，请求4父） 点；严格Gamma覆盖差 0.317912（改善4/4有效父，请求4父） 点；FF面比例差 -0.854649（改善2/4有效父，请求4父） 点。
Native+Geo validation − No-response：宽度绝对误差/L0差 -0.00206791（改善4/4有效父，请求4父）；原双向RMS差 0.10587（改善3/4有效父，请求4父） %L；GT覆盖差 0.935872（改善3/4有效父，请求4父） 点；严格Gamma覆盖差 -0.0891603（改善2/4有效父，请求4父） 点；FF面比例差 -0.65751（改善4/4有效父，请求4父） 点。
Native+Geo validation − Post-scale：宽度绝对误差/L0差 -0.00206791（改善4/4有效父，请求4父）；原双向RMS差 0.134296（改善3/4有效父，请求4父） %L；GT覆盖差 0.301107（改善3/4有效父，请求4父） 点；严格Gamma覆盖差 -0.282646（改善1/4有效父，请求4父） 点；FF面比例差 -0.65751（改善4/4有效父，请求4父） 点。
Native+Geo+Joint train − No-response：宽度绝对误差/L0差 -0.00414793（改善3/4有效父，请求4父）；原双向RMS差 -0.298785（改善3/4有效父，请求4父） %L；GT覆盖差 3.4668（改善3/4有效父，请求4父） 点；严格Gamma覆盖差 0.316327（改善3/4有效父，请求4父） 点；FF面比例差 -0.0651503（改善2/4有效父，请求4父） 点。
Native+Geo+Joint train − Post-scale：宽度绝对误差/L0差 0.00816106（改善1/4有效父，请求4父）；原双向RMS差 -0.0374572（改善1/4有效父，请求4父） %L；GT覆盖差 -4.21549（改善2/4有效父，请求4父） 点；严格Gamma覆盖差 0.284359（改善2/4有效父，请求4父） 点；FF面比例差 -0.0651503（改善2/4有效父，请求4父） 点。
Native+Geo+Joint validation − No-response：宽度绝对误差/L0差 0.000845184（改善2/4有效父，请求4父）；原双向RMS差 0.129805（改善1/4有效父，请求4父） %L；GT覆盖差 1.02539（改善3/4有效父，请求4父） 点；严格Gamma覆盖差 -0.137545（改善2/4有效父，请求4父） 点；FF面比例差 0.378427（改善2/4有效父，请求4父） 点。
Native+Geo+Joint validation − Post-scale：宽度绝对误差/L0差 0.000845183（改善2/4有效父，请求4父）；原双向RMS差 0.174868（改善1/4有效父，请求4父） %L；GT覆盖差 0.577799（改善3/4有效父，请求4父） 点；严格Gamma覆盖差 -0.355334（改善1/4有效父，请求4父） 点；FF面比例差 0.378427（改善2/4有效父，请求4父） 点。

与Post-scale持平或更差的项目不能被解释为超越简单仿射编辑。即使某些项目更好，单次全形状X向仿射任务仍只支持域内整体几何响应，不能证明语义部件、连接重构或复杂结构推理。

20%与40%条件是否不同：以下是各自冻结新C/自身free补集上的终点绝对数值，保留不适用分母。不同K改变了已知面与free目标，不能把这些描述当成仅信息量的严格因果对照。
Native train：20%/40%的RMS为 5.38322/3.90346 %L，GT覆盖为 0.398315/0.648298；有效父/请求父分别 8/8 与 7/8。实际C面积分层另存A_context_area_strata.csv。
Native validation：20%/40%的RMS为 4.17062/2.57693 %L，GT覆盖为 0.297974/0.490723；有效父/请求父分别 8/8 与 6/8。实际C面积分层另存A_context_area_strata.csv。
Native+Geo train：20%/40%的RMS为 5.63938/3.29512 %L，GT覆盖为 0.411133/0.709682；有效父/请求父分别 8/8 与 7/8。实际C面积分层另存A_context_area_strata.csv。
Native+Geo validation：20%/40%的RMS为 4.20751/2.60873 %L，GT覆盖为 0.327148/0.511882；有效父/请求父分别 8/8 与 6/8。实际C面积分层另存A_context_area_strata.csv。
Native+Geo+Joint train：20%/40%的RMS为 5.8331/3.34884 %L，GT覆盖为 0.40564/0.67034；有效父/请求父分别 8/8 与 7/8。实际C面积分层另存A_context_area_strata.csv。
Native+Geo+Joint validation：20%/40%的RMS为 4.23319/2.57153 %L，GT覆盖为 0.318359/0.505208；有效父/请求父分别 8/8 与 6/8。实际C面积分层另存A_context_area_strata.csv。

已观察留出是否新增退化：以下新累计4000减源1000，使用原实际C/噪声与原评分器。未设置临时“明显”阈值；相交、覆盖、RMS与角质量的方向分别保留。
Native：RMS差 -0.00889761（改善3/8有效父，请求8父） %L；GT覆盖差 1.7334（改善5/8有效父，请求8父） 点；严格Gamma覆盖差 0.150558（改善4/8有效父，请求8父） 点；CF面比例差 -2.65234（改善4/8有效父，请求8父） 点；FF面比例差 1.57034（改善2/8有效父，请求8父） 点；小角面差 -0.479432（改善4/8有效父，请求8父） 点。所列指标收益与代价并存。
Native+Geo：RMS差 -0.125452（改善5/8有效父，请求8父） %L；GT覆盖差 1.50146（改善5/8有效父，请求8父） 点；严格Gamma覆盖差 0.0671207（改善3/8有效父，请求8父） 点；CF面比例差 -2.10001（改善4/8有效父，请求8父） 点；FF面比例差 1.39265（改善3/8有效父，请求8父） 点；小角面差 -0.0605261（改善4/8有效父，请求8父） 点。所列指标收益与代价并存。
Native+Geo+Joint：RMS差 -0.116626（改善5/8有效父，请求8父） %L；GT覆盖差 1.83716（改善5/8有效父，请求8父） 点；严格Gamma覆盖差 0.127131（改善3/8有效父，请求8父） 点；CF面比例差 -1.77164（改善4/8有效父，请求8父） 点；FF面比例差 1.6826（改善2/8有效父，请求8父） 点；小角面差 -0.210943（改善4/8有效父，请求8父） 点。所列指标收益与代价并存。

Inherited historical BF16 permutation FAIL: 10/1008 coordinates; max_abs=.064453125; atol=.015625; rtol=.03125. Not rerun, not reclassified PASS. Repeated complete generations: 0 (none claimed).

| B endpoint model | split | alpha | valid parents/requested | width abs error / L0 | slope | abs slope error | monotonic fraction | correct-column advantage / L0 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Native+Geo+Joint | train | 0.9 | 4/4 | 0.0334413 | 0.647605 | 0.568211 | 0.875 | 0.0111859 |
| Native+Geo+Joint | train | 1.0 | 4/4 | 0.0418246 | 0.647605 | 0.568211 | 0.875 | 0.00384614 |
| Native+Geo+Joint | train | 1.1 | 4/4 | 0.0746911 | 0.647605 | 0.568211 | 0.875 | -0.00405119 |
| Native+Geo+Joint | validation | 0.9 | 4/4 | 0.12071 | 0.0613561 | 0.938644 | 0.375 | 0.0138328 |
| Native+Geo+Joint | validation | 1.0 | 4/4 | 0.164925 | 0.0613561 | 0.938644 | 0.375 | 0.00100943 |
| Native+Geo+Joint | validation | 1.1 | 4/4 | 0.211676 | 0.0613561 | 0.938644 | 0.375 | -0.0130457 |
| Native+Geo | train | 0.9 | 4/4 | 0.0480683 | 0.837884 | 0.566769 | 0.875 | 0.012607 |
| Native+Geo | train | 1.0 | 4/4 | 0.0447514 | 0.837884 | 0.566769 | 0.875 | 0.00381953 |
| Native+Geo | train | 1.1 | 4/4 | 0.0732552 | 0.837884 | 0.566769 | 0.875 | -0.00435966 |
| Native+Geo | validation | 0.9 | 4/4 | 0.12131 | 0.080739 | 0.919261 | 0.25 | 0.0137544 |
| Native+Geo | validation | 1.0 | 4/4 | 0.169173 | 0.080739 | 0.919261 | 0.25 | 0.00101765 |
| Native+Geo | validation | 1.1 | 4/4 | 0.210832 | 0.080739 | 0.919261 | 0.25 | -0.0130632 |
| Native | train | 0.9 | 4/4 | 0.0326203 | 0.740303 | 0.543342 | 1 | 0.0111317 |
| Native | train | 1.0 | 4/4 | 0.0515481 | 0.740303 | 0.543342 | 1 | 0.00385415 |
| Native | train | 1.1 | 4/4 | 0.0701595 | 0.740303 | 0.543342 | 1 | -0.00414404 |
| Native | validation | 0.9 | 4/4 | 0.120635 | 0.121278 | 0.878722 | 0.625 | 0.0136889 |
| Native | validation | 1.0 | 4/4 | 0.163966 | 0.121278 | 0.878722 | 0.625 | 0.00103286 |
| Native | validation | 1.1 | 4/4 | 0.207528 | 0.121278 | 0.878722 | 0.625 | -0.0127434 |

| Paired endpoint increment | table | split | K fraction | metric | mean parent delta | median parent delta | improved/valid/requested parents | worst parent delta |
| --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: |
| Native+Geo+Joint_minus_Native | A | train | 0.2 | remainder_symmetric_RMS_bbox_pct | 0.449884 | 0.303614 | 1/8/8 | 2.53785 |
| Native+Geo+Joint_minus_Native | A | train | 0.2 | Gamma_to_free_edges_RMS_bbox_pct | 0.334638 | 0.0969073 | 4/8/8 | 2.38118 |
| Native+Geo+Joint_minus_Native | A | train | 0.2 | fixed_R12_coverage_within_1pct_fraction | 0.00732422 | -0.000976562 | 4/8/8 | -0.0605469 |
| Native+Geo+Joint_minus_Native | A | train | 0.2 | Gamma_to_free_edges_coverage_0.001_fraction | 0.00266457 | -0.000952459 | 2/8/8 | -0.0148546 |
| Native+Geo+Joint_minus_Native | A | train | 0.4 | remainder_symmetric_RMS_bbox_pct | -0.554618 | -0.125301 | 5/7/8 | 0.319845 |
| Native+Geo+Joint_minus_Native | A | train | 0.4 | Gamma_to_free_edges_RMS_bbox_pct | -0.208124 | -0.25222 | 5/7/8 | 0.854226 |
| Native+Geo+Joint_minus_Native | A | train | 0.4 | fixed_R12_coverage_within_1pct_fraction | 0.0220424 | 0.00390625 | 4/7/8 | -0.0214844 |
| Native+Geo+Joint_minus_Native | A | train | 0.4 | Gamma_to_free_edges_coverage_0.001_fraction | -0.0193727 | -0.0147754 | 1/7/8 | -0.070966 |
| Native+Geo+Joint_minus_Native | A | validation | 0.2 | remainder_symmetric_RMS_bbox_pct | 0.0625699 | 0.0880661 | 3/8/8 | 0.444415 |
| Native+Geo+Joint_minus_Native | A | validation | 0.2 | Gamma_to_free_edges_RMS_bbox_pct | 0.105451 | 0.139678 | 2/8/8 | 0.928386 |
| Native+Geo+Joint_minus_Native | A | validation | 0.2 | fixed_R12_coverage_within_1pct_fraction | 0.0203857 | 0.0136719 | 7/8/8 | -0.015625 |
| Native+Geo+Joint_minus_Native | A | validation | 0.2 | Gamma_to_free_edges_coverage_0.001_fraction | 0.000606566 | 0 | 3/8/8 | -0.00191653 |
| Native+Geo+Joint_minus_Native | A | validation | 0.4 | remainder_symmetric_RMS_bbox_pct | -0.00539725 | -0.0407642 | 4/6/8 | 0.377139 |
| Native+Geo+Joint_minus_Native | A | validation | 0.4 | Gamma_to_free_edges_RMS_bbox_pct | -0.0850046 | -0.0692769 | 4/6/8 | 0.612328 |
| Native+Geo+Joint_minus_Native | A | validation | 0.4 | fixed_R12_coverage_within_1pct_fraction | 0.0144857 | 0.0131836 | 5/6/8 | 0 |
| Native+Geo+Joint_minus_Native | A | validation | 0.4 | Gamma_to_free_edges_coverage_0.001_fraction | -0.00153461 | 0.000954576 | 3/6/8 | -0.0189777 |
| Native+Geo+Joint_minus_Native+Geo | A | train | 0.2 | remainder_symmetric_RMS_bbox_pct | 0.193717 | 0.187948 | 1/8/8 | 0.451948 |
| Native+Geo+Joint_minus_Native+Geo | A | train | 0.2 | Gamma_to_free_edges_RMS_bbox_pct | 0.0719264 | 0.0756445 | 4/8/8 | 0.552956 |
| Native+Geo+Joint_minus_Native+Geo | A | train | 0.2 | fixed_R12_coverage_within_1pct_fraction | -0.00549316 | -0.00878906 | 2/8/8 | -0.0244141 |
| Native+Geo+Joint_minus_Native+Geo | A | train | 0.2 | Gamma_to_free_edges_coverage_0.001_fraction | 0.00689665 | 0.000490379 | 4/8/8 | -0.00294755 |
| Native+Geo+Joint_minus_Native+Geo | A | train | 0.4 | remainder_symmetric_RMS_bbox_pct | 0.053719 | -0.00570825 | 5/7/8 | 1.09021 |
| Native+Geo+Joint_minus_Native+Geo | A | train | 0.4 | Gamma_to_free_edges_RMS_bbox_pct | 0.298176 | 0.188414 | 0/7/8 | 1.12901 |
| Native+Geo+Joint_minus_Native+Geo | A | train | 0.4 | fixed_R12_coverage_within_1pct_fraction | -0.0393415 | -0.0136719 | 2/7/8 | -0.259766 |
| Native+Geo+Joint_minus_Native+Geo | A | train | 0.4 | Gamma_to_free_edges_coverage_0.001_fraction | -0.000343018 | 0 | 3/7/8 | -0.0196431 |
| Native+Geo+Joint_minus_Native+Geo | A | validation | 0.2 | remainder_symmetric_RMS_bbox_pct | 0.0256819 | 0.0138035 | 4/8/8 | 0.53399 |
| Native+Geo+Joint_minus_Native+Geo | A | validation | 0.2 | Gamma_to_free_edges_RMS_bbox_pct | -0.115555 | 0.0213902 | 4/8/8 | 0.16642 |
| Native+Geo+Joint_minus_Native+Geo | A | validation | 0.2 | fixed_R12_coverage_within_1pct_fraction | -0.00878906 | -0.00195312 | 2/8/8 | -0.0839844 |
| Native+Geo+Joint_minus_Native+Geo | A | validation | 0.2 | Gamma_to_free_edges_coverage_0.001_fraction | -0.00178233 | -0.000497213 | 1/8/8 | -0.00659119 |
| Native+Geo+Joint_minus_Native+Geo | A | validation | 0.4 | remainder_symmetric_RMS_bbox_pct | -0.0371987 | -0.0169891 | 4/6/8 | 0.229024 |
| Native+Geo+Joint_minus_Native+Geo | A | validation | 0.4 | Gamma_to_free_edges_RMS_bbox_pct | -0.157536 | -0.11068 | 5/6/8 | 0.769628 |
| Native+Geo+Joint_minus_Native+Geo | A | validation | 0.4 | fixed_R12_coverage_within_1pct_fraction | -0.00667318 | -0.00683594 | 1/6/8 | -0.0458984 |
| Native+Geo+Joint_minus_Native+Geo | A | validation | 0.4 | Gamma_to_free_edges_coverage_0.001_fraction | -0.000573438 | -0.000437807 | 3/6/8 | -0.00547623 |
| Native+Geo_minus_Native | A | train | 0.2 | remainder_symmetric_RMS_bbox_pct | 0.256167 | 0.0799779 | 3/8/8 | 2.0859 |
| Native+Geo_minus_Native | A | train | 0.2 | Gamma_to_free_edges_RMS_bbox_pct | 0.262711 | 0.108945 | 4/8/8 | 1.82823 |
| Native+Geo_minus_Native | A | train | 0.2 | fixed_R12_coverage_within_1pct_fraction | 0.0128174 | 0.012207 | 6/8/8 | -0.0498047 |
| Native+Geo_minus_Native | A | train | 0.2 | Gamma_to_free_edges_coverage_0.001_fraction | -0.00423208 | -0.00130962 | 1/8/8 | -0.0304793 |
| Native+Geo_minus_Native | A | train | 0.4 | remainder_symmetric_RMS_bbox_pct | -0.608336 | -0.128539 | 6/7/8 | 0.0855834 |
| Native+Geo_minus_Native | A | train | 0.4 | Gamma_to_free_edges_RMS_bbox_pct | -0.5063 | -0.50117 | 6/7/8 | 0.839355 |
| Native+Geo_minus_Native | A | train | 0.4 | fixed_R12_coverage_within_1pct_fraction | 0.0613839 | 0.0224609 | 4/7/8 | -0.0419922 |
| Native+Geo_minus_Native | A | train | 0.4 | Gamma_to_free_edges_coverage_0.001_fraction | -0.0190296 | -0.00875064 | 2/7/8 | -0.0652216 |
| Native+Geo_minus_Native | A | validation | 0.2 | remainder_symmetric_RMS_bbox_pct | 0.036888 | 0.0294245 | 3/8/8 | 0.513439 |
| Native+Geo_minus_Native | A | validation | 0.2 | Gamma_to_free_edges_RMS_bbox_pct | 0.221006 | 0.0941861 | 2/8/8 | 0.956064 |
| Native+Geo_minus_Native | A | validation | 0.2 | fixed_R12_coverage_within_1pct_fraction | 0.0291748 | 0.00976562 | 7/8/8 | -0.00390625 |
| Native+Geo_minus_Native | A | validation | 0.2 | Gamma_to_free_edges_coverage_0.001_fraction | 0.00238889 | 0.000971492 | 4/8/8 | -0.00194552 |
| Native+Geo_minus_Native | A | validation | 0.4 | remainder_symmetric_RMS_bbox_pct | 0.0318015 | 0.0599217 | 3/6/8 | 0.275028 |
| Native+Geo_minus_Native | A | validation | 0.4 | Gamma_to_free_edges_RMS_bbox_pct | 0.0725315 | 0.0155033 | 3/6/8 | 0.967782 |
| Native+Geo_minus_Native | A | validation | 0.4 | fixed_R12_coverage_within_1pct_fraction | 0.0211589 | 0.0175781 | 5/6/8 | -0.0195312 |
| Native+Geo_minus_Native | A | validation | 0.4 | Gamma_to_free_edges_coverage_0.001_fraction | -0.000961169 | 0.000485511 | 3/6/8 | -0.0135015 |
