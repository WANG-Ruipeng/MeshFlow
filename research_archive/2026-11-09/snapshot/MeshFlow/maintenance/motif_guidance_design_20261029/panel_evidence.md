# Motif guidance 设计用面板证据（2026-10-29）

本文件只整理现成EXPOSURE面板与预算，不是执行注册。未训练、加载模型、生成、创建噪声、评分或重新图审；原记录只读。四校准父与余19父的划分是本次拟议设计标签，不回写旧manifest的训练/DEV身份。

## 原面板与schema

- 直接入口：[EXPOSURE data/eval_manifest.json](../../experiments/r2_bench_four_routes_v1/exposure/data/eval_manifest.json)。schema=`bench_high_joint_coverage_evaluation_v1_panel`，status=`FROZEN`，46个`requests`、23个不同父、23个不同C条件，每C原n0/n1两噪声。
- 它逐字节复用[W原manifest](../../experiments/r2_bench_high_joint_coverage_v1/data/eval_manifest.json)；既有登记SHA256为`d261dbd7a3246613f994411f60fa082be933abe2b8c08fc167948dcc5fab8f18`。本次没有重算hash。
- 原manifest顶层`labels`仍是`B,W`，不是EXPOSURE身份。当前M/EXPOSURE标签及预算在[data/evaluation_plan.json](../../experiments/r2_bench_four_routes_v1/exposure/data/evaluation_plan.json)，schema=`bench_four_routes_exposure_evaluation_v1`。新的设计应旁置自己的标签和划分，不修改旧文件。
- 每条请求核心字段为`request_id, condition_id, parent_id, N, K, noise_index, seed, noise_npz, noise_sha256, inference_npz, metadata_npz, source_npz, query_descriptor, report_group, eval_group, split, closed_C, gallery_bounds, gallery_depth_bounds`及各文件身份。实际Gaussian数组来自`noise_npz`，不能只据seed重新随机生成。
- 部分HIGH条目没有`group`；应读`report_group`及原`split/eval_group`，不要因`group=null`丢掉HIGH条目。`NEW_WIDE_TRAIN`是历史组名，对EXPOSURE确为已训练父。
- `metric_fields`精确列出原177字段。原阶段为`RAW, BASE_POST, CLEAN_FINAL_POST`，后者对应报告FINAL；保留字段定义、分母、NA和失败原因，新增motif指标须独立命名。

## 拟校准四父与八个旧请求

请求ID严格由下表完整condition_id追加`__n0`或`__n1`得到。四父全部是EXPOSURE已训练父。

| 完整parent_id | report_group | K / N / free | 完整condition_id | 原seed n0 / n1 |
|---|---|---:|---|---:|
| `6b894c1c57ba921c98d10ca8df5ecbe7` | OLD | 4 / 133 / 129 | `6b894c1c57ba921c98d10ca8df5ecbe7__SEAT_SURFACE_1__V2` | 1766820057 / 1239493768 |
| `a8e987a6a6486b962d6c8e51365a5a87` | FOCUS | 39 / 189 / 150 | `a8e987a6a6486b962d6c8e51365a5a87__COMPLETE_SEAT_CANDIDATE__COVERAGE_V1` | 2406924831 / 3003755421 |
| `28b9f2908f0cbb179c21c139c0613648` | HIGH_TRAIN | 10 / 384 / 374 | `28b9f2908f0cbb179c21c139c0613648__ALL_SEAT_BOARD_UPPER_SURFACES__EXPOSURE_PREP_V1` | 1781792028 / 1249980453 |
| `bcb4f576747e10bdd7f6e86d7d335698` | NEW_WIDE_TRAIN | 2 / 520 / 518 | `bcb4f576747e10bdd7f6e86d7d335698__COMPLETE_FLAT_SEAT_TOP_SURFACE__HIGH_JOINT_V1` | 1239404145 / 1808988508 |

## 余下19父、38请求

建议称“非校准评价集”，不称19个独立未见测试父：其中12个适配训练父、4个反复使用的旧DEV父、3个后训练留出父；后者也不保证官方预训练完全未见。每父仍只有一个C、两个既有噪声。

| report_group | 完整parent_id | K / N | 原I状态 |
|---|---|---:|---|
| OLD | `35948e4d06a9b07e8ba8ff316aec8d3d` | 2 / 215 | PASS |
| OLD | `4cd23aae912b7de019fb4103277a6b93` | 2 / 256 | PASS |
| OLD | `7b58a3e5d7f4dddcad1d789f3b2120d0` | 28 / 256 | PASS |
| OLD | `838c48b41d513a2dd0992406ba4944f5` | 22 / 158 | PASS |
| OLD | `91e169ea3ceb587beff42b9e13c388bc` | 51 / 202 | PASS |
| OLD | `943dde2754ddc7822e8ff3556a90169` | 24 / 256 | PASS |
| OLD | `9e429b7f77646b4ccc8f8e7f17e9d7b6` | 26 / 256 | PASS |
| FOCUS | `27eb79ddf26fd7ac7898580cb9d16864` | 29 / 255 | PASS |
| HIGH_TRAIN | `abf250a01f526137a035c800c58595d9` | 44 / 308 | LOW_SUPPORT |
| NEW_WIDE_TRAIN | `9452149917cb2e27f4fb4dee5181bee` | 14 / 448 | LOW_SUPPORT |
| NEW_WIDE_TRAIN | `af264c02f69566cdf4fb4dee5181bee` | 84 / 636 | LOW_SUPPORT |
| NEW_WIDE_TRAIN | `77aa33479c2ec92f610c2a68437007d6` | 68 / 776 | PASS；闭合C |
| DEV | `2d1e5eeb255e79dcab53bf7fa22f3636` | 2 / 196 | PASS |
| DEV | `459f90aa2162b1f1d46c340938e2ff1c` | 3 / 198 | PASS |
| DEV | `f495bd67af8d9f0cadd1cf65e8dc4de` | 5 / 160 | PASS |
| DEV | `fd355d06238853c7f0d9805ad6c2059c` | 2 / 154 | PASS |
| HOLDOUT | `df6fd36040a5719662701a18243b45d3` | 38 / 255 | PASS |
| HIGH_HOLDOUT | `617fa8d6d65af35d71295682a9280b8b` | 44 / 724 | PASS；闭合C |
| HIGH_HOLDOUT | `756108c9b72e95aa57d85583f4ad07cf` | 10 / 375 | PASS |

| 原组 | 全面板父/请求 | 校准父/请求 | 余下父/请求 |
|---|---:|---:|---:|
| OLD | 8 / 16 | 1 / 2 | 7 / 14 |
| FOCUS | 2 / 4 | 1 / 2 | 1 / 2 |
| HIGH_TRAIN | 2 / 4 | 1 / 2 | 1 / 2 |
| NEW_WIDE_TRAIN | 4 / 8 | 1 / 2 | 3 / 6 |
| DEV | 4 / 8 | 0 / 0 | 4 / 8 |
| HOLDOUT | 1 / 2 | 0 / 0 | 1 / 2 |
| HIGH_HOLDOUT | 2 / 4 | 0 / 0 | 2 / 4 |
| 合计 | 23 / 46 | 4 / 8 | 19 / 38 |

## 缓存路径与读取边界

记`E=E:/MeshFlow/experiments/r2_bench_four_routes_v1/exposure`、`rid=request_id`、`cid=condition_id`。下列模板由实际EXPOSURE代码和文件目录只读核对；全部46请求涉及的460条不同输入/噪声/查询/RAW/BASE/FINAL/评分/双阶段图片路径均存在。未读取大数组或计算新评分。

| 内容 | 现有路径 |
|---|---|
| RAW mesh | `E/evaluation/EXPOSURE/<rid>/raw/raw.npz` |
| RAW轨迹/身份 | 同目录`trajectory.npz, receipt.json, audit.json, calls.jsonl` |
| 单次D0/GEO后的BASE | `E/evaluation/EXPOSURE/<rid>/post/post.npz`；同目录有`proposal.npz, pre_geo.npz, receipt.json` |
| 最终链FINAL | `E/cpu/EXPOSURE/<rid>/CLEAN_FINAL_POST/post.npz` |
| 原RAW完整评分 | `E/evaluation/scores/EXPOSURE/<rid>/RAW/{row.json,score.json,complete.json}` |
| 原BASE/FINAL完整评分 | `E/cpu/EXPOSURE/<rid>/<BASE_POST或CLEAN_FINAL_POST>/scores/{row.json,score.json,complete.json}` |
| 原177字段合表 | `E/results/per_output.csv`；276行包含M与EXPOSURE各46请求×3阶段，选baseline必须筛`label=EXPOSURE` |
| 固定图页 | `E/gallery/<RAW或BASE_POST或CLEAN_FINAL_POST>__<rid>.png`；历史页面是M/EXPOSURE同框 |
| 历史实际图审 | `E/visual_review/summary_v1/actual_observations.csv`；包含label/stage、逐轴、reviewer、actually_viewed和图片身份 |
| 原评分器版本身份 | `E/evaluation/score_registration.json` |

模型输入仅来自`inference_npz`的C和部署元数据。`source_npz`、来源owner、Y仅作现有后端来源角序/评分/展示身份，不能转成motif指导的来源完成oracle；具体motif允许信息应另冻结。原`gallery_bounds`、`gallery_depth_bounds`和front/side位置继续固定，不能按候选结果移动相机。来源形状不是开放生成唯一正确款式。

噪声、查询和输入由每条manifest字段寻址，不全放在EXPOSURE目录。四校准父的缓存基目录如下（均相对`E:/MeshFlow/experiments/`）：

| 父 | 缓存基目录B |
|---|---|
| 6b894c | `r2_bench_full_seat_curriculum_v1` |
| a8e987 | `r2_bench_seat_coverage_v1` |
| 28b9 | `r2_bench_natural_n_recipe_v1` |
| bcb4f5 | `r2_bench_high_joint_coverage_v1` |

噪声=`B/evaluation_inputs/requests/<rid>/noise.npz`；输入/元数据/来源=`B/evaluation_inputs/conditions/<cid>/{condition.npz,metadata.npz,source_condition.npz}`；查询=`B/evaluation/queries/<cid>.npz`。原`query_descriptor`冻结2048点、权重、所属C面、重心、法向及边界距离，L0=9.208138264587333、全点权重和1。固定C不变时直接复用，不重新随机查询。

历史177字段和旧图审只读复用。新候选图审仍须实际查看同尺度图，不能把EXPOSURE历史标签复制给G0–G3，也不能以评分自动生成图审标签。

## I的支持身份与真实分母

现有M/EXPOSURE×三个阶段逐请求的I状态、内部支持数及支持权重完全一致；这里是旧缓存事实，不预保证新输出仍全为PASS。新的未知相交或截线提取失败可能产生PARTIAL，须保留原因，不能当低冲突或成功。

| 子集 | 预期父/请求 | 原I PASS父/请求 | 原LOW_SUPPORT父/请求 |
|---|---:|---:|---:|
| 全46 | 23 / 46 | 19 / 38 | 4 / 8 |
| 校准8 | 4 / 8 | 3 / 6 | 1 / 2 |
| 非校准38 | 19 / 38 | 16 / 32 | 3 / 6 |

四校准父内部支持：6b为1620/2048、a8为69/2048、28b9为0/2048、bcb为1636/2048。28b9是LOW_SUPPORT，不能以I=0参加lambda选择；a8虽PASS但内部支持仅3.369140625%，必须披露。全面板LOW_SUPPORT为28b9、abf、945、af，均0/2048；移除28b9后仍有3父6请求LOW_SUPPORT。617和77aa是闭合C，开放接口要求NA，不计接续失败。

**原I每输出的分母仍是全部2048点的权重和1，不是内部支持点数。** 原源码`conflict_metrics`先算`union=cross|dup`，再用`dot(weights,union)/weights.sum()`；dup检测限制在距开放边≥0.01L的内部点，cross不作相同内部限制；内部点少于32触发LOW_SUPPORT（未知情况另有PARTIAL优先级）。不能将I除以内部权重，或按a8的69个点重新归一。现有a8/n0 RAW的I=.330078125、内部权重=.03369140625就是两者不可混用的实例。原源码位置由[score_registration](../../experiments/r2_bench_four_routes_v1/exposure/evaluation/score_registration.json)指向`E:/MeshFlow-Control/src/meshflow_control/evaluation/control.py`。

建议汇总沿用同父n0/n1平均，再父等权；主要效果判断使用非校准38，I的原可用宏分母16父/32请求，校准则3父/6请求；完整46仅作含调参数据的描述表，I为19父/38请求。新候选如出现额外NA/PARTIAL，应列每臂有效数、缺失原因和共同PASS配对身份，不静默缩分母，不把缺失补零。CF/FF继续用各阶段实际free面分母，同时列绝对非法面对数、free面/面积及删面；RAW拓扑孤岛不直接等于视觉碎片。

## 200新输出的精确预算

以下假设G0、G1、G2、G3均是需新生成的候选臂，EXPOSURE46只读复用；G3只在固定8校准请求上试预定三个lambda，随后选**一个全局lambda**用于所有其余38请求，不逐父挑lambda。

| 部分 | 新输出 |
|---|---:|
| G0/G1/G2各完整46请求 | 3×46=138 |
| G3三个lambda×校准8请求 | 3×8=24 |
| 冻结所选lambda后G3补余38请求 | 38 |
| 新输出总计 | **200** |

等价为四臂主面板184，加未选两个lambda的16份校准输出。选中lambda的8份校准结果直接复用，重抽或重生会变成208。EXPOSURE缓存另46，因此留档246份唯一输出；选定四臂+EXPOSURE的主面板230份，其中校准40份（四臂32+baseline8）只作调参/描述，非校准五臂主比较190份。

以固定Euler50且每积分步一次生成器前向为前提，200 RAW对应10000次生成器前向；motif能量/梯度计算应独立记账，不能因没有新权重而记作零计算。若算法另有模型前向、重复去噪或额外rollout，上述预算不覆盖，设计必须事前明确；本文件不为它们追加预算。

后端/评分应在协议中明确选一种，不执行后再改变口径：

- **全留档方案：200份新RAW均走单次官方D0→GEO_C/来源定向→HINGE_OPP015→ALPHA→HOLE。** 200 D0、600个新三阶段评分单元；EXPOSURE缓存138行另算，总738行。仅RAW/FINAL展示为400新行+92缓存=492行。原五模型选定主面板是690三阶段行，额外48行为未选lambda校准。
- 若预定未选lambda16份只评RAW，则200 RAW、184 D0；主面板552个新三阶段行+额外16 RAW=568新评分单元，另复用138 baseline行。该方案不得用未运行的FINAL质量选lambda，也不得称200份全部完成后端。

**lambda选择与RAW屏障存在阶段依赖。** 若坚持全部200新RAW保存后才执行任何D0，lambda只能先按冻结RAW规则选，再生成余38请求。若lambda选择依赖FINAL，则应明确两个屏障：校准24 RAW全部保存→校准24 D0/评分→锁定lambda；随后剩余176 RAW全部保存→其176 D0/评分。后一方案仍200份、200 D0，但不能宣称全200RAW前零D0。阈值、失败处理、并列选择及是否需要FINAL必须在未来执行授权前冻结。

若G0被定义成与既有EXPOSURE逐项完全相同的模型、输入、算法和链，则其46份可成为缓存对照而非新臂，此时新输出会是154而非200。当前200预算假定G0确为新候选；不要一边将G0称原EXPOSURE复用，一边仍计200新输出。

以上仅提供协议所需的面板事实和算术，未创建实验目录、冻结执行manifest或启动任何任务。
