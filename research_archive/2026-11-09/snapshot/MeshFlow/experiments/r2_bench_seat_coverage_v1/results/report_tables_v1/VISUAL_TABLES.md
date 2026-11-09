# 实际AI_PROXY五轴与同输出严格三轴交集

每格：类别请求数/全部已评请求〔父等权比例；父数〕。不确定项保留；没有评测的模型/面板写NA。
严格三轴仅是同一输出 PLAUSIBLE + REASONABLE + PRESENT，不代表拓扑合法接续或几何质量通过。

## OLD_TRAIN / RAW

| 轴 | 类别 | Native6000 | Candidate1500 | 旧SEAM3000 |
|---|---|---:|---:|---:|
| overall_shape | REASONABLE | 14/16〔0.875; 8父〕 | 14/16〔0.875; 8父〕 | 15/16〔0.9375; 8父〕 |
| overall_shape | INCOMPLETE | 1/16〔0.0625; 8父〕 | 1/16〔0.0625; 8父〕 | 0/16〔0; 8父〕 |
| overall_shape | UNCERTAIN | 1/16〔0.0625; 8父〕 | 1/16〔0.0625; 8父〕 | 1/16〔0.0625; 8父〕 |
| C_use | PLAUSIBLE | 16/16〔1; 8父〕 | 15/16〔0.9375; 8父〕 | 14/16〔0.875; 8父〕 |
| C_use | REPLACED_OR_IGNORED | 0/16〔0; 8父〕 | 1/16〔0.0625; 8父〕 | 1/16〔0.0625; 8父〕 |
| C_use | UNCERTAIN | 0/16〔0; 8父〕 | 0/16〔0; 8父〕 | 1/16〔0.0625; 8父〕 |
| support_retention | PRESENT | 16/16〔1; 8父〕 | 16/16〔1; 8父〕 | 16/16〔1; 8父〕 |
| support_retention | MISSING_OR_DAMAGED | 0/16〔0; 8父〕 | 0/16〔0; 8父〕 | 0/16〔0; 8父〕 |
| support_retention | UNCERTAIN | 0/16〔0; 8父〕 | 0/16〔0; 8父〕 | 0/16〔0; 8父〕 |
| visible_fragment_or_gap | YES | 3/16〔0.1875; 8父〕 | 6/16〔0.375; 8父〕 | 2/16〔0.125; 8父〕 |
| visible_fragment_or_gap | NO_OBVIOUS | 12/16〔0.75; 8父〕 | 9/16〔0.5625; 8父〕 | 14/16〔0.875; 8父〕 |
| visible_fragment_or_gap | UNCERTAIN | 1/16〔0.0625; 8父〕 | 1/16〔0.0625; 8父〕 | 0/16〔0; 8父〕 |
| layering_or_fold | YES | 1/16〔0.0625; 8父〕 | 2/16〔0.125; 8父〕 | 3/16〔0.1875; 8父〕 |
| layering_or_fold | NO_OBVIOUS | 15/16〔0.9375; 8父〕 | 13/16〔0.8125; 8父〕 | 12/16〔0.75; 8父〕 |
| layering_or_fold | UNCERTAIN | 0/16〔0; 8父〕 | 1/16〔0.0625; 8父〕 | 1/16〔0.0625; 8父〕 |
| strict_visual_three_axis | YES | 14/16〔0.875; 8父〕 | 13/16〔0.8125; 8父〕 | 14/16〔0.875; 8父〕 |
| strict_visual_three_axis | NO | 1/16〔0.0625; 8父〕 | 2/16〔0.125; 8父〕 | 1/16〔0.0625; 8父〕 |
| strict_visual_three_axis | UNCERTAIN | 1/16〔0.0625; 8父〕 | 1/16〔0.0625; 8父〕 | 1/16〔0.0625; 8父〕 |

## OLD_TRAIN / BASE_POST

| 轴 | 类别 | Native6000 | Candidate1500 | 旧SEAM3000 |
|---|---|---:|---:|---:|
| overall_shape | REASONABLE | 8/16〔0.5; 8父〕 | 8/16〔0.5; 8父〕 | 9/16〔0.5625; 8父〕 |
| overall_shape | INCOMPLETE | 5/16〔0.3125; 8父〕 | 6/16〔0.375; 8父〕 | 5/16〔0.3125; 8父〕 |
| overall_shape | UNCERTAIN | 3/16〔0.1875; 8父〕 | 2/16〔0.125; 8父〕 | 2/16〔0.125; 8父〕 |
| C_use | PLAUSIBLE | 16/16〔1; 8父〕 | 15/16〔0.9375; 8父〕 | 15/16〔0.9375; 8父〕 |
| C_use | REPLACED_OR_IGNORED | 0/16〔0; 8父〕 | 1/16〔0.0625; 8父〕 | 1/16〔0.0625; 8父〕 |
| C_use | UNCERTAIN | 0/16〔0; 8父〕 | 0/16〔0; 8父〕 | 0/16〔0; 8父〕 |
| support_retention | PRESENT | 11/16〔0.6875; 8父〕 | 11/16〔0.6875; 8父〕 | 11/16〔0.6875; 8父〕 |
| support_retention | MISSING_OR_DAMAGED | 5/16〔0.3125; 8父〕 | 5/16〔0.3125; 8父〕 | 5/16〔0.3125; 8父〕 |
| support_retention | UNCERTAIN | 0/16〔0; 8父〕 | 0/16〔0; 8父〕 | 0/16〔0; 8父〕 |
| visible_fragment_or_gap | YES | 8/16〔0.5; 8父〕 | 12/16〔0.75; 8父〕 | 7/16〔0.4375; 8父〕 |
| visible_fragment_or_gap | NO_OBVIOUS | 7/16〔0.4375; 8父〕 | 4/16〔0.25; 8父〕 | 8/16〔0.5; 8父〕 |
| visible_fragment_or_gap | UNCERTAIN | 1/16〔0.0625; 8父〕 | 0/16〔0; 8父〕 | 1/16〔0.0625; 8父〕 |
| layering_or_fold | YES | 1/16〔0.0625; 8父〕 | 2/16〔0.125; 8父〕 | 2/16〔0.125; 8父〕 |
| layering_or_fold | NO_OBVIOUS | 15/16〔0.9375; 8父〕 | 13/16〔0.8125; 8父〕 | 12/16〔0.75; 8父〕 |
| layering_or_fold | UNCERTAIN | 0/16〔0; 8父〕 | 1/16〔0.0625; 8父〕 | 2/16〔0.125; 8父〕 |
| strict_visual_three_axis | YES | 8/16〔0.5; 8父〕 | 8/16〔0.5; 8父〕 | 9/16〔0.5625; 8父〕 |
| strict_visual_three_axis | NO | 6/16〔0.375; 8父〕 | 6/16〔0.375; 8父〕 | 5/16〔0.3125; 8父〕 |
| strict_visual_three_axis | UNCERTAIN | 2/16〔0.125; 8父〕 | 2/16〔0.125; 8父〕 | 2/16〔0.125; 8父〕 |

## OLD_TRAIN / CLEAN_FINAL_POST

| 轴 | 类别 | Native6000 | Candidate1500 | 旧SEAM3000 |
|---|---|---:|---:|---:|
| overall_shape | REASONABLE | 8/16〔0.5; 8父〕 | 8/16〔0.5; 8父〕 | 9/16〔0.5625; 8父〕 |
| overall_shape | INCOMPLETE | 5/16〔0.3125; 8父〕 | 6/16〔0.375; 8父〕 | 5/16〔0.3125; 8父〕 |
| overall_shape | UNCERTAIN | 3/16〔0.1875; 8父〕 | 2/16〔0.125; 8父〕 | 2/16〔0.125; 8父〕 |
| C_use | PLAUSIBLE | 16/16〔1; 8父〕 | 15/16〔0.9375; 8父〕 | 15/16〔0.9375; 8父〕 |
| C_use | REPLACED_OR_IGNORED | 0/16〔0; 8父〕 | 1/16〔0.0625; 8父〕 | 1/16〔0.0625; 8父〕 |
| C_use | UNCERTAIN | 0/16〔0; 8父〕 | 0/16〔0; 8父〕 | 0/16〔0; 8父〕 |
| support_retention | PRESENT | 11/16〔0.6875; 8父〕 | 11/16〔0.6875; 8父〕 | 11/16〔0.6875; 8父〕 |
| support_retention | MISSING_OR_DAMAGED | 5/16〔0.3125; 8父〕 | 5/16〔0.3125; 8父〕 | 5/16〔0.3125; 8父〕 |
| support_retention | UNCERTAIN | 0/16〔0; 8父〕 | 0/16〔0; 8父〕 | 0/16〔0; 8父〕 |
| visible_fragment_or_gap | YES | 8/16〔0.5; 8父〕 | 12/16〔0.75; 8父〕 | 7/16〔0.4375; 8父〕 |
| visible_fragment_or_gap | NO_OBVIOUS | 7/16〔0.4375; 8父〕 | 4/16〔0.25; 8父〕 | 8/16〔0.5; 8父〕 |
| visible_fragment_or_gap | UNCERTAIN | 1/16〔0.0625; 8父〕 | 0/16〔0; 8父〕 | 1/16〔0.0625; 8父〕 |
| layering_or_fold | YES | 1/16〔0.0625; 8父〕 | 2/16〔0.125; 8父〕 | 2/16〔0.125; 8父〕 |
| layering_or_fold | NO_OBVIOUS | 15/16〔0.9375; 8父〕 | 13/16〔0.8125; 8父〕 | 12/16〔0.75; 8父〕 |
| layering_or_fold | UNCERTAIN | 0/16〔0; 8父〕 | 1/16〔0.0625; 8父〕 | 2/16〔0.125; 8父〕 |
| strict_visual_three_axis | YES | 8/16〔0.5; 8父〕 | 8/16〔0.5; 8父〕 | 9/16〔0.5625; 8父〕 |
| strict_visual_three_axis | NO | 6/16〔0.375; 8父〕 | 6/16〔0.375; 8父〕 | 5/16〔0.3125; 8父〕 |
| strict_visual_three_axis | UNCERTAIN | 2/16〔0.125; 8父〕 | 2/16〔0.125; 8父〕 | 2/16〔0.125; 8父〕 |

## NEW_TRAIN / RAW

| 轴 | 类别 | Native6000 | Candidate1500 | 旧SEAM3000 |
|---|---|---:|---:|---:|
| overall_shape | REASONABLE | 4/4〔1; 2父〕 | 4/4〔1; 2父〕 | NA |
| overall_shape | INCOMPLETE | 0/4〔0; 2父〕 | 0/4〔0; 2父〕 | NA |
| overall_shape | UNCERTAIN | 0/4〔0; 2父〕 | 0/4〔0; 2父〕 | NA |
| C_use | PLAUSIBLE | 2/4〔0.5; 2父〕 | 4/4〔1; 2父〕 | NA |
| C_use | REPLACED_OR_IGNORED | 0/4〔0; 2父〕 | 0/4〔0; 2父〕 | NA |
| C_use | UNCERTAIN | 2/4〔0.5; 2父〕 | 0/4〔0; 2父〕 | NA |
| support_retention | PRESENT | 4/4〔1; 2父〕 | 4/4〔1; 2父〕 | NA |
| support_retention | MISSING_OR_DAMAGED | 0/4〔0; 2父〕 | 0/4〔0; 2父〕 | NA |
| support_retention | UNCERTAIN | 0/4〔0; 2父〕 | 0/4〔0; 2父〕 | NA |
| visible_fragment_or_gap | YES | 0/4〔0; 2父〕 | 0/4〔0; 2父〕 | NA |
| visible_fragment_or_gap | NO_OBVIOUS | 0/4〔0; 2父〕 | 4/4〔1; 2父〕 | NA |
| visible_fragment_or_gap | UNCERTAIN | 4/4〔1; 2父〕 | 0/4〔0; 2父〕 | NA |
| layering_or_fold | YES | 1/4〔0.25; 2父〕 | 0/4〔0; 2父〕 | NA |
| layering_or_fold | NO_OBVIOUS | 2/4〔0.5; 2父〕 | 4/4〔1; 2父〕 | NA |
| layering_or_fold | UNCERTAIN | 1/4〔0.25; 2父〕 | 0/4〔0; 2父〕 | NA |
| strict_visual_three_axis | YES | 2/4〔0.5; 2父〕 | 4/4〔1; 2父〕 | NA |
| strict_visual_three_axis | NO | 0/4〔0; 2父〕 | 0/4〔0; 2父〕 | NA |
| strict_visual_three_axis | UNCERTAIN | 2/4〔0.5; 2父〕 | 0/4〔0; 2父〕 | NA |

## NEW_TRAIN / BASE_POST

| 轴 | 类别 | Native6000 | Candidate1500 | 旧SEAM3000 |
|---|---|---:|---:|---:|
| overall_shape | REASONABLE | 3/4〔0.75; 2父〕 | 0/4〔0; 2父〕 | NA |
| overall_shape | INCOMPLETE | 0/4〔0; 2父〕 | 4/4〔1; 2父〕 | NA |
| overall_shape | UNCERTAIN | 1/4〔0.25; 2父〕 | 0/4〔0; 2父〕 | NA |
| C_use | PLAUSIBLE | 3/4〔0.75; 2父〕 | 4/4〔1; 2父〕 | NA |
| C_use | REPLACED_OR_IGNORED | 0/4〔0; 2父〕 | 0/4〔0; 2父〕 | NA |
| C_use | UNCERTAIN | 1/4〔0.25; 2父〕 | 0/4〔0; 2父〕 | NA |
| support_retention | PRESENT | 4/4〔1; 2父〕 | 0/4〔0; 2父〕 | NA |
| support_retention | MISSING_OR_DAMAGED | 0/4〔0; 2父〕 | 4/4〔1; 2父〕 | NA |
| support_retention | UNCERTAIN | 0/4〔0; 2父〕 | 0/4〔0; 2父〕 | NA |
| visible_fragment_or_gap | YES | 3/4〔0.75; 2父〕 | 4/4〔1; 2父〕 | NA |
| visible_fragment_or_gap | NO_OBVIOUS | 0/4〔0; 2父〕 | 0/4〔0; 2父〕 | NA |
| visible_fragment_or_gap | UNCERTAIN | 1/4〔0.25; 2父〕 | 0/4〔0; 2父〕 | NA |
| layering_or_fold | YES | 0/4〔0; 2父〕 | 0/4〔0; 2父〕 | NA |
| layering_or_fold | NO_OBVIOUS | 3/4〔0.75; 2父〕 | 4/4〔1; 2父〕 | NA |
| layering_or_fold | UNCERTAIN | 1/4〔0.25; 2父〕 | 0/4〔0; 2父〕 | NA |
| strict_visual_three_axis | YES | 3/4〔0.75; 2父〕 | 0/4〔0; 2父〕 | NA |
| strict_visual_three_axis | NO | 0/4〔0; 2父〕 | 4/4〔1; 2父〕 | NA |
| strict_visual_three_axis | UNCERTAIN | 1/4〔0.25; 2父〕 | 0/4〔0; 2父〕 | NA |

## NEW_TRAIN / CLEAN_FINAL_POST

| 轴 | 类别 | Native6000 | Candidate1500 | 旧SEAM3000 |
|---|---|---:|---:|---:|
| overall_shape | REASONABLE | 3/4〔0.75; 2父〕 | 0/4〔0; 2父〕 | NA |
| overall_shape | INCOMPLETE | 0/4〔0; 2父〕 | 4/4〔1; 2父〕 | NA |
| overall_shape | UNCERTAIN | 1/4〔0.25; 2父〕 | 0/4〔0; 2父〕 | NA |
| C_use | PLAUSIBLE | 3/4〔0.75; 2父〕 | 4/4〔1; 2父〕 | NA |
| C_use | REPLACED_OR_IGNORED | 0/4〔0; 2父〕 | 0/4〔0; 2父〕 | NA |
| C_use | UNCERTAIN | 1/4〔0.25; 2父〕 | 0/4〔0; 2父〕 | NA |
| support_retention | PRESENT | 4/4〔1; 2父〕 | 0/4〔0; 2父〕 | NA |
| support_retention | MISSING_OR_DAMAGED | 0/4〔0; 2父〕 | 4/4〔1; 2父〕 | NA |
| support_retention | UNCERTAIN | 0/4〔0; 2父〕 | 0/4〔0; 2父〕 | NA |
| visible_fragment_or_gap | YES | 3/4〔0.75; 2父〕 | 4/4〔1; 2父〕 | NA |
| visible_fragment_or_gap | NO_OBVIOUS | 0/4〔0; 2父〕 | 0/4〔0; 2父〕 | NA |
| visible_fragment_or_gap | UNCERTAIN | 1/4〔0.25; 2父〕 | 0/4〔0; 2父〕 | NA |
| layering_or_fold | YES | 0/4〔0; 2父〕 | 0/4〔0; 2父〕 | NA |
| layering_or_fold | NO_OBVIOUS | 3/4〔0.75; 2父〕 | 4/4〔1; 2父〕 | NA |
| layering_or_fold | UNCERTAIN | 1/4〔0.25; 2父〕 | 0/4〔0; 2父〕 | NA |
| strict_visual_three_axis | YES | 3/4〔0.75; 2父〕 | 0/4〔0; 2父〕 | NA |
| strict_visual_three_axis | NO | 0/4〔0; 2父〕 | 4/4〔1; 2父〕 | NA |
| strict_visual_three_axis | UNCERTAIN | 1/4〔0.25; 2父〕 | 0/4〔0; 2父〕 | NA |

## NEW_PARENT_HOLDOUT / RAW

| 轴 | 类别 | Native6000 | Candidate1500 | 旧SEAM3000 |
|---|---|---:|---:|---:|
| overall_shape | REASONABLE | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| overall_shape | INCOMPLETE | 2/2〔1; 1父〕 | 2/2〔1; 1父〕 | NA |
| overall_shape | UNCERTAIN | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| C_use | PLAUSIBLE | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| C_use | REPLACED_OR_IGNORED | 2/2〔1; 1父〕 | 2/2〔1; 1父〕 | NA |
| C_use | UNCERTAIN | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| support_retention | PRESENT | 2/2〔1; 1父〕 | 2/2〔1; 1父〕 | NA |
| support_retention | MISSING_OR_DAMAGED | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| support_retention | UNCERTAIN | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| visible_fragment_or_gap | YES | 2/2〔1; 1父〕 | 2/2〔1; 1父〕 | NA |
| visible_fragment_or_gap | NO_OBVIOUS | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| visible_fragment_or_gap | UNCERTAIN | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| layering_or_fold | YES | 2/2〔1; 1父〕 | 2/2〔1; 1父〕 | NA |
| layering_or_fold | NO_OBVIOUS | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| layering_or_fold | UNCERTAIN | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| strict_visual_three_axis | YES | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| strict_visual_three_axis | NO | 2/2〔1; 1父〕 | 2/2〔1; 1父〕 | NA |
| strict_visual_three_axis | UNCERTAIN | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |

## NEW_PARENT_HOLDOUT / BASE_POST

| 轴 | 类别 | Native6000 | Candidate1500 | 旧SEAM3000 |
|---|---|---:|---:|---:|
| overall_shape | REASONABLE | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| overall_shape | INCOMPLETE | 2/2〔1; 1父〕 | 2/2〔1; 1父〕 | NA |
| overall_shape | UNCERTAIN | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| C_use | PLAUSIBLE | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| C_use | REPLACED_OR_IGNORED | 2/2〔1; 1父〕 | 2/2〔1; 1父〕 | NA |
| C_use | UNCERTAIN | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| support_retention | PRESENT | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| support_retention | MISSING_OR_DAMAGED | 2/2〔1; 1父〕 | 2/2〔1; 1父〕 | NA |
| support_retention | UNCERTAIN | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| visible_fragment_or_gap | YES | 2/2〔1; 1父〕 | 2/2〔1; 1父〕 | NA |
| visible_fragment_or_gap | NO_OBVIOUS | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| visible_fragment_or_gap | UNCERTAIN | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| layering_or_fold | YES | 2/2〔1; 1父〕 | 2/2〔1; 1父〕 | NA |
| layering_or_fold | NO_OBVIOUS | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| layering_or_fold | UNCERTAIN | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| strict_visual_three_axis | YES | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| strict_visual_three_axis | NO | 2/2〔1; 1父〕 | 2/2〔1; 1父〕 | NA |
| strict_visual_three_axis | UNCERTAIN | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |

## NEW_PARENT_HOLDOUT / CLEAN_FINAL_POST

| 轴 | 类别 | Native6000 | Candidate1500 | 旧SEAM3000 |
|---|---|---:|---:|---:|
| overall_shape | REASONABLE | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| overall_shape | INCOMPLETE | 2/2〔1; 1父〕 | 2/2〔1; 1父〕 | NA |
| overall_shape | UNCERTAIN | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| C_use | PLAUSIBLE | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| C_use | REPLACED_OR_IGNORED | 2/2〔1; 1父〕 | 2/2〔1; 1父〕 | NA |
| C_use | UNCERTAIN | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| support_retention | PRESENT | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| support_retention | MISSING_OR_DAMAGED | 2/2〔1; 1父〕 | 2/2〔1; 1父〕 | NA |
| support_retention | UNCERTAIN | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| visible_fragment_or_gap | YES | 2/2〔1; 1父〕 | 2/2〔1; 1父〕 | NA |
| visible_fragment_or_gap | NO_OBVIOUS | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| visible_fragment_or_gap | UNCERTAIN | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| layering_or_fold | YES | 2/2〔1; 1父〕 | 2/2〔1; 1父〕 | NA |
| layering_or_fold | NO_OBVIOUS | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| layering_or_fold | UNCERTAIN | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| strict_visual_three_axis | YES | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |
| strict_visual_three_axis | NO | 2/2〔1; 1父〕 | 2/2〔1; 1父〕 | NA |
| strict_visual_three_axis | UNCERTAIN | 0/2〔0; 1父〕 | 0/2〔0; 1父〕 | NA |

## DEV / RAW

| 轴 | 类别 | Native6000 | Candidate1500 | 旧SEAM3000 |
|---|---|---:|---:|---:|
| overall_shape | REASONABLE | 8/8〔1; 4父〕 | 8/8〔1; 4父〕 | 8/8〔1; 4父〕 |
| overall_shape | INCOMPLETE | 0/8〔0; 4父〕 | 0/8〔0; 4父〕 | 0/8〔0; 4父〕 |
| overall_shape | UNCERTAIN | 0/8〔0; 4父〕 | 0/8〔0; 4父〕 | 0/8〔0; 4父〕 |
| C_use | PLAUSIBLE | 0/8〔0; 4父〕 | 1/8〔0.125; 4父〕 | 1/8〔0.125; 4父〕 |
| C_use | REPLACED_OR_IGNORED | 7/8〔0.875; 4父〕 | 6/8〔0.75; 4父〕 | 6/8〔0.75; 4父〕 |
| C_use | UNCERTAIN | 1/8〔0.125; 4父〕 | 1/8〔0.125; 4父〕 | 1/8〔0.125; 4父〕 |
| support_retention | PRESENT | 8/8〔1; 4父〕 | 8/8〔1; 4父〕 | 8/8〔1; 4父〕 |
| support_retention | MISSING_OR_DAMAGED | 0/8〔0; 4父〕 | 0/8〔0; 4父〕 | 0/8〔0; 4父〕 |
| support_retention | UNCERTAIN | 0/8〔0; 4父〕 | 0/8〔0; 4父〕 | 0/8〔0; 4父〕 |
| visible_fragment_or_gap | YES | 2/8〔0.25; 4父〕 | 2/8〔0.25; 4父〕 | 2/8〔0.25; 4父〕 |
| visible_fragment_or_gap | NO_OBVIOUS | 4/8〔0.5; 4父〕 | 5/8〔0.625; 4父〕 | 3/8〔0.375; 4父〕 |
| visible_fragment_or_gap | UNCERTAIN | 2/8〔0.25; 4父〕 | 1/8〔0.125; 4父〕 | 3/8〔0.375; 4父〕 |
| layering_or_fold | YES | 8/8〔1; 4父〕 | 7/8〔0.875; 4父〕 | 7/8〔0.875; 4父〕 |
| layering_or_fold | NO_OBVIOUS | 0/8〔0; 4父〕 | 1/8〔0.125; 4父〕 | 1/8〔0.125; 4父〕 |
| layering_or_fold | UNCERTAIN | 0/8〔0; 4父〕 | 0/8〔0; 4父〕 | 0/8〔0; 4父〕 |
| strict_visual_three_axis | YES | 0/8〔0; 4父〕 | 1/8〔0.125; 4父〕 | 1/8〔0.125; 4父〕 |
| strict_visual_three_axis | NO | 7/8〔0.875; 4父〕 | 6/8〔0.75; 4父〕 | 6/8〔0.75; 4父〕 |
| strict_visual_three_axis | UNCERTAIN | 1/8〔0.125; 4父〕 | 1/8〔0.125; 4父〕 | 1/8〔0.125; 4父〕 |

## DEV / BASE_POST

| 轴 | 类别 | Native6000 | Candidate1500 | 旧SEAM3000 |
|---|---|---:|---:|---:|
| overall_shape | REASONABLE | 5/8〔0.625; 4父〕 | 5/8〔0.625; 4父〕 | 5/8〔0.625; 4父〕 |
| overall_shape | INCOMPLETE | 2/8〔0.25; 4父〕 | 3/8〔0.375; 4父〕 | 2/8〔0.25; 4父〕 |
| overall_shape | UNCERTAIN | 1/8〔0.125; 4父〕 | 0/8〔0; 4父〕 | 1/8〔0.125; 4父〕 |
| C_use | PLAUSIBLE | 0/8〔0; 4父〕 | 1/8〔0.125; 4父〕 | 1/8〔0.125; 4父〕 |
| C_use | REPLACED_OR_IGNORED | 7/8〔0.875; 4父〕 | 6/8〔0.75; 4父〕 | 6/8〔0.75; 4父〕 |
| C_use | UNCERTAIN | 1/8〔0.125; 4父〕 | 1/8〔0.125; 4父〕 | 1/8〔0.125; 4父〕 |
| support_retention | PRESENT | 6/8〔0.75; 4父〕 | 5/8〔0.625; 4父〕 | 6/8〔0.75; 4父〕 |
| support_retention | MISSING_OR_DAMAGED | 2/8〔0.25; 4父〕 | 3/8〔0.375; 4父〕 | 2/8〔0.25; 4父〕 |
| support_retention | UNCERTAIN | 0/8〔0; 4父〕 | 0/8〔0; 4父〕 | 0/8〔0; 4父〕 |
| visible_fragment_or_gap | YES | 5/8〔0.625; 4父〕 | 5/8〔0.625; 4父〕 | 5/8〔0.625; 4父〕 |
| visible_fragment_or_gap | NO_OBVIOUS | 2/8〔0.25; 4父〕 | 2/8〔0.25; 4父〕 | 2/8〔0.25; 4父〕 |
| visible_fragment_or_gap | UNCERTAIN | 1/8〔0.125; 4父〕 | 1/8〔0.125; 4父〕 | 1/8〔0.125; 4父〕 |
| layering_or_fold | YES | 7/8〔0.875; 4父〕 | 6/8〔0.75; 4父〕 | 6/8〔0.75; 4父〕 |
| layering_or_fold | NO_OBVIOUS | 1/8〔0.125; 4父〕 | 2/8〔0.25; 4父〕 | 2/8〔0.25; 4父〕 |
| layering_or_fold | UNCERTAIN | 0/8〔0; 4父〕 | 0/8〔0; 4父〕 | 0/8〔0; 4父〕 |
| strict_visual_three_axis | YES | 0/8〔0; 4父〕 | 1/8〔0.125; 4父〕 | 1/8〔0.125; 4父〕 |
| strict_visual_three_axis | NO | 8/8〔1; 4父〕 | 7/8〔0.875; 4父〕 | 7/8〔0.875; 4父〕 |
| strict_visual_three_axis | UNCERTAIN | 0/8〔0; 4父〕 | 0/8〔0; 4父〕 | 0/8〔0; 4父〕 |

## DEV / CLEAN_FINAL_POST

| 轴 | 类别 | Native6000 | Candidate1500 | 旧SEAM3000 |
|---|---|---:|---:|---:|
| overall_shape | REASONABLE | 5/8〔0.625; 4父〕 | 5/8〔0.625; 4父〕 | 5/8〔0.625; 4父〕 |
| overall_shape | INCOMPLETE | 2/8〔0.25; 4父〕 | 3/8〔0.375; 4父〕 | 2/8〔0.25; 4父〕 |
| overall_shape | UNCERTAIN | 1/8〔0.125; 4父〕 | 0/8〔0; 4父〕 | 1/8〔0.125; 4父〕 |
| C_use | PLAUSIBLE | 0/8〔0; 4父〕 | 1/8〔0.125; 4父〕 | 1/8〔0.125; 4父〕 |
| C_use | REPLACED_OR_IGNORED | 7/8〔0.875; 4父〕 | 6/8〔0.75; 4父〕 | 6/8〔0.75; 4父〕 |
| C_use | UNCERTAIN | 1/8〔0.125; 4父〕 | 1/8〔0.125; 4父〕 | 1/8〔0.125; 4父〕 |
| support_retention | PRESENT | 6/8〔0.75; 4父〕 | 5/8〔0.625; 4父〕 | 6/8〔0.75; 4父〕 |
| support_retention | MISSING_OR_DAMAGED | 2/8〔0.25; 4父〕 | 3/8〔0.375; 4父〕 | 2/8〔0.25; 4父〕 |
| support_retention | UNCERTAIN | 0/8〔0; 4父〕 | 0/8〔0; 4父〕 | 0/8〔0; 4父〕 |
| visible_fragment_or_gap | YES | 5/8〔0.625; 4父〕 | 5/8〔0.625; 4父〕 | 5/8〔0.625; 4父〕 |
| visible_fragment_or_gap | NO_OBVIOUS | 2/8〔0.25; 4父〕 | 2/8〔0.25; 4父〕 | 2/8〔0.25; 4父〕 |
| visible_fragment_or_gap | UNCERTAIN | 1/8〔0.125; 4父〕 | 1/8〔0.125; 4父〕 | 1/8〔0.125; 4父〕 |
| layering_or_fold | YES | 7/8〔0.875; 4父〕 | 6/8〔0.75; 4父〕 | 6/8〔0.75; 4父〕 |
| layering_or_fold | NO_OBVIOUS | 1/8〔0.125; 4父〕 | 2/8〔0.25; 4父〕 | 2/8〔0.25; 4父〕 |
| layering_or_fold | UNCERTAIN | 0/8〔0; 4父〕 | 0/8〔0; 4父〕 | 0/8〔0; 4父〕 |
| strict_visual_three_axis | YES | 0/8〔0; 4父〕 | 1/8〔0.125; 4父〕 | 1/8〔0.125; 4父〕 |
| strict_visual_three_axis | NO | 8/8〔1; 4父〕 | 7/8〔0.875; 4父〕 | 7/8〔0.875; 4父〕 |
| strict_visual_three_axis | UNCERTAIN | 0/8〔0; 4父〕 | 0/8〔0; 4父〕 | 0/8〔0; 4父〕 |

