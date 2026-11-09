# 2026-10-25 实验归档与存储清理

**两阶段最终总计：删除390份旧权重和1022份重复展开副本，共1412文件；逻辑大小137.098 GiB，E盘可用空间实测增加135.639 GiB。**

最终重新验证605项实际保留资产SHA256与4,301条当前非权重引用；均通过。最终依据为[合并回执](FINAL_CLEANUP_RECEIPT.json)及[有效保留资产](effective_retained_assets.json)。

补充阶段清理9月旧工作树AE/FM/PONQ的237份小权重（2.892GiB）。它们在第一遍仅因小文件暂留，随后按17组旧报告确认已封存、无当前维护依赖；不是官方权重。补充清单只覆盖这237项第一遍泛化KEEP判断，核心保留模型不变。见[补充审计](legacy_small_weights_audit.json)、[补充删除账本](legacy_small_deletion_ledger.jsonl)。

下面保留第一阶段的分项事实；其删除量不得与上述最终总计再次相加。

本轮已按逐文件白名单完成清理，并复核 842 项保护资产的 hash。清理状态：`COMPLETE_VERIFIED`；完成时间：`2026-10-24T17:40:41.8357340Z`。

共移除 **1175 个文件，逻辑大小 134.206 GiB**；E 盘可用空间实测增加 **132.748 GiB**。

逻辑大小是被移除路径的文件长度之和，包含硬链接；它不等于独立物理分配。卷可用空间变化也可能含其他进程同期活动，两个数分开报告。

| 清理类型 | 文件数 | 逻辑 GiB |
| --- | ---: | ---: |
| 已在保留 ZIP 中逐字节验证的展开审阅副本 | 1022 | 0.280 |
| 旧候选、非保留端点与中间/滚动权重 | 153 | 133.926 |

## 本轮实验收口

RAW 局部修复试点保持 `NO_TRAINING_DATA_GATE_FAILED`：12 请求中 6 份有非空 EDIT、4 份满足当前保守数值资格；新增最终局部修复集中在91e两份，并伴随未获得对照后处理新增的5条合法 F/F 共边。局部几何正证据保留，不升级为通用后端，也没有因此新增训练或权重。

已归档 159 个科学文件，包含全部12份目标数组/掩码、逐例评分与配对表、实际图审、实际代码及12张同链 FINAL 对照图。审阅包 3,508,889 bytes（3.509 MB）：[R2_BENCH_RAW_LOCAL_REPAIR_V1_REVIEW_SMALL.zip](<../../deliverables/R2_BENCH_RAW_LOCAL_REPAIR_V1_REVIEW_SMALL.zip>)。

ZIP SHA256：`d25723116e28a4a567372b4bef3014cfcefa3658fcd0bd31f3cd5d1c6f1b9acf`。完整原实验及跨阶段 RAW/D0/POST 缓存仍保留；小包不是可独立执行整个训练/模型流程的重放包。原实验记录见 [REPORT.md](<../../experiments/r2_bench_raw_local_repair_v1/REPORT.md>)。

## 保留与删除边界

继续保留的模型和使用身份见 [RETAINED_MODELS.md](<RETAINED_MODELS.md>)。Chair S2000及其配套头、官方D0与CPU后处理保持；MULTIPART的BACK正例分支、FULL_S6000、START保持；Bench COMMON12000、Native6000、SEAM3000、Coverage1500和SHAPE500的推理/恢复资产保持。它们各有证据范围，没有合并称为单一全面最优模型。

官方/公开模型和原始数据不删；实验数据集、来源网格、合法条件、面/角点身份、固定噪声/查询、RAW/POST、评分、图审、代码、训练日志与历史失败记录继续保留。数据侧本轮只移除已证明与保留 ZIP 成员逐字节相同的 `review_package` 展开副本，不把旧结果或唯一调试证据当缓存删掉。

保留资产的旧报告和 `COMPLETE/completion` 不改写，原 hash 仍表示当时完成状态。删除权重的事实以本次清单、账本及旁置退役告示为准。已退役端点的完整历史重放/续训不可用；保存结果的离线分析仍可继续。不得为了通过旧校验而自动下载、恢复或重训已退役权重。

## 逐实验退役告示

外部 `MeshFlow-Control-artifacts` 目录的告示写入本中央目录的镜像树；没有向外部目录新增文档。

| 实验/资产目录 | 移除文件数 | 退役说明 |
| --- | ---: | --- |
| `E:/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1` | 4 | [退役告示](<retirement_notices/External/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow-Control-artifacts/experiments/r2_user_condition_finetune_pilot_v1` | 1 | [退役告示](<retirement_notices/External/MeshFlow-Control-artifacts/experiments/r2_user_condition_finetune_pilot_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/.worktrees/a-layout-value-20260924/output/a_system_official_closed16_20260925_01/training/seed_10/invocations/0001` | 1 | [退役告示](<retirement_notices/Other/MeshFlow/.worktrees/a-layout-value-20260924/output/a_system_official_closed16_20260925_01/training/seed_10/invocations/0001/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/mf_h1_local0` | 23 | [退役告示](<../../experiments/mf_h1_local0/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_bench_adaptation_strength_v1` | 9 | [退役告示](<../../experiments/r2_bench_adaptation_strength_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_bench_backend_relative_deformation_v1` | 3 | [退役告示](<../../experiments/r2_bench_backend_relative_deformation_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_bench_backend_shape_preservation_v1` | 4 | [退役告示](<../../experiments/r2_bench_backend_shape_preservation_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_bench_clean_multipart_seam_v1` | 2 | [退役告示](<../../experiments/r2_bench_clean_multipart_seam_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_bench_coverage_expansion_v1` | 3 | [退役告示](<../../experiments/r2_bench_coverage_expansion_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_bench_early_fm_v1` | 3 | [退役告示](<../../experiments/r2_bench_early_fm_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_bench_full_seat_curriculum_v1` | 8 | [退役告示](<../../experiments/r2_bench_full_seat_curriculum_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_bench_joint_shift_v1` | 5 | [退役告示](<../../experiments/r2_bench_joint_shift_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_bench_newdata_joint_v1` | 5 | [退役告示](<../../experiments/r2_bench_newdata_joint_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_bench_parts_convergence_v1` | 9 | [退役告示](<../../experiments/r2_bench_parts_convergence_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_bench_relation_training_v1` | 5 | [退役告示](<../../experiments/r2_bench_relation_training_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_bench_seat_coverage_v1` | 2 | [退役告示](<../../experiments/r2_bench_seat_coverage_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_bench_state_schedule_v1` | 6 | [退役告示](<../../experiments/r2_bench_state_schedule_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_bench_user_v2_training_v1` | 3 | [退役告示](<../../experiments/r2_bench_user_v2_training_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_c_reference_frame_v1` | 1 | [退役告示](<../../experiments/r2_c_reference_frame_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_full_contour_readout_v1` | 1 | [退役告示](<../../experiments/r2_full_contour_readout_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_full_seat_learning_curve_v1` | 4 | [退役告示](<../../experiments/r2_full_seat_learning_curve_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_owner_layout_v1` | 2 | [退役告示](<../../experiments/r2_owner_layout_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_seam_abc_v1` | 8 | [退役告示](<../../experiments/r2_seam_abc_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_seam_hinge_quality_v1` | 2 | [退役告示](<../../experiments/r2_seam_hinge_quality_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_seam_pool47_couple_v1` | 4 | [退役告示](<../../experiments/r2_seam_pool47_couple_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_seam_quality_four_v1` | 6 | [退役告示](<../../experiments/r2_seam_quality_four_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_seam_quality_wave1_wave2_v1` | 6 | [退役告示](<../../experiments/r2_seam_quality_wave1_wave2_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_seam_slot_c200_v1` | 5 | [退役告示](<../../experiments/r2_seam_slot_c200_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_seat_quick_learnability_v1` | 1 | [退役告示](<../../experiments/r2_seat_quick_learnability_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_single_step_denoiser_distillation_v1` | 1022 | [退役告示](<../../experiments/r2_single_step_denoiser_distillation_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_target_cleanup_c200_v1` | 4 | [退役告示](<../../experiments/r2_target_cleanup_c200_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_terminal_reward_nft_v1` | 2 | [退役告示](<../../experiments/r2_terminal_reward_nft_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_user_data_scaling_32_67_v1` | 3 | [退役告示](<../../experiments/r2_user_data_scaling_32_67_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_user_mix_from_official_v1` | 2 | [退役告示](<../../experiments/r2_user_mix_from_official_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/experiments/r2_window_policy_rl_v1` | 1 | [退役告示](<../../experiments/r2_window_policy_rl_v1/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/native_t1/checkpoints` | 2 | [退役告示](<retirement_notices/Other/MeshFlow/native_t1/checkpoints/WEIGHTS_RETIRED_20261025.md>) |
| `E:/MeshFlow/native_t1/checkpoints/ablations` | 3 | [退役告示](<retirement_notices/Other/MeshFlow/native_t1/checkpoints/ablations/WEIGHTS_RETIRED_20261025.md>) |

## 核验与边界

- [准备白名单与原文件 hash](<prepared_manifest.json>)
- [逐文件实际删除账本](<deletion_ledger.jsonl>)
- [最终清理回执](<cleanup_receipt.json>)
- [本轮审阅归档回执](<local_repair_archive_receipt.json>)

本轮清理不训练、不运行生成模型、不重新评分、不改变模型默认、不 commit/push。保留恢复权重只意味着资产仍可用，不授权继续训练。
