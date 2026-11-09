# GEOMETRY_CONTEXT_REPORT — 数值门禁早停

**本轮没有新增训练。** 实际完成3次纯T0前向后，预登记置换数值门禁失败；新增编码器尚未进入GPU验证。

本轮在训练前数值门禁停止。纯T0在组合面重排与面内S3角置换后的前向，未满足预登记BF16 allclose容差；失败发生在新增编码器接入GPU之前。
实现与CPU测试已经完成，新增几何/图分支的真实GPU接线、梯度、训练和自由生成尚未通过本轮验证。不能据此判断单面几何或邻接编码有益或无益。
T_base相对T0、T_geo相对T_base、T_graph相对T_geo以及原片/新片收益均未评价。没有新训练终点、生成图、CF/FF成绩或架构优势结论。
门槛、精度和核心数值路径未因失败而放宽；不继续训练，不追加模型前向，不自动进入下一阶段。

## 实际执行与预算

授权预算是三臂各500次更新；实际执行按下表记账。未将授权预算写成已完成工作。

| 项目 | 实际值 |
| --- | --- |
| forward_returns | 3 |
| forward_attempts | 3 |
| backward_returns | 0 |
| backward_attempts | 0 |
| optimizer_updates | 0 |
| complete_rollouts | 0 |
| euler_updates | 0 |
| OT_calls | 0 |

门禁耗时：9.2623秒。CPU测试中的fixture梯度/OT与模型调用分开记录；测试阶段不等于新增训练。

## 失败证据

共同初始化：`/mnt/e/MeshFlow/experiments/mf_h1_local0/outputs/a_then_fc_20260930_001700/A_continue/additional_step500_cumulative_step1500.pt`；SHA256 `d8366d8c172407ec97ee4288143863825fae583d2c879d200f3862315747e400`，累计1500步纯A。
严格加载记录：`PASS`；初始state SHA `f162c79cc0d0d8dbd3ff208fecc6f433f348096d950652678a06cfaa328cc90b`。
固定状态：P0/K12，t=0.5，CPU噪声seed12345，未运行OT；面/角排列seed24680。该状态独立于训练输入流seed1010。
这是本轮新诊断状态，不是历史完全相同case；仅沿用历史容差。置换同时包含面顺序与S3角顺序，因此本次失败不能单独归因到其中一种。
失败前没有记录峰值显存：NOT_RECORDED。没有用事后GPU当前占用替代峰值。

| 量 | 记录值 |
| --- | --- |
| max_abs | 0.0645 |
| RMS | 0.0088 |
| atol | 0.0156 |
| rtol | 0.0312 |

原错误：`RuntimeError("Fixed numerical gate failed: {'finite': True, 'bitwise': False, 'allclose': False, 'max_abs': 0.064453125, 'RMS': 0.008849970916358356, 'atol': 0.015625, 'rtol': 0.03125, 'status': 'FAIL'}")`。
单步原始输出：`/mnt/e/MeshFlow/native_t1/runs/geometry_context_20260930_142000/gate_single_step_proof.npz`；SHA256 `c586d996cfb0db5952eaf576e86465ccce76901a6bd1a8e822d1441e6a340f2c`。
失败比较采用原登记atol/rtol组合判定；不能只凭最大绝对误差单独替代allclose判据。首次门禁尚未完成的后续参数哈希/零分支/梯度/重放检查不计为通过。

## CPU测试记录

各套件有重叠，下列测试数量不能相加解释为互不重复的测试总数。

| 记录 | 状态 | 测试/家族数 | 耗时秒 |
| --- | --- | --- | --- |
| external_audit_fixture_test.json | PASS | 8 | 2.7581 |
| final_cpu_tests.json | PASS | 64 | 10.2803 |
| new_checkpoint_tests.json | PASS | 6 | 0.0022 |
| original_cpu_tests.json | PASS | 40 | 5.0330 |
| all_cpu_tests.log | PASS | 52 | 4.3000 |

## 已冻结的评价输入

评价注册：`/mnt/e/MeshFlow/native_t1/runs/geometry_context_20260930_142000/registration/evaluation_inputs.npz`；SHA256 `2a27cf5d576272947b6944f71b7f17ab32ca1489d43d267204303c114a82248e`。
原两对象C12与每对象两块新片已在训练前完成选择及hash登记；64条正式生成计划未执行。新片按原始源面ID的共享整边BFS选择，不根据生成结果挑片；各片保留自己的100面自由补集与固定评价点。

| 片 | 旧片重合面 | 新增面 | 面积占比 | 边界边数 |
| --- | --- | --- | --- | --- |
| P0_patch0_K12 | 12 | 0 | 0.3216 | 12 |
| P0_patch1_K12 | 2 | 10 | 0.0828 | 8 |
| P0_patch2_K12 | 0 | 12 | 0.0919 | 6 |
| P1_patch0_K12 | 12 | 0 | 0.1949 | 14 |
| P1_patch1_K12 | 1 | 11 | 0.1005 | 8 |
| P1_patch2_K12 | 0 | 12 | 0.1665 | 6 |

## 离线门禁审计

状态：PASS_REPRODUCED_BASELINE_PERMUTATION_FAILURE。此记录只读取既存速度数组/排列/代码证据，不是新的模型运行。
完整证据：[offline_gate_audit.json](offline_gate_audit.json)；完整内容另保留在results.json。
同输入重复逐位一致：True。离线复现成功不代表原数值门禁通过。

| 区域 | 元素数 | 越阈元素 | 涉及面 | 最大绝对差 | RMS | allclose |
| --- | --- | --- | --- | --- | --- | --- |
| 全部 | 1008 | 10 | 7 | 0.0645 | 0.0088 | False |
| known | 108 | 1 | 1 | 0.0312 | 0.0141 | False |
| free | 900 | 9 | 6 | 0.0645 | 0.0080 | False |
NumPy与既存Torch比较的阈值mask及排列对账见离线原记录。离线审计未追加模型前向、反向、OT或更新；没有将失败归因于尚未接入GPU的新编码器。

## 交付与完整性

- delivery_integrity.json：PASS；SHA256 `6b02e94fa38b36c0519d30960debfa89aed14b8fe3b18be9f33a8eaa16e95eb4`。
- code_manifest.json：RECORDED；SHA256 `5989c4ba5b002d29c4e65c181707f731ab3957de0449ec9dd418a2c257c3a165`。
- post_stop_guard_amendment.json：RECORDED；SHA256 `5704ffd16711cd160d9d60be3302b549065f0f7eb072f0bbcd505bca156864f5`。
完整源文件清单、代码diff与资产保护验证以相应原始JSON为准，并保留在results.json。
实际失败GPU调用采用原registration记录的执行源码。交付代码随后增加仅入口的fail-closed保护，阻止直接调用generate阶段绕过失败门禁；该修订单独记录，未改变数学或容差，也没有追加GPU执行。原registration pins和gates证据保持原样。

## 未执行事项

| 阶段 | 状态 |
| --- | --- |
| training | NOT_RUN |
| endpoint_generation | NOT_RUN |
| sampling_replays | NOT_RUN |
| strict_CF_FF_audit | NOT_RUN |
| full_topology_audit | NOT_RUN |
| fixed_view_figures | NOT_RUN |
| encoder_GPU_zero_branch_and_gradient_tests | NOT_RUN |
| encoder_delta_step10_100_500 | NOT_RUN |
| endpoint_save_reload | NOT_RUN |
| comparative_benefit | NOT_EVALUATED |

三个500步终点：T_base、T_geo、T_graph均NOT_RUN；没有对应新checkpoint路径或hash。
固定首噪声full/free图：NOT_RUN。没有采样数据，因此不创建空图或伪图。
本轮不能回答几何编码/邻接是否增强补全；下一步首先需要澄清纯T0置换门禁的数值契约与失败原因。没有擅自放宽容差或启动后续实验。
交付为已实现代码diff、实际CPU记录、该报告、results.json、test_summary.csv及既存门禁原始速度证据。没有新增训练权重或生成输出，也没有复制官方大权重。
