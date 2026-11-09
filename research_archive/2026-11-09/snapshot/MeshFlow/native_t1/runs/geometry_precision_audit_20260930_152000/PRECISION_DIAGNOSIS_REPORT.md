# PRECISION_DIAGNOSIS_REPORT

本次固定状态的BF16置换差异，首次在已观测的第一层SDPA输出（投影前）出现。FP32完整前向的组合置换误差明显缩小，但不是逐位一致。此结论只定位到已记录的模块边界，未定位SDPA内部更细的矩阵乘、softmax或归约步骤。

生产训练与采样仍保留原BF16路径；FP32仅为本次诊断。没有改变模型参数、生产精度或原BF16置换容差。

## 完整前向对比

BF16栏复用上一轮保存的速度；本次没有重跑这些BF16完整前向。FP32诊断allclose定义为每元素 `|a-b| <= 1e-5 + 1e-5*|b|`，不是只比较最大绝对误差与1e-5。

| 置换 | BF16 max abs | BF16 RMS | 原BF16 allclose | FP32 max abs | FP32 RMS | FP32诊断allclose |
| --- | --- | --- | --- | --- | --- | --- |
| 仅面序 | 0.064453125 | 0.00884997091636 | False | 1.13248825073e-05 | 1.22850699198e-06 | True |
| 仅S3角序 | 0 | 0 | True | 9.05990600586e-06 | 1.05254121118e-06 | True |
| 面序＋S3角序 | 0.064453125 | 0.00884997091636 | False | 1.15633010864e-05 | 1.4644505353e-06 | True |

原BF16组合置换仍为FAIL：10/1008元素超过atol=0.015625、rtol=0.03125；该FAIL没有因FP32诊断通过而改记为PASS。本状态BF16仅角序置换逐位一致，不能从单个状态推导任意输入都成立。
FP32执行了reference、face_only、corners_only、combined各一次。reference的自比较为零误差；这不构成额外的FP32同输入重复运行验证。

## 第一层的首次可见差异

两次独立第一层重放分别复现对应已保存完整前向轨迹，均逐位一致。对齐面/角排列后，各观测点如下；保存数组转为FP32以便记录，dtype栏保留实际算子输出dtype。

| 观测点 | 实际dtype | 对齐后逐位一致 | max abs | RMS |
| --- | --- | --- | --- | --- |
| pooled_face_before_norm | torch.bfloat16 | True | 0 | 0 |
| normalized_face | torch.float32 | True | 0 | 0 |
| QKV_input | torch.float32 | True | 0 | 0 |
| QKV_output | torch.bfloat16 | True | 0 | 0 |
| Q_normalized | torch.bfloat16 | True | 0 | 0 |
| K_normalized | torch.bfloat16 | True | 0 | 0 |
| SDPA_output_before_projection | torch.bfloat16 | False | 0.015625 | 6.01679211784e-05 |
| attention_after_projection | torch.bfloat16 | False | 0.0625 | 0.000418821375406 |
| hidden_after_attention | torch.bfloat16 | False | 0.0625 | 0.00024647794392 |
| MLP_output | torch.bfloat16 | False | 0.25 | 0.00208154702402 |

面内池化、归一化、QKV输入/输出以及Q/K归一化均逐位一致；SDPA_output_before_projection首次非逐位一致，max=0.015625、RMS=6.01679211784e-5。随后投影、残差和MLP观测点也出现差异。
因此，已保存证据支持该状态的差异首先在注意力聚合结果处可见；不能据此声称已证明某个具体SDPA内部算子的唯一根因，也不能由有限精度数值误差直接否定数学置换等变性。

## 实际调用与保护

| 操作 | 实际次数 |
| --- | --- |
| model_forward_attempts | 4 |
| model_forward_returns | 4 |
| model_backward_calls | 0 |
| optimizer_updates | 0 |
| complete_rollouts | 0 |
| OT_calls | 0 |
| standalone_block_calls | 2 |
| standalone_condition_embedder_calls | 3 |

其中4次是完整FP32模型前向；2次第一层及3次条件embedding独立调用单列，不换算为完整模型前向。零反向、零optimizer更新、零OT、零完整采样。此前失败门禁3次与另一次只读诊断9次前向属于各自既有账本，不重复混入本次4次。
耗时：11.389799秒；本次记录峰值CUDA allocated：578148352 bytes。该数值不替代此前失败run未记录的峰值。
共同checkpoint SHA256：`d8366d8c172407ec97ee4288143863825fae583d2c879d200f3862315747e400`。
参数hash前/后均为 `f162c79cc0d0d8dbd3ff208fecc6f433f348096d950652678a06cfaa328cc90b`；parameters_unchanged=True，source_unchanged=True，no_hooks_remaining=True。

## 证据与结论范围

- [results.json](results.json)，SHA256 `4c28afed3ee306015faba3932180b60173e6baa2fe85196e522a7ae6383555dc`。
- [fp32_velocities.npz](fp32_velocities.npz)，SHA256 `9d1a1e1112266ccc6d0fca6ac637d3e61f5003eb15caf2affaeab770f3464a5e`。
- [first_block_proof.npz](first_block_proof.npz)，SHA256 `8508ac21580ba947f05b531b37858ca5d3f192020a2380d2c84004e7da372277`。
- [registration.json](registration.json) 保留固定预算、源码及历史速度身份。

本报告只读取已经完成的JSON证据；没有新增模型/算子调用、重算几何或修改结果。它支持将原BF16完整主干置换误差作为明确披露的数值边界；是否继续实验由新授权的硬接线契约控制，不以本诊断改动生产精度或隐去原FAIL。
