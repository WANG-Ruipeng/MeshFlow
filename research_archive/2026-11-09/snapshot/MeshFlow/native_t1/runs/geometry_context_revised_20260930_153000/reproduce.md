# 本轮复现入口与门禁范围

本文件是交付附录，不是新的实验授权。当前run正在/已经执行；不得对同一输出目录重新运行训练或生成。所有复现命令都必须指定一个新的、尚不存在的目录。

代码基准为 `2502d5f89c54706cc4700d3cdea4a12fd6792529`。本目录 `code.diff` 只包含16个本轮源码/测试/独立审计包装器，未包含历史实验源码、权重、原数据、生成数组或缓存。已在当前工作树执行 `git apply --check --reverse code.diff`，仅校验、没有应用补丁或修改Git索引；文件清单和原字节哈希见 `code_manifest.json`。补丁保留源文件实际换行和“末尾无换行”标记。

## 本轮实际采用的初始化与v2门禁

本轮使用 `--init-profile a-continue`，初始权重是纯A累计1500步：

```text
SHA256 d8366d8c172407ec97ee4288143863825fae583d2c879d200f3862315747e400
```

本轮显式传入：

```text
--baseline-regression-evidence native_t1/runs/geometry_gate_code_audit_20260930_150400/results.json
```

这会登记 `baseline_regression_and_branch_inheritance_v2`。证据文件、配套 `velocities.npz`、checkpoint/state/input/源码身份和当前速度必须通过严格核对，才能采用本轮获授权的分层门禁：旧BF16全主干置换容差FAIL保留为非阻断数值诊断；原版T0回归、同输入重复、零分支匹配对应T0、特征/图/h1置换、梯度、冻结hash与C51保持仍是硬门禁。

这不是通用的“忽略数值失败”开关。证据位置及身份限定为本轮已验证的诊断；换checkpoint或换输入不能自动继承此豁免。旧失败run、9次只读前向、4次FP32诊断及局部算子调用分别记账，不能当成新run的训练或正式生成。

## 本机同资产复现模板

需要外部纯A1500权重、固定两对象的 prepared NPZ、已验证的v2证据，以及默认A_continue历史轨迹重放引用。它们都不随代码diff复制。若缺少必需资产，入口会停止，不能换随机模型或悄悄重训初始化。

在仓库根目录的已验证Python环境中执行。以下使用Bash语法，并按阶段失败即停：

```bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
NEW_RUN=native_t1/runs/geometry_context_reproduction_unique_name
COMMON=(
  --init-profile a-continue
  --inputs output/t1_publish_20260930/chair_n112.npz
  --input-seed 1010
  --updates 500
  --out "$NEW_RUN"
  --baseline-regression-evidence native_t1/runs/geometry_gate_code_audit_20260930_150400/results.json
)
for STAGE in register gates train generate; do
  python -B -m native_t1.geometry_pilot "${COMMON[@]}" --stage "$STAGE"
done
```

如果同一已验证A_continue文件存放在其他位置，可以额外显式提供 `--checkpoint /external/path/A_continue1500.pt`；其身份仍必须是上述固定权重。注册固定六块K12评价条件、9701–9704噪声与源码hash。训练为三臂各500次新更新；完整生成只用各固定终点，正式64条，重放最多4条。任何硬门禁或数值错误均停止；不自动重试、换阈值、扩训或增加噪声。

调用现有独立严格审计器是一个单独CPU进程：

```bash
python -B output/geometry_context_pilot_audit_20260930/audit_outputs.py \
  --manifest "$NEW_RUN/generated_outputs.json" \
  --out "$NEW_RUN/strict_audit"
python -B -m native_t1.geometry_pilot "${COMMON[@]}" \
  --stage report --audit-results "$NEW_RUN/strict_audit/audit.json"
```

独立包装器依赖本机既有严格审计实现；这些旧实现没有塞入 `native_t1`，也没有复制进本次补丁。其他机器若没有该审计器，应省略上述独立调用，使用 `--stage report`；报告会将CF/FF与完整拓扑明确记为NOT_RUN，不能将轻量距离指标当作无自交或水密认证。报告生成不会运行模型或重新采样。

## 没有本轮专用证据时：portable严格v1

新的portable纯T1累计1500步checkpoint使用：

```bash
python -B -m native_t1.geometry_pilot \
  --init-profile trained \
  --checkpoint /external/path/native_t1_cumulative1500.pt \
  --inputs /external/path/chair_n112.npz \
  --input-seed 1010 --updates 500 \
  --out native_t1/runs/portable_geometry_unique_name \
  --stage all
```

此命令没有 `--baseline-regression-evidence`，默认采用 `strict_permutation_v1`：原BF16置换容差仍是硬门禁。必须通过该checkpoint自己的门禁，失败会停止；不能声称新portable权重与本机历史A_continue相同，也不能自动套用v2的已知基线数值边界。

若尚未准备两对象NPZ，可先使用独立 `native_t1 prepare --data-root /external/shapenet --output /external/chair_n112.npz`。该准备器只读官方外部ShapeNet源文件，保留真实112面及原坐标处理，不执行模型或OT。此步骤不提供任何预训练权重，也不把一个随机模型当成累计1500步初始化。

生产设置继续是主干BF16、FP32参数/积分、新几何分支FP32；FP32全主干只出现在独立诊断中。不会通过复现入口自动切换生产精度、放大残差、加损失或进行后处理。