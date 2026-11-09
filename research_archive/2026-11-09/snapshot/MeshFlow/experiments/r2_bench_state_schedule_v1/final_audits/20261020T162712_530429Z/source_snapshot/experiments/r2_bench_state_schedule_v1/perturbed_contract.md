# PERTURBED_STATE：off-path 终点纠偏回归合同

此候选使用原 Bench 冻结训练行、来源目标、受限 OT、HYBRID0.25、增强、
固定角点身份与原 free scalar 分母。它是**额外覆盖偏移输入态的终点纠偏回归**，
不是所写偏移路径的真实 CFM 导数，不宣称保持原条件概率路径。

对原存储 FP32 时间 `t`，当 `0.3 <= t <= 0.9` 且独立确定性行随机门禁
`u_gate < 0.25` 时：

```text
mu_C = 原有效 C 三角形的全部角点坐标平均
mu_Y = 目标中全部有效 free 角点平均（包含 seam tip；排除 C 和 seam 固定锚点）
alpha ~ Uniform[0.3, 0.8)
d = alpha * (mu_C - mu_Y)
xt'[free] = xt[free] + t*d
u'[free] = (x1[free] - xt'[free]) / (1-t)
```

`mu_C` 不是按唯一顶点去重的平均，也不是面积加权平均；`mu_Y` 不包含固定
SEAM 端点、C 或 padding。中心累加使用 FP64，实际 d、状态和速度目标使用
FP32。C 与 SEAM 锚点坐标和零速度目标保留原值，padding 原样保留。
未触发或关闭时，所有原数组保持原字节，不重新计算旧速度。
没有新增模型参数、辅助 loss 或推理侧来源目标；推理仍使用原同步 SEAM。

随机门禁与 alpha 由**原冻结 schedule row**的排序键 JSON、固定 salt 和不同
SHA256 域导出；使用各摘要前64位中的高53位映射到 `[0,1)`。不访问全局
Python、NumPy 或 torch RNG，不改变旧噪声、OT、增强、时间或样本顺序，也不以
worker ID、完成顺序或臂名生成随机数。调用方必须传原 schedule row，不能加入
当前 worker/arm 等临时字段。

## 数学边界

把同一个 d 当作时间无关路径元数据时，

```text
x_tilde(t) = (1-t) Z_H + t (Y + d)
真实导数     = Y - Z_H + d
本候选监督   = Y - Z_H - t/(1-t)*d
```

二者差为 `-d/(1-t)`，且所写偏移路径终点是 `Y+d`。本候选有意选择附件要求的
“从偏移状态一步指回原 Y”的回归目标，**没有擅自改成端点保持的抛物线桥**。
因此不能将它表述为原 CFM 的数学等价变换、概率传输正确性证明或 DAgger：
没有模型 rollout 聚合、专家逐状态标注或迭代数据回收。

时间上限 0.9 避免分母趋零，但不保证速度目标或梯度幅度不升高；记录每行
偏移范数、实际触发、alpha、原/新速度 SSE 与 L2/MSE 比例，不事后钳制幅度。
原 norm 为0的比例标为 NA，不伪造有限比值。

HYBRID 含独立 Gaussian epsilon，任意 `t<1` 的有效 free 状态并非严格缺失
整个重影区域；候选只检验提高所选合成偏移态训练频率是否有益。
向 C 与 free 质心之间整体偏移也不保证生成了真实坏例的折层/套娃几何。
成功须由冻结自由生成面板确认，不能由构造后一步终点恒等式或纠偏回归下降认证。

实现：`r2/pilots/bench_state_schedule_v1/perturbed_core.py`。
CPU 合同测试：`test_perturbed_core.py`。两文件不导入模型或 torch，
测试不调用前向、反向、优化器、GPU 或 OT。
