# SEAM_SLOT：共享原边的条件生成候选

本次整理日期：2026-10-17。SEAM 在四条已见完整座面条件的 200 步配对实验中带来了实际接续正信号；用户对已有图片的整体观感也更好。**FF 和 I 没有同步改善，因此当前定位是值得保留的研究候选，而非全面质量升级或新的默认模型。**

- [实际结果、完整逐例表与全部 16 张对照图](seam_slot_results.md)
- [可安装的核心实现](../src/meshflow_control/seam/README.md)
- [实际训练、采样与评分源码的来源清单](../research/seam_slot_c200_v1/README.md)
- [本次整理的 CPU 验证与发布边界](seam_slot_validation.md)

## 相对原 Native/T1 + role + Geo 改了什么

输入仍然是用户提供的真实三角面 C、显式总面数 N 和高斯噪声。需要保存 C 自身的原顶点身份、原绕序及内部角点排列；不需要推理时提供完整来源椅子。

1. 从 C 的原索引拓扑提取开放边。独立的 24→32→1 占用头（833 参数）读取 C 的几何特征与 K/N，按固定 `logit >= 0` 判断每条边是否启用 seam。
2. 在原有 N−K 个 free 槽中为每条开放边预留一个候选槽。启用时，前两个角引用对应 C 边的反向端点，只生成第三个角。未启用的槽作为普通 bulk 三角面使用，**不删 Null 槽，也不增加总 N**。
3. 复用生成器参数。原 C 面的 attention 时间为 1，所有非 C 面仍为 t；role、MLP 和最终层改为逐角 known/free 选择，启用 seam 的三个角为 `[1, 1, t]`。Geo 仍只读取原 C 面。
4. 训练时来源 owner 面固定到启用 seam 槽，剩余目标只在 bulk 中做原 OT；然后执行 HYBRID λ=0.25。FM 只对自由角标量归一化：占用 S 条边时分母为 `9(N−K)−6S`。占用 BCE 单独训练小头，不向生成器传回 BCE 梯度。
5. Euler 每步保持 C 和启用 seam 的两个锚角。后处理为单次官方 D0 → 恢复 C 与 seam 锚角 → 原 GEO_C → 来源角序恢复与 free 定向，并核验保留锚角最终确实引用同一 C 顶点索引。

固定底边会天然增加 C/F 共边，所以完整结果还报告 **从合法 seam 沿不经过 C 的合法 free 共边图到达 bulk 的面积**，分母排除 seam 面。该指标与方向一致可达面积是不同定义；都不认证完整水密、无自交或整圈封口。

## 代码的两种用途

### 可安装的机制核心

在已有适用 Python 环境中执行：

```bash
python -m pip install -e './r2[test]'
```

安装后，核心通过 `meshflow_control.seam` 使用，不依赖本地实验目录：

```python
from meshflow_control.seam.slot_data import edge_features, inference_layout
from meshflow_control.seam.corner_contract import decode_shared_indices
from meshflow_control.seam.model_adapter import seam_forward, corner_masked_fm_loss
from meshflow_control.seam.occupancy import OccupancyHead, occupancy_bce_loss
```

函数签名、输入合同和来源见核心 README。这些是对现有实现的原样整理，没有修改占用阈值、BCE 权重、匹配规则、生成器参数或 GEO_C。`seam_forward` 必须在原 `native_execution` 上下文中调用；布局、积分保持和导出由调用方显式管理。

### 已执行实验的完整计算路径

仓库保留 `r2/prototypes/seam_slot_c200_v1/` 原型与 `r2/pilots/seam_slot_c200_v1/` 的训练、采样、后处理入口及必需辅助源码。它们对应已冻结的四条件实验，**不作为任意新数据池的通用训练 CLI**。现有公开 `python -m r2 train/sample/denoise` 的行为不变；普通 `denoise` 不会自动理解 seam 锚角。

历史入口适用 Linux/WSL，额外资源记录依赖 `psutil`。从仓库根目录可以仅检查参数帮助，不启动任务：

```bash
PYTHONPATH=r2/src:. python -B -m r2.pilots.seam_slot_c200_v1.train --help
PYTHONPATH=r2/src:. python -B -m r2.pilots.seam_slot_c200_v1.generate --help
PYTHONPATH=r2/src:. python -B -m r2.pilots.seam_slot_c200_v1.postprocess --help
```

真正重放还需要本地冻结的 registration、输入数组、训练配对缓存、源码身份及匹配权重/AdamW/RNG。原登记包含本地绝对路径与哈希检查；不能在新机器上直接改掉检查后声称重现了旧实验。源码清单给出这些边界和保留权重身份。权重、原始数据、RAW/POST 网格不提交 Git。

`prototypes/.../DESIGN.md` 是开训前保留的历史设计，其“尚未训练”描述不是当前状态。当前执行结果以本页链接的最终结果文档和原报告副本为准。

## 结论边界

- 正式 C-only 结果使用预测占用。`SOURCE_ORACLE` 使用训练来源布局，只作诊断；解析去权重后全激活也不是已修好的分类器。
- 四条 TRAIN C 的短训结果不等于 DEV 泛化。来源占用是参考完成的监督，不是所有合理完成唯一正确的占用。
- 当前最小表示不支持同一来源 free 面拥有多条 C 边、或 seam 第三角本身引用 C 的训练特例；实现明确拒绝，不静默复制或丢面。
- 输入要求规范世界坐标 Chair、N=128..256，且开放边候选能放进 N−K 个槽。没有验证任意旋转、类别或面数范围。
- 容量保护 GEO、全激活与来源占用均单独保留。未自动加入新 loss、第三点角度门禁、RBF、RL、去重目标或多次去噪。
- FF 记录非法 free/free 相交或正面积重叠涉及的 free 面比例；I 记录 C 表面上穿越影响带与近重复覆盖的并集。外观改善与这两项恶化可以同时成立，不能以观感替代几何检查。

本次是代码与证据整理，不包含新训练、生成、权重清理或默认切换。
