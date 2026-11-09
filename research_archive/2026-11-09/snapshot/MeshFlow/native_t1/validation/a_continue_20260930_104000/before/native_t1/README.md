# Native In-Context T1

后续 T1 开发使用本目录。旧 `experiments/mf_h1_local0/` 作为历史记录保留；本包不导入其中的 Python 代码。官方 `models/` 与结构配置保持原样。

当前基准是 **Native_correct step500**：112 个面，已知 K=2/4/8/12，已知面时间为 1，自由面时间为 t；50 步 Euler，只更新自由面。没有适配器、额外 CFG、几何 Guidance 或导出前修补。此处不扩展模型结构、面数或坐标归一化范围。

## 代码入口

| 文件 | 职责 |
| --- | --- |
| `model.py` | 原 Native 逐面时间、角色 embedding 和官方主干运算 |
| `checkpoint.py` | 直接严格加载完整 T1 检查点，验证文件/模型/主干哈希；不读取官方 EMA |
| `runtime.py` | 严格确定性、BF16 前向、FP32 参数与坐标、math SDPA |
| `sampling.py` | 原 50 步采样；输入只有 Gaussian 和 C，不接受 GT |
| `data.py` | 归档加载、已知/自由划分、独立训练数据流 |
| `training.py` | 原 masked FM 损失、AdamW 与单步训练 API；导入不会训练 |
| `metrics.py`、`metric_geometry.py` | 固定采样的双向剩余/完整表面 RMS、绑定、覆盖与边界诊断 |
| `artifacts.py` | 哈希与结果写入 |
| `__main__.py` | `sample`、`verify` 两个显式命令 |
| `verify.py`、`tests/` | 保存轨迹回归、数据/指标/导入边界检查 |

## 使用

在仓库根目录运行，使用现有 WSL 环境 `/home/warpwang/miniforge3/envs/meshflow/bin/python`。未安装或更新依赖。具体官方代码依赖和已验证版本见 [DEPENDENCIES.md](DEPENDENCIES.md)。

```bash
python -B -m native_t1 --help

python -B -m native_t1 sample \
  --condition experiments/mf_h1_local0/outputs/native_patch_20260929_010000/inputs.npz \
  --condition-key parent0_K12_constraints \
  --seed 9401 \
  --out native_t1/runs/example_p0
```

也可以直接提供只含 C 的 `.npy`，省略 `--condition-key`。C 必须是原模型坐标系中的 FP32 `[K,3,3]` 或 `[K,9]`；不会自动改变尺度或角顺序。独立采样不需要目标网格或训练数据。输出为 `raw.npz`（完整结果、51 状态、初始噪声）和 `run.json`（严格加载、运行配置、调用数、哈希）。已有输出目录会拒绝覆盖。

默认权重引用现有 `experiments/mf_h1_local0/outputs/native_patch_20260929_010000/Native_correct/endpoint_step500.pt`，没有复制权重。`--checkpoint` 可指定同一检查点的其他位置，仍须通过固定哈希校验；这不是任意模型加载接口。

CPU 检查与显式 GPU 回归：

```bash
python -B -m unittest discover -s native_t1/tests -v
python -B -m native_t1 verify --out native_t1/runs/replay_check
```

`verify` 每次严格限制为 P0/P1、K12、seed9401 两次完整采样，共 100 次模型前向；比较旧档的全部 51 个状态。它需要历史输入、原模型定义文本与参考轨迹文件，仅作只读验证，不导入旧实验模块。已有验证目录不能覆盖。普通 `sample` 不依赖这些回归文件。

离线评价示例（只读，不修改生成结果）：

```python
import numpy as np
from native_t1.data import load_archive
from native_t1.metrics import evaluate_case

archive = load_archive("experiments/mf_h1_local0/outputs/native_patch_20260929_010000/inputs.npz")
with np.load("native_t1/runs/example_p0/raw.npz") as saved:
    metrics = evaluate_case(saved["output"], archive.cases, parent_index=0)
```

## 保留的范围

训练接口保留原混合 K4/8/12 的两父对象数据流、free-only OT、时间分布和角/面排列，以及 effective batch=8、microbatch=1、AdamW lr=1e-5、betas=(0.9,0.95)、weight_decay=0、grad clip=1。`load_t1` 默认冻结；训练必须由调用方显式解冻并调用单步 API。本次没有执行新训练。

历史的 500 步实验控制器、teacher 调度、断点恢复和报告编排没有搬进生产代码。首批数据流一致不等于已经重跑或复现完整训练。新模型初始化/新数据规模的支持也不在此次重构范围内。

轻量评价保留原核心数值口径；**完整 CF/FF 判交、精确拓扑审计未迁入，标为 NOT_RUN**。历史完整审计和旧结果仍保留，不能把旧审计当作新包执行结果。代码隔离不意味着新增了泛化或几何质量证据。

## 本次验证与清理

实际记录见 [validation/SUMMARY.md](validation/SUMMARY.md)。

- T1 两次完整回归：100 次前向，51 状态逐位一致，模型参数未变。
- 训练输入：首个真实 effective batch 与历史逐项对账；模型反向、参数更新均为 0。
- 评价：使用已保存输出对账，不新增生成矩阵。
- 历史代码、权重、数据、固定噪声、输入 manifest 和实验报告保持原位置。
- 只清理可重建的 Python 缓存；实际目录与大小见 [validation/cleanup.json](validation/cleanup.json)。不删除或移动 `.git`、`.worktrees`、历史输出或环境。

后续功能修改集中在 `native_t1/`；历史实验代码不再作为新功能的依赖。
