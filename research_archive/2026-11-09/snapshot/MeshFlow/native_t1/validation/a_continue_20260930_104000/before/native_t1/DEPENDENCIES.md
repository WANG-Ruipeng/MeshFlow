# Native T1 的依赖边界

本目录是 T1 正确上下文终点的独立代码入口。新代码不导入 `experiments` 中的 Python 模块，也不复用旧适配器、几何引导或修补入口。历史权重、输入归档和回归轨迹仍可作为只读文件引用；代码独立不意味着复制或删除这些资产。

## 必要官方代码与配置

官方结构沿用当前仓库基线 `8cc74da013d3d2feec1853cccbb1f9ac37814b95` 中的文件，本次不修改它们：

| 文件 | 使用关系 | SHA256 |
| --- | --- | --- |
| [models/equidit.py](../models/equidit.py) | `checkpoint.py` 构造 `DiT`；原始 Transformer 子模块 | `f64bdc1901f7b9c3ee4a2a74d0a1975eace272196863fe0678db3306aebb167a` |
| [models/attention.py](../models/attention.py) | `equidit.py` 使用 `SelfAttention`；稳定运行关闭外部 FlashAttention | `5aec770043085b6d44fb87bd7f555be47b7ae6d1395172916253057550edf436` |
| [models/utils.py](../models/utils.py) | RMSNorm、SwiGLUFFN、坐标编码；加载时重建未注册的频率常量 | `cf700d864e9118d1731b43fbdec1d8d0d48eb7d91f57fd80f88610ba501369e4` |
| [configs/snet/base-120m-ot-v-chair.yaml](../configs/snet/base-120m-ot-v-chair.yaml) | T1 结构参数和 velocity 类型校验 | `d29f1e43ac468047b025751b9795b0c36a64dcf9f9e02dea381b57d54eed6442` |

训练数据流还保留一个独立官方依赖：[utils/ot_utils.py](../utils/ot_utils.py)，SHA256 `52c24f5da16fb5e897e73c6eaa5451a4b6ed8f6eba1ee606b4378fcb80e57772`。它只在显式取得训练样本时通过 NumPy/SciPy 做 Hungarian OT 排列；推理不调用它。`utils/__init__.py` 保留为包标记。

精确配置路径为 `E:\MeshFlow\configs\snet\base-120m-ot-v-chair.yaml`；WSL 对应 `/mnt/e/MeshFlow/configs/snet/base-120m-ot-v-chair.yaml`。`models/__init__.py` 作为包标记保留。

模型为 hidden=768、12 层、12 heads、version=3、pe_freq=20、RMSNorm 与 QK normalization、面数标签 `face_bin=20`、结构上限 800。已验证采样用实际 `N=y=112`，K 个已知面保留时间 1，自由面时间为 `k/50`。

YAML 中通用 `sample.cfg_scale=2.0`、优化器及通用训练设置不自动进入 Native T1 执行路径。已验证的 Native 采样每步直接调用一次模型速度，共 50 步；不能因读取该 YAML 额外添加 CFG 调用或替换时间表。

## 权重与输入文件

T1 是完整微调模型，检查点包含主干与形状为 `2×768` 的角色 embedding。推理直接加载下列完整终点，不先读取官方 2GB EMA，也不随机初始化替代缺失权重：

```text
experiments/mf_h1_local0/outputs/native_patch_20260929_010000/Native_correct/endpoint_step500.pt
file SHA256:     7928d77507bf1fe610876f61b5b11e614263323b5fc0f595cc6740ffd24af634
full state SHA:  37e31fe5925b8f870b9a4d3bbbbc84edc7bf30177946a6f67f4529a7f8e97172
backbone SHA:    e342012659cf7b747b6426a500e496895ca2b334c056e2ddab280eb131a2ea0f
```

`checkpoint.load_t1` 校验文件哈希、`native_connected_patch_v1 / Native_correct / completed_updates=500`，以 `weights_only=True` 读取，并严格加载完整 state。官方坐标编码器闭包中的未注册频率常量在 CPU 按原公式重建，不能保留 meta 张量或替换成不同编码。实际 tensor 哈希在加载后再次校验。

采样只需固定 C 和初始 Gaussian；GT 不属于模型或采样函数的输入。`data.load_archive(path, expected_sha256=None)` 通过显式参数读取现有 `inputs.npz`，不硬编码历史目录或 manifest；验证 N=112、K∈{2,4,8,12} 及精确 C/free 分解。数据适配与离线评分可以只读目标、面 ID、bbox 元数据及旧回归归档。原始坐标、C、权重和旧输出不得由新包重写。仅为回归使用旧轨迹，不把旧 Python 模块带入新执行进程。

## Python 包

以下是本机 WSL `meshflow` 环境的已安装版本记录，不是本次安装或升级结果：

| 包 | 当前版本 | 用途 |
| --- | --- | --- |
| PyTorch | `2.7.1+cu128` | 官方主干、Native 模型、CUDA BF16 与严格确定性执行 |
| NumPy | `1.26.4` | 数组、输入/输出归档、CPU 指标 |
| einops | `0.8.1` | 官方 attention 模块的导入依赖 |
| PyYAML | `6.0.3` | 严格结构配置读取 |
| trimesh | `4.10.0` | CPU 离线几何指标；不参与模型速度或采样修改 |
| SciPy | `1.15.3` | 显式训练数据配对调用官方 OT 时需要；trimesh 也可间接使用 |
| matplotlib | `3.10.9` | 已有环境中的可视化工具，核心模型/采样不依赖它 |

`flash-attn` 未安装；官方 attention 允许该导入失败，新包稳定运行主动使用 PyTorch math SDPA。POT 未安装，本次不增装依赖。仓库根 `requirements.txt` 面向完整研究仓库，含 UI、云服务、原训练等依赖，不是 Native T1 推理的最小依赖集合。

`runtime.py` 在 CUDA 初始化前要求 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，开启严格确定性并禁止仅告警降级；关闭 TF32、外部 FlashAttention 和非 math SDPA。推理保持 BF16 autocast、FP32 参数与积分坐标，冻结参数并使用 eval。导入新包不加载权重、不初始化 CUDA、不执行采样或训练。

## 当前代码职责

| 文件 | 职责 |
| --- | --- |
| `__init__.py` | 包入口及运行环境前置配置 |
| `checkpoint.py` | 独立严格加载完整 T1 权重 |
| `model.py` | Native 逐面时间与角色 embedding，使用官方子模块 |
| `runtime.py` | 确定性与精度配置、实际执行断言 |
| `artifacts.py` | 文件、数组和 state 哈希及结果写出 |
| `sampling.py` | 原 Native 50 步 clamped Euler；不接收 GT，不附加引导或修补 |
| `data.py` | 显式输入归档读取、真实面数与固定上下文校验、可复用训练样本流 |
| `training.py` | 显式 masked FM loss、优化器与单步训练 API；本轮未执行模型训练或优化器更新 |
| `__main__.py` | 显式 sample/verify CLI，main guard 阻止导入时执行 |
| `verify.py` | 有界两条逐 bit 回归；只读旧源码做 AST 对照，不导入旧实验模块 |
| `metrics.py`、`metric_geometry.py` | CPU 离线完成度、方向距离及边界指标；无模型/引导调用 |
| `tests/test_isolation.py` | AST 与全新 CPU 子进程导入边界检查 |
| `tests/test_metrics.py` | CPU 指标检查 |
| `tests/test_data.py` | CPU 数据流与 masked loss 检查 |
| `validation/` | 本次隔离、回归和来源记录；不是模型训练入口 |

完整判交及拓扑审计的历史证据保留在原实验记录；新轻量指标没有执行的项目明确 `NOT_RUN`，不能把旧完整审计自动算作新包已经运行。

## 导入隔离检查

从仓库根执行：

```bash
python -B -m unittest discover -s native_t1/tests -p test_isolation.py -v
```

检查所有生产 `.py` 的直接和相对 import、字面量动态 import，以及常见文件动态加载和 `sys.path` 变更。指向历史检查点的字符串是允许的只读资产路径，不是旧代码导入。

动态检查在全新子进程清空可见 GPU、禁止 `torch.load` 和 CUDA 初始化，导入所有生产模块但不调用 CLI `main`、loader、模型或训练函数，最后核对 `sys.modules` 没有旧实验或引导模块。它验证代码边界，不替代根进程执行的两条 51 状态逐 bit 采样回归，也不是针对任意恶意 Python 的安全沙箱。

2026-09-29 本机首次执行隔离检查：4 项 unittest 全部 PASS，耗时 3.606 秒；动态子进程导入全部 12 个生产模块，旧实验模块加载、checkpoint 读取、模型前向、CUDA 初始化、优化器更新均为 0。测试另含 9 个非法 import/动态加载小例以及合法包导入和历史资产路径小例。此记录不包含其它代理执行的 CPU 几何或数据测试。

## 可重建缓存与清理边界

本次只允许清理可从源文件重新生成的 Python `__pycache__` 目录。审计确认过的候选包括：

- `E:\MeshFlow\__pycache__`
- `E:\MeshFlow\models\__pycache__`
- `E:\MeshFlow\experiments\mf_h1_local0\__pycache__`

不能把 `experiments/`、各 `outputs/`、`output/`、`downloaded_data/`、`datasets/`、权重或数据包视为缓存删除。`native_t1` 当前仍引用只读 T1 终点和回归资产，移动或清除它们必须先修改显式路径并保留哈希验证。`.git`、`.worktrees`、Conda/PyTorch/CUDA 环境和驱动也不属于这份清理候选。本依赖审计本身没有删除文件。实际执行的缓存删除名单、大小、范围与结果以 [validation/cleanup.json](validation/cleanup.json) 为准；未出现记录的路径不推定已清理。不复制权重，不执行模型训练或环境升级。
