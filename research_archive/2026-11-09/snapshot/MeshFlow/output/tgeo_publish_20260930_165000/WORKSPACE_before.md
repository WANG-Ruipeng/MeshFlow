# 当前工作区：纯 Native T1 / A_continue

日常开发集中在 [native_t1/](native_t1/README.md)。当前工作模型是上一轮纯 FM 的 **A_continue，累计1500步**：原 T1 step500 → A 继续500步 → A_continue 再继续500步。它与累计1000步的旧 A，以及同为累计1500步但加入 FC 的 A_then_FC，是不同的检查点。

| 位置 | 当前用途 |
| --- | --- |
| `native_t1/` | 当前维护的独立模型、加载、稳定运行、采样、纯 FM 训练 API、轻量评价 |
| `models/`、`configs/`、`utils/ot_utils.py` | 原官方基础代码与结构配置，保持原样 |
| `experiments/mf_h1_local0/` | 历史研究代码，只作复现和证据参考；新包不导入它 |
| `experiments/mf_h1_local0/outputs/` | 历史权重、输入、生成与报告，原路径只读保留 |
| `native_t1/runs/` | 今后的独立采样输出，已加入 Git 忽略 |
| `native_t1/validation/` | 代码隔离、入口切换、回归与清理记录 |
| `downloaded_data/`、`datasets/`、`output/` | 已有数据、官方脚本和旧资产；不作为临时目录清理 |

## 日常入口

在现有 WSL 环境、仓库根目录运行：

```bash
/home/warpwang/miniforge3/envs/meshflow/bin/python -B -m native_t1 sample \
  --condition experiments/mf_h1_local0/outputs/native_patch_20260929_010000/inputs.npz \
  --condition-key parent0_K12_constraints \
  --seed 9601 \
  --out native_t1/runs/a_continue_example
```

默认 profile 为 `a-continue`。只需原坐标尺度下的 FP32 已知三角形 C；也可传仅含 C 的 NPY。输出 `raw.npz` 和 `run.json`；已有输出目录拒绝覆盖。

当前为 N=112、50步 Euler、已知面固定、逐面时间、BF16 前向、FP32 参数与积分、严格确定性。纯 FM 训练 API 保留；没有自动训练入口。没有接口新层、FC/FF 辅助项、几何引导或后处理。

## 固定工作检查点

```text
experiments/mf_h1_local0/outputs/a_then_fc_20260930_001700/A_continue/additional_step500_cumulative_step1500.pt
SHA256: d8366d8c172407ec97ee4288143863825fae583d2c879d200f3862315747e400
```

权重按原路径引用，不复制。固定身份定义在 [native_t1/baseline.py](native_t1/baseline.py)。程序调用用 `from native_t1.baseline import load_baseline`；旧 `checkpoint.load_t1` 保留原 step500 语义，仅供历史调用。历史回归命令显式命名为 `verify-historical`，旧 `verify` 是兼容别名。

当前证据仍限于两个训练对象的 N112 沙盒，不代表新对象泛化。历史实验依赖和哈希互相引用，因此这次做入口隔离与缓存清理，没有移动或删除历史实验。版本控制忽略生成资产，但仍显示源代码改动。

本次切换记录：[native_t1/validation/a_continue_20260930_104000/SUMMARY.md](native_t1/validation/a_continue_20260930_104000/SUMMARY.md)。
