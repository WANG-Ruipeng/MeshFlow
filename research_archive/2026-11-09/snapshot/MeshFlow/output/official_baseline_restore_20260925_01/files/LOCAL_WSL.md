# RTX 5080 / WSL2 本地运行

环境：Ubuntu WSL2，独立 Conda 环境 `~/miniforge3/envs/meshflow`，Python 3.10，PyTorch 2.7.1 + CUDA 12.8。
代码和数据使用当前 Windows 仓库的 WSL 路径 `/mnt/e/MeshFlow`。

## 启动短训练

在 WSL 终端执行：

```bash
cd /mnt/e/MeshFlow
bash tools/run_wsl_smoke.sh
```

脚本使用单张 GPU 0、BF16、离线 W&B 日志。配置位于 `configs/local/wsl5080-smoke.yaml`：120M 级模型，batch size 1，梯度检查点，100 步，最后保存检查点并生成两个 OBJ。

预期输出：

- `output/wsl5080-smoke/log.txt`
- `output/wsl5080-smoke/checkpoints/00000100.pt`
- `output/wsl5080-smoke/infer_00000100/000000.obj`
- `output/wsl5080-smoke/infer_00000100/000001.obj`

重跑时可以传入不同实验名，保留先前结果：

```bash
bash tools/run_wsl_smoke.sh train.exp_name=wsl5080-smoke-2
```

## 环境重建

在已有 Miniforge 的 WSL 中：

```bash
~/miniforge3/bin/conda create -n meshflow python=3.10 pip -y
~/miniforge3/envs/meshflow/bin/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
~/miniforge3/envs/meshflow/bin/python -m pip install -r requirements-wsl.txt
```

不要再叠加原版 requirements.txt 中固定的旧 CUDA 和 Triton 版本。首次验证使用 PyTorch SDPA，无需外部 FlashAttention 或 CUDA Toolkit。

## 已做的代码适配

- `inference.py`：Chamfer 扩展改为按需加载；本地配置关闭 Chamfer 指标，避免导入时编译 CUDA 扩展。
- `utils/logger.py`：日志模式遵循 `WANDB_MODE`，本地脚本设置为 offline。

## 数据与结果的含义

`ss_overfit` 含 128 个训练条目、1 个验证条目，实际为同一个 526 面几何体的副本。可用于检查学习和采样链路，不能用于证明泛化能力。

100 步属于冒烟验证，不代表已达到论文质量。默认 EMA 衰减为 0.9999，短训练时 EMA 更新较慢。

现有训练恢复逻辑没有恢复优化器状态；长时间实验前应补齐，不能将其视为完全连续的断点恢复。

## 本机实测结果（2026-09-24）

- GPU：RTX 5080，PyTorch 显示 sm_120 支持，BF16 前向与反向计算通过。
- Python：3.10.21；PyTorch：2.7.1+cu128；完整安装版本见 `output/wsl-setup/requirements-lock.txt`。
- `pip check` 通过；不依赖外部 FlashAttention，也没有编译 Chamfer 扩展。
- 100 步训练、验证、检查点保存与重载、两个 OBJ 导出全部通过。
- 额外完成 1000 步训练，训练速度约 5.3–5.9 步/秒，观察到整卡显存约 6.2 GB（包含 Windows/其他程序占用，不是峰值统计）。
- 1000 步实验的每 100 步平均训练 loss 从 0.63297 降至 0.06886。
- 固定一个验证噪声样本，在 t=0.1/0.3/0.5/0.7/0.9 上取平均：零速度基线 MSE 1.22089，100 步当前模型 0.46331，1000 步当前模型 0.09293。两个实验独立初始化，这些结果不是泛化评估。
- 1000 步 EMA 模型在上述固定检查中的 MSE 为 1.16615；默认 EMA 衰减 0.9999，短训练时明显滞后。
- 当前训练权重的 50 时间点 CFG 采样已出现小牛轮廓，但面片仍破碎；默认 EMA 采样仍接近噪声。不能视为完整质量复现。

### 1000 步实验复跑

```bash
cd /mnt/e/MeshFlow
bash tools/run_wsl_smoke.sh \
  train.exp_name=wsl5080-overfit1000-rerun \
  train.max_steps=1000 train.ckpt_every=1000 train.log_every=100
```

实际保存的结果：

- 检查点：`output/wsl5080-overfit1000/checkpoints/00001000.pt`（包含当前模型、EMA、优化器与配置）。
- 日志：`output/wsl5080-overfit1000/log.txt`。
- 默认 EMA 输出：`output/wsl5080-overfit1000/infer_00001000/`。
- 当前权重单独预览：`output/wsl5080-overfit1000/current_weights_preview/sample.obj`。
- 真实网格对照：`output/wsl5080-overfit1000/current_weights_preview/ground_truth.obj`。
- 对照图：`output/wsl5080-overfit1000/mesh_comparison.png`。当前权重预览与默认 EMA 输出使用不同随机噪声/采样步数，图用于定性检查，不是受控采样质量比较。
- 检查结果：`output/wsl5080-overfit1000/verification.json`。

当前权重预览的复现命令（使用已有检查脚本）：

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 ~/miniforge3/envs/meshflow/bin/python \
  output/wsl-setup/verify_training.py \
  output/wsl5080-overfit1000/checkpoints/00001000.pt --preview
```
