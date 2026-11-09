# 使用与复现边界

## 环境与源码隔离

本目录是独立 Python 包，不依赖运行时导入旧的 `E:\MeshFlow-Control` 或根目录 `native_t1`。骨干及必要适配代码随包提供，来源见 [清单](source_manifest.json)。

支持的计算环境为 Linux/WSL、Python >=3.10、PyTorch 2.7.x；使用 SDPA math，不安装外部 FlashAttention。推荐新环境进行可编辑安装：

```bash
python -m venv /path/to/r2-venv
source /path/to/r2-venv/bin/activate
python -m pip install -e "/path/to/MeshFlow/r2[test]"
meshflow-r2 recipe
```

Torch 的 CUDA 构建需匹配实际机器。此轮验证复用了本地已有 PyTorch 2.7.1 环境，没有安装或修改环境。由于保留了 `meshflow_control` 导入名，不要与旧独立项目安装在同一环境；在仓库根目录使用 `python -B -m r2` 会优先加载本目录源码。

训练注册需要源文件和 `tools/` 的相对路径，因此训练使用源码目录或可编辑安装，不以普通 wheel 安装作为复现入口。

## 已有本地资产

以下路径是本机 WSL 的真实路径，不代表这些文件包含在 Git：

```bash
ART=/mnt/e/MeshFlow-Control-artifacts
RUN=$ART/experiments/chair_condition_recipe_factorial_v1
DATA=$ART/data/train_manifest.json
STREAM=$RUN/streams
```

先只读检查：

```bash
python -B -m r2 check-assets --data-manifest "$DATA" --stream-root "$STREAM"
python -B -m r2 inspect \
  --checkpoint "$RUN/inference/R2_MIX_H_ALL_generator_step6000.pt" \
  --expected-sha256 6770a575e03e7845a7b0f491d201dc61c3fc42a254fa949b84c498eb8ea149e8
```

`inspect` 也支持确认实验的 generator 和两份完整训练 checkpoint，均在 CPU 验证。完整状态会核对模型/优化器/RNG/流状态哈希，generator 会在 CPU 严格装载，无前向。

| 终点 | generator 文件位置（相对 RUN） | generator SHA256 |
|---|---|---|
| 主实验 | inference/R2_MIX_H_ALL_generator_step6000.pt | 6770a575e03e7845a7b0f491d201dc61c3fc42a254fa949b84c498eb8ea149e8 |
| 确认实验 | confirmation/inference/R2_MIX_H_ALL_generator_step6000.pt | 9dc93885043121ea82287a105cb9ad3ab373827ce4c3b41802fcfe3b746f59a8 |

两份终点采用相同配方、不同输入流；本轮没有为用户自动选择其一。完整状态的 SHA256 另存于 `r2_spec.RETAINED`。

`check-assets` 核对完整 48000 条配对声明及计划身份，**不表示已经读取全部 OT 文件**。训练在使用每条记录前校验缓存与输入哈希；缺失缓存会报错，不自动重算 OT。

## 显式运行 R2

以下是未来启动已授权训练时的入口说明，本次整理没有执行这些命令。先指定官方 Chair EMA、外部的新运行目录及登记文件：

```bash
OFFICIAL=/path/to/official/chair_ema.pt
OUT=/path/outside/repository/r2_new_run
REG=/path/outside/repository/r2_registration.json

python -B -m r2 prepare-run \
  --official "$OFFICIAL" --data-manifest "$DATA" --stream-root "$STREAM" \
  --out "$OUT" --registration "$REG"

python -B -m r2 train \
  --official "$OFFICIAL" --data-manifest "$DATA" --stream-root "$STREAM" \
  --out "$OUT" --registration "$REG" --attempt-id attempt01
```

官方文件的 SHA256 固定为
`bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d`。

登记只绑定已经冻结的输入与源代码，**不会生成新 C、计算新 OT 或启动训练**。一次命令只运行 R2，不分派 R1/R3/R4/课程组，也不自动做采样、评分或重试。训练入口保留历史运行时的共享 GPU 锁、资源快照、累计六小时所有权上限；此上限不等于授权额外训练。新实验仍须遵守仓库 AGENTS 的预算和资源调度要求。

安全中断后的恢复使用同一登记、同一资产路径、同一源码、完整 `latest.pt` 和新的 attempt ID：

```bash
python -B -m r2 train \
  --official "$OFFICIAL" --data-manifest "$DATA" --stream-root "$STREAM" \
  --out "$OUT" --registration "$REG" --attempt-id attempt02 \
  --resume "$OUT/training/R2_MIX_H_ALL/latest.pt"
```

仅允许状态为 `INTERRUPTED_SAFE` 且小于 6000 步的最新完整状态。OOM、未闭合账本和其他失败需先核实恢复合同，不自动重试。已完成的 6000 步终点不通过此入口追加更新；不能更换条件池后沿用旧登记/动量身份。

## 显式采样

```bash
python -B -m r2 sample \
  --checkpoint "$RUN/inference/R2_MIX_H_ALL_generator_step6000.pt" \
  --expected-sha256 6770a575e03e7845a7b0f491d201dc61c3fc42a254fa949b84c498eb8ea149e8 \
  --condition /path/to/condition.npz --condition-key C \
  --num-faces 205 --seed 1234 --out /path/outside/repository/sample01
```

C 为 FP32 的 `[K,3,3]` 或 `[K,9]` 数组，0<K<N；总 N 必须显式给出且在 128..256。坐标应在训练坐标系，加载器不自动缩放、量化、焊接或修复。支持可变 K 的接口不证明所有条件占比都有已验证的质量。

采样只接受 generator 导出，使用一次 Gaussian/Euler50，持久账本最多一次 rollout、50 次前向。失败调用也计数；已有输出目录和账本不能被静默覆盖。保存 `raw.npz`、`raw.ply`、轨迹、条件保真检查和执行账本，不自动后处理。

## 当前不能声称的复现能力

本包不是从任意 ShapeNet 文件自动构建训练集的通用框架。历史清单含固定父池和绝对资产路径；换机器须按冻结数据合同恢复资产位置。重新构建数据、迁移缓存路径、修改条件选择器、扩大 N 或延长训练都需要新的显式版本，不能只改一个命令参数后声称复现旧实验。

CPU 验证证明源码/状态/抽查输入兼容，不证明此次迁移后已经完成 GPU 训练或生成等价性验证。本轮未执行这类计算。

## CPU 测试

```bash
PYTHONPATH=r2/src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -B -m pytest r2/tests -q -p no:cacheprovider
```

协议测试使用小型合成张量和替身模型检查序列化、匹配/混噪算术、优化器和错误处理；不会执行实际 MeshFlow 模型前向。
