# R2：全程混合条件 + 全程 HYBRID

R2 是从官方 Chair EMA 权重开始的条件后训练配方：**Native/T1 + role + Geo，20%/40% 条件全程混合，HYBRID λ=0.25 全程开启，只使用 free-only FM**。不从 START 续训，也不使用 B、教师或其他辅助损失。

这是现有 MeshFlow 仓库中的独立源码目录，权重、数据和运行产物保留在仓库外。旧 `native_t1` 的 START 默认采样入口继续保留。

- [仓库训练结构与调用关系](docs/training_structure.md)
- [使用与复现边界](docs/reproduction.md)
- [GitHub 云端核查](docs/cloud_audit.md)
- [迁移验证](docs/validation.md)
- [逐文件来源和 SHA256](docs/source_manifest.json)
- [官方骨干来源](docs/model_source_map.json)

## 入口

在 MeshFlow 仓库根目录执行，无需把旧 `E:\MeshFlow-Control` 加入路径：

```bash
python -B -m r2 --help
python -B -m r2 recipe
```

需要安装时，在独立的 Python 环境中使用 `pip install -e "./r2[test]"`，之后可执行 `meshflow-r2`。训练和 GPU 采样使用 Linux/WSL、PyTorch 2.7.x、SDPA math；不要与原版 README 的 FlashAttention 环境混用。新包为保持历史身份保留了 `meshflow_control` 导入名，不能在同一环境中同时安装旧独立工程和本目录。

| 命令 | 行为 |
|---|---|
| `recipe` | 打印固定配方，不导入 torch、不读取权重 |
| `inspect` | CPU 验证 R2 权重和协议，零模型前向 |
| `check-assets` | 核对历史数据清单、冻结计划与配对声明，不重新计算 OT |
| `prepare-run` | 为已有冻结输入登记一个新的 R2 运行，不启动训练 |
| `train` | 显式执行已登记的单条 R2 6000 步训练，或恢复安全中断 |
| `sample` | 显式给定 R2 导出权重、C、N 和种子，执行一次 Euler50 |

## 固定配方

| 项目 | 定义 |
|---|---|
| 初始化 | 官方 Chair EMA + 新 role/Geo；fresh AdamW；global step 从 0 开始 |
| 有效 batch / microbatch | 8 / 1；每批 8 个不同父对象 |
| 每批条件 | C20 原样 2 + C20 宽度增广 2 + C40 原样 2 + C40 宽度增广 2 |
| 增广 | X 宽度系数 [0.9, 1.1] |
| 匹配与混噪 | free-only OT 后再 HYBRID 混噪，λ=0.25，不重匹配 |
| 损失 | FM；按整个有效批次的有效 free 标量总数归一化 |
| 优化器 | AdamW，lr=1e-5，betas=(0.9,0.95)，weight decay=0，clip=1 |
| 训练预算 | 6000 次更新，48000 个条件样本曝光；每 100 步保存 |
| 精度 | BF16 前向，FP32 参数/优化器状态/积分；严格确定性 |
| 采样 | 标准 Gaussian，Euler50，C 在全部 51 个状态中硬保持 |
| 面数接口 | 总 N=128..256，FP32 C，0<K<N；训练只覆盖历史 C20/C40 |

**“全程混合”混合的是条件面数占比，不是多个模型。** “全程 HYBRID”指每一步训练均使用同一混噪配方；推理仍从标准 Gaussian 出发。

## 当前边界

训练加载器仍绑定历史 32 父/453 任务导入清单，R2 使用其中 26 父/411 条件池。20%/40% 是面数比例，不是表面积，也不代表当前筛选器已经满足用户输入质量要求。此版本不提供新条件构造或洗数据入口。

`train --resume` 只继续同一登记运行中安全中断且小于 6000 步的完整状态；不能据此在旧终点上更换数据并追加训练。主实验与确认实验的两份 R2 终点均可验证和采样，但没有自动选定默认终点。

为保留旧冻结输入与检查点的源文件身份，计算核心按原字节迁入。内部依赖中仍有旧协议名称和未启用的辅助模块；公共 R2 入口不会分派其他配方、加载教师或构建辅助头。旧 `tools/train_recipe_*.py` 只作为历史注册依赖保留，日常使用以上入口。

本次整理没有新增训练、真实模型前向或生成。代码发布依据用户明确的提交与推送请求，权重和数据继续保留在仓库外。
