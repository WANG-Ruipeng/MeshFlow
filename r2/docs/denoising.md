# 单步去噪与来源方向导出

`meshflow-r2 denoise` 将已经验证的处理顺序收进独立 R2 包：

```text
已保存 RAW = C + F
  → 官方 D0 单次完整预测（此时暂不写回 C）
  → 直接恢复 C，或显式选择 RBF 协调恢复
  → GEO_C 几何整理
  → 恢复原输入 C 角序
  → 固定 C，求解 free 方向
  → post.npz / post.ply + 状态与来源账本
```

新命令默认使用单次 D0、无 RBF、来源方向导出。已有 `sample` 仍输出 RAW；旧 `native_t1` 后处理合同不变。D0 不负责条件生成，没有新增训练、辅助损失或自动多轮去噪。

## 安装与输入

在仓库根目录执行 `pip install -e ./r2`，或者安装由 `r2/` 构建的 wheel。去噪需要 Linux/WSL、CUDA 和 PyTorch 2.7.x。纯 CPU 的来源捕获、缓存处理及方向导出不导入 torch，也不依赖 `native_t1`、实验脚本或旧独立工程。

权重单独提供，仅支持已经核验的官方 Chair D0 EMA：

- SHA256：`6a31f4c7dcbea7a6ac58c26075b20c897fd19aad4c33b5295cc4c93ed7078c36`
- 文件大小：7,289,979,546 字节；模型参数：455,592,963；严格加载全部 296 个 EMA 张量。
- 一次 BF16 decoder 前向，FP32 参数、残差相加与输出；严格确定性、math SDPA、TF32 关闭。内部尺度固定为 `0.95 * 0.3762`。
- 完整输入为有限 FP32 `(N,3,3)` 数组，前 K 面是 C；模型接口允许 N=1..800，但本地迁移面板不代表整个范围都经过效果验证。

RAW、C 必须已经处于同一模型坐标系。这里不自动规范化坐标、不推断物体尺度、不从完整 GT 提取条件。

## 在输入规范化之前保存 C

推荐输入一个只含 `vertices`（FP32）和 `faces`（整数索引）的 NPZ：

```bash
meshflow-r2 prepare-condition \
  --input /data/user_C.npz --input-key indexed \
  --out /runs/request/source_condition.npz
```

也可读取 FP32 三角面数组 `C.npy`，或只含 `C` 的 NPZ。没有显式顶点 ID 时，仅建立占位角点 ID，连接身份标为 UNKNOWN；不会通过坐标相等猜测来源连接，此类输入不能启用 RBF。

若外部前处理重排每面的角点，应在只含相关数组的输入 NPZ 中同时提供原 C、原顶点 ID 与排列 P：

```bash
meshflow-r2 prepare-condition \
  --input /data/source_with_mapping.npz --input-key C_original \
  --vertex-ids-key source_vertex_ids --permutation-key P \
  --out /runs/request/source_condition.npz
```

约定 `C_canonical[j] = C_original[P[j]]`，P 对每面给出 0、1、2 的排列。输出同时保存 `C_original`、规范化 `C`、P、来源顶点及面 ID。后续生成应读取这份文件中的 `C`；导出使用逆排列恢复原角序。未给 P 时，明确记录恒等排列：这是捕获当前输入，不能用它冒称恢复了历史上已经遗失的角序。

原实验保存的 `C`、`C_source_corner_permutations`、`C_source_vertex_ids` 也可直接使用，信任级别标记为 `SAVED_MAPPING_ONLY`。新版显式保存的原 C 会额外逐字节校验。没有映射的旧文件不会自动猜方向。

当前只承诺几何与已定义的来源字段；UV、颜色、逐角法线等未实现。捕获时出现未选中的数组、导出出现未知属性，会报错而非静默丢弃。

## 处理已有 RAW

```bash
meshflow-r2 denoise \
  --input /runs/request/raw.npz --input-key raw --known-faces 6 \
  --condition /runs/request/source_condition.npz \
  --checkpoint /weights/chair_denoiser_0030000.pt \
  --out /runs/request/d0_source_export
```

命令先校验 RAW 前缀与 C、权重身份及资源余量，再加载模型。单条调用只拥有一个模型，不自动增加 worker 或重试。多任务调度应由调用方按实测显存、计算利用率和 CPU/RAM 预算安排。

已有官方完整预测可跳过模型：将 `--checkpoint` 换成以下参数。

```bash
--proposal /cache/denoised.npz --proposal-receipt /cache/receipt.json
```

缓存 NPZ 必须含完整 FP32 `official_unconstrained_proposal`，包括 D0 预测过的 C；receipt 必须记录 `status=PASS`、RAW 文件 SHA256、官方 checkpoint SHA256 和缓存文件 SHA256。仅有已写回 C 的结果不能冒充完整预测供 RBF 使用。缓存验证检查文件身份与数组合同，不重新证明 receipt 的来源。

只需修正已有 GEO_C 的方向时：

```bash
meshflow-r2 orient \
  --input /runs/request/post_canonical.npz \
  --condition /runs/request/source_condition.npz \
  --out /runs/request/source_export
```

该入口不调用 D0，不重复 GEO_C。输入必须是本包兼容的标准 GEO_C NPZ，不能直接把任意 PLY 当成带完整来源的中间结果。

## 可选 RBF

在 `denoise` 上显式增加 `--coordination rbf --reference-length L`。L 必须是事先确定、跨方法一致的参考长度；不能从生成结果包围盒临时计算，也不应为了启用 RBF 而要求未知完整 GT。

实现沿用已验证参数：Wendland C2，半径 `0.1L`，ridge `1e-6`，重复来源顶点倒数权重，free 位移上限 `0.02L`。RBF 从尚未写回 C 的完整 D0 预测计算协调位移，再精确恢复 C；随后使用与 D0 基准相同的 GEO_C 和来源方向导出。

RBF 是显式候选，不是默认质量改进开关。旧面板上有 3/12 输出建立合法且方向一致的 C/F 共边，但座面 0/4，FF 与 I 仍有代价。完整逐例结果及指标口径见[研究结果](research_results.md)。本入口不搜索半径、位移限幅或其他求解器参数。

## 输出、恢复与边界

| 文件 | 用途 |
|---|---|
| `proposal.npz` / `proposal_receipt.json` | 原始完整 D0 预测及缓存身份；保留预测的 C |
| `denoised.npz` | 恢复 C 后、进入 GEO_C 前的三角面 |
| `post_canonical.npz/.ply` | 几何整理完成、仍采用模型内部角序 |
| `post_restored_C.npz/.ply` | 仅恢复来源 C 角序的中间结果 |
| `post.npz/.ply` | 固定来源 C 并整理 free 方向后的最终输出 |
| `source_restoration.json` | 原 C 恢复、来源顶点身份及几何不变量检查 |
| `orientation.json` | 分量、边约束、未解冲突和非流形状态 |
| `registration.json` / `ledger.jsonl` / `denoise.json` | 输入与实现 hash、阶段事件、最终产物 hash |
| `failure.json` | 失败状态与前向尝试数；禁止自动重试 |

每份输出使用新目录，输入与旧结果只读。`denoise --resume` 可验证并复用已完成结果；也可从完整缓存预测、尚未开始几何处理的位置继续。模型阶段或几何阶段不完整时会停止，不隐式重放。`D0_FORWARD_STARTED` 统计进入预测函数的尝试，`D0_FORWARD_RETURNED` 才表示成功返回。

定向阶段保留顶点坐标、无序面集合和既有索引连接，只改变允许的角序。GEO_C 自身按坐标建索引，并不保证任意用户文件中的重合但独立顶点 ID 都原样保留；因此导出另报来源连接身份 `PASS`、`CHANGED_BY_GEO_C` 或 `UNKNOWN_SOURCE_VERTEX_IDS`，不以顶点位置未变替代连接身份检查。

奇偶环矛盾、固定 C 冲突或不合格边会明确报告；不通过移动 C、删面、拆点或放松 C 解决。整体执行 `COMPLETE` 仅表示流程完成，不代表整网格可定向、水密、无相交、法线朝外或接合成功。纯 free 冲突分量可保留未满足约束。

需要旧规范化合同的调用方必须显式使用 `--winding canonical`；该选项不宣称恢复原输入方向。不同合同的输出分别保存，不能覆盖旧结果。

[迁移与安装验证](denoise_validation.md)记录真实验证范围；[几何来源](denoise_geometry_provenance.json)与[模型来源](../src/meshflow_control/denoiser_models/PROVENANCE.md)记录实际实现与许可出处。
