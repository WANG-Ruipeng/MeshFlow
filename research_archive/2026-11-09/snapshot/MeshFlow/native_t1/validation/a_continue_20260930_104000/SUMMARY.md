# A_continue 工作入口固化与代码整理

日期：2026-09-30。状态：**PASS**。本次没有新增训练，也没有新的实验矩阵。

## 当前默认模型

`python -B -m native_t1 sample ...` 默认使用纯 FM **A_continue，累计1500步**（原 T1 500 + A 500 + A_continue 500）。程序调用入口为 `native_t1.baseline.load_baseline()`。

```text
experiments/mf_h1_local0/outputs/a_then_fc_20260930_001700/A_continue/additional_step500_cumulative_step1500.pt
bytes: 1563278431
file SHA256: d8366d8c172407ec97ee4288143863825fae583d2c879d200f3862315747e400
full state SHA256: f162c79cc0d0d8dbd3ff208fecc6f433f348096d950652678a06cfaa328cc90b
backbone SHA256: 25d5414e6a3af1cbf1f0f71f27a68010f28318578aaf092af150d80b00e4a1a2
```

检查文件、配置、完整模型及主干哈希；严格加载，无缺失/额外参数；130,251,267参数，推理时全部冻结。不复制检查点、不恢复优化器、不读取官方EMA中转。未修改任何历史权重。

## 改动范围

- 新增 `baseline.py` 固定工作检查点身份，拒绝其他分支或计数。
- CLI 默认 profile 为 `a-continue`；显式 `original-t1` 仍使用原 step500。
- 旧 `checkpoint.py`、`load_t1()` 和 `verify.py` 保持字节不变；历史回归显式命名 `verify-historical`，旧 `verify` 保留兼容别名。
- 模型、runtime、sampler、数据流、纯 FM 训练 API、指标实现保持字节不变。官方模型和配置哈希仍匹配。
- 新增根 `WORKSPACE.md` 并更新包文档；生成资产加入Git忽略，源码仍可审阅。
- 历史研究代码继续留在原位置，新包不导入其中Python代码。历史多臂、FC/FF、Guidance、修补路径不作为日常入口。
- 改动前保留19份小源码/文档快照，共118,597字节；快照Python后缀为 `.py.txt`，不进入包导入。当前CLI文档已变化，不能声称整个源码与旧实验快照逐字相同。

## 实际验证

12/12 CPU检查通过：新增8项工作profile/身份/历史路由测试，以及原4项导入隔离测试。实际checkpoint读取、模型构造、GPU、训练调用在CPU门禁中均为0。见 [tests.json](tests.json)。

只执行一次**实际默认CLI**的 P0 / K12 / seed9601 完整重放，没有传 `--profile` 或 `--checkpoint` 覆盖默认：

- 严格加载检查点身份通过，前后模型state哈希相同。
- 与已有 A_continue 轨迹的全部51状态及最终输出逐位一致。
- 自由Gaussian逐位相同；C在全部51状态逐位保持。
- 实际50次模型前向、50次Euler更新、1次完整采样；反向0、优化器更新0、OT0、几何梯度0、后处理0。
- 模型采样耗时2.5824秒；加载及身份校验11.7028秒；CPU门禁2.9579秒。
- 严格确定性、TF32关闭、BF16前向、FP32参数/积分和math SDPA在实际路径中生效。

本次只是入口等价性检查，不产生新的模型质量或泛化结论。未运行旧 `verify-historical` 的两条生成，也未新增几何评分矩阵。

参考轨迹：
`experiments/mf_h1_local0/outputs/a_then_fc_20260930_001700/raw/A_continue_P0_K12_seed9601.npz`，
SHA256 `94834c2136410a6c6fcf3cc63a5c5dfb3954967d1b6a642729605a667bf1df6d`。

## 实际清理

删除2处已核查的可再生缓存，共18个 `.pyc`、302,507字节（约295.4 KiB）：

- `native_t1/__pycache__`：3文件、21,177字节。
- `experiments/mf_h1_local0/__pycache__`：15文件、281,330字节。

删除前确认绝对路径位于工作区、目录名为 `__pycache__`、没有reparse point、文件全部是 `.pyc`。未删除或移动历史源码、模型、输入、输出、原始数据、官方 `output/`、`.git`、`.worktrees`或环境。

## 记录

- [results.json](results.json)：模型身份、实际计数、逐位比较、源码保护与清理。
- [code.diff](code.diff)：本次源代码、文档与忽略规则的实际改动。
- [replay/run.json](replay/run.json)：实际默认CLI记录。
- [cleanup.json](cleanup.json)：具体删除路径、文件与字节数。
- [before_manifest.json](before_manifest.json)：小源码快照哈希。

工具层曾有一次PowerShell/bash内联引号错误，以及一次apply_patch因Windows sandbox logon1385未能执行；均未运行模型或改变实验计数。随后用已授权的文件写入和直接WSL命令完成。模型/测试没有失败重试。

当前Git仍有未提交的本地研究源码，未执行reset、clean、提交或历史重写。整理指工作代码与归档的用途隔离，不是清空历史研究记录。
