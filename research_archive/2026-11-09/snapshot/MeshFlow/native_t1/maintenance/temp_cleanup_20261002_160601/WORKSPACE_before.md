# 当前工作区：Native T1 + Geo，JEdge5 可选

维护入口为 `native_t1/`。恢复 Pair 实验前的主线：Native T1 + C-only Geo + 原 FM；JEdge5 作为可选训练 loss 保留，后续候选也沿用可选配方接口。Pair 活动代码、临时权重与生成数组已清理，不在训练或采样路径中。

## 使用 JEdge5

- 训练：`python -B -m native_t1 train-edge5 ...`，等价于 `train --context-encoder geo --loss-recipe edge5`。必须显式指定初始化、更新预算和新输出目录，不会自动开训。
- 采样：`python -B -m native_t1 sample --profile edge5 ...`，使用保留的 JEdge5 权重；不加 Guidance、投影、焊接或其他后处理。
- 未指定 profile 时，默认仍是 `fm-geo`，不会静默改成 JEdge5。FM、surface 与 edge5 通过同一可选 loss 接口选择。

JEdge5 = T1 + Geo + FM + 有限接口边 D_edge + 表面测度 L5。两项辅助 loss 只在训练使用；网络和原50步Euler采样器保持不变。具体命令与公式见 [主说明](native_t1/README.md)、[JEdge5配方](native_t1/docs/jedge5.md)、[可选loss接口](native_t1/docs/loss_recipes.md)。

## 代码边界

|模块|职责|
|---|---|
|model.py / context_geometry.py|时间非对称原生T1与已知面Geo编码|
|data.py / prepare.py|真实输入、原坐标变换、free-only nested OT和排列|
|training.py / train_cli.py|显式预算、fresh AdamW、FM训练路径|
|objectives.py / losses/|可选loss配方；没有Pair训练分支或七臂扫描器|
|geometry_checkpoint.py / working_model.py|严格区分FM、surface、JEdge5身份|
|sampling.py / runtime.py|原50步Euler、51状态C保持、确定性BF16/FP32|
|tests/|CPU数值、标签、身份、入口和导入边界测试|

## 保留权重

|角色|路径|累计步数|
|---|---|---|
|JEdge5可选工作模型|native_t1/checkpoints/jedge5_cumulative3000.pt|3000|
|FM默认基线|native_t1/checkpoints/ablations/fm_cumulative3000.pt|3000|
|共同起点G0|native_t1/checkpoints/tgeo_g0_cumulative2500.pt|2500|
|surface参照|native_t1/checkpoints/ablations/surface_cumulative3000.pt|3000|
|Edge-only参照|native_t1/checkpoints/ablations/edge_only_cumulative3000.pt|3000|

JEdge5文件SHA256：`cc7a248fa9a8570059f225cd1c313f21f08585925407a79418df6d63c9cae0e9`；已记录生成器state SHA256：`b9c1df5326ad4815ec3c32ed20e08dc34ff6248ff05be2c412e051fcb728aae3`。本次清理前后文件哈希一致，没有重新训练或覆盖权重。

## 2026-10-02 Pair清理

删除两处隔离目录：`experiments/mf_h1_local0/geometric_pair_closure_pilot/`、`native_t1/runs/geometric_pair_closure_20261001_171356/`。四份Pair step250/500 checkpoint、诊断及正式原始数组、连接候选、活动实验源码均退役。

报告、results、逐输出表、训练记录、固定图、协议和code.diff保存在 [精简档案](native_t1/maintenance/pair_cleanup_20261002_100559/CLEANUP_REPORT.md)。档案内历史报告的checkpoint/raw路径是执行时记录，现已删除；code.diff可恢复实验代码，但不恢复已删除的权重/数组。

官方权重、原数据、G0/FM/JEdge5和其他历史实验不在删除范围。本轮没有模型前向、训练更新、完整生成、reset/clean、commit或push；公共代码保持HEAD `5c11eb2`。权重、数据、runs、maintenance及本地WORKSPACE说明仍Git忽略。

本地已准备输入为 `output/t1_publish_20260930/chair_n112.npz`；公开代码可依据官方数据和固定recipe重新准备。当前证据仍来自两个观察过的父对象，不代表新对象泛化、水密或全指标优势。

## GPU任务调度约定（2026-10-02）

后续每次实验及新增GPU任务前按 [AGENTS.md](AGENTS.md) 检查显存、GPU忙碌率、已有进程及CPU/RAM；独立任务优先尝试有资源约束的并行，不再默认逐臂串行。并发数来自同类任务实际峰值和吞吐，保留显存余量，逐个启动并动态复核。

未知峰值时先观察一个已授权任务，不额外新增探针训练。训练、评价、采样分开估算；不改变microbatch、精度、随机流、确定性或实验预算来塞入更多任务。CPU匹配/OT/离线审计可与独立GPU工作重叠，独立输出和固定seed不随完成顺序变化。

本次只读取GPU状态和记录执行偏好，没有启动训练、采样或新的调度服务。检查时RTX5080总显存16303 MiB、可用13080 MiB、GPU利用率0%；这是空闲快照，不是运行峰值或固定并发数依据。

已有显存参考：`experiments/mf_h1_local0/outputs/tgeo_m10_20260930_173200/results.json` 中两臂 `arms.*.peak_cuda_allocated` 为约2.520/2.521 GiB；其72条 `generated[].sampling.peak_cuda_allocated` 为约0.536 GiB。这只是历史PyTorch allocated峰值，不含完整CUDA进程占用，也不是当前JEdge5所有loss的峰值；不能按16GB直接相除定并发。当前采样入口记录allocated峰值，公共训练入口尚无完整reserved/进程峰值记录，后续须随首个已授权任务补齐观测。allocated属于reserved的一部分，不把两者相加重复计算。
