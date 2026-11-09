# FM baseline 主线恢复与临时文件清理

状态：COMPLETE。用户明确确认 JEdge5 作为可选项，后续继续在共同 baseline 上试新 loss。本轮只维护和清理，没有真实模型训练、完整生成、commit 或 push。

## 当前代码与默认模型

主线：Native T1 + C-only Geo + 原 masked FM。默认采样 profile 为 `fm-geo`，严格要求 FM 训练身份。

默认权重：`native_t1/checkpoints/ablations/fm_cumulative3000.pt`，累计3000步（历史L0）。保留原文件名和位置，未搬移或复制。

- 文件 SHA256：`0f22ce498ca2c27dc74b43890fb8aad20f0f3e7153aa74ee6d3c14ea445a2660`。
- 模型 state SHA256：`f006c49c8c70efa4bf2df6ca96431c844a25a1bdd38e2db4756115a37a23545b`。
- CPU 严格加载通过：无 missing/unexpected keys，模型参数不变。

JEdge5 保留为明确可选项：训练用 `train-edge5`，或 `train --context-encoder geo --loss-recipe edge5`；采样用 `sample --profile edge5`。其原权重不变。原始 T1、A_continue、通用 trained-geo 兼容入口保留，不混淆模型身份。

主训练命令显式传 `--context-encoder geo --loss-recipe fm`。底层训练 CLI 的历史 context 默认 `none` 仍用于原T1预热，不暗改其含义。文档列出从官方 EMA、原始数据到纯 FM 累计3000的完整六阶段训练命令。

架构和50步采样器未变。`model.py/context_geometry.py/sampling.py` 为共同生成路径；`training.py` 保留原FM，`objectives.py/losses/` 为可选训练配方。后续loss候选不需要将JEdge5固定到主架构中。

## 实际清理

共删除 **240 个已登记文件，3,180,734,285 bytes**。扣除新增的精简终点和源码ZIP，净释放 **2,658,943,197 bytes，约2.66 GB / 2.48 GiB**（维护小报告未计入这一净载荷数）。

- Edge-only step250中间权重和step500含optimizer的训练状态。
- 两轮最近试验的活动源码包：Edge factorial、Boundary cancellation；完整代码先ZIP备份并逐成员哈希核验。
- 五张已被_v2取代的绘图首版；最终图保留，旧引用有资产位置说明。
- 两个发布隔离测试checkout副本及一次性编辑脚本。
- 在核查范围未发现额外__pycache__、pyc、tmp或bak缓存。

唯一 Edge-only 终点没有丢失：去除optimizer动量后，以同一实验schema保留在 `native_t1/checkpoints/ablations/edge_only_cumulative3000.pt`（521710363 bytes）。模型每个tensor及输入RNG逐位一致，CPU严格加载通过；新文件SHA `61d09ce272d2b677ed0ae045e81e8a0c6229cbf77b08de0a612e4a76ff10015f`。

Edge-only不是主入口，其旧schema需恢复归档的专用loader使用，不能冒充FM或JEdge5。完整可恢复源在 `retired_recent_experiment_source.zip`，共23成员/80725 bytes。

409 个保护文件逐一核对大小与SHA，全部保持：包括主模型、JEdge5、L5、G0起点、最新两轮报告/results/raw、固定评价输入、最终图与源快照。原数据、官方资产和既有基础模型不在删除范围；独立几何评分器依赖也保留。Boundary候选没有训练checkpoint，报告和失败证据全部保留。

科学报告不回写。两个运行目录新增 `ASSET_RELOCATION.md`，说明退役路径及恢复方式。删除清单、原hash和替代资产均在 `cleanup_manifest.json`，实际删除记录在 `deletion_result.json`。

## 验证

- 完整CPU套件：89/89通过，5.527秒；此前针对profile切换10/10通过。
- 默认L0及精简Edge-only，共2次真实模型CPU严格加载；前向/反向/训练更新/完整生成均为0，CUDA未初始化。
- CLI help与默认路由正确；FM profile拒绝误传JEdge5/surface权重，JEdge5显式profile仍受原身份检查保护。
- 删除后409个保护文件哈希全部未变，240个目标文件全部已删除。
- `git diff --check`通过。未reset/clean；原有未提交的JEdge5代码继续保留，未删除用户工作树改动。
- 本目录 `code.diff` 包含当前公开代码的实际patch及未跟踪源码，排除所有权重、缓存和数据。

## GitHub 当前状态

已实时核查 `https://github.com/WANG-Ruipeng/MeshFlow.git` 的 `master`：

`5ef35b421ad00ccd3a97867ce0e33e5f77f65b1c` — `Add portable T_geo alongside pure Native T1`。

因此 **Native T1 + Geo 的可移植训练/采样代码已经在GitHub**；但是后续JEdge5损失支持、此次FM默认入口和文档整理仍是本地未提交修改，尚未推送。模型权重没有上传，继续遵守只发布代码的要求。

本轮用户询问提交状态，未追加新的发布操作；没有自行commit/push。代码和必要审计已整理为可审阅patch。
