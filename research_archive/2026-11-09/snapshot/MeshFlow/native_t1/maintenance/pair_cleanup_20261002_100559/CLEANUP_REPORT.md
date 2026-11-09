# Pair本地临时文件清理完成

状态：COMPLETE。2026-10-02按用户授权清理Pair隔离实验，回到原T1+Geo主线及可选JEdge5工作方式。没有训练、采样、模型加载、环境升级或Git提交。

## 实际清理

- 删除登记文件：642个，共6,483,112,507 bytes（约6.483 GB）。
- 保留精简科学证据：41个文件，33,768,728 bytes（约33.77 MB）。
- 净释放载荷：6,449,343,779 bytes，约6.45 GB / 6.01 GiB；小型清理清单及本报告开销不含在载荷数中。
- 删除Pair的P_Time/P_Conf两臂各step250/500共四份checkpoint、teacher/生成原始数组、匹配/连接候选及私有活动源码。
- 删除范围只有 `native_t1/runs/geometric_pair_closure_20261001_171356/` 与 `experiments/mf_h1_local0/geometric_pair_closure_pilot/`。清理前验证绝对路径在E:/MeshFlow内，目标及子目录无junction/symlink；无运行中的Pair模块进程。

## 保留证据及旧路径说明

本目录 `evidence/` 保存原报告、完整results、160行raw/post表、code.diff与源码manifest、冻结协议、测试/调用计数、两臂training.json、阶段A汇总及固定对照图。所有41个复制文件均前后SHA一致。

[原实验报告](evidence/GP_CLOSURE_REPORT.md) · [结果JSON](evidence/results.json) · [逐输出表](evidence/raw_post_per_output.csv) · [代码diff](evidence/code.diff) · [固定图](evidence/figures/overview_patch4_seed10301.png)

历史报告未改写。其运行时checkpoint、raw、Stage2候选及部分详细诊断路径现在已经删除，不能直接再加载。代码diff可恢复27个Python/Markdown源码与必要测试文件；不会恢复删除的四个权重或原始数组。保留旧哈希是身份记录，不代表权重仍在。

## JEdge5与主线状态

公共代码未接入Pair，无需回滚；HEAD仍为 `5c11eb2d4a5297612eb77089d3840cb2c1036bde`，Git工作树clean。

- 显式JEdge5训练：`python -B -m native_t1 train-edge5 ...`，等价于 `train --context-encoder geo --loss-recipe edge5`。
- 显式JEdge5采样：`python -B -m native_t1 sample --profile edge5 ...`。
- 保留此前用户要求的可选loss设计：未指定profile仍默认 `fm-geo`，没有将JEdge5强制加入所有后续loss。
- JEdge5权重 `native_t1/checkpoints/jedge5_cumulative3000.pt`：521691011 bytes，文件SHA `cc7a248fa9a8570059f225cd1c313f21f08585925407a79418df6d63c9cae0e9`。
- 官方权重、原始数据、G0、FM、JEdge5及其他历史实验不在删除目标。G0/FM/JEdge5和5份公共入口/训练/采样源码清理前后哈希全部一致，见deletion_result.json。
- 修正本地ignored `WORKSPACE.md` 中过期的“默认edge5”描述，明确FM默认与JEdge5显式使用方式；原版本留为WORKSPACE_before.md。

## 验证与实际调用

全部642个目标文件已不存在，两目标目录已不存在。档案41个文件逐一哈希核对通过，保护文件全部未变；公开入口无Pair引用。

模型加载/前向/反向、optimizer更新、完整采样、科学几何审计均为0。本轮仅文件清理、哈希与静态入口检查，不需重跑训练测试。没有reset/clean、commit或push。

删除清单见cleanup_manifest.json，实际执行见deletion_result.json。Pair结果没有被当作新基线；后续若试新loss，继续通过可选配方接入并明确预算。
