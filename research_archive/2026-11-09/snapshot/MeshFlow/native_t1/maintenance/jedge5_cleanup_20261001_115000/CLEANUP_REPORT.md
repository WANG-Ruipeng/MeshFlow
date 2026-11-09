# JEdge5 主线整理与磁盘清理

状态：COMPLETE。仅维护与清理，没有启动新实验、训练或完整生成，也没有commit/push。

## 当前维护入口

主方法为 Native T1 + C-only Geo + FM + D_edge + L5。`train-edge5` 固定选择该配方；`sample` 默认加载本地JEdge5。原模型、C逐位保持机制与50步采样器未改。源码位于 `native_t1/`，训练损失集中在 `objectives.py` 与 `losses/`，不依赖历史experiment目录。纯FM和surface保留为显式消融，不作为并列默认方案。

README和方法文档包含官方资产到工作模型的训练步骤；固定系数沿用已完成pilot，没有根据生成结果重标定。现有证据仍是两个观察过的训练父对象，保留覆盖、接口尾部与质量取舍，未据清理声明新的实验优势。

## 保留的精简终点

| 角色 | 相对路径 | bytes | 文件SHA256 |
| --- | --- | --- | --- |
| JEdge5 | `native_t1/checkpoints/jedge5_cumulative3000.pt` | 521691011 | `cc7a248fa9a8570059f225cd1c313f21f08585925407a79418df6d63c9cae0e9` |
| L0 | `native_t1/checkpoints/ablations/fm_cumulative3000.pt` | 521689571 | `0f22ce498ca2c27dc74b43890fb8aad20f0f3e7153aa74ee6d3c14ea445a2660` |
| L5 | `native_t1/checkpoints/ablations/surface_cumulative3000.pt` | 521691211 | `93546de419582302e70cf140dae3aa44aad9ab913bd11b394482a8c347203999` |

三份各约522 MB，完整生成器与输入RNG逐位保留；CPU严格加载无缺失/多余key，模型state hash与旧终点一致。优化器动量被剥离，后续使用fresh AdamW，不可声称无缝恢复旧优化器轨迹。

G0累计2500起点、原T1/A、Geo2000、官方EMA及原数据保持不变。精简终点文件hash因序列化格式改变而改变，模型state hash不变；详见 `checkpoint_migration.json`。

## 实际清理

- 删除七臂筛选L0–L6各step250/500，共14个完整训练checkpoint。
- 删除J35/JEdge5各step250/500，共4个原完整训练checkpoint；JEdge5的模型与RNG已迁移。
- 共移除权重 28,171,791,430 bytes，新增三份精简权重 1,565,071,793 bytes，权重净释放 **26.607 GB（24.779 GiB）**。
- 旧筛选与pilot两个活动代码包的33个文件已删除，原代码存于126333-byte ZIP，逐成员hash验证通过；两个空目录已删除。
- L1/L2/L3/L4/L6/J35与中间权重退役。历史报告中的这些权重路径只保留为原始记录，不再可加载；每个删除项有原hash与替代/退役原因。
- 所有旧报告、results、表、图、原始数组、teacher状态和source快照保留。829份保护文件大小/hash全部不变，两棵worktree仍在。

`cleanup_manifest.json` 是完整删除与保留清单，`post_cleanup_verification.json` 为独立事后核验。官方大包没有复制或重打包，权重/数据/maintenance均保持Git忽略。

## 验证与实际调用

1. 原16个保存teacher速度状态、两项loss共32次对照：标量、完整速度梯度、诊断计数均逐位一致，最大梯度差0；8个源标签映射hash一致。
2. 7项几何CPU fixture通过；独立FP64有限差分最大误差3.0263e-10。相关CPU autograd记录在 `loss_migration_verification.json`，不是新的模型反向或训练。
3. 迁移前完整CPU套件87/87通过；删除旧依赖后再次87/87通过，耗时5.216秒，日志 `post_cleanup_tests.log`。两个CLI help/import检查通过。
4. 3次完整模型CPU严格加载；生成器前向0、模型参数反向0、optimizer更新0、完整生成0、CUDA初始化0。
5. 严格拒绝将纯FM schema附加伪造metadata后冒充edge5；已加入回归测试。
6. Git diff检查通过，实际patch为 `code.diff`。一次WSL默认换行配置检查误报全仓CRLF，随后使用与Windows一致的命令级配置核对通过；未改全仓换行或Git持久配置。

保存的GT-only核函数常量仅在当前批中使用；没有跨更新缓存模型编码或预测梯度。新入口未经本轮GPU训练/完整采样复跑，验证范围是迁移数学一致性、CPU合同、严格权重加载和历史证据保护。
