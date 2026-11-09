# B TRAIN47 CPU 数据准备完成

`B_DATA_MANIFEST.json` 已完成全部 800 个更新批、6400 条训练样本、6400 次受限 bulk OT。两 CPU worker，每个单线程；数据阶段墙钟 135.687 秒，含末尾保护资产核验（最后一个批完成为134.583秒），不含前面的来源包准备和故障修复。没有 GPU、模型前向/反向、训练更新、评测噪声新抽样或质量重评分。

每个样本实际检查：来源面恰好覆盖原 N 一次、OT 面及角点身份对应目标 x1、来源绕序逆映射可回到原 source_faces、C 与 owner 两角锚点精确、FM 自由标量分母正确、third-in-C owner 仍为自由第三点。最终重新核验原输入、冻结源码与全部 C-only/label/metadata 资产 hash 未变。完整样本路径/hash/父身份/alpha/样本位置见 manifest；逐样本 `.started.json` 与 `.json` 留存，OT 没有失败或重试。

训练池：47 父，N=133..256，K=2..20；没有剔除任何“坏父”。43 父各曝光 136 次、4 父各 138 次，共 3200 clean 和 3200 x-width 增强。跨47排列边界的3个batch有同父两个位置，使用已冻结的独立seed命名；不是按接口边数量加权。

400 条 C 开放边中，366 条有来源合法相邻面，只实例化 360 个独立 owner。6 条 alias 回普通 bulk 候选槽，因此 head 标签是 owner-slot，不是“边是否需要接续”。18 条合法接口第三点已属于 C，其中12个实际 owner仍学习自由 tip；没有加入额外 GT 锚点。不能将360-owner召回与366-source-supported边召回混用。24维 C-only 特征不含父ID、source Y 或源边索引；推理不读取标签执行 alias 消解，误判/双激活风险需在结果中保留。

旧四条 C 的所有原有 case 数组逐项相等。但扩47需要原原型未支持的 owner/third-C 解释扩展，并改变父曝光与随机流，不能称全部实现和逐样本流完全相同的纯扩父消融。原 weighted BCE 常数、head 优化器政策不因400边的类别比例另行调整。

已交付的共同评测资产片段 `B_EVAL_DEV_ASSETS.json` 包含历史 DEV 三父、每父四个既有噪声。保存的是旧 FULL 请求高斯数组的精确 `z[K:]`，没有新噪声，C 与原请求逐位相等。标签文件与推理 C-only 文件分离；没有读取来源布局作为推理条件。原四 TRAIN 请求继续复用既有面板。

准备阶段有一次已记录工程错误：来源核验代码对 DEV 误取了不存在的 TRAIN 专用 `mask_path`。当时仅写好47父来源包，尚未执行任何 OT。`B_PREP_ENGINEERING_REPAIR01.json` 和修复前源码保留；显式恢复逐数组核对既有包，未覆盖它们，只补 DEV。这个错误不是训练/OT重试。

资源记录：`B_RESOURCES_CPU.json`、`B_RESOURCES_ASSET_REPAIR01.json`、`B_RESOURCES_DURING_OT.json`。中途每个 OT worker RSS 约65 MB，仅为一次快照，不声称峰值。初始可用内存约65.9 GB，24 CPU；GPU只读取状态，没有派遣GPU工作。

本文件只确认数据准备；不包含 B 模型训练或生成结论。训练入口由主任务统一冻结和调度，不能凭来源覆盖数量预判接续或质量会获胜。
