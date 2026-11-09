# MOMENT 封存与 START 正式接入

2026-10-05，按用户明确要求完成回退、清理与正式代码整理。当前 `native_t1 sample` 默认是 `chair-hybrid`，即原 OT_HYBRID pilot1000 / conditional_total5000 的 START。没有把 H_CONT 或 MOMENT 终点当作 START。

## 实际清理

逐文件删除 8061 项，共 **15,677,075,950 bytes（15.677 GB / 14.600 GiB）**。这是删除文件的逻辑字节合计，不是磁盘空闲空间增量测量。

|类别|文件数|字节数|
|---|---:|---:|
|MOMENT_XY V2 正式 step200/400/600/800/1000 权重|5|7,831,804,343|
|同一 V2 实验 H_CONT 正式 step200/400/600/800/1000 权重|5|7,825,414,599|
|V2 派生 OT 映射 NPZ|8000|19,319,007|
|MOMENT V2 活动源码包|20|208,665|
|Finite Surface V1 活动源码包|19|223,152|
|Finite Surface V1R1 活动源码包|12|106,184|

三个活动源码包先封存至 [retired_activity_source.zip](retired_activity_source.zip)，51 个成员逐一 SHA256 和 ZIP CRC 校验通过。归档 SHA256 为 `719cf69c39cf0117fe69dfa0c6a86dc739674676f169ba9220926a570e2604b4`。随后按已解析的绝对路径逐文件删除，仅移除确认空的包目录，没有递归删除 runs。

原始报告、逐例/逐曲线 CSV、冻结输入、raw、训练日志、逐点诊断与评分源码快照均保留。8000 个 OT attempt JSON 与 10 个权重 sidecar JSON 留作历史证据。1,276,851 bytes 的小型合成 CPU 恢复测试 fixture 也保留；它不是正式训练权重。封存源代码不再是活动运行依赖。

旧报告、manifest 和 sidecar 中的已删除 PT 路径是当时的历史记录，不表示这些权重仍可恢复或可重新采样。三个退役 run 根目录新增 RETIRED.json 说明；原报告没有改写。

## START 身份和代码

- 正式路径：`native_t1/checkpoints/chair_hybrid_start_total5000.pt`。
- 原路径：`native_t1/runs/chair_hybrid_coupling_v1/checkpoints/OT_HYBRID_pilot1000_total5000.pt`。
- 两路径为同一 NTFS 硬链接文件，1,565,074,939 bytes，没有再复制一份大权重。
- 文件 SHA256：`d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80`。
- 模型状态 SHA256：`166c71fef888f26494fc4a0af1776b77a484efa72a6f3ce06552c604fbecc972`。

正式代码包含独立 ChairModel、固定身份加载器、原始 MT19937 噪声与 Euler50 采样路径、CLI、身份 manifest 和 CPU 测试。模型 forward AST 与原 MVPModel 一致，无 MOMENT 关系参数或 retired experiment imports。N 必须显式指定在 128..256，C 位于前 K 个槽位，0<K<N。

原 fm-geo/edge5 等 N112 profile 保持显式可选。旧 N112 训练入口未改作 START 续训入口；本次没有启动训练、修改 loss 或扫描求解器。START 完整原 AdamW/RNG/输入流状态仍保存在本地 checkpoint 中，推理加载不恢复它们。

Git 提交仅包含正式代码、文档、测试及小型身份 manifest。权重、raw、归档和运行记录继续留在本地忽略目录。具体提交号与提交文件清单见 [git_commit_receipt.json](git_commit_receipt.json)；本次没有 push。

## 验证与账本

1. [删除清单](delete_plan.json)与[逐文件执行账本](deletions.jsonl)共 8061 条，全部确认已不存在；[执行结果](cleanup_result.json)为 DONE。
2. 删除前后 11,488 个保留文件逐一比较 SHA256 与字节数，缺失或变化 **0**，包含三组退役 run 的保留证据、两条 START 路径及既有决策 ZIP。见 [保护校验](protection_verification_after.json)。其他数据/官方权重不在删除范围；不把此数解释为全仓库哈希覆盖。
3. 原 START 真实 checkpoint 的 CPU strict load 通过：194 个 FP32 参数张量，130,401,155 参数，missing/unexpected keys 均 0；冻结/eval，未初始化 CUDA。见 [加载审计](start_loader_audit.json)与[硬链接审计](start_hardlink_audit.json)。加载审计先于硬链接创建，其当时 canonical_present=false 已由后者及删除后校验补齐。
4. 完整公共 CPU 测试 **109/109 PASS**，failure/error/skip 均 0，禁止真实 backbone forward、CUDA 初始化与 AdamW 更新的 guard 未触发。测试有小型合成模块运算、坐标 autograd、临时合成 checkpoint 读取；不计作真实模型训练。见 [测试账本](full_cpu_tests.json)及 [日志](full_cpu_tests.log)。
5. 删除后 `python -B -m native_t1 sample --help` 成功；独立只读代码终审通过，无退休活动包依赖。

本次真实模型前向、训练更新、生成均为 **0**。验证范围是身份、加载、代码/公式一致性与 CPU 合同测试；没有新增 GPU 生成或声称新的 GPU 逐位复现证据。

先前交付的 [决策材料 ZIP](../../reviews/CHAIR_MOMENT_CHANNEL_V2_DECISION_MATERIALS.zip) 保持原 SHA256 `c246ff3d45aa587d9e4b1ae63d79790c08789d7d2cffabc32e4041aed91a23f3`。本次封存是用户决定；V2 的有限实验结果不足以晋级 MOMENT，也不据此声称所有 moment 类方法普遍无效。
