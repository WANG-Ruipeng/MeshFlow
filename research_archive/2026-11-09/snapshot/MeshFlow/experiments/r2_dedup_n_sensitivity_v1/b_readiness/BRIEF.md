# B：SEAM 原目标／去重目标 200 步配对准备

状态：**PREPARATION_ONLY / WAIT_FOR_A_GATE**。本文件只复用已保存的来源接口、去重映射、检查点审计与源码，未训练、未运行占用头或生成器、未计算新 OT、未产生新几何。A 的小 N 敏感性门禁未通过前，不启动 B。本文件不是新的自动执行入口。

## 最小可用共同池：14 父，不是旧四父，也不是全部16父

从 `r2_target_cleanup_v1` 的16个确实改变目标的父对象，与已完成的 `idea3/per_condition.csv`、`per_source_interface.csv` 按 condition_id 联接，现有 SEAM 单 owner 表示支持其中14父。它们的 C 均无已登记非流形边、内部同向边或坐标别名；每个合法 owner 只负责一条 C 边，第三点不是 C 顶点；原／去重后的 N 均为128..256，开放边数不超过 free 面数。

| 父 ID | 原 N | 去重 N | K | 删除 free 面 | C 开放边 | 来源占用 | Null |
|---|---:|---:|---:|---:|---:|---:|---:|
| 34b1b2ee54ea314486a1cb4b695edbd9 | 154 | 152 | 2 | 2 | 4 | 4 | 0 |
| 432ed2387d4982a635836c728d324152 | 256 | 248 | 4 | 8 | 6 | 6 | 0 |
| 473afe27e45b37101c8d18742f91c015 | 220 | 200 | 16 | 20 | 18 | 17 | 1 |
| 5d3cfbc8bae19e701f580a0f9153317f | 209 | 173 | 2 | 36 | 4 | 4 | 0 |
| 621c62aa667c4d3cfe2fba8e5d6db330 | 216 | 208 | 6 | 8 | 8 | 8 | 0 |
| 6c36a5b107e83d331191025061735ea3 | 159 | 132 | 6 | 27 | 8 | 6 | 2 |
| 776cdc2ed288076882eddefc99ebacfd | 198 | 187 | 2 | 11 | 4 | 4 | 0 |
| 7a5e63ee142a424c5f6d0af6de791c3 | 196 | 172 | 8 | 24 | 10 | 6 | 4 |
| a7ae914aba9eb59d84498bc295cd5a4a | 222 | 146 | 2 | 76 | 4 | 4 | 0 |
| a7b82d009e29bafe705fef512417e4c9 | 220 | 217 | 20 | 3 | 22 | 22 | 0 |
| d645bf9037e7dabda341228b21d337a9 | 224 | 166 | 7 | 58 | 9 | 9 | 0 |
| e1e7e66978b888703f146f1fbd99cb1a | 174 | 172 | 4 | 2 | 6 | 4 | 2 |
| e8e586ccf4dcac74cad9a65dbad13fea | 166 | 128 | 10 | 38 | 12 | 1 | 11 |
| f2a4d96e77d1c92a3d6cc735e3493c99 | 140 | 128 | 2 | 12 | 4 | 4 | 0 |
| 合计 | 2754 | 2429 | 91 | 325 | 119 | 99 | 20 |

两臂都排除 `5420d4699cf0125fd08f8c5dee475682` 与 `b192cda468f9390aa3f22b4b00de6dfb`：前者有1个多边 owner、2条第三点引用 C 的接口；后者有2个多边 owner、4条第三点引用 C 的接口。现有 `slot_data.supervised_sample` 明确拒绝这两类；不为凑16父引入 alias 解码或新的离散预测设计。它们分别只删除1面、4面，因此14父仍覆盖325/330个去重面。

旧 SEAM 四父73/3f/85/8c均没有本次精确去重变化，不能作为 B 的训练干预对照。它们仍可作冻结评测输入，必须称为已有四 C 迁移/回归面板，不能称新父泛化。

## 已有接口能复用，但旧入口不能直接改参数运行

`slot_data.py` 的布局和受限 OT 支持可变 N、K、开放边数：来源占用 owner 固定分配至 K+edge_id 槽；每个活跃 seam 的前两角点固定，第三点自由；Null 槽仍作为普通 bulk；bulk 仅在剩余真实 free 目标中 OT。去重后须用 `source_face_mapping.csv` 将原 owner face row 映射到 clean face row，保留原顶点身份/来源角序，并核验所有99个 owner 留存。不能将原面序号直接用于 compact 后数组。现有 `legal_interface_preservation.csv` 已记录原119条接口全部保留；14父子集对应99条。

后续真正启动时仍需物化两臂各自的新 case/OT 缓存并核对：C 数值与身份相同、owner+bulk 恰好覆盖全部目标且无重复、固定角点速度为0、FM 分母 `9(N-K)-6S`、原角序往返一致。这里尚未执行这些新缓存检查，不能写成已经有14父端到端 SEAM 数据包。

旧 `prepare.py` 和 `trainer_support.validate_batch` 写死四父每批各两次、132条边/124个正标签；旧 trainer 又写死 FULL_S6000→200 的恢复游标及头优化器。B 需要独立入口，保留这些旧文件只读。尤其不能用 `head=None` 实现冻结头：旧 update 用它选择普通 FM 路径，会误取消 seam_forward。应显式区分 `use_seam=True` 与 `train_head=False`。

## 同一起点及占用政策

建议最小 B 从同一份保留的 `SEAM_SLOT_C200_step200.pt` 克隆生成器 live 参数、匹配 AdamW、RNG；这是 SEAM 继续适配，不是从官方模型重新训练。该检查点已在先前 CPU readiness 实读核验：生成器 global_step=12200，194个 AdamW 状态均为12200；头 step=200，4个优化器状态齐全。新增200步后生成器 global_step=12400，分支 local_step=200；继承的 SEAM 累计适配400步须另列，不能混为12200→12200。

最小比较**无需新训占用头**：两臂使用同一冻结的833参数头、同一现有特征定义与同一 logit≥0 政策；训练 FM 的 owner 布局仍来自各自来源监督，不使用预测结果代替标签。头的覆盖能力是已知局限，必须报告预测占用与来源占用的差异；冻结头负结果不能泛化为“SEAM去重无效”。本轮不顺便修 BCE、改阈值或改成来源 oracle 推理。头不能读取父ID、Y或来源标签。

一个容易漏掉的混杂：现有 edge24 特征含 `N/256`。即使 C 与头都相同，若推理 N 不同，布局也可能变。B 主评测须预先固定**两臂相同的每请求 N、C与噪声**，由同一个冻结头计算一次布局并共享；这样不会把布局变化混入生成器的目标去重比较。训练目标 N 的改变及其 OT/分母改变仍属于此次数据干预的一部分，不能称为纯表面重数的完全隔离实验。无需假造 dummy 面保持训练 N。

若改成继续训练头，那是另一个设计选择：两臂同初始头/动量仍会因 N 特征不同产生不同梯度与布局，且旧四 C 权重8.25与66/124不等于14父的119边、99正20负先验。不能无声沿用或重新校准。为当前单变量快筛，建议先不引入这条额外路径。

## 最小预算建议与待锁定项

仅在 A 门禁通过后，建议固定两臂各200更新、batch8/microbatch1，原模型/FM/OT后HYBRID0.25/精度/增强不变。每臂1600实例，以同一预登记的14父平衡循环取样，父曝光为114或115次；增强/时间/随机流身份配对。N不同意味着实际噪声数组与受限 OT 必须各自构造，不能宣称 tensor 逐位相同。两臂合计400生成器更新、3200训练前向/反向、3200个生产 bulk OT；冻结头0次更新。

最小可复用评测提案是旧 SEAM 四 C×四噪声，两个新模型共32 Euler50和32次单次D0；冻结 SEAM 原16份作现状参照。若需要六 FULL C×四噪声的24请求面板，应正式锁为48 Euler50/48 D0并建立新的 C-only 布局，不冒称全部已有 SEAM 缓存。**最终评测面板、推理 N、预算仍待 A 结果后预登记；这里没有启动或扩预算。** 新 RAW 全部完成后再进入 D0，保持原硬锚、GEO_C、来源定向与原评分；不叠容量 GEO、RBF、第三点惩罚或 RL。

必须同时报告来源双面语义边界：已去重的330组是原始绕序相反的精确同位面，canonical训练将其呈现为相同目标；不能称为330个随机坏三角。B的325面子集保留无向表面并集，不保留全部原始双面重数/绕序语义。成败仍看真实图审与RAW/POST完整配对代价，不能只看训练 FM。

## 复用证据及哈希

以下均为本轮只读核对，未重新运行相交检测或检查点网络：

| 文件（相对仓库） | SHA256 |
|---|---|
| experiments/r2_target_cleanup_v1/paired_parent_summary.csv | 9548b6c43809b65bc846d4f67745779962bab56ba932c90b365f0de4301b4fb1 |
| experiments/r2_target_cleanup_v1/source_face_mapping.csv | 32d4132b46fcb4819a8eba981825fef4b5ec7109cd3c64a03436c0cd736946aa |
| experiments/r2_target_cleanup_v1/legal_interface_preservation.csv | d1c4b0fb5810d80f9fa0b05df89a115a631f7afaaeb8efa1ea7f6e851fcbd715 |
| experiments/r2_idea_1b_3_4a_diagnostic_v1/idea3/per_condition.csv | 3347e6aa502e8abdc152d07bd30280ebb9396c6ff3ad605c247f999a0174668e |
| experiments/r2_idea_1b_3_4a_diagnostic_v1/idea3/per_source_interface.csv | 667c1b4f7c9ba88073a6c7429526656908a68b51103ca2e56e1d175ee9267ef0 |
| experiments/r2_quality_followup_readiness_v1/checkpoint_audit.json | 1eb67559497e5a5ae9e26ee82696a4ceb1dfd835edea9231aeffabb6b410fa46 |
| r2/prototypes/seam_slot_c200_v1/slot_data.py | a5d6253e4da33b633e21e74858e29c813cbebe4787f04f6d9dd87c5f6478ceb6 |
| r2/pilots/seam_slot_c200_v1/trainer_support.py | b945f958a61c7b05d51c678beb050c89306d0b260b97d62b99ead63e28300735 |

保留起点检查点文件 SHA256：`c0c26ad26d2123062d3fd38039227d6cd05e269b5ca16e312602c45266fb3d65`；生成器状态 `fb8133fbd9095ea8d840f4fe4e8dace573ab5aa86c1e5bafc53b0f47985a4880`；头状态 `59534a5eb0cbde547a02e9f352ba8286feb1880df25c85afbe22ad5089003eb4`。本次复用了先前已完成的文件/内部状态核验，没有为写简报重新加载1.56GB检查点。
