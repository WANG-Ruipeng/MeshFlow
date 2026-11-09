# NFT 终态奖励路线负结果收口报告

日期：2026-10-15。范围：`r2_terminal_reward_nft_v1` 及其冻结训练条件补充检查 `r2_nft_training_c_frozen_v1`。本报告汇总已完成证据，不替换或回写两轮原报告、源码、评分与图审。

**当前配置按负结果收口，不继续扩大训练量、轮数或覆盖。** 第一轮没有扩展固定面板的明确联合成功覆盖；第二轮在原八条奖励训练 C 上检查新噪声，仍没有得到跨多条 C、可归因于奖励的稳定独有收益。NFT 有一个真实局部新增请求，不能写成“所有样本均无改善”；但该新增也由无奖励 SELF 实现，且共同已知的冻结原始奖励没有提高。这些证据不足以支持当前配置晋级，也不构成“RL 整体失败”的结论。

FULL_S6000 继续作为充分训练的方法比较参照。维护 CLI 默认仍按原 Chair HYBRID START 合同，本次没有切换默认。若未来重开 NFT/RL 路线，须提出相对本版的实质机制变化、可检验假说及独立预算，并另行获得授权、重新登记；本收口不授权沿原配置追加训练或新实验。

## 实际测试的配置与证据范围

两臂均从同一 FULL_S6000 live 生成器开始，local6000/global12000，非 EMA。冻结原主干、role 与 Geo，仅训练 48 个主干线性层的 rank4、alpha4 LoRA，**每臂 589,824 个可训练参数、256 次有效 AdamW 更新**，固定使用 round15 终点；没有选择最佳 checkpoint。lr=1e−5、microbatch1、累积8，BF16 前向、FP32 参数与 Euler50 积分，确定性运行合同保持不变。

NFT 使用八条按输入特征预选的 TRAIN FULL 条件，16 轮共生成512份在线 RAW，每 C 64份；对单次官方 D0、GEO_C、来源绕序后的 POST 进行原几何评分和实际用途图审，再给终态奖励。学习仍使用原定长 RAW 端点，不把删面后的 POST 当作训练端点。

SELF_REFLOW 的代码标签为 `SELF_FM`。它使用 **NFT 生成的同一端点池和同一重加噪、时间、曝光流**，做无奖励端点回归及相同参考约束；它是共享端点的 off-policy 对照，不是原来源 GT 上的普通续训，也没有自己的在线生成池。两臂匹配参数、端点与有效更新预算，前向计算量并不相等，因此不能把二者差异解释成所有形式的 RL 与非 RL 的普遍比较。 共享池还可能把 NFT 的端点分布变化传给 SELF，不能据此严格隔离奖励经端点分布产生的全部影响；这里的负结果仅指未观察到本配置相对该对照的稳定独有优势。

第一轮最终面板是六个父的36个配对请求：旧噪声24个、新噪声12个，TRAIN/DEV 各18个，每模型36份输出。仅85b169与八条奖励 C 重合。第二轮则把**原八条奖励 C 全部保留，每 C 四份未参与训练或前轮评价的新噪声**，三模型共96份输出，零训练、零反向、零失败重放。第二轮直接复用八个冻结 C2048 查询 bank 和原 reward/post/scorer。两轮条件与噪声构成不同，不合并为一个68请求成功率。

第一轮原报告提出的“原八条训练 C 内冻结检查”已经由第二轮完成，不能再将它当成尚未执行的建议。

## 第一轮：36请求未增加明确联合覆盖

“联合”为 RAW 采用 YES 且 POST 至少一条合法 CF 共边；“严格联合”还要求 POST 采用 YES、主体完整 YES。接口存在不认证完整接缝、主体可靠连接或可用成品。UNKNOWN 单独保留。

| 模型 | RAW采用 YES /36（UNKNOWN） | POST有合法接口 /36 | 严格联合 YES /36（UNKNOWN） | POST CF | POST FF | POST I下界＝上界 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FULL_S6000 | 22（9） | 11 | 9（2） | 5.2698% | 17.5258% | 2.4143% |
| NFT | 22（9） | 11 | 9（2） | 5.1603% | 16.0414% | 2.7262% |
| SELF_REFLOW | 22（10） | 10 | 9（1） | 4.6732% | 14.6409% | 3.1942% |

FULL 与 NFT 的 RAW/POST 采用标签、是否有合法接口逐请求一致。旧24请求三模型均有6个明确联合正例，新12请求均有3个；新噪声上的三个正例 FULL 已经具备，不能记成 NFT 学到的新能力。全部9个明确联合都在 TRAIN，DEV18请求无明确联合；FULL/NFT 各有2个未知联合，不能将其填作失败制造排序。

NFT 的合法边总数93→94、长度总和71.2134→71.4523模型单位，是已有接口输出的局部变动，没有扩展成功覆盖。POST FF 均值下降1.4844个百分点，同时 I 上升0.3120个百分点；SELF 的 FF 降幅更大、I 代价也更大，不能把 FF 改善独归因于奖励。

不利尾部保持可见：DEV608旧噪声 n1 的 POST FF 为16.4179%→25.5639%（+9.1460个百分点）；DEV608新噪声 n4 的 POST I 为4.2480%→15.5273%（+11.2793个百分点），同请求 RAW I 也增加9.0820个百分点。这一单例基本解释全组 I 均值的增量。逐例和父均值见[原分析表](../r2_terminal_reward_nft_v1/evaluation/analysis/per_output.csv)、[配对表](../r2_terminal_reward_nft_v1/evaluation/analysis/paired_per_request.csv)及[尾部表](../r2_terminal_reward_nft_v1/evaluation/numeric_crosscheck/tail_requests.csv)。

## 第二轮：八条训练 C 的新噪声检查仍不支持扩大本版

三模型的 RAW/POST 采用及主体标签逐请求完全一致；RAW 采用均26/32、UNKNOWN 均4。严格联合与普通联合在本轮相同。

| 模型 | POST有合法接口 /32 | 严格联合 /32 | 合法边总数 | 合法边长度总和 | POST CF | POST FF | POST I下界＝上界 | free保留率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FULL_S6000 | 11 | 11 | 61 | 70.2352 | 5.0763% | 13.0230% | 3.0594% | 74.8172% |
| NFT | 12 | 12 | 58 | 66.6572 | 4.8420% | 12.2591% | 3.1097% | 74.8219% |
| SELF_REFLOW | 11 | 11 | 62 | 70.4296 | 4.6039% | 13.0058% | 2.8610% | 74.9259% |

唯一扩展发生在 a7b82d：严格联合从 FULL 的2/4变为 NFT 的3/4。新增 n0 同样由 SELF 实现；SELF 另丢失原有 n2，而 NFT 保留了它，因此 SELF 该 C 仍为2/4。其余七条 C 的严格联合计数不变。这个局部正结果保留，但它不是跨多 C 的增益，也不是新增请求上的奖励独有收益。

固定奖励沿用原 `geometry_review_reward` 的原始 R，范围0至1.5，**不使用各模型自行组标准化的 r_NFT 比较**。用途门依赖 RAW采用、POST采用、主体完整三个实际标签；几何项包含合法接口/长度、CF、FF、I、退化及保留率。每 C 内平均四噪声，再对 C 等权；共同已知配对优先，未知原始 R 仍为缺失。

| 配对比较 | 共同已知请求 /32 | 已知C /8 | 候选平均R | 参照平均R | 配对ΔR | 含未知的全32请求ΔR区间 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| NFT − FULL | 28 | 7 | 0.464992 | 0.468106 | −0.003113 | [−0.025333, +0.019896] |
| SELF − FULL | 28 | 7 | 0.467465 | 0.468106 | −0.000641 | [−0.023169, +0.021814] |
| NFT − SELF | 28 | 7 | 0.464992 | 0.467465 | −0.002472 | [−0.024538, +0.020457] |

NFT 相对 FULL 在七条可判定 C 上三升四降。e6ea5e 的四份噪声、三模型采用均 UNKNOWN，所以整条 C 的原始 R 不可判定；它没有被删除。该 C 的接口确定为零，因此联合事件为否，这是合取关系，并非把未知采用改成 NO。表中全32区间是固定奖励公式及未知门产生的可行界，**不是统计置信区间，也不是未知填零**；区间跨零，不能声称八条 C 的真实平均奖励已被证明下降。

NFT 的 POST CF、FF 均值分别下降0.2343、0.7639个百分点，I 增加0.0504个百分点；合法接口输出多一份，但合法边总数和长度总和反而减少。e6ea5e/n2 的 FF 增加6.631个百分点，7a5e63/n2增加5.497个百分点，5f2441/n3的 I 增加1.904个百分点。二值接口增量不等于完整奖励或综合质量提升。完整八 C 与全部请求保存在[逐C表](../r2_nft_training_c_frozen_v1/evaluation/analysis/per_condition.csv)、[逐请求配对表](../r2_nft_training_c_frozen_v1/evaluation/analysis/paired_per_request.csv)、[逐C配对表](../r2_nft_training_c_frozen_v1/evaluation/analysis/paired_per_condition.csv)和[全部几何尾部](../r2_nft_training_c_frozen_v1/evaluation/analysis/geometry_tails.csv)。

## 保留的有效观察与结论边界

- 并非没有可行输出或没有更新：512训练端点中434个原始奖励已知、423个有有效偏好，61/64条件组有区分信息，严格联合171/512；两臂都实际完成256次有效更新。NFT 有效 LoRA 范数从16更新时0.01707增至终点0.08585，SELF 从0.02199增至0.31509。参数变化不等于任务收益，较小变化也不能单独证明训练不足。
- 冻结 FULL 的 round0 四条件、32输出搜索池中，随机八份平均 R 为0.660758，Best-of8为1.141174，差0.480416。这是同池候选选择收益，只覆盖四条 TRAIN C，不是学成的策略增益，也不代表 DEV 搜索效果。
- 当前结果不支持“只因训练条件收益未迁移到前轮面板”这个简单解释：原八 C 的冻结补充检查也未呈现多 C 稳定收益。也没有“冻结奖励已上升而任务不改善”的证据，不能将本次直接定性为已证实的 reward hacking 或已证明只需换奖励函数。
- 第一轮在线奖励曲线用不同策略、不同噪声和变化的已知覆盖；末批端点来自240更新策略。不能把在线涨跌当最终256更新策略的固定测试改善。
- 两轮用途评阅均为实际看图的模型标签盲审 **AI_PROXY**，不是人类金标准。第一轮训练奖励图审非盲；同一 BASE 旧24输出的采用标签也有历史16 YES与本轮13 YES的差异。统一使用各轮自身封存标签配对，不混用历史标签；仍需承认评阅者、视图、遮挡与薄层判断的局限。
- 第一轮仅六个反复观察的父、FRESH每 C 两份噪声；第二轮仅八条已受奖励监督的 C、每 C 四份噪声。没有统计显著优势结论，也不能外推到新父总体、物理可靠性或其他 RL 机制。两轮均保持 RAW/POST 分开；CF/FF 分母是该阶段实际 free 面数，I 是固定 C 查询的冲突覆盖，不能当作全网格相交体积。
- 第一轮角点离散度未显示整体收缩，但依赖三角划分与角点密度；不同数组哈希不证明语义多样性，不能据此排除模式塌缩。完整负例、未知与不利代价均保留。

第一轮逻辑更新总计512；记录故障导致 SELF 一次获准工程恢复，物理完成526、尝试527、训练前向10468，多出的14个完整更新、15次尝试和228次前向已单列，NFT没有重放。第二轮仅冻结采样，实际4800生成前向、96次单D0、192评分、零训练或重放。不能将工程恢复伪称为等计算预算，也不能把已保存的损失/梯度复现扩大为未保存原完整状态的逐字节证明。训练明细见[学习分析](../r2_terminal_reward_nft_v1/learning_analysis/snapshots/bec1ae29f546f1eec90e/summary.json)。

## 证据交付与权重清理状态

| 证据 | 第一轮 | 第二轮 |
| --- | --- | --- |
| 原报告 | [REPORT.md](../r2_terminal_reward_nft_v1/REPORT.md) | [REPORT.md](../r2_nft_training_c_frozen_v1/REPORT.md) |
| 完整阶段数值CSV | [216条RAW/POST](../r2_terminal_reward_nft_v1/evaluation/analysis/per_output_stage.csv) | [192条RAW/POST](../r2_nft_training_c_frozen_v1/evaluation/analysis/per_output_stage.csv) |
| 原尺寸匿名图集 | [36页](../r2_terminal_reward_nft_v1/evaluation/blind_gallery/index.html) | [32页](../r2_nft_training_c_frozen_v1/evaluation/blind_gallery/index.html) |
| 审阅包ZIP | [原训练及36请求证据包](../r2_terminal_reward_nft_v1/evidence_pack/terminal_reward_nft_v1_e5051ffa87a3de06.zip) | [八训练C冻结证据包](../r2_nft_training_c_frozen_v1/evidence_pack/NFT_TRAINING_C_FROZEN_REVIEW.zip) |
| ZIP校验回执 | [26,902,941字节](../r2_terminal_reward_nft_v1/evidence_pack/terminal_reward_nft_v1_e5051ffa87a3de06.receipt.json) | [4,219,656字节](../r2_nft_training_c_frozen_v1/evidence_pack/receipt.json) |
| 独立数值核验 | [3214项PASS](../r2_terminal_reward_nft_v1/evaluation/numeric_crosscheck/comparison.json) | [14128项PASS](../r2_nft_training_c_frozen_v1/numeric_crosscheck/comparison.json) |

两个 ZIP 均有成员哈希和 CRC 核验，包含完整逐例表与统一图像预览；大基座、D0、全量 RAW 和原尺寸图按外部路径保留，小包不是完整训练资产。第一包 SHA256 为 `e5051ffa87a3de0662e2e68d148f549db2349fe0234a041806c5fae1d139c19d`；第二包为 `a83a7f8241876670620ef00f4317af5bcab7a1d5d1025ea11e9ee6a5bc823a70`。

**权重清理已完成（2026-10-15）。** 按精确白名单删除原实验 `checkpoints/NFT/round00.pt`至`round14.pt`及`checkpoints/SELF_FM/round00.pt`至`round14.pt`，共30个中间恢复点；删除前逐文件SHA256与原sidecar及独立清单核对一致。逻辑删除287,949,354字节（274.61 MiB），清理窗口实际可用磁盘增加288,022,528字节（约274.68 MiB）；后者可能包含同期文件系统活动，不能将两者混作同一口径。两份round15与FULL基线清理后重算SHA256，均保持原身份。官方D0受保护，本次只核大小/修改时间，没有重复读取7.29 GB权重计算hash。

保留[最终NFT round15.pt](../r2_terminal_reward_nft_v1/checkpoints/NFT/round15.pt)、[最终SELF round15.pt](../r2_terminal_reward_nft_v1/checkpoints/SELF_FM/round15.pt)及其元数据，保留外部 FULL_S6000 与官方 D0；原数据、RAW/POST、全部评分、图审、训练事件、失败/恢复记录、源码、CSV和ZIP不属于该删除清单。清理后可以使用保留的最终适配器复现终点评价，但不能再从已删除的任意中间轮恢复；已有学习曲线、参数变化与训练审计记录仍是清理前保存的历史证据，不因此回写或重算。

实际删除路径、删除前哈希、删除后存在性、保留资产验证与空间口径见 [cleanup_receipt.json](cleanup_receipt.json)，独立候选核对见 [checkpoint_review.json](checkpoint_review.json)。原报告、原ZIP和执行源码保持原hash；全部30,485个原非权重证据文件按清理前清单核查存在性、大小和修改时间，最终核验结果见 [verification.json](verification.json)。这一全量文件清单校验不是重新计算网格或评分。

当前NFT/SELF轮次已收口；最终模型推理所需的资产仍在，但本次未重跑终点评价。中间轮次重放、依赖32份权重的旧全量审计不再具备完整磁盘前提，禁止把主动清理误报成原实验未完成。新增退役说明保留在原实验目录的RETIREMENT.md，不覆盖任何旧报告或审计。

本次仅整理文档、追加研究记忆并清理已核验的冗余权重：零模型加载、零前向、零训练、零新生成，未commit或push，未切换默认。
