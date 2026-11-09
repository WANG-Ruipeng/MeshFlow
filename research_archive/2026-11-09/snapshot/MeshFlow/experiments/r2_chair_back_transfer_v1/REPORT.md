# Chair BACK_FULL 冻结迁移检查

状态：完成。日期：2026-10-19。实验：`R2_CHAIR_BACK_TRANSFER_V1`。

**结论：当前 Chair SEAM 能在多数被检查的靠背输入上生成可辨认的椅身，并让 C 承担靠背用途；但座面强结果中的主体接续能力尚未稳定迁移。** 实际图审14/16靠背用途为 YES、2/16不确定；最终只有2/16达到预登记的 bulk 可达面积≥30%。这两类证据分开保存，不把“有合法 C/F 边”当成完整接续成功。

此次只检查完整靠背，没有运行椅腿、扶手或其他新部位。没有训练、选新权重或改变默认模型。

## 固定输入与证据边界

从现有 BACK_FULL 条件中，先核对 C 连贯性、来源拓扑/绕序、非退化与可逆身份，要求 K≥2。候选按 `(K, parent_id, condition_id)` 排序，取四分位固定秩；没有按生成结果或来源 Y 质量挑选。熟悉组可用31条，DEV可用18条，各取4条，每条预先保存2份新噪声。5条不连贯输入在生成前排除，原因见 `eligibility.json`。全部8条 C 已在生成前实际查看。

| 组别 | 父ID前缀 | K | N | C开放边数 |
| --- | --- | ---: | ---: | ---: |
| FAMILIAR | f4f1aba6 | 2 | 133 | 4 |
| FAMILIAR | 6c36a5b1 | 12 | 159 | 14 |
| FAMILIAR | 8c9b2023 | 22 | 183 | 20 |
| FAMILIAR | b631b78c | 32 | 208 | 34 |
| DEV | fb8c3a83 | 2 | 216 | 4 |
| DEV | abbe7279 | 11 | 194 | 11 |
| DEV | f199965d | 17 | 128 | 19 |
| DEV | 9d81bc69 | 29 | 176 | 31 |

FAMILIAR表示父对象属于TRAIN47；其BACK_FULL可能已经在早期EXP67训练中出现，不能称“全新条件泛化”。DEV未进入条件适配训练，但官方预训练是否见过这些父为UNKNOWN。每父一个C、两个噪声，规模只适合范围检查。完整来源模型不作为唯一应当生成的款式。

## 固定模型、处理与执行

生成器为保留的 `PLAIN_S2000_generator.pt`（这里的PLAIN仍是SEAM表示；live global15000），布局为冻结 `LONG_occupancy.pt`，logit阈值0。新8条C都实际经过C-only头，137条开放边全部激活；这不是owner准确率，也没有使用来源Y布局。

固定流程：标准Gaussian → Euler50 → 单次官方Chair D0 → 恢复C/seam锚点 → GEO_C和来源绕序恢复 → HINGE_OPP015 → ALPHA → HOLE。`FINAL_POST`是最后一个阶段。FP32参数/积分、BF16前向、严格确定性运行合同保持。

实际执行800次生成器前向、8次小头前向、16次D0前向；0训练更新、0反向、0失败工作进程、0科学重试。全部RAW保存后才开始D0。模型、输入、噪声、51个积分状态的锚点和导出索引经独立CPU验收PASS。

资源：RTX5080 16GiB、24 CPU核；先测首个授权任务再派第二个8请求分区。两个RAW进程的实际rollout时间区间没有重叠，不声称并行加速。D0一个进程加载一次处理16份，CPU评分/后处理2workers。稀疏整卡快照最高占用4379MiB、最低空闲11601MiB；不是连续进程峰值。RAW allocator最大reserved618MiB、D0为1812MiB，不包含完整驱动/其他进程显存。16次RAW积分合计51.45秒，D0阶段含加载/保存52.88秒；CPU后处理186.33秒、80阶段评分55.83秒、图库7.06秒（部分并行，不能直接当隔离benchmark相加）。

账本：[execution_ledger.json](execution_ledger.json)。工程预检修正记录在 `audits/score_preflight_before_execution/`，发生于正式评分前，原定义/输入/旧结果保留。

## 完整数值结果

以下为8父等权均值（本面板每父2份，亦等于逐输出均值）。CF、FF是非法面对所影响的free面占比；I是固定C查询上的ConflictCov，含横穿/近重复覆盖，不是精确相交面积。I两分项可能重叠，不可直接相加。≥30%仅为预登记的描述阈值，不能认证整个接缝封闭或物体可用。

| 阶段 | CF↓ | FF↓ | I↓ | seam→原bulk可达面积 | 可达≥30% | 小角<5°↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RAW | 11.14% | 89.44% | 19.75% | 0.00% | 0/16 | 15.21% |
| BASE_POST | 7.46% | 21.06% | 21.26% | 8.23% | 1/16 | 14.77% |
| HINGE_OPP015 | 6.74% | 20.03% | 21.18% | 16.70% | 2/16 | 14.69% |
| ALPHA | 6.74% | 18.85% | 21.18% | 16.93% | 2/16 | 14.58% |
| FINAL_POST | 6.73% | 18.83% | 21.18% | 16.96% | 2/16 | 14.57% |

RAW三角汤没有GEO建立的自由共享索引，RAW图可达为0不能单独当作几何没有相邻关系。RAW的FF实际为89.44%；BASE大幅降低包含D0及GEO的共同效果，不能归功于其中单独一个操作。

| 组别 | 图审用途YES / 不确定 | BASE→FINAL ≥30% | 平均bulk可达 BASE→FINAL | 最终FF | 最终I |
| --- | ---: | ---: | ---: | ---: | ---: |
| FAMILIAR | 8 / 0 | 0 → 0 / 8 | 2.07% → 5.52% | 19.49% | 18.45% |
| DEV | 6 / 2 | 1 → 2 / 8 | 14.39% → 28.40% | 18.17% | 23.91% |
| ALL | 14 / 2 | 1 → 2 / 16 | 8.23% → 16.96% | 18.83% | 21.18% |

全部16份BASE/FINAL均有合法且反向绕序的C/F接口，但多数仍只连到小片seam。两个≥30%的最终输出为 **DEV f199965d/n1（99.66%）和9d81bc69/n0（95.21%）**；后者在HINGE阶段新增达到阈值。其余14份均未达，熟悉组为0/8。不能由DEV均值较高反推DEV整体更易或已泛化，只是两个大连通分量抬高均值。

额外CPU后处理的实际收益：FF 21.06%→18.83%，CF 7.46%→6.73%，I 21.26%→21.18%；合法图单面孤岛13.31→9.88个/请求，原始边图单面孤岛8.31→5.31。非流形边11.13个/请求不变，GEO删free面14.19张/请求不变。

HINGE接受69次、ALPHA39次，HOLE添加4张桥面。BASE原有2567张面全部保留；原合法CF边234/234、FF边2136/2136保留；最终新增合法CF边1条、FF边148条。按原检测器定义，新增非法CF/FF面对均为0，旧非法CF面对248→229、FF443→400。固定BASE面分母的最终FF为18.855%，因此收益不是只用4张新增面稀释分母。4个请求平均形状因子略恶化，不能称“所有指标零退化”。桥面仅来源映射ID为−1，输出面ID仍唯一非负。

完整179个数值字段、80阶段行均保存在 [per_output.csv](evaluation/tables/per_output.csv)；另有逐条件、逐父、配对增量、面对和合法边记录。验收：[cpu_post_acceptance.json](cpu_post_acceptance.json)。

## 实际图审与截面

16页全部实际打开查看，每页4阶段、两个相对表面视角和前/侧固定位置截面；共64条阶段记录。评阅身份是非盲AI_PROXY，非人类盲评。颜色蓝=C、灰=生成；相同C跨噪声/阶段共用尺度和剖切位置。图片切线不自动等同真实裂缝，所有精确拓扑/水密判断在视觉记录中保留UNKNOWN，并用独立几何评分解释。

14份能明确辨认C的靠背角色，2份fb8c保留不确定：灰色弧背遮挡蓝色平板、上角突出，侧面存在分层关系。背板前后层也可能是合理厚度，本轮不凭截图宣布所有灰背层非法。16份都能辨认座面、腿/支撑与靠背；整体合理性13份YES、3份不确定。

熟悉组的整板、斜板、条栅和曲面靠背均有可辨认用途，但f4f/n1的蓝板伸到座下、b631/n1有横跨蓝背的灰条片/厚箱体，8c/n0有腿间小悬片候选。DEV的abbe/f199仍有背缘薄片和侧面间隔。9d81/n1填充了原C框的中心开口；没有给定空域禁止条件，因此不自动判作失败。

BASE→FINAL大部分页面在当前视图尺度下外观接近；f199/n1可见局部小缺口收拢。未观察到明显新主体破损，不等于没有任何局部形状代价。图审结果不由指标阈值生成。

[完整图库与截面](gallery/S2000_BACK_FULL/index.html) · [64条图审](visual_review/per_stage.csv) · [逐请求视觉变化](visual_review/paired.csv)

| 请求（点击看四阶段） | 最终靠背用途 | bulk可达 BASE→FINAL | 最终FF | 最终I |
| --- | --- | ---: | ---: | ---: |
| [f4f1aba6_BACK_FULL_n0](gallery/S2000_BACK_FULL/f4f1aba6_BACK_FULL_n0.png) | YES | 0.00% → 0.00% | 5.60% | 0.00% |
| [f4f1aba6_BACK_FULL_n1](gallery/S2000_BACK_FULL/f4f1aba6_BACK_FULL_n1.png) | YES | 0.00% → 0.00% | 38.60% | 17.14% |
| [6c36a5b1_BACK_FULL_n0](gallery/S2000_BACK_FULL/6c36a5b1_BACK_FULL_n0.png) | YES | 0.49% → 0.49% | 11.63% | 44.92% |
| [6c36a5b1_BACK_FULL_n1](gallery/S2000_BACK_FULL/6c36a5b1_BACK_FULL_n1.png) | YES | 0.00% → 0.00% | 47.52% | 0.00% |
| [8c9b2023_BACK_FULL_n0](gallery/S2000_BACK_FULL/8c9b2023_BACK_FULL_n0.png) | YES | 5.26% → 19.37% | 18.35% | 23.14% |
| [8c9b2023_BACK_FULL_n1](gallery/S2000_BACK_FULL/8c9b2023_BACK_FULL_n1.png) | YES | 0.00% → 9.93% | 10.06% | 21.14% |
| [b631b78c_BACK_FULL_n0](gallery/S2000_BACK_FULL/b631b78c_BACK_FULL_n0.png) | YES | 10.83% → 14.39% | 15.85% | 21.92% |
| [b631b78c_BACK_FULL_n1](gallery/S2000_BACK_FULL/b631b78c_BACK_FULL_n1.png) | YES | 0.00% → 0.00% | 8.33% | 19.34% |
| [fb8c3a83_BACK_FULL_n0](gallery/S2000_BACK_FULL/fb8c3a83_BACK_FULL_n0.png) | UNCERTAIN | 0.00% → 0.00% | 20.50% | 24.56% |
| [fb8c3a83_BACK_FULL_n1](gallery/S2000_BACK_FULL/fb8c3a83_BACK_FULL_n1.png) | UNCERTAIN | 0.00% → 0.00% | 8.04% | 54.83% |
| [abbe7279_BACK_FULL_n0](gallery/S2000_BACK_FULL/abbe7279_BACK_FULL_n0.png) | YES | 0.11% → 0.11% | 17.78% | 30.37% |
| [abbe7279_BACK_FULL_n1](gallery/S2000_BACK_FULL/abbe7279_BACK_FULL_n1.png) | YES | 2.42% → 19.93% | 36.72% | 25.98% |
| [f199965d_BACK_FULL_n0](gallery/S2000_BACK_FULL/f199965d_BACK_FULL_n0.png) | YES | 12.17% → 12.13% | 7.27% | 4.20% |
| [f199965d_BACK_FULL_n1](gallery/S2000_BACK_FULL/f199965d_BACK_FULL_n1.png) | YES | 99.66% → 99.66% | 12.04% | 17.92% |
| [9d81bc69_BACK_FULL_n0](gallery/S2000_BACK_FULL/9d81bc69_BACK_FULL_n0.png) | YES | 0.76% → 95.21% | 18.18% | 15.23% |
| [9d81bc69_BACK_FULL_n1](gallery/S2000_BACK_FULL/9d81bc69_BACK_FULL_n1.png) | YES | 0.00% → 0.19% | 24.83% | 18.16% |

## 此轮决定与限制

**保留“Chair完整靠背有可辨认用途与少量真实主体接续”的结果，不能写成各种部位都具备稳定控制。** 后处理在这个新面板仍减少非法面对、保住原连接，但最终I仍高，连接覆盖仍有限。跨部位输入值得继续研究，现有证据不足以承诺和座面强结果一样。

如果下一轮继续，我优先检验“SEAM阶段的条件部位覆盖是否限制靠背接续”：将已有合法BACK_FULL加入固定训练任务，并保留本面板与座面回归作预先固定的比较。待检验假说是：在不加大后处理位移预算的情况下，背部任务曝光能提高seam与bulk的共同几何组织，而不只是增加被结构锁定的C/F边。此轮没有开始该训练，也没有由单终点检查证明原因就是遗忘或训练不足。

没有比较早期EXP67在相同新噪声上的结果，没有统计跨所有Chair父或椅腿等部位的成功率；官方预训练暴露UNKNOWN。没有把来源GT款式作为成功标准，没有重新抽噪声、调头或挑输出。

## 文件与恢复

- 输入/噪声：`eval_manifest.json`、`inputs/`、`queries/`；manifest SHA256 `0b751c9d803adb0a31552f2ae50d3ed5597baf7a626952cb6f0c0dfccf7b9c81`。
- 所有产物：`evaluation/S2000/<request>/raw/`、`post/`、`cpu/`，原RAW/BASE保留。
- 数值：`evaluation/tables/`、`evaluation/scores/`、`key_per_output.csv`。
- 图像/审阅：`gallery/S2000_BACK_FULL/`、`visual_review/`。
- 实现：`r2/pilots/chair_back_transfer_v1/`；源码身份 `source_manifest.json`、新增代码 `code.diff`。
- 恢复入口：`python -m r2.pilots.chair_back_transfer_v1.resume` 默认只显示状态；`--execute`仅补已冻结任务的缺失阶段，未决神经调用禁止自动重放。本轮已完成，不需要重新执行。

此轮无新权重、无清理、无默认切换、无commit/push。完成固定任务后停止。
