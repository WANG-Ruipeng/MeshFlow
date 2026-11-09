# MF-H1-LOCAL0：Phase 0–2 预检与结果报告

日期：2026-09-28。运行目录：`E:\MeshFlow\experiments\mf_h1_local0\outputs\run_20260928_01`。

**结论：实现接入通过；条件响应尚未与数值波动分离；架构优势未建立。** 标签为 `IMPLEMENTATION_PASS_ONLY` / `NEEDS_LEARNING_DIAGNOSIS`。本次已完成 Q/G 各 200 次正式更新，总计 400 次，seed 10。没有启动 Phase 3，没有扩大数据或训练，没有改损失或增添机制模块。资源和资产已齐，但当前证据不支持直接扩大机制实验；建议另行授权数值重复性和学习诊断后再决定。

本报告同时保留一个未通过的原始数值对照：官方 FP32 linspace 时间网格与任务指定的固定 dt=1/50 循环，最终坐标不满足原定误差界。后续使用相同浮点网格和官方整数时钟重参数参考均逐位一致，证明积分接线，但不抹去原始失败，也不证明 BF16 条件生成可重复。详见下文。

**资产、版本与实际环境**

| 项目 | 已核验结果 |
|---|---|
| 仓库 | `E:\MeshFlow`；HEAD `8cc74da013d3d2feec1853cccbb1f9ac37814b95` |
| 来源 | origin 为 WANG-Ruipeng fork，upstream 为 qiisun/MeshFlow；与任务冻结提交相同 |
| 已有规则 | 检查了仓库及适用父目录，没有适用 AGENTS.md；其他工作树的 AGENTS.md 不适用于本目录 |
| 官方权重 | `C:\Users\warpwang\Downloads\last.pt`，2,084,287,208 bytes，完全本地读取，上传 0 bytes |
| 官方身份 | HF `qsun2001/meshflow` dataset 的 `v1/120m-ot-v-chair/checkpoints/last.pt`；大小与官方 LFS SHA256 一致 |
| 数据源 | `C:\Users\warpwang\Downloads\shapenet-cls.tar.gz`，25,483,419 bytes |
| 实际 chair 数据 | `experiments/mf_h1_local0/assets/shapenet-03001627`；3,135 个真实 mesh，另有 2 个 split 文件 |
| 作者划分 | train 2,822，test 313，UID 重叠 0、缺失 mesh 0 |
| GPU | RTX 5080，16,303 MiB；检测时约 13,605 MiB 空闲，非本任务独占 |
| CPU/RAM/磁盘 | WSL 24 logical CPUs，RAM 67,125,248,000 bytes，E 盘检测时可用 2,205,159,895,040 bytes |
| 环境 | WSL Ubuntu，Python 3.10.21，PyTorch 2.7.1+cu128，NumPy 1.26.4，Trimesh 4.10.0 |
| 驱动 | Windows 591.86（WSL nvidia-smi 工具显示 590.57），未修改 |
| 精度与注意力 | 小型 BF16 反传探针通过；正式 Q/G 均 BF16 autocast、FP32 参数及坐标、SDPA math，无 compile |

权重 SHA256：`bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d`。

数据压缩包 SHA256：`06707caff4ce465f402e6b69f81056204c579921d371eb2c9e4e9a3929f7aef2`。

安全读取全部 tar 成员并仅提取 chair 到新目录，未修改原压缩包、其他下载数据或 checkpoint。其他三个压缩包不是本次的必需输入；已有 bench checkpoint、ss_overfit 及旧合成代理均未使用。本任务没有缺失的必要下载资产。

加载使用 `torch.load(map_location='cpu', weights_only=True, mmap=True)`；选 `ema`，182 个张量，130,249,731 个参数。`strict=True` 的 missing/unexpected keys 均为空。model/data/transport/sample 关键配置匹配 chair 参考；checkpoint 历史训练配置并非全部相同：max_steps 2,000,000 对参考 1,000,000，batch 72 对 32，lr 2e-5 对 1e-4，另有 resume/keep_last_ckpts。实际 adapter 使用任务规定 lr 1e-4。meta 初始化后的未注册 Fourier 频率闭包常量按官方公式在 CPU 重建；学习参数全部严格来自 EMA。

主干状态 SHA256：`b32197096ecf8d662bfee18bbc4abf16cc4b4690cd28e5c47d14850729d2c222`，官方探针、短训、生成、计时后不变。结束时再次核验原 last.pt SHA 与 tracked Git 文件，均未改变。首次 WSL Git 把 CRLF 误判为修改，Windows Git 和临时 `-c core.autocrlf=true` 复核已排除；未修改 Git 配置。原始审计记录及更正同时保留。

证据：[资产清单](../asset_audit_20260928_02/asset_manifest.json)、[环境](../asset_audit_20260928_02/environment.json)、[Git 更正](../asset_audit_20260928_02/git_cross_platform_verification.json)、[最终源与权重核验](final_source_verification.json)。

**实现及数据路径**

所有新增代码在 `experiments/mf_h1_local0/`，没有修改官方主干、数据加载器或 transport 源码。wrapper 每次只调用一次原 forward，在第 3/6/9/12 层后加残差；主干 eval 且参数冻结，保留对早层 adapter 的反向传播。没有 GT 终点输入接口、额外主干终点预测、直接网格投影或修补。

| 分支 | 消息/相对几何 | 可训练参数 |
|---|---|---:|
| S_static | 分组点均值 token，静态 value | 950,144 |
| Q_center | 当前 Xt 面角减去原语质心 | 1,178,496 |
| G_closest | 当前 Xt 面角减去有限线段/三角片最近点 | 1,178,496 |
| P_point | 点特征与分组均值，当前 Xt 面角减采样点，logit 减 log(m) | 1,178,496 |

Q/G/P state keys 和参数形状完全相同、共享初始状态；Q/G 唯一机制差异为相对向量。点编码 41→128→128，动态消息 316→128→128，4 heads。只把输出投影置零，其余正常初始化；无额外 gate、null token、几何 logit bias 或虚设参数。finite triangle 退化时采用与顶点顺序无关的确定处理，不用有向法线。Q/G/P 的几何关系都随当前 Xt 更新。

正式数据复用官方 bbox 归一化、噪声、面排列及 nested OT；原 target 在 OT 后乘 2，与原训练路径相同。原语坐标来自该最终 target 坐标系，不逐原语重新归一化。训练 target 仅用于原 velocity FM MSE，GT 引导原语的构造不等于把完整终点喂给 adapter。

16 个训练父对象及另外 4 个诊断父对象在生成前固定，均来自作者 train split，20 个 UID/无序几何 hash 互异；诊断对象未用于本轮 adapter 训练。这里只能验证作者 UID 身份和精确几何去重，数据没有更强的祖先关系元数据；不能声称这些诊断对象未被官方预训练见过。完整 mesh 用于正式训练，未截取前 16 面；只有早期官方接口单测使用少量真实面。

每个诊断父对象固定单边、单面、混合两边一面三类提示，共 12 组。每组总计 32 个点，混合为 8/8/16，包含原顶点。每个原语另采 128 个独立检查点，表面一致性检查通过。对象、归一化逆变换、提示顶点、采样点、检查点、噪声种子、输入及每个 microbatch 的 hash 均保留在 [split_manifest.json](split_manifest.json)、[prompt_manifest.json](prompt_manifest.json)、`fixed_inputs/` 和训练日志中。

**测试与门禁**

| 项目 | 实际结果与边界 |
|---|---|
| 独立 CPU 数学/adapter 单测 | attempt01 191/191；最终 attempt02 201/201，均保留 |
| 严格官方 EMA 加载 | PASS，0 missing / 0 unexpected |
| 控制关闭及零初始化 | FP32 原始主干对照通过；关闭控制完整轨迹逐位一致 |
| 梯度 | 输出零初始化行为、非零分支、早层 adapter 梯度通过；主干无参数更新 |
| 主干冻结 | 前后逐张量状态 hash 一致 |
| 单 token 性质 | S 静态单 value 性质与 Q/G 查询相关消息性质通过；不把该测试当生成有效证据 |
| 独立有限几何 oracle | 1,000 组 FP64 随机案例，最大绝对误差约 1.55e-15，阈值 1e-10 |
| Q/G 唯一相对向量差异 | 捕获真实动态 MLP 输入，独立参考差约 4.94e-6，低于 1e-5 |
| 非零分支置换/翻转/padding | 点序、原语序、顶点顺序、混批 padding 均通过；CPU 代表最大误差约 8.2e-8 |
| 官方 FP32 网络测试 | SDPA math、禁 TF32、确定性算法；全部通过 |
| 保存恢复 | 官方 Q 的连续 10 步与 5+恢复+5 参数/优化器/RNG 一致；独立 CPU 测试覆盖 Q/G 恢复，均为一次性探针 |
| 独立评价数学测试 | 26 项通过，CPU Trimesh 距离不复用训练几何 helper |
| BF16 完整 mesh 资源探针 | 有限 loss/反传与资源检查通过；不等于 BF16 全轨迹确定性保证 |
| 原 linspace 对照 | **FAIL，保留**：最大终点差 0.1146656573，大于预设 1e-4 |
| 相同浮点时间网格复核 | PASS：每步输入、velocity、终点与未改官方 transport 逐位一致 |
| 官方整数时钟重参数参考 | PASS：s=50t，官方积分 s=0..50、51 节点，返回 v/50；与任务固定 dt 循环逐位一致 |

原 linspace 与 `t=k/50, dt=1/50` 有 9 个 FP32 时间节点不同（最大 5.96e-8），46 个 dt 不同（最大 4.10e-8），多步高频位置编码会放大差别。因此原始“默认浮点网格端到端相等”门禁没有变成 PASS；后续诊断证明的是相同数值定义下实现一致。Phase 2 在这些诊断完成后才开始，生产采样仍保持任务固定 dt，没有改成另一采样规则或放宽阈值。该数值敏感性是本次结果的实质限制。

日志：[CPU 最终测试](../../tests/results/cpu_units_attempt02.json)、[官方测试](official_tests/official_unit_tests.json)、[原始采样失败](sampling_precheck.json)、[匹配网格](sampler_diagnosis.json)、[整数时钟参考](sampler_integer_clock_reference.json)、[错误及限制记录](failures.jsonl)。此前广播维度错误在官方测试前修复；失败记录没有删除。

**400 步短训和完整生成**

AdamW，lr=1e-4，betas=(0.9,0.95)，weight_decay=0，clip=1；microbatch=2、累积=4、有效 batch=8。每次更新以全部 8 个样本的有效坐标数归一化原 velocity MSE。无 scheduler、adapter EMA、额外 loss。Q/G 均从新的 seed10 共同初始状态开始，不使用单测、旧 adapter 或 Phase 3 状态。100/200 步 adapter、optimizer、RNG checkpoint 全部保存。

共同初始状态 hash：`a3fce00eadde61f5480d4cefbcf06689a96ec1b04ddad3c4d2676adca6b34079`。
共同训练输入 rolling hash：`9330c9cce5097fff9a30c2ef44cf9da7569e77ecedd78766218e9d74380d9d31`，Q/G 相等。

| 项目 | Q_center | G_closest |
|---|---:|---:|
| 正式 optimizer 更新 | 200 | 200 |
| 训练 microbatch 前向 | 800 | 800 |
| 固定 teacher 诊断前向 | 28 | 28 |
| 训练过程用时 | 232.21 s | 290.85 s |
| 固定 teacher MSE 训练前 | 0.00104900718 | 0.00104900718 |
| 固定 teacher MSE 训练后 | 0.00105202251 | 0.00104995290 |

固定 teacher MSE 没有清晰下降。不同随机批次的训练损失也不能作为生成改善证明。正式运行 CPU 峰值 RSS 2,336,526,336 bytes；所记录 CUDA allocated 峰值 2,237,264,896 bytes 为进程累计峰值，不能解释成隔离测得的逐臂资源差。

每次生成都是 512 raw face slots、50 次显式 Euler 更新、51 个网格点、face/geometry CFG=1，独立 Gaussian 初噪声，与 GT 无 OT 配对。固定条件编码公平缓存，Xt 关系每步更新。保存未经量化、合并、投影或修复的 `.npz` 和 `.obj`。

主要评价实际完成 96 次条件采样：Q/G × 4 父对象 × 3 提示 × 2 噪声 × 正确/循环错配。循环错配会复用另一个父对象的正确提示，因此只有 48 个不同的臂/提示/噪声输入，不是 96 个独立条件。另实际采样 2 次无几何条件 base，在 24 个提示评分中复用，不冒充 24 次独立 base 生成。统计单位只有 4 个诊断父对象、1 个 adapter 训练 seed；没有 Phase 3 的多 seed/2000 次 bootstrap 推断。

**生成误差、质量与条件响应**

下表为正确条件的原语等权 point-to-triangle RMS，以目标 bbox 对角线百分比表示，越小越好；先对噪声取均值，再按父对象与提示类型等权汇总。检查点独立于条件采样点；不是同点最近邻训练指标。

| 提示 | 无几何条件 base | Q_center | G_closest |
|---|---:|---:|---:|
| 单边 | 9.554175% | 9.607713% | 9.533667% |
| 单面 | 3.298219% | 3.341748% | 3.326462% |
| 两边一面 | 5.064053% | 5.067936% | 5.069072% |
| 三类等权总体 | 5.972149% | 6.005799% | 5.976400% |

总体 G−Q = −0.029399 个百分点，相对变化 −0.4895%；远小于任务 Phase 3 的 10% 目标约定，也没有超过 base。这里不作显著性、优越性或等效性推断。G 对 P 的强点基线比较 **NOT_RUN**；S/P 已实现并经 CPU 测试，本次没有训练，不能推导其生成效果。

唯一实际输出的最小角 <5° 面比例：base 21.1914%，Q 22.0378%，G 21.8994%。记录中非有限坐标和按实现阈值判定的退化面均为 0。其余分位数、coverage、max、completion 诊断见逐输出记录。自交、水密、流形检查和后处理均 **NOT_RUN**，不以零非有限值或图片替代拓扑结论。未引导区双噪声 surface RMS（Q 4.554%、G 4.619% bbox）只表示几何变化，不证明合理多样性。

**不能确认条件响应的关键证据：** 48 对相同提示、相同噪声的重复运行全部不是逐位一致，最大坐标差约 2.61885。Q 改条件后的坐标 RMS 均值 0.069780，同输入重复为 0.066570；G 分别 0.072395 和 0.066075。重复生成引起的 guide 分数绝对差均值 Q 0.032407、G 0.054703 个百分点，与总体 G−Q 差同量级。正确/错配坐标变化与两个端点重复 RMS 之和的比值中位数为 Q 0.4703、G 0.5496；只有 Q 1/24、G 3/24 超过该经验量。这不是正式噪声置信界，也不是因果检验。

专门计时中的同输入重复采样也观察到变化。可能涉及 BF16/CUDA 算子及轨迹数值敏感性，但具体算子未定位，不能写成已证实原因。故 `condition_response = RESPONSE_NOT_SEPARATED_FROM_NUMERICAL_VARIATION`，`causal_response = NOT_ESTABLISHED`。单次改提示产生非零输出差不算条件控制已成立。

目视对照图中的 Q/G 整体仍接近无条件 base，多处黄色引导未得到明显跟随；这只与定量限制一致，不构成独立效果认证。已保存全部 12 张最终联系表，统一每行坐标范围和视角，不裁掉离群几何；早期标签布局修订图也保留。[混合提示，seed4101](figures/phase2_contacts_v3/mixed_2edges_1face_seed4101_target_base_correct.png)、[错配对照](figures/phase2_contacts_v3/mixed_2edges_1face_seed4101_correct_vs_mismatch.png)、[全部最终图](figures/phase2_contacts_v3/)。

证据：[逐输出 CSV](per_output.csv)、[原始 NPZ/OBJ](raw/)、[配对汇总](paired_summary.json)、[条件与重复性分析](condition_response.json)。CSV 120 个评分行包括复用 base 的 24 个评分；不等于 120 次独立采样。

**实际效率**

专门计时使用同一个提示、同一个噪声、同一精度/后端、512 面、随机交替顺序。组件各预热 3 次、计时 10 次；完整采样各预热 1 次、计时 3 次。全采样前移除计时 hooks。

| 中位耗时 | Q_center | G_closest |
|---|---:|---:|
| 条件编码 | 0.940 ms | 0.918 ms |
| 所有面角的几何关系计算一次 | 0.076 ms | 9.137 ms |
| 主干扣除 adapter 的估算 | 20.038 ms | 20.000 ms |
| 完整前向含编码 | 37.252 ms | 85.190 ms |
| 50 步完整生成 | 2.502 s | 5.031 s |

G 在这一配置约为 Q 的 2 倍采样时间，仅三个计时样本，不作普遍或显著速度结论。主干时间为嵌套 CUDA events 相减的估算，含调度开销。整次计时 GPU allocated 峰值 588.96 MiB、CPU RSS 1.772 GiB，两臂共享过程峰值。该测量额外 426 次主干前向、8 次完整采样、0 次更新，全量计入下面账本，原始计时输出单独保存。[组件及完整采样计时](components_benchmark.json)。

**完整执行账本**

| 阶段 | 官方主干前向调用 | 完整生成 | 正式更新 | 一次性官方单测更新 |
|---|---:|---:|---:|---:|
| 官方严格接线/梯度/恢复测试 | 45 | 0 | 0 | 30 |
| 初始 base/关闭控制/采样对照 | 301 | 6 | 0 | 0 |
| 相同浮点网格诊断 | 100 | 2 | 0 | 0 |
| 整数时钟官方参考 | 50 | 1 | 0 | 0 |
| Phase 2 训练与主要生成 | 6,557 | 98 | 400 | 0 |
| 专门效率测量 | 426 | 8 | 0 | 0 |
| 合计 | **7,479** | **115** | **400** | **30** |

7,479 次是函数调用次数，包含 batch 前向，不能当成样本数。wrapper 6,977 次、直接官方 forward 502 次。115 次完整采样合计 NFE=5,750。Phase 2 的 6,557 = 1,600 训练 microbatch +56 固定 teacher +1 BF16 完整 mesh 探针 +4,900 生成。首次基线阶段为 1 次单前向预热 +6×50 次生成前向=301；六次生成是四个 base、一次 wrapper-off、一次原 reference。

另两次独立 CPU 单测运行各 60 次小模型 optimizer 探针，共 120 次，无官方主干前向；CPU adapter layer 调用 704、encoder 319。加上官方探针 30 次，共有 150 次一次性测试更新，均未作为正式训练初始化。全部 recorded optimizer 调用为 550 = 400 正式 +150 探针；正式 400 步预算没有扩张。主干参数更新 0，Phase 3 更新 0。文件解析、评价和渲染未增加主干前向。[机器可读计数](execution_counts.json)。

**交付和停止边界**

代码入口、复现命令及范围在 [README](../../README.md)。本目录交付 `code.patch`、本报告、`resolved_protocol.yaml`、资产/测试 JSON、失败原记录、逐步训练 JSONL、100/200 步 checkpoint、固定输入、原始完整生成及最终图片。补丁只新增本实验目录文件；在当前已存在代码的工作区只做反向适用性检查，没有应用反向修改。便携压缩包另外附内容 SHA256 manifest；不包含 2 GB 官方权重或原始数据资产，其路径和 hash 在报告中。

已回答本轮问题：官方 EMA 的身份和加载正确；零初始化/关闭控制/梯度/冻结/几何接线通过；完整生成确已执行；但条件响应未可靠分离，G 的有限原语距离没有显示可信优势，且本实现带来可见时间开销。因此本轮停在 Phase 2。若未来继续，应先获得重复性与学习效果的可信证据，再评估是否投入 Phase 3；本次没有实施该后续工作。
