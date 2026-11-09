# 当前工作区：Chair HYBRID START

维护入口为 `native_t1/`。2026-10-05 按用户明确要求退回并正式接入 START：原 OT_HYBRID pilot1000 / conditional_total5000，Native T1 + C-only Geo + FM，训练 coupling lambda=0.25。默认采样 profile 现为 `chair-hybrid`；MOMENT V2 及其 finite-surface 前序活动包已封存并移除。

## 当前使用方式与边界

```bash
python -B -m native_t1 sample --condition /path/to/C.npy --num-faces 163 --seed 2452526625 --out native_t1/runs/chair_start_sample
```

总 N 必须显式指定 128..256；C 为原坐标尺度的有限 FP32 数据，0<K<N。使用原 MT19937 free-face 噪声、50 步 Euler、BF16 forward / FP32 参数与积分，已知 C 在 51 个状态均精确保留。详见 [START 说明](native_t1/docs/chair_start.md)。

正式本地权重为 `native_t1/checkpoints/chair_hybrid_start_total5000.pt`，是原 `chair_hybrid_coupling_v1/checkpoints/OT_HYBRID_pilot1000_total5000.pt` 的硬链接。文件 SHA256：`d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80`。原完整 AdamW、RNG 和输入流状态保留；推理不恢复优化器。Git 只纳入正式源码、测试、文档及身份 manifest，权重继续本地保存。

旧 `--profile fm-geo`、`edge5` 等保持显式可用的 N112 路径。`train` / `train-edge5` 仍是显式预算、fresh AdamW 的旧 N112 训练入口，不是 START 续训入口。任何新训练需要独立的数据、coupling 和预算合同，不因改变默认模型自动开始。

## 保留的正式和历史权重

|角色|路径|累计步数|
|---|---|---:|
|START 默认|native_t1/checkpoints/chair_hybrid_start_total5000.pt|5000|
|旧 FM 显式 profile|native_t1/checkpoints/ablations/fm_cumulative3000.pt|3000|
|JEdge5 显式可选|native_t1/checkpoints/jedge5_cumulative3000.pt|3000|
|旧共同起点 G0|native_t1/checkpoints/tgeo_g0_cumulative2500.pt|2500|
|surface 旧参照|native_t1/checkpoints/ablations/surface_cumulative3000.pt|3000|
|Edge-only 旧参照|native_t1/checkpoints/ablations/edge_only_cumulative3000.pt|3000|

## 2026-10-05 MOMENT 封存完成

删除 MOMENT/H_CONT 正式 V2 权重各 5 份、8000 个派生 OT NPZ 和三个退役活动源码包，共 8061 文件、15,677,075,950 bytes（约 15.68 GB 逻辑字节）。源码先行归档；原始结果、报告、逐例 CSV、日志、评分源码快照、决策 ZIP 和小型合成 CPU 测试 fixture 保留。

11,488 个保留文件前后 SHA256 一致。START CPU strict load 通过，公共 CPU 测试 109/109 通过，删除后 CLI 导入通过；真实模型前向、训练更新和生成均 0。根据本轮用户明确授权完成本地 Git 提交，没有 push。详见 [清理报告](native_t1/maintenance/moment_retirement_20261005/CLEANUP_REPORT.md) 和其中的账本。

后续仍遵守 [AGENTS.md](AGENTS.md) 的资源调度与保护边界。以下是按日期保留的历史记录；其中旧的默认模型、HEAD、实验状态及已删除路径以其记录日期为准。

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

## 2026-10-02 正式多对象实验前的数据盘点

本地 `experiments/mf_h1_local0/assets/shapenet-03001627` 实际已有 3135 个官方 UID、3125 组精确几何；官方 train/test 为 2822/313，未发现 UID 或精确几何跨划分重复。近重复与同源家族尚未排查。当前两对象 N112 是维护训练入口的限制，不代表本地只有两个对象。正式实验应区分条件微调未见与官方预训练未见，不直接把旧双对象模型作为公平的纯 T1 / Geo 比较起点。

面数为 28–800，中位 470；原始索引拓扑/退化面诊断及逐对象记录见 [数据扫描](native_t1/maintenance/temp_cleanup_20261002_160601/dataset_scan.json)。先准备现有椅子数据即可推进；跨类别可补下载 MeshFlow 的 `shapenet-cls.tar.gz`（25.48 MB，历史归档清单确认包含 bench/chair/lamp/table），语义部件控制可申请 PartNet。具体链接、体积与适用边界见 [数据下载清单](native_t1/maintenance/temp_cleanup_20261002_160601/DATASET_DOWNLOADS.md)。本轮未下载数据、未启动训练或采样。

历史 checkpoint 已按用户明确批准的固定清单完成清理：删除 27 份退役权重及 30 个临时文件，共 35,971,879,861 bytes（约 35.97 GB）。official、无 Geo T1 500/1000/1500、canonical 五份及关键 Base/Geo 和 EMA 终点继续保留。3,711 个关键文件 SHA256 校验一致，原盘点中清单外缺失为 0；实验报告、指标和原始采样证据均保留。已删历史终点不能直接重新采样，相关分支若重新研究需重训；历史报告内旧 checkpoint 路径以 [清理报告](native_t1/maintenance/temp_cleanup_20261002_160601/CLEANUP_REPORT.md) 的删除记录为准。
## 2026-10-02 已接收 ShapeNet class split 归档

用户下载的 `shapenet-cls.tar.gz` 已从 Downloads 移至 `downloaded_data/archives/shapenet-cls.tar.gz`（25,483,419 bytes）；SHA256 `06707caff4ce465f402e6b69f81056204c579921d371eb2c9e4e9a3929f7aef2` 与官方元数据一致。来源及校验记录在同目录 `shapenet-cls.manifest.json`。原始归档已保留，尚未解压；既有 chair 数据不变，后续可从归档提取缺失的 bench/lamp/table 类别。
## 2026-10-04 Chair 第一档旧恢复点清理

经用户批准，删除 chair_domain_control_v1 的 T1/Geo/Full_latest.pt（phase2800、total3800），以及 control_mvp48_budget_single_v1 的 T1/Geo/Full_latest.pt（total4800）和 T1_fit2_probe_latest.pt（probe800），共 7 份、10,950,114,685 bytes（约10.95 GB）。只执行已评估的第一档；第二、第三档未删除。

全部固定中间评价状态和终点继续保留。下一步 Hybrid Coupling Pilot 的指定起点仍为 native_t1/runs/chair_domain_control_v1/checkpoints/Geo_phase3000_total4000.pt，模型与完整优化器状态未改变，文件 SHA256 为 190a16e2aec96a1f84f63460b5f97b00e23499338102f82e7530218ecb185ebf。

10,773 个受保护文件前后 SHA256 一致，清单外意外缺失为0；原数据、冻结C/噪声/GT查询点、raw、源码、报告和归因包保留。旧清单与同名JSON保留为历史证据，其中上述七个PT路径已明确退役；不能再用它们恢复旧步数。详见[清理报告与账本](native_t1/maintenance/weight_cleanup_stage_a_20261004/CLEANUP_REPORT.md)。本轮没有模型加载、前向、训练或生成，没有启动下一步试验。


## 2026-10-04 Chair Hybrid Coupling 有限试验完成

`native_t1/runs/chair_hybrid_coupling_v1/` 已实际完成同源 Geo4000 两臂各1000更新、固定A/B/C与响应诊断。两新终点为本实验 pilot1000/total5000，身份见 checkpoint_manifest.json；不要与历史预算实验5000步混称。未切换FM默认模型或JEdge5可选模型。

实际总计2000更新、432登记请求（18原始NA）、292完整采样、122历史raw缓存、2400 BF16交叉诊断、72 FP32诊断及10预检前向；总前向33082，模型失败/未决0。866个保护依赖前后hash一致；冻结新实现未改变，训练曝光、主指标算术及三个检查点轨迹独立审计PASS。

结果为 GEOMETRY_SIGNAL_WITH_TRADEOFF：HYBRID相对BASE验证A20固定查询MSE下降约40%，6/8父改善，覆盖与p95改善；但小角面增加、跨噪声控制分歧、40%上下文部分指标及已观察留出退化。优先下一步是现有raw的小角面收益归因，检验误差收益是否主要由新增细长面贡献；该后续未执行，没有追加训练、loss或求解器扫描。

详见本轮 CHAIR_HYBRID_COUPLING_REPORT.md 与 DELIVERY_INDEX.md。纯CPU审计读取器/报告文案修正单独保留原版、diff和哈希；正式实验没有静默重跑。当前固定任务已停止。


## 2026-10-04 Chair Finite Surface Attention 门禁停止

`native_t1/runs/chair_surface_attention_v1/` 已完成资产登记、独立实现、CPU门禁及受限真实预检，结论为 IMPLEMENTATION_BLOCKED。共同起点固定为上一轮 OT_HYBRID pilot1000/conditional_total5000（SHA256 d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80）。没有新增正式训练权重，没有切换默认模型。

192点求积在4个真实训练任务均未通过0.01精度门禁，7504个普通query/known对中1459个失败；误差集中在t=0.9/1。96个密集参考对子集中另有1个未达到0.001收敛要求，故保留求积规则差异的表述，不声称真积分误差界。

实际真实预检仅10前向、1次零更新反向；新增显存日志器重复传入time_unix/pid导致停止，模型调用失败0。没有修正后重跑，真实三臂成本比较与MOMENT/SURFACE接线检查均未完成。H_CONT参数、AdamW与RNG恢复通过。三臂正式更新均0，所有新质量比较NA。登记576请求，START只读复用138个完整raw，零新采样；隔离CPU小模块1次测试更新另记。

固定任务已停止，不应自动恢复训练或把控制器DONE解释为真实预检通过。优先后续假说是晚期窄核相对长三角边的n8子网格欠分辨，先在同一冻结失败输入上建立误差受控参考；本轮未执行该后续。日志接口缺陷是未来重新审计的工程前置修正。

详见本轮 CHAIR_FINITE_SURFACE_ATTENTION_REPORT.md、DELIVERY_INDEX.md 和 RUN_AND_RESUME.md。原失败源码、自动报告、完整CSV、实际评分源码与hash均保留，最终呈现修正位于postrun_review，未改旧结果。


## 2026-10-04 Chair Finite Surface Attention 明确授权修复续跑完成

用户明确要求继续下一步、不要因实现缺失停止后，在独立 `chair_surface_attention_v1r1` 代码与结果目录修复了显存日志保留字段重复问题。CPU回归、真实只读nvidia-smi及实际三臂遥测均通过，旧v1所有文件未改。

本次实际新增76前向/20次零更新反向，与旧轮累计86/21，仍在原96/24硬上限内。三臂全部完成预定检查，无调用/遥测失败、无未决调用；原参数、AdamW与RNG恢复。H/S各任务2次成本测量，M每任务1次限制已登记；未用余量不授权追加任务。MOMENT/SURFACE真实FP32及BF16零初始化均与H输出逐位一致。

独立CPU自适应积分完成固定8例，34次积分、1472625核点、24次旧求积CPU调用，无重试。两档精度自洽通过；192点最坏均值差约0.111808，确认晚期窄核欠分辨。12288点参考在部分难例也有偏差；49152点在本8例低于0.001，但不是全任务或严格认证。

真实成本已完成：SURFACE/H完整forward在普通40%/最大40%为3.05084/3.13547，略超冻结3倍门槛；F+B四任务均低于3倍。严格置换容差H本身已失败，M/S在最大N另有padding数值差异，全部保留、未放宽阈值。实现已实际跑通，当前阻塞由数值精度、成本及数值不变性合同构成。

正式训练和新采样仍为0，没有改变默认模型。优先后续是保持数学定义、在冻结难例上验证能解析边/角窄核的有限三角积分实现；该新生产算法与训练本次未启动。修订固定任务已完成，详见 `native_t1/runs/chair_surface_attention_v1r1/CHAIR_FINITE_SURFACE_ATTENTION_R1_REPORT.md` 及 `DELIVERY_INDEX.md`。

## 2026-10-05 Chair Moment Channel Learning V2 实际训练对照完成

`native_t1/runs/chair_moment_channel_v2/` 已实际完成从同一 HYBRID total5000 起点恢复的 H_CONT/MOMENT_XY 两臂各1000更新，终点均 moment1000/total6000。20个冻结新实现文件和2464个保护依赖hash不变；旧AdamW延续、新通道参数与状态单列。未切换FM默认模型或JEdge5可选模型，未覆盖旧START。

实际16000训练forward/backward、2000 optimizer更新、120预检forward/44零更新backward、276新终点Euler50与2初始回放、138 START raw缓存、18原NA；采样NFE13900，总forward30020，模型失败/未决0。两臂8000训练样本与1000批次逐项配对，完整检查点/raw/账本独立审计PASS。孤立CPU模块仅1次测试更新另记。padding压缩修复在四真实fixture的变长padding逐层与输出均bitwise通过；原严格逐元素置换FAIL仍保留，新归一化FP32合同通过，BF16非零置换仅作披露。

质量结论 NO_CLEAR_LEARNING_SIGNAL：验证A20主MSE START/H/M分别0.0015310739/0.0015961439/0.0016433661，M比H高2.96%、仅3/8父改善，比START高7.33%。验证正确条件列H/M均9/24、三目标全对均0/8；通吃稍减与平均斜率误差稍降未形成跨噪声一致控制收益。M训练循环约43.8分钟、H约25.3分钟，采样均值4.6352秒/2.7163秒；不宣称已证明可靠内核加速。

暂留简单HYBRID/START，不扩大本通道训练。唯一后续建议是用既有raw与冻结查询定位父c1b8c2f0c4f3cc43aff7c08b060f5ed6的noise1两臂共同退化区域，检验是否局部缺面/错位而非整体宽度偏差；该后续本轮未启动。科学协议偏离0；训练前恢复状态修复、1次独立审计元数据工程错误修复、报告采样计时字段与编码修订均保留原件/diff。固定任务已停止。详见本轮 `CHAIR_MOMENT_CHANNEL_V2_REPORT.md`、`DELIVERY_INDEX.md` 和 `delivery_hashes.json`。
