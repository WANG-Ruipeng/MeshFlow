# R2_BENCH_LAYOUT_ORACLE_V1 实际执行账本

本账本仅从已完成调用日志、生产receipt及资源记录汇总。读取权重做流式SHA256核验，没有反序列化检查点、训练、模型调用或重评分。固定终点保留，不代表晋级。

## 科学计算与阶段预算

| 阶段 | 实际执行 | 原始记录 |
|---|---:|---|
| PLAIN训练 | 1000次更新、8000次前向、8000次反向、8000实例曝光 | training/PLAIN/status.json、calls.jsonl |
| LAYOUT训练 | 1000次更新、8000次前向、8000次反向、8000实例曝光 | training/LAYOUT/status.json、calls.jsonl |
| 新生成 | 3模式×12请求=36 RAW；Euler50共1800次逻辑向量场/生成器前向 | evaluation/*/raw_complete_full.json |
| 官方D0 | 36次单次D0，36次基础GEO_C；所有新RAW完成后才进入D0 | evaluation/*/post_complete_full.json |
| 新统一评分 | 36 RAW＋36 BASE_POST＋36 CLEAN_FINAL_POST=108条 | results/completion.json |
| 冻结参照 | EXPOSURE12请求×3阶段=36条历史评分，只读复用 | audits/reference_cache_audit.json |
| 图页 | 36页，每页4方法，共144方法阶段单元；CPU 2 worker | gallery/complete.json |

两臂正式调用账本均无失败、无pending调用；没有生产重跑、额外更新或新噪声抽样。旧EXPOSURE没有重新生成或评分。LAYOUT_NULL使用同一个LAYOUT终点，并非第三条训练臂。布局支路包含在生成器前向内，未额外调用旧占用头。

## 实际墙钟与吞吐

以下为各阶段生产计时器的实际范围，不把多进程时间求和称为用户等待时间，也不将并行训练与单卡独占测速等同。数据构造、阅读、实际AI_PROXY图审及全部收口没有统一启动/结束计时器，记为NA，不由文件时间猜算。

| 阶段 | 秒 | 口径 |
|---|---:|---|
| PLAIN训练 | 1843.141 | 生产训练阶段；1000更新；实际重叠运行 |
| PLAIN正式更新计算累计 | 1678.478 | 每次update计时之和；均值1.6785秒/更新 |
| LAYOUT训练 | 1919.891 | 生产训练阶段；1000更新；实际重叠运行 |
| LAYOUT正式更新计算累计 | 1761.565 | 每次update计时之和；均值1.7616秒/更新 |
| PLAIN1000 RAW12份 | 53.337 | 含本阶段模型准备/采样的producer计时，0.2250输出/秒 |
| LAYOUT1000 RAW12份 | 54.268 | 含本阶段模型准备/采样的producer计时，0.2211输出/秒 |
| LAYOUT_NULL RAW12份 | 53.351 | 含本阶段模型准备/采样的producer计时，0.2249输出/秒 |
| PLAIN1000 D0/GEO12份 | 33.131 | 原producer D0/GEO阶段计时；不是纯D0 kernel时间 |
| LAYOUT1000 D0/GEO12份 | 32.841 | 原producer D0/GEO阶段计时；不是纯D0 kernel时间 |
| LAYOUT_NULL D0/GEO12份 | 32.304 | 原producer D0/GEO阶段计时；不是纯D0 kernel时间 |
| CPU post LAYOUT1000+LAYOUT_NULL | 91.896 | 24请求，2 workers，包含相应评分/后端工作 |
| CPU post PLAIN1000 | 46.832 | 12请求，2 workers，包含相应评分/后端工作 |
| CPU raw PLAIN1000+LAYOUT1000+LAYOUT_NULL | 74.016 | 36请求，2 workers，包含相应评分/后端工作 |
| 36页CPU渲染 | 15.879 | 2 workers；不含后续实际图审 |

## 资源与并发的证据边界

实际WSL设备为NVIDIA GeForce RTX 5080，nvidia-smi总显存16303MiB；24逻辑CPU，初始可用RAM约61.25GiB。首臂dispatch时整卡已用3503MiB、可用12477MiB、GPU利用率0%。PLAIN正式更新之后重新观察，第二臂dispatch时已用8057MiB、可用7923MiB、利用率25%、可用RAM约59.53GiB。新增worker的保守估计4750MiB，1.2倍余量加2048MiB安全边际，需要7748MiB，满足当时7923MiB可用量。该4750MiB是派发估计，不是已测进程显存峰。

训练确实有两个独立PID（PLAIN491、LAYOUT761）重叠运行；每臂独立模型、优化器和冻结随机流。训练资源observer每2秒采样。nvidia-smi逐进程显存返回N/A，不能从整卡峰或PyTorch allocator峰推算独立进程峰。整卡观测含另一worker、CUDA/context和用户背景进程。未停止无关进程。

| 训练臂 | allocator allocated峰 MiB | allocator reserved峰 MiB | observer整卡已用峰 MiB | 进程显存峰 | observer样本数 |
|---|---:|---:|---:|---|---:|
| PLAIN | 3847.767 | 4166.000 | 12971 | NA（nvidia-smi不可得） | 884 |
| LAYOUT | 3864.728 | 4178.000 | 12960 | NA（nvidia-smi不可得） | 920 |

allocator为每个正式完整更新前重置后记录，再跨1000更新取最大；不是训练进程总VRAM。观察到的整卡最大12971MiB也不是连续采样保证捕获的硬上界。

采样三个独立模式进程按PLAIN1000→LAYOUT1000→LAYOUT_NULL增量派发，单进程一次处理一份请求；前一组达到代表性输出后才观察/派发后续组。记录有前一进程退出与后一进程启动的短暂重叠，不能称三路全程同时采样或宣称三路并发峰。D0同样逐组增量派发，每个dispatch记录workers=1；记录包含前一进程收尾和下一进程准备重叠，但没有证据支持三路D0持续并发。CPU RAW评分可与D0独立工作重叠；CPU评分/后端每批最多2 workers，每worker1线程。

| 模式 | RAW秒级observer整卡峰 MiB | RAW逐请求allocated/reserved峰 MiB | D0秒级observer整卡峰 MiB |
|---|---:|---:|---:|
| PLAIN1000 | 4560 | 586.090/632.000 | 5773 |
| LAYOUT1000 | 5515 | 593.373/646.000 | 5769 |
| LAYOUT_NULL | 5443 | 593.373/646.000 | 5776 |

采样与D0 observer均1秒间隔，覆盖本次实际任务；逐进程显存全部NA。D0完成记录未提供独立allocator峰，记NA。RAW逐请求allocator值不是含模型加载及CUDA上下文的进程峰。各observer开始/结束UTC和全部计时字段保存在output/execution_ledger_evidence.json；不可将不同observer的峰相加。

## 工程事件与保留资产

CPU测试入口曾因维护包路径未插入而导入失败；只修测试harness后通过，生产模型/数据/评分未重跑。训练收口CPU汇总曾将paired[-50]/paired[-100]误写成元素而非尾部切片，引发TypeError；analysis/training_complete_v1保留partial，新版写training_complete_v2。该修正新增生产前向、反向、更新、评分均为0，不改冻结生产代码。事件详见engineering_events两份JSON。数据准备还修正过nullable review字段，均在冻结前且零模型调用。

CPU reduced-width模型合同测试另列：最终15项测试计144次微型native block调用、14次layout模块调用、3次backward、2次输入VJP、2次微型AdamW更新；不是正式生产计算。此前14/15/15项CPU开发测试未逐项计数，明确NA；所有均为缩小CPU fixture，无生产检查点、无CUDA。见audits/model_cpu_gate.json。

artifacts/retained_weights.json列出两臂各一个full恢复终点及一个推理导出，另列只读EXPOSURE full/export来源，实际读取文件size和流式SHA256核验通过；没有删除、覆盖任何权重。checkpoint_latest仍保留，不作为额外科学终点。

本轮新增四个终点/导出文件总计 4175261458 bytes。源码、原数据、原模型、旧RAW/POST与评分不动；未切默认、未commit/push。

## 核验来源

- audits/training_completion.json 与 training/*/status.json、calls.jsonl：正式训练计数及资源。
- dispatch/*_go.json：逐臂派发前实测与headroom。
- evaluation/*/raw_complete_full.json、post_complete_full.json：36 RAW/36 D0。
- evaluation/completions/*.json、results/completion.json：CPU实际批次、108新评分/36旧评分。
- evaluation/*/sampling_attempt*/resources.json、D0_resource_observer*/resource_observation.json：整卡资源监测。
- gallery/complete.json：渲染预算；visual_review/completed/*.json：单独实际图审记录。
- output/execution_ledger_evidence.json：本账本依赖的完成记录SHA256与细节。
