# 历史 checkpoint 清理报告（已完成）

状态：用户明确批准具体清单后，已严格完成全部 57 项删除，未扩大范围。删除后校验 PASS：3,711 个保留文件 SHA256 全部不变；原盘点中没有清单外文件缺失。未启动训练或采样。

已删除 27 份历史 checkpoint，35,971,606,153 bytes（35.972 GB / 33.501 GiB）；同时删除 30 个可再生成缓存或已验证的重复临时文件，273,708 bytes。总计 57 个文件，35,971,879,861 bytes。逐文件删除，无目录递归删除、无权重改写、无文件系统压缩。

## 保留

- 官方权重、原始数据、所有 worktree 和既有精简档案。
- native_t1/checkpoints 的 G0、FM、JEdge5、surface 和 edge-only 五份权重。
- 无 Geo 原生 T1 step500、pure A1000、pure A1500；前后两个是维护代码固定读取的路径，A1500也是引入Geo前的共同起点。
- 旧 Base2000、Geo2000，以及匹配重放实验的两个 EMA3000 独有终点。
- 所有实验报告、指标、源代码、日志、原始采样、固定图、协议和最新 Equal-NFE 整个运行目录。

## 删除后果与判据

早期 K1/K2、独立条件 T2、FC/FF、边界注入和 Graph 分支的历史权重已退役。其已保存结果仍可分析，但无法直接用已删权重重新采样；若将来重新研究这些分支，需要再训练。保留原始哈希不等于保留可恢复的权重备份。

Geo_continue2500 已经在 CPU 上确认模型状态及输入 RNG 与 canonical G0 一致；旧文件额外携带 optimizer，而维护续训使用 fresh AdamW。重放 raw3000 的 B/J 模型状态与 canonical FM/JEdge5 一致，封装文件本身不是字节级副本。EMA3000 不相同，因此保留。

## 已删除的 27 份 checkpoint

|序号|相对路径|GB|理由|
|---:|---|---:|---|
|1|`experiments/mf_h1_local0/outputs/a_then_fc_20260930_001700/A_then_FC/additional_step500_cumulative_step1500.pt`|1.563|Retired late FC-only exploration, outside next M1/M2/M3 recipes. Pure-FM sibling A_continue1500 retained at its original path.|
|2|`experiments/mf_h1_local0/outputs/native_inpainting_20260928_233911/Native_correct/endpoint_step500.pt`|1.563|Retired early K1/K2 two-object Native experiment. Connected-patch pure T1 step500 and later pure T1 endpoints are retained; next formal models share official or pre-Geo initialization. Keep this run reports, source, raw outputs and metrics.|
|3|`experiments/mf_h1_local0/outputs/native_inpainting_20260928_233911/Native_independent/endpoint_step500.pt`|1.563|Retired early K1/K2 two-object Native experiment. Connected-patch pure T1 step500 and later pure T1 endpoints are retained; next formal models share official or pre-Geo initialization. Keep this run reports, source, raw outputs and metrics.|
|4|`experiments/mf_h1_local0/outputs/native_patch_20260929_010000/Native_independent/endpoint_step500.pt`|1.563|Historical connected-patch independent-context T2 control; not a maintained public loader dependency or a planned formal initialization. Native_correct step500 and historical raw T1/T2 comparisons remain. New-object causal controls require new matched training.|
|5|`experiments/mf_h1_local0/outputs/t1_gated_loss_factorial_20260929_225800/M01/additional_step500_cumulative_step1000.pt`|1.563|Retired gated FC-only or FF-only exploratory branch. Next planned joint recipe is JEdge5, not these older FC/FF vertex-relation losses; keep reports, fixed raw outputs and scores.|
|6|`experiments/mf_h1_local0/outputs/t1_gated_loss_factorial_20260929_225800/M10/additional_step500_cumulative_step1000.pt`|1.563|Retired gated FC-only or FF-only exploratory branch. Next planned joint recipe is JEdge5, not these older FC/FF vertex-relation losses; keep reports, fixed raw outputs and scores.|
|7|`experiments/mf_h1_local0/outputs/t1_interface_3arm_20260929_181500/A/additional_step250_cumulative_step750.pt`|1.563|Pure-T1 step750 intermediate superseded by retained same-arm step1000 and step1500; no maintained default depends on step750.|
|8|`experiments/mf_h1_local0/outputs/t1_interface_3arm_20260929_181500/B/additional_step250_cumulative_step750.pt`|1.563|Retired known-boundary linear injection experiment, absent from the maintained M1/M2/M3 interface. Keep diagnostic evidence; remove intermediate/final exploration weights.|
|9|`experiments/mf_h1_local0/outputs/t1_interface_3arm_20260929_181500/B/additional_step500_cumulative_step1000.pt`|1.563|Retired known-boundary linear injection experiment, absent from the maintained M1/M2/M3 interface. Keep diagnostic evidence; remove intermediate/final exploration weights.|
|10|`experiments/mf_h1_local0/outputs/t1_interface_3arm_20260929_181500/C/additional_step250_cumulative_step750.pt`|1.563|Retired FM+0.1FC+0.1FF shared-vertex relation branch, absent from next recipes. Keep diagnostic evidence; remove intermediate/final exploration weights.|
|11|`experiments/mf_h1_local0/outputs/t1_interface_3arm_20260929_181500/C/additional_step500_cumulative_step1000.pt`|1.563|Retired FM+0.1FC+0.1FF shared-vertex relation branch, absent from next recipes. Keep diagnostic evidence; remove intermediate/final exploration weights.|
|12|`experiments/mf_h1_local0/outputs/t1_time_gate_20260929_213000/C_gate/additional_step250_cumulative_step750.pt`|1.563|Retired gated FC+FF exploration or its intermediate checkpoint. No next-phase initialization/runtime dependency; preserve fixed evidence.|
|13|`experiments/mf_h1_local0/outputs/t1_time_gate_20260929_213000/C_gate/additional_step500_cumulative_step1000.pt`|1.563|Retired gated FC+FF exploration or its intermediate checkpoint. No next-phase initialization/runtime dependency; preserve fixed evidence.|
|14|`experiments/mf_h1_local0/outputs/tgeo_m10_20260930_173200/Geo_continue/additional_step250_cumulative_step2250.pt`|1.565|Intermediate Geo FM step2250 superseded by retained canonical G0 step2500; no maintained runtime depends on this intermediate.|
|15|`experiments/mf_h1_local0/outputs/tgeo_m10_20260930_173200/Geo_continue/additional_step500_cumulative_step2500.pt`|1.565|Canonical tgeo_g0_cumulative2500.pt has CPU-verified identical learned model tensor SHA256 and stream_state. Old optimizer moments are unnecessary under maintained fresh-AdamW continuation. Keep canonical G0 and old run evidence.|
|16|`experiments/mf_h1_local0/outputs/tgeo_m10_20260930_173200/Geo_M10/additional_step250_cumulative_step2250.pt`|1.565|Retired Geo plus gated FC-only exploratory branch, distinct from maintained JEdge5. Keep reports/raw evidence; delete intermediate/final exploration checkpoints.|
|17|`experiments/mf_h1_local0/outputs/tgeo_m10_20260930_173200/Geo_M10/additional_step500_cumulative_step2500.pt`|1.565|Retired Geo plus gated FC-only exploratory branch, distinct from maintained JEdge5. Keep reports/raw evidence; delete intermediate/final exploration checkpoints.|
|18|`native_t1/runs/jedge5_diagnosis_20261002_102215/stage_d/R_B/raw_additional250_cumulative2750.pt`|0.522|Retired replay intermediate; training logs, raw outputs and final EMA retained|
|19|`native_t1/runs/jedge5_diagnosis_20261002_102215/stage_d/R_B/EMA_additional250_cumulative2750.pt`|0.522|Retired replay intermediate; training logs, raw outputs and final EMA retained|
|20|`native_t1/runs/jedge5_diagnosis_20261002_102215/stage_d/R_B/raw_additional500_cumulative3000.pt`|0.522|Replay raw final has canonical generator state; canonical FM/JEdge5 retained; wrapper is not byte-identical|
|21|`native_t1/runs/jedge5_diagnosis_20261002_102215/stage_d/R_J/raw_additional250_cumulative2750.pt`|0.522|Retired replay intermediate; training logs, raw outputs and final EMA retained|
|22|`native_t1/runs/jedge5_diagnosis_20261002_102215/stage_d/R_J/EMA_additional250_cumulative2750.pt`|0.522|Retired replay intermediate; training logs, raw outputs and final EMA retained|
|23|`native_t1/runs/jedge5_diagnosis_20261002_102215/stage_d/R_J/raw_additional500_cumulative3000.pt`|0.522|Replay raw final has canonical generator state; canonical FM/JEdge5 retained; wrapper is not byte-identical|
|24|`native_t1/runs/geometry_context_revised_20260930_153000/T_base/additional_step250_cumulative_step1750.pt`|1.563|Retired intermediate; Base/Geo final endpoints and scientific evidence retained|
|25|`native_t1/runs/geometry_context_revised_20260930_153000/T_geo/additional_step250_cumulative_step1750.pt`|1.565|Retired intermediate; Base/Geo final endpoints and scientific evidence retained|
|26|`native_t1/runs/geometry_context_revised_20260930_153000/T_graph/additional_step250_cumulative_step1750.pt`|1.565|Retired intermediate; Base/Geo final endpoints and scientific evidence retained|
|27|`native_t1/runs/geometry_context_revised_20260930_153000/T_graph/additional_step500_cumulative_step2000.pt`|1.565|Retired graph candidate outside maintained model and next M1/M2/M3 comparison|

## 30 个临时文件

- numeric_reference_A 的两个 code_diff_roundtrip 临时目录内共 26 个文件；逐项 SHA256 与保留的 source_snapshot 一致。历史 final_audit.json 的 roundtrip_directory 仅代表当时补丁验证位置，清理后不可再打开。
- Equal-NFE 源码目录里的两个 cpu_tests_preflight JSON；与最新运行 tests/sampler_initial.json、sampler_final.json 完全一致。
- datasets/__pycache__ 下两份 Python 字节码；对应源码保留。

## 审计

- [逐文件删除清单与 SHA256](deletion_manifest.json)
- [3,711 个关键保留文件的删除前 SHA256](protected_before.json)
- [执行计划和空间统计](plan_summary.json)
- [实验分支逐项保留/退役理由](experiments_checkpoint_recommendations.json)
- [数据集扫描与下载建议](DATASET_DOWNLOADS.md)

自动审批拒绝理由：命令会不可逆删除 57 个文件、约 35.97 GB 历史 checkpoint 及临时文件；用户已授权清理目标，但尚未明确批准这份具体的大规模删除清单，且部分历史终点可能难以重建。用户随后明确回复“看了一下没问题，你严格按照清单删除就好”，本轮依据该批准完成执行。
## 执行与验证结果

- 57 条删除回执与冻结清单的路径、文件大小和 SHA256 逐项一致，无重复或额外回执；所有目标已不存在。
- 27 份 checkpoint 加 30 个临时文件，合计 35,971,879,861 bytes（35.972 GB / 33.501 GiB）。未额外删除空目录。
- 3,711 个受保护文件重新计算 SHA256，全部与删除前一致；对最初 51,543 个文件的盘点检查，清单外缺失为 0。
- 官方资产、五份 canonical、无 Geo T1 500/1000/1500、Base/Geo2000、两个 EMA3000、最新 Equal-NFE 全部证据及本地数据均保留。
- 本轮没有模型前向、训练更新、数据下载、权重改写、NTFS 压缩、Git commit 或 push。
- 磁盘可用空间是实时卷级快照，其他活动也会改变它；本轮清理空间按上述实际删除文件大小计账，不把全卷空闲增量全部归因于本操作。

证据：[用户明确批准](user_approval.json)、[逐文件删除回执](deletion_receipts.jsonl)、[保留资产与缺失检查](verification.json)、[完成摘要](cleanup_summary.json)。