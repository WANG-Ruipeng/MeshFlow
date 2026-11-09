# README_RESUME

本轮输出仅位于 experiments/r2_anchor_aware_denoiser_v1，代码仅位于 r2/pilots/anchor_aware_denoiser_v1。运行环境为已有WSL Ubuntu与 /mnt/e/MeshFlow-Control-artifacts/venv/bin/python，无新依赖安装。

在 /mnt/e/MeshFlow 下执行：

    /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B r2/pilots/anchor_aware_denoiser_v1/run.py --stage all

可选阶段：prepare、preflight、train、inference、score、gallery、finalize。prepare验证冻结注册/原资产，不重新物化噪声。stage all依赖完整数据准备产物；本轮已有这些产物。最终图审需要实际AI_PROXY阅读，不由渲染脚本自动判成功；finalize只接受完整审阅与审计记录。

训练阶段只运行两个登记臂；每臂1000更新，每500保存恢复检查点，1000保存完整状态和推理导出。完整模型、AdamW、CPU/CUDA RNG与流位置保留。completed终点验证哈希后跳过；最终导出缺失时从已完成1000步完整检查点恢复导出，不重复训练。

GPU训练支持在完整更新边界响应SIGINT/SIGTERM、保存恢复状态。恢复仅接受已知完成步数与前向/反向/更新账本完全相符的干净边界；自动恢复不允许回放已经发生但未保存的更新。若进程硬崩溃、OOM、数值错误或存在未决调用，先检查failure/phase/ledger，禁止静默重试。500恢复点不意味着可以无视后续已消耗更新预算重跑500—1000。

推理完成的每份输出通过checkpoint/input/registration/source/output哈希核验后跳过；存在START无完整receipt时停止，不重抽噪声或重新生成。CPU评分与几何导出使用原子暂存目录，已完成项验证后复用；中断的staging目录需检查，不自动覆盖。

累计GPU阶段进程计时上限10800秒，保守包含加载和保存时间。预检32F/16B/2丢弃更新单列；正式16000逻辑F/16000B/2000更新；真实推理96F；合成54F；生成器/Euler/OT均0。激活重算为同一逻辑反向内的原层重算，不当作新增数据曝光。达到时间上限时保存边界并如实标部分完成，不通过恢复入口重新获得额外预算。

重要文件：
- registration.json、pre_gpu_registration_v0.json：固定科学设置及GPU前工程修订。
- data_audit/：112条件白名单、5条排除理由、真实来源接口、源hash。
- stream/：同一8000曝光、1000实际批次、18合成输入，不能删掉后再假装原样恢复。
- gpu_phases/、gpu_calls.jsonl、training/*/ledger.jsonl：资源时间、调用和更新。
- checkpoints/*/step0500.pt、step1000.pt、inference.pt：恢复点、唯一终点、推理导出。
- inference/、synthetic_results/：新去噪结果与合成诊断。
- evaluation/：144 POST索引、固定质量与新连接诊断、全阈值和配对。
- gallery/：全部完整图、局部图与实际AI_PROXY审阅。
- independent_audit/：独立计数、配对和数值核验。
- source_snapshot/、source_hashes.csv、code.diff：实际运行代码证据。
- REPORT.md、METHODS.md、review包：可审阅交付。

旧官方权重、生成器、RAW、评分器和报告均只读。不自动commit/push、切换默认模型、延长训练或开启新loss。任务完成后不要再次调度GPU。

## 本轮完成状态（2026-10-10）

两臂各1000更新，96次真实RAW推理与54次合成推理全部完成；144份POST评分及56张原PNG的实际图审完成，独立最终审计PASS。累计GPU阶段3939.515653秒，未发生GPU重试。最终结论见REPORT.md。

finalization_completion.json存在时，run.py --stage all先核验冻结输入/训练源码，再直接调用finalize的完成态校验并返回；不会进入GPU、重新汇总、重新渲染或改写已完成结果。单独指定某阶段时仍执行该阶段自己的身份/hash检查。完整终态复验不要删除completion、注册、stream或旧输入来强制重跑。

review ZIP是便携证据包，不是完整恢复环境。恢复仍需要本地官方权重、独立源码目录、原RAW/查询、完整stream及checkpoint；大权重与数组按local_binary_artifacts.csv保留在本地，未嵌入ZIP。
