# R2_BENCH_COVERAGE_EXPANSION_V1 执行与恢复入口

2026-10-21。本文按已冻结模块的实际 CLI 编写，不替代 `PROTOCOL.md`、`registration.json`、`training_registration.json` 或调用账本。本文件建立时唯一训练臂已经启动；不要照抄首次训练命令再启动第二个进程。本文写入不执行训练、采样、评分或恢复。

冻结范围：`COVERAGE_SEAM` 新增 1,500 更新，终点 `COVERAGE_S1500`；321 条 TRAIN 条件/79 父，其中 55 条闭合 C、42 条开放但无来源 owner；原 DEV 39 条保留。相同 A step0 初始化与 fresh AdamW。评测严格复用原 48 请求，旧 `A_S1500` 只读复用。

## 1. 环境和资源

下面是 Windows PowerShell 中的 WSL 命令。发行版固定 `Ubuntu`，工作目录固定 `/mnt/e/MeshFlow`，解释器固定为本轮实际使用的 `/mnt/e/MeshFlow-Control-artifacts/venv/bin/python`；不使用默认 `python3`，不要新建环境、下载权重或重新生成噪声。每条命令独立执行并检查退出状态，不把全过程放进无条件自动重试循环。

每次新的 GPU 阶段先读取实际资源，保留原始 stdout/JSON 到独立 dispatch 记录；下面的命令仅查询，不构造模型：

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -c "import json; from r2.pilots.bench_coverage_expansion_v1.common import resources; print(json.dumps(resources(),ensure_ascii=False,indent=2))"
```

记录 GPU 名称、总/已用/空闲显存、利用率和进程，CPU/RAM 与其他科研进程负载。额外保留至少 `max(2048 MiB, 0.1 × 总显存)`；WDDM 的单进程显存可能 NA，不能把 allocator 峰值冒充进程峰值。唯一训练臂不另加训练 worker；CPU 评分或图集可在资源允许时与独立 GPU 工作重叠。

相近任务的规划值：训练进程 4,500 MiB、RAW 2,800 MiB、D0 3,500 MiB。采样/D0 CLI 的估计值是调度门槛，不改变模型或数值定义。实际首个已授权批次/输出之后重新观察资源；不额外跑 probe。相近训练 allocator 峰值为 allocated 约 2,809 MiB、reserved 3,206–3,224 MiB，这不包含所有进程开销。历史 A 1,500 步约 2,820 秒、48 RAW 约 257 秒，存在同期并行，不能当作独占速度保证。

## 2. 只读查看训练及精确恢复

状态、账本和当前完整恢复点：

```powershell
Get-Content -Encoding UTF8 -Raw E:\MeshFlow\experiments\r2_bench_coverage_expansion_v1\training\COVERAGE_SEAM\status.json
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_coverage_expansion_v1.resume
```

同时查看 `training/COVERAGE_SEAM/training_log.jsonl`、`exposure.jsonl`、`calls.jsonl`、`checkpoint_history.jsonl`、`attempts/*/final_status.json` 和实际进程。`status.json` 的最后一步不等于一定已落盘的检查点步数，应读取其中的 `checkpoint.local_step` 和哈希。正在 TRAINING/LOADING 时不得启动第二个 writer，也不要删除 `writer.lock` 或权重。

| 状态 | 允许动作 |
|---|---|
| `TRAINING` / `LOADING` 且进程在运行 | 只监看；保持唯一 worker |
| `INTERRUPTED_SAFE` 且 `exact_resume_available=true` | 重新记录资源，精确恢复已保存完整游标，不回退、不增加步数 |
| `FAILED` / `OOM` / 进程消失但不是安全中断 | 保留现场、失败记录和调用账本；停止自动推进，先明确工程处理和额外计算记录 |
| `COMPLETE` | 核验 local_step=1500、global_step=13500、检查点和 generator 导出，再进入评测 |
| `NOT_STARTED` | 当前任务正常不应返回此状态；先检查路径/环境，不据此自动重开任务 |

只有安全恢复条件成立，才使用：

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_coverage_expansion_v1.dispatch
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_coverage_expansion_v1.resume --execute
```

`dispatch` 会重新测资源并写新的资源记录；已有相同 release 时不覆盖旧 go 文件。`resume --execute` 拒绝不安全状态。训练器要求账本与检查点游标完全相同，并校验模型、优化器和 RNG；没有获准的丢步重算或静默 retry。

首次训练的实际入口仅作记录，**已经启动后不再次执行**：

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_coverage_expansion_v1.training --arm COVERAGE_SEAM
```

CPU 合同核验入口为 `training --check-cpu`，它读取哈希和元数据、不加载检查点/构造模型，不应拿它额外跑训练探针。已存在 `cpu_gate.json` 和 `evaluation_cpu_checks.json` 为 PASS 时，不必反复重做准备。

完整训练预算是 12,000 train-forward、12,000 backward、1,500 optimizer-update；检查点 `step0/step500/step1500` 和 `rolling.pt` 仅恢复用途，正式评测只用 `generator_step1500.pt`。不能按 loss 或图像挑别的终点。

## 3. 固定评测输入检查

以下入口检查 48 条 C/N/固定噪声、旧 C-only 布局和来源导出元数据，不调用模型：

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_coverage_expansion_v1.eval_plan
```

它复核新旧 eval manifest 完全相等，唯一新标签为 `COVERAGE_S1500`，生成器终点 local1500/global13500。已冻结同内容结果可复用；哈希冲突不能覆盖。不要重新运行 `prepare candidates/publish` 或 `register` 来改变已经开训的数据合同。

## 4. 新 RAW：48 请求、2,400 前向

确认训练 COMPLETE 且终点导出存在，资源允许后执行：

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_coverage_expansion_v1.sample --label COVERAGE_S1500 --estimated-peak-mib 2800
```

它仅加载本轮终点，使用原同步 `SEAM` Euler50、BF16 前向与 FP32 积分，每个请求 50 次生成器前向。布局直接复用 `OLD_S15000` 缓存；没有占用头新前向、来源 owner oracle、新噪声或 ASYNC。

完成后检查：

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -c "from r2.pilots.bench_coverage_expansion_v1.eval_plan import all_raw_barrier; print({'verified_RAW_receipts':len(all_raw_barrier())})"
```

必须返回 48。原始证据位于 `evaluation/COVERAGE_S1500/<request_id>/raw/`，聚合完成文件为 `raw_complete_full.json`。已经完整且哈希吻合的请求复用；存在无完成 receipt 的目录会报错保留现场，不能删目录来绕过未决尝试。不要重新生成历史 A。

## 5. 单次官方 D0 + GEO_C + 来源定向

全部 48 RAW 先保存且通过 barrier 后，重新检查资源，再执行：

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_coverage_expansion_v1.post_d0 --label COVERAGE_S1500 --estimated-peak-mib 3500
```

每请求最多一份单次官方 Bench D0 提议，恢复原 C 与 seam 锚点，执行原 GEO_C 和来源定向。阶段资产在各请求的 `post/`：`proposal.npz`、`pre_geo.npz`、`post_canonical.npz`、`post.npz` 及来源/几何/方向 receipt。聚合为 `post_complete_full.json`，48 个请求各 `D0_forwards=1`、`GEO_calls=1`。

若 D0 提议已完整保存而 CPU 几何尚未开始，现有实现可以复用提议继续几何，不再调用 D0；若已有未完成的 `geometry_attempt.json` 或模型前向未决，则入口拒绝静默重跑。先核查该请求调用账本再决定工程恢复。

## 6. 固定评分与 CPU 后处理

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_coverage_expansion_v1.evaluation prepare
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_coverage_expansion_v1.evaluation raw --labels COVERAGE_S1500 --workers 2
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_coverage_expansion_v1.evaluation post --labels COVERAGE_S1500 --workers 2
```

`--workers 2` 是上限请求，入口先执行第一份已授权任务、测 CPU/RAM 后选择实际 1 或 2 worker，线程数固定为 1，不额外运行 probe。RAW 评分可与独立 D0 阶段交叠，但 POST 评分/CPU级联需要全部对应 POST producer 完成。新科学设置不变。

RAW 48 行；POST 入口每请求对 `BASE_POST` 和 `CLEAN_FINAL_POST` 各评分一次，共 96 行。固定级联是 `BASE_POST → HINGE_OPP015 → ALPHA → HOLE`，原 `require_opposite_winding=True` 等合同保留。HINGE/ALPHA 中间产物保存但不另加评分阶段。总计 144 新评分，不重新评分旧 A。所有模型调用为零。

完成文件在 `evaluation/completions/raw__COVERAGE_S1500.json`、`post__COVERAGE_S1500.json`。原始逐请求评分在 `evaluation/scores/.../RAW/`、`cpu/.../BASE_POST/scores/`、`cpu/.../CLEAN_FINAL_POST/scores/`。部分目录无完成 receipt 时拒绝覆盖，不能直接重试绕过失败。

## 7. 完整表格、图集与实际审阅

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_coverage_expansion_v1.collect --check-inputs
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_coverage_expansion_v1.collect
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_coverage_expansion_v1.gallery prepare
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_coverage_expansion_v1.gallery run --workers 2
```

collect 必须收齐 `A_S1500` 和 `COVERAGE_S1500` 各 48 请求、3 阶段，共 288 行。旧 A 的 row/receipt 只读认证，共享评分源码必须与旧注册一致；不从旧错误汇总表复制值。输出为 `results/per_output.csv`、`per_condition.csv`、`per_parent.csv`、`parent_macro_summary.csv`、`paired_per_output.csv`、`paired_per_condition.csv`、`paired_per_parent.csv` 和 `completion.json`。

图集 144 页：每请求 RAW/BASE/FINAL 三页，每页两模型对照；两视角和 front/side 中央截面保持原坐标、尺度和位置，输出 `gallery/index.html`。渲染完成状态为 `RENDER_COMPLETE_PENDING_ACTUAL_REVIEW`，不是已图审。必须实际看每张登记图片并另存非盲 `AI_PROXY` 记录；不靠已算指标自动生成用途标签，不把截线端点直接称为物理裂缝。

图审和分析的口径见 `preparation/METRIC_READING.md`。保存逐输出记录、改善/恶化父、坏尾部和未决判断；不存在唯一来源款式正确答案。

完成 collector 后，可运行只读 CSV 分析入口；缺少完成表格时只返回 WAIT，不造占位结果：

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B experiments/r2_bench_coverage_expansion_v1/analysis_results.py --check-inputs
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B experiments/r2_bench_coverage_expansion_v1/analysis_results.py
```

输出 `analysis/key_metrics.json` 与关键指标、逐父配对、配对父汇总、`split×part_type×stage` 配对汇总及状态CSV；保留原表全部字段的路径/哈希/列名，不重评分。接口布尔率以明确的 `OPEN_C_SCOPE` 排除闭合C并报告有效n；真正碎片面积未实现时记NA，不把脱离C的大主体自动当碎片。该入口不生成结论或图审。

## 8. 完成与停止

固定产物齐全、实际图审完成后写报告，首先回答扩充条件覆盖是否值得继续；保留相反代价和有限样本限制。报告引用真实调用账本、资源/用时、最终权重哈希、数据/输入哈希、阶段完成 receipt。正常全程新增模型调用上限：训练 12,000F/12,000B/1,500U，采样 2,400F，D0 48F，head/probe 0。

完成后停止；不自动加训、换默认模型、删除旧证据、commit 或 push。任何失败都保持真实状态和实际调用数，不能为了收齐表格静默补跑或换请求。

## 9. 本次完成状态与只读整理入口

本次正式训练、生成、评分和实际图审均已完成。最终核验收据见 `final_verify_latest.json`，总收据见 `completion.json`，结论见 `REPORT.md`。不要因重新整理报告而执行训练、采样或评分步骤。

以下入口只读取已完成证据；缺少真实图审或核验未通过时不会生成伪完成报告：

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B experiments/r2_bench_coverage_expansion_v1/analysis_review.py
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B experiments/r2_bench_coverage_expansion_v1/build_closeout.py
```

独立核验脚本 `final_verify.py` 已首次执行PASS，检查调用账本、保护资产、网格C保持、评分与实际图审身份，不导入torch，不反序列化检查点，不重评分。其收据不会被报告整理覆盖。终点权重、RAW及所有阶段产物仍保留；无需重跑实验即可复算表格和查看截面。
