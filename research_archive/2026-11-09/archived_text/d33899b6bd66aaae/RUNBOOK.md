# R2_BENCH_C_EDGEFLIP_V1：恢复与核验

本轮是同一物理 C、同 K/N/外边界的内部翻边快测，随后执行用户明确批准的 PLAIN200 / FLIP200 各 200 更新。没有新增网络、loss 或噪声。旧 EXPOSURE 与原评分器保持不变。

## 已冻结的定义

- `registration.json`：冻结模型的 4 个新翻边 RAW；6 个原 RAW 缓存复用。
- `training_registration.json`：同 EXPOSURE live 参数、AdamW、RNG，固定 200 行计划、batch8/micro1，分别 200 更新。整体 CLEAN/X_WIDTH 配对交替翻边；几何门槛不通过时保留原输入。
- `candidate_evaluation/registration.json`：三模型各 6 个原输入、4 个合法翻边输入。原 EXPOSURE 六份 RAW/FINAL 复用；本轮共 24 新 RAW、24 单次官方 Bench D0。全部 RAW 保存后才执行 D0。
- 模型终点固定 local200/global25700，不选 best。推理 Euler50，原 D0 → GEO_C/来源绕序 → HINGE_OPP015 → ALPHA → HOLE。
- 图审是实际非盲 AI_PROXY，不是人类认证。原输入三父与合法翻边两父分开汇总；f495 不满足翻边几何门槛，没有替换父或放宽门槛。

## 当前权重

两臂各保留 `training/<arm>/checkpoint_step200.pt`（完整恢复状态）及 `generator_step200.pt`（推理导出）。路径与 SHA256 在各臂 `status.json`；这些新分支没有改变默认模型。旧 EXPOSURE 起点见训练登记。滚动恢复点不是额外候选。

## 执行环境

Windows 工作区 `E:\MeshFlow`；WSL Python `/mnt/e/MeshFlow-Control-artifacts/venv/bin/python`，模块工作目录 `/mnt/e/MeshFlow`。

所有命令前先读现有收据。`COMPLETE.json` 存在时优先运行只读最终核验；不要重跑注册、改写冻结输入或通过删除 partial 目录绕过失败。未完成科学调用需要按账本定位，不能静默重复。资源并行合同见 AGENTS.md，调度记录在 `dispatch/`、各 worker 资源文件中。

```powershell
# 最终完成态只读核验（不加载模型、不训练、不重新评分）
wsl.exe --cd /mnt/e/MeshFlow /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_c_edgeflip_v1.closeout
```

下面是本轮实际阶段入口，供定位阶段与读取完成收据；不是授权再次训练或生成。已完成入口应验证后返回。任何 incomplete/failed 状态先核对原调用账本。

```text
python -B -m r2.pilots.bench_c_edgeflip_v1.run status
python -B -m r2.pilots.bench_c_edgeflip_v1.training --arm PLAIN200
python -B -m r2.pilots.bench_c_edgeflip_v1.training --arm FLIP200
python -B -m r2.pilots.bench_c_edgeflip_v1.candidate_eval sample --label PLAIN200 --worker plain_eval_w0
python -B -m r2.pilots.bench_c_edgeflip_v1.candidate_eval sample --label FLIP200 --worker flip_eval_w0
python -B -m r2.pilots.bench_c_edgeflip_v1.candidate_post verify
python -B -m r2.pilots.bench_c_edgeflip_v1.candidate_post d0 --worker d0_w0 --estimated-peak-mib 3500
python -B -m r2.pilots.bench_c_edgeflip_v1.candidate_eval score_raw --workers 2
python -B -m r2.pilots.bench_c_edgeflip_v1.candidate_eval post --workers 2
python -B -m r2.pilots.bench_c_edgeflip_v1.candidate_analysis all --workers 2
```

## 结果路径

- `REPORT.md`：结论、训练合同、局限与完整对照。
- `candidate_evaluation/analysis/`：逐输出原177字段、共同C、图/形状统计、逐父与逐请求配对、删面成本、完整训练日志与曲线。
- `candidate_evaluation/gallery/index.html`：20页同尺度双视图和原位置双截面，每页三个模型。
- `visual_review/candidates_actual.json`：60格实际图审，保留不确定项。
- `analysis/`、`gallery/`：前置冻结模型翻边诊断。
- `source_archive/`、`code.diff`、`source_hashes.json`：实际新增代码与原评分代码快照。
- `COMPLETE.json`：实际执行计数和产物校验；ZIP不含大权重和轨迹，完整产物留在本地。

没有从小面板结果自动扩大训练、清理旧资产、切默认、commit或push。
