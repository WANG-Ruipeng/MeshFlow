# 本轮执行与恢复入口

代码目录：`experiments/mf_h1_local0/chair_control_feedback_auto_v1`。在 `E:\MeshFlow` 对应的 WSL 工作目录执行，使用已有 `/home/warpwang/miniforge3/envs/meshflow/bin/python`。不需要安装新依赖。

本目录是本轮的冻结结果目录，不应删除账本后重跑。当前入口会复用经过 hash 核验的完整结果；未完成的科学调用不会自动重试。工程保存失败必须先保留原失败和账本，并留下具体的有限恢复收据。

```bash
PY=/home/warpwang/miniforge3/envs/meshflow/bin/python
# 实际执行过的准备、冻结及输入参考入口（均无模型调用）
$PY -B -m experiments.mf_h1_local0.chair_control_feedback_auto_v1.assets
$PY -B -m experiments.mf_h1_local0.chair_control_feedback_auto_v1.protocol
$PY -B -m experiments.mf_h1_local0.chair_control_feedback_auto_v1.evaluation --prepare

# 总入口：START 真实采样/评分/反馈池；完成原 FM、轨迹重放、匹配 teacher 对照
$PY -B -u -m experiments.mf_h1_local0.chair_control_feedback_auto_v1.run explore

# 单臂可恢复入口；每100步恢复权重，500/1000永久权重和全DEV
$PY -B -u -m experiments.mf_h1_local0.chair_control_feedback_auto_v1.run train --phase explore --arm TRAJECTORY_REPLAY

# 独立 CPU 评分与全量图集
$PY -B -m experiments.mf_h1_local0.chair_control_feedback_auto_v1.run score --label explore_TRAJECTORY_REPLAY_1000
$PY -B -m experiments.mf_h1_local0.chair_control_feedback_auto_v1.run visual --labels START explore_REPLAY_FM_CONTROL_1000 explore_TRAJECTORY_REPLAY_1000
$PY -B -m experiments.mf_h1_local0.chair_control_feedback_auto_v1.run compare --candidate explore_TRAJECTORY_REPLAY_1000 --matched explore_REPLAY_FM_CONTROL_1000

# 训练期间用 --live；全部结束后省略该参数执行严格最终账本审计
$PY -B -m experiments.mf_h1_local0.chair_control_feedback_auto_v1.accounting_audit --live
```

`worker.py sample` 是上述总入口调用的唯一 GPU 采样实现，可以指定已保存权重、标签和冻结 manifest。它始终使用原 Euler50，保存 51 态及 50 速度，随后导出与重读 PLY/NPZ。评分和渲染分别由 `evaluation.py`、`visualization.py` 执行，均不调用生成模型。

续训权重须使用本目录 `checkpoint.py` 的 `load`，它恢复完整模型、AdamW、随机状态和数据流游标，核验累计步数 `5000 + feedback_step`。不能用旧 pilot 的训练总入口误读历史 `pilot_step=1000`。原 START 使用旧严格加载器核验，文件保持原位且未复制。

独立重复受 `decisions/repeat_authorization.json` 的真实 DEV 正面门槛约束；CONFIRM 还要求冻结五模型身份、两重复臂各1000步完成及原确认 manifest 的 hash。未触发时不读取确认输出。

本轮固定墙钟窗口记录在 protocol.json；新的独立复现实验必须先登记新的输出目录、窗口及预算。不要通过修改本轮冻结定义、删除账本或重置计数绕过上限。默认模型、公共代码与旧数据/结果均保持原状。

## 已实际执行的末轮修订与终点反馈

H3 的 `register` 只在 3000 个既有更新及全组反馈完成后执行一次；`decisions/H3_recipe.json` 和两份冻结证据子目录保留当时版本。已登记目录不要再次执行 `register`，其时间及证据身份不允许重写。

```bash
# 最后一个探索配方；原始 h3 register 已执行，下面是实际训练入口
$PY -B -u -m experiments.mf_h1_local0.chair_control_feedback_auto_v1.h3 train

# 读取原三个1000终点，在原TRAIN_FEEDBACK16上做预先登记的真实生成诊断
$PY -B -u -m experiments.mf_h1_local0.chair_control_feedback_auto_v1.postfeedback

# H3终点在同一反馈池的已登记诊断；不是新训练目标或CONFIRM
$PY -B -u -m experiments.mf_h1_local0.chair_control_feedback_auto_v1.h3 diagnostic

# 最终纯CPU诊断和清单；训练全部完成后才用最终模式
$PY -B -m experiments.mf_h1_local0.chair_control_feedback_auto_v1.training_diagnostics
$PY -B -m experiments.mf_h1_local0.chair_control_feedback_auto_v1.accounting_audit
$PY -B -m experiments.mf_h1_local0.chair_control_feedback_auto_v1.checkpoint_inventory
# Read-only attribution with the registered matched and coefficient-ablation maps
$PY -B -m experiments.mf_h1_local0.chair_control_feedback_auto_v1.error_attribution \
  --matched-map native_t1/runs/chair_control_feedback_auto_v1/derived/attribution/DEV/matched_map.json \
  --ablation-map native_t1/runs/chair_control_feedback_auto_v1/derived/attribution/DEV/coefficient_ablation_map.json
$PY -B -m experiments.mf_h1_local0.chair_control_feedback_auto_v1.view_catalog
```

上述 GPU 命令是复现说明，不授权固定任务结束后自动再开实验。全部正式步骤的实际状态以最终报告和账本为准；未完成的阶段不得从这些示例命令推定为已完成。

已有的 `run compare` 不带 `--visual-review` 时会把视觉状态写成 NOT_RUN。复核全组结果必须显式传入对应已实际看图的 review JSON，例如：

```bash
$PY -B -m experiments.mf_h1_local0.chair_control_feedback_auto_v1.run compare \
  --candidate explore_TRAJECTORY_REPLAY_1000 --matched explore_REPLAY_FM_CONTROL_1000 \
  --visual-review native_t1/runs/chair_control_feedback_auto_v1/visuals/reviews/explore_TRAJECTORY_REPLAY_1000_vs_explore_REPLAY_FM_CONTROL_1000.json
```

新权重使用本轮 checkpoint schema，不能直接假定公共 `native_t1 sample` 的旧 schema 加载器接受它。实际采样路径是本轮 `worker sample` → 本轮严格 `checkpoint.load` → 原 `chair_hybrid_coupling_v1.sampling.sample_case`。模型结构和推理输入不变；所有终点诊断均从磁盘重新加载实际新权重后生成，以验证加载路径。默认公共 START 入口未改。

二进制 PLY 与 NPZ 均导出在登记的模型坐标系；逐位保持指该坐标系内的原始 C。源文件 C_world、源面/顶点关联与明确 transform 在条件 NPZ 中保留。没有输出近似逆变换的 world PLY 后冒称原始文件坐标逐位相同，也没有修复或去重原 C。