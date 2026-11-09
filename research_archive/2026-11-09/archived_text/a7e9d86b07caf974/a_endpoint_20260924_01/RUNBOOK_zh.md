本轮复现入口

在WSL Ubuntu中进入固定工作树：

```bash
cd /mnt/e/MeshFlow/.worktrees/a-layout-value-20260924
PY=/home/warpwang/miniforge3/envs/meshflow/bin/python
RUN=output/a_endpoint_reproduction_new
$PY -B -m experiments.a_radial.layout_endpoint.prepare --source-run output/a_taskaligned_20260924_01 --run "$RUN"
$PY -B -m pytest -q -p no:cacheprovider experiments/a_radial/layout_endpoint
$PY -B -m experiments.a_radial.layout_endpoint.correctness --source-run output/a_taskaligned_20260924_01 --output-dir output/a_endpoint_reproduction_correctness_new
$PY -B -m experiments.a_radial.layout_endpoint.smoke --source-run "$RUN" --out output/a_endpoint_reproduction_smoke_new
$PY -B -m experiments.a_radial.layout_endpoint.smoke --source-run "$RUN" --diagnostic
```

correctness全模型全设备strict pass为false不能忽略：原A GPU压力对象的数值等价性存在局限。先读取本轮IMPLEMENTATION_GATE.json与TRAINING_GATE.json，逐条检查CPU镜像、实际v1初始化GPU、连续恢复和单对象下降；不能仅看命令退出码就把未通过压力检查重命名为全部PASS。新的复现gate必须引用该次实际证据、源码与协议hash，原gate不能直接复用到不同协议。

正式三臂从同seed的v1 final副本出发，各1000步，仅用final：

```bash
for ARM in c0 c1 c2; do
  for SEED in 10 11; do
    $PY -B -m experiments.a_radial.layout_endpoint.train --run "$RUN" --arm "$ARM" --seed "$SEED"
  done
done
$PY -B -m experiments.a_radial.layout_endpoint.evaluate --run "$RUN"
$PY -B -m experiments.a_radial.layout_endpoint.final_parity --run "$RUN"
$PY -B -m experiments.a_radial.layout_endpoint.summarize --run "$RUN"
$PY -B -m experiments.a_radial.layout_endpoint.visualize --run "$RUN"
```

run_all.py是本次已验TRAINING_GATE的串行执行器。逐条train命令是底层复现入口，不能用其绕过方法要求的实际门禁。

续训使用同预算、同源码、同协议：--resume <step_00250.pt等> --output-dir training/resumed_c1_s10_new。输出必须是该run下尚不存在的新目录，checkpoint包括完整Adam、scheduler、Python/NumPy/Torch CPU/CUDA随机状态、抽样游标和trace。若续训输出采用新目录，最终评测身份路径也必须明确登记，不得随意把任意checkpoint冒充final。

部署与训练的区别：正式表始终调用原layout_student.deployment.deploy（CPU32+最终方向半径重查+精确审核）。FINAL_DEPLOYMENT_PARITY仅测量GPU/可微CPU镜像的数值差异，不替换主表、不选择权重。全部质量失败、实际面数、原始输出和回退输出均保留。

图表：figures/training、all_parent_outcomes、error_quality_tradeoff，均提供PNG和PDF。表面几何与全部尝试见attempts；封存后另提供OBJ。完整三臂损失/数据/门槛说明见METHOD_zh.md，最终统计见REPORT_zh.md和SUMMARY.json。

本轮没有新下载数据。source_run提供的16对象仍是既有开发池，12/4只是内部划分；更大未见确认集和完整MeshFlow生成实验需要另行实施。
封存前先完成上面的评测、最终设备差异测量、统计和作图；本次还保存了最终单测XML与CHECKS.json。只读预检与封存入口：

```bash
$PY -B -m experiments.a_radial.layout_endpoint.package_results --run "$RUN" --smoke-run output/a_endpoint_reproduction_smoke_new --audit-only
$PY -B -m experiments.a_radial.layout_endpoint.package_results --run "$RUN" --smoke-run output/a_endpoint_reproduction_smoke_new
```

封存采用排他路径，已存在的包不覆盖。封存会校验旧实验877项文件、准备数据20项、六组最终完整训练状态和配对随机序列、160次尝试与288个结果槽位、网格逻辑/存储/文件哈希，然后导出OBJ、复制源码与证据、压缩并重新解压。在隔离Python进程中仅加载解压包内源码与c2 seed10最终权重，重放预定首UID，要求原始输出与回退策略的网格及判定一致。ZIP校验与重放报告写在包外，避免修改已封存内容。封存通过只说明证据完整，不会将科学门槛或设备差异变成PASS。