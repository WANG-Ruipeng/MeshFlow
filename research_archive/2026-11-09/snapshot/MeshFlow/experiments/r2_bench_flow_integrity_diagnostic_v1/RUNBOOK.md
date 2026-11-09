# 恢复与核验

工作目录 `E:\MeshFlow`。WSL Python为`/mnt/e/MeshFlow-Control-artifacts/venv/bin/python`；环境`PYTHONPATH=/mnt/e/MeshFlow:/mnt/e/MeshFlow/r2/src`。

总入口：

```bash
python -B -m r2.pilots.bench_flow_integrity_v1.run
```

若COMPLETE存在，只验证交付hash，不跑GPU/OT/评分。阶段已开始而未结束的科学attempt拒绝静默重放；人工先查相应failure/calls账本。当前所有阶段已完成，Euler100门未触发，不得用恢复入口开启它。

本轮实际阶段入口（记录用途，不用于重复执行）：

```bash
python -B -m r2.pilots.bench_flow_integrity_v1.register
python -B -m r2.pilots.bench_flow_integrity_v1.contract_audit run
python -B -m r2.pilots.bench_flow_integrity_v1.trajectory --run
python -B -m r2.pilots.bench_flow_integrity_v1.preflight --manifest experiments/r2_bench_flow_integrity_diagnostic_v1/audits/source_contract/probe_manifest.json
python -B -m r2.pilots.bench_flow_integrity_v1.dispatch --manifest experiments/r2_bench_flow_integrity_diagnostic_v1/audits/source_contract/probe_manifest.json
python -B -m r2.pilots.bench_flow_integrity_v1.analyse_probes --manifest experiments/r2_bench_flow_integrity_diagnostic_v1/audits/source_contract/probe_manifest.json
```

源码/模型/数据均未改原版本。轨迹图中的COVERAGE_FINAL与probe中的COVERAGE_S1500是同一个global19500冻结权重的显示别名。完整逐pair表约50MB；未复制大权重或训练原数据。
