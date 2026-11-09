# Delivery index

- [16×3 全量图库](gallery/index.html)、[逐例 CSV](per_output_metrics.csv)、[全部父均表](aggregation/per_parent_metrics.csv)。
- [父配对](aggregation/paired_parent.csv)、[噪声配对](aggregation/paired_noise.csv)、[父摘要](aggregation/paired_parent_summary.csv)。
- [协议](protocol.json)、[projection config](projection_config.json)、[模型与输入manifest](registration/model_input_manifest.json)、[RAW身份审计](registration/raw_identity_audit.json)。
- 每个 outputs/{request_id}/ 保存实际 raw.npz、mesh.ply、result.json；修改输出另含 intervention.json 和 geometry_arrays.npz。
- [账本](accounting.json)、ledger/calls.jsonl、tests/project/、[端点执行摘要](endpoint_summary.json)、[图库帧审计](gallery/frames.json)。
- [质量成本紧凑表](quality_cost.csv)、[实际源码hash](source_hashes.json)、[代码diff](code.diff)、[保护依赖审计](audit/protected_dependencies.json)、[全部制品清单](artifact_manifest.json)。
- 实际新源码及旧评分依赖位于 source_snapshots/new 与 source_snapshots/dependencies；[恢复入口](README_RESUME.md)。

