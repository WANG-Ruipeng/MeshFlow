# CHAIR_CONTROL_ATTRIBUTION_REVIEW

先读 ATTRIBUTION_REPORT.md；全部数字从原chair_domain_control_v1逐例raw/CSV/评分JSON导出。

- width_per_output.csv=216行；width_per_curve.csv=72；noise_paired.csv=24。
- cross_matrix.csv=648个原单元；cross_direction_audit.csv保留旧行解释和正确列比较；no_response_matrix.csv只是原F1三行重复的解析正控，不是新评分。
- coverage_partition.csv=656行（164配对×4组）；coverage_paired.csv/coverage_per_parent.csv保留接口、角质量、pair数和受影响面比例。coverage_vectors和coverage_pair_vectors含实际固定Q坐标和逐点数值。标签0/1/2/3=both_in/enter/leave/both_out。
- pullback_per_case.csv=96个离线候选；pullback_cache每候选仅两类距离，包含真实采样点和平方距离。原模型输出未改写。
- evidence_arrays含144份原B终点free/C/GT/Q及hash。original_tables保存原B逐例三表、teacher、A、原协议和清单。
- original_scoring_code为实际相关评分源码全文及trimesh最近点依赖源码，不是伪代码；source_code_hashes.json提供来源和hash。analysis_code与code.diff仅为独立新增代码。
- source_manifest.json与source_hash_verification.json记录源前后hash；每个打包文件另列package_manifest.json。
- 恢复命令见restart.txt；源数据在本机原位置。完成阶段和候选会校验hash后复用，不重复冒充新证据。

CSV中的距离/L是fraction；原bbox_pct是%L；coverage fraction乘一次100才是百分点。NA不等于0。教师与生成输入不同，不能当逐状态因果配对。所有模型/训练/OT/CUDA调用为0。
