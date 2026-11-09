# CHAIR_CONDITION_RECIPE_FACTORIAL_V1 交付索引

先读 [最终实验报告](EXPERIMENT_REPORT.md)，再读 [模型与检查点索引](MODEL_INDEX.md)。后者区分历史 START、R1–R4、两课程和独立复验的10个生成器实例；同名配方在 confirm 阶段不是原权重续训。

## 完整逐例结果

- [640个独立模型输出的完整原字段表](analysis/recipe_factorial_delivery_tables_v1/complete_per_output_640.csv)：用 evidence_phase/model_instance_id 区分阶段与权重。
- [参考诊断与NA完整表](analysis/recipe_factorial_delivery_tables_v1/reference_A20_per_output_all.csv)：保留320行A20诊断域和320行A40显式NA，不将来源重建用作开放补全晋级门槛。
- [主阶段逐输出](phases/main/evaluation/per_output.csv)、[课程逐输出](phases/course/evaluation/per_output.csv)、[复验逐输出](phases/confirm/evaluation/per_output.csv)。
- 主分析、课程分析、全历史基线共同表和独立复验分析均位于 analysis/ 对应子目录；其中包含全部逐父/逐噪声配对、分层、尾部、留一父和I分项，而非只保留有利列。

## 实际图片与原始观察

完整原PNG在本地以下图集。审阅ZIP只带索引、SHA和全部逐例观察，不包含这些PNG；解压ZIP内HTML没有原图时，请打开本地规范路径。

- 主实验：E:/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1/phases/main/gallery/INDEX.html
- 课程同噪声三方对照：E:/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1/comparisons/curriculum/gallery/INDEX.html
- 独立复验：E:/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1/phases/confirm/gallery/INDEX.html

[主初评](audit/visual_review/main_complete/per_output.csv)、[课程初评](audit/visual_review/course_complete/per_output.csv)、[复验初评](audit/visual_review/confirm_complete/per_output.csv) 保留原始AI观察及不确定性。课程包含对同64个R2结果的再次对照观察；704条观察记录对应640个独立模型输出，不把重复查看当新样本。这不是人类可用性研究。

## 执行、协议与代码

- [最终执行账本](audit/final_execution_accounting/ACCOUNTING.md)；JSON和逐阶段CPU/GPU资源表在同目录。
- [协议澄清](PROTOCOL_CLARIFICATIONS.md)：条件候选筛选并非严格同一规则，不能把完整池配方差异称为纯比例因果效应。
- registrations/ 保留实际输入、源码和阶段授权登记；source_snapshot/ 是原冻结源码，source_external_snapshot/ 是必要历史评分依赖及来源。
- additional_source/ 仅在审阅ZIP中用于收集后续适配器源码；每份实际来源/哈希见 BUNDLE_MANIFEST.json。code.diff 表示完整文本相对 /dev/null 的新增，不冒充未知Git基线的差异。

权重、RAW、查询数组和大曝光日志保留本地，路径/哈希在各绑定与审阅包manifest中。默认START不因本报告自动切换，本轮没有自动commit/push。
