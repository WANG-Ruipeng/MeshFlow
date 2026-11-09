**Chair Control Usability Quick Pilot V1：小于10MB的云端复核包**

本包按用户要求，只保留报告、完整逐输出/逐父CSV、实际评分源码、协议、身份及关键审计。前一版67.5MB完整包已删除；原实验数据仍在本地保留。

建议读取顺序：

1. results/QUICK_CONTROL_USABILITY_REPORT.md 和 task_contract.txt：本轮目标、真实执行账本、定义、完整结果与证据边界。
2. results/per_output_metrics.csv：全部80个唯一输出的完整标量指标，包括主面板64项及嵌套附加16项；两面板重合8项不重复计数。
3. results/per_condition_summary.csv、paired_parent_summary.csv：候选内均值、固定同winner的GT-oracle Best@2/4、全部8父配对分布。
4. results/scoring/primary_summary.csv、diversity_pairs.csv、sensitivity_checks.csv：配对均差/中位差/改善父/最差父、完整96候选对、固定4项加密复核。
5. results/multi_context_summary.csv、scoring/multi_context_global.csv、registration/conditions.json：2父嵌套条件、面积/片数/free预算，以及同一公共未观测区域的公平比较。
6. results/source_code/new/scoring.py、aggregation.py 和 source_code/original/：实际执行的新旧评分代码；定义、源hash和工程diff一并保留。

所有CSV保留完整行与列，没有只保留汇总或有利案例。包内另含请求登记、模型身份、调用账本、独立评分审计及空人评表。主要数值仍是HYBRID平均D下降11.70%但仅4/8父改善、F@1%下降、接口RMS改善而细容差贴合混合；Best@4不是部署选择器，也不是人类可用率。

**本精简包不含：** 权重、raw、NPZ/NPY、源网格、Gaussian数组、逐点距离向量、图库图片及HTML。原报告和代码保持原文，里面的绝对路径以及指向这些省略项的链接仅供本地追溯，不能在本包中直接打开。附带 omitted_materials.json 明确记录范围。本包支持标量结果/分母/汇总与代码定义复核，不支持重新计算原始几何距离或重新看图。不要假称已检查被省略的原始数据。

AI_PROXY文件只是上一轮实际看图后的定性记录；云端本次没有收到图片，不能把它当独立视觉核验或人工评阅。人工U_single/U_at4仍为NA，user_study=NOT_RUN。空表不是已执行的研究结果。

无需新增训练、模型前向、生成或重新评分。唯一优先建议仍待云端依据现有标量证据判断：先组织真实盲评，检验候选潜力能否转成用途收益，还是现有失败已足以改变下一步优先级。

bundle_manifest.json记录全部文件字节数和SHA256。文件已在ZIP内逐项核验；解压后运行 python verify_bundle.py 可再次只读校验。
