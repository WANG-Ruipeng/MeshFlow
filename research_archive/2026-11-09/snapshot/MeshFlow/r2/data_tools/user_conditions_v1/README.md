# R2 用户式条件数据准备（CPU）

本目录是独立的数据准备工具，接续 R2 配方；不修改已有 R2 训练/采样入口，不加载模型。
实际主清单在：

`E:/MeshFlow-Control-artifacts/experiments/chair_r2_user_conditions_v1/active_conditions.json`

117 条 train（25 父、47 区域）+ 6 条新 DEV。原129份导出和全部图片保持，6条小档细边条作为HOLD证据。26原训练父中的400a因座/背均不能可靠分区而不进共同池，不能解读为“过滤掉有缺陷父”。语义为AI_PROXY而非人工标注。

在已有WSL NumPy/Matplotlib环境中检查或恢复：
```bash
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B /mnt/e/MeshFlow/r2/data_tools/user_conditions_v1/run.py --stage check
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B /mnt/e/MeshFlow/r2/data_tools/user_conditions_v1/run.py --stage resume
```

总入口核验现有来源、全部NPZ和PNG及活动清单身份，写入ledger.jsonl。resume仅补建缺失条件构造和条件质量分区；已有同身份结果复用。缺失语义审阅、活动选择或最终审计时明确BLOCKED，需要从对应保存证据恢复；不会自动猜标签、重抽噪声或开训。JSON阶段文件默认不可用不同内容覆盖。

实际阶段源码：
- prepare.py / inventory.py：来源、ID与几何候选；候选不是语义标签。
- quality.py：调用保留的原CPU面对分类器，复用26父既有缓存并检查8DEV；不是重新评分模型输出。
- freeze_regions.py：记录本次逐父实际图审决策，实例专用，不是可泛化的自动语义分割器。
- construct.py：完整原边图上嵌套区域、去重、原FP32的C/Y及映射。
- render.py：真实来源网格固定相机的CPU图，不使用生成图。
- review_usability.py：本次6条细边条图审HOLD决策，保留原导出。
- verify_export.py / verify_active.py：独立几何、映射、来源和活动子集回读检查。
- summarize.py / publish.py：全部候选表及最终主池表、图集、报告。

查看结果优先打开输出目录 gallery/index.html、DATA_PREPARATION_REPORT.md、dataset_handoff_active.json。初始conditions.json/summary/handoff仅是122train+7DEV候选阶段，不能误用作后续主池。

后续训练仍走R2（Native/T1+role+Geo、FM、全程HYBRID=.25），拟从主实验完整step6000及对应AdamW建立新的1000步续训分支。当前R2公共入口绑定旧26父/C20/C40/6000步，不直接支持这个清单；后续实现需独立分支合同。该目录没有训练授权、训练入口、权重或新噪声。C区域标签不输入模型，Y是来源补集，不是唯一正确生成答案。

独立审计不是“全网格干净”认证；主池仍有源几何CF/FF问题与UNKNOWN。后续评价应同时看C实际用途、整体形状与几何代价。
