# Bench多部位实验交付与图审检查表（草稿）

状态：**准备清单；不是全部项目已完成的声明。** 创建时第二次5250→5362尾段重放仍待授权。所有完成状态应由实际receipt/completion核对，不从预期路径存在推断成功。

|交付项|位置／内容|当前边界|
|---|---|---|
|冻结合同|`registration.json`、全部输入与代码hash|已冻结；恢复工程变化另列，不覆盖科学合同|
|数据与真实曝光|`data/manifest.json`、`data/input_schedule.json`、训练`exposure.jsonl`|316条已准备；最终实际曝光汇总PENDING|
|固定评测C与噪声|`evaluation_inputs/seed_plan.json`、`manifest.json`、条件／metadata／source／noise NPZ|已冻结；模型只读C，不读Y或来源owner|
|固定监督探针|`fixed_loss_probe/manifest.json`及噪声、逐checkpoint JSON|早期已存在；必须补齐到实际终点|
|正式权重与恢复状态|`training/COMMON`、`training/PLAIN`、`training/SEAM`，模型／优化器／RNG、SHA|COMMON可恢复5250；分叉与最终权重PENDING|
|训练曲线与有限停止判断|逐更新loss、每1000步probe、逐臂平台检查、共同停止原因|最终PENDING；封顶不冒称收敛|
|RAW／POST及中间处理产物|每标签每请求的raw、D0 proposal、pre_geo、post_canonical、来源定向post及调用回执|NATIVE_S2000过程8请求已完成；正式完整配对PENDING|
|完整124字段及配对表|每标签score、`evaluation/tables`的per_output/per_condition/per_parent/paired_per_output等|过程16cell已完成；正式48请求／臂PENDING|
|角色面对与删面细节|`illegal_pairs_by_role.csv`、`deleted_faces.csv`，保留分母与缺失理由|随对应真实评测生成；不能只保留汇总|
|原评分代码和独立wrapper|`scoring_source_snapshot`、`evaluation.py`及hash|不改旧定义；正式报告列实际版本|
|完整画廊和截面|固定每请求RAW/POST同尺度，两个表面视角＋前侧有限截面|过程8页已完成；后续按到达checkpoint执行|
|实际AI_PROXY图审|`visual_review/<label_or_comparison>/`，jsonl/md、图片hash、不确定原因|NATIVE_S2000已实际8页／16阶段记录；最终PENDING|
|资源与真实物理计算账本|`dispatch/`、训练／采样／D0ledger、恢复会计|最终合并PENDING；不能只报唯一更新游标|
|故障及恢复证据|两次原始尾段快照、授权、CPU核验、重放字段比对、工程diff/hash|第一次恢复已验证；第二次重放授权与验证PENDING|
|最终报告|由`REPORT_DRAFT.md`填入全部真实结果与限制|PENDING；禁止把本草稿更名冒充完成|

## 图审执行顺序

1. 只用冻结manifest列出的请求，先核对图像hash与模型／噪声／RAW或POST身份；不得按数值或美观挑页。
2. 过程页全部实际查看并独立标记，正式12000、15000及实际终点使用完整48请求。相同实际终点不重复采样或伪造第二份独立证据。
3. 逐请求、逐阶段记录主体完整性、C合理参与、构造角色诊断、错层／重复层、薄支撑和悬片外观。组合C分别写座面及支撑表现，避免用单一成功标签掩盖局部问题。
4. 每张图明确AI_PROXY与是否非盲；允许UNKNOWN/UNCERTAIN。不能以metric自动填图审标签，也不能把未真正查看的图记为已审阅。
5. 先形成逐图记录，再汇总逐条件、逐父、TRAIN/DEV及部位结果；读者应能定位相反代价。图像不足以判断共享索引或三维断片时，保留未知并引用独立数值字段，不能互相冒充。
6. 图审固定训练合同；结果不授权重新抽噪声、换父、改定义、追加checkpoint或挑最好看的模型。

## 表与口径检查

- 监督loss使用ΣSSE/Σ有效标量；生成质量使用已登记的噪声→条件→父等权，并保存输出级分布。两者不能互换。
- 124字段完整保留，包括不改善的FF、I、删面或碎片；每项给有效／注册分母及NA原因。
- 闭合C无开放接缝时相关字段NA；开放C预测全Null仍为零接续。不能用NA让失败消失。
- `SOURCE_FIXTURE`仅为工程预检，不能混入RAW/POST生成评价。
- D0提议、硬恢复及GEO_C产物应保留；视觉支撑消失时，不能只以冲突下降宣布后处理获益。
- 既有早期probe材料只作过程解释，最终需覆盖全部到达步骤；DEV不选停止点或best checkpoint。

## 恢复和完成状态

第一次额外77更新已执行并验证重放。第二次112更新已执行但未进入5250持久化状态，重放仍待授权；如获准完成，将对原112条更新和896曝光进行相同比对，并将总恢复额外更新189单独报告。没有对应验证回执时不得填“已恢复”。

最终若达到共同平台，写明有限规则通过而非数学收敛；若到达各分支9000步上限仍未平台，写`BUDGET_EXHAUSTED_NOT_CONVERGED`并停止。不得自动延长、切换默认模型、commit或push。
