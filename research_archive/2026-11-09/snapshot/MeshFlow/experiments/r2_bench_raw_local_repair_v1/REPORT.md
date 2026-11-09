# Bench：真实 RAW 局部修复目标试点

实验：`R2_BENCH_RAW_LOCAL_REPAIR_V1`；2026-10-24。

**局部坐标修复能消除部分真实穿刺，但这版数据尚不足以启动后端训练。**

固定12请求中，6份获得非空EDIT，低于事先冻结的至少7份门槛。完整处理链的保守数值资格为4/12；新增最终修复集中在91e两份输出，伴随未获得对照后处理新增的5条合法F/F连接。没有为了启动训练而扩大半径、替换样本或把UNKNOWN当监督。

这次实际制作了固定12份候选目标、逐角点状态和全部失败提案，完成原GEO及完整后处理配对评分和真实图审；没有把脚本交回后停止。数据阶段模型调用、训练、生成、D0前向均为0。这里的12请求按已知现象选出，只是开发试点，不估计一般成功率。

## 1. 实际检验的问题

RELDEF500相对SHAPE500没有新增严格视觉正例，并存在DEV FF代价。本轮结束该loss配置，转而检验：能否从真实生成的RAW中，制作有明确局部几何依据、在现有同槽坐标输出和后处理合同下仍有效的修复监督。

六父为a8、27、838c、91e、fd355、2d，每父固定n0/n1，不换样本、不重抽噪声、不借来源完整物体规定唯一正确款式。已有RAW、官方/SHAPE/RELDEF提议与BASE/FINAL均只读。48份BASE的顶点、槽位与角序对应核验通过，481个原缓存由hash保护。

最接近历史R05有界锚定、R27有界协调恢复及SHAPE/RELDEF保形。本次不同之处是建立真实RAW局部监督对和UNKNOWN掩码；没有扩大旧焊接半径，也没有将某个低FF后端自动当真值。

## 2. 冻结的唯一修复规则

在所有12请求上使用同一规则，从原有限三角面检测确认的非法C/F面对出发，仅尝试座面上表面附近的侧壁顶沿小幅穿刺：C局部法向朝+Y且Y分量≥0.7；自由面近直立，法向绝对点积≤0.5；有稳定下侧角点及1—2个仅小幅穿出的顶角。

采用**原GEO_ONLY真实角点簇**，把同簇的所有free角点赋到有限C表面最近点；不自造半径簇，不只拉一个角点。每个成员相对原RAW的位移≤0.005L，L=9.2081382646。所有簇成员、指定要消除的CF面对、受影响面和拒绝理由逐项保存。C、N、槽位、原RAW角序固定。

每个patch只尝试一次。要求指定非法关系真正消除、没有新增CF/FF或UNKNOWN、原合法边不丢、面积比处于[0.8,1.25]、法向变化≤30°，不得移动KEEP。失败回滚并保留提案，不扫描更大半径或换一种算法挑最好结果。+Y规则仅适用于这12份已审座面输入，不宣称是通用修复器。

这是一种有限的关系投影，不解决缺腿、重复主体、任意错层、删面或增面。局部目标仍可包含UNKNOWN区域；它不是完整无缺陷GT。

完整合同：[PROTOCOL.md](PROTOCOL.md)、[strategy.json](targets/strategy.json)。

## 3. 先补齐了同处理链对照

原GEO_ONLY只做到RAW→GEO/来源定向，没有后续HINGE/ALPHA/HOLE。若直接比较新目标与官方D0 FINAL，就会把省去D0本身的收益算给本轮编辑。

因此预先登记了必要CPU对照：RAW_IDENTITY与LOCAL_REPAIR使用同一GEO和HINGE_OPP015→ALPHA→HOLE链；RAW_IDENTITY复用原GEO缓存，不重新生成。其BASE的12×177=2,124个字段复算与原缓存精确相等。两臂共48行原始评分，24条同阶段逐输出配对，完整保留177字段、分母及NA。

**主要编辑增量是LOCAL_REPAIR−RAW_IDENTITY。** 官方、SHAPE、RELDEF仅作历史参照。目标不再送D0一次。

见[CPU_CONTROL_ADDENDUM.md](CPU_CONTROL_ADDENDUM.md)、[完整逐输出指标](scores/per_output.csv)、[配对差值](scores/paired_per_output.csv)。

## 4. 冲突归属复算

SHAPE→RELDEF的FF新增91对中，62对发生在共同保留的原RAW槽位之间，29对涉及新增保留的面；另有54对共同保留面的旧冲突消失。不能把全部代价归因于“多保住了面”。面被删掉后的面对不再可观察，也不当作几何修复。

| 类别（原RAW面身份） | 面对数 |
| --- | --- |
| CF\|COMMON_RETAINED\|ILLEGAL_BOTH | 176 |
| CF\|COMMON_RETAINED\|NEW_ILLEGAL_COMMON_FACES | 3 |
| CF\|COMMON_RETAINED\|RESOLVED_COMMON_FACES | 6 |
| CF\|NEWLY_RETAINED\|ILLEGAL_NEWLY_RETAINED_FACES | 3 |
| CF\|NO_LONGER_RETAINED\|NO_LONGER_OBSERVABLE_FACE_REMOVED | 1 |
| FF\|COMMON_RETAINED\|ILLEGAL_BOTH | 249 |
| FF\|COMMON_RETAINED\|NEW_ILLEGAL_COMMON_FACES | 62 |
| FF\|COMMON_RETAINED\|RESOLVED_COMMON_FACES | 54 |
| FF\|NEWLY_RETAINED\|ILLEGAL_NEWLY_RETAINED_FACES | 29 |
| FF\|NO_LONGER_RETAINED\|NO_LONGER_OBSERVABLE_FACE_REMOVED | 4 |

这张表描述面身份和冲突的转移，不单独证明形成冲突的动力学原因。逐面对记录：[conflict_attribution.csv](data/conflict_attribution.csv)。

## 5. 实际目标与后处理存活

构造得到56个通过局部检查的patch、23个拒绝patch，共229个EDIT角点，覆盖6/12请求。下面的“额外FINAL修复”只计原RAW_IDENTITY FINAL仍有指定冲突、而目标FINAL保留相关原面并消除冲突的patch；原GEO已经解决的关系不记为新增收益。

| 请求 | CF候选种子 | 接受/尝试patch | EDIT角点 | 自动KEEP角点 | 经BASE/FINAL存活patch | 额外FINAL修复patch | 对照FINAL连接未保留 | 数值局部资格 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a8e987/n0 | 62 | 9/11 | 36 | 0 | 9 | 0 | 0 | True |
| a8e987/n1 | 65 | 7/9 | 81 | 0 | 7 | 0 | 0 | True |
| 27eb79/n0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | False |
| 27eb79/n1 | 1 | 0/1 | 0 | 0 | 0 | 0 | 0 | False |
| 838c48/n0 | 48 | 3/5 | 36 | 0 | 3 | 0 | 0 | True |
| 838c48/n1 | 31 | 5/7 | 18 | 0 | 5 | 0 | 0 | True |
| 91e169/n0 | 25 | 8/11 | 14 | 0 | 8 | 1 | 1 | False |
| 91e169/n1 | 87 | 24/35 | 44 | 0 | 24 | 16 | 4 | False |
| fd355d/n0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | False |
| fd355d/n1 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | False |
| 2d1e5e/n0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | False |
| 2d1e5e/n1 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | False |

56个接受patch的指定关系在真实BASE/FINAL中均保持消除，相关原面均保留。但**4/12满足无新增冲突、无对照合法边代价的数值资格**（a8与838c各两噪声），它们的指定关系原GEO也能解决。真正多修复了FINAL指定关系的是91e两噪声（1与16个patch）；它们没有新增非法CF/FF，但相对对照FINAL少了1与4条合法F/F共边。

独立索引核查确认：这5条边在RAW、IDENTITY_BASE、TARGET_BASE都没有合法共边，相关面始终保留。**这是没有获得identity后续处理本来会新增的连接，不是撕裂RAW/GEO既有共边，也不是通过删面降低冲突。** 所以局部修复有几何正证据，却还不满足完整链下无连接代价的保守准入。见[实际共边配对账本](analysis/paired_original_legal_edges.csv)和[独立拓扑审计](audits/independent_topology_audit.json)。

拒绝理由允许一patch同时触发多个条件，所以各行之和不是拒绝patch数：

| 拒绝理由 | 出现次数 |
| --- | --- |
| AREA_RATIO_OUTSIDE_LIMIT | 6 |
| CF_NEW_UNKNOWN_FIXED_GEO | 1 |
| CF_NEW_UNKNOWN_RAW | 2 |
| CLUSTER_MEMBER_DISPLACEMENT_EXCEEDED | 2 |
| DESIGNATED_CF_NOT_RESOLVED | 1 |
| FF_NEW_ILLEGAL_FIXED_GEO | 7 |
| FF_NEW_ILLEGAL_RAW | 12 |
| NORMAL_TURN_EXCEEDED | 2 |

原自动KEEP规则要求已审a8/27的完整下侧支撑面同时满足局部位置和无已检冲突条件。若自动KEEP为空，代表此规则没有认证出KEEP，不代表RAW没有正确支撑；未认证角点仍是UNKNOWN，未偷偷作为identity监督。后续人工标注如有，保留为独立版本，不覆盖原mask或伪造通过门槛。

实际补充图审在a8/27四份输出中标注了**12个支撑下端KEEP角点**（每份a8四点、每份27两点），保存完整RAW slot/corner/坐标与8张已查看图的hash。只认证局部支撑端点位置，不认证整张面的自交或整物质量。它们另存于[manual_KEEP_annotations.json](visual_review/manual_KEEP_annotations.json)，没有修改冻结target的UNKNOWN状态和loss_mask；也不把KEEP-only的27变成EDIT正例。

数据：[逐请求](analysis/per_request.csv)、[逐patch存活](analysis/patch_post_survival.csv)、[KEEP/EDIT面存活](analysis/KEEP_EDIT_slot_survival.csv)、[原面对净进出](analysis/paired_illegal_original_faces.csv)。每个`targets/<request>/target.npz`含真实候选坐标、原RAW、KEEP/EDIT/UNKNOWN/C状态、loss_mask及角点身份；`proposals/`保留拒绝候选。

## 6. 同链FINAL质量

CF/FF为受非法关系影响的free面比例；I沿用原ConflictCov_C支持域/查询定义。面数、非法面对数和完整原字段仍在CSV中，不能只根据百分比判断改善。每父两噪声，表中父均值；全组为六父等权。

| 父 | 同链方法 | CF % | FF % | I % | 有效free面 | 合法C/F边 | 单面孤岛 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 全组6父 | RAW_IDENTITY | 3.4299 | 24.5781 | 1.5340 | 149.8333 | 11.1667 | 17.0833 |
| 全组6父 | LOCAL_REPAIR | 2.3159 | 24.4546 | 1.5055 | 149.8333 | 11.2500 | 17.0000 |
| a8e987 | RAW_IDENTITY | 0.0000 | 16.5257 | 0.0000 | 147.5000 | 29.5000 | 4.0000 |
| a8e987 | LOCAL_REPAIR | 0.0000 | 16.5257 | 0.0000 | 147.5000 | 29.5000 | 4.0000 |
| 27eb79 | RAW_IDENTITY | 3.2119 | 8.4439 | 0.3418 | 171.5000 | 6.5000 | 17.5000 |
| 27eb79 | LOCAL_REPAIR | 3.2119 | 8.4439 | 0.3418 | 171.5000 | 6.5000 | 17.5000 |
| 838c48 | RAW_IDENTITY | 0.3704 | 33.1012 | 0.0000 | 134.5000 | 20.0000 | 11.0000 |
| 838c48 | LOCAL_REPAIR | 0.3704 | 33.1012 | 0.0000 | 134.5000 | 20.0000 | 11.0000 |
| 91e169 | RAW_IDENTITY | 10.0861 | 12.8165 | 1.7822 | 132.0000 | 11.0000 | 19.0000 |
| 91e169 | LOCAL_REPAIR | 3.4022 | 12.0758 | 1.6113 | 132.0000 | 11.5000 | 18.5000 |
| fd355d | RAW_IDENTITY | 6.0385 | 26.3273 | 4.4189 | 140.5000 | 0.0000 | 25.5000 |
| fd355d | LOCAL_REPAIR | 6.0385 | 26.3273 | 4.4189 | 140.5000 | 0.0000 | 25.5000 |
| 2d1e5e | RAW_IDENTITY | 0.8723 | 50.2540 | 2.6611 | 173.0000 | 0.0000 | 25.5000 |
| 2d1e5e | LOCAL_REPAIR | 0.8723 | 50.2540 | 2.6611 | 173.0000 | 0.0000 | 25.5000 |

RAW级修复与GEO后的增量分开记录。某个指定关系虽然在原始坐标上被消除，但若原GEO已经会消除它，就不能宣称本轮提升了最终质量。被处理链删去的指定面也不算成功。

全组FINAL的CF为3.4299%→2.3159%，FF为24.5781%→24.4546%，I为1.5340%→1.5055%；有效free面数均为149.8333/输出。按原面身份，FINAL消除28对CF与1对FF，0新增非法原面对，改善不是多删面得到的。收益集中在91e，不能据全组均值声称六父普遍提升；它与上面的5条最终连接机会损失需要同时报告。

## 7. 实际图审与训练决定

图册使用同相机、原截面，并补两侧及支撑处截面；包含完整目标、状态叠加、接受与拒绝patch局部放大，以及RAW_IDENTITY FINAL对LOCAL_REPAIR FINAL的直接对照。真实评阅标为非盲AI_PROXY，不能当作独立人类评价。

已实际查看固定12请求的多视角、同位置截面、支撑截面、状态图和局部提案图。整体形状变化很小，当前图片不足以把微小编辑判为明确的视觉升级；2d的宏观错层未解决。a8/27的已有细支撑保留有价值，但应区分绕过D0的保形收益与本轮编辑增量。非盲AI_PROXY逐例记录与少量独立KEEP角点注释保留在visual_review；后者不覆盖原目标mask、不增加EDIT请求数。

冻结的数据准入为至少7/12有真实可信EDIT并通过原后处理及结构保护；KEEP-only、UNKNOWN占位或因为省掉D0而保住结构均不计为编辑正例。非空EDIT上限已只有**6/12**，数值资格进一步为**4/12**；图审无法把其余未修复样本变成合格训练对。决定：**NO_TRAINING_DATA_GATE_FAILED**。这不是按结果追加了一个门槛：即使将全部6份候选都算合格，仍未达到事先冻结的7份。

没有启动训练、前向或新生成，没有新增权重。用户授权的是值得训练时继续；这版数据未通过已冻结的准备门槛，所以保留真实局部对与反例后收口。

本次12请求涉及的6个开发父不能一边进入训练一边宣称后端未见。只读盘点还确认，旧SHAPE16个后端训练父没有现成同一Coverage模型的RAW缓存；其他历史模型的18份RAW不具有相同生成分布身份，不能直接混作本轮训练数据。若将来准备新训练池，需要独立冻结父、RAW与修复标签。

这次门槛的结果仅约束这项具体修复规则与这版数据；不证明所有真实RAW局部修复都不可行，不证明D0存在理论上限，也不把所有残余错误统一判成必须删面或必须重新生成。

## 8. 实际执行、工程记录与恢复

构造CPU时间约283.67秒，原CF/FF检测器累计调用数按账本为292（一个全网格扫描含CF与FF两项）。评分资源：

| 阶段 | CPU workers | 墙钟秒 | worker阶段秒之和 | 峰值RSS MiB |
| --- | --- | --- | --- | --- |
| identity | 3 | 152.68 | 385.00 | 197.73 |
| target | 3 | 188.33 | 437.83 | 221.60 |

并行时间不能和单进程时间当作相同计算基准。初始资源为RTX5080、16,303MiB显存、24逻辑CPU、约61GiB可用RAM；GPU在数据阶段没有任务进程。

构造初次在预检发现RAW模型角序与导出来源角序不同，**在0候选、0检测器、0模型调用处停止**。原失败和源码保留；用已保存排列的逆映射修复索引账本后明确恢复，12份逐位往返验证通过，没有重放科学计算。target的source_corner_ids是canonical identity，旧导出排列另存input_export_source_corner_ids；原GEO映射不变。

另有91e/n1的patch033使用无GEO映射的−1标识，被构造为一个无效提案；它在位移/面积/法向预检即遭拒，未执行该提案的检测器或解码，也未贡献目标坐标。独立补充将其明确列为UNMAPPABLE工程失败：可解释的真实簇候选为78个（56接受、22拒绝），另有1个无映射失败；没有偷偷删掉日志或把它当物理不可修的证据。

渲染时新增支撑截面触发旧显示函数仅接受三个固定索引的限制。保留失败账本和已生成页面后，仅修正显示索引的调用，真实截面位置不变，随后完成图册；没有重算目标、评分或模型。

本阶段总计：12新目标GEO、12复用identity GEO、24完整cascade、48完整评分；0训练、0新权重、0生成、0D0、0新噪声、0GPU调用。原资产hash保持；无清理、默认切换、commit或push。

可恢复总入口（已完成阶段只做hash验证，不静默覆盖半成品）：

```powershell
wsl.exe -d Ubuntu -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_raw_local_repair_v1.run verify
```

数据/目标/原链评分/渲染/分析分别可用`assets`、`targets`、`ready`、`evaluate`、`render`、`analysis`子命令。实际图审和训练准入不会由总入口自动伪造。源码与原评分代码归档、code.diff、源hash及独立审计随本目录交付。

## 9. 下一步的证据边界

本轮应保留的正证据是：小幅、明确的真实C/F穿刺可以用同槽坐标修复，且未靠删面改善冲突。它尚不能支撑普遍的后端训练收益；安全通过的a8/838c修复多为原GEO已能做到，91e的额外收益又与下游连接竞争。因此暂不追加RELDEF或训练这版自动标签。若继续投入，应先取得直接改善最终残余缺陷、且保留同链连接的可信修复目标；对于2d式主体错层，仍应放在生成阶段的条件采用问题中处理。当前证据不支持把所有失败都转成删面/增面需求，也不构成D0或坐标修复的理论上限。

当前任务在上述数据与评价交付后收口。未自动延长RELDEF、调loss权重、扩大吸附阈值或切换默认后端。
