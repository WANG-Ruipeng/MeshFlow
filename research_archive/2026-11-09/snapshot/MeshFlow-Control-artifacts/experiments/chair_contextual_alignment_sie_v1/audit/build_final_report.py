"""Build final report from completed saved evidence; zero model/scoring calls."""
from pathlib import Path
import csv,json
R=Path(__file__).resolve().parents[1]
def js(p):return json.loads((R/p).read_text(encoding="utf-8-sig"))
def cr(p):
 with (R/p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def n(x):return float(x) if x not in ("",None) else None
def fmt(x,scale=1,d=3):
 x=n(x);return "NA" if x is None else f"{x*scale:.{d}f}"
def tab(head,rows):return "\n".join(["| "+" | ".join(head)+" |","|"+"|".join("---" for _ in head)+"|"]+["| "+" | ".join(map(str,row))+" |" for row in rows])+"\n"
labels=["START","A_FM_1000","B_FREE_ALIGN_1000","C_KNOWN_ALIGN_1000"]
short=dict(zip(labels,["START","A / FM","B / free","C / known"]))
contrasts=[labels[3]+"-"+labels[1],labels[3]+"-"+labels[2],labels[2]+"-"+labels[1],labels[1]+"-START",labels[2]+"-START",labels[3]+"-START"]
cs=dict(zip(contrasts,["C−A","C−B","B−A","A−START","B−START","C−START"]))
results=js("analysis/results.json");assert results["status"]=="COMPLETE" and results["scored"]==192 and results["diversity"]["complete_pairs"]==288
assert js("audit/training_endpoint_independent.json")["status"]=="PASS"
assert js("audit/call_closure_final.json")["status"]=="PASS"
review=js("visual_review/final_review_receipt.json");assert review["status"]=="PASS" and review["blind_output_rows"]==192 and review["reference_output_rows"]==128
summary=cr("sie_summary.csv");absolute=cr("audit/report_ready/P1_ALL_absolute.csv");pairs=cr("analysis/comparison_summary.csv");parents=cr("paired_parent_metrics.csv")
def s(label,cohort,metric,panel="P1"):
 return next(x for x in summary if x["label"]==label and x["cohort"]==cohort and x["metric"]==metric and x["panel"]==panel)
def p(con,cohort,metric,panel="P1"):
 return next(x for x in pairs if x["contrast"]==con and x["cohort"]==cohort and x["metric"]==metric and x["panel"]==panel)
def a(label,metric):return next(x for x in absolute if x["label"]==label and x["metric"]==metric)["value"]
parts=["# CHAIR_CONTEXTUAL_ALIGNMENT_SIE_PILOT_V1 实验报告\n",
"""实验已实际完成：三臂各1000步、160条新Euler50、32份可信START缓存、192份输出评分、288对多样性比较，以及完整图集与实际AI审阅。固定任务到此停止，默认模型保持START。

**结论：本轮不支持known侧对齐比free侧更有控制价值。** B相对普通FM续训有小幅、噪声敏感的参考接缝收益；C没有形成跨父、新噪声一致的接合增量。B/C均未通过预先冻结的开发性SIE筛选，不能以辅助loss下降替代control成功，也不能把此次适配结果推广成对完整CoAl方法的否定。

必须单列的局部代价：f596…在NEW两噪声下，C−A的I增加11.304个百分点；其n7单输出I由A的30.029%升到B的43.799%、C的52.051%。即B−A +13.770、C−A +22.021个百分点。ALL四噪声父均值没有超过10个百分点的I/E退化，不代表这些NEW和单输出代价不存在。

## 1. 真实账本、身份与完成性
"""]
parts.append(tab(["项目","实际执行/核验"],[
["优化器更新","A1000 + B1000 + C1000 = 3000；无训练重放"],
["生成器正式训练","24,000 forward / 24,000 combined backward"],
["辅助头正式训练","8,000 forward / 8,000 combined backward参与；后者与生成器同一次反传"],
["额外正确性检查","12生成器F / 10零更新B，其中6F为三个导出开/关capture检查"],
["辅助头额外图参与","预检4F/8B；稀疏aux观察6次穿过已有头图，无新增头前向"],
["稀疏梯度观测","FM6 + aux6 = 12次额外autograd.grad，均单独记账"],
["教师","32唯一完整训练网格、128图、8个batch forward；冻结、无教师反向"],
["新生成","160/160完整Euler50；8,000 NFE；失败0，额外rollout0"],
["缓存和输出","32可信START缓存 + 160新生成 = 192；缓存不重新评分"],
["计算/资格","192输出评分PASS；P1 S/I均PASS，E为32 PASS + 96 PARTIAL；P2的S/E 64份NA"],
["C保持","192输出的7项合同全部恰True，含51状态C、最终C、NPZ/PLY、N、实际Gaussian和有限性"],
["独立终点核验","94冻结源文件一致；三臂8000样本位置/1000批完全配对；6训练checkpoint与3导出PASS"],
["优化器身份","194原参数张量、130,401,155参数；原AdamW step5000→6000，B/C头step0→1000"],
["GPU累计持锁","5,133.034秒 = 85.551分钟，预算21,600秒；所有持锁区间闭合"],
["完成与停止","可恢复总入口COMPUTE_COMPLETE；实际AI审阅另行完成；human usability=NOT_RUN"]
]))
parts.append("""账本证据：[调用闭合](audit/call_closure_final.json)、[实际资源](audit/final_accounting.json)、[终点独立审计](audit/training_endpoint_independent.json)、[权重清单](checkpoint_manifest.json)、[训练日志](training_logs/manifest.json)。全轮生成器前向总数为32,012（训练24,000+采样8,000+预检/导出12）；反向路径调用为24,022（正式24,000+预检10+观察12），不能再加一次头的combined backward。

实际RTX5080总显存16,303MiB。5秒nvidia-smi设备级采样训练峰值A/B/C为6,177/6,256/6,441MiB，最低空闲9,539MiB；这些包含桌面占用，并非精确的单进程瞬时峰值。训练/教师/采样按任务单GPU worker运行，CPU评分实际2 workers、每worker2线程。协议准备资源上限曾记4个CPU评分worker，实际采用工具已限制的2个；没有改变查询密度、训练batch或科学定义。阶段持锁A/B/C分别1472.460/1553.412/1536.037秒，采样START/A/B/C分别58.248/149.071/143.582/150.683秒。嵌套调用计时不可相加当独立GPU成本。

## 2. 固定任务和方法边界

A=普通HYBRID-FM续训；B=第4层attention输出的free token对齐；C=同一输出的known token对齐。B/C仅选择域不同，头初始化/容量/教师/输入流/损失系数完全相同。捕获位于attention内部输出投影之后、外部gate/残差/角广播之前，hook不改输出。头3,934,720参数，只用于训练。教师与头不进入纯生成器推理依赖；严格加载导出且capture开关预测逐位一致。

所有臂继承同一START和原AdamW moments；batch8、micro1，8不同父，4原尺度+4连续宽度增强；FM沿用整batch有效free标量分母。仅4个alpha=1样本使用0.1×平均余弦损失，避免未变宽教师监督变宽目标。generator/head分别lr1e−5/1e−4、分别clip1，保持原BF16主干/FP32参数与积分和确定性路径。完整定义、教师revision/hash、恢复命令见[METHOD_AND_DEVIATIONS](METHOD_AND_DEVIATIONS.md)、[protocol](protocol.json)、[README_RESUME](README_RESUME.md)。

参考[Learning to Read the Contextual Tokens in Diffusion Transformers](https://arxiv.org/html/2610.06844v1)的§4/A.1.1/A.2；本轮是CoAl-inspired mesh pilot。未做空条件加权/LLM Reader，多视图mesh教师和alpha=1限制是本任务适配，不能称完整论文复现。

## 3. S/I/E主表

以下为百分数，S↑、I↓、E↑。S=REFERENCE_INTERFACE，E=REFERENCE_EXPOSURE。先父内噪声平均，再8父等权；OLD/NEW每父2噪声，ALL每父4噪声。NEW仍是相同DEV父上的新噪声，并非新父确认。E显示点值[保守下界,上界]；本面板S/I上下界重合。原始全精度、分母和PARTIAL/NA见[逐输出CSV](per_output_metrics.csv)。这些检测界不是统计置信区间。
""")
rows=[]
for co in ("OLD","NEW","ALL"):
 for lab in labels:
  ss,ii,ee=[s(lab,co,m) for m in ("SeamCov","ConflictCov_C","ExposedRetain")]
  rows.append([co,short[lab],fmt(ss["value"],100),fmt(ii["value"],100),fmt(ee["value"],100)+" ["+fmt(ee["lower"],100)+", "+fmt(ee["upper"],100)+"]"])
parts.append(tab(["噪声","模型","S %","I %","E % [L,U]"],rows))
parts.append("""## 4. 配对效果与冻结筛选

差值单位为百分点，正ΔI表示代价。所有比较使用完全相同的C和实际Gaussian。
""")
rows=[]
for con in contrasts:
 for co in ("OLD","NEW","ALL"):
  sp,ip,ep=[p(con,co,m) for m in ("SeamCov","ConflictCov_C","ExposedRetain")]
  rows.append([cs[con],co,fmt(sp["mean_delta"],100),fmt(ip["mean_delta"],100),fmt(ep["mean_delta"],100),fmt(sp["median_delta"],100),sp["positive_parents"]+"/"+sp["expected_parents"]])
parts.append(tab(["比较","噪声","ΔS","ΔI","ΔE","父ΔS中位数","S改善父"],rows))
parts.append("""B−A：ALL的ΔS +1.100pp，中位+0.219pp、5/8父为正；但OLD +2.227pp、NEW −0.026pp，新增噪声没有接缝增量。该NEW失败很小，应如实保留“小幅旧噪声收益”，不能改写成完全无效，也不能放宽已登记规则使其PASS。

C−A：ALL仅+0.195pp，中位−0.346pp、3/8父为正，NEW −1.075pp。C−B的S在OLD/NEW/ALL分别−0.762/−1.049/−0.905pp；ALL有5/8父下降、2/8上升、1不变。C相对B的E平均提高0.761pp，但I平均增加0.523pp；这是一种取舍，不能称known侧接合整体更强。

相对START，ALL平均S分别A +1.160、B +2.260、C +1.355pp，平均I均下降、E均上升；但S改善父仅A3/8、B3/8、C2/8，父中位都为负。部分共同改善来自普通续训，不能全部归给新增对齐。

""")
screens=js("audit/report_ready/screens_unchanged.json")
for screen in screens["screens"]:
 parts.append("### "+short[screen["candidate"]]+"：固定筛选 "+screen["status"]+"\n")
 parts.append(tab(["条件","点值","保守区间","阈值","判断"],[[x["name"],fmt(x["point"],1 if "parents" in x["name"] else 100), "["+fmt(x["lower"],1 if "parents" in x["name"] else 100)+", "+fmt(x["upper"],1 if "parents" in x["name"] else 100)+"]",x["operator"]+" "+fmt(x["threshold"],1 if "parents" in x["name"] else 100),x["status"]] for x in screen["criteria"]]))
parts.append("""两组均为明确FAIL，不是未执行或因未决界导致INCOMPLETE。规则是本轮开发筛选，不是文献标准/显著性检验/论文录用门槛。未使用GT MSE +5%作为开放式控制否决器。

## 5. 完整逐父配对与集中性

下列为P1 ALL，差值单位pp。父ID缩写只用于显示，完整ID/逐噪声/上下界均在[逐父CSV](paired_parent_metrics.csv)与[逐噪声CSV](paired_noise_metrics.csv)。
""")
for con in contrasts:
 rows=[]
 for uid in sorted({r["parent_id"] for r in parents if r["panel"]=="P1"}):
  vals=[]
  for met in ("SeamCov","ConflictCov_C","ExposedRetain"):
   rr=next(x for x in parents if x["panel"]=="P1" and x["cohort"]=="ALL" and x["contrast"]==con and x["parent_id"]==uid and x["metric"]==met);vals.append(fmt(rr["delta"],100))
  rows.append([uid[:8]+"…"]+vals)
 parts.append("### "+cs[con]+"\n"+tab(["父","ΔS","ΔI","ΔE"],rows))
parts.append("""B−A的S增量中，f596…贡献6.434pp的父均值变化，占8父总增量约73.1%；排除该父的描述性均值仍为+0.339pp，但低于预设+1pp。B的留一父S范围+0.339至+1.376pp，符号未翻转。C−A留一父范围−0.356至+0.577pp，排除f596…或866e…会翻负。C−B留一父范围−1.301至−0.695pp，方向一致。留一父不改变正式8父分母。

相对START的S均值主要由866e…和86ec…推动；不能围绕单一最好/最坏父讲故事。全部父和OLD/NEW结果保留，未删坏例。详见[留一父表](analysis/leave_one_parent_out.csv)。

## 6. I/E尾部、几何与参考距离成本

ALL父平均层面未出现I增加或E下降超过10pp；NEW子组f596…的C−A I增加11.304pp，属于必须披露的父级代价。单输出最差I：f596…n7，B−A +13.770pp、C−A +22.021pp、C−B +8.252pp。单输出最差E：866e…n0，B−A −7.227pp、C−A −7.080pp；C−B最差为bdd29…n1 −1.563pp。[全部尾部及保守界](audit/report_ready/control_cost_tails.csv)与[单输出极值](audit/report_ready/single_output_extrema.csv)均保留。

I必须按crossing和near-duplicate的并集理解，不能直接相加。P1 ALL的分解如下（%）：
""")
parts.append(tab(["模型","cross","near-dup","交集","I并集"],[[short[l],fmt(a(l,"ConflictCov_C_cross"),100),fmt(a(l,"ConflictCov_C_dup"),100),fmt(a(l,"ConflictCov_C_cross_and_dup"),100),fmt(s(l,"ALL","ConflictCov_C")["value"],100)] for l in labels]))
parts.append("""B/C相对A新增的I主要含near-duplicate覆盖变化；具体输出的几何位置仍需结合图像，不能只看此分解认定唯一机制。

P1 ALL几何诊断：
""")
parts.append(tab(["模型","CF对数","CF受影响free%","FF对数","FF受影响free%","小角<5°%","free/参考面积"],[[short[l],fmt(a(l,"CF_proper_pair_count")),fmt(a(l,"CF_proper_affected_free_fraction"),100),fmt(a(l,"FF_illegal_pair_count")),fmt(a(l,"FF_illegal_affected_free_fraction"),100),fmt(a(l,"free_minimum_angle_lt5_fraction"),100),fmt(a(l,"free_area_over_GT_free_area"))] for l in labels]))
parts.append("""四模型的精确重复面、零面积和near-degenerate面比例均为0；这不意味着没有近重复、交叉或薄长面。FF受影响free比例仍约90%，沿用旧定义，不能称论文全网格Ri。原始精确坐标C/free共边数均为0，因此几何接近或S改善不等于拓扑焊接。完整连接统计在逐输出表保留；未做任何后处理。

P1 ALL参考距离（不是唯一合法形状约束）：
""")
tails=cr("analysis/distance_tails/summary_distance_p95.csv")
rows=[]
for l in labels:
 t=next(x for x in tails if x["label"]==l and x["panel"]=="P1" and x["cohort"]=="ALL")
 rows.append([short[l],fmt(a(l,"F_to_GT_MSE2048_L0"),d=7),fmt(a(l,"GT_to_F_MSE2048_L0"),d=7),fmt(t["F_to_GT_p95_2048_L0"],d=5),fmt(t["GT_to_F_p95_2048_L0"],d=5),fmt(a(l,"fixed_query_p95_L"),d=5)])
parts.append(tab(["模型","F→GT MSE/L0²","GT→F MSE/L0²","F→GT p95/L0","GT→F p95/L0","旧固定512查询p95/L0"],rows))
parts.append("""双向2048点p95是在报告阶段从已有冻结距离向量取线性95%分位数的独立补充：每输出先取p95，再在父内平均噪声，最后8父等权；表中数值不是把全部点池化后的总体p95。原256项双向MSE复核一致，新增查询/几何距离调用/模型调用均0。旧固定512查询p95保持原定义，没有混为同一指标。[逐输出/逐父p95与hash](analysis/distance_tails/receipt.json)。

## 7. 多样性与开放式P2

同条件四候选的free两两表面距离，共12条件×4模型×6对=288对，方向距离共576次已完成CPU计算。下表是原始对称均方表面距离的条件等权均值，不能称原论文TMD，也没有越大越好的自动解释。
""")
div=cr("audit/report_ready/diversity_summary.csv")
parts.append(tab(["模型","P1 OLD","P1 NEW","P1 ALL","P2 OPEN"],[[short[l]]+[fmt(next(x for x in div if x["label"]==l and x["panel"]==pa and x["cohort"]==co)["mean"],d=7) for pa,co in (("P1","OLD"),("P1","NEW"),("P1","ALL"),("P2","OPEN"))] for l in labels]))
parts.append("""C的P1 ALL多样性相对A约低4.50%，B相对A约低0.07%；C仍高于START，P2也未显示总体收缩，不能据此声称模式塌缩。但“合理多样性”仍要结合几何与图像；更大距离也可能来自缺陷。

P2没有完整GT，S/E/来源距离全部NA；四种条件不是四个独立父，seat2/seat32也不是独立形状证据。
""")
parts.append(tab(["模型","P2 I%","S/E"],[[short[l],fmt(s(l,"OPEN","ConflictCov_C","P2")["value"],100),"NA"] for l in labels]))
parts.append("""P2的C−A I为+0.098pp，C−B +0.372pp（四条件均增加），B−A −0.275pp；C相对A没有数值上的冲突覆盖收益。普通续训A相对START已经降低约3.015pp，不能将这一大部分共同变化归因于known对齐。P2用途相容性以如下实际图审阅为补充，不伪造来源S/E。

## 8. 实际AI视觉审阅
""")
parts.append((R/"visual_review/visual_summary.md").read_text(encoding="utf8"))
parts.append("""## 9. 辅助目标是否真正作用于主干

预检确认：捕获不改生成器输出，B/C读取同一第4层张量；selected token梯度、早期主干辅助梯度均非零，未选中token不直接进入头；后续层无直接aux路径符合结构定义。真实预检没有参数更新。导出后194张量保持新版本严格身份，不冒充START。

B/C的头不是仅有日志数值变化：记录了头梯度/实际参数更新和生成器更新。B/C初始头逐张量相同；step1/100/1000在同一早期参数集合观察到真实aux梯度，且与FM大致近正交。B的aux梯度范数约0.010778/0.000721/0.000195，C约0.010677/0.001153/0.000174。观测范围是每个注册步首个alpha=1 microbatch、固定7个早期参数张量，aux已含0.1/4权重；不是全网、全batch梯度范数。这只能证明信号接入，无法从梯度本身证明控制变好。
""")
parts.append(tab(["训练在线统计","A","B","C"],[
["全部1000步FM均值",".114649680",".114675129",".114703259"],
["末100步FM均值",".112801802",".112792877",".112804576"],
["全部4000未增强样本在线align均值","NA",".025802339",".028914290"],
["末100步raw align均值","NA",".001919199",".002328119"],
["末100步cos均值","NA",".998080801",".997671881"],
["生成器clip次数","15","16","15"],
["辅助头clip次数","NA","0","0"]
]))
parts.append("""教师32目标的两两余弦均值0.876316，范围0.736769–0.990787；仅预测训练频率加权教师均值的常量loss为0.061828。B/C明显低于常量基准，但以上是不断变化checkpoint在已见32训练父上的在线统计，不是额外固定终点/未见父测试。全局语义教师可能学会形状类别/风格而不敏感于C附近接缝、错层或遮挡；本轮没有为验证该解释另加模型调用。[完整训练解释](audit/training_interpretation_complete.md)。

## 10. 证据等级与停止结论

1. **实现正确性：PASS。** 严格起点、共同数据流、梯度、预算、checkpoint/独立推理均有实证。
2. **表征信号：存在。** B/C能学会冻结教师目标并影响早期主干；不能由此证明泛化或control。
3. **生成质量：没有统一、独立的提升证明。** 普通续训贡献明显，参考距离/几何/图像各有取舍，未以同一SigLIP作为唯一质量裁判。
4. **局部控制：B有有限、噪声敏感的参考接缝小信号；C没有known优于free的证据。** C−B在ALL/OLD/NEW的S均下降，I平均上升，E部分改善；不能称整体接合增强。
5. **模型晋级：不进行。** B/C预注册screen均FAIL，局部尾部代价保留；START仍为默认。即使B的开发性平均S更高，也不足以据此替换START。

这些结论限定在一份训练流、32训练父和8个已见DEV父的小面板；NEW只改变噪声，CONFIRM未读取。没有人类可用率测评。不同完整形状可能都是合法补全，来源差异不自动判错。多视图全局教师、仅alpha=1对齐及未复现空条件加权都限制了对原论文方法的外推。

## 11. 交付、工程记录与恢复

所有RAW、旧START和原评分实现保持只读；新增工作位于既有MeshFlow-Control repo与外部本轮制品目录。未commit/push、未切默认模型、未添加训练外推理分支、未后处理、未扫描新层位/教师/lambda/seed。

必要工程修正均在正式科学定义不变下记录：教师CPU raster初始布尔赋值错误在首图/首模型调用前修正；旧缓存run无status字段时改为依赖完整已核合同和score状态，未重算旧分数；终点审计读取器的teacher计数口径修正；报告p95读取器兼容旧hashes/新files字段，首轮读取失败发生于任何p95输出之前。最后一项保留于engineering/distance_tail_summary_reader_before.py。实际视觉CSV合并器也补充兼容JSON数组/竖线分隔视角字段及Windows七位小数UTC时间；原观察CSV保持不变，旧读取器保留于engineering/visual_review_reader_before.py。上述不属于模型数值失败或训练重试。正式账本ERROR0，训练重放0。

原始94冻结源码hash、47原src文件前后hash、实际评分源码19文件及来源许可保留在[source_code](source_code/README.md)。[code.diff](code.diff)记录本轮新增repo源码/config/test/docs；独立repo原本无HEAD，故diff显式使用新增文件形式，不能伪称对某个旧commit的差异。制品侧审计/报表脚本也保留其源码和hash。没有重新打包第三方大权重或创建重复ZIP。

核心入口：
- [完整图集](OPEN_GALLERY.html)；[逐输出指标](per_output_metrics.csv)；[逐父配对](paired_parent_metrics.csv)；[逐噪声配对](paired_noise_metrics.csv)；[SIE汇总](sie_summary.csv)。
- [实际视觉记录](visual_review.csv)；[匿名原记录与参考复核](visual_review/final_review_receipt.json)；human usability=NOT_RUN。
- [方法/偏离](METHOD_AND_DEVIATIONS.md)；[冻结协议](protocol.json)；[教师manifest](teacher_manifest.json)；[教师训练渲染接触表](teacher_render_contact_sheet.png)。
- [checkpoint_manifest](checkpoint_manifest.json)；[training_logs](training_logs/manifest.json)；[code.diff](code.diff)；[可恢复入口说明](README_RESUME.md)。

固定任务已完成后停止。保留可恢复入口只为已有任务的完整可核验性，不据此启动新实验。
""")
(R/"EXPERIMENT_REPORT.md").write_text("\n\n".join(parts)+"\n",encoding="utf8")
print(json.dumps(dict(status="REPORT_WRITTEN",bytes=(R/"EXPERIMENT_REPORT.md").stat().st_size,model_calls=0)))
