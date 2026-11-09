"""Read-only result aggregation, report and artifact packaging metadata. No models."""
from pathlib import Path
import csv,json,statistics,hashlib,difflib,datetime,re
from common import ROOT,OUT,PREV,CODE,read,write,sha,verify,checked

def rows(name):return list(csv.DictReader((OUT/name).open(encoding='utf-8-sig')))
def csvwrite(name,data):
 with checked(OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def save(name,text):checked(OUT/name).write_text(text,encoding='utf-8')
def table(head,data):return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|']+['| '+' | '.join(map(str,r))+' |' for r in data])
def pct(v):return f'{float(v)*100:.2f}'
reg=read(OUT/'registration.json');qc=rows('quality_per_output.csv');qs=rows('quality_summary.csv');sm=rows('summary.csv');ts=rows('topology_summary.csv');stages=['RAW','D1','D3','D5']
assert read(OUT/'denoise_completion.json')['total_forward_attempts']==48
assert read(OUT/'score_completion.json')['new_scored_outputs']==24
assert read(OUT/'independent_review/REPORT.json')['status']=='PASS'
assert (OUT/'gallery/AI_PROXY_REVIEW.md').exists()
q={(r['stage'],r['group']):r for r in qs};a={(r['stage'],r['group'],r['setting']):r for r in sm};top={(r['stage'],r['group']):r for r in ts}
qfields=['CF_proper_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','free_minimum_angle_lt5_fraction']
main=[]
for st in stages:
 main.append([st]+[pct(q[st,'ALL'][k]) for k in qfields]+[pct(a[st,'ALL','raw_tau_1e-6']['A_VMR']),pct(a[st,'ALL','raw_tau_1e-6']['PMR']),top[st,'ALL']['outputs_with_C_free_shared_edge']+'/12'])
mt=table(['阶段','CF↓%','FF↓%','I↓%','小角↓%','严格A-VMR↑%','严格PMR↑%','存在C/free共边'],main)
paired=[]
for st in ('D3','D5'):
 for group in ('ALL','seat','back'):
  rr=[r for r in rows('quality_paired_vs_D1.csv') if r['stage']==st and (group=='ALL' or r['region_type']==group)]
  for k in qfields+['ConflictCov_C_cross','ConflictCov_C_dup']:
   ds=[float(r['delta_'+k]) for r in rr]
   paired.append(dict(stage=st,reference='D1',group=group,metric=k,outputs=len(ds),mean_delta=statistics.mean(ds),median_delta=statistics.median(ds),min_delta=min(ds),max_delta=max(ds),improved=sum(x<-1e-12 for x in ds),tied=sum(abs(x)<=1e-12 for x in ds),worsened=sum(x>1e-12 for x in ds)))
csvwrite('paired_delta_summary.csv',paired)
curves=[]
for n in (2,3,4,5):
 rr=[read(p) for p in (OUT/'denoised'/f'D{n}').glob('*/receipt.json')]
 curves.append(dict(round=n,outputs=len(rr),mean_step_free_displacement_model=statistics.mean(r['mean_step_free_displacement_model'] for r in rr),maximum_step_free_displacement_model=max(r['max_step_free_displacement_model'] for r in rr),mean_from_RAW_free_displacement_model=statistics.mean(r['mean_total_free_displacement_model'] for r in rr),maximum_from_RAW_free_displacement_model=max(r['max_total_free_displacement_model'] for r in rr)))
csvwrite('round_displacement_summary.csv',curves)
qr={(r['request_id'],r['stage']):r for r in qc};cost=[]
for st in stages:
 for group in ('ALL','seat','back'):
  rr=[r for r in qc if r['stage']==st and (group=='ALL' or r['region_type']==group)];before=[qr[r['request_id'],'RAW'] for r in rr]
  original=sum(int(float(r['free_face_denominator'])) for r in before);retained=sum(int(float(r['free_face_denominator'])) for r in rr)
  cost.append(dict(stage=st,group=group,outputs=len(rr),original_free_total=original,retained_free_total=retained,deleted_free_total=original-retained,pooled_deleted_fraction=(original-retained)/original,macro_deleted_fraction=statistics.mean(1-float(r['free_face_denominator'])/float(o['free_face_denominator']) for r,o in zip(rr,before)),macro_free_area_ratio_to_RAW=statistics.mean(float(r['free_total_surface_area'])/float(o['free_total_surface_area']) for r,o in zip(rr,before))))
csvwrite('deletion_summary.csv',cost)
costt=table(['阶段','原free总数','保留free总数','删面数','池化删面比例%','逐例free面积/RAW均值%'],[[r['stage'],r['original_free_total'],r['retained_free_total'],r['deleted_free_total'],pct(r['pooled_deleted_fraction']),pct(r['macro_free_area_ratio_to_RAW'])] for r in cost if r['group']=='ALL'])
fullt=table(['阶段']+[x['name'] for x in reg['settings']],[[st]+[pct(a[st,'ALL',x['name']]['A_VMR'])+'/'+pct(a[st,'ALL',x['name']]['PMR']) for x in reg['settings']] for st in stages])
regiont=table(['区域','阶段','CF↓%','FF↓%','I↓%','小角↓%'],[[g,st]+[pct(q[st,g][k]) for k in qfields] for g in ('seat','back') for st in stages])
pairst=table(['对比','指标','改善/持平/退化（12例）','平均变化pp','中位变化pp','最小/最大变化pp'],[[r['stage']+'−D1',r['metric'],f"{r['improved']}/{r['tied']}/{r['worsened']}",f"{100*r['mean_delta']:+.3f}",f"{100*r['median_delta']:+.3f}",f"{100*r['min_delta']:+.3f}/{100*r['max_delta']:+.3f}"] for r in paired if r['group']=='ALL' and r['metric'] in qfields])
caserid='NEWDEV__33ec1e64a02ff718faaadcdc0221eb72__seat__medium__n0'
case=table(['阶段','保留free面','FF↓%','小角↓%','free面积（模型单位²）'],[[st,int(float(qr[caserid,st]['free_face_denominator'])),pct(qr[caserid,st]['FF_illegal_affected_free_fraction']),pct(qr[caserid,st]['free_minimum_angle_lt5_fraction']),f"{float(qr[caserid,st]['free_total_surface_area']):.3f}"] for st in stages])
findings=f'''# USER_MIX 累计1／3／5次去噪：固定面板结论

实际完成同一12份USER_MIX RAW的对照。复用第一次结果，新增48次官方去噪前向；每轮恢复原C，未在中间合并/删面，各终点最后执行一次同配置GEO_C。没有训练、蒸馏、新生成或新噪声。

**多次去噪能够继续降低部分相交指标，但这批结果不支持直接将3或5次设为默认。** I没有改善，PMR仍全0；而且出现明确的细腿支撑消失。不能仅因FF更低就称成品更好。

{mt}

D5相对D1的FF改善10/12、退化2/12，I改善4/12、持平3/12、退化5/12。严格匹配少量增加；1%L A-VMR为5.56→7.54→9.39%，但2%L为16.40→16.07→15.61%。四份座面在2%L下始终0。新增的一份共边输出不满足“双面且反向绕序”的检查，不能算新增可靠接口。

最明显坏例为33ec中等座面n0：D1还有细腿，D3/D5两个对向视角都看不到支撑腿，只余座背；C与灰座面的错层仍在。D1保留136个free面，D3只85，D5只66，D5的FF=0但小角比例54.55%。[该例图](gallery/{caserid}__v0.png)及[反向视角](gallery/{caserid}__v1.png)。

也有局部整理收益：866e n1前沿细长面减少但盒形/短脚仍在，f596 n1减面后主轮廓保留。全24图已实际AI_PROXY审阅；其余11请求未见同等级支撑消失，但有限视角不能认证它们完全无损。FF改善不全由删面造成：D1的208个非法面对中，到D5有140对两面仍在且变为合法接触/分离，同时40对涉及删面、28对仍非法，另有72个新非法对。

**建议继续以一次去噪作为当前比较基准；这轮没有提供采用3/5次作为统一后处理或蒸馏目标的充分证据。** 这是六父开发面板结论，不是所有网格或所有重复修正方式的定论。固定任务已停止，不自动启动训练/蒸馏、改loss、切换默认模型或commit/push。

[完整报告](REPORT.md) · [全部24张对照图](gallery/index.html) · [逐例几何指标](quality_per_output.csv) · [逐例边界匹配](per_output.csv) · [终点网格索引](post_index.csv)
'''
save('FINDINGS.md',findings)
report=f'''# R2 USER_MIX 多次官方去噪固定面板实验

日期2026-10-10；版本{reg['version']}。状态 COMPLETE。本轮回答：同一模型、同一RAW连续去噪3/5次，能否较一次进一步改善最终mesh？

结论：**FF/CF有进一步改善和少量精确边界匹配增量；I与完全匹配没有改善，且一份细腿椅子出现严重支撑丢失。暂不支持全局替换一次基准。** 完整指标与坏例一并保存。

## 固定范围与重复定义

仅当前研究候选R2_USER_MIX的12份已有RAW：6 USER开发条件（2座面、4靠背）×2固定噪声。没有重跑四个生成器，也没有训练、蒸馏、生成器前向或新抽Gaussian。全部来源、C/RAW/checkpoint/noise hash在registration.json。此开发面板已反复检查，不能外推独立泛化。

令X0为已有RAW，Xr为官方去噪后的连续三角面；每轮直接读取前一轮尚未几何整理的输出，执行：

`Xr = restore_original_C(denoiser(Xr−1))`

原始N、面顺序/槽位始终不变，r=2严格从前轮保存的未Geo X1开始。每轮模型输入乘固定0.35739，BF16网络前向后在FP32累加残差并除回同尺度，原C逐字节恢复；没有输出bbox归一、噪声日程或噪声注入。推理使用同一官方EMA/同一face-count条件、SDPA math、TF32关闭、严格deterministic。官方identity encoder未采样，48个receipt均核验前后RNG无消耗。

D1／D3／D5分别是`GEO_C(X1)`、`GEO_C(X3)`、`GEO_C(X5)`；终点都只做一次同配置顶点合并和free退化/重复删除。D3的Geo结果绝不作为D4输入。D2/D4仅保存中间几何、位移和receipt，没有追加终点评分。D1及RAW复用旧冻结结果。**这不是循环执行完整Geo，也不是降低噪声强度的扩散采样。**

新前向预算12×4=48，实际48成功、0重试；新Geo24/24、新CPU几何评分24/24。原C中间48/48及全部POST36/36逐位保持。统一4阶段共48条mesh记录，其中24引用旧产物、24新增。

权重、匹配源码及运行时适配沿用[前轮完整报告](../r2_postprocess_avmr_pmr_v1/REPORT.md)：官方Chair0030000.pt、455,592,963参数、固定SHA256 `6a31f4c7dcbea7a6ac58c26075b20c897fd19aad4c33b5295cc4c93ed7078c36`，对应官方commit db86bc32423c316c00bcc2f645a26243e8010f20。没有再次下载或复制7.29GB训练包。仍是发布权重的C保持适配：最终GEO_C替代原量化/无约束导出，不能称官方原脚本逐位复现。

## 总体数值

百分比为逐输出宏平均；每父2噪声相同，等于先噪声均值再父等权。↓为低好，↑为高好，但每项仅支持其明确性质。

{mt}

CF=与C严格穿插的受影响free面比例；FF=与其他free发生严格穿插或正面积共面重叠的受影响free面比例。它们不是非法面对占所有面对的比例，POST分母随实际free面数变化。I=冻结C表面2048查询上的cross与dup候选并集；近并行覆盖候选可能含合法薄壁，不能直接当非法重复层。原检测器、固定L、阈值、冻结查询和定义全部沿用，未按本轮结果调整。

{pairst}

FF均值改善没有覆盖所有请求；I两终点均有5/12退化，dup候选均值D1 1.009%→D3 1.156%→D5 1.245%。CF median变化0，不能把均值下降称普遍改善。

## 座面与靠背分别看

{regiont}

座面小角均值D1 33.28%→D3 34.67%→D5 33.81%，全组小角改善主要由靠背贡献。座面CF降到0也不保证座面/腿的结构合理，严重坏例正好出现在座面组。无唯一GT重建排名，无外露任务或必要接口注册，未强行算S/E或新增S_join。

## A-VMR／PMR及实际连接

来源规则沿用[指标政策](../../native_t1/docs/control_metrics.md#meshfim-boundary-metrics)：源顶点ID拓扑的全部C开放边界端点，仅匹配F角点，严格欧氏距离小于tau，逐输出匹配比例宏平均、全部边界顶点匹配的输出比例为PMR。L固定为sqrt(3)×2/.3762=9.208138264587333，不按输出bbox改变。模型单位1e-6设置名沿用raw_tau_1e-6。论文评分尺度兼容性仍UNKNOWN，全部C开放边并非已注册必须接合接口；匹配不认证水密、主体接入或语义用途。

全部8档（每格A-VMR/PMR，%）：

{fullt}

1%L有增量，2%L反而下降，不存在所有容差都改善。4份seat在2%L仍全0，不能据此把整片C到F表面距离也说成0/无穷；这里查的是边界顶点到F顶点。

实际C/free共享边输出D1 1/12→D3/D5 2/12。都是86ec back full：n1原已有一条双面反向绕序边；n0新增一条共享边，但双面反向绕序检查为0。因此新增计数不能当新增有效完整接口。POST共边分量均数14.25→7.00→6.25，非流形边均数6.67→8.25→8.25；更少碎分量不保证合法拓扑，详见topology_per_output.csv。

## 减面、位移与面对变化

{costt}

池化删面比与按输出先算比例再平均不同，deletion_summary.csv同时保留。D1→D5净少235free面，但终点Geo独立、保留面集合并非嵌套：D1保留面中241在D5删除，同时6个D1已删面在D5被保留。不能描述为沿D1索引依次删除235面。

连续去噪的平均每轮free角点移动仍非零：D2约0.01554、D3约0.01061、D4约0.00866、D5约0.00772模型单位；各轮最大移动分别0.26736/0.21597/0.19410/0.19490。到D5累计相对RAW最大0.46633。均值递减不等于已收敛或无局部大漂移。逐轮、相对RAW/D1、Geo单独位移及源面映射均保存，后续合并半径0.0419709561不能冒充全流程位移上界。

FF下降并非只有删面贡献。读取既有面对分类与source_face_ids对齐，无重评分：D1的208个非法面对，到D3有136对保留双面变非非法、31对涉及删面、41仍非法，另93个新非法对；到D5为140、40、28，另72个新非法对。D5的140对具体为31共享边接触、68共享点接触、41分离，没有把UNKNOWN当改善。详见FF_pair_fates_*.csv与独立FF_fates_REPORT.json。该诊断按面对计数，不按穿透体积或面积计，不提供对去噪和Geo的单独因果归因。

## 全图审阅与严重反例

[完整24张图集](gallery/index.html)：12请求×2对向视角，每张RAW/D1/D3/D5四面板共用范围和相机，C蓝、F灰，全三角面保留，不读GT或为美观隐藏面。

33ec seat medium n0：

{case}

D1已有腿部变细，D3/D5两个对向视角都看不到任何支撑腿，只余座背；C低于灰座面的错层继续存在。D5删除原146/212free、FF=0而小角54.55%，是“更低局部冲突不能代替完整形状质量”的明确反例。[视角0](gallery/{caserid}__v0.png) · [视角1](gallery/{caserid}__v1.png)。该观察对应最终完整流程，不能仅据终点图片将全部损伤唯一归因于神经位移或最后合并。

也如实保留正面变化：866e n1前沿细长层面整理、盒形短脚仍在；f596 n1大量减面但主轮廓保留；f596 n0部分脚踝间隙收拢，但背孔较D1更锯齿、局部中断仍在。其余11请求未观察到同等级整组支撑消失，不表示它们通过严格完整性认证。

[逐图审阅](gallery/AI_PROXY_REVIEW.md)与48条逐阶段、12条逐请求表全部保留。实际看过全部24PNG，标为非盲AI_PROXY，审阅者知道之前图审及部分数值提示；不包装成人评、盲评或接合认证。无图像证据支持普遍改变C的宏观用途关系。

## 执行、恢复与独立核验

Windows/WSL测资源后启动：RTX5080总16303MiB，前置约2702MiB使用、13278MiB空闲，GPU利用1%；CPU24逻辑核、Windows可用RAM约63.54GiB。记录活动进程，没有停止任何用户进程。按旧同设置及本轮首个真实前向测量选择一个GPU进程；48个短前向仅数秒，重复模型加载/进程成本超过潜在收益。每请求各轮必须按依赖执行，显存富余本身不构成增加worker收益。

去噪进程39.39秒，48个前向计时之和2.777秒；登记forward_phase_wall 8.873秒包含I/O、资源和终态哈希核验，不是纯神经时间。单次scope含常驻FP32权重的PyTorch峰allocated 1,881,729,536bytes、reserved 1,910,505,472bytes；WDDM逐进程显存不可可靠提供，不能用PyTorch计数冒充进程总显存。CPU评分首例测量后2worker各2线程，24份21.445秒；图集1CPU/1线程并行4.248秒。并行耗时不当隔离benchmark。

独立审计不导入主评分器，用NumPy源ID边统计、SciPy cKDTree、手写binary PLY解析与edge incidence/union-find复算：384指标、496距离、96摘要、288对D1配对、48拓扑一致，最大距离差0；24新PLY和12旧D1 PLY均与NPZ逐位一致。48轮未Geo输入链、原N/FP32、C恢复、proposal的F和位移全部核验。官方权重重新读全SHA通过，895保护文件终态一致。质量不是另重写几何检测器，而是独立核对保存来源1968数值及各汇总、分母/固定查询；明确区分这两种核验范围。证据在independent_review/REPORT.json、aggregate_REPORT.json、FF_fates_REPORT.json。

工程预检有两项均已记账：GPU前CPU评分schema允许五个明确D1来源字段的修正，保留pre_gpu_registration_v0.json及新版hash，预算/科学设置未改；面对去向派生脚本首次assert把total计入求和，已修正，原失败/修正执行写账本。它们没有消耗额外GPU前向或重算24正式几何评分。原数据/旧报告/权重没有被写入。

可恢复总入口：`r2/tools/multistep_denoise_v1/run.py --stage all`；可分别prepare/denoise/score/gallery。已有有效receipt复用；已开始的模型调用若没有成功receipt将拒绝静默重放。完成本固定任务后不再调用额外阶段。

## 交付与后续判断

outputs/D3和D5保存24最终NPZ/PLY，post_index.csv给出4阶段48条完整来源；denoised/D2..D5保存48中间网格/原proposal/位移和receipt。per_output/summary保存全部8阈值；quality/topology/paired/per_condition_noise保存逐例、配对和分组；scores保存真实逐点/逐面对诊断；实际执行源码快照、code.diff、source_hashes及资源账本一并保存，权重只引用不复制。

本轮完成并停止。建议保留一次作为当前比较基准，暂不将3/5次升为统一后处理或蒸馏教师目标。更少局部穿插伴随明显个例结构代价，且主要C接合缺口仍在。没有自动训练、蒸馏、改loss、切默认、阈值扫描或commit/push。
'''
save('REPORT.md',report)
# Freeze actual code and dependencies, with flat inherited names to avoid nested path inflation.
source_rows=[]
for file in sorted(CODE.glob('*.py')):
 dst=OUT/'source_snapshot/local'/file.name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(file.read_bytes())
 source_rows.append(dict(original=str(file),snapshot=dst.relative_to(OUT).as_posix(),sha256=sha(file),bytes=file.stat().st_size))
for file in sorted({Path(p) for p in reg['protected'] if p.endswith('.py')}):
 digest=sha(file);dst=OUT/'source_snapshot/inherited'/(digest[:12]+'_'+file.name);dst.parent.mkdir(parents=True,exist_ok=True)
 if not dst.exists():dst.write_bytes(file.read_bytes())
 source_rows.append(dict(original=str(file),snapshot=dst.relative_to(OUT).as_posix(),sha256=digest,bytes=file.stat().st_size))
for file in (OUT/'independent_review').glob('*.py'):
 dst=OUT/'source_snapshot/independent'/file.name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(file.read_bytes());source_rows.append(dict(original=str(file),snapshot=dst.relative_to(OUT).as_posix(),sha256=sha(file),bytes=file.stat().st_size))
csvwrite('source_hashes.csv',source_rows)
diff=[]
for file in sorted(CODE.glob('*.py')):diff.extend(difflib.unified_diff([],file.read_text().splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+file.relative_to(ROOT).as_posix()))
save('code.diff',''.join(diff))
protected_count=verify(reg)
expected={'per_output.csv':384,'per_vertex_distances.csv':496,'quality_per_output.csv':48,'topology_per_output.csv':48,'summary.csv':96,'post_index.csv':48,'paired_vs_D1.csv':288,'quality_paired_vs_D1.csv':36,'per_condition_noise_quality.csv':24,'gallery/AI_PROXY_per_stage_review.csv':48,'gallery/AI_PROXY_per_request_review.csv':12}
counts={k:len(rows(k)) for k in expected};assert counts==expected
links=0
for name in ('REPORT.md','FINDINGS.md'):
 for link in re.findall(r'\]\(([^)]+)\)',(OUT/name).read_text()):
  if link.startswith(('https:','http:','#')):continue
  assert (OUT/link.split('#')[0]).exists(),(name,link);links+=1
write(OUT/'completion.json',dict(status='COMPLETE',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),generator='R2_USER_MIX',requests=12,stages=stages,reused_RAW_outputs=12,reused_D1_outputs=12,new_denoiser_forwards=48,forward_retries=0,new_GEO=24,new_geometry_scores=24,generator_forwards=0,training_updates=0,distillation_updates=0,new_noise=0,new_intermediate_C_exact=48,total_POST_C_exact=36,new_PLY_NPZ_consistent=24,protected_hashes=protected_count,independent_audit='PASS',quality_aggregate_audit='PASS',FF_fates_audit='PASS',AI_PROXY_reviewed_PNG=24,CPU_score_workers=2,GPU_workers=1,render_workers=1,row_counts=counts,links_verified=links,no_default_change=True,no_git_operations=True,task_stopped=True))
# Persistent project notes are maintained documentation, never immutable experiment inputs.
note='''\n## 2026-10-10 USER_MIX 多次去噪对照完成：暂不采用3/5次默认\n\n同一12 USER_MIX RAW复用D1，实际新增48官方去噪前向得到D2..D5，每轮C原字节恢复、N不变，无中途Geo；D3/D5终点各一次同配置GEO_C、24 CPU评分与24两视角图全部完成。零训练/蒸馏/新生成/新噪声。ALL D1/3/5：FF12.02/8.02/6.12%，CF4.88/4.49/4.32%，I3.69/3.71/3.77%，小角17.06/16.28/16.06%；严格A-VMR1.39/2.31/3.37%，PMR全0，共边输出1/2/2，但新增共边未通过双面反向绕序检查。四座面在2%L仍0。\n\nD5 FF改善10/12、退化2；I改善4/12、持平3、退化5。33ec seat n0在D3/D5两视角整组腿消失，D1/3/5 free数136/85/66，D5 FF=0却小角54.55%，C错层未解决。其他11请求未见同等级支撑消失，但图审非盲AI_PROXY，不能认证无损。实际FF改善不仅靠删面：140个D1非法面对两面保留后变合法接触/分离，也有新非法对。保留一次为比较基准，未自动改默认或蒸馏目标。\n\n[结论](experiments/r2_multistep_denoise_v1/FINDINGS.md)、[完整报告](experiments/r2_multistep_denoise_v1/REPORT.md)、[全部图集](experiments/r2_multistep_denoise_v1/gallery/index.html)、384指标行/496顶点距离/48质量与拓扑/逐父噪声/24新PLY及实际源码diff/hash均保存。独立数值、网格和48前向链审计PASS，895受保护文件/官方权重hash一致。固定任务完成后停止；没有commit/push、新loss、模型切换或后续训练。\n\n'''
w=ROOT/'WORKSPACE.md';b=w.read_bytes();mark='## 2026-10-10 USER_MIX 多次去噪对照完成：暂不采用3/5次默认'.encode()
assert str(w) not in reg['protected']
if mark not in b:
 ix=b.index(b'\n')+1;w.write_bytes(b[:ix]+note.encode()+b[ix:])
metricnote='''### 2026-10-10 多次去噪补充证据：低冲突不等于完整成品\n\n固定USER_MIX12输出的D1/D3/D5复评分已完成，见[完整报告](../../experiments/r2_multistep_denoise_v1/REPORT.md)。每轮恢复C，中间不Geo；终点各Geo一次。FF12.02→8.02→6.12%，I3.69→3.71→3.77%，PMR全0。33ec座面n0的D5虽FF=0，却在两视角失去整组支撑腿、小角54.55%。这是必须同时报告完整形状、删面和薄构件损伤的直接证据，不能只取FF最低点。严格A-VMR略升及共边输出1→2也未解决四座面边界匹配，新增共边没有通过双面反向绕序检查。所有阈值、配对及AI_PROXY局限保留；不新增S_join或改旧定义，不自动升为默认。\n\n'''
m=ROOT/'native_t1/docs/control_metrics.md';b=m.read_bytes();assert str(m) not in reg['protected'];mark='### 2026-10-10 多次去噪补充证据：低冲突不等于完整成品'.encode()
if mark not in b:
 token='## 长期执行要点'.encode();assert token in b;m.write_bytes(b.replace(token,metricnote.encode()+token,1))
manifest=[]
for f in sorted(OUT.rglob('*')):
 if f.is_file() and f.name!='artifact_manifest.csv':manifest.append(dict(path=f.relative_to(OUT).as_posix(),bytes=f.stat().st_size,sha256=sha(f)))
csvwrite('artifact_manifest.csv',manifest)
print(json.dumps(dict(status='COMPLETE',protected_hashes=protected_count,files=len(manifest),artifact_MiB=sum(x['bytes'] for x in manifest)/1048576,row_counts=counts,links=links),ensure_ascii=False))