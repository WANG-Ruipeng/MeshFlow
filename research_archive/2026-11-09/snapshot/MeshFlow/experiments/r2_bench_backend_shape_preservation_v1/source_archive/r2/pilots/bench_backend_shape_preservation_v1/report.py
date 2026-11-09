"""Render the bounded result report from preserved tables and actual reviews."""
from __future__ import annotations
from .common import *
from .analysis import records

def table(headers,rows):
 return '| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)
def f(x):return f'{float(x):.2f}'
def pct(x):return f(float(x)*100)
def run():
 dest=OUT/'report_tables';dest.mkdir(exist_ok=True)
 vals=records(OUT/'scores/parent_macro_summary.csv');v=records(OUT/'visual_review/group_summary.csv')
 groups=['CURRENT_TRAIN','DEV','NEW_TRAIN','NEW_PARENT_HOLDOUT']
 specs=[('官方FINAL','COVERAGE_FINAL','CLEAN_FINAL_POST'),('GEO_ONLY','COVERAGE_FINAL','GEO_ONLY'),('保形FINAL','SHAPE_D0','CLEAN_FINAL_POST')]
 key={(r['label'],r['stage'],r['eval_group']):r for r in vals}
 keys=['CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','actual_free_faces','free_area','free_minimum_angle_lt5_fraction','hinge_diag_isolated_1face_edge_components','nonmanifold_edges','C_F_has_valid_edge','C_F_valid_edge_total_length','opposite_winding_reached_free_area_fraction','CF_illegal_pair_count','FF_illegal_pair_count']
 rows=[dict(group=g,method=name,**{k:key[(label,stage,g)][k] for k in keys}) for g in groups for name,label,stage in specs]
 csvwrite(dest/'key_metrics.csv',rows)
 mtable=table(['分组','流程','CF%','FF%','I%','有效free面','小角%','单面孤岛','有向可达面积%'],
  [[r['group'],r['method'],pct(r[keys[0]]),pct(r[keys[1]]),pct(r[keys[2]]),f(r[keys[3]]),pct(r[keys[5]]),f(r[keys[6]]),pct(r[keys[10]])] for r in rows])
 visual=table(['分组','请求','RAW','官方FINAL','GEO_ONLY','保形BASE','保形FINAL'],[[g,next(r['requests'] for r in v if r['eval_group']==g),*[next(r['strict_visual_joint_count'] for r in v if r['eval_group']==g and int(r['column'])==c) for c in range(5)]] for g in groups+['ALL']])
 old=records(OLD/'visual_review/per_cell.csv');old={(r['request_id'],r['label'],r['stage']):r for r in old}
 current=records(OUT/'visual_review/per_cell.csv');drift=[]
 axes=['overall_shape','C_use','support_retention','visible_fragment_or_gap','layering_or_fold']
 for r in current:
  a=old.get((r['request_id'],r['label'],r['stage']))
  if a is None:continue
  out=dict(request_id=r['request_id'],label=r['label'],stage=r['stage'],old_visual_joint=a['visual_joint'],new_visual_joint=r['strict_visual_joint'])
  for x in axes:out[x+'_old']=a[x];out[x+'_new']=r[x]
  out['any_axis_changed']=any(a[x]!=r[x] for x in axes);drift.append(out)
 csvwrite(dest/'historical_visual_label_comparison.csv',drift)
 dr=[r for r in drift if r['stage']=='CLEAN_FINAL_POST']
 changed=[r['request_id'] for r in dr if (r['old_visual_joint']=='1')!=(r['new_visual_joint']=='True')]
 src=read(OUT/'analysis/COMPLETE.json')['source_summary'];st=records(OUT/'analysis/source_stage_summary.csv')
 source_table=table(['来源检查','官方D0','保形D0'],[
  ['identity masked坐标MSE',*[f"{next(r['masked_model_MSE'] for r in src if r['kind']=='IDENTITY' and r['label']==l):.8f}" for l in ('OFFICIAL_SOURCE','CANDIDATE_SOURCE')]],
  ['sigma=.03 masked坐标MSE',*[f"{next(r['masked_model_MSE'] for r in src if r['kind']=='CORNER_NOISE' and r['label']==l):.8f}" for l in ('OFFICIAL_SOURCE','CANDIDATE_SOURCE')]],
  ['identity 提议下方区域面积比%',*[pct(next(r['below_C_proxy_area_ratio'] for r in st if r['stage']=='PROPOSAL' and r['label']==l)) for l in ('OFFICIAL_SOURCE','CANDIDATE_SOURCE')]],
  ['identity GEO后下方区域面积比%',*[pct(next(r['below_C_proxy_area_ratio'] for r in st if r['stage']=='GEO' and r['label']==l)) for l in ('OFFICIAL_SOURCE','CANDIDATE_SOURCE')]],
  ['identity GEO后平均free面数',*[f(next(r['free_faces'] for r in st if r['stage']=='GEO' and r['label']==l)) for l in ('OFFICIAL_SOURCE','CANDIDATE_SOURCE')]]])
 # A new plotting presentation, preserving the original analysis figure/hash.
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 samples=records(OUT/'analysis/training_instances.csv');fig,ax=plt.subplots(figsize=(9,4.5),dpi=150)
 for kind in ('IDENTITY','CORNER_NOISE','VERTEX_NOISE'):
  xx=list(range(1,501,20));ys=[np.mean([float(r['mse_denoiser']) for r in samples if r['kind']==kind and i<=int(r['update'])<i+20]) for i in xx]
  ax.plot(np.array(xx)+9.5,ys,label=kind)
 ax.set_yscale('log');ax.set_xlabel('Optimizer update');ax.set_ylabel('MSE (denoiser units)');ax.set_title('Masked free coordinates; 20-update average');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(dest/'training_loss_readable.png');plt.close(fig)
 text=f'''# Bench后端保形校准：500步实际训练与30请求配对评价

**结论：有可保留的保形与视觉正信号，但未通过预先冻结的整体质量门槛，不替换官方D0或当前默认流程。** 新后端让来源identity误差下降约19%，真实生成的严格视觉联合从本轮复评的10/30升到14/30，新增来自7b58和a8各两份噪声、没有丢失本轮旧正例。但旧TRAIN/DEV有3父触发几何告警，FF/CF/I没有整体改善，DEV合法接口仍为0/8。

这不是“后处理没有用”，也不是完整解决了保形。它证明了当前D0的部分收缩倾向可通过少量校准减轻；同时揭示仅靠完整来源的identity/合成噪声训练，还不足以同时修好真实RAW里的相交、错层与缺损。本轮固定任务已完成并停止。

## 1. 做了什么，与此前路线有什么不同

固定Coverage生成器及已保存的30份RAW，只适配官方Bench D0 EMA（455,592,963参数），未改生成器、网络结构、输入角色或评分器。最近路线R22是Chair的sigma=.02独立扰动适配，identity只用于评价；本轮显式训练50% identity、25%独立角点噪声、25%同来源顶点相关噪声，sigma按denoiser单位.01/.03/.05循环。没有把不同面上的近邻误当成同源点。

实际训练16个独立父、每父125次曝光，共2000实例。15个评价父全部从后端训练池排除；下文CURRENT_TRAIN/NEW_TRAIN等标签仅指生成器历史，不是本次后端见过这些父。训练父由已审计USER v2候选复用，每父一个同版本C/Y；未重新修目标或扩大数据。4个训练父各有一个已知非流形顶点，原件与风险保留，不认证为全部严格流形。来源评价4父有已知缺陷，可信mask及排除面身份在data清单中明列。

**真实RAW的可信修复配对仍为0。** 没有将RAW强制配对唯一来源款式，也没有把旧D0/GEO_ONLY当成真值。因此本次是“来源保形校准迁移到真实RAW”的小试，不是已经完成真实RAW修复监督。

固定500更新、batch4/micro1、fresh AdamW（lr=1e-5，betas=.9/.95，wd=0，clip=1），FP32参数/残差、BF16前向、math SDPA、TF32关闭、严格确定性和梯度检查点。只监督可信free坐标MSE，每实例按坐标数归一、四实例等权；C不扰动、不计loss。坐标尺度为.95×.3762，face-count dropout=0。step500是唯一主终点，没有挑中间权重。

## 2. 来源保形诊断：有改善，但薄支撑仍未完全恢复

15父各identity与固定sigma=.03独立扰动，共30输入；官方和候选各一次D0。MSE是同实例恢复诊断，单位为模型坐标平方，不能当开放生成的来源款式排名。

{source_table}

identity的MSE从0.00173196降到0.00140251（约−19.02%）；噪声恢复MSE几乎持平（约−0.10%），且均低于输入噪声MSE0.00710488。本轮不是只把模型推成完全恒等，但合成去噪结果不能替代真实RAW质量。

“下方区域”是来源面身份固定的几何proxy：面最低点低于C最低点0.02L且面心低于C，求这些面面积比；不是语义支撑标签，三角面面积相加，已知来源缺陷的MSE mask与全区域面积统计也不是同一口径。每父等权，15父都保留。

实际看过全部15页来源图（双视角、同位置双截面）：a8的GEO后下方区域面积比8.51%→63.75%，f495为0.40%→30.16%，薄腿有可见恢复；27仅5.33%→13.31%，仍明显收缩。厚端板/箱体样本通常维持主体；9e自由端板的缝隙仍在。不能把所有“多留面”记为完整支撑恢复。

[来源逐例MSE](source_analysis/per_instance.csv) · [逐阶段面积与面数](source_analysis/per_stage.csv) · [15页来源图](source_analysis/index.html) · [实际来源图审](audits/source_visual_review.json)

## 3. 固定30份真实生成：保形收益与几何代价并存

候选统一执行单次D0→原C硬恢复→原GEO_C及来源角序导出→原HINGE_OPP015/ALPHA/HOLE。原RAW/官方BASE、FINAL/GEO_ONLY只读复用。候选BASE和FINAL各30份完整177字段，含分母、NA、失败状态；没有新增随机查询点或重新生成。

下表为父等权（每父2噪声、每组先父内平均）。CF/FF为受非法面对影响的free面比例，I为原有效C支持域ConflictCov_C，不是全网格自交率；单面孤岛是合法共边图中的单面分量，不等于所有都是肉眼漂浮面。Native无SEAM，对应seam指标保持NA。

{mtable}

相对官方FINAL，候选平均free面保留更多：CURRENT_TRAIN87.50→109.25、DEV70→99、NEW_TRAIN30.75→71.75、df6 126.50→171.50。但CURRENT_TRAIN的FF15.80%→18.07%、DEV22.26%→24.45%，不能以面数增加宣布质量提高。NEW_TRAIN的FF比例1.79%→1.09%，其非法FF面对均数却0.25→0.50，必须保留分母变化这项矛盾证据。

相对GEO_ONLY，候选显著压低FF：CURRENT_TRAIN31.66%→18.07%、DEV49.06%→24.45%、df6 70.40%→20.09%。代价是仍丢失一些RAW结构。它处于原D0和GEO_ONLY之间，尚未实现两者优点兼得。

接续也并非全组改善：CURRENT_TRAIN有合法C/F接口8/16→10/16、平均合法长度0.831→4.239，有向可达面积7.09%→13.48%；DEV仍0/8、df6仍0/2。NEW_TRAIN接口仍3/4，但平均合法长度3.603→2.458、有向可达面积37.90%→15.40%。面积可达、共边和视觉合理性分别保存，不能互相替代。

预定几何门槛：旧CURRENT_TRAIN/DEV中若至少2父出现FF≥+10pp或CF/I≥+5pp即不晋级。实际3父：838c（FF+22.43、CF+6.12pp），91e（CF+5.31pp），2d（I+5.005pp）。即使把2d的临界值单独标注，另两父也已触发门槛。没有据结果调整阈值。

[完整270评分行](scores/per_output.csv) · [完整177字段父均表](scores/per_parent.csv) · [逐请求配对](scores/paired_per_output.csv) · [逐父告警](analysis/geometry_gate_per_parent.csv) · [重点字段](report_tables/key_metrics.csv)

## 4. 实际图审与截面：哪些保住了，哪些没解决

全部30页、150格均由AI实际看过，使用相同双视角与原位置双截面；非盲AI_PROXY，不是独立人评。严格视觉联合定义为同一输出“整体合理 AND C采用合理 AND 支撑存在”；碎片/折层另列，不把此联合当作无交、流形或合法接续认证。

{visual}

本轮成对复评中官方FINAL10/30、候选14/30，保住10个旧正例并新增4个：7b58两噪声端部支撑少缺损，a8两噪声腿更完整。改善并非只限a8/27，但跨父范围仍小。27支撑仍不足或不确定；9e端架破损、df6中央碎片、2d/459等灰座面与蓝C错层没有普遍解决。BASE与FINAL的视觉联合相同，不能把最后CPU整理另记为4个新增视觉成功。

原GEO_ONLY报告曾记官方FINAL9/30、GEO_ONLY18/30；这次重新图审对应10/30和19/30。共同旧FINAL有{len(changed)}个请求的联合标签变化：{', '.join(changed)}。旧记录未改。完整五轴历史转移见[复评差异](report_tables/historical_visual_label_comparison.csv)，主结论使用本轮同页配对，不能用历史9直接减新14来宣称新增5例。

[30页完整对照图集](gallery/index.html) · [150格实际图审](visual_review/per_cell.csv) · [逐请求转移](visual_review/paired_transitions.csv) · [原正例丢失表](visual_review/original_positive_losses.csv)

## 5. 执行账本、工程事件与恢复

真实执行2000次训练实例前向、2000次反向、500次AdamW更新；梯度检查点内部重计算属于正常反向，不是额外训练实例。评价90次单次D0，生成器前向0、新自由生成噪声0。CPU新增30请求GEO及60完整评分、30来源identity GEO；旧210评分行只读复用。

第100步完整检查点保存后发生日志关键字path重名TypeError。故障、旧源码和快照保留；修复仅改日志参数名并提供专用边界恢复。从101继续，296个AdamW状态全step100及Python/NumPy/CPU/CUDA RNG逐项恢复一致，重算更新/前向/反向均0。最终step500与推理文件296个FP32张量完全相同。故障不是“全程零失败”，也没有掩盖一次额外训练。

RTX5080 16GB，训练单worker；实测整卡约10.8GB、余量约5.2GB，因此训练期间不叠加GPU评价。两个执行段内计时累计656.31秒＝原段164.15秒＋恢复段492.16秒，含段内装载、训练、保存和部分校验；不含排障间隔、启动前检查及计时点后的最终hash/收尾。500个更新循环计时累计392.34秒；complete.json的training_seconds=461.65秒仅为恢复段装载后的范围，不能代表完整训练。训练PyTorch峰allocated8.342GB/reserved8.512GB不代表整进程显存。去噪评价实测整卡约4.44GB，单次allocator约1.88GB；三组30份评价活动墙钟103.60/36.19/103.44秒，含文件校验和装载，不能当作单次去噪延迟。

CPU24逻辑核、评分派发时约61GiB可用RAM；3个评分worker、每进程1线程，与独立单worker图集渲染/来源分析重叠。完整几何评分约287.49秒，worker活动时间合计789.69秒，不能与独占串行耗时直接当硬件加速比。无OOM、无科学预算缩减，未终止用户进程。

保留模型：[推理权重](training/candidate_denoiser.pt)，SHA256 `60f0aaf0003907913ed06d7d40f516fa88cf912517d37e8b155056acd298a5cc`；[完整step500](training/checkpoints/step0500.pt)，SHA256 `ee34b1605f1a37437556407940c537154c611bc310c20f1b0dd48ce2c9ee94a1`。前者约1.82GB，后者约5.47GB。中间恢复快照仍保留，本轮未清理旧资产。

[训练曲线](report_tables/training_loss_readable.png) · [逐更新](analysis/training_updates.csv) · [2000实例账本](analysis/training_instances.csv) · [独立训练审计](audits/training_exact_budget.json) · [工程修复](engineering_amendment.json) · [独立结果复核](audits/results_review.json)

## 6. 本轮决定与唯一后续优先问题

**保留这份权重和保形正证据，不晋级为综合后端，不替换当前默认，也不自动延长训练。** 结果支持“D0的部分薄结构收缩可校准”，不支持“合成保形校准已经解决真实生成的整体质量”。来源样本MSE改善与真实RAW质量之间仍有差距，不能把它直接命名为经典过拟合或唯一根因。

若继续，优先缺口是可信的真实RAW结构修复监督：在保留正确支撑的同时，给出哪些自由面确需修正的训练证据。待检验假说是：加入这种监督后，相对当前固定后端，能维持7b58/a8的保形收益，同时降低838c/91e的冲突代价，而非再次依靠更多删面获得低FF。此处仅记录方向，不授权或启动下一轮训练；当前可用可信RAW修复对为0，不虚构已经具备这类数据。

## 7. 交付与复算入口

数据、固定噪声、来源身份、mask和2000曝光：[data/manifest.json](data/manifest.json)。代码与原评分依赖快照见`source_archive/`、`code.diff`、`source_hashes.json`；完成清单`COMPLETE.json`绑定所有交付物hash。恢复入口与已完成阶段复用见[RUNBOOK](RUNBOOK.md)。

```text
python -B -m r2.pilots.bench_backend_shape_preservation_v1.run verify
```

完成后此入口只验证，不自动重训/重采样。原资产、评分器、报告和默认配置保持不变；无commit/push。
'''
 with (OUT/'REPORT.md').open('x',encoding='utf-8') as h:h.write(text)
 write(dest/'COMPLETE.json',dict(status='COMPLETE',historical_final_joint_changed_requests=changed,files={str(p):sha(p) for p in dest.iterdir() if p.is_file()}))
 print('REPORT_WRITTEN',len(text),changed)
if __name__=='__main__':run()
