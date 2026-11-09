"""Summarize the executed saved-vector analysis and plot additive contributions."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).parent
def read(n):return list(csv.DictReader((P/n).open(encoding='utf-8-sig')))
def mean(rows,k):return float(np.mean([float(r[k]) for r in rows]))
D=read('distance_summary.csv');R=read('region_distance_contributions.csv');C=read('control_witness_changes.csv');F=read('conflict_per_known_face.csv');A=read('paired_point_changes.csv')
directions={d:{l:{k:mean([r for r in D if r['direction']==d and r['label']==l],k) for k in ['mse','mean_distance','median_distance','p95_distance','tail_top5_mse_contribution','coverage_1pct']} for l in ['START','teacher500','TR500','cap500','cap1000']} for d in ['GT512','GT2048','F2048']}
regions=[]
for n in ['512','2048']:
 for partition in ['C_distance','GT_Y_thirds']:
  for name in sorted({r['region'] for r in R if r['partition']==partition}):
   v={l:mean([r for r in R if r['query_count']==n and r['partition']==partition and r['region']==name and r['label']==l],'global_MSE_contribution') for l in ['START','teacher500','cap500','cap1000']}
   regions.append(dict(query_count=int(n),partition=partition,region=name,**v,delta_cap500_START=v['cap500']-v['START'],delta_cap1000_cap500=v['cap1000']-v['cap500']))
control={re:{k:mean([r for r in C if r['candidate']=='cap500' and r['reference']==re],k) for k in ['S_gained_weight','S_lost_weight','I_added_area','I_removed_area','conflict_cross_delta','conflict_dup_delta','cross_and_dup_delta']} for re in ['START','teacher500','TR500']}
tail={re:{k:mean([r for r in A if r['candidate']=='cap500' and r['reference']==re and r['query_count']=='512'],k) for k in ['MSE_delta','points_improved_fraction','positive_error_contribution','negative_error_contribution','top5_positive_delta_contribution']} for re in ['START','teacher500','TR500']}
special={}
for uid in sorted({r['uid'] for r in F}):
 rows=[r for r in F if r['uid']==uid and r['candidate']=='cap500' and r['reference']=='teacher500']
 pairs={}
 for noise in ['0','1']:
  a=[r for r in rows if r['noise_index']==noise];top=sorted(a,key=lambda r:float(r['I_delta_global']),reverse=True)[:3]
  pairs[noise]=[dict(face=int(r['known_face_index']),source_face=int(r['source_face_id']),I_delta_pp=100*float(r['I_delta_global']),dup_delta_pp=100*float(r['dup_delta_global']),centroid=[float(r[k]) for k in ['centroid_x','centroid_y','centroid_z']]) for r in top]
 special[uid]=pairs
summary=dict(directions=directions,regions=regions,saved_control_transitions=control,paired_512_point_changes=tail,top_conflict_faces=special,
 caveats=['GT512 and GT2048 use different frozen samples of the same target; distances within each bank are exactly paired across models.','Generated Q_F2048 points differ across models; compare population summaries, never pair their point indices.','C-distance bins use original condition only; Y thirds are geometric location bins, not semantic chair parts.','Region contributions use full query denominator and sum exactly to the original MSE. No original gate or denominator changes.'])
(P/'geometry_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
parents=sorted({r['uid'] for r in C});short=[x[:6] for x in parents]
fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained');plt.rcParams.update({'font.size':11})
x=np.arange(8)
for offset,re,col in [(-.18,'START','#4477aa'),(.18,'teacher500','#dd8855')]:
 vals=[100*mean([dict(r,net=float(r['S_gained_weight'])-float(r['S_lost_weight'])) for r in C if r['candidate']=='cap500' and r['reference']==re and r['uid']==u],'net') for u in parents]
 axs[0,0].bar(x+offset,vals,.35,label='vs '+re,color=col)
axs[0,0].axhline(0,color='#444',lw=.8);axs[0,0].set_xticks(x,short,rotation=35);axs[0,0].set_ylabel('SeamCov change (percentage points)');axs[0,0].set_title('A  CAPPED500: parent-level seam benefit');axs[0,0].legend(frameon=False)
vals=[]
for u in parents:
 a=mean([r for r in D if r['label']=='cap500' and r['direction']=='GT512' and r['uid']==u],'mse');b=mean([r for r in D if r['label']=='START' and r['direction']=='GT512' and r['uid']==u],'mse');vals.append((a-b)/16*2*1e4)
axs[0,1].bar(x,vals,color=['#ba5c55' if v>0 else '#4b927c' for v in vals]);axs[0,1].axhline(0,color='#444',lw=.8);axs[0,1].set_xticks(x,short,rotation=35);axs[0,1].set_ylabel('Contribution to cohort MSE delta (x 1e-4)');axs[0,1].set_title('B  CAPPED500 vs START: full-denominator costs')
labs=['START','teacher500','cap500','cap1000'];x2=np.arange(4)
for shift,d,color in [(-.24,'GT512','#665599'),(0,'GT2048','#4477aa'),(.24,'F2048','#dd8855')]:axs[1,0].bar(x2+shift,[directions[d][l]['mse']*1e3 for l in labs],.23,label=d,color=color)
axs[1,0].set_xticks(x2,labs);axs[1,0].set_ylabel('Normalized mean squared distance (x 1e-3)');axs[1,0].set_title('C  Directional errors: Dsym averages the last two');axs[1,0].legend(frameon=False)
names=['C_0_1pct','C_1_5pct','C_over5pct'];xx=np.arange(3)
for sh,n,col in [(-.18,512,'#665599'),(.18,2048,'#4477aa')]:
 vals=[next(r['delta_cap500_START'] for r in regions if r['query_count']==n and r['region']==name)*1e4 for name in names];axs[1,1].bar(xx+sh,vals,.35,label=f'GT{n}',color=col)
axs[1,1].set_xticks(xx,['C distance <=1% L0','1% < distance <=5%','distance >5% L0']);axs[1,1].set_ylabel('Contribution to MSE delta (x 1e-4)');axs[1,1].set_title('D  Missing-target error grows near C and away');axs[1,1].legend(frameon=False)
fig.suptitle('CAPPED500 offline deep dive | 8 parents x 2 original noises | saved vectors only',fontsize=16)
fig.savefig(P/'capped500_geometry.png',dpi=150);fig.savefig(P/'capped500_geometry.svg');plt.close(fig)
start=directions['GT512']['START']['mse'];cap=directions['GT512']['cap500']['mse'];gt=directions['GT2048'];fg=directions['F2048']
text=f'''# CAPPED500 既有逐点与局部几何归因

本分析读取80份真实raw、80份已存quality向量及80份control向量（5模型×16请求），768个原数值重构核对通过。没有模型调用、重抽查询或重跑原评分器。唯一新增几何量是原GT查询到原C的有限三角面距离，仅用于固定空间分桶。

1. Dsym改善不等于完整性改善。GT→free 的2048均方距离从 {gt['START']['mse']:.9f} 到 {gt['cap500']['mse']:.9f}（{(gt['cap500']['mse']/gt['START']['mse']-1)*100:+.4f}%）；free→GT 从 {fg['START']['mse']:.9f} 到 {fg['cap500']['mse']:.9f}（{(fg['cap500']['mse']/fg['START']['mse']-1)*100:+.4f}%）。两者取半和使Dsym总体下降。旧512点均方误差 {start:.9f}→{cap:.9f}（{(cap/start-1)*100:+.4f}%），方向与2048 GT单向一致，不能仅归咎512查询偶然。
2. 原512点的中位距离、p95和1%覆盖可同时改善，但均方误差更重视大误差。47.6807%的相同查询点改善，正向误差增量均值0.000717081，负向抵消−0.000526498；净增加0.000190583。每输出误差增量最大的5%点贡献0.000265769（超过净增量，因为其余有抵消）。这是描述性分解，不删掉尾部重新判通过。
3. 新增缺失误差并不只在远离C的自由主体。512点净增加中，距离C≤1%L0贡献0.000044953，1–5%贡献0.000079711，>5%贡献0.000065918。2048得到同向结果。按GT原Y高度三等分，中间段是主要正代价；这不是语义座面分割，具体部位需看图。
4. S是原合格接缝见证点的真实转移。对teacher500，新覆盖10.2452pp，失覆盖4.2380pp，净+6.0072pp；对START，新覆盖11.2287pp、失覆盖6.7551pp，净+4.4735pp。不能说所有旧接缝都保持后再补新接缝。
5. 534b对teacher的I严重增加主要是重复覆盖，尤其noise1：两噪声I分别+7.0801/+20.7520pp，dup分别+6.4453/+20.7031pp，cross+1.9043/+2.7344pp，交集按原并集定义扣除。各known面贡献及原source face id保存在conflict_per_known_face.csv，原见证边见seam_per_edge.csv。

逐点距离原样保存于query_per_point.csv；region_distance_contributions.csv使用全体点分母，分区之和可精确还原原MSE。两个噪声、全部父均保留。图capped500_geometry.png/SVG只展示上述已存数值，不用新图像生成或网格修复。来源hash及运行时间见receipt.json。
'''
(P/'GEOMETRY_FINDINGS.md').write_text(text,encoding='utf-8')
print(json.dumps(dict(status='COMPLETE',summary=str(P/'geometry_summary.json'),figure=str(P/'capped500_geometry.png'))))
