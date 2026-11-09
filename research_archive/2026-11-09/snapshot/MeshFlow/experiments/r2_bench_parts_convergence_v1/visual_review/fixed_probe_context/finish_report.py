import csv
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
def csvread(name):
    with (OUT/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=json.loads((OUT/'summary.json').read_text())
parts=csvread('per_part.csv');parents=csvread('per_parent.csv');times=csvread('per_time.csv');conditions=csvread('per_condition.csv');deltas=csvread('condition_deltas.csv')
overall={(r['step'],r['split']):r for r in s['overall']}
parent_index={(int(r['step']),r['parent_id']):r for r in parents}
devp=sorted([r for r in parents if r['step']=='2000' and r['split']=='DEV'],key=lambda r:float(r['sse']),reverse=True)
top2=devp[:2];total=overall[2000,'DEV']
outside_mse=(total['sse']-sum(float(r['sse']) for r in top2))/(total['scalar_count']-sum(int(r['scalar_count']) for r in top2))
extra={}
for start,end in [(0,2000),(1000,2000)]:
    key=f'{start}_to_{end}'
    extra[key]={}
    for split in ('TRAIN','DEV'):
        dd=[r for r in deltas if int(r['from_step'])==start and int(r['to_step'])==end and r['split']==split]
        pp=[r for r in parents if int(r['step'])==end and r['split']==split]
        extra[key][split]=dict(conditions_better=sum(float(r['delta_mse'])<0 for r in dd),conditions_worse=sum(float(r['delta_mse'])>0 for r in dd),
            parents_better=sum(float(r['mse'])<float(parent_index[start,r['parent_id']]['mse']) for r in pp),
            parents_worse=sum(float(r['mse'])>float(parent_index[start,r['parent_id']]['mse']) for r in pp))
extra['top2_DEV_parent_step2000']={
    'parents':[r['parent_id'] for r in top2],
    'sse_share':sum(float(r['sse']) for r in top2)/total['sse'],
    'scalar_count_share':sum(int(r['scalar_count']) for r in top2)/total['scalar_count'],
    'remaining_six_parent_pooled_mse_descriptive_only':outside_mse,
    'not_a_filtering_or_replacement_rule':True}
ti={(int(r['step']),r['split'],float(r['t'])):r for r in times}
dev_delta=overall[2000,'DEV']['mse']-overall[1000,'DEV']['mse']
extra['DEV_1000_2000_time_delta_shares']={str(t):((float(ti[2000,'DEV',t]['mse'])-float(ti[1000,'DEV',t]['mse']))/3)/dev_delta for t in (.2,.5,.8)}
(OUT/'interpretation.json').write_text(json.dumps(extra,indent=2)+'\n',encoding='utf-8')

md=['# 固定监督探针：0／1000／2000步过程解释','',
 '只复算已经保存的96条固定监督探针（32条C×3个t），没有新前向、采样、评分、训练或数据筛选。它测量训练插值状态上的速度MSE，不是自由生成质量，不是唯一来源款式相似度，也不能独自证明收敛或开放生成泛化。','',
 '**结论：差距很大，但并非由一条异常条件或大面数加权制造；而且大部分差距在第0步已经存在。** 本轮0→2000步TRAIN误差继续下降、DEV总体略升，表现为现有固定探针上的差距扩大。不能把第0步已有差距全部归因于这2000步新训练，也不能从该表单独确定其数据或模型原因。','',
 '|分组|step0|step1000|step2000|固定标量分母|条件数／父数|','|---|---:|---:|---:|---:|---:|']
for split in ('TRAIN','DEV'):
    rr=[overall[k,split] for k in (0,1000,2000)]
    md.append('|'+split+'|'+'|'.join(f"{r['mse']:.6f}" for r in rr)+f"|{rr[-1]['scalar_count']}|{rr[-1]['n_conditions']}／{rr[-1]['n_parents']}|")
md += ['',
 '口径为 `Σ all_free_velocity_sse / Σ all_free_scalar_count`，不是逐条件MSE的简单平均。当前COMMON为普通Native，逐条分母严格等于 `9×(N−K)`；每条件3个固定t合计为 `27×(N−K)`。全部96条在三个checkpoint的身份、分母都一致，逐条汇总与原记录汇总一致。C不进入free分母；此阶段bulk与ALL_free相同，seam_tip计数为0，属于NA而非零误差成绩。','',
 '作为敏感性检查，DEV条件等权均值从0.940292→0.939182→0.982311，父等权均值从0.917884→0.923792→0.969653；同样显示后半段上升。这些等权结果只作解释，不替换预注册的SSE/有效坐标口径。','',
 '1000→2000步：DEV有11/16条件、6/8父变差；TRAIN有14/16条件、11/12父改善。DEV中位条件MSE从0.937253升到0.980057，因此不是只有最坏一条在动。','',
 '按已复核的部位标签汇总如下（括号为条件数；条件ID保留历史命名，不能用ID中的SEAT/LEG字符串覆盖复核后的part字段）。每格仍是该部位的ΣSSE/Σ标量数。','',
 '|分组／部位（C数）|step0|step1000|step2000|','|---|---:|---:|---:|']
pi={(int(r['step']),r['split'],r['part']):r for r in parts}
for split in ('TRAIN','DEV'):
    for p in sorted({r['part'] for r in parts if r['split']==split}):
        rs=[pi[k,split,p] for k in (0,1000,2000)]
        md.append(f"|{split} / {p}（{rs[-1]['n_conditions']}）|"+'|'.join(f"{float(r['mse']):.6f}" for r in rs)+'|')
md += ['',
 'DEV的5个已有部位分层在1000→2000步均上升；其中SEAT_SUPPORT总体误差最高，但只有2条条件，不应外推成整个部位必然困难。TRAIN所有部位分层均下降。','',
 '|DEV父前缀|N|step0|step1000|step2000|2000步SSE占比|','|---|---:|---:|---:|---:|---:|']
for r in sorted(devp,key=lambda r:r['parent_id']):
    rs=[parent_index[k,r['parent_id']] for k in (0,1000,2000)]
    md.append(f"|{r['parent_id'][:6]}|{r['min_N']}|"+'|'.join(f"{float(v['mse']):.6f}" for v in rs)+f"|{100*float(r['sse'])/total['sse']:.2f}%|")
md += ['',
 f"两个高误差父9d5608、fd355d合计占DEV约{100*extra['top2_DEV_parent_step2000']['sse_share']:.2f}%的SSE，但只占{100*extra['top2_DEV_parent_step2000']['scalar_count_share']:.2f}%的标量分母。即使仅作描述地查看其余6父，合并MSE仍为{outside_mse:.6f}，明显高于TRAIN的0.115627。因此有尾部集中，但不是去掉一两个父就能解释全部差距；这里不提出排除任何父的建议。",
 '',
 '面数也不是充分解释：最高误差父fd355d只有N=154，9d5608为N=244；N=224的5dda4e却只有0.112980。2000步DEV每条件最大分母权重仅8.94%，条件MSE与N的描述性Pearson相关约−0.194，与free面数约−0.293；样本很小，不能把相关当因果，但不支持“大N加权把均值虚抬”的解释。','',
 '1000→2000步DEV净增SSE为3245.15。最大单条件贡献是9d5608的历史 `SEAT_SURFACE_1` ID（当前复核part为BACK_SURFACE），贡献全DEV MSE增量+0.013505；其次fd355d组合条件+0.007227、459f90座面+0.007028。前3条占正向增量约57.49%，同时还有其他条件退化与5条改善抵消，不能只围绕这3条讲故事。','',
 '|分组／t|step0|step1000|step2000|','|---|---:|---:|---:|']
for split in ('TRAIN','DEV'):
    for t in (.2,.5,.8):
        md.append(f'|{split} / {t}|'+'|'.join(f"{float(ti[k,split,t]['mse']):.6f}" for k in (0,1000,2000))+'|')
md += ['',
 f"DEV的1000→2000步增量中，t=0.8贡献约{100*extra['DEV_1000_2000_time_delta_shares']['0.8']:.2f}%；该时间层在step0就高于另两个时间层。TRAIN三个时间层均下降。这说明差距并非只发生在高噪声起点，但本次只给出已有记录的分层现象，不新增机制归因。",
 '',
 '完整逐条件、逐父、逐部位、逐t及配对增量分别保存在同目录CSV；`summary.json`记录原始文件hash、分子分母核对及描述性统计。正式训练继续遵守已冻结预算与停止规则，本材料不改变父对象、条件、噪声、权重选择或训练定义。','']
(OUT/'CONTEXT.md').write_text('\n'.join(md),encoding='utf-8')
receipt=json.loads((OUT/'receipt.json').read_text())
receipt['files'].update({str(OUT/n):sha(OUT/n) for n in ('CONTEXT.md','interpretation.json','finish_report.py')})
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(extra,indent=2))
