"""Final report from immutable diagnostics; never trains, samples or scores."""
from pathlib import Path
import argparse,json,numpy as np
from .common import read_json,write_json,file_sha256
RMS='remainder_symmetric_RMS_bbox_pct';GAMMA='boundary_RMS_bbox_pct';P95='boundary_p95_bbox_pct';FF='FF_illegal_pair_count'

def mean(rows,key):return float(np.mean([r['scalars'][key] for r in rows]))
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x) for x in row)+' |' for row in rows])
def f(x):return f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x)

def run(out):
 out=Path(out);notes=read_json(out/'interpretation.json');A={m:read_json(out/'scores'/('A_'+m)/'results.json')['rows'] for m in ('B','J')}
 configs=[('bf16',50),('bf16',100),('fp32',50),('fp32',100)];a_table=[];a_stats=[]
 for pr,n in configs:
  b=[r for r in A['B'] if r['precision']==pr and r['nfe']==n];j=[r for r in A['J'] if r['precision']==pr and r['nfe']==n]
  idx={(r['parent'],r['patch'],r['seed']):r for r in b};d=[r['scalars'][RMS]-idx[r['parent'],r['patch'],r['seed']]['scalars'][RMS] for r in j]
  row=dict(precision=pr,nfe=n,B_RMS=mean(b,RMS),J_RMS=mean(j,RMS),delta_mean=float(np.mean(d)),delta_median=float(np.median(d)),J_lower=sum(v<0 for v in d),B_FF=mean(b,FF),J_FF=mean(j,FF))
  a_stats.append(row);a_table.append([pr,n,*[f(row[k]) for k in ('B_RMS','J_RMS','delta_mean','delta_median')],str(row['J_lower'])+'/20',f(row['B_FF']),f(row['J_FF'])])
 b_stats=[]
 for cut in (20,50,80):
  for source in ('B','J'):
   continuation='J' if source=='B' else 'B'
   cross=[r for r in read_json(out/'scores'/('B_'+continuation)/'results.json')['rows'] if r['start_step']==cut]
   idx={(r['parent'],r['patch'],r['seed']):r for r in A[source] if r['precision']=='fp32' and r['nfe']==100}
   sign=1 if source=='B' else -1
   d={k:[sign*(r['scalars'][k]-idx[r['parent'],r['patch'],r['seed']]['scalars'][k]) for r in cross] for k in (RMS,FF,P95)}
   b_stats.append(dict(source=source,cut=cut,mean_RMS=float(np.mean(d[RMS])),median_RMS=float(np.median(d[RMS])),J_worse=sum(x>0 for x in d[RMS]),mean_FF=float(np.mean(d[FF])),mean_Gamma_p95=float(np.mean(d[P95]))))
 c_stats=[]
 for parent in (0,1):
  for group in ('training_C','other_observed_C'):
   row=dict(parent=parent,group=group)
   for precision in ('bf16','fp32'):
    br=read_json(out/'stage_c/teacher'/('B_'+precision+'.json'))['rows'];jr=read_json(out/'stage_c/teacher'/('J_'+precision+'.json'))['rows']
    br=[r for r in br if r['parent']==parent and r['context_group']==group and r['teacher_t']>=.5];jr=[r for r in jr if r['parent']==parent and r['context_group']==group and r['teacher_t']>=.5]
    for metric in ('FM','joint_actual'):
     bm=np.mean([r['losses'][metric] for r in br]);jm=np.mean([r['losses'][metric] for r in jr]);row[precision+'_'+metric+'_relative_percent']=float(100*(jm/bm-1))
   c_stats.append(row)
 d=read_json(out/'analysis/D.json');replays={a:read_json(out/'stage_d'/a/'replay.json') for a in ('R_B','R_J')}
 full={m+'_raw3000':[r for r in A[m] if r['precision']=='bf16' and r['nfe']==50] for m in ('B','J')};curve={}
 if d['status']=='PASS':
  for label in ('G0','R_B_raw250','R_J_raw250','R_B_EMA250','R_J_EMA250','R_B_EMA500','R_J_EMA500'):
   rows=read_json(out/'scores'/('D_'+label)/'results.json')['rows']
   if label.endswith('500'):full[label]=rows
   else:curve[label]=rows
  for label,rows in full.items():curve[label]=[r for r in rows if r['patch'] in (0,4) and r['seed']==10401]
 d_stats=[dict(label=k,**{m:mean(v,m) for m in (RMS,'free_to_R12_RMS_bbox_pct','R12_to_free_RMS_bbox_pct',GAMMA,P95,FF,'CF_proper_pair_count','free_total_surface_area')}) for k,v in full.items()]
 curve_stats=[dict(label=k,**{m:mean(v,m) for m in (RMS,GAMMA,P95,FF)}) for k,v in curve.items()]
 counts=dict(A_formal_full=168,A_production_repeat=1,B_cross_suffix=72,B_same_model_suffix=6,C_teacher_forward=400,C_parameter_forward=8,C_velocity_VJP=1200,C_parameter_VJP=24,C_OT=20,D_updates=sum(r['counts'].get('optimizer_updates',0) for r in replays.values()),D_training_forward=sum(r['counts'].get('forward_returns',0) for r in replays.values()),D_training_backward=sum(r['counts'].get('backward_returns',0) for r in replays.values()),D_OT=sum(r['counts'].get('OT_returns',0) for r in replays.values()))
 evals=[read_json(p) for p in sorted((out/'D_evaluation').glob('*/results.json'))]
 counts.update(D_complete_full=sum(r['counts']['complete_rollouts'] for r in evals),D_sampling_forward=sum(r['counts']['model_forward_returns'] for r in evals),D_teacher_forward=sum(r['teacher_counts'].get('teacher_forward_returns',0) for r in evals))
 counts['total_full']=169+counts['D_complete_full'];counts['total_model_forward']=13650+3900+408+counts['D_training_forward']+counts['D_sampling_forward']+counts['D_teacher_forward'];counts['sampling_failures']=0
 cp=[]
 for arm,r in replays.items():
  for step,kinds in r['checkpoints'].items():
   for kind,v in kinds.items():cp.append(dict(v,step=int(step)))
 result=dict(schema='jedge5_diagnosis_final_v1',status='COMPLETE' if d['status']=='PASS' else 'COMPLETE_WITH_D_LIMITATION',protocol='registration/protocol.json',actual_counts=counts,A_configuration_comparison=a_stats,B_same_source_comparison=b_stats,C_active_teacher=c_stats,D_endpoint20=d_stats,D_curve4=curve_stats,checkpoints=cp,findings=notes,
  references=dict(A='analysis/A/results.json',B='analysis/B/results.json',C='stage_c/analysis.json',D='analysis/D.json',binding='analysis/binding.json',audit='final_audit.json',timing='timing.json',figures='figures',diff='code.diff'),default_model_changed=False,committed_or_pushed=False)
 write_json(out/'results.json',result)
 lines=['# JEdge5 工作条件诊断报告','',notes['summary'],'',
  '本轮按 A→B→C→D 执行。两个已见父对象、十个已观察上下文，固定新生成噪声 10401/10402；patch0 是原微调条件，patch1–4 是已见对象上的其他上下文。两噪声不能支持显著性、尾部概率或新对象泛化声明。',
  '', '## 执行与保护', '',
  'B 是 T1+Geo+FM 累计3000步；J 是同架构 JEdge5 累计3000步。共同重放起点 G0 累计2500步。checkpoint、输入数组和架构严格加载；身份、文件哈希见 registration/protocol.json。原生产采样器、默认FM、旧权重及历史结果保持不变；未提交或推送 Git。',
  '', '严格确定性、TF32关闭、SDPA math、FP32参数与积分保持。BF16/50独立入口与原生产采样器的51状态逐位一致；FP32路径的嵌入、QKV、注意力、MLP和速度输出实际均为FP32。六条同模型后缀恢复整段轨迹逐位一致。历史BF16置换误差沿用 baseline-inheritance，本轮没有重测或把旧FAIL改为PASS。',
  '',table(['实际调用','次数'],[[k,v] for k,v in counts.items()]),
  '', 'A/B/C optimizer更新为0；D每臂500，总1000。C的1200次velocity VJP与24次双参数VJP分开计数；D的8000次微批反向分开计数。完整生成与后缀续跑、复用行均未混计。CPU夹具、一次历史输出评分回归和静态CC审计另记于测试/审计记录。',
  '', '## Q1：J−B 差距是否依赖步长或BF16？', '',notes['q1'],'',
  'RMS单位为bbox百分比；差值单位为百分点。先逐case配对后统计，中位差不是两模型各自中位数之差。FF列为平均非法对数。','',
  table(['精度','Euler步数','B RMS','J RMS','J−B均值','J−B中位差','J较低','B FF','J FF'],a_table),
  '', '全部双向误差、固定覆盖、Gamma到表面/有限边、CF/FF、面积与角度指标及parent/patch/seed分组见 analysis/A。没有另造总分。',
  '', '## Q2：较细积分是否收敛？', '',notes['q2'],
  '', '## Q3：前段状态还是后段速度场？', '',notes['q3'],'',
  '下表始终比较同一来源状态上“J续跑−B续跑”。正RMS表示J后段较差。B→B/J→J复用A原终点；其余仅执行剩余80/50/20步。','',
  table(['来源','切点t','RMS均值差','RMS中位差','J较差/12','FF均值差','Gamma p95均值差'],[[r['source'],r['cut']/100,f(r['mean_RMS']),f(r['median_RMS']),r['J_worse'],f(r['mean_FF']),f(r['mean_Gamma_p95'])] for r in b_stats]),
  '', '所有中间状态是带噪Xt，未按最终网格质量判失败。交叉续跑混合了两个已训练速度场，不是标准模型分布，也不提供严格因果分解。',
  '', '## Q4：监督收益是否只在训练C？', '',notes['q4'],'',
  '下表只统计t≥.5固定teacher状态，列出J相对B均值的变化百分比；联合项使用原Edge/L5系数，诊断ramp=1。训练时间分布没有改变。','',
  table(['父对象','上下文','BF16 FM变化%','BF16联合变化%','FP32 FM变化%','FP32联合变化%'],[[r['parent'],r['group'],*[f(r[k]) for k in ('bf16_FM_relative_percent','bf16_joint_actual_relative_percent','fp32_FM_relative_percent','fp32_joint_actual_relative_percent')]] for r in c_stats]),
  '', 'teacher Gaussian/GT配对与自由生成噪声独立，不把teacher与生成样本逐个当作同一轨迹。完整400行损失/几何/梯度表见 stage_c/teacher_context_time.csv。',
  '', '## Q5：标量和两种梯度空间是否一致？', '',notes['q5'],
  '', '## Q6：raw重放与匹配EMA对照', '',notes['q6'],'',
  'D沿用fresh AdamW(lr1e−5, betas=.9/.95, wd0, clip1)、batch8/micro1和原流seed1010/batch1000/sample8000。Edge=4.071385484299878，L5=0.3266104383520167，t≥.5门控和50更新ramp均未更改。独立FP32 EMA半衰期100，初始为G0，500次后初始系数1/32；仅成功optimizer.step后更新，未进入raw计算。',
  '',table(['模型','RMS','free→GT','GT→free','Gamma RMS','Gamma p95','FF','CF','free面积'],[[r['label'],*[f(r[k]) for k in (RMS,'free_to_R12_RMS_bbox_pct','R12_to_free_RMS_bbox_pct',GAMMA,P95,FF,'CF_proper_pair_count','free_total_surface_area')]] for r in d_stats]),
  '', '固定四case曲线（两parent×patch0/4×10401），与上方20case均值不能直接横比：','',
  table(['checkpoint','RMS','Gamma RMS','Gamma p95','FF'],[[r['label'],*[f(r[k]) for k in (RMS,GAMMA,P95,FF)]] for r in curve_stats]),
  '', 'checkpoint本地路径、大小、文件SHA及state SHA见 results.json/checkpoints；未复制官方大权重。20状态teacher曲线、raw/EMA匹配差分与逐case变化见 analysis/D.json、analysis/D_per_case.csv。四case不能证明训练收敛或最佳EMA尺度。',
  '', '## Q7：支持什么，尚不能说明什么？', '',notes['q7'],
  '', '## 质量、资源与交付', '',notes['quality'],
  '', 'CF/FF沿用原独立浮点检测器及阈值，未执行的精确谓词、精确拓扑/水密认证明确NOT_RUN。已知C保持和112面数逐轨迹断言；没有Pair、Stage2、Guidance、焊接或几何修复。条件绑定四格复用输出评分，见 analysis/binding.csv；绑定成立不等于候选有新增质量收益。',
  '', 'GPU并发按首个正式任务实测递增，采样和训练分别估算，预留至少2GiB；详情见 registration/resource_snapshots.jsonl 与 dispatch.jsonl。WDDM下进程显存查询不可得，记录全卡占用增量估计和PyTorch allocated/reserved峰值，未把allocated当完整进程占用。并发时长不是隔离算力基准。网络前向CUDA事件、host时间、显式同步、轨迹保存、CPU OT、训练更新、EMA更新、checkpoint保存与离线几何审计在 timing.json 分账；嵌套计时不相加为总墙钟时间。',
  '', '固定首seed同视角full/free图见 figures/A_*（全部十个上下文）及 figures/D_*（四个预登记曲线case）；完整边界统一范围，未删除不利案例。代码补丁code.diff、源快照、协议、必要测试和逐case表均在本目录。',
  '', '本轮已停止：未改loss、lambda、EMA半衰期或默认权重，未延长至3500步，未启动下一阶段。','']
 (out/'JEDGE5_DIAGNOSIS_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
 print(json.dumps(dict(status=result['status'],actual_counts=counts),ensure_ascii=False))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)

