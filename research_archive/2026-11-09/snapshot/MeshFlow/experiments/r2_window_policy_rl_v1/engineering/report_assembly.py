from pathlib import Path
import csv,json
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads((R/p).read_text(encoding='utf-8-sig'))
def rows(p):return list(csv.DictReader((R/p).open(encoding='utf-8-sig')))
def table(head,data):return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in data)+'\n'
sumry=read('evaluation/analysis/summary.json')
assert sumry['status']=='COMPLETE' and sumry['review_rows']==132
labels=['FULL_DET','FULL_STOCH','RL_STOCH'];splits=['TRAIN','DEV']
idx={(r['label'],r['split'],r['stage']):r for r in rows('evaluation/analysis/per_model_split_stage.csv')}
rev={(r['label'],r['group']):r for r in sumry['review_summaries'] if r['scope']=='SPLIT'}
main=[];quality=[];visual=[]
fields=['RAW_adoption_known','POST_adoption_known','POST_body_complete_known','joint_RAW_adoption_POST_legal_CF','joint_both_adoptions_body_and_CF']
for split in splits:
 for label in labels:
  r=idx[label,split,'POST'];main.append([split,label,r['legal_CF_outputs']+'/'+r['outputs'],int(float(r['legal_CF_edges_sum'])),format(float(r['legal_CF_length_sum']),'.5f')])
  quality.append([split,label]+[format(100*float(r[m+'_mean']),'.4f') for m in ['CF_illegal_fraction','FF_illegal_fraction','I_lower','free_small_angle_fraction']])
  visual.append([split,label]+['/'.join(str(rev[label,split][m][k]) for k in ('YES','NO','UNKNOWN')) for m in fields])
pc=rows('evaluation/analysis/per_condition_stage.csv');ct=[]
for cid in sorted({r['condition_id'] for r in pc}):
 x={r['label']:r for r in pc if r['condition_id']==cid and r['stage']=='POST'}
 ct.append([cid[:8],x[labels[0]]['split']]+[x[l]['legal_CF_outputs']+'/4' for l in labels])
pointer=read('analysis/training_latest.json');training=json.loads(Path(pointer['summary_path']).read_text())
expo=[[x['parent_id'][:8],x['completed_opportunities'],x['reward_observations'],x['reward_1'],x['active_reward_groups'],x['all_zero_groups']] for x in training['conditions']]
vp=[]
for r in sumry['paired_review_summaries']:
 if r['candidate_label']=='RL_STOCH' and r['baseline_label']=='FULL_STOCH' and r['scope']=='SPLIT' and r['group'] in splits:
  for m,n in zip(fields,['RAW采用','POST采用','POST主体','J2','J4']):
   d=r[m];vp.append([r['group'],n,d['common_known_pairs'],d['gained'],d['lost'],d['unknown_pairs'],format(100*d['full_panel_mean_delta_lower'],'.2f')+' / '+format(100*d['full_panel_mean_delta_upper'],'.2f')])
figs=[]
for p in sorted((R/'figures/scientific').glob('*/plot_data.json')):
 d=json.loads(p.read_text())
 if d.get('status')=='COMPLETE' and d.get('kind') in ['training_ledger','final_comparison']:
  for f in d['artifacts']:
   if f.endswith('.png'):figs.append('!['+d['kind']+']('+Path(f).relative_to(R).as_posix()+')')
assert len(figs)==2,figs
parts={
 'MAIN':table(['面板','模型','合法接口输出','合法共边条数','共边总长度'],main),
 'CONDITIONS':table(['父/C前缀','面板','FULL_DET','FULL_STOCH','RL_STOCH'],ct),
 'EXPOSURE':table(['TRAIN C','批机会','轨迹','在线R=1','活跃批','全零跳过'],expo),
 'VISUAL':table(['面板','模型','RAW采用','POST采用','POST主体','J2','J4'],visual),
 'QUALITY':table(['面板','模型','CF %','FF %','I %','小角 %'],quality),
 'VISUAL_INTERPRETATION':table(['面板','图审事件','共同已知配对','新增YES','丢失YES','未决配对','全分母Δ下/上界pp'],vp)+'\n以上仅比较共同已知项的确定转换，同时保留全分母未知上下界；不能把UNKNOWN填成失败/成功来放大收益。原逐例notes保留图像中用途与主体关系，像素审阅不认证真实接缝。',
 'FIGURES':'\n\n'.join(figs)}
s=(R/'engineering/REPORT_TEMPLATE.md').read_text()
for k,v in parts.items():s=s.replace('{{'+k+'}}',v)
assert '{{' not in s
p=R/'REPORT.md';assert not p.exists();p.write_text(s,encoding='utf-8')
print({'status':'REPORT_WRITTEN','path':str(p),'bytes':p.stat().st_size})
