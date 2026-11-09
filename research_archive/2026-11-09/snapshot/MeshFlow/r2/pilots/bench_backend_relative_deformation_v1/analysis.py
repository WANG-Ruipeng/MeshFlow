"""Summarize fixed candidate losses and paired source/real-output evidence."""
from __future__ import annotations
from .common import *

def records(p):
 with resolve(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def mean(xs):
 vals=[float(x) for x in xs if x not in ('',None)]
 return float(np.mean(vals)) if vals else None
def run():
 dest=OUT/'analysis';done=dest/'COMPLETE.json'
 if done.exists():r=read(done);verify(r['files']);return r
 dest.mkdir(exist_ok=True)
 losses=[json.loads(x) for x in (OUT/'training/losses.jsonl').read_text().splitlines()]
 assert [r['update'] for r in losses]==list(range(1,501))
 samples=[dict(update=r['update'],**s) for r in losses for s in r['samples']]
 assert len(samples)==2000
 csvwrite(dest/'training_updates.csv',[{k:v for k,v in r.items() if k!='samples'} for r in losses])
 csvwrite(dest/'training_instances.csv',samples)
 lsum=[]
 for lo,hi in [(1,50),(451,500)]:
  for kind in ('IDENTITY','CORNER_NOISE','VERTEX_NOISE'):
   a=[r for r in samples if lo<=r['update']<=hi and r['kind']==kind]
   lsum.append(dict(first_update=lo,last_update=hi,kind=kind,instances=len(a),**{k:mean(r[k] for r in a) for k in ('mse_denoiser','mse_model','relative_huber','weighted_relative','total_loss')}))
 csvwrite(dest/'loss_first_last50.csv',lsum)
 old=[json.loads(x) for x in (PREV/'training/losses.jsonl').read_text().splitlines()]
 old_samples=[dict(update=r['update'],**s) for r in old for s in r['samples']]
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(12,4.5),dpi=160,layout='constrained')
 for kind,color in zip(('IDENTITY','CORNER_NOISE','VERTEX_NOISE'),('tab:blue','tab:orange','tab:green')):
  xs=np.arange(1,501,20)+9.5
  for data,style,label in ((old_samples,'--','SHAPE'),(samples,'-','RELDEF')):
   ys=[mean(r['mse_denoiser'] for r in data if r['kind']==kind and i<=r['update']<i+20) for i in range(1,501,20)]
   axes[0].plot(xs,ys,style,color=color,label=label+' '+kind)
  axes[1].plot(xs,[mean(r['relative_huber'] for r in samples if r['kind']==kind and i<=r['update']<i+20) for i in range(1,501,20)],color=color,label=kind)
 for ax in axes:ax.set_yscale('log');ax.set_xlabel('Optimizer update');ax.grid(alpha=.2);ax.legend(fontsize=7)
 axes[0].set_ylabel('Coordinate MSE (denoiser units)');axes[1].set_ylabel('Relative radial Huber (dimensionless)')
 fig.savefig(dest/'training_loss.png');plt.close(fig)
 vals=records(OUT/'source_analysis/per_instance.csv');source=[]
 available=sorted({r['label'] for r in vals})
 for kind in sorted({r['kind'] for r in vals}):
  for label in available:
   a=[r for r in vals if r['kind']==kind and r['label']==label]
   numeric_keys=[k for k in ('masked_model_MSE','input_masked_model_MSE','below_C_proxy_area_ratio','relative_huber_mean','relative_r_mean','relative_r_p95') if k in a[0]]
   source.append(dict(kind=kind,label=label,parents=len(a),**{k:mean(r[k] for r in a) for k in numeric_keys}))
 csvwrite(dest/'source_summary.csv',source)
 stage=records(OUT/'source_analysis/per_stage.csv');st=[]
 for label in sorted({r['label'] for r in stage}):
  for s in ('SOURCE','PROPOSAL','RESTORED_C','GEO'):
   a=[r for r in stage if r['label']==label and r['stage']==s]
   assert len(a)==15,(label,s,len(a))
   st.append(dict(label=label,stage=s,parents=len(a),**{k:mean(r[k] for r in a) for k in ('free_faces','free_area','below_C_proxy_area_ratio','free_y_span')}))
 csvwrite(dest/'source_stage_summary.csv',st)
 parents=records(OUT/'scores/per_parent.csv')
 refs={(r['label'],r['parent_id'],r['stage']):r for r in parents};gate=[]
 fields={'FF':'FF_illegal_affected_free_fraction','CF':'CF_illegal_affected_free_fraction','I':'ConflictCov_C'}
 for row in parents:
  if row['label']!='RELDEF_D0' or row['stage']!='CLEAN_FINAL_POST':continue
  for label in ('COVERAGE_FINAL','SHAPE_D0'):
   old=refs[(label,row['parent_id'],'CLEAN_FINAL_POST')]
   item=dict(parent_id=row['parent_id'],eval_group=row['eval_group'],reference_label=label)
   for k,f in fields.items():item[k+'_delta_pp']=(float(row[f])-float(old[f]))*100
   item['alert']=item['FF_delta_pp']>=10 or item['CF_delta_pp']>=5 or item['I_delta_pp']>=5
   gate.append(item)
 csvwrite(dest/'geometry_gate_per_parent.csv',gate)
 alerts={l:sum(r['alert'] and r['eval_group'] in ('CURRENT_TRAIN','DEV') and r['reference_label']==l for r in gate) for l in ('COVERAGE_FINAL','SHAPE_D0')}
 result=dict(status='COMPLETE',training_steps=500,source_summary=source,loss_summary=lsum,geometry_alert_old_parents=alerts['COVERAGE_FINAL'],geometry_gate_pass=alerts['COVERAGE_FINAL']<2,alerts_by_reference=alerts,
  files={str(p):sha(p) for p in dest.iterdir() if p.is_file()})
 write(done,result);print(json.dumps({k:v for k,v in result.items() if k!='files'}));return result
if __name__=='__main__':run()
