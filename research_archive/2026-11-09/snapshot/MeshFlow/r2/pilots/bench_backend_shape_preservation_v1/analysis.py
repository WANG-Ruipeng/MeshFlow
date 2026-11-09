"""Collect fixed losses and source diagnostics without model calls."""
from __future__ import annotations
import csv, json
from collections import defaultdict
import numpy as np
from .common import *

def records(p):
 with resolve(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def num(v):
 try:return float(v)
 except (TypeError,ValueError):return None
def mean(v):
 a=[num(x) for x in v];a=[x for x in a if x is not None];return float(np.mean(a)) if a else None
def run():
 path=OUT/'analysis';path.mkdir(exist_ok=True)
 if (path/'COMPLETE.json').exists():return read(path/'COMPLETE.json')
 losses=[json.loads(x) for x in (OUT/'training/losses.jsonl').read_text().splitlines()]
 assert len(losses)==500 and [r['update'] for r in losses]==list(range(1,501))
 samples=[dict(update=r['update'],**s) for r in losses for s in r['samples']]
 csvwrite(path/'training_updates.csv',[{k:v for k,v in r.items() if k!='samples'} for r in losses]);csvwrite(path/'training_instances.csv',samples)
 ls=[]
 for lo,hi in [(1,50),(451,500)]:
  for kind in ('IDENTITY','CORNER_NOISE','VERTEX_NOISE'):
   a=[r for r in samples if lo<=r['update']<=hi and r['kind']==kind]
   ls.append(dict(first_update=lo,last_update=hi,kind=kind,samples=len(a),mse_denoiser=mean(r['mse_denoiser'] for r in a),mse_model=mean(r['mse_model'] for r in a)))
 csvwrite(path/'loss_first_last50.csv',ls)
 vals=records(OUT/'source_analysis/per_instance.csv');ss=[]
 for kind in sorted({r['kind'] for r in vals}):
  for label in ('OFFICIAL_SOURCE','CANDIDATE_SOURCE'):
   a=[r for r in vals if r['kind']==kind and r['label']==label]
   ss.append(dict(kind=kind,label=label,parents=len(a),masked_model_MSE=mean(r['masked_model_MSE'] for r in a),input_masked_model_MSE=mean(r['input_masked_model_MSE'] for r in a),below_C_proxy_area_ratio=mean(r['below_C_proxy_area_ratio'] for r in a)))
 csvwrite(path/'source_summary.csv',ss)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(9,4),dpi=160)
 for kind in ('IDENTITY','CORNER_NOISE','VERTEX_NOISE'):
  ys=[mean(r['mse_denoiser'] for r in samples if r['kind']==kind and i<=r['update']<i+20) for i in range(1,501,20)]
  ax.plot(np.arange(1,501,20)+9.5,ys,label=kind)
 ax.set_yscale('log');ax.set_xlabel('Optimizer updates');ax.set_ylabel('Masked coordinate MSE (denoiser units), 20-update mean');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(path/'training_loss.png');plt.close(fig)
 stage=records(OUT/'source_analysis/per_stage.csv');st=[]
 for label in ('OFFICIAL_SOURCE','CANDIDATE_SOURCE'):
  for s in ('SOURCE','PROPOSAL','RESTORED_C','GEO'):
   a=[r for r in stage if r['label']==label and r['stage']==s]
   st.append(dict(label=label,stage=s,parents=len(a),**{k:mean(r[k] for r in a) for k in ('free_faces','free_area','below_C_proxy_area_ratio','free_y_span')}))
 csvwrite(path/'source_stage_summary.csv',st)
 # The geometry alert is inherited from the preceding development acceptance.
 parents=records(OUT/'scores/per_parent.csv');ref={(r['parent_id'],r['stage']):r for r in parents if r['label']=='COVERAGE_FINAL'};gate=[]
 keys={'FF':'FF_illegal_affected_free_fraction','CF':'CF_illegal_affected_free_fraction','I':'ConflictCov_C'}
 for row in parents:
  if row['label']!='SHAPE_D0' or row['stage']!='CLEAN_FINAL_POST':continue
  old=ref[(row['parent_id'],'CLEAN_FINAL_POST')];item=dict(parent_id=row['parent_id'],eval_group=row['eval_group'])
  for k,f in keys.items():item[k+'_delta_pp']=(float(row[f])-float(old[f]))*100
  item['alert']=item['FF_delta_pp']>=10 or item['CF_delta_pp']>=5 or item['I_delta_pp']>=5;gate.append(item)
 csvwrite(path/'geometry_gate_per_parent.csv',gate)
 result=dict(status='COMPLETE',training_steps=500,source_summary=ss,loss_summary=ls,geometry_alert_old_parents=sum(r['alert'] and r['eval_group'] in ('CURRENT_TRAIN','DEV') for r in gate),geometry_gate_pass=sum(r['alert'] and r['eval_group'] in ('CURRENT_TRAIN','DEV') for r in gate)<2,files={str(p):sha(p) for p in path.iterdir() if p.is_file()})
 write(path/'COMPLETE.json',result);print(json.dumps({k:v for k,v in result.items() if k!='files'}));return result
if __name__=='__main__':run()
