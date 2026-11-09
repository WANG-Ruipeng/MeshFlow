"""Derived presentation tables and learning figures. No model or metric changes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import csv,collections,json,math
from pathlib import Path
import numpy as np
from . import common as C
OUT=C.OUT/'report_tables'
def readrows(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def number(v):
 try:return float(v) if v not in ('',None) else None
 except (TypeError,ValueError):return None
def mean(group,field):
 a=[number(r.get(field)) for r in group];a=[v for v in a if v is not None];return float(np.mean(a)) if a else None

def parent_mean(group,field):
 parents=collections.defaultdict(list)
 for r in group:parents[(r['label'],r['parent_id'])].append(r)
 generators=collections.defaultdict(list)
 for (label,parent),rows in parents.items():
  value=mean(rows,field)
  if value is not None:generators[label].append(value)
 values=[float(np.mean(v)) for v in generators.values() if v]
 return float(np.mean(values)) if values else None

def main():
 eq=C.OUT/'evaluation';q=readrows(eq/'quality_per_output.csv');top=readrows(eq/'topology_per_output.csv');vm=readrows(eq/'per_output.csv');mods=readrows(eq/'modifications_per_output.csv')
 assert len(q)==144 and len(top)==144
 fields=['CF_proper_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','ConflictCov_C_cross','ConflictCov_C_dup','free_minimum_angle_lt5_fraction','free_q_gt10_fraction','free_near_degenerate_fraction','FF_illegal_pair_count','CF_illegal_pair_count']
 tf=['C_F_valid_edge_count','C_F_valid_opposite_winding_edge_count','reached_free_area_fraction','reached_free_faces']
 # Full tables are authoritative; pool noise within parent, parents within generator, then generators.
 result=[]
 for scope in ['ALL']+sorted({r['label'] for r in q}):
  for method in ('IDENTITY_GEO','D0_GEO','STUDENT_GEO'):
   choose=lambda rows:[r for r in rows if r['method']==method and (scope=='ALL' or r['label']==scope)]
   group=choose(q);tg=choose(top);mg=choose(mods)
   result.append(dict(scope=scope,method=method,outputs=len(group),pooling='noise within parent, equal parents within generator, equal generators',**{f:parent_mean(group,f) for f in fields},**{f:parent_mean(tg,f) for f in tf},**{f:parent_mean(mg,f) for f in ('deleted_free_faces','deleted_free_fraction','POST_free_surface_area')}))
 C.csvwrite(OUT/'quality_primary.csv',result)
 table=[]
 for scope in ['ALL']+sorted({r['label'] for r in q}):
  for setting in sorted({r['setting'] for r in vm}):
   for method in ('IDENTITY_GEO','D0_GEO','STUDENT_GEO'):
    rows=[r for r in vm if r['method']==method and r['stage']=='POST' and r['setting']==setting and (scope=='ALL' or r['label']==scope)]
    table.append(dict(scope=scope,method=method,setting=setting,outputs=len(rows),pooling='noise within parent, equal parents within generator, equal generators',A_VMR=parent_mean(rows,'VMR'),PMR=parent_mean(rows,'perfect')))
 C.csvwrite(OUT/'AVMR_PMR_all8.csv',table)
 for filename in ('FF_fates_per_output','GEO_sensitivity_per_output'):
  rows=readrows(eq/(filename+'.csv'));names=[k for k in rows[0] if any(number(r.get(k)) is not None for r in rows)]
  groups=collections.defaultdict(list)
  for r in rows:groups[r['method'],r['label']].append(r)
  C.csvwrite(OUT/(filename+'_summary.csv'),[dict(method=m,label=l,outputs=len(v),**{k+'_mean':mean(v,k) for k in names}) for (m,l),v in sorted(groups.items())])
 paired=readrows(eq/'paired_per_output.csv');direction={f+'_delta':-1 for f in fields};direction.update({f+'_delta':1 for f in tf})
 counts=[]
 for scope in ['ALL']+sorted({r['label'] for r in q}):
  for comparison in sorted({r['comparison'] for r in paired}):
   g=[r for r in paired if r['comparison']==comparison and (scope=='ALL' or r['label']==scope)]
   for field,sign in direction.items():
    vals=[number(r.get(field)) for r in g];vals=[v for v in vals if v is not None]
    counts.append(dict(scope=scope,comparison=comparison,metric=field,outputs=len(vals),preferred_direction='higher' if sign==1 else 'lower',improved=sum(v*sign>1e-12 for v in vals),tied=sum(abs(v)<=1e-12 for v in vals),worse=sum(v*sign< -1e-12 for v in vals),delta_mean=float(np.mean(vals)) if vals else None))
 C.csvwrite(OUT/'paired_direction_counts.csv',counts)
 phases=[C.read(p) for p in sorted((C.OUT/'gpu_phases').glob('*.json'))]
 C.csvwrite(OUT/'phase_resources.csv',[dict(phase=p['name'],status=p['status'],seconds=p['gpu_occupancy_seconds'],**p['counts']) for p in phases])
 # Standard static scientific plots, no generated mesh/image content.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 curves=readrows(C.OUT/'training/learning_curve.csv');monitor=readrows(C.OUT/'monitor/summary.csv')
 fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
 x=np.array([int(r['step']) for r in curves]);window=100
 for name,color in [('real','#2874a6'),('synthetic','#d68910')]:
  for ax,field in [(axes[0],name+'_E'),(axes[1],name+'_R')]:
   y=np.array([float(r[field]) for r in curves]);xx=[x[i:i+window].mean() for i in range(0,len(x),window)];yy=[y[i:i+window].mean() for i in range(0,len(y),window)]
   ax.plot(xx,yy,label=name,color=color)
 for split,domain,style in [('internal_val','real','o-'),('fixed_training','real','s--'),('fixed_training','synthetic','^--')]:
  rows=[r for r in monitor if r['split']==split and r['domain']==domain]
  axes[2].plot([int(r['step']) for r in rows],[float(r['pooled_R']) for r in rows],style,label=split+'/'+domain)
 axes[0].set_title('Update-time KD E (100-update means)');axes[0].set_yscale('log')
 axes[1].set_title('Update-time E/B (not endpoint eval)');axes[1].set_yscale('log')
 axes[2].set_title('Fixed monitoring inputs: pooled E/B');axes[2].set_yscale('log')
 for ax in axes:
  ax.set_xlabel('Optimizer update');ax.grid(True,alpha=.25);ax.legend(fontsize=7)
 axes[1].axhline(1,color='gray',lw=1);axes[2].axhline(1,color='gray',lw=1)
 image=C.checked(OUT/'learning_curves.png');image.parent.mkdir(parents=True,exist_ok=True);fig.savefig(image,dpi=160);plt.close(fig)
 C.write(OUT/'completion.json',dict(status='PASS',quality_outputs=144,all_thresholds=8,new_model_calls=0,aggregation='Noise within parent, equal parents within generator, equal generators; six repeated DEV parents, no independent48-parent CI',sources={str(p):C.sha(p) for p in [eq/'quality_per_output.csv',eq/'topology_per_output.csv',eq/'per_output.csv',C.OUT/'training/learning_curve.csv',C.OUT/'monitor/summary.csv']}))
 print('Presentation tables and learning curves complete')
if __name__=='__main__':main()