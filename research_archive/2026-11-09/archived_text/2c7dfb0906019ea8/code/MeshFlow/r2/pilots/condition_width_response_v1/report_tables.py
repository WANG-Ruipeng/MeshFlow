"""Summarize existing scalar tables; no model/geometry execution."""
import csv,json,collections
from pathlib import Path
import numpy as np
P=Path('/mnt/e/MeshFlow/experiments/r2_condition_width_response_v1')
def rd(name):
 with (P/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
a=rd('evaluation/per_output.csv');curves=rd('analysis/per_curve_response.csv')
names={'CF_pct':'CF_illegal_affected_free_fraction','FF_pct':'FF_illegal_affected_free_fraction','I_pct':'ConflictCov_C','AVMR_strict_pct':'A_VMR__raw_tau_1e-6','PMR_strict_pct':'PMR__raw_tau_1e-6','small_angle_pct':'free_minimum_angle_lt5_fraction','degenerate_pct':'free_near_degenerate_fraction','boundary_vertex_mean':'boundary_distance_mean_model','boundary_vertex_p95':'boundary_distance_p95_model','boundary_surface_mean':'DESC__C_boundary_to_F_surface_mean_model','boundary_surface_p95':'DESC__C_boundary_to_F_surface_p95_model','boundary_edge_mean':'DESC__C_boundary_to_F_all_finite_edges_mean_model','boundary_edge_p95':'DESC__C_boundary_to_F_all_finite_edges_p95_model','valid_edges':'C_F_valid_edge_count','oriented_reach_pct':'opposite_winding_reached_free_area_fraction','retained_F':'actual_free_faces'}
rows=[]
for stage in ('RAW','POST'):
 for variant in ('m095','base','p105'):
  rr=[r for r in a if r['stage']==stage and r['variant']==variant];assert len(rr)==16
  item=dict(stage=stage,variant=variant,outputs=16,parents=8)
  for key,name in names.items():
   by=collections.defaultdict(list)
   for r in rr:
    if r[name]!='':by[r['parent_id']].append(float(r[name]))
   values=[np.mean(x) for x in by.values()]
   item[key]=float(np.mean(values))*(100 if key.endswith('_pct') else 1) if values else None
   item[key+'_valid_parents']=len(values)
  item['outputs_with_legal_edge']=sum(int(r['C_F_has_valid_edge']) for r in rr)
  item['total_legal_edges']=sum(int(r['C_F_valid_edge_count']) for r in rr)
  item['parents_with_legal_edge']=len({r['parent_id'] for r in rr if int(r['C_F_has_valid_edge'])})
  rows.append(item)
perparent=[]
for parent in dict.fromkeys(r['parent_id'] for r in a):
 for stage in ('RAW','POST'):
  cs=[r for r in curves if r['parent_id']==parent and r['stage']==stage]
  rr=[r for r in a if r['parent_id']==parent and r['stage']==stage]
  item=dict(parent_id=parent,kind=cs[0]['region_type']+'_'+cs[0]['tier'],stage=stage,K=int(cs[0]['K']),N=int(cs[0]['N']),width_slope_mean=np.mean([float(c['width_slope']) for c in cs]),width_monotone=sum(c['width_monotone']=='True' for c in cs))
  for v in ('m095','base','p105'):
   r0=[r for r in rr if r['variant']==v]
   for key in ('CF_pct','FF_pct','I_pct','boundary_surface_mean','boundary_vertex_mean'):
    item[v+'_'+key]=np.mean([float(r[names[key]]) for r in r0])*(100 if key.endswith('_pct') else 1)
   item[v+'_legal_outputs']=sum(int(r['C_F_has_valid_edge']) for r in r0)
  perparent.append(item)
for name,rr in [('report_quality_summary.csv',rows),('report_parent_summary.csv',perparent)]:
 with (P/'analysis'/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
print(json.dumps(rows,ensure_ascii=False,indent=2))
print('LEGAL OUTPUTS:',json.dumps([{k:r[k] for k in ('request_id','stage','C_F_valid_edge_count','opposite_winding_reached_free_area_fraction')} for r in a if int(r['C_F_has_valid_edge'])],indent=2))
