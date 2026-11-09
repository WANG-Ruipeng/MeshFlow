"""Report-only reductions of completed frozen scores; no model or geometry calls."""
from pathlib import Path
import csv, collections, json, sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from r2.pilots.anchor_aware_denoiser_v1 import common as C
DEST=C.OUT/'report_tables'
ARMS=('D0_OFFICIAL',*C.ARMS)
GENS=('R2_FROZEN','R2_CONT_OLD','R2_CONT_USER','R2_USER_MIX')
def read(name):
 with (C.OUT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def number(row,key):return float(row[key]) if row.get(key) not in ('',None) else None
def avg(rows,key):
 vals=[number(r,key) for r in rows]
 return float(np.mean(vals)) if vals and all(v is not None for v in vals) else None
def total(rows,key):return sum(number(r,key) for r in rows)
def ident(r):return r['denoiser'],r['label'],r['request_id']
def md(rows,cols):
 def fmt(x):
  if x is None:return 'NA'
  if isinstance(x,float):return f'{x:.5g}'
  return str(x)
 return '\n'.join(['| '+' | '.join(cols)+' |','|'+'|'.join('---' for _ in cols)+'|']+['| '+' | '.join(fmt(r.get(k)) for k in cols)+' |' for r in rows])
def main():
 assert C.read(C.OUT/'evaluation/completion.json')['status']=='PASS'
 assert C.read(C.OUT/'synthetic_results/aggregation.json')['status']=='COMPLETE'
 q=read('evaluation/quality_per_output.csv');top=read('evaluation/topology_per_output.csv')
 mod=read('evaluation/modifications_per_output.csv');vm=read('evaluation/per_output.csv')
 dist=read('evaluation/boundary_distance_per_output.csv')
 assert len(q)==len(top)==len(mod)==144
 top={ident(r):r for r in top};mod={ident(r):r for r in mod}
 vm={(ident(r),r['stage'],r['setting']):r for r in vm}
 dist={(ident(r),r['stage']):r for r in dist}
 enriched=[]
 quality_fields={'CF_pct':'CF_proper_affected_free_fraction','FF_pct':'FF_illegal_affected_free_fraction','I_pct':'ConflictCov_C','I_cross_pct':'ConflictCov_C_cross','I_dup_pct':'ConflictCov_C_dup','I_intersection_pct':'ConflictCov_C_cross_and_dup','small_angle_pct':'free_minimum_angle_lt5_fraction','q_gt10_pct':'free_q_gt10_fraction','near_degenerate_pct':'free_near_degenerate_fraction'}
 for r in q:
  k=ident(r);t=top[k];m=mod[k];s=vm[(k,'POST','raw_tau_1e-6')]
  z={x:r[x] for x in ('denoiser','label','request_id','parent_id','candidate','region_type')}
  z.update({dst:100*number(r,src) for dst,src in quality_fields.items()})
  z.update(valid_CF_edges=number(t,'C_F_valid_edge_count'),has_valid_CF_edge=number(t,'C_F_has_valid_edge'),
   valid_CF_length=number(t,'C_F_valid_edge_total_length'),shared_CF_edges=number(t,'C_F_shared_edge_count'),
   reachable_free_area_pct=100*number(t,'reached_free_area_fraction'),strict_AVMR_pct=100*number(s,'VMR'),strict_PMR_pct=100*number(s,'perfect'),
   AVMR_1pctL_pct=100*number(vm[(k,'POST','L_tau_0.01')],'VMR'),distance_mean_model=number(dist[(k,'POST')],'distance_mean_model'),
   pre_GEO_strict_AVMR_pct=100*number(vm[(k,'PRE_GEO_C_RESTORED','raw_tau_1e-6')],'VMR'),
   pre_GEO_distance_mean_model=number(dist[(k,'PRE_GEO_C_RESTORED')],'distance_mean_model'),
   FF_illegal_pairs=number(r,'FF_illegal_pair_count'),deleted_free_faces=number(m,'deleted_free_faces'),raw_free_faces=number(m,'raw_free_faces'),
   deletion_pct=100*number(m,'deleted_free_fraction'),free_area=number(m,'POST_free_surface_area'),
   max_free_displacement=number(m,'denoise_free_corner_displacement_max'),mean_free_displacement=number(m,'denoise_free_corner_displacement_mean'),
   nonmanifold_edges=number(t,'nonmanifold_edges'),nonmanifold_CF_edges=number(t,'C_F_nonmanifold_edge_count'))
  enriched.append(z)
 C.csvwrite(DEST/'real_per_output.csv',enriched)
 numeric=[key for key,value in enriched[0].items() if isinstance(value,(int,float))]
 def summarize(rows,group):
  out=[]
  for key in sorted({tuple(r[k] for k in group) for r in rows}):
   rs=[r for r in rows if tuple(r[k] for k in group)==key]
   d=dict(zip(group,key));d.update(outputs=len(rs),**{k:avg(rs,k) for k in numeric})
   d.update(valid_connected_outputs=int(total(rs,'has_valid_CF_edge')),parents_with_valid_connection=len({r['parent_id'] for r in rs if r['valid_CF_edges']>0}),
    valid_CF_edge_sum=int(total(rs,'valid_CF_edges')),valid_CF_length_sum=total(rs,'valid_CF_length'),
    deleted_free_sum=int(total(rs,'deleted_free_faces')),deletion_weighted_pct=100*total(rs,'deleted_free_faces')/total(rs,'raw_free_faces'),
    FF_p95_pct=float(np.quantile([r['FF_pct'] for r in rs],.95)),FF_max_pct=max(r['FF_pct'] for r in rs))
   out.append(d)
  return out
 mainrows=summarize(enriched,['denoiser','label'])
 C.csvwrite(DEST/'real_main.csv',mainrows)
 C.csvwrite(DEST/'real_parent.csv',summarize(enriched,['denoiser','label','parent_id']))
 C.csvwrite(DEST/'real_region.csv',summarize(enriched,['denoiser','label','region_type']))
 C.csvwrite(DEST/'real_noise.csv',summarize(enriched,['denoiser','label','candidate']))
 overall=summarize(enriched,['denoiser']);C.csvwrite(DEST/'real_all_sources.csv',overall)
 pairs=[];directions=[]
 for new,base in ((ARMS[1],ARMS[0]),(ARMS[2],ARMS[0]),(ARMS[2],ARMS[1])):
  ref={(r['label'],r['request_id']):r for r in enriched if r['denoiser']==base}
  for r in enriched:
   if r['denoiser']!=new:continue
   b=ref[r['label'],r['request_id']]
   pairs.append(dict(comparison=new+'_minus_'+base,**{k:r[k] for k in ('label','request_id','parent_id','candidate','region_type')},**{k+'_delta':r[k]-b[k] for k in numeric}))
 for cmp,label in sorted({(r['comparison'],r['label']) for r in pairs}):
  rs=[r for r in pairs if (r['comparison'],r['label'])==(cmp,label)]
  for metric in numeric:
   ds=[r[metric+'_delta'] for r in rs]
   directions.append(dict(comparison=cmp,label=label,metric=metric,count=len(rs),mean_delta=float(np.mean(ds)),median_delta=float(np.median(ds)),minimum_delta=min(ds),maximum_delta=max(ds),increased=sum(x>1e-12 for x in ds),decreased=sum(x< -1e-12 for x in ds),equal=sum(abs(x)<=1e-12 for x in ds)))
 C.csvwrite(DEST/'paired_deltas.csv',pairs);C.csvwrite(DEST/'paired_directions.csv',directions)
 synth=read('synthetic_results/per_input.csv');sp=[]
 for new,base in ((ARMS[1],ARMS[0]),(ARMS[2],ARMS[0]),(ARMS[2],ARMS[1])):
  ref={r['synthetic_id']:r for r in synth if r['denoiser']==base}
  for r in synth:
   if r['denoiser']!=new:continue
   b=ref[r['synthetic_id']]
   sp.append(dict(comparison=new+'_minus_'+base,**{k:r[k] for k in ('synthetic_id','uid','region_type','clean','noise_index')},**{k+'_delta':number(r,k)-number(b,k) for k in ('free_mse','supported_interface_mse','free_movement_corner_l2_mean')}))
 C.csvwrite(DEST/'synthetic_paired.csv',sp)
 sections=['# Completed-score tables','All percentages use macro per-output means unless explicitly weighted; differences are percentage points. No added metric or scorer calls.',
  md(mainrows,['denoiser','label','outputs','valid_connected_outputs','parents_with_valid_connection','valid_CF_edge_sum','reachable_free_area_pct','strict_AVMR_pct','strict_PMR_pct','CF_pct','FF_pct','I_pct','small_angle_pct','deletion_weighted_pct']),
  '## Synthetic same-shape diagnostics (denoiser coordinate units)',md(read('synthetic_results/summary.csv'),['denoiser','input_kind','count','free_mse_macro','supported_interface_mse_macro','free_movement_corner_l2_mean_macro'])]
 C.textfile(DEST/'TABLES.md','\n\n'.join(sections)+'\n')
 C.write(DEST/'completion.json',dict(status='PASS',source_hashes={str(C.OUT/f):C.sha(C.OUT/f) for f in ('evaluation/quality_per_output.csv','evaluation/topology_per_output.csv','evaluation/modifications_per_output.csv','evaluation/per_output.csv','evaluation/boundary_distance_per_output.csv','synthetic_results/per_input.csv')},real_outputs=144,paired_outputs=144,synthetic_pairs=54,model_calls=0,scorer_calls=0))
 print(json.dumps(dict(status='PASS',destination=str(DEST))))
if __name__=='__main__':main()
