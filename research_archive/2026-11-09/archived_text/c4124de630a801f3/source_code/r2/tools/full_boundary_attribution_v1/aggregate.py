"""Descriptive aggregation; preserves inherited metrics and exact source-boundary edge mappings."""
from . import common as C
import numpy as np,json

def main():
 if (C.OUT/'aggregation_completion.json').exists():
  done=C.read(C.OUT/'aggregation_completion.json')
  for p,h in done['table_hashes'].items():assert C.sha(p)==h,p
  print('Cached aggregation hashes verified');return
 r=C.read(C.OUT/'registration.json');rows=[];edges=[];partitions=[];bounds=[]
 for cell in r['cells']:
  folder=C.OUT/'results'/cell['cell_id'];s=C.read(folder/'summary.json');v=C.arrays(folder/'vectors.npz');a=C.arrays(cell['geometry_path']);old=C.read(cell['inherited_path']);num=old['numeric']
  ident={k:cell[k] for k in ['cell_id','request_id','parent_id','candidate','label','stage','K','N']}
  row=dict(ident,source_boundary_length=s['source_boundary_length'],edge_queries=s['edge_midpoint_queries'],vertex_queries=s['boundary_vertex_queries'],F_faces=s['free_faces'],F_components=s['observed_F_components'],
   inherited_adoption=old['review']['C_adoption'],inherited_legal_edges=int(num['C_F_valid_edge_count']),inherited_CF=float(num['CF_illegal_affected_free_fraction']),inherited_FF=float(num['FF_illegal_affected_free_fraction']),inherited_I=float(num['ConflictCov_C']),
   inherited_A_VMR_001L=float(num['A_VMR__L_tau_0.001']),inherited_PMR_001L=float(num['PMR__L_tau_0.001']),required_interface_status='NA_NO_REQUIRED_INTERFACE')
  for domain,stats in s['boundary_distances'].items():
   for k,value in stats.items():row[domain+'__'+k]=value
  samples=C.csvread(folder/'samples.csv');w=v['length_weights'];mask=w>0;w=w[mask];den=w.sum()
  for metric in ['surface_saved_CF_any_illegal','surface_saved_FF_any_illegal','surface_parallel_15deg']:
   row[metric+'__length_fraction']=float(sum(ww for ww,item in zip(v['length_weights'],samples) if item[metric]=='True')/den)
  row['multiple_normal_groups__length_fraction']=float(w@(v['corridor_layer_count'][mask]>=2)/den)
  row['surface_signed_offset__weighted_mean_over_L']=float(sum(ww*float(x['surface_signed_source_normal_offset_over_L']) for ww,x in zip(v['length_weights'],samples))/den)
  legal=[x for x in old['valid_edges'] if x['is_C_F']=='True' and x['valid_edge']=='True'];assert len(legal)==row['inherited_legal_edges']
  def ek(a,b):return tuple(sorted((tuple(a),tuple(b))))
  lm={ek(a['mesh_vertices'][int(x['vertex_u'])],a['mesh_vertices'][int(x['vertex_v'])]):x for x in legal}
  mapped=set();legal_length=0
  for e in C.csvread(folder/'boundary_edges.csv'):
   key=ek([float(e['start_'+t]) for t in 'xyz'],[float(e['end_'+t]) for t in 'xyz']);match=lm.get(key);yes=match is not None
   if yes:mapped.add(key);legal_length+=float(e['length'])
   edges.append(dict(ident,**e,saved_legal_CF_shared_edge=yes,saved_valid_edge_record=match,definition='Exact source boundary edge endpoints matched to inherited actual legal CF edge; no near welding.'))
  assert len(mapped)==len(lm),(cell['cell_id'],'legal edge missing source boundary map')
  row['exact_legal_source_boundary_edges']=len(mapped);row['exact_legal_source_boundary_length']=legal_length
  row['exact_legal_source_boundary_length_fraction']=legal_length/den
  d={name:v[name+'_distance'][mask]/C.L for name in ['surface','edge','open_edge','vertex']}
  for tau in r['geometry_config']['distance_thresholds_over_L']:
   groups=dict(surface_far=d['surface']>=tau,surface_near_all_edge_far=(d['surface']<tau)&(d['edge']>=tau),
    all_edge_near_vertex_far=(d['edge']<tau)&(d['vertex']>=tau),vertex_near=d['vertex']<tau)
   assert np.all(sum(groups.values())==1)
   partitions.append(dict(ident,tau_over_L=tau,**{k:float(w@m/den) for k,m in groups.items()},
    interpretation='Descriptive midpoint partitions, not failure labels. Even a perfectly shared long edge normally has interior points far from vertices.'))
   for name,dd in d.items():
    radius=w/(2*C.L)
    bounds.append(dict(ident,domain=name,tau_over_L=tau,midpoint_length_fraction=float(w@(dd<tau)/den),
     guaranteed_length_fraction_lower=float(w@(dd+radius<tau)/den),possible_length_fraction_upper=float(w@(dd-radius<tau)/den),
     mean_distance_quadrature_error_bound_over_L=float(np.sum(w*w)/(4*C.L*den)),
     definition='Distance to a fixed finite set is 1-Lipschitz; per-bin midpoint +/- half-bin length bounds threshold crossing. Mean integral error bound=sum(h^2)/(4*sum(h)*L).'))
  rows.append(row)
 C.csvwrite(C.OUT/'tables/per_output.csv',rows);C.csvwrite(C.OUT/'tables/source_boundary_legal_edges.csv',edges)
 C.csvwrite(C.OUT/'tables/distance_partitions.csv',partitions);C.csvwrite(C.OUT/'tables/sampling_bounds.csv',bounds)
 ix={(x['request_id'],x['label'],x['stage']):x for x in rows};arm=[];stage=[]
 metrics=[k for k,v in rows[0].items() if isinstance(v,(float,int)) and k not in ['candidate','K','N']]
 for q in r['requests']:
  for st in ['RAW','POST']:
   a=ix[q['request_id'],'SMALL_C200',st];b=ix[q['request_id'],'CREF_C200',st]
   arm.append(dict(request_id=q['request_id'],parent_id=q['parent_id'],candidate=q['candidate'],stage=st,SMALL_adoption=a['inherited_adoption'],CREF_adoption=b['inherited_adoption'],**{k+'__CREF_minus_SMALL':(b[k]-a[k] if b[k] is not None and a[k] is not None else None) for k in metrics}))
  for label in ['SMALL_C200','CREF_C200']:
   a=ix[q['request_id'],label,'RAW'];b=ix[q['request_id'],label,'POST']
   stage.append(dict(request_id=q['request_id'],parent_id=q['parent_id'],candidate=q['candidate'],label=label,**{k+'__POST_minus_RAW':(b[k]-a[k] if b[k] is not None and a[k] is not None else None) for k in metrics}))
 C.csvwrite(C.OUT/'tables/paired_models.csv',arm);C.csvwrite(C.OUT/'tables/paired_stages.csv',stage)
 summary=[]
 for parent in list(dict.fromkeys(x['parent_id'] for x in rows)):
  for label in ['SMALL_C200','CREF_C200']:
   for st in ['RAW','POST']:
    ss=[x for x in rows if x['parent_id']==parent and x['label']==label and x['stage']==st]
    summary.append(dict(valid_counts={k:sum(x[k] is not None for x in ss) for k in metrics},parent_id=parent,label=label,stage=st,outputs=len(ss),adoption_yes=sum(x['inherited_adoption']=='YES' for x in ss),legal_positive_outputs=sum(x['inherited_legal_edges']>0 for x in ss),**{k+'__macro_mean':(float(np.mean([x[k] for x in ss if x[k] is not None])) if any(x[k] is not None for x in ss) else None) for k in metrics}))
 C.csvwrite(C.OUT/'tables/per_parent.csv',summary)
 C.write(C.OUT/'aggregation_completion.json',dict(status='COMPLETE',outputs=44,source_boundary_rows=len(edges),partition_rows=len(partitions),bounds_rows=len(bounds),new_model_calls=0,new_quality_scans=0,selection_limit='Observed-case diagnostic subset; not population success rates',
   source_hashes={str(C.OUT/'registration.json'):C.sha(C.OUT/'registration.json'),str(C.OUT/'analysis_completion.json'):C.sha(C.OUT/'analysis_completion.json')},
   table_hashes={str(p):C.sha(p) for p in sorted((C.OUT/'tables').glob('*.csv'))}))
 C.event('DESCRIPTIVE_AGGREGATION_COMPLETE',outputs=44,legal_source_boundary_mapping='ALL_MATCHED',sampling_bounds='1_LIPSCHITZ_MIDPOINT_ERROR')
 print('Aggregated44cells; all saved legal CF edges matched to original source boundary.')
if __name__=='__main__':main()