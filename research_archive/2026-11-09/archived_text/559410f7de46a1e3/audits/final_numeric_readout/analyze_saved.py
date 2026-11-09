"""Read saved endpoint tables and completed ordinary-forward diagnostics only."""
from pathlib import Path
import csv,datetime,hashlib,json,math,statistics,sys
ROOT=Path('/mnt/e/MeshFlow')
OUT=ROOT/'experiments/r2_full_contour_readout_v1'
OLD=ROOT/'experiments/r2_full_seat_learning_curve_v1'
DEST=OUT/'audits/final_numeric_readout'
L=math.sqrt(3)*2/.3762
SOURCES={}
METRICS=['CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','ConflictCov_C_upper','ConflictCov_C_uncertain_fraction','CF_illegal_pair_count','FF_illegal_pair_count','C_F_valid_edge_count','C_F_valid_edge_total_length','C_F_has_valid_edge','C_F_valid_opposite_winding_edge_count','opposite_winding_reached_free_area_fraction','actual_free_faces','deleted_free_faces','free_face_denominator','free_total_surface_area','free_minimum_angle_lt5_fraction','free_near_degenerate_fraction','EXACT_COORD_face_components']
BOUNDARY_STATS=['length_weighted_mean_over_L','length_weighted_median_over_L','length_weighted_p95_over_L','max_over_L','length_fraction_lt_0.001L','length_fraction_lt_0.005L','length_fraction_lt_0.01L','length_fraction_lt_0.02L']
BOUNDARY_METRICS=['boundary_'+domain+'__'+s for domain in ('surface','edge','open_edge') for s in BOUNDARY_STATS]
SELECTED=METRICS+['C_F_valid_edge_total_length_over_L']+BOUNDARY_METRICS

def capture(p):
 p=Path(p);b=p.read_bytes();SOURCES[str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)};return b

def js(p):return json.loads(capture(p).decode('utf-8-sig'))
def cs(p):return list(csv.DictReader(capture(p).decode('utf-8-sig').splitlines()))
def num(v):
 if v is None or v=='':return None
 x=float(v);return x if math.isfinite(x) else None

def csvwrite(name,rows):
 rows=list(rows);keys=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
 p=DEST/name
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def jsonwrite(name,data):
 (DEST/name).write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')

def load_arm(label):
 table=cs(OUT/'evaluation/tables'/label/'per_output.csv')
 receipt=js(OUT/'evaluation/boundary_completion'/f'{label}.json')
 assert receipt['status']=='COMPLETE' and receipt['cells']==48
 assert len(table)==48
 result={}
 for r in table:
  key=(r['request_id'],r['stage']);assert key not in result
  assert r['label']==label
  b=js(OUT/'evaluation/boundary'/label/r['request_id']/r['stage']/'completion.json')
  assert b['status']=='COMPLETE'
  summary_path=b['summary_path'];summary=js(summary_path)
  assert SOURCES[summary_path]['sha256']==b['files'][summary_path]
  assert abs(float(summary['L'])-L)<1e-14 and int(summary['source_C_faces'])==int(r['K'])
  d={**r}
  for m in METRICS:d[m]=num(r.get(m))
  d['C_F_valid_edge_total_length_over_L']=d['C_F_valid_edge_total_length']/L
  for domain in ('surface','edge','open_edge'):
   for s in BOUNDARY_STATS:d['boundary_'+domain+'__'+s]=num(summary['boundary_distances'][domain].get(s))
  d['source_boundary_length']=summary['source_boundary_length'];d['edge_midpoint_queries']=summary['edge_midpoint_queries']
  d['boundary_summary_path']=summary_path;d['boundary_summary_sha256']=SOURCES[summary_path]['sha256']
  d['all_score_status_PASS']=(r['I_status']=='PASS' and r['boundary_distance_status']=='PASS')
  result[key]=d
 return result

def mean(values):return math.fsum(values)/len(values) if values else None

def aggregate(pairs,group,identity):
 rows=[]
 for m in SELECTED:
  ok=[p for p in pairs if p['FULL__'+m] is not None and p['CONTOUR__'+m] is not None]
  a=[p['FULL__'+m] for p in ok];b=[p['CONTOUR__'+m] for p in ok];d=[p['delta__'+m] for p in ok]
  rows.append(dict(group=group,identity=identity,stage=pairs[0]['stage'],metric=m,requests_total=len(pairs),paired_available=len(ok),NA_pairs=len(pairs)-len(ok),FULL_mean=mean(a),CONTOUR_mean=mean(b),delta_mean=mean(d),FULL_median=statistics.median(a) if a else None,CONTOUR_median=statistics.median(b) if b else None,FULL_min=min(a) if a else None,FULL_max=max(a) if a else None,CONTOUR_min=min(b) if b else None,CONTOUR_max=max(b) if b else None,delta_min=min(d) if d else None,delta_max=max(d) if d else None,n_lower=sum(x<0 for x in d),n_equal=sum(x==0 for x in d),n_higher=sum(x>0 for x in d),denominator='equal-weight paired requests; each parent has four fixed noises'))
 return rows

def training_tables():
 inputs={};defs={}
 for label,root in [('FULL',OLD),('CONTOUR',OUT)]:
  pointer=js(root/'training_analysis/latest.json');assert pointer['status']=='COMPLETE' and pointer['last_complete_diagnostic_step']==6000
  completion=js(pointer['path']);assert SOURCES[pointer['path']]['sha256']==pointer['sha256']
  d=Path(pointer['path']).parent
  inputs[label]={name:cs(d/(name+'.csv')) for name in ['fixed_window_diagnostics','fixed_window_time_bins','learning_curve_100_steps','per_update','parent_exposure']}
  assert len(inputs[label]['per_update'])==6000
  assert all(r['status']=='COMPLETE' for name in ['fixed_window_diagnostics','fixed_window_time_bins','learning_curve_100_steps','per_update'] for r in inputs[label][name])
  defs[label]=dict(completion_path=pointer['path'],completion_sha256=pointer['sha256'])
 values=['samples','interface_scalar_count','other_scalar_count','all_free_scalar_count','interface_velocity_mse','other_velocity_mse','all_free_velocity_mse','interface_endpoint_scaled_mse','other_endpoint_scaled_mse','all_free_endpoint_scaled_mse','interface_over_other_velocity_mse_ratio','logged_FM_mean','logged_FM_scalar_weighted_mean']
 for name in ['fixed_window_diagnostics','fixed_window_time_bins','learning_curve_100_steps']:
  def key(r):return(int(r['window_start']),int(r['window_end']),r['time_bin'])
  a={key(r):r for r in inputs['FULL'][name]};b={key(r):r for r in inputs['CONTOUR'][name]};assert set(a)==set(b)
  out=[]
  for k in sorted(a,key=lambda x:(x[0],x[1],x[2])):
   x,y=a[k],b[k];r=dict(window_start=k[0],window_end=k[1],time_bin=k[2],time_bin_lower=y.get('time_bin_lower',''),time_bin_upper=y.get('time_bin_upper',''))
   for field in ['samples','interface_scalar_count','other_scalar_count','all_free_scalar_count']:assert x[field]==y[field],(name,k,field)
   for field in values:
    v,w=num(x.get(field)),num(y.get(field));r.update({f'FULL__{field}':v,f'CONTOUR__{field}':w,f'delta__{field}':w-v if v is not None and w is not None else None})
   r['denominator']='pooled recorded scalar counts; not average of sample MSEs'
   r['interpretation']='Changing training states; same paired replay, not fixed probes and not autonomous rollout.'
   out.append(r)
  csvwrite('training_'+name+'_paired.csv',out)
 exposure=[]
 for label in inputs:
  data=inputs[label]['parent_exposure'];counts=[int(r['completed_update_exposures']) for r in data]
  exposure.append(dict(label=label,parents=len(data),completed_exposures=sum(counts),min_parent_exposures=min(counts),max_parent_exposures=max(counts),interface_scalar_count=sum(int(r['completed_interface_scalar_exposure']) for r in data),free_scalar_count=sum(int(r['completed_free_scalar_exposure']) for r in data)))
 csvwrite('training_exposure_totals.csv',exposure)
 return defs

def main():
 required=[OUT/'evaluation/tables/CONTOUR_S6000/per_output.csv',OUT/'evaluation/boundary_completion/CONTOUR_S6000.json']
 missing=[str(p) for p in required if not p.exists()]
 pointer=OUT/'training_analysis/latest.json'
 if not pointer.exists() or json.load(pointer.open())['status']!='COMPLETE':missing.append('final training_analysis COMPLETE snapshot')
 if missing:print(json.dumps({'status':'WAITING_EXISTING_EVIDENCE','missing':missing}));return
 if (DEST/'completion.json').exists():raise RuntimeError('Completed independent analysis already exists; preserve it')
 baseline=load_arm('FULL_S6000');candidate=load_arm('CONTOUR_S6000');assert set(baseline)==set(candidate)
 pairs=[]
 for key in sorted(baseline):
  a,b=baseline[key],candidate[key]
  for field in ['parent_id','candidate','N','K','split','condition_id']:assert a[field]==b[field],(key,field)
  assert a['source_boundary_length']==b['source_boundary_length'] and a['edge_midpoint_queries']==b['edge_midpoint_queries']
  r={f:a[f] for f in ['request_id','parent_id','candidate','stage','N','K','split','condition_id']}
  for m in SELECTED:
   x,y=a[m],b[m];r.update({f'FULL__{m}':x,f'CONTOUR__{m}':y,f'delta__{m}':y-x if x is not None and y is not None else None})
  for label,item in [('FULL',a),('CONTOUR',b)]:
   for f in ['score_path','score_sha256','boundary_summary_path','boundary_summary_sha256','all_score_status_PASS']:r[label+'__'+f]=item[f]
  pairs.append(r)
 csvwrite('endpoint_per_output_pairs.csv',pairs)
 parent=[];strata=[]
 for stage in ['RAW','POST']:
  scope=[p for p in pairs if p['stage']==stage]
  for ident in sorted({p['parent_id'] for p in scope}):parent+=aggregate([p for p in scope if p['parent_id']==ident],'parent',ident)
  for ident in ['ALL','TRAIN','DEV']:strata+=aggregate([p for p in scope if ident=='ALL' or p['split']==ident],'stratum',ident)
 csvwrite('endpoint_per_parent_paired_metrics.csv',parent);csvwrite('endpoint_strata_paired_metrics.csv',strata)
 transitions=[]
 for p in pairs:
  if p['stage']!='POST':continue
  a=p['FULL__C_F_valid_edge_count'];b=p['CONTOUR__C_F_valid_edge_count']
  transitions.append({**{f:p[f] for f in ['request_id','parent_id','candidate','split']},'FULL_legal_edges':a,'CONTOUR_legal_edges':b,'edge_presence_transition':('PRESENT' if a>0 else 'ABSENT')+'_TO_'+('PRESENT' if b>0 else 'ABSENT'),**{k:v for k,v in p.items() if any(k.endswith(m) for m in ['CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','C_F_valid_edge_total_length_over_L','actual_free_faces','deleted_free_faces'])}})
 csvwrite('POST_interface_transitions_and_costs.csv',transitions)
 tails=[]
 for stage in ['RAW','POST']:
  for m in ['CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','boundary_edge__length_weighted_mean_over_L','boundary_open_edge__length_weighted_p95_over_L','deleted_free_faces']:
   ordered=sorted([p for p in pairs if p['stage']==stage and p['delta__'+m] is not None],key=lambda p:p['delta__'+m],reverse=True)
   for rank,p in enumerate(ordered,1):tails.append(dict(stage=stage,metric=m,delta_descending_rank=rank,request_id=p['request_id'],parent_id=p['parent_id'],candidate=p['candidate'],split=p['split'],FULL=p['FULL__'+m],CONTOUR=p['CONTOUR__'+m],delta=p['delta__'+m],note='All paired requests retained; rank is descriptive, no threshold or selected example subset.'))
 csvwrite('all_requests_tail_ranks.csv',tails)
 train=training_tables()
 counts={}
 for group in ['ALL','TRAIN','DEV']:
  q=[p for p in pairs if p['stage']=='POST' and (group=='ALL' or p['split']==group)]
  counts[group]={label:{'requests':len(q),'outputs_with_legal_edge':sum(p[label+'__C_F_valid_edge_count']>0 for p in q),'legal_edge_count_sum':sum(p[label+'__C_F_valid_edge_count'] for p in q),'legal_edge_length_sum_over_L':sum(p[label+'__C_F_valid_edge_total_length_over_L'] for p in q),'actual_free_faces_sum':sum(p[label+'__actual_free_faces'] for p in q),'deleted_free_faces_sum':sum(p[label+'__deleted_free_faces'] for p in q)} for label in ['FULL','CONTOUR']}
 definitions=dict(CF='CF_illegal_affected_free_fraction: affected free-face union for illegal C/F pairs / actual free-face denominator, not proper-only.',FF='FF_illegal_affected_free_fraction: affected free-face union for illegal F/F pairs / actual free-face denominator.',I='ConflictCov_C: frozen C query-weighted conflict surface coverage; not global intersection-pair count or free-face fraction.',interface='C_F_valid_edge_count/total_length: saved exact indexed C/F edge filter, rejects duplicate/illegal/unknown/nonmanifold relationships; nonzero does not certify entire boundary closure.',boundary='Existing deterministic length-weighted midpoint queries along source C boundary to finite F surface/all edges/open edges, normalized by fixed L. All C boundary is observational domain, not mandatory closure target.',face_counts='actual_free_faces and deleted_free_faces are saved stage-specific counts; POST quality fractions use surviving free faces.',mean='Equal paired-request macro mean; four fixed noises per parent. Tail table contains every paired request.',training='Pooled recorded SSE/scalar count in fixed windows1-1000/1001-3000/3001-6000 and bins0-.2/.2-.4/.4-.6/.6-.8/.8-1; interface and other counts preserved, no new forward. Endpoint SSE=(1-t)^2 velocity SSE.',limits='No actual image labels assessed. No inference about adoption/completeness/style from numerical metrics alone. Familiar DEV, not new blind generalization; no best checkpoint selection.')
 jsonwrite('metric_definitions.json',dict(L=L,definitions=definitions))
 jsonwrite('source_hashes.json',SOURCES)
 result=dict(status='COMPLETE_NUMERICAL_ONLY',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),paired_outputs=48,requests=24,parents=6,comparison='CONTOUR_S6000 minus FULL_S6000 only',POST_counts=counts,training_snapshots=train,new_model_calls=0,new_scores=0,new_visual_reviews=0,source_files=len(SOURCES),source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 jsonwrite('completion.json',result)
 print(json.dumps(result))

if __name__=='__main__':main()