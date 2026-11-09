"""Read completed scores and real review shards; no model, detector or new labels."""
import csv, hashlib, json, math, statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'experiments/r2_bench_four_routes_v1/exposure'
DEST=OUT/'results/delivery_v1'
STAGES=('RAW','BASE_POST','CLEAN_FINAL_POST')
GROUPS=('ALL','OLD','FOCUS','DEV','HOLDOUT','HIGH_TRAIN','HIGH_HOLDOUT','NEW_WIDE_TRAIN')
METRICS=('CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C',
 'free_minimum_angle_lt5_fraction','free_near_degenerate_fraction','C_F_valid_edge_count',
 'C_F_valid_edge_total_length','C_F_has_valid_edge','nonmanifold_edges','C_F_nonmanifold_edge_count',
 'hinge_diag_legal_opposed_edge_C_reached_free_area_fraction','hinge_diag_fully_C_disconnected_free_area_fraction',
 'hinge_diag_isolated_1face_edge_components','actual_free_faces','deleted_free_faces',
 'free_only_reached_bulk_area_fraction','retained_bulk_area','legal_seam_bulk_edge_count',
 'free_total_surface_area','new_bridge_faces')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readcsv(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def number(x):
 try:
  y=float(x)
  return y if math.isfinite(y) else None
 except (TypeError,ValueError):return {'True':1.,'False':0.}.get(x)
def aggregate(rows,key,pass_i=False):
 d=defaultdict(list);excluded=Counter()
 for r in rows:
  if pass_i and r['I_status']!='PASS':excluded[r['I_status']]+=1;continue
  v=number(r.get(key))
  if v is None:excluded[str(r.get(key))]+=1;continue
  d[r['parent_id']].append(v)
 pv={p:statistics.fmean(v) for p,v in d.items()}
 return dict(mean=statistics.fmean(pv.values()) if pv else None,valid_parents=len(pv),valid_requests=sum(map(len,d.values())),parent_values=pv,excluded=dict(excluded))
def subset(rows,label,stage,group):return [r for r in rows if r['label']==label and r['stage']==stage and (group=='ALL' or r['report_group']==group)]
def joint(r):return r['overall_shape']=='REASONABLE' and r['C_use']=='PLAUSIBLE' and r['support_retention']=='PRESENT'
def csvwrite(path,rows):
 with path.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 sources=[OUT/'results/per_output.csv',OUT/'visual_review/summary_v1/actual_observations.csv',OUT/'results/analysis_v1/deletion_area_per_output.csv']
 hashes={str(p):sha(p) for p in sources};sc,vi,de=map(readcsv,sources)
 assert len(sc)==len(vi)==276 and len(de)==184
 DEST.mkdir(exist_ok=True)
 done=DEST/'complete.json'
 if done.exists():
  x=json.loads(done.read_text());assert x['sources']==hashes
  for p,h in x['files'].items():assert sha(p)==h
  print(json.dumps({k:v for k,v in x.items() if k not in ('files','sources')}));return
 numeric=[];visual=[];area=[];transitions=[]
 for stage in STAGES:
  for group in GROUPS:
   for label in ('M','EXPOSURE'):
    ss=subset(sc,label,stage,group);vs=subset(vi,label,stage,group)
    for metric in METRICS:
     a=aggregate(ss,metric,metric=='ConflictCov_C')
     numeric.append(dict(label=label,stage=stage,group=group,metric=metric,expected_requests=len(ss),expected_parents=len({r['parent_id'] for r in ss}),**a))
    counts={axis:dict(Counter(r[axis] for r in vs)) for axis in ('overall_shape','C_use','support_retention','visible_fragment_or_gap','layering_or_fold')}
    visual.append(dict(label=label,stage=stage,group=group,requests=len(vs),joint_count=sum(joint(r) for r in vs),axis_counts=counts))
    if stage!='RAW':
     ds=subset(de,label,stage,group)
     for r in ds:
      r['post_raw_area_ratio']=float(r['actual_final_free_area'])/float(r['raw_free_area'])
      r['retained_slot_fraction']=float(r['retained_original_free_faces'])/float(r['raw_free_faces'])
      r['pre_geo_raw_area_ratio']=float(r['pre_geo_free_area'])/float(r['raw_free_area'])
     for metric in ('post_raw_area_ratio','pre_geo_raw_area_ratio','deleted_raw_area_fraction','retained_slot_fraction','deleted_original_free_faces','actual_final_free_area','deleted_pre_geo_reference_area'):
      area.append(dict(label=label,stage=stage,group=group,metric=metric,**aggregate(ds,metric)))
   ba={r['request_id']:r for r in subset(vi,'M',stage,group)};sa={r['request_id']:r for r in subset(vi,'EXPOSURE',stage,group)}
   for rid in sorted(ba):
    transitions.append(dict(stage=stage,group=group,request_id=rid,parent_id=ba[rid]['parent_id'],M_joint=joint(ba[rid]),EXPOSURE_joint=joint(sa[rid]),transition=('GAIN' if joint(sa[rid]) else 'LOSS') if joint(ba[rid])!=joint(sa[rid]) else 'SAME'))
 details=dict(numeric=numeric,visual=visual,area=area,transitions=transitions,
  I_status={l:{s:dict(Counter(r['I_status'] for r in subset(sc,l,s,'ALL'))) for s in STAGES} for l in ('M','EXPOSURE')},
  seam_status={s:dict(Counter(r['seam_to_bulk_status'] for r in subset(sc,'EXPOSURE',s,'ALL'))) for s in STAGES},
  identity={'C_preserved_exact':dict(Counter(r['C_preserved_exact'] for r in sc))},
  caution='Descriptive three-axis intersection is not a formal success metric; area is generated-slot geometry, not GT completeness; M/EXPOSURE differs by 3000 added Native updates, including OLD and FOCUS exposure; this is not a data-only causal comparison.')
 (DEST/'summary.json').write_text(json.dumps(details,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 for name,rr in [('visual_joint_transitions',transitions),('numeric',numeric),('area',area),('visual_counts',visual)]:
  csvwrite(DEST/(name+'.csv'),[{k:json.dumps(v,sort_keys=True) if isinstance(v,dict) else v for k,v in r.items()} for r in rr])
 result=dict(status='COMPLETE',model_calls=0,scoring_calls=0,new_ratings=0,actual_observations=276,score_rows=276,sources=hashes,code_sha256=sha(__file__),files={str(p):sha(p) for p in DEST.iterdir() if p.is_file()})
 done.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items() if k not in ('sources','files')}))

if __name__=='__main__':main()
