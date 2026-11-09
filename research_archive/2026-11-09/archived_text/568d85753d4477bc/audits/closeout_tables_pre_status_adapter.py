"""Aggregate saved scores and literal image reviews; no scans or model calls."""
from pathlib import Path
import csv,json,hashlib,statistics,collections

ROOT=Path(__file__).resolve().parents[3]/'experiments/r2_seam_slot_c200_v1'
ARMS=('FULL4_C200','SEAM_SLOT_C200')
METRICS=('C_F_has_valid_edge','C_F_valid_edge_count','C_F_valid_edge_total_length',
 'reached_free_area_fraction','opposite_winding_reached_free_area_fraction',
 'CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C',
 'free_minimum_angle_lt5_fraction','free_q_gt10_fraction','nonmanifold_edges',
 'C_F_nonmanifold_edge_count','A_VMR__raw_tau_1e-6','PMR__raw_tau_1e-6','deleted_free_faces')
def readcsv(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def savecsv(p,rows):
 with p.open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def write(p,obj):
 with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 e=ROOT/'evaluation';g=e/'gallery';rows=readcsv(e/'per_output.csv')
 reviews=readcsv(g/'review_73_3f_AI_PROXY.csv')+readcsv(g/'review_85_8c_AI_PROXY.csv')
 keys=lambda r:(r['label'],r['request_id'],r['stage'])
 assert len(rows)==len(reviews)==64 and {keys(r) for r in rows}=={keys(r) for r in reviews}
 assert len({keys(r) for r in reviews})==64
 for r in reviews:
  assert r['review_status']=='ACTUALLY_VIEWED_NONBLIND' and r['reviewer'].startswith('AI_PROXY')
  assert sha(g/(r['request_id']+'.png'))==r['reviewed_image_sha256']
 review={keys(r):r for r in reviews};scores={keys(r):r for r in rows}
 savecsv(e/'visual_review_all64.csv',sorted(reviews,key=keys))
 ids=sorted({r['request_id'] for r in rows});comparisons=[];paired=[];parents=[];seam=[]
 for stage in ('RAW','POST'):
  for metric in METRICS:
   a=[[float(r[metric]) for r in rows if r['label']==arm and r['stage']==stage] for arm in ARMS]
   comparisons.append(dict(stage=stage,metric=metric,n=16,FULL_mean=statistics.mean(a[0]),
    SEAM_mean=statistics.mean(a[1]),delta=statistics.mean(y-x for x,y in zip(*a)),
    FULL_sum=sum(a[0]),SEAM_sum=sum(a[1])))
 savecsv(e/'endpoint_comparison.csv',comparisons)
 for rid in ids:
  q=dict(request_id=rid)
  for arm,short in zip(ARMS,('FULL','SEAM')):
   raw=review[arm,rid,'RAW'];post=review[arm,rid,'POST'];s=scores[arm,rid,'POST']
   q.update({short+'_RAW_adoption':raw['C_adoption'],short+'_POST_adoption':post['C_adoption'],
    short+'_body':post['whole_body_complete'],short+'_POST_legal_CF_edges':int(s['C_F_valid_edge_count']),
    short+'_screened_adoption_plus_any_legal_CF':int(raw['C_adoption']=='YES' and float(s['C_F_valid_edge_count'])>0),
    short+'_POST_CF':float(s['CF_illegal_affected_free_fraction']),short+'_POST_FF':float(s['FF_illegal_affected_free_fraction']),
    short+'_POST_I':float(s['ConflictCov_C']),short+'_POST_oriented_reach':float(s['opposite_winding_reached_free_area_fraction'])})
  q['SEAM_free_only_bulk_reach']=float(scores[ARMS[1],rid,'POST']['free_only_reached_bulk_area_fraction'])
  paired.append(q)
 savecsv(e/'paired_screening.csv',paired)
 fields=('retained_seam_faces','retained_seam_area','legal_CF_seed_seam_faces','retained_bulk_faces','retained_bulk_area',
  'free_only_reached_bulk_faces','free_only_reached_bulk_area','free_only_reached_bulk_area_fraction',
  'legal_seam_bulk_edge_count','legal_seam_bulk_edge_length','deleted_seam_faces')
 for rid in ids:
  s=scores[ARMS[1],rid,'POST'];seam.append(dict(request_id=rid,**{k:float(s[k]) for k in fields}))
 savecsv(e/'seam_bulk_per_request.csv',seam)
 for prefix in sorted({r.split('_')[0] for r in ids}):
  for arm,short in zip(ARMS,('FULL','SEAM')):
   subset=[r for r in rows if r['label']==arm and r['stage']=='POST' and r['request_id'].startswith(prefix)]
   ps=[p for p in paired if p['request_id'].startswith(prefix)]
   parents.append(dict(parent_prefix=prefix,label=arm,n=4,RAW_adoption_YES=sum(p[short+'_RAW_adoption']=='YES' for p in ps),
    any_legal_CF=sum(float(r['C_F_valid_edge_count'])>0 for r in subset),
    screened_adoption_plus_any_legal_CF=sum(p[short+'_screened_adoption_plus_any_legal_CF'] for p in ps),
    legal_CF_edges=sum(int(r['C_F_valid_edge_count']) for r in subset),
    legal_CF_length=sum(float(r['C_F_valid_edge_total_length']) for r in subset),
    **{k:statistics.mean(float(r[k]) for r in subset) for k in ('CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','opposite_winding_reached_free_area_fraction')},
    candidate_bulk_ge30=sum(float(r['free_only_reached_bulk_area_fraction'])>=.3 for r in subset) if arm==ARMS[1] else None,
    candidate_bulk_mean=statistics.mean(float(r['free_only_reached_bulk_area_fraction']) for r in subset) if arm==ARMS[1] else None))
 savecsv(e/'outcome_by_parent.csv',parents)
 result=dict(status='COMPLETE',actual_reviewed_pages=16,actual_reviewed_cells=64,
  visual_counts={arm:{stage:{field:dict(collections.Counter(r[field] for r in reviews if r['label']==arm and r['stage']==stage))
    for field in ('C_adoption','whole_body_complete','fragments_or_folded_layers','visible_join')} for stage in ('RAW','POST')} for arm in ARMS},
  screened_counts={short:sum(r[short+'_screened_adoption_plus_any_legal_CF'] for r in paired) for short in ('FULL','SEAM')},
  seam_bulk=dict(nonzero=sum(r['free_only_reached_bulk_area_fraction']>0 for r in seam),ge30=sum(r['free_only_reached_bulk_area_fraction']>=.3 for r in seam),
   zero_requests=[r['request_id'] for r in seam if r['free_only_reached_bulk_area_fraction']==0],
   median=statistics.median(r['free_only_reached_bulk_area_fraction'] for r in seam)),
  warnings=['screened_adoption_plus_any_legal_CF is a minimal screen, not complete attachment, quality, or new semantic capability.',
   'Visible_join is a visual contact judgment, not an indexed-topology certificate.',
   'Candidate-only seam-bulk diagnostics are NA for FULL, not zero. 30% is descriptive only.',
   'Four seen TRAIN C; repeated noise is not independent parent replication.'],
  no_model_calls=True,no_new_scans=True)
 write(e/'closeout_summary.json',result)
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
