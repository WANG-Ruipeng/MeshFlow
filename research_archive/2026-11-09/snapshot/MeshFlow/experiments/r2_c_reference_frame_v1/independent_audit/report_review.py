"""Independent read-only report / saved-mask / historical-label check."""
import csv,json,hashlib,collections
from pathlib import Path
import numpy as np
O=Path('/mnt/e/MeshFlow/experiments/r2_c_reference_frame_v1');E=O/'evaluation';A=O/'independent_audit'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def arr(p):
 with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
errors=[];ncheck=0
def check(v,msg):
 global ncheck
 ncheck+=1
 if not v:errors.append(msg)
def close(a,b):return abs(float(a)-float(b))<1e-10
R=read(E/'registration.json');qs={q['condition_id']:arr(q['path']) for q in R['queries']}
cost=rows(O/'results/I_cost_per_output.csv');check(len(cost)==64,'64 I records')
for x in cost:
 s=read(x['source_score']);flat={**s['metrics'],**s['topology']};v=arr(Path(x['source_score']).parent/'vectors.npz');w=qs[x['condition_id']]['C_weights'];den=w.sum()
 c,d,u,un=[v[k] for k in ('conflict_cross','conflict_dup','conflict_union','conflict_unknown')]
 check(np.array_equal(c|d,u),'union saved masks')
 masks={'cross':c,'near_parallel':d,'union':u,'both':c&d,'cross_only':c&~d,'near_only':d&~c,'unknown':un}
 for k,m in masks.items():
  check(close(x[k+'_pct'],np.dot(w,m)/den*100),'I saved mask weighted '+k)
  check(int(x[k+'_samples'])==int(m.sum()),'I sample count '+k)
 for k,name in [('union','ConflictCov_C'),('cross','ConflictCov_C_cross'),('near_parallel','ConflictCov_C_dup'),('both','ConflictCov_C_cross_and_dup')]:check(close(x[k+'_pct'],100*s['metrics'][name]),'I CSV source score '+name)
 check(int(x['exact_duplicate_faces'])==int(flat['exact_duplicate_faces'])==0,'no exact duplicate faces')
 check(close(x['query_weight_denominator'],den) and int(x['query_samples'])==len(w)==2048,'full query denominator')
 pairs=read(Path(x['source_score']).parent/'pairs.json')
 for part in ('CF','FF'):
  counts=collections.Counter(z['classification'] for z in pairs[part]['contacts'])
  for kind in ('proper_transverse_penetration','coplanar_positive_area_overlap'):check(int(x[part+'__'+kind])==counts[kind],'saved pair classification '+part+kind)
pn=rows(O/'results/per_panel.csv');N=rows(E/'per_output.csv');pmap={(x['panel'],x['label'],x['stage']):x for x in pn}
dec=rows(O/'results/paired_decisions.csv');main=[x for x in dec if x['comparison']=='CREF_C200-minus-SMALL_C200' and x['stage']=='RAW']
gains=[x for x in main if x['C_adoption_pair']=='GAIN'];losses=[x for x in main if x['C_adoption_pair']=='LOSS'];yu=[x for x in main if (x['C_adoption_before'],x['C_adoption_after'])==('YES','UNCERTAIN')]
check(len(gains)==7 and len({x['condition_id'] for x in gains})==5,'seven clear gains /five C')
check(collections.Counter(x['panel'] for x in gains)=={'TRAIN_FAMILIAR':4,'DEV_HELDOUT':3},'gains panel split')
check(len(losses)==0 and len(yu)==2,'zero definite loss/two yes-to-uncertain')
for x in N:
 if x['stage']=='RAW':check(all(float(x[k])==0 for k in ('A_VMR__raw_tau_1e-6','PMR__raw_tau_1e-6','C_F_has_valid_edge')),'RAW strict metrics zero')
group={}
for prefix in ('3f','73b'):
 z=[x for x in cost if x['parent_id'].startswith(prefix) and x['stage']=='POST']
 old=sorted([x for x in z if x['label']=='SMALL_C200'],key=lambda x:int(x['candidate']));new=sorted([x for x in z if x['label']=='CREF_C200'],key=lambda x:int(x['candidate']))
 a=np.array([float(x['union_pct']) for x in new]);b=np.array([float(x['union_pct']) for x in old])
 group[prefix]={'old_I':float(b.mean()),'new_I':float(a.mean()),'median':float(np.median(a)),'increases':int((a>b).sum()),'n2_share_percent':float(100*a[2]/a.sum()),'n4_n6_share_percent':float(100*(a[4]+a[6])/a.sum()),'old_FF_pairs':sum(int(x['FF__proper_transverse_penetration']) for x in old),'new_FF_pairs':sum(int(x['FF__proper_transverse_penetration']) for x in new)}
check(group['3f']['increases']==7 and group['73b']['increases']==6,'tail breadth')
check(group['73b']['old_FF_pairs']==102 and group['73b']['new_FF_pairs']==533,'independent FF proper pairs')
oldreviews=rows('/mnt/e/MeshFlow/experiments/r2_seat_noise_transfer_v1/evaluation/gallery/AI_PROXY_review_unblinded.csv');newreviews=rows(E/'gallery/AI_PROXY_review_unblinded.csv')
oldmap={(x['label'],x['request_id'],x['stage']):x for x in oldreviews};newmap={(x['label'],x['request_id'],x['stage']):x for x in newreviews}
for x in rows(O/'results/historical_review_sensitivity.csv'):
 baseline=x['comparison'].split('-historical-review-')[1];counts=collections.Counter()
 for q in R['requests']:
  if q['panel']!=x['panel']:continue
  av=oldmap[baseline,q['request_id'],x['stage']]['C_adoption'];bv=newmap['CREF_C200',q['request_id'],x['stage']]['C_adoption']
  v='UNCERTAIN' if 'UNCERTAIN' in (av,bv) else 'GAIN' if (av,bv)==('NO','YES') else 'LOSS' if (av,bv)==('YES','NO') else 'SAME';counts[v]+=1
 for k in ('GAIN','LOSS','SAME','UNCERTAIN'):check(int(x.get(k) or 0)==counts[k],'historical sensitivity '+k)
drift={}
for label in ('SMALL_C200','EXP67_FROZEN'):
 d=[]
 for q in R['requests']:
  av=oldmap[label,q['request_id'],'RAW']['C_adoption'];bv=newmap[label,q['request_id'],'RAW']['C_adoption']
  if av!=bv:d.append({'parent':q['parent_id'],'candidate':q['candidate'],'before':av,'after':bv})
 drift[label]=d
check(len(drift['SMALL_C200'])==1 and len(drift['EXP67_FROZEN'])==5,'rereview drift counts')
report=(O/'REPORT.md').read_text(encoding='utf-8-sig')
for x in pn:
 if x['stage']=='POST':
  for k in ('CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','free_minimum_angle_lt5_fraction','A_VMR__raw_tau_1e-6','PMR__raw_tau_1e-6','opposite_winding_reached_free_area_fraction'):
   check(f'{100*float(x[k]):.3f}%' in report,'report percentage '+x['label']+'/'+x['panel']+'/'+k)
costshare=sum(group[k]['new_I']-group[k]['old_I'] for k in group)/(400*(float(pmap['TRAIN_FAMILIAR','CREF_C200','POST']['ConflictCov_C'])-float(pmap['TRAIN_FAMILIAR','SMALL_C200','POST']['ConflictCov_C'])))
out={'status':'PASS' if not errors else 'FAIL','checks':ncheck,'errors':errors,'report_sha256':hashlib.sha256((O/'REPORT.md').read_bytes()).hexdigest(),'scope':'Independent report text review, per-panel percentages, strict RAW metrics, clear gain/loss/uncertainty counts, saved I masks and pair counts, and historical-label sensitivity. No scorer/model call.','I_records':64,'I_tails':group,'familiar_I_increment_share_3f_73b':costshare,'clear_RAW_gains':[{k:x[k] for k in ('panel','parent_id','candidate')} for x in gains],'YES_to_UNCERTAIN':[{k:x[k] for k in ('panel','parent_id','candidate')} for x in yu],'historical_RAW_review_drift':drift,'text_judgment':'The report discloses RAW adoption evidence alongside worsened POST I/FF, weak exact interfaces, CREF0 damage, uncertainty and prior familiarity. It does not equate proximity with connection or the near-parallel I proxy with certified penetration. No promotion, full-equivariance claim, unseen-test claim, improved-diversity claim or unique-source GT ranking is made.','caveat':'Review ZIP contents/size are a subsequent packaging step and are not certified by this text check.','new_model_calls':0,'new_geometry_scans':0,'new_scorer_calls':0,'audit_engineering_note':'First in-memory audit attempt stopped at a missing metrics.exact_duplicate_faces key; this quantity is in the topology section. Corrected read-only accessor; original data unchanged.'}
(A/'report_review.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({k:out[k] for k in ('status','checks','errors','report_sha256','I_tails','familiar_I_increment_share_3f_73b')}))

