"""Independent set-based pair provenance accounting from frozen full-scan evidence."""
import csv,json,hashlib,collections
from pathlib import Path
import numpy as np
BASE=Path('/mnt/e/MeshFlow');OUT=BASE/'experiments/r2_multistep_denoise_v1';OLD=BASE/'experiments/r2_postprocess_avmr_pmr_v1';DEST=OUT/'independent_review'
def read(p):return json.loads(Path(p).read_text())
def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig',newline='')))
def check(v,msg):
 if not v:raise AssertionError(msg)
idx={(r['request_id'],r['stage']):r for r in rows(OUT/'post_index.csv')}
labels={'proper_transverse_penetration','coplanar_positive_area_overlap'}
def pairsets(r):
 with np.load(r['post_path'],allow_pickle=False) as a:source=a['source_face_ids'][int(r['K']):].copy()
 path=(OLD/'scores/R2_USER_MIX'/r['request_id'] if r['stage']=='D1' else OUT/'scores'/r['stage']/r['request_id'])/'quality/pairs.json'
 f=read(path)['FF'];check(f['expected_full_pairs']==len(source)*(len(source)-1)//2,'Full pair set expected')
 actual={}
 for p in f['contacts']:
  a,b=[int(source[int(k)]) for k in p['pair']];actual[frozenset((a,b))]=p['classification']
 bad={p for p,c in actual.items() if c in labels};return set(source.tolist()),bad,actual
per=rows(OUT/'FF_pair_fates_D1_to_endpoints.csv');check(len(per)==24,'24 pair fate totals');sets={};counts={}
for r in per:
 rid,stage=r['request_id'],r['stage'];a,ab,ac=pairsets(idx[(rid,'D1')]);b,bb,bc=pairsets(idx[(rid,stage)])
 gone={p for p in ab if not p<=b};retained=ab-gone;still=retained&bb;resolved=retained-bb
 c=dict(D1_illegal_pairs=len(ab),endpoint_illegal_pairs=len(bb),D1_illegal_both_faces_retained_still_illegal=len(still),D1_illegal_both_faces_retained_now_nonillegal=len(resolved),D1_illegal_one_or_both_faces_deleted=len(gone),endpoint_illegal_pairs_not_illegal_at_D1=len(bb-ab),D1_retained_faces_later_deleted=len(a-b),D1_deleted_faces_reappeared=len(b-a))
 for key,val in c.items():check(int(r[key])==val,'Fate count mismatch '+key)
 sets[(rid,stage)]=(gone,still,resolved,bc);counts[(rid,stage)]=c
for r in rows(OUT/'FF_pair_fates_detail.csv'):
 gone,still,resolved,bc=sets[(r['request_id'],r['stage'])];pair=frozenset((int(r['source_face_a']),int(r['source_face_b'])))
 expected='one_or_both_faces_deleted' if pair in gone else 'both_faces_retained_still_illegal' if pair in still else 'both_faces_retained_now_nonillegal' if pair in resolved else None
 check(expected==r['fate'],'Detailed source pair fate');classification='NA_DELETED' if pair in gone else bc.get(pair,'not_in_contact_list_full_scan_disjoint')
 check(classification==r['endpoint_classification'],'Saved classification identity')
for r in rows(OUT/'FF_pair_fates_summary.csv'):
 sub=[v for v in per if v['stage']==r['stage'] and (r['group']=='ALL' or v['region_type']==r['group'])]
 check(int(r['outputs'])==len(sub),'Fate summary denominator')
 for key in next(iter(counts.values())):check(int(r[key])==sum(counts[(v['request_id'],v['stage'])][key] for v in sub),'Fate sum '+key)
report=dict(status='PASS',per_output_rows=24,detail_rows=len(rows(OUT/'FF_pair_fates_detail.csv')),summary_rows=6,model_forwards=0,geometry_recomputations=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limitation='Recombines saved classifications; does not independently classify triangle geometry.')
(DEST/'FF_fates_REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))