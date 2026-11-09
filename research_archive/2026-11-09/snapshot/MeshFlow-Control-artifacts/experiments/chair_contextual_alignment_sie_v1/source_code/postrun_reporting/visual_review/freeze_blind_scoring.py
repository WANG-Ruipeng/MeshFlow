"""Serialize actual manually recorded source-hidden visual observations.
No image synthesis, mesh scoring, method-map access, or reference-image access.
"""
import csv,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/mnt/e/MeshFlow-Control-artifacts/experiments/chair_contextual_alignment_sie_v1')
D=ROOT/'visual_review'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def resolve(s):
 s=s.replace('\\','/')
 return Path('/mnt/'+s[0].lower()+s[2:]) if s[1:2]==':' else Path(s)
def frozen_write(p,b):
 if p.exists():
  if p.read_bytes()!=b:raise RuntimeError('Refuse overwrite frozen review '+str(p))
 else:p.write_bytes(b)
assign=[r for r in read(ROOT/'audit/visual_assignments_anonymous.json')['rows'] if r['reviewer']=='scoring']
notes=read(D/'blind_scoring_observations.json')
assert len(assign)==len(notes)==16
assert [n['index'] for n in notes]==list(range(16))
views=['front_3q','back_3q','front','side','wire']
rows=[];pages=[]
confirmed_large={(9,0),(13,0),(14,2)}
for a,n in zip(assign,notes):
 assert len(n['methods'])==4
 p=resolve(a['blind_page']);h=sha(p)
 pages.append(dict(request_id=a['request_id'],image_path=a['blind_page'],image_sha256=h,
                   viewed_utc=n['observed_utc'],view_scope=views,
                   display=('original_pixels_vertical_crops_0_970_and_950_end' if n['index'] in (4,5,6,7,13) else 'full_original_resolution'),
                   image_modification_saved=False))
 for j,m in enumerate(n['methods']):
  assert len(m)==5
  rows.append(dict(request_id=a['request_id'],parent_id=a['parent_id'],condition_id=a['condition_id'],panel=a['panel'],
    cohort=a['cohort'],candidate=a['candidate'],anonymous_method=f'Method {j+1}',stage='SOURCE_HIDDEN',
    page=a['blind_page'],page_sha256=h,review_kind='AI_VISUAL_REVIEW',reviewer='scoring',
    observed_utc=n['observed_utc'],view_scope=json.dumps(views),review_status='REVIEWED_SOURCE_HIDDEN',
    C_compatibility=m[0],overall_coherence=m[1],missing_body='未见确定主体整体缺失；不据此断言闭合、水密或人类可用。',
    duplicate_overlap=m[2],long_thin_sheet=m[3],style_vs_fragmentation=m[4],
    certainty=json.dumps(dict(body_recognizable='evident',missing_body='not_apparent',
       large_anomalous_sheet='evident' if (n['index'],j) in confirmed_large else 'not_confirmed_or_possible_as_described',
       actual_intersection_or_weld='unable_to_judge_from_images'),ensure_ascii=False),
    render_readability='已实际查看四视图和wire，主体可读；小裂隙、穿插及焊接不可仅凭像素确证。',
    notes='匿名标注、先隐藏来源的AI审阅。此前准备交付时已暴露方法映射；不称严格方法盲评。',
    human_usability='NOT_RUN',prior_mapping_exposure=True,reference_viewed_before_this_observation=False))
assert len(rows)==64 and len({(r['request_id'],r['anonymous_method']) for r in rows})==64
import io
f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
frozen_write(D/'blind_scoring.csv',f.getvalue().encode('utf-8'))
receipt=dict(status='COMPLETE_ASSIGNED_SOURCE_HIDDEN_REVIEW',scope='four assigned P1 parents x four noises x four anonymous methods',
  actual_pages_viewed=16,actual_output_rows=64,view_scope=views,pages=pages,
  observations_sha256=sha(D/'blind_scoring_observations.json'),csv_sha256=sha(D/'blind_scoring.csv'),
  frozen_utc=datetime.now(timezone.utc).isoformat(),prior_mapping_exposure=True,
  exposure_note='Reviewer encountered method mapping during earlier implementation/schema work. Current image review did not reopen mapping, numerical results, or reference images. This is not strict method-blind review.',
  reference_images_opened_this_stage=0,numerical_results_read_this_stage=False,
  human_usability='NOT_RUN',usable_rate_computed=False,model_calls=0,scoring_calls=0,render_calls=0,
  report_scope='Image observations only; no exact intersection, connectivity, weld, watertightness, or usability claim.',
  next_stage='Wait for root confirmation that global source-hidden review is frozen before opening reference pages.')
frozen_write(D/'blind_scoring_receipt.json',(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
print(json.dumps({k:v for k,v in receipt.items() if k!='pages'},ensure_ascii=False,indent=2))
