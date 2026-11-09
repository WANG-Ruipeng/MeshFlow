"""Recorded visual screen after export; preserve every rejected artifact."""
from prepare import DEFAULT_OUT,read,write,sha,event
def main(out=DEFAULT_OUT):
 src=read(out/'conditions.json');rows=src['rows']
 held={('21f2','seat'),('21f2','back'),('47da','seat'),('47da','back'),('73b3','seat'),('c1b8','seat')}
 decisions=[]
 for r in rows:
  hold=r['tier']=='small' and any(r['uid'].startswith(p) and r['region_type']==k for p,k in held)
  if not hold:continue
  image=out/'gallery/conditions'/(r['condition_id']+'.png')
  assert image.exists() and r['region_area_fraction']<.1
  decisions.append(dict(condition_id=r['condition_id'],uid=r['uid'],split=r['split'],
   status='HOLD_TINY_EDGE_STRIP_AI_PROXY',include_in_active_pool=False,
   reason='Actual two-view five-column image shows only a narrow boundary strip/needle, not a substantial small patch. Whole-face prefix cannot approximate the intended small region at this source resolution; preserve as rejected preparation evidence, omit from this pilot main pool. This is not a universal claim that an edge strip can never be a valid user input.',
   region_area_fraction=r['region_area_fraction'],image=str(image),image_sha256=sha(image),
   image_actually_viewed=True,reviewer='root AI_PROXY',original_npz=r['npz'],original_npz_sha256=r['sha256']))
 assert len(decisions)==6 and sum(r['split']=='train' for r in decisions)==5
 write(out/'usability_review.json',dict(schema='post_construction_input_fidelity_review_v1',
  policy='AI visual input-fidelity review before any training/output generation; region area fraction below 10 percent triggered close inspection, not an automatic geometric-invalidity label. All six inspected candidates were thin strips and held. No data or pictures deleted, no new seed or task substituted.',
  source_manifest_sha256=sha(out/'conditions.json'),rows=decisions,model_output_selection=False,
  coverage_limitation='Not every one of 129 extent images separately visually audited. All source regions and these six low-area cases reviewed; full geometry/partition programmatic validation covers all129.',
  training_authorization=False))
 exclude={r['condition_id'] for r in decisions}
 active=[r for r in rows if r['condition_id'] not in exclude]
 value=dict(schema='r2_user_conditions_active_v1',rows=active,source_all_exported_manifest=str(out/'conditions.json'),
  source_all_exported_sha256=sha(out/'conditions.json'),usability_review_sha256=sha(out/'usability_review.json'),
  train_condition_count=sum(r['split']=='train' for r in active),dev_condition_count=sum(r['split']=='dev' for r in active),
  train_parents=sorted({r['uid'] for r in active if r['split']=='train'}),
  held_condition_ids=sorted(exclude),new_generated_arrays=0,
  readiness='REVIEW_PACKAGE_COMPLETE; future training requires its own R2 continuation contract',
  future_recipe='R2 Native/T1+role+Geo, FM, full-course HYBRID=.25')
 assert value['train_condition_count']==117 and value['dev_condition_count']==6 and len(value['train_parents'])==25
 write(out/'active_conditions.json',value)
 event(out,'POST_CONSTRUCTION_USABILITY_REVIEW',active_train=117,active_dev=6,held_train=5,held_dev=1,model_calls=0)
 print('Active117 train /6DEV; all129 original exports retained; six narrow strips held.')
if __name__=='__main__':main()
