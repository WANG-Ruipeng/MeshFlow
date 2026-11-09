"""Apply recorded AI visual decisions; this is not a semantic classifier."""
from pathlib import Path
from collections import Counter
from prepare import DEFAULT_OUT,read,write,sha,geometry,event
from construct import nested_extents
from quality import partition_condition

def main(out=DEFAULT_OUT):
 out=Path(out); draft=read(out/'reviews/draft_regions.json')['rows']
 parents={p['uid']:p for p in read(out/'parent_inventory.json')['rows']}
 supplemental={'12c2','3ede','5869','71db','73b3','e682'}
 rows=[]
 for r in draft:
  r=dict(r);uid=r['uid'];kind=r['region_type']
  r['original_review_status']=r['status']
  r['reviewer']='root AI_PROXY, using independent per-parent AI visual reviews'
  r['human_semantic_annotation']=False
  r['labels_fed_to_model']=False
  r['final_approval']=False
  r['data_preparation_decision']='FINAL_FOR_REVIEW_PACKAGE; no training authorization implied'
  if uid.startswith('5574') and kind=='back':
   r['status']='NA_DISCONNECTED_REGION'
   r['reason']='The two seat-facing supporting panels p01+p04 do not share complete source edges. This pilot does not bridge them or relabel one isolated panel as the complete back region.'
  elif uid.startswith('e682') and kind=='back':
   r['source_face_ids']=[168,170,172,173,174,175,176]
   r['status']='REVIEWED_SUBSET'
   r['reason']='Supplemental views confirm the complete fan-shaped seat-facing supporting panel. Omit face30, which extends below the seat to a rear leg. The bevel/top shell and reverse face remain in Y; full means this front panel R.'
  if r['source_face_ids'] and not r['status'].startswith('NA'):
   a=geometry(parents[uid])
   ext,growth=nested_extents(a['full_target'],a['source_faces'],r['source_face_ids'],kind)
   q=partition_condition(uid,r['source_face_ids'],out/'quality')
   r['quality_at_R']={k:dict(confirmed_pairs=v['confirmed_illegal_pair_count'],unknown_pairs=v['unknown_pair_count']) for k,v in q['partitions'].items()}
   assert not q['C_nonfinite_source_face_ids'] and not q['C_zero_area_source_face_ids'] and not q['C_duplicate_groups']
   risk=q['parent_confirmed_illegal_pair_count']>0 or q['parent_unknown_pair_count']>0
   r['status']='ACCEPTED_WITH_RISK_AI_PROXY' if risk else 'ACCEPTED_AI_PROXY'
   r['whole_source_risk']=risk
   r['extent_preview']=[dict(K=len(e['source_face_ids']),tier=e['tier'],requested_tiers=e['requested_tiers'],region_area_fraction=e['region_area_fraction']) for e in ext]
   if uid.startswith('3ede') and kind=='seat':
    pp=q['partitions']['CF']['problem_pairs']
    overlap_sum=sum(p['detector']['primary'].get('coplanar_overlap_area',0) for p in pp)
    assert len(pp)==6 and all(p['category']=='coplanar_positive_overlap' for p in pp)
    r['reason']+=' Retain with localized known risk: six positive-area overlap pairs with support end caps at rear seat corners; pair-area sum '+str(overlap_sum)+' (not union area). No duplicate full seat layer is inferred. M remains unchanged.'
   if uid.startswith('7f82') and kind=='back':
    r['reason']+=' One internal pair (192,195) is UNKNOWN in the frozen detector; retained explicitly as unresolved, not certified intersection-free.'
   if uid.startswith('3ede') and kind=='back':
    r['reason']='The sole upper backrest board front face, excluding both descending posts. Full means the entire declared supporting board R, not a filled solid rectangle between seat and top nor the whole frame.'
   if uid.startswith('71db'):
    r['reason']+=' Curved inner shell divided at its approximately 45-degree transition; an operational whole-face boundary, not a semantic ground-truth boundary. Border/transition faces outside the declared R stay in Y.'
   if any(uid.startswith(prefix) for prefix in supplemental):
    image=out/'gallery/subset_02'/f'{uid}__regions.png'
    assert image.exists()
    r['visual_evidence'].append(dict(path=str(image),sha256=sha(image),actually_viewed=True,reviewer='root AI_PROXY',purpose='Exact whole-source-face subset review'))
  rows.append(r)
 rows.extend(read(out/'reviews/dev_root.json')['rows'])
 assert len([r for r in rows if r['split']=='train'])==52
 assert len(rows)==60
 accepted=[r for r in rows if r['status'].startswith('ACCEPTED')]
 value=dict(schema='chair_r2_user_conditions_reviewed_regions_v1',rows=rows,
  counts=dict(Counter(r['status'] for r in rows)),semantic_authority='AI_PROXY only; not human annotated or externally validated',
  fraction_denominator='Declared coherent surface region R, not necessarily all faces of an anatomical part or of M',
  model_input_semantics=False,training_authorization=False,
  original_reviews={str(p):sha(p) for p in [out/'reviews/train_first13.json',out/'reviews/train_last13.json',out/'reviews/dev_root.json',out/'reviews/draft_region_checks.json']})
 write(out/'regions.json',value)
 event(out,'REGIONS_FROZEN_FOR_REVIEW',train_regions=sum(r['split']=='train' for r in accepted),dev_regions=sum(r['split']=='dev' for r in accepted),review_kind='AI_PROXY')
 print(value['counts'])
if __name__=='__main__':main()
