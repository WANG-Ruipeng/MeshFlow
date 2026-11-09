"""Merge already completed AI_PROXY reviews without image reinterpretation.
Standard library only: reads judgments/registration and hashes original PNG files.
"""
from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,json
ROOT=Path('/mnt/e/MeshFlow-Control-artifacts/experiments/r2_user_condition_finetune_pilot_v1')
REV=ROOT/'review';AUDIT=ROOT/'audit'
LABELS=['R2_FROZEN','R2_CONT_OLD','R2_CONT_USER']
COMPS=['USER_vs_OLD','USER_vs_FROZEN'];METRICS=['overall','C_use']
RATINGS=['improved','similar','degraded','uncertain']
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,sort_keys=True)+'\n',encoding='utf-8')
def rows(x):return x if isinstance(x,list) else x['rows']
def csvwrite(p,rr):
 fields=list(dict.fromkeys(k for r in rr for k in r))
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rr:w.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()})

def main():
 paths={n:REV/n for n in ['new_review_notes.json','old_first8.json','paired_old_first8.json','old_last8.json','paired_old_last8.json']}
 paths['notes_recovery']=AUDIT/'old_first8_notes_utf8_recovery.json'
 paths['registration']=ROOT/'evaluation/registration.json'
 source_hashes={str(p):sha(p) for p in paths.values()}
 data={n:read(p) for n,p in paths.items()};reg=data['registration'];requests=reg['requests']
 assert len(requests)==28 and reg['labels']==LABELS
 reqmap={r['request_id']:r for r in requests};order={r['request_id']:i for i,r in enumerate(requests)}
 assert len(reqmap)==28
 newnotes=data['new_review_notes.json'];first=data['old_first8.json'];fp=data['paired_old_first8.json'];last=data['old_last8.json'];lp=data['paired_old_last8.json'];recovery=data['notes_recovery']
 assert len(newnotes['rows'])==12 and len(rows(first))==len(rows(last))==24 and len(rows(fp))==len(rows(lp))==8
 assert len(recovery['output_notes'])==24 and len(recovery['paired_notes'])==8
 last_new_condition=newnotes['rows'][-1]['prefix']
 image_table={}
 for r in requests:
  image_table[r['request_id']]={}
  for mode in ['colored','neutral']:
   p=ROOT/'gallery'/mode/(r['request_id']+'.png');assert p.is_file()
   image_table[r['request_id']][mode]=dict(path=str(p),sha256=sha(p),actually_viewed=True,views_actually_viewed=['front_oblique','opposite_oblique'],model_columns_actually_viewed=LABELS,merge_reinspection=False)
 expected_images={v['path'] for e in image_table.values() for v in e.values()}
 actual_images={str(p) for mode in ['colored','neutral'] for p in (ROOT/'gallery'/mode).glob('*.png')}
 assert expected_images==actual_images and len(actual_images)==56
 outputs=[];paired=[]
 def base(r,reviewer,src,index,disclosure):
  return dict(request_id=r['request_id'],registration_order_index=order[r['request_id']],parent_id=r['parent_id'],condition_id=r['condition_id'],candidate=r['candidate'],panel=r['panel'],region_type=r['region_type'],tier=r['tier'],N=r['N'],K=r['K'],input_path=r['input_path'],input_sha256=r['input_sha256'],reviewer=reviewer,review_type='AI_PROXY',human_rating=False,labels_visible=True,model_labels_visible=True,blinded=False,actually_viewed=True,both_modes_examined=True,both_views_examined=True,evidence=image_table[r['request_id']],source_review_path=str(paths[src]),source_review_sha256=source_hashes[str(paths[src])],source_row_index=index,review_disclosure=disclosure)
 def verify_evidence(row):
  rid=row['request_id'];ev=row.get('evidence',row.get('images',{}))
  for mode in ['colored','neutral']:
   oldsha=ev.get(mode,{}).get('sha256',row.get(mode+'_sha256'))
   if oldsha is not None:assert oldsha==image_table[rid][mode]['sha256']
 def new_disclosure(r):
  return dict(numeric_scores_seen_before_review=False,detailed_numeric_scores_read_before_review=False,aggregate_hint_received_during_last_condition=True,aggregate_hint_possible_for_this_condition=r['condition_id']==last_new_condition,aggregate_hint_condition_id=last_new_condition,aggregate_hint_content='During the last NEWDEV condition review the root received a teammate aggregate: NEWDEV CF/FF slightly lower, I/slender quality worse. No detailed per-output score table was read before review.',numeric_information_blinding='NOT_FULLY_INDEPENDENT_OF_NUMERIC_INFORMATION',exposure_timing_within_last_two_noise_requests='not separately recorded; conservatively mark both noises of last condition',disclosure_source='Root merge-task instruction; original notes retained verbatim')
 for i,n in enumerate(newnotes['rows']):
  matches=[r for r in requests if r['panel']=='NEWDEV' and r['condition_id']==n['prefix'] and r['candidate']==n['noise']]
  assert len(matches)==1;r=matches[0];assert all(len(n[k])==3 for k in ['notes','overall','C_use'])
  b=base(r,'root','new_review_notes.json',i,new_disclosure(r))
  for j,label in enumerate(LABELS):
   outputs.append(dict(b,model=label,label=label,overall=n['overall'][j],C_use=n['C_use'][j],overall_status=n['overall'][j],C_use_status=n['C_use'][j],overall_notes=n['notes'][j],C_use_notes=n['notes'][j],artifact_observations=None,uncertainty=None,original_combined_note=n['notes'][j],absolute_status_vocabulary='root NEWDEV original codes; not recoded',notes_encoding_recovered=False))
  paired.append(dict(b,USER_vs_OLD=dict(n['vs_old']),USER_vs_FROZEN=dict(n['vs_frozen']),notes=None,notes_encoding_recovered=False))
 first_disclosure={k:v for k,v in first.items() if k!='rows'}
 first_disclosure.update(aggregate_hint_received_during_last_condition=False,numeric_information_blinding='CURRENT_NUMERICAL_SCORES_NOT_READ_BY_REVIEWER; model labels visible')
 for i,row in enumerate(rows(first)):
  r=reqmap[row['request_id']];assert r['panel']=='OLDREG' and row['actually_viewed'];verify_evidence(row)
  assert row['model']==LABELS[i%3]
  oldnotes=[row[k] for k in ['overall_notes','C_use_notes','artifact_observations','uncertainty']]
  restored=recovery['output_notes'][i];assert len(restored)==4 and all('?' in x for x in oldnotes)
  b=base(r,row['reviewer'],'old_first8.json',i,first_disclosure)
  outputs.append(dict(b,model=row['model'],label=row['model'],overall=row['overall'],C_use=row['C_use'],overall_status=row['overall'],C_use_status=row['C_use'],overall_notes=restored[0],C_use_notes=restored[1],artifact_observations=restored[2],uncertainty=restored[3],original_combined_note=None,absolute_status_vocabulary='old_first8 original codes; not recoded',notes_encoding_recovered=True,notes_recovery_path=str(paths['notes_recovery']),notes_recovery_sha256=sha(paths['notes_recovery'])))
 for i,row in enumerate(rows(fp)):
  r=reqmap[row['request_id']];verify_evidence(row)
  b=base(r,row['reviewer'],'paired_old_first8.json',i,first_disclosure)
  paired.append(dict(b,USER_vs_OLD=dict(row['USER_vs_OLD']),USER_vs_FROZEN=dict(row['USER_vs_FROZEN']),notes=recovery['paired_notes'][i],notes_encoding_recovered=True,notes_recovery_path=str(paths['notes_recovery']),notes_recovery_sha256=sha(paths['notes_recovery'])))
 for i,row in enumerate(rows(last)):
  r=reqmap[row['request_id']];assert row['actually_viewed'] and row['both_modes_examined'] and row['both_views_examined'];verify_evidence(row)
  disclosure={k:row[k] for k in ['quality_scores_read','source_GT_compared','prior_experiment_familiarity','review_completed_utc','uncertainty']}
  disclosure.update(aggregate_hint_received_during_last_condition=False,numeric_information_blinding='CURRENT_NUMERICAL_SCORES_NOT_READ_BY_REVIEWER; model labels visible')
  b=base(r,row['reviewer'],'old_last8.json',i,disclosure)
  outputs.append(dict(b,model=row['label'],label=row['label'],overall=row['overall'],C_use=row['C_use'],overall_status=None,C_use_status=row.get('C_use_status'),overall_notes=row['overall'],C_use_notes=row['C_use'],artifact_observations=row['artifacts'],uncertainty=row['uncertainty'],original_combined_note=None,absolute_status_vocabulary='old_last8 original prose plus original C_use_status; no inferred overall code',notes_encoding_recovered=False))
 for i,row in enumerate(rows(lp)):
  r=reqmap[row['request_id']];verify_evidence(row)
  disclosure={k:row[k] for k in ['quality_scores_read','source_GT_compared','prior_experiment_familiarity','review_completed_utc','uncertainty']}
  disclosure.update(aggregate_hint_received_during_last_condition=False,numeric_information_blinding='CURRENT_NUMERICAL_SCORES_NOT_READ_BY_REVIEWER; model labels visible')
  b=base(r,row['reviewer'],'paired_old_last8.json',i,disclosure)
  paired.append(dict(b,USER_vs_OLD=dict(overall=row['vs_OLD_overall'],C_use=row['vs_OLD_C_use']),USER_vs_FROZEN=dict(overall=row['vs_FROZEN_overall'],C_use=row['vs_FROZEN_C_use']),notes=row['comparison_note'],notes_encoding_recovered=False))
 outputs.sort(key=lambda x:(order[x['request_id']],LABELS.index(x['model'])));paired.sort(key=lambda x:order[x['request_id']])
 expected_outputs={(r['request_id'],label) for r in requests for label in LABELS}
 assert len(outputs)==84 and {(x['request_id'],x['model']) for x in outputs}==expected_outputs
 assert len(paired)==28 and {x['request_id'] for x in paired}==set(reqmap)
 assert all(x[c][m] in RATINGS for x in paired for c in COMPS for m in METRICS)
 assert all(x['actually_viewed'] and x['labels_visible'] and not x['human_rating'] for x in outputs+paired)
 metadata=dict(schema='r2_user_condition_ai_proxy_merged_v1',review_type='AI_PROXY',human_rating=False,labels_visible=True,model_labels_visible=True,blinded=False,aggregate_hint_received_during_last_condition=True,numerical_information_blinding='MIXED_REVIEWER_DISCLOSURES; ROOT_NEWDEV_LAST_CONDITION_RECEIVED_AGGREGATE_HINT',merge_reinspection=False,judgments_changed=False,source_hashes=source_hashes,merge_script_path=str(Path(__file__)),merge_script_sha256=sha(Path(__file__)),notes_encoding_recovery=dict(applied_to='old_first8 notes only',reason='PowerShell-to-WSL stdin encoded prior Chinese prose as question marks; judgment codes, IDs and image hashes were intact',recovery_source='exact earlier conversation review-writing text',original_files_preserved=True,recovery_path=str(paths['notes_recovery']),recovery_sha256=sha(paths['notes_recovery']),rejudgment=False),execution_counts=dict(model_calls=0,geometry_recomputation=0,OT_calls=0,noise_draws=0,render_calls=0,new_image_reviews=0),interpretation='Pair categories preserved exactly. Similar or uncertain is not counted as successful C use. Absolute reviewer vocabularies remain separate and are not silently harmonized.')
 save(REV/'per_output_ai_proxy.json',dict(metadata,schema='r2_user_condition_ai_proxy_per_output_v1',rows=outputs))
 save(REV/'paired_ai_proxy.json',dict(metadata,schema='r2_user_condition_ai_proxy_paired_v1',rows=paired))
 csvwrite(REV/'per_output_ai_proxy.csv',outputs);csvwrite(REV/'paired_ai_proxy.csv',paired)
 scopes=[('NEWDEV','ALL'),('NEWDEV','seat'),('NEWDEV','back'),('OLDREG','ALL')]
 summaries=[];summary_csv=[]
 for panel,region in scopes:
  rr=[x for x in paired if x['panel']==panel and (region=='ALL' or x['region_type']==region)]
  group=dict(panel=panel,region=region,request_count=len(rr),condition_count=len({x['condition_id'] for x in rr}),parent_count=len({x['parent_id'] for x in rr}))
  for comp in COMPS:
   for metric in METRICS:
    counts=Counter(x[comp][metric] for x in rr);entry=dict(group,comparison=comp,metric=metric,counts={k:counts[k] for k in RATINGS});summaries.append(entry)
    for rating in RATINGS:summary_csv.append(dict(row_type='paired_request_count',**group,comparison=comp,metric=metric,rating=rating,count=counts[rating],condition_id='',parent_id='',noise0_rating='',noise1_rating='',two_noise_consistent='',contains_uncertain=''))
 consistency=[]
 grouped=defaultdict(list)
 for x in paired:grouped[(x['panel'],x['condition_id'])].append(x)
 assert len(grouped)==14
 for (panel,condition),rr in grouped.items():
  rr=sorted(rr,key=lambda x:x['candidate']);assert len(rr)==2 and [x['candidate'] for x in rr]==[0,1] and len({x['parent_id'] for x in rr})==1
  for comp in COMPS:
   for metric in METRICS:
    ratings=[x[comp][metric] for x in rr];same=ratings[0]==ratings[1]
    entry=dict(panel=panel,region=rr[0]['region_type'],condition_id=condition,parent_id=rr[0]['parent_id'],comparison=comp,metric=metric,request_ids=[x['request_id'] for x in rr],noise0_rating=ratings[0],noise1_rating=ratings[1],two_noise_consistent=same,contains_uncertain='uncertain' in ratings,condition_verdict=ratings[0] if same else 'mixed_across_two_noises',both_noise_improved=ratings==['improved','improved'])
    consistency.append(entry)
    summary_csv.append(dict(row_type='condition_two_noise_consistency',panel=panel,region=rr[0]['region_type'],request_count=2,condition_count=1,parent_count=1,comparison=comp,metric=metric,rating=entry['condition_verdict'],count=1,condition_id=condition,parent_id=rr[0]['parent_id'],noise0_rating=ratings[0],noise1_rating=ratings[1],two_noise_consistent=same,contains_uncertain=entry['contains_uncertain']))
 absolute=[]
 for panel,region in scopes:
  rr=[x for x in outputs if x['panel']==panel and (region=='ALL' or x['region_type']==region)]
  for label in LABELS:
   cc=Counter(x['C_use_status'] if x['C_use_status'] is not None else 'NOT_CODED' for x in rr if x['model']==label)
   absolute.append(dict(panel=panel,region=region,model=label,original_C_use_status_counts=dict(cc),automatic_success_mapping=False))
 summary=dict(metadata,schema='r2_user_condition_ai_proxy_summary_v1',paired_request_counts=summaries,condition_two_noise_consistency=consistency,absolute_original_C_use_status_counts=absolute,denominator_policy='28 requests on14 panel-specific conditions and8 unique parents; two noises are not independent parents. NEWDEV6 parents are also included among OLDREG8. Panels and NEWDEVseat/back are reported separately; no pooled success rate.',uncertain_is_success=False,similar_is_success=False)
 save(REV/'summary_ai_proxy.json',summary);csvwrite(REV/'summary_ai_proxy.csv',summary_csv)
 coverage_rows=[]
 for r in requests:
  rr=[x for x in outputs if x['request_id']==r['request_id']]
  coverage_rows.append(dict(request_id=r['request_id'],panel=r['panel'],region_type=r['region_type'],condition_id=r['condition_id'],parent_id=r['parent_id'],candidate=r['candidate'],models=[x['model'] for x in rr],output_rows=len(rr),paired_rows=sum(x['request_id']==r['request_id'] for x in paired),reviewers=sorted({x['reviewer'] for x in rr}),colored=image_table[r['request_id']]['colored'],neutral=image_table[r['request_id']]['neutral'],all_viewed_by_original_reviewer=all(x['actually_viewed'] for x in rr),aggregate_hint_possible=any(x['review_disclosure'].get('aggregate_hint_possible_for_this_condition',False) for x in rr)))
 assert all(x['output_rows']==3 and x['paired_rows']==1 and x['all_viewed_by_original_reviewer'] for x in coverage_rows)
 assert all(sha(Path(v['path']))==v['sha256'] for e in image_table.values() for v in e.values())
 assert all(sha(Path(p))==h for p,h in source_hashes.items())
 coverage=dict(metadata,schema='r2_user_condition_ai_proxy_review_coverage_v1',status='PASS',per_output_rows=84,paired_rows=28,original_png_count=56,fully_covered=True,missing_outputs=[],missing_requests=[],missing_images=[],input_and_image_hashes_unchanged=True,panel_condition_counts={'NEWDEV':6,'OLDREG':8},unique_parent_count=len({r['parent_id'] for r in requests}),reviewer_source_metadata={'root':{k:v for k,v in newnotes.items() if k!='rows'},'old_first8':{k:v for k,v in first.items() if k!='rows'},'old_last8':{k:rows(last)[0][k] for k in ['reviewer','review_type','labels_visible','blinded','quality_scores_read','prior_experiment_familiarity','review_completed_utc']}},rows=coverage_rows)
 save(REV/'review_coverage.json',coverage)
 print(json.dumps(dict(status='PASS',outputs=84,paired=28,original_images=56,unique_parents=coverage['unique_parent_count'],summary=summaries,condition_consistency_count=len(consistency)),ensure_ascii=False,indent=2))

if __name__=='__main__':main()