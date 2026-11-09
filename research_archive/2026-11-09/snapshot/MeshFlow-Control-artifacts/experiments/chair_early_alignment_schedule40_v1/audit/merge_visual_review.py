import csv,json,hashlib,time
from pathlib import Path
r=Path('/mnt/e/MeshFlow-Control-artifacts/experiments/chair_early_alignment_schedule40_v1')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
p1=r/'evaluation/visual_review_part1.csv';p2=r/'evaluation/visual_review_part2.csv'
a=list(csv.DictReader(p1.open(encoding='utf-8-sig')));b=list(csv.DictReader(p2.open(encoding='utf-8-sig')))
assert len(a)==len(b)==24
mapping_path=r/'gallery/method_mapping.json'
mapping=json.loads(mapping_path.read_text())
rev={v:k for k,v in mapping.items()}
pages=json.loads((r/'gallery/manifest.json').read_text())['pages']
page_by_id={p['parent_id']:p for p in pages}
rows=[]
for source,arr in [('part1',a),('part2',b)]:
 for x in arr:
  parent=x.get('parent_id',x.get('parent'))
  page=page_by_id[parent]
  d=dict(parent_id=parent,request_id=page['request_id'],label=rev[x['anonymous_method']],anonymous_method=x['anonymous_method'],review_source=source,whole_shape_observation=x['whole_shape_observation'],method_observation=x.get('method_observation',''),condition_placement_observation=x.get('condition_placement_observation',''),visible_layering_or_intersection_observation=x.get('visible_layering_or_intersection_observation',''),fragments_or_collapse_observation=x.get('fragments_or_collapse_observation',''),input_C_caveat=x.get('input_C_caveat',x.get('input_c_observation')),views_reviewed='front_3q;back_3q;front;side;wire',render_modes='neutral;colored',limitations=x.get('limitations',x.get('scope')),human_rating='NOT_RUN',mapping_read_before_anonymous_recording=False,numerical_results_exposure_before_review='YES_SUMMARIES' if source=='part1' else 'NO',source_anonymous_csv='evaluation/visual_review_'+source+'.csv')
  for mode in ['neutral','colored']:
   path=r/'gallery'/mode/(page['request_id']+'.png')
   assert sha(path)==page[mode+'_sha256']
   d[mode+'_page']=str(path.relative_to(r));d[mode+'_sha256']=sha(path)
  rows.append(d)
rows.sort(key=lambda x:(x['parent_id'],x['anonymous_method']))
assert len(set((x['parent_id'],x['label']) for x in rows))==48
out=r/'evaluation/visual_review_per_output.csv'
with out.open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
m={'status':'COMPLETE','outputs_reviewed':48,'parents_reviewed':8,'actual_png_pages_reviewed':16,'views_per_mode':5,'source_anonymous_records_frozen_before_mapping_read':True,'human_rating':'NOT_RUN','formal_blinded_human_study':False,'root_had_received_numeric_summaries':True,'new_renders':0,'original_render_manifest_left_unchanged':True,'render_manifest_pending_is_generation_time_snapshot':'Superseded for actual AI review status by this separate completion receipt; original image artifact metadata unchanged.','files':{str(p.relative_to(r)):sha(p) for p in [p1,p2,r/'evaluation/visual_review_part1.md',r/'evaluation/visual_review_part2.md',mapping_path,out]},'script_sha256':sha(Path(__file__)),'generated_unix':time.time()}
(r/'audit/visual_review_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps({'status':'COMPLETE','rows':len(rows),'per_output_csv_sha256':sha(out)}))