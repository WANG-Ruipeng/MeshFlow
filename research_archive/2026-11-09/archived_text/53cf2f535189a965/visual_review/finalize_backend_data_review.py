"""Attach immutable evidence to manually authored actual-image observations; no scoring."""
from pathlib import Path
import json, hashlib

OUT = Path('/mnt/e/MeshFlow/experiments/r2_bench_raw_local_repair_v1')
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
notes = read(OUT/'visual_review/manual_notes_backend_data.json')
manifest = read(OUT/'data/manifest.json')
axes = ['overall_shape','C_use','support_retention','visible_fragment_or_gap','layering_or_fold']
columns = ['RAW','TARGET','GEO_TARGET','FINAL_TARGET','SHAPE_FINAL','RELDEF_FINAL']
records=[]
accepted_count=0
for n in notes['records']:
    t=manifest['tasks'][n['index']]
    rid=t['request_id']
    assert rid.startswith(n['request_prefix']) and t['noise_index']==n['noise_index']
    gd=OUT/'gallery'/rid
    viewed=n['viewed_files']
    assert len(viewed)==len(set(viewed))
    image_sources=[{'path':str(gd/f),'sha256':sha(gd/f),'actually_viewed':True} for f in viewed]
    pd=read(gd/'patch_display.json')
    patch_rows=[]
    for p in pd['rows']:
        # Labels come only from the preceding manual image review. This merely verifies coverage.
        image_file=Path(p['image_path']).name
        assert image_file in viewed,(rid,image_file)
        patch_rows.append({'patch_id':p['patch_id'],'decision':p['decision'],'actually_viewed':True,
                           'image_path':str(gd/image_file),'image_sha256':sha(gd/image_file),
                           'visual_local_benefit':'UNCERTAIN' if p['decision']=='ACCEPTED' else 'NOT_A_COMMITTED_TARGET',
                           'note':'See manually authored accepted_patch_review; image review does not certify exact intersection removal.'})
        accepted_count+=int(p['decision']=='ACCEPTED')
    cells=[]
    assert len(n['axes_by_column'])==6
    for label, values in zip(columns,n['axes_by_column']):
        assert len(values)==5
        cells.append({'label':label,'actually_viewed':True,'reviewer_kind':'AI_PROXY',**dict(zip(axes,values))})
    records.append({'request_id':rid,'parent_id':t['parent_id'],'noise_index':t['noise_index'],
      'eval_group':t['eval_group'],'actually_viewed':True,'reviewer_kind':'AI_PROXY','blind':False,
      'image_sources':image_sources,'cells':cells,
      'matched_final_visible_regression':n['matched_final_visible_regression'],
      'protected_support_loss':n['protected_support_loss'],'credible_local_EDIT':n['credible_local_EDIT'],
      'notes':n['notes'],'accepted_patch_review':n['accepted_patch_review'],'patches':patch_rows})
assert len(records)==12 and len({r['request_id'] for r in records})==12
assert accepted_count==56,accepted_count
result={'schema':'bench_raw_local_repair_actual_image_review_v1','reviewer':'backend_data',
 'reviewer_kind':'AI_PROXY','blind':False,'actually_viewed':True,
 'scope':'12 requests; six columns, paired no-D0 FINAL control, support sections, corner states, all 56 accepted patches and first two rejected per request where present.',
 'limitations':['Not independent human review. Named methods were visible.',
 'Numerical progress messages were received, but five-axis and local credibility labels were assigned from actual images, not score thresholds.',
 'No obvious gross damage does not establish exact topology preservation or trustworthy training targets.',
 'Subpixel or occluded local changes remain UNCERTAIN. UNKNOWN target coordinates are not ground truth.',
 'A cut missing a thin support does not establish absence; both surface views and fixed support sections were inspected.'],
 'source_manifest_sha256':sha(OUT/'data/manifest.json'),
 'manual_notes_sha256':sha(OUT/'visual_review/manual_notes_backend_data.json'),
 'requests':12,'cells':72,'accepted_patches_actually_viewed':accepted_count,
 'images_actually_viewed':sum(len(r['image_sources']) for r in records),
 'credible_EDIT_summary':{'YES':0,'UNCERTAIN':6,'NA_NO_EDIT':6},
 'records':records}
p=OUT/'visual_review/review_backend_data.json'
assert not p.exists(),'No silent overwrite'
p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'path':str(p),'sha256':sha(p),'requests':12,'cells':72,'accepted_patch_images':accepted_count,'images':result['images_actually_viewed']}))
