"""Bounded table inspection and grid repair; preserves initial partial grid."""
from .run import *

def main():
    r=read(OUT/'render_plan.json');trainids={v['parent_id'] for v in r['conditions'].values() if '__V2' in v['condition_id']}
    parents=[p for p in r['parents'] if p['parent_id'] in trainids]
    assert len(parents)==25
    event('GRID_CORRECTION',reason='Historical parent eligibility strings differ by production batch. Initial grid contained only first6; membership now comes from frozen TRAIN condition IDs. Original partial grid preserved.',workers=1,resources=resources())
    grid=render_one(('parent_grid','accepted_targets_all25',parents))
    jobs=[('held','train_named_assemblies',[v for v in r['conditions'].values() if v['parent_id'] in trainids and v['part_type']=='PART_ASSEMBLY'])]
    pages=[grid]+[render_one(j) for j in jobs]
    old=read(OUT/'gallery_manifest.json')
    kept=[p for p in old['pages'] if p['kind']!='parent_grid']
    save(OUT/'gallery_manifest_v2.json',dict(pages=pages+kept,superseded='Initial partial6-parent grid is not full25 evidence'))
    html='<meta charset="utf-8"><title>Bench relation coverage</title><h1>Bench relation coverage</h1><p>Source data only; blue C / gray Y. Same-frame comparisons. See visual_review.json for actual AI_PROXY notes.</p>'
    for p in pages+kept:
        n=Path(p['path']).name;html+=f'<h2>{n}</h2><a href="gallery/{n}"><img style="width:100%;max-width:1400px" src="gallery/{n}"></a>'
    put(OUT/'review.html',html.encode())
    rows=list(csv.DictReader((OUT/'condition_descriptors.csv').open(encoding='utf-8-sig')))
    for part in ('SEAT_SURFACE','BACK_SURFACE','LEG_SUPPORT','PART_ASSEMBLY'):
        rr=[d for d in rows if d['split']=='TRAIN' and d['part_type']==part]
        print(part)
        for d in rr:
            print(d['parent_id'][:6],d['K'],round(float(d['C_centroid_y']),3),round(float(d['C_extent_z']),3),round(float(d['C_horizontal_normal_area']),3),d['actual_training_samples'],d['extent_description'])
    diag=list(csv.DictReader((OUT/'diagnostic_conditions.csv').open(encoding='utf-8-sig')))
    print('DIAGNOSTICS')
    for d in diag:print(d['parent_id'][:6],d['K'],d['C_centroid_y'],d['C_extent_x'],d['C_extent_y'],d['C_extent_z'],d['C_M_extent_ratio_z'],d['C_centroid_Y_relative_M_bbox'],d['actual_training_samples'])
    print('RELEVANT_HELD')
    for v in r['special_conditions']:
        if v.get('review_status')=='HOLD_ADAPTER':
            q=read(v['quality_file'],v.get('quality_sha256'));print(v['condition_id'],v['admission_reasons'],json.dumps(q)[:5500])
    save(OUT/'supplement_input_hashes.json',SOURCES)

if __name__=='__main__':main()
