"""Freeze a stratified visual-review list and make contact sheets of saved PNGs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from . import bench_unified_gallery as R
C=R.C
OUT=R.ROOT/'visual_review/BENCH_UNIFIED_FIXED96'
PARTS=('BACK_SURFACE','SEAT_SURFACE','LEG_SUPPORT','PART_ASSEMBLY')
BAD='f5d370e9fe5b7452dbbc9440457e303e__BACK_SURFACE_1__n1'


def prepare():
    path=OUT/'selection.json'
    if path.exists():return C.read(path)
    m=R.prepare();chosen=[]
    for checkpoint in ('SEAM_S15000','SEAM_S21000'):
        for part in (*PARTS,'SEAT_SUPPORT'):
            for split in ('TRAIN','DEV'):
                eligible=[p for p in m['pages'] if (p['checkpoint'],p['part_type'],p['split'])==(checkpoint,part,split)]
                p=min(eligible,key=lambda p:p['request_id'])
                if part=='BACK_SURFACE' and split=='DEV':p=next(p for p in eligible if p['request_id']==BAD)
                chosen.append(dict(page_id=p['page_id'],checkpoint=checkpoint,part_type=part,split=split,
                                   request_id=p['request_id'],condition_id=p['condition_id'],parent_id=p['parent_id'],
                                   planned_review_set='CORE16' if part in PARTS else 'ADDITIONAL_SEAT_SUPPORT4',
                                   selection='Known support-loss diagnostic case, preselected' if p['request_id']==BAD else 'Lexicographically first request within model/part/split',
                                   image_path=str(R.OUT/(p['page_id']+'.png'))))
    result=dict(status='FROZEN_SELECTION_PENDING_ACTUAL_VIEW',rows=chosen,planned_pages=20,
                manifest_path=str(R.MANIFEST),manifest_sha256=C.sha(R.MANIFEST),
                numeric_metric_selection=False,reviewer='AI_PROXY',blind=False,
                policy='16 main strata plus4 SEAT_SUPPORT strata; known f5d/n1 explicitly targeted. Same requests across endpoints. No whole96 visual success estimate.',
                exact_topology_from_images='UNKNOWN')
    C.write(path,result);return result


def contact_sheets():
    from PIL import Image,ImageOps,ImageDraw,ImageFont
    sel=prepare();sheets=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    for checkpoint in ('SEAM_S15000','SEAM_S21000'):
        rows=[r for r in sel['rows'] if r['checkpoint']==checkpoint]
        for i in range(0,len(rows),4):
            subset=rows[i:i+4]
            if any(not Path(r['image_path']).exists() for r in subset):continue
            path=OUT/'contact_sheets'/f'{checkpoint}_{i//4+1:02d}.png'
            if path.exists():continue
            page=Image.new('RGB',(2360,1520),'white');draw=ImageDraw.Draw(page)
            sources=[]
            for j,r in enumerate(subset):
                image=Image.open(r['image_path']).convert('RGB');image.thumbnail((1160,700))
                x,y=(j%2)*1180,(j//2)*760
                draw.text((x+10,y+7),r['part_type']+' '+r['split']+' | '+r['request_id'][:12],font=font,fill='black')
                page.paste(image,(x+10,y+35));sources.append(dict(image_path=r['image_path'],sha256=C.sha(r['image_path'])))
            path.parent.mkdir(parents=True,exist_ok=True);page.save(path)
            receipt=dict(path=str(path),sha256=C.sha(path),sources=sources,selection_sha256=C.sha(OUT/'selection.json'),
                         actual_viewed=False,model_calls=0,new_geometry=0,new_scores=0)
            C.write(path.with_suffix('.json'),receipt);sheets.append(receipt)
    return dict(new_contact_sheets=len(sheets),sheets=sheets)


def summarize_actual():
    """Attach hashes to manually entered, actually viewed observations only."""
    sel=prepare();manifest=R.prepare();pages={p['page_id']:p for p in manifest['pages']}
    rows=[]
    for filename in ('actual_notes_agent.json','actual_notes_root.json'):
        p=OUT/filename
        if not p.exists():continue
        notes=C.read(p)
        for n in notes['rows']:
            page=pages[n['page_id']];image_path=R.OUT/(n['page_id']+'.png')
            receipt=C.read(image_path.with_suffix('.json'))
            assert C.sha(image_path)==receipt['image_sha256']
            rows.append(dict(**n,reviewer=notes['reviewer'],blind=False,actual_viewed=True,
                part_type=page['part_type'],split=page['split'],checkpoint=page['checkpoint'],
                request_id=page['request_id'],image_path=str(image_path),image_sha256=C.sha(image_path),
                image_receipt_sha256=C.sha(image_path.with_suffix('.json')),
                manifest_sha256=C.sha(R.MANIFEST),definition_sha256=C.sha(R.OUT/'definition.json'),
                source_notes_sha256=C.sha(p),exact_topology='UNKNOWN',watertightness='UNKNOWN',
                metrics_used_to_assign_visual_label=False,
                observed_columns=['OLD_BASE','OLD_HINGE015','GUARD_GEO','UNIFIED_FINAL'],
                observed_views=['azimuth_-60','azimuth_120','fixed_front_section','fixed_side_section']))
    planned={r['page_id'] for r in sel['rows']};done={r['page_id'] for r in rows}
    result=dict(status='PLANNED_SUBSET_REVIEWED' if planned<=done else 'PARTIAL_ACTUAL_REVIEW',
        reviewer_kind='AI_PROXY',blind=False,actual_pages=len(done),page_reviews=len(rows),
        planned_pages=len(planned),planned_pages_reviewed=len(planned&done),
        extra_pages=len(done-planned),actual_stage_views=4*len(rows),
        missing_planned_pages=sorted(planned-done),rows=rows,
        scope='Stratified visual sample, not a review or success estimate of all 96 requests.',
        limitations='Fixed central cuts omit off-plane geometry. Visible surface retention or small gaps do not certify legal shared edges, source intent, manifoldness, watertightness, or load-bearing support.',
        model_calls=0,new_geometry=0,new_scores=0)
    snapshot=OUT/'review_snapshots'/f'pages_{len(done):03d}_reviews_{len(rows):03d}'
    C.write(snapshot/'actual_review.json',result)
    lines=['# Bench unified rescue: actual visual review','',
           f"AI_PROXY, nonblind. Actually reviewed {len(done)} distinct pages ({len(planned&done)}/{len(planned)} planned plus {len(done-planned)} extra), four columns per page.",
           '',result['scope'],'',result['limitations'],'']
    for r in rows:
        lines += [f"## {r['page_id']}",'',f"Reviewer: {r['reviewer']}. Image SHA256: `{r['image_sha256']}`.",'',
                  r['overall'],'',r['sections'],'',f"GUARD vs OLD: {r['guard_vs_old']}",'',
                  f"FINAL vs GUARD: {r['final_vs_guard']}",'',r['remaining_concern'],'']
    (snapshot/'ACTUAL_REVIEW.md').write_text('\n'.join(lines),encoding='utf8')
    # This pointer is explicitly mutable; each underlying review snapshot is retained.
    (OUT/'latest_review.json').write_text(json.dumps(dict(snapshot=str(snapshot),
        actual_review_sha256=C.sha(snapshot/'actual_review.json'),actual_pages=len(done),
        page_reviews=len(rows),planned_pages_reviewed=len(planned&done)),indent=2),encoding='utf8')
    return {k:v for k,v in result.items() if k!='rows'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','contact','summarize-actual'));a=p.parse_args()
    result=prepare() if a.action=='prepare' else (contact_sheets() if a.action=='contact' else summarize_actual())
    print(json.dumps(result,ensure_ascii=False,indent=2))
