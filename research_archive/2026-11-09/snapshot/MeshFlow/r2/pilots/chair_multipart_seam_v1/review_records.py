"""Prepare/finalize human-readable actual-view records; never rates images.

Templates deliberately contain null ratings and NOT_VIEWED markers. Finalize
only validates manually entered observations and hashes completed gallery files.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from . import gallery as G

FLAGS=('C_role_use','body_plausible_complete','visible_fragments_cracks',
       'obvious_layering_or_cover','support_thinning_or_loss','aperture_fill_observed')
RATINGS={'YES','NO','UNCERTAIN'}
CHANGES={'BETTER','MIXED','WORSE','NO_CLEAR_CHANGE','UNCERTAIN'}
GROUPS={'BACK_FULL':('BACK_FULL',None,16),
        'seat_part_01':('SEAT_FULL',('73','3f','85'),12),
        'seat_part_02':('SEAT_FULL',('8c','2d','610','608'),16)}

def selected_pages(group):
    part,prefixes,count=GROUPS[group]
    pages=[p for p in G.prepare()['pages'] if p['part_type']==part and
           (prefixes is None or p['parent_id'].startswith(prefixes))]
    assert len(pages)==count
    return part,pages

def make_template(group):
    part,pages=selected_pages(group)
    dest=G.OUT/'visual_review_templates'/(group+'.json')
    if dest.exists():return dest
    records=[]
    for page in pages:
        ident={k:page[k] for k in ('request_id','condition_id','parent_id','noise_index','part_type','split','K','N')}
        stages=[dict(model=c['model'],stage=c['stage'],**{key:None for key in FLAGS},
                     topology_by_image='UNKNOWN',notes='') for c in page['cells']]
        comparison=[dict(stage=stage,change=None,positive_observations='',costs_or_uncertainties='')
                    for stage in ('RAW','BASE_POST','UNIFIED_FINAL')]
        processing=[dict(model=model,RAW_to_FINAL_change=None,observations='')
                    for model in ('S2000','MULTIPART_S1500')]
        records.append(dict(**ident,image_path=str(G.OUT/'gallery'/G.KIND/(page['page_id']+'.png')),
                            image_sha256=None,actual_view_status='NOT_VIEWED',viewed_at_utc=None,
                            evaluator='AI_PROXY',nonblind=True,stages=stages,
                            old_new_comparisons=comparison,within_model_processing=processing,
                            request_notes=''))
    value=dict(schema='chair_multipart_actual_nonblind_review_v1',part_type=part,review_group=group,
               gallery_manifest=str(G.OUT/'gallery_manifests'/(G.KIND+'.json')),
               gallery_manifest_sha256=G.C.sha(G.OUT/'gallery_manifests'/(G.KIND+'.json')),
               status='TEMPLATE_NOT_REVIEWED',requests=records,
               policy='Inspect all six cells, both complete-object views and fixed finite sections. Do not infer absence of fine defects from an unclear view. No source style requirement or undeclared empty-hole constraint. Exact connectivity/intersections remain UNKNOWN from images. A finite section endpoint is not proof of a physical crack.',
               flags=list(FLAGS),flag_values=sorted(RATINGS),comparison_values=sorted(CHANGES),
               model_calls=0,scoring_calls=0)
    G.C.write(dest,value);return dest

def finalize(group,source):
    source=Path(source).resolve();review=G.C.read(source);plan=G.prepare()
    part,expected=selected_pages(group)
    pages={p['request_id']:p for p in expected};records=review['requests']
    assert review['schema']=='chair_multipart_actual_nonblind_review_v1' and review['part_type']==part and review['review_group']==group
    assert len(records)==len(expected) and {r['request_id'] for r in records}==set(pages)
    manifest=G.OUT/'gallery_manifests'/(G.KIND+'.json')
    assert review['gallery_manifest_sha256']==G.C.sha(manifest)
    folder=G.OUT/'gallery'/G.KIND;completion=G.C.read(folder/'completion.json')
    assert completion['requests']==44 and completion['status']=='RENDER_COMPLETE_PENDING_ACTUAL_REVIEW'
    images={im['request_id']:im for im in completion['images']}
    stage_rows=[];comparison_rows=[];processing_rows=[];view_rows=[]
    for r in records:
        q=pages[r['request_id']];im=images[r['request_id']]
        ident={k:q[k] for k in ('request_id','condition_id','parent_id','noise_index','part_type','split','K','N')}
        assert all(r[k]==v for k,v in ident.items())
        assert r['actual_view_status']=='ACTUALLY_VIEWED' and r['viewed_at_utc']
        assert r['evaluator']=='AI_PROXY' and r['nonblind'] is True
        assert G.C.resolve(r['image_path'])==G.C.resolve(im['path'])
        assert r['image_sha256']==im['image_sha256']==G.C.sha(im['path'])
        assert len(r['stages'])==6
        assert [(s['model'],s['stage']) for s in r['stages']]==[(c['model'],c['stage']) for c in q['cells']]
        for s in r['stages']:
            assert all(s[k] in RATINGS for k in FLAGS)
            assert s['topology_by_image']=='UNKNOWN' and s['notes'].strip()
            stage_rows.append(dict(**ident,evaluator='AI_PROXY',nonblind=True,
                                   image_path=r['image_path'],image_sha256=r['image_sha256'],
                                   viewed_at_utc=r['viewed_at_utc'],**s))
        assert [s['stage'] for s in r['old_new_comparisons']]==['RAW','BASE_POST','UNIFIED_FINAL']
        for s in r['old_new_comparisons']:
            assert s['change'] in CHANGES
            assert s['positive_observations'].strip() or s['costs_or_uncertainties'].strip()
            comparison_rows.append(dict(**ident,old_model='S2000',new_model='MULTIPART_S1500',**s))
        assert [s['model'] for s in r['within_model_processing']]==['S2000','MULTIPART_S1500']
        for s in r['within_model_processing']:
            assert s['RAW_to_FINAL_change'] in CHANGES and s['observations'].strip()
            processing_rows.append(dict(**ident,**s))
        view_rows.append(dict(**ident,evaluator='AI_PROXY',nonblind=True,viewed_at_utc=r['viewed_at_utc'],
                              image_path=r['image_path'],image_sha256=r['image_sha256'],
                              actual_view_status=r['actual_view_status'],request_notes=r['request_notes']))
    out=G.OUT/'visual_review'/group;out.mkdir(parents=True,exist_ok=True)
    if (out/'completion.json').exists():
        old=G.C.read(out/'completion.json');assert old['entered_review_sha256']==G.C.sha(source)
        return old
    files={}
    for name,rows in (('per_stage',stage_rows),('old_new_comparisons',comparison_rows),
                      ('within_model_processing',processing_rows),('actual_view_log',view_rows)):
        dest=out/(name+'.csv');assert not dest.exists(),'Do not overwrite an incomplete review export'
        G.C.csvwrite(dest,rows);files[str(dest)]=G.C.sha(dest)
    G.C.write(out/'review.json',dict(review,status='COMPLETE_ACTUAL_AI_PROXY_REVIEW'))
    files[str(out/'review.json')]=G.C.sha(out/'review.json')
    value=dict(status='COMPLETE_ACTUAL_AI_PROXY_REVIEW',part_type=part,review_group=group,requests=len(records),
               stage_observations=len(stage_rows),old_new_comparisons=len(comparison_rows),
               processing_comparisons=len(processing_rows),evaluator='AI_PROXY',nonblind=True,
               topology_by_image='UNKNOWN',entered_review_sha256=G.C.sha(source),
               gallery_manifest_sha256=G.C.sha(manifest),gallery_completion_sha256=G.C.sha(folder/'completion.json'),
               gallery_definition_sha256=G.C.sha(folder/'definition.json'),review_tool_sha256=G.C.sha(__file__),
               files=files,model_calls=0,new_scoring_calls=0,
               limitations='Observations of actual displayed views, not a human study, exact topology certification, or source-style reconstruction grade.')
    G.C.write(out/'completion.json',value);return value

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('template','finalize'))
    p.add_argument('--group',choices=tuple(GROUPS),required=True);p.add_argument('--input');a=p.parse_args()
    if a.action=='template':print(make_template(a.group))
    else:
        if not a.input:p.error('--input is required for finalize')
        print(json.dumps(finalize(a.group,a.input),ensure_ascii=False))
