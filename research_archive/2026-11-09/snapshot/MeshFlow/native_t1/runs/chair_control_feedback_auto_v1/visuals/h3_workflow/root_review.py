"""Persist manual root observations; merge only after both real reviews exist."""
import argparse,copy,datetime
from pathlib import Path
from experiments.mf_h1_local0.chair_control_feedback_auto_v1.common import *
from experiments.mf_h1_local0.chair_control_feedback_auto_v1.aggregation import compare_sets
from experiments.mf_h1_local0.chair_control_feedback_auto_v1.representatives import select

def bind(step):
    here=Path(__file__).parent;notes_path=here/f'root_notes_{step}.json';notes=read_json(notes_path)
    state=read_json(here/f'step{step}.json');assert state['status']=='COMPLETE'
    gallery=Path(state['gallery']);assert digest(gallery/'manifest.json')==state['manifest_sha256']
    requests=read_json(OUT/'manifests/DEV.json')['rows'];lookup={(r['uid'],r['noise_index']):r for r in requests}
    parents=sorted({r['uid'] for r in requests})[:4]
    assert len(notes['rows'])==8 and {(r['uid'],r['noise_index']) for r in notes['rows']}=={(u,n) for u in parents for n in (0,1)}
    candidate=f'explore_TRAJECTORY_REPLAY_CAPPED_{step}';matched=f'explore_REPLAY_FM_CONTROL_{step}';tr=f'explore_TRAJECTORY_REPLAY_{step}'
    images={};rows=[];abl=[]
    for note in notes['rows']:
        assert note['actually_viewed'] and all(note[k] in ('BETTER','WORSE','MIXED','SIMILAR') for k in ('judgment','versus_TR_judgment','versus_START_judgment'))
        paths=[gallery/(note['uid']+suffix+'.png') for suffix in ('','_feedback')]
        for p in paths:images[str(p)]=dict(path=str(p),sha256=digest(p),actually_viewed=True)
        row=dict(note,request_id=lookup[(note['uid'],note['noise_index'])]['id'],candidate_label=candidate,
            matched_label=matched,ablation_reference_label=tr,reference_label='START',review_type='AI_VISUAL_REVIEW',reviewer='AI agent /root',
            image_paths=[str(p) for p in paths],image_sha256={str(p):digest(p) for p in paths},
            mesh_views=['front_3q','back_3q','front','side','wire'],feedback_views=['S','I','E'],scope_status='COMPLETE',human_usability='NOT_RUN',
            geometry_observation=note['geometry_vs_teacher'],
            status='FAIL' if note['new_obvious_main_body_collapse_vs_matched'] or note['new_obvious_main_body_collapse_vs_START'] else 'PASS')
        rows.append(row)
        abl.append(dict(row,matched_label=tr,judgment=note['versus_TR_judgment'],geometry_observation=note['geometry_vs_TR'],
            new_obvious_main_body_collapse_vs_matched=note['new_obvious_main_body_collapse_vs_TR'],
            status='FAIL' if note['new_obvious_main_body_collapse_vs_TR'] or note['new_obvious_main_body_collapse_vs_START'] else 'PASS'))
    result=dict(status='FAIL' if any(r['status']=='FAIL' for r in rows) else 'INCOMPLETE',scope_status='PARTIAL_FULL_COHORT',
        assigned_scope_status='COMPLETE',review_type='AI_VISUAL_REVIEW',reviewer_type='AI_VISUAL_REVIEW',reviewer='AI agent /root',
        candidate_label=candidate,matched_label=matched,ablation_reference_label=tr,rows=rows,records=rows,ablation_records=abl,
        human_usability='NOT_RUN',source_images=list(images.values()),source_notes=dict(path=str(notes_path),sha256=digest(notes_path)),
        source_manifest=dict(path=str(gallery/'manifest.json'),sha256=digest(gallery/'manifest.json')),
        display_method=notes['display_method'],model_calls=0,scorer_calls=0,CONFIRM_read=False)
    path=OUT/'visuals/reviews'/f'{candidate}_vs_{matched}_root_partial.json';freeze_json(path,result)
    return path

def merge(step):
    candidate=f'explore_TRAJECTORY_REPLAY_CAPPED_{step}';matched=f'explore_REPLAY_FM_CONTROL_{step}';tr=f'explore_TRAJECTORY_REPLAY_{step}'
    folder=OUT/'visuals/reviews';paths=[folder/f'{candidate}_vs_{matched}_{who}_partial.json' for who in ('root','coupling')]
    parts=[read_json(p) for p in paths];ids={r['id'] for r in read_json(OUT/'manifests/DEV.json')['rows']}
    rows=sorted([r for d in parts for r in d['rows']],key=lambda r:r['request_id'])
    abl=sorted([r for d in parts for r in d['ablation_records']],key=lambda r:r['request_id'])
    assert len(rows)==len(abl)==16 and {r['request_id'] for r in rows}==ids
    for r in rows:
        assert r['actually_viewed'] and r['candidate_label']==candidate and r['matched_label']==matched
        for p,h in r['image_sha256'].items():assert digest(Path(p))==h
    full=dict(status='FAIL' if any(r['new_obvious_main_body_collapse_vs_matched'] or r['new_obvious_main_body_collapse_vs_START'] for r in rows) else 'PASS',
        scope_status='COMPLETE',reviewer_type='AI_VISUAL_REVIEW',review_type='AI_VISUAL_REVIEW',human_usability='NOT_RUN',
        candidate_label=candidate,matched_label=matched,ablation_reference_label=tr,rows=rows,records=rows,
        source_reviews={str(p):digest(p) for p in paths},source_manifest=parts[0]['source_manifest'],
        source_images=[im for d in parts for im in d['source_images']],
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        interpretation='Narrow no-new-obvious-body-collapse check against both START and specified comparator; local WORSE remains material. AI image review is not human usability or a manifoldness certificate.')
    target=folder/f'{candidate}_vs_{matched}.json';freeze_json(target,full)
    ar=dict(full,matched_label=tr,rows=abl,records=abl,status='FAIL' if any(r['new_obvious_main_body_collapse_vs_matched'] or r['new_obvious_main_body_collapse_vs_START'] for r in abl) else 'PASS')
    at=folder/f'{candidate}_vs_{tr}.json';freeze_json(at,ar)
    print('matched',compare_sets(OUT,candidate,matched,visual_review=target)['status'])
    print('coefficient_ablation',compare_sets(OUT,candidate,tr,visual_review=at)['status'])
    print('representatives',select(OUT,candidate,matched,target)['status'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['bind','merge']);p.add_argument('--step',type=int,choices=[500,1000],required=True);a=p.parse_args()
    if a.stage=='bind':print(bind(a.step))
    else:merge(a.step)
