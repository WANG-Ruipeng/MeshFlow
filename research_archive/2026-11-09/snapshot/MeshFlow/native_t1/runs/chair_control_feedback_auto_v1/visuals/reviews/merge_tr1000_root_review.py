"""Persist actual root image inspection, then merge independently inspected rows."""
from experiments.mf_h1_local0.chair_control_feedback_auto_v1.common import *
from experiments.mf_h1_local0.chair_control_feedback_auto_v1.aggregation import compare_sets
from experiments.mf_h1_local0.chair_control_feedback_auto_v1.representatives import select
import copy, datetime

dest=OUT/'visuals/reviews'
partial=dest/'explore_TRAJECTORY_REPLAY_1000_vs_explore_REPLAY_FM_CONTROL_1000_coupling_partial.json'
other=read_json(partial)
gallery=Path(other['source_manifest']['path']).parent
observations={
'33ec1e64a02ff718faaadcdc0221eb72':[
('BETTER','More coherent rounded back/arm rails and fewer large side/underarm diagonals than teacher. Chair retained; known base/seat mismatch remains.',False,False),
('WORSE','Compared with teacher, fragmented/open seat, lower needle/sliver, larger side panel and back spikes. Main chair still recognizable. New purple band on known seat; mostly green E misses free-region defects.',False,False)],
'341bcae1e7f366a233d93f57729688db':[
('MIXED','All continued models deteriorate markedly against START. FM and teacher have flat seat/back and fragmented undersupports; TR restores some box-side material but adds jagged front spikes and holes. Do not attribute this shared failure uniquely to TR.',False,True),
('WORSE','START, FM and teacher retain a coherent box chair. TR has pronounced front/side/support breakup, spikes, holes and compressed body. Known small foot bar remains, with green E that does not certify the free body.',True,True)],
'534b0e6e2c53a8e1a7cdfd3874ed3723':[
('WORSE','Global chair similar; TR known-leg complement becomes a thin separated sliver with a pronounced gap, while back is cleaner. Local junction regression, no whole-body collapse. Red E on seat and mostly red S persist.',False,False),
('MIXED','TR has fewer long back/leg cracks than teacher but seat spike, known-seat gap and known-leg mismatch persist. Local improvements and costs coexist.',False,False)],
'866e7248434968051cc7750d0afca796':[
('SIMILAR','All continued models recover the round back hole relative to START. Teacher and TR bodies are similar, with slight TR seat slit/back raggedness. Shared improvement is not specific to replay. Purple known-leg I band reduces locally.',False,False),
('MIXED','All continued models improve round-hole body relative to START. TR hole is larger and more round but ragged; an under-seat diagonal triangle shifts/grows. No new whole-body collapse.',False,False)]}
rows=[];images=[]
requests=read_json(OUT/'manifests/DEV.json')['rows']
for uid,obs in observations.items():
    paths=[gallery/(uid+'.png'),gallery/(uid+'_feedback.png')]
    for p in paths:images.append(dict(path=str(p),sha256=digest(p),actually_viewed=True))
    for noise,(judgment,note,collapse,collapse_start) in enumerate(obs):
        req=next(r for r in requests if r['uid']==uid and r['noise_index']==noise)
        row=copy.deepcopy(other['rows'][0])
        row.update(request_id=req['request_id'],uid=uid,noise_index=noise,candidate=noise,
            reviewer='AI agent /root',judgment=judgment,geometry_observation=note,
            condition_compatibility_observation='See geometry observation; known blue C is not inferred to be attached solely from preservation.',
            feedback_overlay_observation='All saved S/I/E overlays viewed; marker density is not a numerical score.',
            new_obvious_main_body_collapse_vs_matched=collapse,new_obvious_main_body_collapse_vs_START=collapse_start,
            severe_local_regression=collapse,status='FAIL' if collapse or collapse_start else 'PASS',
            versus_START_judgment='WORSE' if collapse_start else 'MIXED',
            image_paths=[str(p) for p in paths],image_sha256={str(p):digest(p) for p in paths})
        rows.append(row)
root=dict(status='FAIL',scope_status='PARTIAL_FULL_COHORT',reviewer_type='AI_VISUAL_REVIEW',
    review_type='AI_VISUAL_REVIEW',actually_viewed=True,rows=rows,records=rows,source_images=images,
    source_manifest=other['source_manifest'],display_method=other['scope']['display_method'],
    human_usability='NOT_RUN',candidate_label=other['candidate_label'],matched_label=other['matched_label'])
freeze_json(dest/'explore_TRAJECTORY_REPLAY_1000_vs_explore_REPLAY_FM_CONTROL_1000_root_partial.json',root)
allrows=sorted(rows+other['rows'],key=lambda r:r['request_id'])
assert len(allrows)==16 and len({r['request_id'] for r in allrows})==16
assert {r['request_id'] for r in allrows}=={r['request_id'] for r in requests}
full=dict(root,scope_status='COMPLETE',rows=allrows,records=allrows,
    source_images=images+other['source_images'],created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_reviews={str(partial):digest(partial)},
    gate=dict(status='FAIL',reason='341 noise1 has obvious body regression against matched teacher and START; 341 noise0 also regresses against START, shared with controls.'))
target=dest/'explore_TRAJECTORY_REPLAY_1000_vs_explore_REPLAY_FM_CONTROL_1000.json'
freeze_json(target,full)
for step in (500,1000):
    review=dest/f'explore_TRAJECTORY_REPLAY_{step}_vs_explore_REPLAY_FM_CONTROL_{step}.json'
    print(step,compare_sets(OUT,f'explore_TRAJECTORY_REPLAY_{step}',f'explore_REPLAY_FM_CONTROL_{step}',visual_review=review)['status'])
print(select(OUT,other['candidate_label'],other['matched_label'],target)['status'])
