"""Bounded saved-review consistency summary. No images judged, rendered, or scored here."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json

OUT=Path('/mnt/e/MeshFlow/native_t1/runs/chair_control_feedback_auto_v1')
VIS=OUT/'visuals'
DEST=VIS/'reviews'
def read(p):
    return json.loads(Path(p).read_text())
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):
    return dict(path=str(p),sha256=sha(p))
def rid(short,n):
    matches=[r['id'] for r in dev if r['uid'].startswith(short) and r['noise_index']==n]
    assert len(matches)==1
    return matches[0]
dev=read(OUT/'manifests/DEV.json')['rows']
ids={r['id'] for r in dev}
assert len(dev)==len(ids)==16
summaries=[]
for family in ['TRAJECTORY_REPLAY','TRAJECTORY_REPLAY_CAPPED']:
    for step in [500,1000]:
        candidate=f'explore_{family}_{step}'
        matched=f'explore_REPLAY_FM_CONTROL_{step}'
        p=DEST/(candidate+'_vs_'+matched+'.json')
        d=read(p)
        rows=d['rows']
        assert d['candidate_label']==candidate and d['matched_label']==matched
        assert d['scope_status']=='COMPLETE' and len(rows)==16
        assert {r['request_id'] for r in rows}==ids
        for r in rows:
            assert r['actually_viewed'] and r['candidate_label']==candidate and r['matched_label']==matched
            assert r['scope_status']=='COMPLETE'
            assert len(r['mesh_views'])==5 and set(r['feedback_views'])=={'S','I','E'}
            assert r['human_usability']=='NOT_RUN'
        reps=list((VIS/'representatives').glob(candidate+'__vs__'+matched+'__*/selection_manifest.json'))
        assert len(reps)==1
        rp=reps[0]
        rep=read(rp)
        assert rep['status']=='COMPLETE' and rep['candidate_label']==candidate and rep['matched_label']==matched
        assert len(rep['all16_rows'])==16 and {r['request_id'] for r in rep['all16_rows']}==ids
        assert rep['identity']['visual_review']['sha256']==sha(p)
        byid={r['request_id']:r for r in rows}
        pairbyid={r['request_id']:r for r in rep['all16_rows']}
        selected=rep['selections']['metric_improvement_but_reviewed_visual_deterioration']['request_ids']
        counters=[]
        for request_id in selected:
            v=byid[request_id]
            x=pairbyid[request_id]
            assert v['judgment']=='WORSE'
            tests=dict(S_increase=x['S_numeric_pair'] and x['S_delta_lower']>0,
                S_guard_and_I_reduction=x['S_numeric_pair'] and x['S_delta_lower']>=-.005 and x['I_numeric_pair'] and x['I_delta_upper']<0,
                S_guard_and_E_increase=x['S_numeric_pair'] and x['S_delta_lower']>=-.005 and x['E_numeric_pair'] and x['E_delta_lower']>0)
            assert any(tests.values())
            counters.append(dict(request_id=request_id,positive_criteria=[k for k,v in tests.items() if v],
                S_delta=x['S_delta'],I_delta_upper=x['I_delta_upper'],E_delta_lower=x['E_delta_lower'],
                visual_judgment=v['judgment'],observation=v.get('geometry_vs_teacher',v.get('geometry_observation'))))
        summaries.append(dict(candidate_label=candidate,matched_label=matched,step=step,status=d['status'],
            scope_status=d['scope_status'],request_count=16,human_usability=d['human_usability'],
            judgment_counts=dict(Counter(r['judgment'] for r in rows)),
            body_regressions_vs_teacher=[r['request_id'] for r in rows if r.get('new_obvious_main_body_collapse_vs_matched')],
            body_regressions_vs_START=[r['request_id'] for r in rows if r.get('new_obvious_main_body_collapse_vs_START')],
            body_regressions_vs_TR=[r['request_id'] for r in rows if r.get('new_obvious_main_body_collapse_vs_TR')],
            metric_positive_visual_negative=counters,review_source=ref(p),representative_source=ref(rp)))
galleries=[]
for step in [500,1000]:
    state=read(VIS/'h3_workflow'/f'step{step}.json')
    p=Path(state['gallery'])/'manifest.json'
    m=read(p)
    labels=['START']+[f'explore_{x}_{step}' for x in ['FM_CONTROL','REPLAY_FM_CONTROL','TRAJECTORY_REPLAY','TRAJECTORY_REPLAY_CAPPED']]
    assert m['status']=='COMPLETE' and m['labels']==labels and m['requests']==16 and m['parents']==8
    assert set(m['identity']['availability'])=={x+'|'+l for x in ids for l in labels}
    assert all(m['identity']['availability'].values())
    assert m['all_geometry_unchanged'] and m['all_vertices_inside']
    assert m['unique_cached_mesh_panels']==480 and m['overlay_panels']==240
    assert len(m['parent_pages'])==16
    for page in m['parent_pages']:
        assert page['panels_uncropped'] and Path(page['path']).is_file()
        feedback=Path(page['path']).stem.endswith('_feedback')
        expected={(r,c) for r in range(6 if feedback else 10) for c in range(5 if feedback else 7)}
        assert {(b['row'],b['column']) for b in page['panel_boxes']}==expected
    galleries.append(dict(step=step,status='PASS',source=ref(p),columns=['C']+labels+['GT: C + free'],
        full_DEV_requests=16,parents=8,all_original_and_capped_arms_visible=True,
        mesh_layout_slots=560,unique_mesh_panels=480,overlay_panels=240,
        four_views_plus_wire_per_noise=True,three_feedback_overlays_per_noise=True,
        source_receipts_geometry_unchanged=True,source_receipts_all_vertices_inside=True,
        scope='Manifest/layout/availability consistency only; no new rendering or image re-review. Existing source-image hashes are retained by full reviews.'))
transitions=[
dict(request_id=rid('341b',1),family='TRAJECTORY_REPLAY_CAPPED',
    step500='Severe lower/front body loss versus teacher, START and original TR.',
    step1000='Recovers a coherent box chair close to teacher and START; small back-corner cracks remain.',
    interpretation='The cap500 noise1 body failure is absent at cap1000; do not carry that specific failure forward.'),
dict(request_id=rid('bdd',1),family='TRAJECTORY_REPLAY_CAPPED',
    step500='Deep hanging under-seat triangular sheet, nearly reaching the floor.',
    step1000='No deep dangling sheet visible; seat more continuous than teacher, but back slits remain.',
    interpretation='This specific hanging-sheet defect did not persist to the final cap output.'),
dict(request_id=rid('86ec',0),family='TRAJECTORY_REPLAY',
    step500='Ragged open crown and long diagonal back gap versus teacher.',
    step1000='Mostly present crown with small holes; complete chair, with seat/leg tradeoffs.',
    interpretation='Partial recovery of the large back/crown defect; no claim that every local defect is cured.'),
dict(request_id=rid('f596',0),family='TRAJECTORY_REPLAY_CAPPED',
    step500='Exposed blue seat, rough crown; no corresponding raised side assembly or nearly floor-length hanging triangle in the saved review.',
    step1000='New raised side/arm sheets and a long downward triangle nearly to the floor; worse than original TR.',
    interpretation='A final-endpoint local regression; do not substitute the more favorable cap500 appearance.')]
result=dict(schema='chair_control_feedback_final_visual_summary_v1',status='PASS_CONSISTENCY',
    created_utc=datetime.now(timezone.utc).isoformat(),scientific_candidate_success='NONE_BY_VISUAL_GATE',
    scope='Read-only synthesis of four complete candidate AI reviews, four current representative manifests, two full combined-gallery manifests and DEV registration. No additional image judgments.',
    review_summaries=summaries,gallery_coverage=galleries,defect_transitions=transitions,
    worst_body_summary=[
        'Original TR500: 341b noise0 regresses against teacher and START.',
        'Original TR1000: 341b noise1 regresses against teacher and START; noise0 is shared continued-training deterioration against START, not uniquely TR.',
        'Cap500: 341b noise1 regresses against teacher, START and original TR.',
        'Cap1000: 341b noise0 regresses against START; teacher already has major deterioration. No full-review row flags a new whole-body collapse versus matched teacher, but the START-constrained full visual gate remains FAIL.'],
    material_local_summary=[
        'Both final replay candidates: c1b noise1 fragmented upper back/crown versus teacher.',
        'Cap1000: f596 noise0 raised side sheets and long under-seat triangle; 86ec noise1 thicker apron/filled back with more C overlap; 534 noise0 new thin under-seat needle; bdd noise0 larger back cuts.',
        'Control-associated gains remain visible: round-back opening at 866e is shared by FM/teacher as well as replay; it is not replay-specific.'],
    representative_interpretation='Metric-positive counterexamples use the frozen selector: conservative S increase OR S delta lower >= -0.005 plus conservative I decrease/E increase; they are not all S improvements.',
    limitations=['AI review is unblinded; human usability NOT_RUN.',
        'WORSE local judgments differ from the narrow whole-main-body-collapse flag.',
        'S/I/E feedback is defined on C and can miss large free-only artifacts; marker density is not area-weighted evidence.',
        'Shared continuation failures must not be uniquely attributed to the extra replay/cap mechanism.',
        'No checkpoint is selected by this synthesis.'],
    source_script=ref(__file__),DEV_registration=ref(OUT/'manifests/DEV.json'),
    new_render_calls=0,new_score_calls=0,model_calls=0,CONFIRM_read=False)
j=DEST/'FINAL_VISUAL_SUMMARY.json'
j.write_text(json.dumps(result,indent=2)+'\n')
lines=[
'# Final DEV visual consistency summary','',
'All four candidate reviews are COMPLETE for the same 16 DEV requests (8 parents x 2 noises); all four full visual gates are FAIL. Human usability: NOT_RUN. This synthesis reads saved reviews and selections only.','',
'Candidate identities are original TRAJECTORY_REPLAY and TRAJECTORY_REPLAY_CAPPED at exactly 500 and 1000 updates. Each is compared with explore_REPLAY_FM_CONTROL at the matching update count; START and same-step original TR remain separate comparisons.','',
'| Candidate | Matched teacher | Full visual gate | Body regression vs teacher | Body regression vs START |',
'|---|---|---|---|---|']
for s in summaries:
    short=lambda items: ', '.join(x.split('__')[1][:6]+' '+x.split('__')[-1] for x in items) or 'None flagged'
    lines.append(f"| {s['candidate_label']} | {s['matched_label']} | {s['status']} | {short(s['body_regressions_vs_teacher'])} | {short(s['body_regressions_vs_START'])} |")
lines+=['','The cap1000 failure against START at 341b noise0 is shared with substantial teacher deterioration. It is not evidence that the cap alone caused that body failure. Conversely, a missing teacher-relative collapse flag does not make cap1000 pass the START constraint or establish usability.','',
'Important local failures at the final endpoint: cap f596 noise0 adds raised side sheets and a long downward triangle; cap 86ec noise1 becomes bulky and overlaps more known structure; cap 534 noise0 adds an under-seat needle; cap bdd noise0 has larger back cuts. Both replay candidates retain c1b noise1 back/crown fragmentation relative to teacher.','',
'Do not carry intermediate failures forward without checking the endpoint: cap341b noise1 recovers from its severe500 body failure at1000; capbdd noise1 no longer has the deep500 hanging sheet. OriginalTR86ec noise0 also partially recovers its large500 crown/back defect. These recoveries coexist with new endpoint defects and do not overturn the full gates.','',
'| Candidate | Metric-positive / visually WORSE selected requests |',
'|---|---|']
for s in summaries:
    values=', '.join(x['request_id'].split('__')[1][:6]+' '+x['request_id'].split('__')[-1] for x in s['metric_positive_visual_negative'])
    lines.append(f"| {s['candidate_label']} | {values or 'NA'} |")
lines+=['',
'At cap1000, bdd noise0 is both the largest nominal S gain (+0.1612875296 relative to teacher, or +16.1288 percentage points) and the retained visually WORSE counterexample because its back is more cut/fragmented. The full JSON records the frozen conservative positive criteria; some other selected counterexamples qualify through I/E improvement with S within its guard, not through S gain alone.','',
'Both combined seven-column galleries are complete: C / START / FM_CONTROL / REPLAY_FM_CONTROL / original TRAJECTORY_REPLAY / CAPPED / GT, at the exact same500 or1000 step. Every registered DEV request is present in every model column, with four shaded views plus wireframe and S/I/E overlays. Each gallery contains480 unique mesh renders,560 laid-out mesh slots (C/GT reused between noises), and240 feedback panels. No original arm is hidden by representative selection.','',
'Source image hashes remain in the four full reviews. This check verified manifest availability/layout and bound current review hashes into each representative manifest; it did not rerender, rescore, run a model, or read CONFIRM. Round-hole gains on866e are shared by controls and must not be described as unique to replay.','',
'Full sources, exact identities, hashes, counterexample deltas, and defect transitions: FINAL_VISUAL_SUMMARY.json.','']
md=DEST/'FINAL_VISUAL_SUMMARY.md'
md.write_text('\n'.join(lines))
print(json.dumps(dict(status='PASS_CONSISTENCY',json=ref(j),markdown=ref(md),source=ref(__file__)),indent=2))

