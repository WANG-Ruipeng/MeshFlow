"""Bind manually observed 36-output AI review to actual inspection metadata."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]
LABELS=('START','HIST_CAPPED500','HALF_CAPPED500')
VIEWS=['front_3q','back_3q','front','side','wire']
PARENTS=['866e7248434968051cc7750d0afca796','86ec9c796a6c1789e792598ed58dd1cc','bdd29e651e5f6fb2b079317292bdc5d4']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

evidence=read(HERE/'coupling_evidence.json')
notes=read(HERE/'coupling_notes.json')
assert len(evidence)==24 and len(notes)==12
assert len({(e['request_id'],e['overlay']) for e in evidence})==24
assert {n['uid'] for n in notes}==set(PARENTS)
sources={}
for e in evidence:
    assert e['original_panel_pixels_unchanged'] and not e['resizing']
    assert e['rows']==(['S','I','E'] if e['overlay'] else VIEWS)
    assert e['columns']==(list(LABELS) if e['overlay'] else ['Input C',*LABELS,'GT C+free'])
    for path,hash_value in ((e['gallery_manifest'],e['gallery_manifest_sha256']),
                            (e['camera_manifest'],e['camera_sha256']),
                            (e['evidence_page'],e['evidence_page_sha256'])):
        if path in sources:assert sources[path]==hash_value
        sources[path]=hash_value
for path,expected in sources.items():assert sha(path)==expected

# These are manual cross-label observations, not metric-derived decisions.
hist_vs_start={
 ('866e','OLD',0):'Cleaner more continuous round back and seat than START, without new missing body.',
 ('866e','OLD',1):'Less fragmented main body than START, but prominent seat fin and hanging strip remain.',
 ('866e','FRESH',2):'Cleaner round back and fuller apron than START, with a small raised seat fin.',
 ('866e','FRESH',3):'More continuous seat/back than START, but broken back-aperture faces persist.',
 ('86ec','OLD',0):'Broadly similar complete body and fine back cracks; no clear major change.',
 ('86ec','OLD',1):'Broadly similar complete body with irregular slats/top plate.',
 ('86ec','FRESH',2):'Isolated near-ground fragment seen in START is absent; main body retained.',
 ('86ec','FRESH',3):'New long underseat blade relative to START; main body still present.',
 ('bdd2','OLD',0):'Broadly similar complete body and fine cracks.',
 ('bdd2','OLD',1):'Definite new large floor-reaching underseat triangular sheet and more back slits than START.',
 ('bdd2','FRESH',2):'START raised seat-level fin absent; free-back slits persist.',
 ('bdd2','FRESH',3):'Broadly similar complete but slitted-back geometry.',
}
rows=[]
for note in notes:
    group=[e for e in evidence if e['uid']==note['uid'] and e['candidate']==note['candidate'] and e['panel']==note['panel']]
    assert len(group)==2
    main=next(e for e in group if not e['overlay'])
    overlay=next(e for e in group if e['overlay'])
    baseline_comparison=hist_vs_start[(note['uid'][:4],note['panel'],note['candidate'])]
    for label in LABELS:
        cstart='SELF: this is the START reference.' if label=='START' else baseline_comparison if label=='HIST_CAPPED500' else note['half_vs_start']
        chist='SELF: this is the HIST_CAPPED500 reference.' if label=='HIST_CAPPED500' else note['half_vs_hist'] if label=='HALF_CAPPED500' else 'Reference comparison (HIST relative to this START): '+baseline_comparison
        rows.append(dict(label=label,request_id=main['request_id'],uid=note['uid'],candidate=note['candidate'],panel=note['panel'],
            notes=note[label],comparison_to_START=cstart,comparison_to_HIST_CAPPED500=chist,
            uncertainty=note['uncertainty'],feedback_overlay_observation=note['overlay'],
            actually_viewed=True,view_scope=VIEWS,feedback_views=['S','I','E'],
            gallery_manifest=main['gallery_manifest'],gallery_manifest_sha256=main['gallery_manifest_sha256'],
            camera_manifest=main['camera_manifest'],camera_manifest_sha256=main['camera_sha256'],
            evidence_pages=[dict(path=e['evidence_page'],sha256=e['evidence_page_sha256'],
                inspection_png_sha256=e['png_sha256'],overlay=e['overlay']) for e in group]))
assert len(rows)==36 and len({(r['request_id'],r['label']) for r in rows})==36
assert Counter(r['label'] for r in rows)==Counter({x:12 for x in LABELS})
assert Counter((r['uid'],r['panel']) for r in rows)==Counter({(u,p):6 for u in PARENTS for p in ('OLD','FRESH')})
result=dict(status='PASS',reviewer='AI_VISUAL_REVIEW /root/coupling',reviewer_type='AI_VISUAL_REVIEW',
    reviewed_outputs=36,reviewed_requests=12,scope_status='COMPLETE_ASSIGNED_3_PARENTS',
    full_experiment_scope='PARTIAL_36_OF_96_TO_BE_MERGED_BY_ROOT',parent_ids=PARENTS,
    actual_inspection_images=24,human_review='NOT_RUN',
    status_meaning='PASS means assigned inspection completed with hash-bound evidence, not model promotion or universal quality improvement.',
    observations_origin='Manual actual image inspection; no numerical score used to manufacture a visual judgment.',
    summary='Most chair body regions persist. HALF removes clear hanging sheets in bdd OLD n1 and 86ec FRESH n3, but worsens local protrusions/fragmentation versus HIST in 866e FRESH n2/n3. No universal improvement claim.',
    significant_recoveries=[dict(uid=PARENTS[2],panel='OLD',candidate=1,description='HIST large floor-reaching underside triangle absent in HALF'),
                           dict(uid=PARENTS[1],panel='FRESH',candidate=3,description='HIST long hanging underside blade absent in HALF')],
    local_regression_cases=[dict(uid=PARENTS[0],panel='FRESH',candidate=2,description='More forward seat/back spikes and fragmented apron than HIST'),
                            dict(uid=PARENTS[0],panel='FRESH',candidate=3,description='Ragged seat rim and aperture-filling back patches compared with HIST'),
                            dict(uid=PARENTS[1],panel='OLD',candidate=1,description='New broad patch bridging lower slat openings; physical overlap uncertain')],
    rows=rows,evidence_file=str(HERE/'coupling_evidence.json'),evidence_sha256=sha(HERE/'coupling_evidence.json'),
    manual_notes_file=str(HERE/'coupling_notes.json'),manual_notes_sha256=sha(HERE/'coupling_notes.json'),
    source_page_checks=sources,script_sha256=sha(__file__),
    no_style_difference_automatic_failure=True,exact_intersections_or_watertightness_not_inferred=True,
    model_calls=0,scoring_calls=0,new_mesh_render_calls_for_review=0,utc=datetime.now(timezone.utc).isoformat())
write(HERE/'coupling.json',result)
render=[]
for panel in ('OLD','FRESH'):
    latest=read(OUT/'visuals'/('latest_'+panel+'.json'))
    assert sha(latest['manifest'])==latest['sha256']
    m=read(latest['manifest'])
    assert m['status']=='COMPLETE' and m['camera_status']=='FINAL_COMPLETE'
    assert m['all_vertices_inside'] and m['all_geometry_unchanged']
    render.append({key:m[key] for key in ('panel','status','camera_status','unique_mesh_panels','overlay_panels',
        'expected_overlay_panels','wall_seconds','mesh_render_seconds','all_vertices_inside','all_geometry_unchanged','GL_workers','CPU_threads')})
write(OUT/'visuals/preparation/render_completion.json',dict(status='PASS',panels=render,
    phases_serial=True,software_GL=True,model_calls=0,new_metric_calls=0,geometry_edits=0))
print(json.dumps(dict(status='PASS',reviewed_outputs=len(rows),file=str(HERE/'coupling.json'),
    sha256=sha(HERE/'coupling.json'),render=render)))
