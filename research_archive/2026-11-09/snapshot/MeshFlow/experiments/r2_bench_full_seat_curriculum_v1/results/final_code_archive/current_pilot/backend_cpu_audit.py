"""Preflight backend routing without model loading, generation or geometry scoring.
Pure metadata stubs test applicability; production scorer formulas are untouched.
"""
import ast, json, sys, types
from pathlib import Path
import numpy as np
from .common import OUT, ROOT, sha, read
from . import eval_runtime as R


def run():
    from . import evaluate as E, collect as C, gallery as G
    for name in ('post_d0.py','evaluate.py','collect.py','gallery.py','eval_runtime.py','panel.py'):
        ast.parse(Path(__file__).with_name(name).read_text())
    reg=read(OUT/'registration.json');qs=R.requests()
    assert reg['expected_new_raw']==reg['expected_new_d0']==72 and reg['sampling_steps']==50
    assert reg['occupancy_eval_head_forward_budget']==12
    assert {label:R.label_spec(label)['mode'] for label in R.LABELS}==dict(zip(R.LABELS,('PLAIN','PLAIN','SEAM')))
    barrier_status='ALL_COMPLETE_ALREADY'
    try:R.all_raw_barrier()
    except FileNotFoundError as exc:
        assert 'raw_complete_full.json' in str(exc)
        barrier_status='BLOCKS_MISSING_LATER_RAW_BEFORE_D0'
    # No preparation manifests written: capture planned metadata only.
    old=R.P.immutable;captured=[]
    R.P.immutable=lambda path,value:captured.append((str(path),value))
    try:score_plan=E.prepare();gallery_plan=G.prepare()
    finally:R.P.immutable=old
    assert score_plan['chain']=='BASE_POST -> HINGE_OPP015 -> ALPHA -> HOLE'
    assert sum(score_plan['outputs'].values())==72 and len(score_plan['stages'])==3
    assert len(gallery_plan['pages'])==72 and sum(len(p['cells']) for p in gallery_plan['pages'])==216
    qmap={q['request_id']:q for q in qs}
    for p in gallery_plan['pages']:
        q=qmap[p['request_id']]
        assert p['bounds']==q['gallery_bounds'] and p['depth_bounds']==q['gallery_depth_bounds']
        assert [c['model'] for c in p['cells']]==list(R.LABELS)
        for c in p['cells']:
            assert c['request_id']==q['request_id'] and c['stage']==p['sheet']
            assert str(OUT) in c['path'] and c['model'] in c['path']
    # Authenticate current COMMON products, without rescoring or rendering them.
    for q in qs:
        c=G.cell(q,R.LABELS[0],'RAW');G.G.authenticate(c)
    # Exercise the production applicability wrapper with a metadata-only scorer stub.
    fake=types.ModuleType('metadata_only_score_stub')
    exec("def role_reach(*a,**kw): return ({'status':'APPLICABLE','retained_seam_faces':0},{},[])\ndef score(*a,**kw): return role_reach()[0]",fake.__dict__)
    original=E.CPU.M;E.CPU.M=fake
    try:
        native=E.CPU.score_current(qs[0],{},None,'RAW',False)
        seam=E.CPU.score_current(qs[0],{},None,'RAW',True)
    finally:E.CPU.M=original
    assert native['status']=='NA_NOT_SEAM_REPRESENTATION'
    assert all(native[k] is None for k in E.CPU.E.INTERNAL_FIELDS)
    assert seam['status']=='APPLICABLE' and seam['retained_seam_faces']==0
    q=qs[0];source={k:q[k] for k in ('request_id','condition_id','parent_id','split','noise_index','N','K')}
    source.update(boundary_edges=123,retained_seam_faces=None,seam_bulk_area=None)
    row=C.decorate_with_status(source,R.LABELS[0],q,'RAW',native['status'])
    assert row['boundary_edges']==123 and row['C_boundary_edges']==q['boundary_edges'] and row['seam_applicability']==native['status']
    mean=C.aggregate([row],('label',),['retained_seam_faces','seam_bulk_area'])[0]
    assert mean['retained_seam_faces'] is None and mean['retained_seam_faces__valid_n']==0
    metric_hashes={p:h for p,h in score_plan['source_hashes'].items() if 'bench_full_seat_curriculum_v1' not in p}
    R.verify(metric_hashes)
    assert 'torch' not in sys.modules
    result=dict(status='PASS',planned_RAW=72,planned_D0=72,scoring_rows=216,gallery_pages=72,gallery_cells=216,
        RAW_barrier=barrier_status,COMMON_actual_RAW_authenticated=24,
        Native_seam_fields='NA_NOT_SEAM_REPRESENTATION; None values; valid_n=0',
        open_SEAM_all_null='Applicable zeros retained',whole_mesh_boundary_metric='Preserved separately from input C_boundary_edges',
        source_C_preservation='Original D0 geometry restores all fixed C coordinates and validates C vertex identity bijection; actual end outputs await authorized D0',
        inference_source_owner='No Y/source owner available in model input bundles',
        original_metric_source_hashes=metric_hashes,
        checked_backend_code={str(Path(__file__).with_name(n)):sha(Path(__file__).with_name(n)) for n in ('post_d0.py','evaluate.py','collect.py','gallery.py','eval_runtime.py','panel.py','backend_cpu_audit.py')},
        engineering_supplement=str(OUT/'engineering/collector_native_applicability_fix.json'),
        model_calls=0,geometry_scoring_calls=0,geometry_processing_calls=0,rendered_images=0,
        limitation='Metadata/array/source preflight only. Does not claim D0 or scoring already completed.')
    R.P.immutable(OUT/'backend_cpu_audit.json',result)
    return result

if __name__=='__main__':
    r=run();print({k:v for k,v in r.items() if k not in ('checked_backend_code','original_metric_source_hashes')})
