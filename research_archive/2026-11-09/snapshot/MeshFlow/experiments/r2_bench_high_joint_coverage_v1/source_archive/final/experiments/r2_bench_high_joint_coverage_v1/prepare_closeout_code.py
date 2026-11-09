"""Prepare bounded evidence checks from the previous fully executed audit."""
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'r2/pilots/bench_natural_n_recipe_v1/evaluation_closeout.py'
DST=ROOT/'r2/pilots/bench_high_joint_coverage_v1/evaluation_closeout.py'
PREFLIGHT='''def preflight():
    dest=OUT/'audits/pre_evaluation_audit.json'
    if dest.exists():
        result=E.read(dest);E.verify(result['files']);return result
    from . import report_results as R
    cache=E.read(OUT/'audits/B_cache_audit.json')
    contract=E.read(OUT/'audits/eval_cpu_contract.json')
    assert cache['status']==contract['status']=='PASS'
    assert contract['requests']==46 and contract['model_calls']==0
    assert contract['manifest_sha256']==E.sha(E.MANIFEST)
    expected=dict(requests=46,new_RAW=54,new_D0=54,generator_forwards=2700,
                  new_score_rows=162,reused_score_rows=114,total_score_rows=276)
    assert E.budget()==expected
    routes=collections.Counter()
    for label in E.LABELS:
        assert len(E.new_requests(label))=={'B':8,'W':46}[label]
        for q in E.requests():
            for stage in E.STAGES:
                cell=R.cell(q,label,stage)
                root=E.root_for(label,q)
                assert str(root) in cell['path']
                routes['REUSED' if root==E.OLD else 'NEW']+=1
    assert routes=={'REUSED':114,'NEW':162}
    assert R.groups()==('OLD','FOCUS','DEV','HOLDOUT','HIGH_TRAIN','HIGH_HOLDOUT','NEW_WIDE_TRAIN')
    gallery=E.read(OUT/'gallery/manifest.json')
    assert gallery['sheets']==138 and gallery['cells']==276
    denoiser=E.ROOT/'r2/src/meshflow_control/denoise_model.py'
    text=denoiser.read_text(encoding='utf-8')
    assert '1 <= len(raw) <= 800' in text and 'reshape(1, number, 9)' in text
    files={str(p):E.sha(p) for p in (Path(__file__),Path(E.__file__),Path(R.__file__),
        E.MANIFEST,OUT/'audits/B_cache_audit.json',OUT/'audits/eval_cpu_contract.json',
        OUT/'gallery/manifest.json',OUT/'visual_review/protocol.json',denoiser)}
    result=dict(status='PASS',checked_at_utc=utc(),budget=expected,routing_cells=dict(routes),
        new_requests_per_label={'B':8,'W':46},groups=list(R.groups()),
        cached_B38='Read-only original arrays, queries, scores and camera/sections',
        D0_shape_contract='1..800 source contract; not GPU quality certification',
        gallery_pages=138,expected_actual_visual_cells=276,files=files,
        model_calls=0,scoring_calls=0,tensor_loads=0,new_noise=0)
    E.immutable(dest,result);return result


'''
text=SRC.read_text(encoding='utf-8-sig')
start=text.index('def preflight():');end=text.index('class Evidence:')
text=text[:start]+PREFLIGHT+text[end:]
# Adapt only the old fixed evaluation population; training's 1500-step audit is unchanged.
for old,new in [('raw_complete[\'outputs\']==38','raw_complete[\'outputs\']==46'),
                ('==38-len(new)','==46-len(new)'),('reused_outputs=38-len(new)','reused_outputs=46-len(new)'),
                ('==84','==54'),('RAW=84,generator_forward=4200,D0=84,GEO=84','RAW=54,generator_forward=2700,D0=54,GEO=54'),
                ('All84 RAW','All54 RAW'),('(342,252,90)','(276,162,114)'),
                ('==342','==276'),("{'NEW_REGISTERED':252,'REUSED_READ_ONLY':90}","{'NEW_REGISTERED':162,'REUSED_READ_ONLY':114}"),
                ("gallery['sheets']==114","gallery['sheets']==138"),
                ('cells=342,new=252,reused=90','cells=276,new=162,reused=114'),
                ('gallery_pages=114,actual_AI_PROXY_observations=342','gallery_pages=138,actual_AI_PROXY_observations=276'),
                ('original_P0_cache_unchanged','original_B_cache_unchanged')]:
    if old not in text:raise ValueError('Missing '+old)
    text=text.replace(old,new)
compile(text,str(DST),'exec')
if DST.exists():
    if DST.read_text(encoding='utf-8')!=text:raise ValueError('Existing derived code changed')
else:DST.write_text(text,encoding='utf-8')
print('PREPARED evaluation_closeout.py; no scientific calls')
