"""Freeze source identity and finite experiment definition before model execution."""
from pathlib import Path
import collections
import difflib
import json
import shutil
from .common import *
from .ledger import LIMITS


def resolve(path,base=CHAIR):
    p=Path(path)
    return p if p.is_absolute() else Path(base)/p


def records():
    panels=read_json(CHAIR/'panels.json')
    return [dict(r,table=table) for table in ('A','B','C') for r in panels[table]]


def original_sources():
    roots=[ROOT/'native_t1', ROOT/'models', ROOT/'utils', ROOT/'experiments/mf_h1_local0']
    paths=set()
    for d in roots[:3]:
        paths.update(d.glob('*.py'))
    for name in ('control_mvp48','control_mvp48_budget_single','chair_domain_control_v1','chair_control_attribution_v1'):
        paths.update((roots[3]/name).glob('*.py'))
    paths.update((ROOT/'native_t1/losses').glob('*.py'))
    paths.add(ROOT/'configs/snet/base-120m-ot-v-chair.yaml')
    return paths


def prepare(out=OUT):
    out=Path(out);(out/'registration').mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():
        verify_identity(out);return read_json(out/'protocol.json')
    from .stream import prepare_training_plan
    from experiments.mf_h1_local0.control_mvp48.train_io import resource_snapshot
    ready=read_json(SOURCE/'DATA_READY.json')
    assert ready['object_count']==48 and ready['splits']==dict(train=32,validation=8,test=8)
    manifest=read_json(CHAIR/'checkpoint_manifest.json')
    start=next(r for r in manifest['checkpoints'] if r['path']==str(START))
    assert digest(START)==START_SHA==start['checkpoint_sha256']
    assert start['metadata']['full_recovery_state'] and start['metadata']['conditional_total_step']==4000
    assert start['metadata']['context_encoder']=='geo' and start['metadata']['loss_recipe']=='fm'
    binding=read_json(SOURCE/'registration/train_data_binding.json')
    paths=original_sources()|{START,CHAIR/'checkpoint_manifest.json',CHAIR/'protocol.json',CHAIR/'panels.json',
        CHAIR/'task_manifest.json',SOURCE/'DATA_READY.json',SOURCE/'registration/train_data_binding.json'}
    for rel,expected in binding['files'].items():
        p=resolve(rel,SOURCE)
        if digest(p)!=expected:raise ValueError('Frozen data binding changed: '+str(p))
        paths.add(p)
    panels=read_json(CHAIR/'panels.json')
    for rel,expected in panels['frozen_files'].items():
        p=resolve(rel)
        if digest(p)!=expected:raise ValueError('Frozen evaluation input changed: '+str(p))
        paths.add(p)
    oldledger=read_json(CHAIR/'evaluation/phase3000/Geo/ledger.json')
    assert oldledger['model_hashes']['checkpoint_sha256']==START_SHA
    for r in oldledger['rows']:
        if r.get('raw_path'):
            p=resolve(r['raw_path'])
            if digest(p)!=r['raw_sha256']:raise ValueError('Historical raw changed: '+str(p))
            paths.add(p)
        if r.get('score_path'):
            p=resolve(r['score_path'])
            if digest(p)!=r['score_sha256']:raise ValueError('Historical score changed')
            paths.add(p)
    paths.add(CHAIR/'evaluation/phase3000/Geo/ledger.json')
    plan=prepare_training_plan(SOURCE,CHAIR,out,seed=20261004)
    rows=records();assert len(rows)==144
    diagnostic_parents={split:sorted({r['uid'] for r in panels['B'] if r['split']==split})[:2] for split in ('train','validation')}
    diagnostic_cases=[r['case_id'] for r in panels['B'] if r['uid'] in sum(diagnostic_parents.values(),[]) and r['alpha'] in (.9,1.1)]
    assert len(diagnostic_cases)==16
    protocol=dict(schema='NATIVE_CHAIR_HYBRID_COUPLING_PILOT_V1',source=str(SOURCE),source_chair=str(CHAIR),
        source_checkpoint=start, arms={'OT_BASE':dict(lambda_variance=0.),'OT_HYBRID':dict(lambda_variance=.25)},
        pilot_start=0,pilot_end=1000,conditional_start=4000,conditional_end=5000,training_seed=20261004,
        budget=LIMITS,overall_forward_limit=40204,unique_generation_request_limit=432,additional_repeat_limit=2,repeats_planned=0,
        effective_batch=8,microbatch=1,optimizer=dict(name='AdamW',lr=1e-5,betas=[.9,.95],weight_decay=0.,clip=1.),
        scheduler=None,ema=None,model_mode='eval with all parameters trainable, inherited original convention',
        precision='BF16 backbone forward; FP32 parameters, optimizer and integration; SDPA math; TF32 off; strict deterministic',
        FM_denominator='9*sum(valid_free_face_count over all 8 microbatches)',loss='original masked FM only',
        coupling='original pre-scale nested OT(Z,Y/2), inverse face/corner map Y to original Z slots, then Z_lambda=sqrt(1-lambda)*Z+sqrt(lambda)*epsilon; lambda0 exact branch',
        training_plan=plan, task_manifest_sha256=digest(CHAIR/'task_manifest.json'),source_panels_sha256=digest(CHAIR/'panels.json'),
        panel_counts={t:dict(requests=sum(r['table']==t for r in rows),ready=sum(r['table']==t and r['status']=='READY' for r in rows)) for t in ('A','B','C')},
        diagnostics=dict(parents=diagnostic_parents,case_ids=diagnostic_cases,curves_per_checkpoint=8,steps=50,
                         fp32_parents={k:v[0] for k,v in diagnostic_parents.items()},fp32_noise=0,fp32_steps=[0,25,49],
                         definition='finite-edit symmetric algebraic decomposition; not unique causal attribution or infinitesimal Jacobian'),
        main_endpoint=dict(table='A',split='validation',ratio=.2,alpha=1.,parents=8,noises=2,
                           metric='fixed_query_MSE=mean((distance(frozen_coverage_points,free_triangle_surface)/L)^2)',
                           aggregation='mean two noise values within each parent, then equal parent mean',
                           comparison='OT_HYBRID minus OT_BASE',screen='mean paired delta<0 and median parent paired delta<0 and >=6/8 parents improve'),
        matrix='E[input_alpha,target_beta]=sqrt(mean(d(Q_beta,F_alpha)^2))/L0; advantage fixed target beta COLUMN',
        width='q95-q05 of equal triangle corner X; absolute slope error per curve BEFORE averaging',
        sampler='original Euler50 FP32 state/t/dt, BF16 forward, clean C clamped every accepted state',
        cache_policy='START reuses strictly verified historical raw except 16 registered +/- diagnostic cases replayed for actual velocities; saved budget not reused for new experiments',
        checkpoint_save='full recoverable latest each200; fixed pilot1000 endpoint; pilot0 identity reference only',
        gpu_workers=1,gpu_lock=GPU_LOCK,cpu_threads=2,
        stopping='fixed budget or affected-stage identity/implementation/nonfinite failure; no automatic replay, extra arms or tuning',
        evidence_limits='one paired training stream; previously observed validation/holdout; no significance or mechanism exclusivity claim',
        missing_nonessential_handoff=['Native_Chair_Handoff_20261004','latest_independent_audit/INDEPENDENT_REVIEW_zh.md'],
        available_audits=['native_t1/analysis/chair_control_followup_v1','native_t1/runs/chair_control_attribution_v1'])
    hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)}
    freeze_json(out/'registration/source_hashes_before.json',hashes)
    freeze_json(out/'protocol.json',protocol)
    (out/'protocol.sha256').write_text(digest(out/'protocol.json')+'  protocol.json\n')
    freeze_json(out/'registration/evaluation_records.json',rows)
    freeze_json(out/'registration/start_identity.json',start)
    atomic_json(out/'registration/resources_initial.json',resource_snapshot())
    atomic_json(out/'registration/dispatch_plan.json',dict(gpu_workers=1,cpu_threads_per_process=2,
        reason='Explicit pilot cap is one GPU worker; CPU source/test/scoring tasks may run independently. No synthetic GPU probe.',
        headroom='max(2 GiB, 10% total VRAM) plus resource admission before each GPU worker'))
    return protocol


def verify_identity(out=OUT):
    out=Path(out)
    before=read_json(out/'registration/source_hashes_before.json')
    changed=[rel for rel,h in before.items() if not (ROOT/rel).exists() or digest(ROOT/rel)!=h]
    if changed:raise ValueError('Protected experiment dependency changed: '+str(changed))
    if (out/'protocol.sha256').read_text().split()[0]!=digest(out/'protocol.json'):
        raise ValueError('Frozen protocol changed')
    implementation=out/'registration/implementation_hashes.json'
    if implementation.exists():
        registered=read_json(implementation)
        actual={str(p.relative_to(ROOT)):digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
        if actual!=registered:raise ValueError('Frozen pilot implementation changed; no silent resume')
    return dict(status='PASS',files=len(before))


def code_snapshot(out=OUT,freeze=False):
    out=Path(out);files=sorted(Path(__file__).parent.glob('*.py'))
    hashes={str(p.relative_to(ROOT)):digest(p) for p in files}
    if freeze:freeze_json(out/'registration/implementation_hashes.json',hashes)
    chunks=[]
    for p in files:
        chunks.extend(difflib.unified_diff([],p.read_text().splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+str(p.relative_to(ROOT))))
    (out/'code.diff').write_text(''.join(chunks))
    return hashes


if __name__=='__main__':
    p=prepare();print(json.dumps(dict(status='PREPARED',protocol_sha256=json_hash(p),panels=p['panel_counts'])))
