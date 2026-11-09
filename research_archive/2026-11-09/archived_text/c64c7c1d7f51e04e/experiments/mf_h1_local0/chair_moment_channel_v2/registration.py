"""Freeze the independently authorized V2 experiment before real calls."""
from pathlib import Path
import difflib
import shutil
from .common import *
from .ledger import LIMITS

ATTACHMENT=Path('/mnt/c/Users/warpwang/.codex/attachments/9040b405-1bff-4291-90e1-49095b78a340/Pasted text.txt')

def prepare(out=OUT):
    out=Path(out);(out/'registration').mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():return read_json(out/'protocol.json')
    from .stream import prepare_training_plan
    from experiments.mf_h1_local0.control_mvp48.train_io import resource_snapshot
    start=next(r for r in read_json(HYBRID/'checkpoint_manifest.json')['endpoints'] if r['metadata']['arm']=='OT_HYBRID')
    m=start['metadata'];assert start['checkpoint_sha256']==START_SHA==digest(START)
    assert (m['pilot_step'],m['conditional_total_step'],m['lambda_variance'])==(1000,5000,.25)
    assert m['full_recovery_state'] and m['loss_recipe']=='fm' and m['context_encoder']=='geo'
    assert read_json(HYBRID/'RUN_COMPLETE.json')['status']=='DONE'
    # Retain all old audit files, raw and score evidence. Never copy old weights.
    protected={}
    for rel,h in read_json(OLD/'registration/source_hashes_before.json').items():
        assert digest(ROOT/rel)==h,rel
        protected[rel]=h
    for base in (OLD,R1,ROOT/'experiments/mf_h1_local0/chair_surface_attention_v1',ROOT/'experiments/mf_h1_local0/chair_surface_attention_v1r1'):
        for p in sorted(base.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:protected[p.relative_to(ROOT).as_posix()]=digest(p)
    freeze_json(out/'registration/source_hashes_before.json',protected)
    for name,source in [('geometry_fixtures.json',OLD/'registration/geometry_fixtures.json'),('old_test_manifest.json',R1/'precheck/test_manifest.json')]:
        shutil.copyfile(source,out/'registration'/name)
    fixtures=read_json(out/'registration/geometry_fixtures.json')
    for f in fixtures['fixtures']:assert digest(f['npz_path'])==f['sha256']
    rows=records();assert len(rows)==144 and sum(r['status']=='READY' for r in rows)==138
    replays=[]
    for ratio in (.2,.4):
        candidates=sorted((r for r in rows if r['table']=='A' and r['split']=='train' and r['status']=='READY' and r['ratio']==ratio and r['noise_index']==0 and r['alpha']==1),key=lambda r:(r['uid'],r['case_id']))
        replays.append(candidates[0])
    plan=prepare_training_plan(SOURCE,CHAIR,out,seed=TRAIN_SEED)
    resources=resource_snapshot();fields=resources['gpu']['stdout'].strip().split(',')
    total,used,free=map(float,fields[-4:-1]);reserve=max(2048.,.15*total)
    protocol=dict(schema='NATIVE_CHAIR_MOMENT_CHANNEL_LEARNING_V2',experiment='chair_moment_channel_v2',
        source=str(SOURCE),source_chair=str(CHAIR),source_checkpoint=start,
        arms={a:dict(lambda_variance=.25,extra_parameters=0 if a=='H_CONT' else 104496) for a in ARMS},
        labels=LABELS,seed=TRAIN_SEED,training_seed=TRAIN_SEED,relation_initialization_seed=INIT_SEED,
        training_plan=plan,updates_per_arm=1000,base_conditional_total_step=5000,moment_start=0,moment_end=1000,
        conditional_start=5000,conditional_end=6000,lambda_variance=.25,checkpoint_interval=200,
        effective_batch=8,microbatch=1,alpha='4 exactly1 and 4 Uniform[.9,1.1]; shared actual tensors/hashes',
        optimizer=dict(name='AdamW',lr=1e-5,betas=[.9,.95],weight_decay=0.,clip=1.,new_parameter_lr=1e-5),
        loss='Original FM sum divided by effective-batch valid free scalar count',scheduler=None,ema=None,
        gradient_checkpointing=False,dropout='original eval, all probabilities effectively zero',
        precision='BF16 forward; FP32 parameters/state/AdamW/relations; strict deterministic; TF32 off; SDPA math',
        architecture=dict(D=768,H=12,layers_1based=[3,6,9,12],new_parameters=104496,
            relation='11 -> 32 SiLU -> H+D; zero last layer; no extra gate; full valid-source softmax',
            carrier='Original SDPA plus exact difference correction; compact valid sources in original order',
            optimization='Retain previously verified low-dimensional value reordering; one compact-view repair; no new source-partition kernel',
            relation_once_per_forward=True,per_layer_MLP=True,cache='Only immutable compact C moments; never Geo hidden'),
        geometry=dict(prior_sigma=1.,bandwidth_squared='(1-t)^2+.1^2',
            covariance='sum((y-c)(y-c)^T)/12',space='c-x, never t*c-x',no_coordinate_sort=True,
            fixtures_path=str(out/'registration/geometry_fixtures.json'),fixtures_sha256=digest(out/'registration/geometry_fixtures.json')),
        real_gate=dict(normalized_e_inf_max=5e-5,normalized_e_rms_max=1e-5,
            old_element_atol=1e-6,old_element_rtol=1e-5,zero_both_precisions='bitwise H equality',
            same_shape_padding='bitwise valid outputs for zero/bounded/large finite values; pad input gradients zero',
            FP64='literal forward/gradient, six corners, face permutations, compact scatter',
            BF16_nonzero='descriptive only, not asserted strict equivariance',
            schedule=dict(zero=16,cost=32,moment_nonzero=40,H_transforms=16,padding_gradient=8,old_M_reference_cost=8),
            planned_forwards=120,planned_backwards=44,reserve_forwards=40,reserve_backwards=4,
            reserve_policy='Only explicit documented engineering recovery within total cap; no automatic extra probes',
            cost_order='alternate H/M each repeat and fixture; warm F+B then 3 measured synchronized F+B'),
        cost_gate=dict(maximum_moment_to_H_CONT_median_ratio=3.,reserve_mib=reserve,
            initial_total_mib=total,initial_used_mib=used,initial_free_mib=free,
            measured_pairs_per_task_arm=3,warmup_pairs_per_task_arm=1,
            old_reference='one warm forward and one F+B per fixture, descriptive same-run reference only'),
        budget=LIMITS,overall_forward_limit=36960,unique_generation_request_limit=432,
        initialization_replays=[r['case_id'] for r in replays],initialization_replay_limit=2,
        expected_sampling=278,sampling_hard_limit=416,expected_sampling_NFE=13900,
        panels={t:dict(requests=sum(r['table']==t for r in rows),ready=sum(r['table']==t and r['status']=='READY' for r in rows)) for t in ('A','B','C')},
        main_endpoint=dict(table='A',split='validation',ratio=.2,alpha=1.,parents=8,noises=2,
            metric='mean((distance(original frozen512 GTfree queries,F)/L0)^2)',
            aggregation='mean two noises within parent, then eight equal parents',
            comparisons=['MOMENT_XY_END-H_CONT_END','H_CONT_END-START','MOMENT_XY_END-START'],
            candidate='mean<0, paired median<0, improved>=6/8; no missing parent/noise'),
        controls='E[input alpha,target beta]; strict column choices/24; each noise/12; all3/8; one-input-all-targets/8; percurve absolute slope errors',
        quality='Both surface directions, bidirectional RMS, coverage,p95,span/median,Gamma,CF/FF,affected faces,angles,C/N/nonfinite/failure',
        sampler='Original Gaussian bytes, Euler50 BF16 forward FP32 state; C bitwise all51; no GT OT/guidance/postprocessing',
        cache_policy='Read-only verified START138 raw and scores; no I/P',gpu_workers=1,gpu_lock=GPU_LOCK,cpu_threads=2,
        stop='Only V2 implementation/numeric/cost/resource failures block training; no SURFACE integral gate',
        recovery='Engineering-only faults may be repaired within budget with original failure/diff/ledger; no scientific version stitching',
        evidence_limits='One continuation stream; previously observed validation/holdout; no significance or default promotion')
    freeze_json(out/'protocol.json',protocol)
    (out/'protocol.sha256').write_text(digest(out/'protocol.json')+'  protocol.json\n',encoding='utf8')
    freeze_json(out/'registration/evaluation_records.json',rows)
    freeze_json(out/'registration/evaluation_requests.json',[dict(r,checkpoint_label=l) for l in LABELS for r in rows])
    freeze_json(out/'registration/initialization_replays.json',replays)
    freeze_json(out/'registration/start_identity.json',start)
    freeze_json(out/'registration/resources_initial.json',resources)
    freeze_json(out/'registration/dispatch_plan.json',dict(gpu_workers=1,cpu_threads=2,reserve_mib=reserve,reason='Explicit V2 single GPU worker; independent CPU audits overlap'))
    shutil.copyfile(ATTACHMENT,out/'registration/task_prompt.txt')
    return protocol

def code_snapshot(out=OUT,name='implementation_hashes'):
    files=sorted(Path(__file__).parent.glob('*.py'))
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in files}
    freeze_json(Path(out)/'registration'/f'{name}.json',hashes)
    diff=[]
    for p in files:diff.extend(difflib.unified_diff([],p.read_text(encoding='utf8').splitlines(True),fromfile='/dev/null',tofile='b/'+p.relative_to(ROOT).as_posix()))
    (Path(out)/'code.diff').write_text(''.join(diff),encoding='utf8')
    return hashes

def verify_identity(out=OUT,code=False):
    out=Path(out);before=read_json(out/'registration/source_hashes_before.json')
    changed=[r for r,h in before.items() if not (ROOT/r).exists() or digest(ROOT/r)!=h]
    if changed:raise ValueError('Protected file changed: '+str(changed))
    assert digest(out/'protocol.json')==(out/'protocol.sha256').read_text().split()[0]
    if code:
        for rel,h in read_json(out/'registration/training_implementation_hashes.json').items():assert digest(ROOT/rel)==h,rel
    return dict(status='PASS',protected_files=len(before),changed=changed)

if __name__=='__main__':prepare()
