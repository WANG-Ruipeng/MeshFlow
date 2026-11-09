"""Freeze identities, geometry probes and finite budgets before numerical gates."""
from pathlib import Path
import difflib
import shutil
import numpy as np
from .common import *
from .ledger import LIMITS
from .stream import SurfaceStream,prepare_training_plan,domain_seed

TASK_ATTACHMENT=Path('/mnt/c/Users/warpwang/.codex/attachments/773459b2-41c6-4939-b543-3bbf6dab875d/Pasted text.txt')

def geometry_fixtures(out):
    manifest=read_json(CHAIR/'task_manifest.json'); selected=[]
    for ratio in (.2,.4):
        tasks=sorted((r for r in manifest['tasks'] if r['role']=='train' and r['status']=='READY' and r['ratio']==ratio),
                     key=lambda r:(r['N'],r['K'],r['uid'],r['task_id']))
        for scale,index in [('ordinary',(len(tasks)-1)//2),('maximum',len(tasks)-1)]:
            selected.append(dict(task=tasks[index],scale=scale,rank=index,population=len(tasks),ratio=ratio))
    freeze_json(out/'registration/fixture_selection.json',selected)
    stream=SurfaceStream(SOURCE,CHAIR,out,'H_CONT',cache_dir=out/'registration/fixture_OT_cache')
    rows=[];times=np.asarray([0.,.5,.9,1.],np.float32)
    for i,item in enumerate(selected):
        task=item['task'];key=task['task_id'];rec=dict(sample_index=100000+i,surface_step=0,
            uid=task['uid'],task_id=key,alpha=1.,N=task['N'],K=task['K'])
        for field,domain in [('gaussian_seed','geometry_Z'),('epsilon_seed','geometry_epsilon'),
                             ('t_seed','geometry_time'),('permutation_seed','geometry_permutation')]:
            rec[field]=domain_seed(domain,key)
        sample=stream._sample(rec);known=sample['known_mask'];free=sample['valid_mask']&~known
        C=sample['context'][known].reshape(-1,3,3).copy()
        x0=sample['x0'][free].reshape(-1,3);x1=sample['x1'][free].reshape(-1,3)
        ids=np.asarray([0,(len(x0)-1)//3,2*(len(x0)-1)//3,len(x0)-1],np.int64)
        c=C[0].mean(0);edge=(C[0,0]+C[0,1])/np.float32(2)
        normal=np.cross(C[0,1]-C[0,0],C[0,2]-C[0,0]);norm=np.linalg.norm(normal)
        if norm==0:raise ValueError('First registered known face is degenerate; preserve failure, no replacement')
        normal=normal/norm
        probes=np.stack([c,edge,c+np.float32(.1)*normal,c+np.asarray([20,-20,20],np.float32)])
        queries=np.stack([np.concatenate([(np.float32(1)-t)*x0[ids]+t*x1[ids],probes]) for t in times])
        path=out/'registration/geometry_fixtures'/f'{i}_{item["scale"]}_{int(item["ratio"]*100)}.npz'
        arrays={k:np.asarray(sample[k]) for k in ('x0','x1','u','xt','context','t','y','known_mask','valid_mask','source_face_ids','corner_permutations')}
        arrays.update(C=C,queries=queries,times=times,free_query_indices=ids)
        filehash=save_npz(path,**arrays)
        rows.append(dict(fixture_id=f'{item["scale"]}_{int(item["ratio"]*100)}',npz_path=str(path),path=str(path),
            sha256=filehash,npz_sha256=filehash,uid=task['uid'],task_id=key,ratio=item['ratio'],scale=item['scale'],
            N=task['N'],K=task['K'],alpha=1.,selection_rank=item['rank'],selection_population=item['population'],
            record=rec,shared_input_sha256=sample['shared_input_sha256'],coupled_input_sha256=sample['coupled_input_sha256'],
            ordinary_query_count=7,extreme_query_indices=[7],reference_query_indices=[0,6],
            reference_known_indices=sorted(set([0,len(C)//2,len(C)-1])),
            known_source_ids=sample['source_face_ids'][known].tolist(),
            free_query_indices=ids.tolist(),normal_probe_status='nonzero source triangle normal, geometry-only probe'))
    fixture=dict(schema='surface_geometry_gate_fixtures_v1',fixtures=rows,times=times.tolist(),
        selection='For each ratio .2/.4, sort all original READY train tasks by (N,K,uid,task_id); lower median and final rank',
        query_rule='4 evenly spaced frozen HYBRID free-corner path indices, first known centroid, edge01 midpoint, centroid+0.1*unit normal; extreme centroid+[20,-20,20] separately',
        seed_domain='Independent geometry_* seeds derived from registered experiment/seed/task identity, alpha=1; no model-output selection',
        OT_counts=stream.counts,model_calls=0)
    freeze_json(out/'registration/geometry_fixtures.json',fixture)
    return fixture

def prepare(out=OUT):
    out=Path(out);(out/'registration').mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():
        verify_identity(out);return read_json(out/'protocol.json')
    from experiments.mf_h1_local0.control_mvp48.train_io import resource_snapshot
    from experiments.mf_h1_local0.chair_hybrid_coupling_v1.registration import original_sources
    start=next(r for r in read_json(HYBRID/'checkpoint_manifest.json')['endpoints'] if r['metadata']['arm']=='OT_HYBRID')
    m=start['metadata']
    assert start['checkpoint_sha256']==START_SHA==digest(START)
    assert m['pilot_step']==1000 and m['conditional_total_step']==5000 and m['lambda_variance']==.25
    assert m['full_recovery_state'] and m['context_encoder']=='geo' and m['loss_recipe']=='fm'
    assert read_json(HYBRID/'RUN_COMPLETE.json')['status']=='DONE'
    protected=original_sources()|set((ROOT/'experiments/mf_h1_local0/chair_hybrid_coupling_v1').glob('*.py'))
    oldhashes=read_json(HYBRID/'registration/source_hashes_before.json')
    for rel,h in oldhashes.items():
        p=ROOT/rel
        if digest(p)!=h:raise ValueError('Original asset differs: '+str(p))
        protected.add(p)
    for p in (HYBRID/'protocol.json',HYBRID/'checkpoint_manifest.json',HYBRID/'accounting.json',
              HYBRID/'CHAIR_HYBRID_COUPLING_REPORT.md',HYBRID/'RUN_AND_RESUME.md',START,START.with_suffix('.json')):
        protected.add(p)
    previous=read_json(HYBRID/'evaluation/OT_HYBRID/generated_rows.json')
    assert previous['status']=='COMPLETE' and previous['sampled']==138 and previous['not_applicable']==6
    for row in previous['rows']:
        if row['status']=='NOT_APPLICABLE':continue
        p=Path(row['raw_path']);assert digest(p)==row['raw_sha256'];protected.add(p)
    protected.add(HYBRID/'evaluation/OT_HYBRID/generated_rows.json')
    protected.update((HYBRID/'evaluation/OT_HYBRID/scoring').rglob('*.json'))
    protected.update((HYBRID/'evaluation/OT_HYBRID/scoring').rglob('*.npz'))
    plan=prepare_training_plan(SOURCE,CHAIR,out,seed=TRAIN_SEED)
    fixtures=geometry_fixtures(out)
    resources=resource_snapshot();gpu_fields=resources['gpu']['stdout'].strip().split(',')
    total=float(gpu_fields[-4]);used=float(gpu_fields[-3]);free=float(gpu_fields[-2])
    reserve=max(.15*total,2048.);process_budget=free-reserve
    assert process_budget>0
    panels=records();assert len(panels)==144
    trajectory=[]
    for split in ('train','validation'):
        uid=sorted({r['uid'] for r in panels if r['table']=='B' and r['split']==split})[0]
        trajectory.extend(r['case_id'] for r in panels if r['table']=='B' and r['uid']==uid and r['noise_index']==0 and r['alpha'] in (.9,1.1))
    protocol=dict(schema='NATIVE_CHAIR_FINITE_SURFACE_ATTENTION_V1',source=str(SOURCE),source_chair=str(CHAIR),
        source_checkpoint=start,arms={a:dict(lambda_variance=.25,relation=a if a!='H_CONT' else None,
            extra_parameters=0 if a=='H_CONT' else 104496) for a in ARMS},
        surface_start=0,surface_end=1000,conditional_start=5000,conditional_end=6000,
        training_seed=TRAIN_SEED,relation_initialization_seed=INIT_SEED,training_plan=plan,
        effective_batch=8,microbatch=1,alpha='4 exactly1 and 4 independent Uniform[.9,1.1] per batch',
        optimizer=dict(name='AdamW',lr=1e-5,betas=[.9,.95],weight_decay=0.,clip=1.,new_parameter_lr=1e-5,
                       recovery='old parameter states/steps restored exactly; new parameters zero AdamW moments and step0'),
        loss='original FM sum divided by 9*sum(valid free faces in all8 microbatches)',
        scheduler=None,ema=None,dropout='all modules eval',gradient_checkpointing=False,
        precision='BF16 original backbone; FP32 parameters, geometry, relations and optimizer; TF32 off; SDPA math; deterministic strict',
        model_output='Original actual velocity output; no change to parameterization or wrapper conversion',
        architecture=dict(L=12,H=12,D=768,layers_1based=[3,6,9,12],
            module_paths=[f'backbone.layers.{i}.attn' for i in (2,5,8,11)],
            network='Linear(11,32), SiLU, Linear(32,H+D); final weight/bias0; no outer zero gate',
            carrier='Original SDPA face message plus exact algebraic correction (softmax(l+b)-softmax(l))@V and pair-value correction; original out_proj weight, original gate; H_CONT remains original',
            free_known_only=True,normalization='all valid source faces',relations_shared_once_per_forward=True),
        geometry=dict(prior_sigma=1.,space='x-y / x-c, no t*y',bandwidth_squared='(1-t)^2+.1^2',
            components=['mu_minus_x_3','cov_xx_yy_zz_xy_xz_yz_6','log1p(max(-z,0))','t'],
            quadrature_points=192,subdivisions=8,quadrature='3 equal-weight (2/3,1/6,1/6) permutations in each of n^2 subtriangles',
            query_chunk=48,known_chunk=8,reference_points=12288,reference_convergence_points=49152,
            reference_convergence_limit=.001,ordinary_errors_strictly_below=.01,
            fixtures_path=str(out/'registration/geometry_fixtures.json'),fixtures_sha256=digest(out/'registration/geometry_fixtures.json')),
        real_gate=dict(zero_init_fp32_atol=1e-6,zero_init_fp32_rtol=1e-5,zero_init_bf16='must equal repeated original output; no tolerance relaxation',
            nonzero_permutation_fp32_atol=1e-6,nonzero_permutation_fp32_rtol=1e-5,
            schedule=dict(zero_init_forwards=16,cost_forwards=36,cost_backwards=24,nonzero_permutation_forwards=24,padding_forwards=12),
            planned_forwards=88,planned_backwards=24,zero_real_optimizer_updates=True),
        cost_gate=dict(task_ids=[r['task_id'] for r in fixtures['fixtures']],precision='BF16',
            warmup_forward_per_task_arm=1,measured_forward_backward_pairs_per_task_arm=2,
            synchronization=True,maximum_surface_to_H_CONT_median_ratio=3.,
            initial_total_mib=total,initial_used_mib=used,initial_free_mib=free,
            reserve_mib=reserve,available_process_budget_mib=process_budget,
            process_and_global_memory_checked=True),
        budget=LIMITS,overall_forward_limit=52996,unique_generation_request_limit=576,additional_repeat_limit=2,repeats_planned=0,
        panels={t:dict(requests=sum(r['table']==t for r in panels),ready=sum(r['table']==t and r['status']=='READY' for r in panels)) for t in ('A','B','C')},
        sampler='original Euler50; FP32 state/t/dt, BF16 forward, clean C all51states; no postprocessing or GT OT',
        main_endpoint=dict(table='A',split='validation',ratio=.2,alpha=1.,parents=8,noises=2,
            metric='mean((distance(original frozen512 GTfree queries,F)/L0)^2)',
            aggregation='two noises within parent, then eight equal parents',
            comparisons=['SURFACE_XY-MOMENT_XY','SURFACE_XY-H_CONT','MOMENT_XY-H_CONT','H_CONT-START','MOMENT_XY-START','SURFACE_XY-START'],
            screen='mean paired delta<0, paired delta median<0, improved>=6/8; complete denominator'),
        controls='same original full target-column 3x3; own F1 No-response/Post-scale; absolute slope errors percurve before means',
        trajectory_quadrature=dict(label='SURFACE_XY',case_ids=trajectory,steps=[0,25,49],
            query_indices='four evenly spaced valid free corner indices',reference_subsets='query0/last and known0/mid/last',no_extra_model_calls=True),
        cache_policy='START references strictly verified preceding OT_HYBRID raw and scores, no I/P replay',
        checkpoint_save='surface0 identity; full latest each200; fixed surface1000/total6000',
        gpu_workers=1,gpu_lock=GPU_LOCK,cpu_threads=2,
        stop='any numerical/cost gate failure prohibits all three formal training arms; no reduced nodes, data narrowing, retries or extra budget',
        optional_missing_assets=['Native_Surface_Attention_CPU_Pilot_20261004','Native_Chair_Handoff_20261004'],
        CPU_prototype_reuse='NOT_AVAILABLE; independently implement supplied complete math, no CPU prototype learning',
        evidence_limits='one paired continuation stream, previously observed validation/holdout; no causal uniqueness/statistical significance or watertight claim')
    freeze_json(out/'protocol.json',protocol)
    (out/'protocol.sha256').write_text(digest(out/'protocol.json')+'  protocol.json\n',encoding='utf8')
    freeze_json(out/'registration/evaluation_records.json',panels)
    freeze_json(out/'registration/start_identity.json',start)
    freeze_json(out/'registration/source_hashes_before.json',{p.relative_to(ROOT).as_posix():digest(p) for p in sorted(protected)})
    freeze_json(out/'registration/resources_initial.json',resources)
    freeze_json(out/'registration/dispatch_plan.json',dict(gpu_workers=1,cpu_threads=2,reason='Explicit one-GPU-worker task; independent CPU implementation/audits overlap',reserve_mib=reserve))
    if TASK_ATTACHMENT.exists():shutil.copyfile(TASK_ATTACHMENT,out/'registration/task_prompt.txt')
    return protocol

def code_snapshot(out=OUT):
    files=sorted(Path(__file__).parent.glob('*.py'))
    freeze_json(Path(out)/'registration/implementation_hashes.json',{p.relative_to(ROOT).as_posix():digest(p) for p in files})
    diff=[]
    for p in files:diff.extend(difflib.unified_diff([],p.read_text(encoding='utf8').splitlines(True),fromfile='/dev/null',tofile='b/'+p.relative_to(ROOT).as_posix()))
    (Path(out)/'code.diff').write_text(''.join(diff),encoding='utf8')

def verify_identity(out=OUT):
    out=Path(out);before=read_json(out/'registration/source_hashes_before.json')
    changed=[rel for rel,h in before.items() if not (ROOT/rel).exists() or digest(ROOT/rel)!=h]
    if changed:raise ValueError('Protected source changed: '+str(changed))
    assert digest(out/'protocol.json')==(out/'protocol.sha256').read_text().split()[0]
    path=out/'registration/implementation_hashes.json'
    if path.exists():
        expected=read_json(path)
        actual={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
        if expected!=actual:raise ValueError('Frozen implementation changed; inspect evidence, no silent continuation')
    for f in read_json(out/'registration/geometry_fixtures.json')['fixtures']:
        assert digest(f['npz_path'])==f['sha256']
    scorers=read_json(CHAIR/'protocol.json')['source_scorer_identity']
    assert all(digest(ROOT/p)==h for p,h in scorers.items()),'Original scoring dependency changed'
    return dict(status='PASS',protected_files=len(before),additional_historical_scoring_sources=len(scorers))

if __name__=='__main__':
    p=prepare();print('PROTOCOL_FROZEN',json_hash(p))
