"""Freeze the user-authorized paired 1500-update comparison before model calls."""
from .common import *
from pathlib import Path


def main():
    if REGISTRATION.exists():
        from .training import checked
        for arm in ARMS:checked(arm)
        return read(REGISTRATION)
    old=read(OLD/'training_registration.json')
    manifest=OUT/'data/manifest.json'
    m=read(manifest)
    streamgate=OUT/'data/stream_gate.json';cpugate=OUT/'data/runtime_cpu_gate.json'
    sg,cg=read(streamgate),read(cpugate)
    if sg['status']!='PASS' or cg['status']!='PASS':raise RuntimeError('CPU preparation did not pass')
    if sha(SOURCE)!=SOURCE_IDENTITIES['source_checkpoint_sha256']:raise RuntimeError('Source checkpoint changed')
    receipt=read(SOURCE.with_suffix('.json'))
    for key,dest in [('generator_state_sha256','source_generator_sha256'),('optimizer_state_sha256','source_optimizer_sha256'),('rng_sha256','source_rng_sha256')]:
        if receipt[key]!=SOURCE_IDENTITIES[dest]:raise RuntimeError('Source matching state identity changed')
    verify(old['training_code_hashes'])
    from r2.pilots.bench_four_routes_v1.resources import snapshot
    resource=OUT/'resources/before_registration.json'
    if not resource.exists():write(resource,snapshot())
    sources={resolve(p) for p in old['training_code_hashes']}
    sources.update(Path(__file__).with_name(n) for n in ('__init__.py','common.py','register.py','training.py','data.py','prepare_data.py'))
    sources.add(ROOT/'r2/data_tools/bench_multilod_v1/sampler.py')
    sources.add(ROOT/'r2/pilots/bench_c_tessellation_v1/common.py')
    sources.add(ROOT/'r2/pilots/bench_four_routes_v1/resources.py')
    from .training import limits
    paths=dict(manifest=manifest,eval_manifest=resolve(old['eval_manifest']),source_checkpoint=SOURCE)
    reg=dict(schema='bench_multilod_training_registration_v1',status='FROZEN',experiment=EXPERIMENT,
        arms=list(ARMS),updates=UPDATES,full_steps=[UPDATES],checkpoint_every=250,
        source_global_step=BASE_STEP,source_native_total_step=NATIVE_BASE,
        final_global_step=BASE_STEP+UPDATES,final_native_total_step=NATIVE_BASE+UPDATES,
        **SOURCE_IDENTITIES,model_config=old['model_config'],runtime_version=old['runtime_version'],
        batch=8,microbatch=1,hybrid_lambda=.25,N_contract=[128,799],loss_normalization=LOSS_NORMALIZATION,
        optimizer='Matching EXPOSURE live/AdamW/RNG; lr1e-5 betas(.9,.95) wd0 clip1; no reset',
        trainable_parameters='All inherited generator parameters; no new parameters',EMA=False,
        per_arm_budget=limits(),total_budget={k:2*v for k,v in limits().items()},
        head_forward_budget=0,head_optimizer_updates=0,probe_forward_budget=0,
        schedules={a:dict(path=str(OUT/'data'/('schedule_'+a+'.json')),sha256=sha(OUT/'data'/('schedule_'+a+'.json'))) for a in ARMS},
        coefficient_schedules={a:dict(path=str(OUT/'data'/('coefficients_'+a+'.json')),sha256=sha(OUT/'data'/('coefficients_'+a+'.json'))) for a in ARMS},
        training_code_hashes={str(p):sha(p) for p in sorted(sources)},
        gate_hashes={str(p):sha(p) for p in (streamgate,cpugate,DATA_ROOT/'verification_receipt.json',resource)},
        input_case_hashes={str(resolve(r['case_path'])):r['case_sha256'] for r in m['train']},
        source_registration=str(OLD/'training_registration.json'),source_registration_sha256=sha(OLD/'training_registration.json'),
        source_data_registration=str(DATA_ROOT/'protocol.json'),source_data_registration_sha256=sha(DATA_ROOT/'protocol.json'),
        stream_policy='Same original parent/augmentation sequence in two arms, newly registered shared random seed identities; LOW chosen after parent per complete CLEAN/X_WIDTH pair',
        coefficient_policy='Same frozen original narrow-reference group MSE weights in both arms; per-arm actual free scalar counts determine gains and normalization. Within-group coordinate weighting remains.',
        inference_policy='Unchanged saved evaluation C/N/Gaussian arrays, Euler50; same single official Bench D0 and GEO_C/source winding/HINGE_OPP015/ALPHA/HOLE; no new evaluation noise',
        evaluation_requests=46,evaluation_new_raw=92,evaluation_new_D0=92,
        evaluation_generator_forwards=4600,evaluation_baseline='Existing EXPOSURE46 RAW/BASE/FINAL and scores reused read-only',
        raw_barrier='All92 new RAW saved before first D0',
        scientific_question='Does adding accepted same-parent fixed-C free-side lower-N variants improve completion beyond equal-budget ordinary continuation?',
        prior_routes=['Natural-N earlier expanded parent/N coverage but had no admitted same-parent LOW variants.','C_EDGEFLIP changed C interior tessellation; present C is unchanged.','JOINT_SHIFT changed geometry coordinates; present coordinates use original frame.'],
        decision='Compare paired adoption/body/support jointly with CF/FF/I, fragments, legal edges and postprocessing losses; no sole-metric or checkpoint selection.',
        failure_contract='Stop new dispatch on OOM/numerical failure. No silent replay/rollback; preserve attempts. Incomplete attempted work blocks recovery unless explicitly authorized.',
        checkpoint_policy='One atomic rolling latest each250; fixed immutable final1500 full/inference. No intermediate model evaluation.',
        no_cleanup_commit_push_default_change=True,predictor_head_training_budget=0)
    for key,p in paths.items():reg[key]=str(p);reg[key+'_sha256']=sha(p)
    write(REGISTRATION,reg)
    from .training import checked
    for arm in ARMS:checked(arm)
    return reg


if __name__=='__main__':
    r=main();print(__import__('json').dumps({k:r[k] for k in ('status','arms','updates','source_global_step','final_global_step','total_budget')}))
