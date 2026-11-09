"""Freeze one W_MATCHED continuation after CPU and independent count gates."""
import json
from pathlib import Path
from .common import *
from . import stream


def _unchanged(mapping,message):
    for path,expected in mapping.items():
        if sha(resolve(path))!=expected:raise ValueError(message+': '+path)


def freeze():
    dest=OUT/'training_registration.json'
    if dest.exists():
        result=read(dest)
        for key in ('training_code_hashes','gate_hashes','protected'):_unchanged(result[key],'Frozen identity changed')
        return result
    paths=dict(manifest=MANIFEST,eval_manifest=OUT/'data/eval_manifest.json',
        data_gate=WIDE_ROOT/'data/cpu_gate.json',stream_gate=OUT/'data/stream_gate.json',
        training_gate=OUT/'training_cpu_gate.json',runtime_gate=OUT/'runtime_cpu_gate.json',
        eval_gate=OUT/'audits/eval_cpu_contract.json',cache_gate=OUT/'audits/BW_cache_audit.json',
        independent_gate=OUT/'coefficient_audit/complete.json',workflow_gate=OUT/'workflow_cpu_gate.json',
        protocol=OUT/'PROTOCOL.md',authorization=OUT/'AUTHORIZATION.json')
    gates=('data_gate','stream_gate','training_gate','runtime_gate','eval_gate','cache_gate','independent_gate','workflow_gate')
    for key in gates:
        if read(paths[key]).get('status')!='PASS':raise ValueError('Required gate not passed: '+key)
    gate=read(paths['training_gate'])
    if gate['source_identities']!=SOURCE_IDENTITIES:raise ValueError('Source state identities differ')
    if any(gate[key]!=0 for key in ('model_forwards','model_backwards','optimizer_updates','OT_calls','neural_model_constructions','tensor_weight_loads')):
        raise ValueError('This gate must not repeat neural/OT/restore probes')
    _unchanged(gate['source_hashes'],'CPU-tested source changed')
    m=read(MANIFEST);pool=stream.validate_manifest(m)
    if (len(m['train']),pool['high_parents'],pool['new_high_parents'])!=(30,6,4):raise ValueError('Exact original W30/HIGH6 data required')
    schedule=read(SCHEDULE);stream.validate(schedule,m['train']);coeffs=stream.checked_coefficients(COEFFICIENTS,schedule,m)
    if len(coeffs)!=1500 or sha(SCHEDULE)!=read(paths['stream_gate'])['original_W_schedule_sha256']:
        raise ValueError('Exact W stream changed')
    if sha(paths['eval_manifest'])!=sha(WIDE_ROOT/'data/eval_manifest.json'):
        raise ValueError('Evaluation manifest must remain a byte-identical W panel')
    ev=read(paths['eval_manifest'])
    if len(ev['requests'])!=46 or len({q['request_id'] for q in ev['requests']})!=46:raise ValueError('46 fixed requests required')
    independent=read(paths['independent_gate'])
    # The independent gate is itself frozen below; it must attest this candidate.
    if sha(COEFFICIENTS) not in json.dumps(independent):raise ValueError('Independent audit does not identify this coefficient schedule')
    source_reg=read(SOURCE_ROOT/'training_registration.json');wide_reg=read(WIDE_ROOT/'training_registration.json')
    inherited=dict(wide_reg['training_code_hashes']);_unchanged(inherited,'Qualified inherited source changed')
    files={resolve(path) for path in inherited}
    files|={Path(__file__).with_name(n) for n in ('__init__.py','common.py','runtime.py','stream.py','training.py','train_tests.py','register.py','workflow.py')}
    protected=dict(m.get('protected',{}))
    protected.update({str(path):sha(path) for path in (SOURCE,SOURCE_INFERENCE,MANIFEST,SCHEDULE,NARROW_MANIFEST,NARROW_SCHEDULE,
        WIDE_ROOT/'training_registration.json',WIDE_ROOT/'data/eval_manifest.json',
        NARROW_ROOT/'training_registration.json',SOURCE_ROOT/'training_registration.json')})
    if protected[str(SOURCE)]!=SOURCE_IDENTITIES['source_checkpoint_sha256']:raise ValueError('Original live source changed')
    _unchanged(protected,'Protected assets changed')
    result=dict(schema='BENCH_GROUP_WEIGHT_MATCH_REGISTRATION_V1',status='FROZEN',experiment=EXPERIMENT,
        arms=list(ARMS),updates=UPDATES,full_steps=[1500],checkpoint_every=250,
        source_global_step=BASE_STEP,final_global_step=22500,source_checkpoint=str(SOURCE),**SOURCE_IDENTITIES,
        model_config=source_reg['model_config'],batch=8,microbatch=1,hybrid_lambda=.25,
        optimizer='Inherited matching exposure B live AdamW/RNG; lr1e-5 betas(.9,.95) wd0 clip1',
        runtime_version='bench_natural_n_native128_799_v1',N_contract=[128,799],
        loss_normalization=LOSS_NORMALIZATION,training_seed=stream.DEFAULT_SEED,
        schedules={'W_MATCHED':dict(path=str(SCHEDULE),sha256=sha(SCHEDULE),original_arm='WIDE_HIGH')},
        manifest=str(MANIFEST),manifest_sha256=sha(MANIFEST),
        eval_manifest=str(paths['eval_manifest']),eval_manifest_sha256=sha(paths['eval_manifest']),
        coefficient_schedule=str(COEFFICIENTS),coefficient_schedule_sha256=sha(COEFFICIENTS),
        coefficient_formula='HIGH: (SSE_OLD+rho*SSE_HIGH)/D_B; rho=m_HIGH_B/m_HIGH_W; D_B=m_OLD+m_HIGH_B; FOCUS unchanged',
        backward_implementation='Retain original SSE_i/D_W, multiply D_W/D_B and additionally rho for HIGH before backward',
        coefficient_boundary='Group MSE coefficients only; neither equal realized losses nor equal gradient shares',
        training_code_hashes={str(path):sha(path) for path in sorted(files)},
        gate_hashes={str(paths[key]):sha(paths[key]) for key in (*gates,'protocol','authorization')},
        protected=protected,protocol_sha256=sha(paths['protocol']),
        data_counts=dict(old_parents=22,focus_parents=2,high_parents=6,conditions=30),
        train_N_range=[min(int(r['N']) for r in m['train']),max(int(r['N']) for r in m['train'])],
        paired_stream=read(COEFFICIENTS)['paired_stream'],
        reused_comparators=dict(B=dict(root=str(NARROW_ROOT),arm='NATURAL_HIGH'),W=dict(root=str(WIDE_ROOT),arm='WIDE_HIGH')),
        head_forward_budget=0,head_optimizer_updates=0,probe_forward_budget=0,
        per_arm_budget=dict(train_forward=12000,train_backward=12000,optimizer_update=1500),
        total_budget=dict(train_forward=12000,train_backward=12000,optimizer_update=1500,sampling_forward=2300,new_RAW=46,new_D0=46),
        evaluation_budget=dict(requests_per_model=46,new_M=46,reused_B=46,reused_W=46,new_RAW=46,new_D0=46,
            sampling_forward=2300,score_cells_total=414,new_score_cells=138,reused_score_cells=276),
        all_RAW_before_D0=True,sampling_steps=50,
        final_selection='Fixed1500 endpoint; no best selection or extension',
        failure_contract='No silent retry/replay; exact safe saved boundary only; all attempted work preserved',
        scientific_contrast='Exact W data stream with narrow B group-MSE coefficients versus W; no new data, network or geometry loss',
        resource_boundary='Reuse actual W max-N peak and inspect current headroom before single authorized dispatch',
        no_cleanup_commit_push_default_change=True)
    write_new(dest,result);write_new(OUT/'registration.json',result);return result


if __name__=='__main__':
    r=freeze();print(json.dumps({key:r[key] for key in ('status','arms','updates','final_global_step','total_budget')}))
