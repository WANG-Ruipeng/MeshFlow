"""Immutable registration of the single authorized wider HIGH-pool branch."""
import collections
import json
from pathlib import Path
from .common import *
from . import stream


def _unchanged(mapping, message):
    for p,h in mapping.items():
        if sha(p)!=h:raise ValueError(message+': '+p)


def freeze():
    dest=OUT/'training_registration.json'
    if dest.exists():
        result=read(dest)
        _unchanged(result['training_code_hashes'],'Frozen source changed')
        _unchanged(result['gate_hashes'],'Frozen gate changed')
        _unchanged(result['protected'],'Protected source changed')
        return result
    paths=dict(manifest=OUT/'data/manifest.json',eval_manifest=OUT/'data/eval_manifest.json',
        data_gate=OUT/'data/cpu_gate.json',stream_gate=OUT/'data/stream_gate.json',
        training_gate=OUT/'training_cpu_gate.json',runtime_gate=OUT/'runtime_cpu_gate.json',
        eval_gate=OUT/'audits/eval_cpu_contract.json',cache_gate=OUT/'audits/B_cache_audit.json',
        workflow_gate=OUT/'workflow_cpu_gate.json',
        protocol=OUT/'PROTOCOL.md')
    manifest=read(paths['manifest']);pool=stream.validate_manifest(manifest)
    if (len(manifest['train']),pool['high_parents'],pool['new_high_parents'])!=(30,6,4):
        raise ValueError('Only the actual accepted 30-parent/HIGH6 pool is authorized')
    for key in ('data_gate','stream_gate','training_gate','runtime_gate','eval_gate','cache_gate','workflow_gate'):
        if read(paths[key]).get('status')!='PASS':raise ValueError('Required gate not passed: '+key)
    gate=read(paths['training_gate'])
    if gate['source_identities']!=SOURCE_IDENTITIES:raise ValueError('CPU gate source differs')
    if (gate['model_forwards'],gate['model_backwards'],gate['optimizer_updates'],gate['OT_calls'])!=(0,0,0,1):
        raise ValueError('CPU validation scope differs')
    _unchanged(gate['source_hashes'],'CPU-tested source changed')
    if sha(SOURCE)!=SOURCE_IDENTITIES['source_checkpoint_sha256']:raise ValueError('Original B source changed')
    ev=read(paths['eval_manifest']);requests=ev['requests']
    if ev.get('status')!='FROZEN' or len(requests)!=46 or len({q['request_id'] for q in requests})!=46:
        raise ValueError('Fixed46 paired evaluation requests required')
    prior_eval=read(NARROW_ROOT/'data/eval_manifest.json')
    by_id={q['request_id']:q for q in requests}
    prior_ids={q['request_id'] for q in prior_eval['requests']}
    if len(prior_ids)!=38 or not prior_ids<=set(by_id):raise ValueError('Original fixed38 panel must remain')
    for q in prior_eval['requests']:
        for key,value in q.items():
            if by_id[q['request_id']].get(key)!=value:raise ValueError('Original fixed request changed: '+key)
    additional=[q for q in requests if q['request_id'] not in prior_ids]
    counts=collections.Counter(q['parent_id'] for q in additional)
    if counts!={p:2 for p in manifest['high_pool_contract']['new_parent_ids']}:
        raise ValueError('Exactly two frozen noises for each actual new HIGH parent required')
    parents={r['parent_id'] for r in manifest['train']}
    for q in requests:
        group=q.get('report_group',q.get('eval_group',''))
        if ('HOLDOUT' in group or group=='DEV') and q['parent_id'] in parents:
            raise ValueError('Held-out evaluation parent entered training')
    schedule_path=OUT/'data/schedule_WIDE_HIGH.json';schedule=read(schedule_path)
    stream.validate(schedule,manifest['train'],1500,'WIDE_HIGH')
    paired=stream.validate_frozen_narrow(schedule,manifest['train'])
    source_reg=read(SOURCE_ROOT/'training_registration.json')
    narrow_reg=read(NARROW_ROOT/'training_registration.json')
    inherited=dict(source_reg['training_code_hashes'],**narrow_reg['training_code_hashes'])
    _unchanged(inherited,'Protected inherited source changed')
    files={resolve(p) for p in inherited}
    files|={Path(__file__).with_name(n) for n in ('__init__.py','common.py','runtime.py','training.py',
        'stream.py','train_tests.py','prepare.py','register.py','workflow.py')}
    protected=dict(manifest.get('protected',{}))
    protected.update({str(p):sha(p) for p in (SOURCE,SOURCE_INFERENCE,
        SOURCE_ROOT/'training_registration.json',NARROW_ROOT/'training_registration.json',
        NARROW_ROOT/'data/manifest.json',NARROW_ROOT/'data/schedule_NATURAL_HIGH.json',
        NARROW_ROOT/'data/eval_manifest.json')})
    _unchanged(protected,'Protected data/source changed')
    result=dict(schema='BENCH_HIGH_JOINT_COVERAGE_REGISTRATION_V1',status='FROZEN',experiment=EXPERIMENT,
        arms=list(ARMS),updates=UPDATES,full_steps=[1500],checkpoint_every=250,
        source_global_step=BASE_STEP,final_global_step=22500,source_checkpoint=str(SOURCE),
        **SOURCE_IDENTITIES,model_config=source_reg['model_config'],batch=8,microbatch=1,hybrid_lambda=.25,
        optimizer='Inherited matching exposure B live AdamW and RNG; lr1e-5 betas(.9,.95) wd0 clip1',
        runtime_version='bench_natural_n_native128_799_v1',N_contract=[128,799],
        loss_normalization='whole_effective_batch_free_coordinates',training_seed=stream.DEFAULT_SEED,
        schedules={'WIDE_HIGH':dict(path=str(schedule_path),sha256=sha(schedule_path))},
        manifest=str(paths['manifest']),manifest_sha256=sha(paths['manifest']),
        eval_manifest=str(paths['eval_manifest']),eval_manifest_sha256=sha(paths['eval_manifest']),
        training_code_hashes={str(p):sha(p) for p in sorted(files)},
        gate_hashes={str(paths[k]):sha(paths[k]) for k in ('data_gate','stream_gate','training_gate',
            'runtime_gate','eval_gate','cache_gate','workflow_gate','protocol')},
        protected=protected,protocol_sha256=sha(paths['protocol']),
        data_counts=dict(old_parents=22,focus_parents=2,high_parents=6,new_high_parents=4,conditions=30),
        train_N_range=[min(int(r['N']) for r in manifest['train']),max(int(r['N']) for r in manifest['train'])],
        paired_stream=paired,reused_comparator=dict(root=str(NARROW_ROOT),arm=NARROW_ARM,
            global_step=22500,not_training_source=True),
        head_forward_budget=0,head_optimizer_updates=0,probe_forward_budget=0,
        per_arm_budget=dict(train_forward=12000,train_backward=12000,optimizer_update=1500),
        total_budget=dict(train_forward=12000,train_backward=12000,optimizer_update=1500,
            sampling_forward=2700,new_RAW=54,new_D0=54),
        evaluation_budget=dict(requests_per_model=46,new_W=46,new_B=8,reused_B=38,
            new_RAW=54,new_D0=54,sampling_forward=2700),
        all_RAW_before_D0=True,sampling_steps=50,
        final_selection='Predeclared1500 endpoint only; no best checkpoint or extension',
        failure_contract='No silent retry/replay; exact safe saved boundary only; preserve all attempts',
        scientific_contrast='HIGH natural parent coverage, higher N range and per-parent repetition change jointly; not isolated N causality',
        low_version_policy='One accepted existing C and unchanged natural full source per parent',
        resource_boundary='Prior maxN384 training peak is not a maxN776 estimate; observe authorized formal updates before another dispatch',
        no_cleanup_commit_push_default_change=True)
    write_new(dest,result);write_new(OUT/'registration.json',result)
    return result


if __name__=='__main__':
    r=freeze();print(json.dumps({k:r[k] for k in ('status','arms','updates','final_global_step','total_budget')}))
