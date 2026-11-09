"""Session C CPU registration and separated frozen inference/teacher banks.

No model is instantiated. Call prepare only after Session A DATA_READY.
Canonical records are explicitly mapped from A's frozen plans by the caller.
"""
from __future__ import annotations
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
from .eval_common import ROOT, RUN, EVALUATION, REGISTRATION, MODELS, sha256, read_json, write_json, owned_path, snapshot


def array_hash(array):
    import hashlib
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str((str(array.dtype), tuple(array.shape))).encode()+array.tobytes()).hexdigest()


def load_data_api():
    path = RUN/'data/load_sample.py'
    spec = importlib.util.spec_from_file_location('control_mvp48_frozen_data_C', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def inventory_data():
    ready = RUN/'DATA_READY.json'
    document = read_json(ready)
    if document.get('status') != 'READY' or document.get('data_revision') != 'repair_v2':
        raise ValueError('Expected latest repair_v2 DATA_READY')
    if document.get('splits') != dict(train=32,validation=8,test=8) or document.get('three_model_generations') != 96 or document.get('teacher_checks') != 24:
        raise ValueError('A budget/split contract changed')
    verified={}
    for relative, record in document['files'].items():
        path=(RUN/relative).resolve()
        if RUN.resolve() not in path.parents or not path.is_file():
            raise ValueError(f'Missing or escaping attested file: {relative}')
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError(f'Attested file size/hash mismatch: {relative}')
        verified[relative]=record['sha256']
    for name in ('contract','manifest','split','independent_audit'):
        record=document[name]
        if verified.get(record['path']) != record['sha256']:
            raise ValueError(f'Unattested READY binding: {name}')
    for record in document['task_plans'].values():
        if verified.get(record['path']) != record['sha256']:
            raise ValueError('Unattested task plan')
    required = ['manifest.json', 'splits.json', 'patches.json', 'test_plan.json',
                'validation_plan.json', 'train_plan.json', 'load_sample.py']
    if any('data/'+name not in verified for name in required):
        raise ValueError('DATA_READY omits required data files')
    for relative,digest in document['code'].items():
        path=(ROOT/relative).resolve()
        if ROOT.resolve() not in path.parents or sha256(path)!=digest:
            raise ValueError(f'A source identity changed: {relative}')
    audit=read_json(RUN/document['independent_audit']['path'])
    if audit.get('status')!='PASS':
        raise ValueError('Independent A audit not PASS')
    return dict(data_ready_sha256=sha256(ready),data_revision='repair_v2',
                files=verified,ready=document,verified_file_count=len(verified),
                contract_sha256=document['contract']['sha256'])


def canonical_records(api=None):
    api=api or load_data_api()
    result=[]
    for kind,want in (('test',32),('validation',8)):
        plan=api.load_plan(kind)
        if len(plan['records'])!=want or plan['case_count']!=want:
            raise ValueError(f'{kind} fixed plan count mismatch')
        rows=[]
        for r in plan['records']:
            if kind=='validation' and r['t']!=.5:
                raise ValueError('Teacher time differs from .5')
            rows.append(dict(parent_id=r['uid'],condition_id=r['patch_id'],
                noise_id=r['noise_index'],gaussian_seed=r['gaussian_seed'],
                N=r['N'],K=r['K'],source_case_id=r['case_id'],source_record=dict(r)))
        result.append(rows)
    _validate_records(*result)
    return tuple(result)


def verify_data_identity(identity):
    if sha256(RUN/'DATA_READY.json') != identity['data_ready_sha256']:
        raise ValueError('DATA_READY changed')
    for path, digest in identity['files'].items():
        if sha256(RUN/path) != digest:
            raise ValueError(f'Frozen data changed: {path}')


def _save_npz(path, **arrays):
    path=owned_path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    return dict(path=str(path.relative_to(RUN)), sha256=sha256(path))


def _validate_records(test_records, teacher_records):
    required={'parent_id','condition_id','noise_id','gaussian_seed'}
    if len(test_records)!=32 or any(not required <= set(r) for r in test_records):
        raise ValueError('Exactly32 explicit test records required')
    parents=sorted({r['parent_id'] for r in test_records})
    if len(parents)!=8:
        raise ValueError('Exactly8 test parents required')
    for parent in parents:
        rows=[r for r in test_records if r['parent_id']==parent]
        if len(rows)!=4 or {(r['condition_id'],r['noise_id']) for r in rows}!={('p0',0),('p0',1),('p1',0),('p1',1)}:
            raise ValueError('Each test parent requires registered p0,p1 by noises0,1')
        for noise in (0,1):
            if len({r['gaussian_seed'] for r in rows if r['noise_id']==noise})!=1:
                raise ValueError('Same parent/noise must share Gaussian across patches')
    if len(teacher_records)!=8 or len({r['parent_id'] for r in teacher_records})!=8:
        raise ValueError('Exactly8 validation parents required')
    if any(r['condition_id']!='p0' or r.get('noise_id',0)!=0 for r in teacher_records):
        raise ValueError('Teacher only first patch and one fixed noise')
    if set(parents)&{r['parent_id'] for r in teacher_records}:
        raise ValueError('Test/validation overlap')


def prepare(test_records, teacher_records):
    """Freeze before any formal forward or score; run real eight OT pairs once."""
    _validate_records(test_records,teacher_records)
    protocol_path=REGISTRATION/'eval_protocol.json'
    if protocol_path.exists() or (EVALUATION/'inputs').exists() or (EVALUATION/'teacher_inputs').exists():
        raise FileExistsError('Preparation already began; inspect evidence, never silently regenerate or repeat OT')
    identity=inventory_data(); api=load_data_api()
    if canonical_records(api)!=(test_records,teacher_records):
        raise ValueError('Canonical records must exactly map latest A plans')
    resources_before=snapshot()
    write_json(REGISTRATION/'eval_resources_before_cpu_prepare.json',resources_before)
    splits=read_json(RUN/'data/splits.json')
    if isinstance(splits.get('splits'),dict):
        splits=splits['splits']
    if set(splits['test']) != {r['parent_id'] for r in test_records} or set(splits['validation']) != {r['parent_id'] for r in teacher_records}:
        raise ValueError('Canonical records differ from A frozen split')
    if set(splits['train']) & ({r['parent_id'] for r in test_records}|{r['parent_id'] for r in teacher_records}):
        raise ValueError('Evaluation parent participated in conditional training')
    from .eval_scoring import source_identity, METRIC_SPECS, prepare_reference_from_sample
    from .eval_teacher import prepare_teacher_sample
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU preparation unexpectedly initialized CUDA')
    protocol=dict(schema='control_mvp48_evaluation_v1', owner='Session C',status='FROZEN',
        frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), data_identity=identity,
        data_revision='repair_v2',data_description='User-authorized minimally cleaned official-source-derived geometry; no further generation/scoring repairs',
        models=list(MODELS), endpoint_step=1000, endpoint_source='Session B load_mvp_checkpoint; final raw weights only, no EMA',
        test_records=test_records,teacher_records=teacher_records,
        formal_requests=96,formal_steps=50,formal_NFE=4800,teacher_forwards=24,
        solver=dict(name='original_Euler50',times='k/50,k=0..49',dt='1/50',saved_states=51),
        runtime=dict(forward='BF16',parameters='FP32',state_integration='FP32',eval=True,no_grad=True,
                     deterministic_algorithms=True,warn_only=False,TF32=False,SDPA='math',CUBLAS_WORKSPACE_CONFIG=':4096:8'),
        inference_allowlist=['C','N','free_Gaussian_or_current_state','t','known_mask','valid_mask'],
        input_layout='actual retained N after A authorized minimal cleanup, no padding; C first K; A Gaussian(N-K) fills free slots',
        geometry=dict(points=512,L='bbox diagonal of full GT, offline only',C_excluded=True,
                      complement='each own condition exact GT free complement',coverage_threshold='0.01L',
                      interface_threshold='0.001L',Gamma='all A-registered actual known/free crossing edges',
                      predicted_surface_RNG='private fixed RandomState9911, area weighted, same arithmetic as original',
                      GT_queries='one frozen bank per condition, reused across all models/noises',
                      exact_topology='NOT_RUN',watertight_certification='NOT_RUN'),
        scorer_identity=source_identity(),metric_specs=METRIC_SPECS,
        aggregation='mean2C*2noise within parent; eight parents equal weight; no best noise; failed requests retained with valid denominators',
        comparisons=['Geo-T1','Full-Geo','Full-T1'],
        teacher=dict(t=.5,free_only_OT_pairs=8,training_loss_calls=0,geometry_gradients=0,
                     FM='mean squared velocity error over valid free scalar coordinates',
                     layout='known first, unchanged A canonical corners, no extra permutation; unused plan permutation_seed remains recorded',
                     diagnostics='original finite/area/near-degenerate geometry on xhat=xt+(1-t)*v; C restored'),
        repeat_checks=dict(allowed_max=2,planned=0,reason='No repeat needed at freeze; no empirical repeatability claim'),
        scheduling=dict(gpu_workers=1,lock='/tmp/wrp_meshflow_jedge5_diag_gpu0.lock',
                        headroom='max(2048MiB,10%total)',CPU_threads_per_process=1,
                        CPU_scoring='may overlap independent later inference under measured CPU/RAM availability'),
        failure='preserve invalid raw/partial outputs and attempts; stop new GPU dispatch on numerical/OOM failure; no seed replacement/retry',
        control_rule='Only reuse complete four-cell outputs from distinct test parents with exactly equal N,K and byte-identical registered Gaussian; each correct diagonal uses its own complement. Otherwise NOT_RUN; no extra generation.',
        views=dict(noise_id=0,elevation=20,azimuth=-55,vertical_axis='y',coordinate_display_order=[0,2,1],projection='ortho',columns=['GT','C','T1','Geo','Full'],
                   limits='shared over two conditions and all displayed raw/GT for each parent'),
        restrictions=['no training/checkpoint selection','no solver scan','no guidance','no generation GT/OT',
                      'no Pair','no welding/deleting/repair','no training feedback before all arms finished'])
    write_json(protocol_path,protocol,exclusive=True)
    start=time.perf_counter();ledger=dict(status='PREPARING',test_inputs=[],conditions=[],teacher_inputs=[],
        OT_attempts=0,OT_returns=0,model_forwards=0,CUDA_initialized=False,protocol_sha256=sha256(protocol_path))
    write_json(REGISTRATION/'eval_preparation_ledger.json',ledger)
    samples={}; expected=[]
    for index, record in enumerate(test_records):
        uid,patch=record['parent_id'],record['condition_id'];condition_key=(uid,patch)
        if condition_key not in samples:
            sample=api.load_sample(uid,patch);samples[condition_key]=sample
            reference=prepare_reference_from_sample(sample)
            # The reference is used by CPU scoring only; formal worker reads inputs exclusively.
            condition_id=f'condition{len(ledger["conditions"]):02d}'
            arrays=dict(C=sample['constraints'],gt_free=sample['free_target'],full_target=sample['full_target'],
                        L=np.asarray(sample['bbox_diagonal']),N=np.asarray(sample['N']),K=np.asarray(sample['K']))
            arrays.update({f'reference_{k}':v for k,v in reference['arrays'].items()})
            location=_save_npz(EVALUATION/'references'/f'{condition_id}.npz',**arrays)
            ref_meta=reference['audit']
            ledger['conditions'].append(dict(parent_id=uid,condition_id=patch,reference_metadata=ref_meta,**location))
        sample=samples[condition_key];N,K=sample['N'],sample['K']
        if (N,K)!=(record['N'],record['K']):
            raise ValueError('Plan N/K differs from actual retained geometry')
        condition=api.load_condition(uid,patch)
        noise=api.gaussian(record['gaussian_seed'],N-K,fp32=True)
        z=np.zeros((1,N,9),np.float32);z[:,K:]=noise.reshape(1,N-K,9)
        C=condition['C'].copy();known=condition['known_mask'];valid=condition['valid_mask']
        if not np.array_equal(known,np.arange(N)<K) or not valid.all() or int(condition['N'])!=N:
            raise ValueError('Actual N/mask boundary mismatch')
        case_id=f'case{index:02d}'
        location=_save_npz(EVALUATION/'inputs'/f'{case_id}.npz',C=C,z=z,N=np.asarray(N),
                          known_mask=known,valid_mask=valid)
        bank=dict(record,case_id=case_id,N=N,K=K,C_sha256=array_hash(C),noise_sha256=array_hash(noise),
                  z_sha256=array_hash(z),known_mask_sha256=array_hash(known),valid_mask_sha256=array_hash(valid),**location)
        ledger['test_inputs'].append(bank)
        write_json(REGISTRATION/'eval_preparation_ledger.json',ledger)
        for model in MODELS:
            expected.append(dict(bank,model=model,request_id=f'{case_id}_{model}'))
    for index,record in enumerate(teacher_records):
        sample=api.load_sample(record['parent_id'],record['condition_id'])
        noise=api.gaussian(record['gaussian_seed'],sample['N']-sample['K'],fp32=False)
        ledger['OT_attempts']+=1;write_json(REGISTRATION/'eval_preparation_ledger.json',ledger)
        arrays,metadata=prepare_teacher_sample(sample,noise)
        ledger['OT_returns']+=1
        location=_save_npz(EVALUATION/'teacher_inputs'/f'teacher{index:02d}.npz',**arrays)
        ledger['teacher_inputs'].append(dict(record,teacher_id=f'teacher{index:02d}',identity=metadata,**location))
        write_json(REGISTRATION/'eval_preparation_ledger.json',ledger)
    groups={}
    for row in ledger['test_inputs']:
        groups.setdefault((row['N'],row['K'],row['noise_sha256']),[]).append(row)
    cross=[rows for rows in groups.values() if len({r['parent_id'] for r in rows})>=2]
    # Do not decide a comparison after viewing outputs; root must explicitly register any eligible full grid before GPU.
    control=dict(status='ELIGIBLE_REQUIRES_FROZEN_FOUR_GRID' if cross else 'NOT_RUN',
        reason='No distinct parents share all of actual N,K and registered free Gaussian bytes' if not cross else 'Matching input groups require explicit four-cell offline exclusion protocol',
        same_N_groups={str(n):sorted({r['parent_id'] for r in ledger['test_inputs'] if r['N']==n}) for n in sorted({r['N'] for r in ledger['test_inputs']})},
        eligible_input_groups=cross,extra_generation=0,
        claim='This round does not comprehensively verify geometry editing control')
    verify_data_identity(identity)
    ledger.update(status='PASS',seconds=time.perf_counter()-start,CUDA_initialized=torch.cuda.is_initialized())
    if ledger['CUDA_initialized'] or ledger['OT_returns']!=8 or len(expected)!=96:
        raise RuntimeError('Preparation count/runtime gate failed')
    write_json(REGISTRATION/'eval_requests.json',expected,exclusive=True)
    write_json(REGISTRATION/'eval_control.json',control,exclusive=True)
    write_json(REGISTRATION/'eval_preparation_ledger.json',ledger)
    write_json(REGISTRATION/'eval_resources_after_cpu_prepare.json',snapshot())
    return ledger
