"""Read-only CPU post-run audit for native_patch_*; no model or OT calls.

This module writes nothing. It loads finalized JSON/NPZ and safely mmap-loads
endpoints, verifies actual bytes/tensors/counters, and prints one JSON summary.
Run only after the main process records PASS_EXECUTION.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np
import torch

BASE_SHA='b32197096ecf8d662bfee18bbc4abf16cc4b4690cd28e5c47d14850729d2c222'
MODELS=('Native0','T1','T2')
PARENTS=(0,1)
TRAIN_K=(4,8,12)
EVAL_K=(2,4,8,12)
SEEDS=(9201,9202,9203,9204)
SHARED_FIELDS=('shared_batch_sha256','shared_input_hashes','parent_indices','K','free_face_counts',
    'effective_free_coordinates','times','Gaussian_hashes','paired_noise_hashes','pre_OT_target_hashes','known_slots','cycle_phase')


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def array_sha(value,legacy_dtype=False):
    value=np.ascontiguousarray(value)
    descriptor=(value.dtype,value.shape) if legacy_dtype else (str(value.dtype),tuple(value.shape))
    return hashlib.sha256(str(descriptor).encode()+value.tobytes()).hexdigest()


def state_sha(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        value=value.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str((tuple(value.shape),value.dtype)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def require(condition,message):
    if not condition:raise AssertionError(message)


def batch_plan(index):
    rotation={k:TRAIN_K[(i+index%3)%3] for i,k in enumerate(TRAIN_K)}
    pairs=((0,4),(0,4),(0,8),(0,12),(1,4),(1,8),(1,8),(1,12))
    return [(parent,rotation[k]) for parent,k in pairs]


def _endpoint(out,report,arm):
    result=report['arms'][arm];record=result['checkpoint'];path=Path(record['path'])
    require(path.resolve()==(out/arm/'endpoint_step500.pt').resolve(),'Endpoint path differs from current arm')
    require(path.stat().st_size==record['bytes'] and file_sha(path)==record['sha256'],'Endpoint file bytes/hash changed')
    saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    require(saved['schema']=='native_connected_patch_v1' and saved['arm']==arm,'Wrong checkpoint schema/arm')
    require(saved['completed_updates']==saved['optimizer_step_attempts']==500,'Endpoint update count differs')
    require(saved['recoverable'] and not saved['optimizer_step_in_progress'],'Endpoint not recoverable clean update boundary')
    require(saved['initial_backbone_sha256']==BASE_SHA,'Endpoint was not initialized from official EMA')
    require(saved['initial_state_sha256']==report['native_initial_state_sha256']==result['initial_state_sha256'],'Full initial model hashes differ')
    require(saved['protocol_sha256']==report['protocol_sha256'] and saved['protocol']==report['protocol'],'Endpoint protocol identity differs')
    state=saved['model'];parameters=sum(v.numel() for v in state.values())
    require(parameters==130251267 and len(state)==183,'Full trainable model tensor/parameter count differs')
    require(all(v.device.type=='cpu' and v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in state.values()),'Nonfinite or non-FP32 endpoint tensor')
    current_hash=state_sha(state)
    backbone_hash=state_sha({name[len('backbone.'):]:value for name,value in state.items() if name.startswith('backbone.')})
    require(current_hash==record['state_sha256'],'Full endpoint tensor hash changed')
    require(backbone_hash==record['backbone_state_sha256']==result['final_backbone_sha256'] and backbone_hash!=BASE_SHA,'Experimental backbone update/hash mismatch')
    optim=saved['optimizer'];require(len(optim['param_groups'])==1,'Unexpected optimizer groups')
    group=optim['param_groups'][0]
    require(group['lr']==1e-5 and tuple(group['betas'])==(.9,.95) and group['weight_decay']==0,'Optimizer hyperparameters changed')
    require(len(optim['state'])==len(group['params'])==183,'Optimizer does not cover full model')
    elements=0
    for item in optim['state'].values():
        require(float(item['step'])==500,'AdamW state step differs from500')
        require(item['exp_avg'].dtype==item['exp_avg_sq'].dtype==torch.float32,'Optimizer moment precision differs')
        require(item['exp_avg'].shape==item['exp_avg_sq'].shape,'Optimizer moment shape mismatch')
        require(bool(torch.isfinite(item['exp_avg']).all()) and bool(torch.isfinite(item['exp_avg_sq']).all()),'Nonfinite optimizer moment')
        elements+=item['exp_avg'].numel()
    require(elements==parameters,'Optimizer moment parameter coverage differs')
    stream=saved['rng']['stream']
    require(stream['batch_index']==500 and stream['sample_index']==4000 and stream['seed']==10 and stream['arm']==arm,'Saved stream boundary differs')
    for name,value in stream['rng'].items():
        rng=np.random.RandomState();rng.set_state((value[0],np.asarray(value[1],dtype=np.uint32),int(value[2]),int(value[3]),float(value[4])))
        restored=rng.get_state()
        require(restored[0]==value[0] and np.array_equal(restored[1],value[1]) and list(restored[2:])==value[2:],'Unsafe or inconsistent RNG restore')
    return dict(arm=arm,path=str(path),bytes=record['bytes'],file_sha256=record['sha256'],model_state_sha256=current_hash,
        backbone_state_sha256=backbone_hash,parameters=parameters,tensors=183,all_FP32_finite=True,
        optimizer_states_step500=183,optimizer_moment_elements=elements,role_nonzero_elements=int(torch.count_nonzero(state['role_embedding'])),
        recoverable=True,stream_batch_index=500,stream_sample_index=4000,stream_RNG_valid=True,
        weights_only_mmap_loaded=True,model_instantiated=False)


def audit(output):
    if torch.cuda.is_initialized():raise RuntimeError('Read-only audit must be a fresh CPU-only process')
    torch.set_num_threads(2)
    out=Path(output).resolve();report=json.loads((out/'results.json').read_text(encoding='utf-8-sig'))
    require(report['status']=='PASS_EXECUTION','Main process is not complete; no audit executed')
    require(file_sha(out/'protocol.json')==report['protocol_sha256'],'Protocol changed')
    inp=report['assets']['inputs'];path=Path(inp['path'])
    require(path.resolve()==(out/'inputs.npz').resolve() and file_sha(path)==inp['sha256'],'Input archive identity differs')
    with np.load(path,allow_pickle=False) as saved:arrays={key:saved[key].copy() for key in saved.files}
    require(set(arrays)==set(inp['arrays']),'Input archive key set changed')
    for name,value in arrays.items():
        record=inp['arrays'][name]
        require(array_sha(value,True)==record['sha256'] and list(value.shape)==record['shape'] and str(value.dtype)==record['dtype'],'Input array changed: '+name)
    for parent in PARENTS:
        previous=None
        for K in EVAL_K:
            C=arrays[f'parent{parent}_K{K}_constraints']
            require(C.shape==(K,3,3) and C.dtype==np.float32,'Known patch shape/dtype differs')
            if previous is not None:require(np.array_equal(C[:len(previous)],previous),'Nested patch C altered')
            previous=C
            target=arrays[f'parent{parent}_target'];free=arrays[f'parent{parent}_K{K}_free_target'];ids=arrays[f'parent{parent}_K{K}_source_face_ids'];free_ids=arrays[f'parent{parent}_K{K}_free_source_ids']
            require(len(set(ids.tolist()+free_ids.tolist()))==112 and len(ids)+len(free_ids)==112,'C/free source decomposition not disjoint and complete')
            require(np.array_equal(target[ids].reshape(K,3,3),C) and np.array_equal(target[free_ids],free),'Patch source geometry changed')
    for seed in SEEDS:
        require(array_sha(arrays[f'noise_seed{seed}'],True)==report['assets']['noise_hashes'][str(seed)],'Actual archived Gaussian changed')
    expected={(m,p,k,s) for m in MODELS for p in PARENTS for k in EVAL_K for s in SEEDS}
    generated=report['generated'];observed={(r['model'],r['parent'],r['K'],r['seed']) for r in generated}
    require(len(generated)==96 and observed==expected,'Formal generation matrix incomplete/duplicated')
    trajectory_hashes={};state_count=0
    for row in generated:
        raw_path=Path(row['raw_path'])
        require(raw_path.resolve().is_relative_to((out/'raw').resolve()),'Raw path escapes current output')
        require(file_sha(raw_path)==row['raw_sha256'],'Raw file hash changed: '+row['key'])
        with np.load(raw_path,allow_pickle=False) as saved:
            require(set(saved.files)=={'output','path'},'Formal raw archive lacks complete path')
            full=saved['output'].copy();states=saved['path'].copy()
        K,parent,seed=row['K'],row['parent'],row['seed']
        C=arrays[f'parent{parent}_K{K}_constraints'].reshape(K,9);z=arrays[f'noise_seed{seed}'][0]
        require(full.shape==(112,9) and states.shape==(51,112,9),'Output/path face count differs')
        require(full.dtype==states.dtype==np.float32 and np.isfinite(full).all() and np.isfinite(states).all(),'Raw nonfinite or wrong dtype')
        require(array_sha(full)==row['output_sha256'],'Raw endpoint array hash changed')
        require(np.array_equal(full,states[-1]),'Raw endpoint is not final saved state')
        require(np.array_equal(states[:,:K],np.broadcast_to(C,(51,K,9))),'C not bitwise preserved in all51 states')
        require(np.array_equal(states[0,K:],z[K:]),'Free initial state is not untouched archived Gaussian slots')
        require(len(full[K:])==112-K and not np.array_equal(states[0,K:],states[-1,K:]),'Free output count/update differs')
        require(row['sampling']['nfe']==50 and row['sampling']['known_all51_bitwise'],'Sampling facts differ')
        trajectory_hashes[row['key']]=dict(file_sha256=row['raw_sha256'],path_sha256=array_sha(states),K=K,known_states_verified=51,free_faces=112-K)
        state_count+=51
    teacher_summary=[];cover=Counter();donors=defaultdict(Counter);noise_hashes=set();shared_sequence=[]
    correct=report['arms']['Native_correct'];independent=report['arms']['Native_independent']
    for arm,result in report['arms'].items():
        require(result['status']=='PASS_EXECUTION' and result['completed_updates']==result['optimizer_step_attempts']==500,'Training arm incomplete')
        require(result['initial_state_sha256']==report['native_initial_state_sha256'] and result['initial_backbone_sha256']==BASE_SHA,'Fresh initialization hashes differ')
        require(not result['optimizer_step_in_progress'] and not result.get('batch_in_progress',False),'Training arm stopped inside a batch')
        require([t['step'] for t in result['teacher']]==[0,100,250,500],'Teacher checkpoints differ')
        for teacher in result['teacher']:
            require(teacher['forward_calls']==60 and len(teacher['rows'])==30,'Teacher must use60 forwards/30paired rows')
            entries={(q['parent'],q['K'],q['t']) for q in teacher['rows']}
            require(entries=={(p,k,t) for p in PARENTS for k in TRAIN_K for t in (.05,.2,.5,.8,.95)},'Teacher failed six-case time coverage')
            require(all(np.isfinite(q[m]['mse']) for q in teacher['rows'] for m in ('correct','mismatch')),'Nonfinite teacher MSE')
        teacher_summary.append(dict(arm=arm,steps=[0,100,250,500],forward_calls=240,paired_rows=120))
    require(len(correct['batch_audits'])==len(independent['batch_audits'])==500,'Training batch count differs')
    for index,(a,b) in enumerate(zip(correct['batch_audits'],independent['batch_audits'])):
        require(a['step']==b['step']==index+1,'Training step labels differ')
        require(all(a[field]==b[field] for field in SHARED_FIELDS),'Two-arm shared input stream differs at step'+str(index+1))
        plan=Counter(batch_plan(index));actual=Counter(zip(a['parent_indices'],a['K']))
        require(plan==actual,'Actual parent/K coverage differs from registered cycle')
        denom=sum((112-k)*9 for _,k in batch_plan(index))
        require(a['effective_free_coordinates']==b['effective_free_coordinates']==a['free_coordinate_denominator']==b['free_coordinate_denominator']==denom,'Effective loss denominator differs')
        require(len(a['shared_input_hashes'])==8,'Wrong microbatch count')
        require(all(p==d for p,d in zip(a['parent_indices'],a['donor_indices'])),'Correct arm donor mismatched')
        cover.update(actual);noise_hashes.update(a['Gaussian_hashes']);shared_sequence.append(a['shared_batch_sha256'])
        for parent,K,donor in zip(b['parent_indices'],b['K'],b['donor_indices']):
            require(donor in (0,1),'Independent donor outside two parents');donors[(parent,K)][donor]+=1
    expected_coverage=Counter(pair for i in range(500) for pair in batch_plan(i))
    require(cover==expected_coverage and sum(cover.values())==4000 and len(noise_hashes)==4000,'Actual full500 input coverage/fresh-noise mismatch')
    source_records={}
    require(report['source_files_unchanged'] and report['source_files']==report['source_files_after'],'Execution source protected comparison failed')
    for name,record in report['source_files'].items():
        p=out.parent.parent/name
        require(p.stat().st_size==record['bytes'] and file_sha(p)==record['sha256'],'Execution source changed: '+name)
        source_records[name]=record['sha256']
    require(report['protected_unchanged'] and report['protected_before']==report['protected_after'],'Historical/official protection check failed')
    endpoints=[_endpoint(out,report,arm) for arm in ('Native_correct','Native_independent')]
    counts=report['counts'];gates=report['model_gates']['actual_counts']
    expected_forwards=8000+480+4900+gates['native_forward_attempts']+gates['reference_forward_attempts']
    require(counts['model_forward_attempts']==counts['model_forward_returns']==expected_forwards,'Actual model-forward ledger differs from all stages')
    require(counts['optimizer_step_attempts']==counts['optimizer_updates']==counts['experimental_backbone_updates']==1000 and counts['protected_official_updates']==0,'Update ledger differs')
    require(counts['formal_rollouts']==96 and counts['diagnostic_rollouts']==2 and counts['complete_rollouts']==98 and counts['euler_updates']==4900,'Rollout/Euler ledger differs')
    require(counts['backward_attempts']==counts['backward_completed']==8000+gates['backward_attempts'],'Backward ledger differs')
    require(counts['geometry_gradient_attempts']==counts['geometry_gradient_completed']==0,'Unexpected geometry guidance used')
    require(len(report['sampling_diagnostics'])==2 and all(x['status']=='PASS' and x['bitwise_equal'] for x in report['sampling_diagnostics']),'Recorded sampling/reload diagnostics failed')
    require(len(report['scores'])==96 and len(report['bindings'])==48,'Scoring/binding grid incomplete')
    require(not torch.cuda.is_initialized(),'Audit initialized CUDA')
    donor_rows=[dict(parent=p,K=k,donor0=c[0],donor1=c[1],samples=sum(c.values()),correct=c[p],correct_fraction=c[p]/sum(c.values())) for (p,k),c in sorted(donors.items())]
    return dict(status='PASS',output=str(out),input_archive_sha256=inp['sha256'],raw_files_verified=96,
        complete_paths_verified=96,known_states_verified=state_count,all_shapes_FP32_finite=True,
        all_known_coordinates_bitwise_preserved=True,all_initial_free_slots_equal_archived_Gaussian=True,
        trajectory_hashes=trajectory_hashes,training_batches_matched=500,training_microbatches_matched=4000,
        fresh_Gaussian_count=4000,coverage=[dict(parent=p,K=k,samples=n) for (p,k),n in sorted(cover.items())],
        shared_batch_sequence_sha256=hashlib.sha256(''.join(shared_sequence).encode()).hexdigest(),
        independent_donor_statistics=donor_rows,teacher=teacher_summary,endpoints=endpoints,
        source_hashes_unchanged=source_records,protected_file_records_unchanged=len(report['protected_before']),
        protected_file_current_bytes_rehashed=False,protected_file_current_hash_scope='Main process before/after record equality; this audit freshly hashes execution sources, inputs, raw outputs and two endpoints',
        runtime_ledger=counts,audit_model_forwards=0,audit_OT_calls=0,audit_optimizer_updates=0,audit_cuda_initialized=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
    try:
        result=audit(args.output)
    except Exception as error:
        print(json.dumps(dict(status='FAIL',error=repr(error),traceback=traceback.format_exc(),
            audit_model_forwards=0,audit_OT_calls=0,audit_optimizer_updates=0),indent=2));raise
    print(json.dumps(result,indent=2,allow_nan=False))
