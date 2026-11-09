"""Isolated C-reference inputs and actual local-OT training stream; CPU only."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from . import common as C
C.package()
from meshflow_control.data.dataset import gaussian, training_time, permutations
from meshflow_control.data.io import array_hash, bits, json_hash
from meshflow_control.training import coupling
from r2.pilots.user_data_scaling_32_67_v1 import training_data as olddata

CONTRACT='CREF_LOCAL_preOT_aug_minus_a_div2__FP32_MODEL_TARGET_aug_minus_a__world_Geo_v1'
PROPOSAL=Path('/mnt/c/Users/warpwang/.codex/attachments/9072b8e9-1a30-483d-818a-f95b937428ce/Pasted text.txt')

def verify(files):
    for p,h in files.items():
        if C.sha(p)!=h:raise ValueError('Frozen dependency changed: '+str(p))

def registration():
    r=C.read(C.OUT/'registration.json');verify(r['source_hashes'])
    if C.sha(C.OUT/'inputs/manifest.json')!=r['inputs_manifest_sha256']:raise ValueError('Eval inputs changed')
    p=C.read(C.OUT/'plans/CREF_C200.json')
    if p['registration_sha256']!=C.sha(C.OUT/'registration.json') or json_hash(p['records'])!=r['records_sha256']:raise ValueError('Plan/registration mismatch')
    return r

def prepare():
    path=C.OUT/'registration.json'
    if path.exists():return registration()
    from meshflow_control.config import MODEL_CONFIG
    baseline_plan=C.QUICK/'plans/SMALL_C200.json';plan=C.read(baseline_plan)
    qreg=C.read(C.QUICK/'registration.json');init=C.read(C.QUICK/'training/SMALL_C200/initialization.json')
    status=C.read(C.QUICK/'training/SMALL_C200/status.json');receipt=C.read(C.SOURCE.with_suffix('.json'))
    inputs=C.NOISE/'inputs/manifest.json';inp=C.read(inputs);records=plan['records']
    assert len(records)==1600 and plan['arm']=='SMALL_C200'
    assert plan['registration_sha256']==C.sha(C.QUICK/'registration.json')
    assert status['status']=='COMPLETE' and status['local_step']==200 and status['global_step']==6200
    assert status['calls']['successful']=={'train_forward':1600,'train_backward':1600,'optimizer_update':200}
    assert not status['calls']['failed'] and not status['calls']['pending_call_ids']
    assert qreg['source_checkpoint_sha256']==receipt['sha256']==init['source_file_sha256']
    assert qreg['source_generator_sha256']==receipt['generator_state_sha256']==init['loaded_generator_sha256']
    assert qreg['source_global_step']==init['restored_global_step']==receipt['global_step']==6000
    assert qreg['source_data_sha256']==C.sha(C.DATA) and set(qreg['train_task_ids'])==set(inp['focused_task_ids'])
    d=olddata.Dataset(C.DATA)
    for b in range(200):
        rr=records[b*8:(b+1)*8]
        assert sorted(Counter(r['uid'] for r in rr).values())==[2,2,2,2]
        assert sum(r['alpha']==1. for r in rr)==4
        for slot,r in enumerate(rr):
            assert r['sample_index']==b*8+slot and r['step']==b+1 and r['global_step']==6001+b
            assert r['task_id'] in inp['focused_task_ids'] and r['arm']=='SMALL_C200'
            assert float(training_time(r['t_seed']))==r['t']
    protected={}
    def protect(p,expected=None):
        h=C.sha(p)
        if expected is not None and expected!=h:raise ValueError('Source hash differs '+str(p))
        protected[str(p)]=h;return h
    # Source file hash certifies the saved state fields. No GPU/model is loaded.
    protect(C.SOURCE,receipt['sha256'])
    for p in [baseline_plan,C.DATA,inputs,C.QUICK/'registration.json',C.QUICK/'training/SMALL_C200/initialization.json',
              C.QUICK/'training/SMALL_C200/status.json',C.QUICK/'training/SMALL_C200/calls.jsonl',C.SOURCE.with_suffix('.json'),PROPOSAL,
              C.ROOT/'AGENTS.md',C.ROOT/'WORKSPACE.md',C.ROOT/'native_t1/docs/control_metrics.md']:
        protect(p)
    for tid in inp['focused_task_ids']:
        d.verify_condition(tid);t=d.tasks[tid];p=d.parents[t['uid']]
        protect(t['npz'],t['sha256']);protect(p['npz'],p['sha256'])
    for i in range(200):
        oldcache=C.QUICK/'ot/SMALL_C200'/f'{i:03d}.npz';oldreceipt=C.read(oldcache.with_suffix('.json'))
        assert oldreceipt['records_sha256']==json_hash(records[i*8:i*8+8])
        protect(oldcache,oldreceipt['sha256']);protect(oldcache.with_suffix('.json'))
    code=[C.CODE/n for n in ('common.py','prepare.py','train.py','model.py')]
    code.extend([C.ROOT/'r2/pilots/seat_quick_learnability_v1/train.py',C.ROOT/'r2/pilots/user_data_scaling_32_67_v1/training_data.py',
                 C.ROOT/'r2/pilots/user_condition_finetune_v1/trainer.py'])
    code.extend((C.ROOT/'r2/src/meshflow_control').rglob('*.py'))
    for p in sorted(set(code)):protect(p)
    # Read checkpoint tensors on CPU through mmap, without constructing a model
    # or changing RNG. Saved hashes must match the actually used baseline init.
    import torch
    payload=torch.load(C.SOURCE,map_location='cpu',weights_only=True,mmap=True)
    for a,b in [('generator_state_sha256','loaded_generator_sha256'),('optimizer_state_sha256','loaded_optimizer_sha256'),('rng_sha256','loaded_rng_sha256')]:
        assert payload[a]==init[b]
    assert payload['model_config']==MODEL_CONFIG and payload['global_step']==6000
    expected_init=dict(generator=payload['generator_state_sha256'],optimizer=payload['optimizer_state_sha256'],rng=payload['rng_sha256'],model_config=payload['model_config'],optimizer_parameter_names=payload['optimizer_parameter_names'])
    del payload
    C._write_bytes(C.OUT/'inputs/manifest.json',inputs.read_bytes())
    C._write_bytes(C.OUT/'plans/SMALL_C200_reused.json',baseline_plan.read_bytes())
    reg=dict(schema='c_reference_training_registration_v1',experiment='R2_C_REFERENCE_FRAME_V1',status='FROZEN',
        source_checkpoint=str(C.SOURCE),source_checkpoint_sha256=receipt['sha256'],source_generator_sha256=receipt['generator_state_sha256'],
        source_data=str(C.DATA),source_data_sha256=C.sha(C.DATA),source_global_step=6000,endpoint_global_step=6200,
        baseline_plan=str(baseline_plan),baseline_plan_sha256=C.sha(baseline_plan),records_sha256=json_hash(records),
        inputs_manifest_sha256=C.sha(inputs),arms=[C.TRAIN_LABEL],updates_per_arm=200,batch=8,microbatch=1,
        train_task_ids=inp['focused_task_ids'],actual_training_samples=1600,expected_initialization=expected_init,
        paired_baseline='Completed SMALL_C200; identical ordered 1600 parent/task/N/K/augmentation/time/Gaussian/epsilon/permutation records; only coordinate strategy differs',
        sampling_policy='four fixed parents twice each batch; four clean/four width augmented; original frozen plan records including original arm remain byte-identical',
        coordinate_contract=CONTRACT,
        coordinate_definition='Augment world C/Y first; FP32-area/centroid accumulation in original C row order gives FP32 C-only anchor a before slot/corner permutation. Local backbone C/Y=world-a. OT costs at world preOT-a/2; model target independently world_target-a, reorder through new OT. Geo consumes separately supplied original augmented world C with identical corner/face slots.',
        Gaussian='Original iid Gaussian and epsilon seeds/arrays, not translated or demeaned; HYBRID0.25 after local target matching; never xt_world-a',
        loss='Original free-only FM, shared effective-batch free-coordinate denominator; no extra loss or params',
        optimizer='Same restored EXP67 AdamW/RNG: lr1e-5 betas(.9,.95), weight_decay0, clip1; no reset/EMA/scheduler',
        runtime='FP32 parameters/AdamW; BF16 forward; strict deterministic; TF32 off; SDPA math',
        checkpoint_policy='100 recovery and fixed200 final normally; exceptional clean safe boundary may save exact actual cursor without replay; only200 evaluated',
        retry='No failed/pending forward/backward/update replay; immutable local OT maps with durable reservations',
        gpu_budget_gib=6,concurrency=1,headroom_mib=2048,training_forward_budget=1600,training_backward_budget=1600,optimizer_update_budget=200,
        extra_training_probes=0,source_hashes=protected)
    C.write(path,reg)
    C.write(C.OUT/'plans/CREF_C200.json',dict(arm=C.TRAIN_LABEL,records=records,registration_sha256=C.sha(path),
        baseline_plan_sha256=C.sha(baseline_plan),parent_counts=plan['parent_counts'],condition_counts=plan['condition_counts']))
    C.write(C.OUT/'preparation/baseline_identity.json',dict(status='PASS',source_initialization=expected_init,
        original_completed_updates=200,original_completed_samples=1600,records_byte_identical=True,baseline_plan_sha256=C.sha(baseline_plan),
        source_checkpoint_sha256=receipt['sha256'],inference_manifest_byte_identical=True,model_forwards=0,backwards=0,optimizer_updates=0))
    return reg

class Builder:
    def __init__(self,d,maps=None):self.data=d;self.maps=maps;self.new_maps={}
    def sample(self,row):
        from .model import area_weighted_center
        case=self.data.task(row['task_id'],row['alpha']);n,k=case['N'],case['K']
        known_ids,free_ids=case['source_face_ids'],case['free_source_ids']
        world=case['full_target'].reshape(n,3,3);anchor=area_weighted_center(world[known_ids])
        local=(world-anchor).astype(np.float32)
        free_order,corners,order=permutations(row['permutation_seed'],n,k);ordered_ids=free_ids[free_order]
        pre_world=case['pre_ot_full'][ordered_ids][np.arange(n-k)[:,None],corners[k:]]
        pre=(pre_world-anchor/np.float32(2)).astype(np.float32)
        target_before=local[ordered_ids][np.arange(n-k)[:,None],corners[k:]]
        noise=gaussian(row['gaussian_seed'],n-k);epsilon=gaussian(row['epsilon_seed'],n-k)
        ident=dict(contract=CONTRACT,pre=array_hash(pre),target_model=array_hash(target_before),raw_noise=array_hash(noise),
            anchor=array_hash(anchor),source_ids=array_hash(ordered_ids),corners=array_hash(corners[k:]),known_ids=array_hash(known_ids),
            alpha=row['alpha'],task_id=row['task_id'],sample_index=row['sample_index'])
        key=json_hash(ident);prefix=str(row['sample_index'])
        if self.maps is None:
            face,corner=coupling.nested_ot_map(pre,noise);cache_status='MISS'
            self.new_maps.update({prefix+'_key':np.asarray(key),prefix+'_face':face,prefix+'_corner':corner})
        else:
            assert str(self.maps[prefix+'_key'].item())==key
            face,corner=coupling.validate_map(self.maps[prefix+'_face'],self.maps[prefix+'_corner']);cache_status='HIT'
        target=coupling.to_noise_slots(target_before,face,corner)
        t=np.asarray(training_time(row['t_seed']),np.float32);mixed,_,_=coupling.coupling_path(noise,target,epsilon,t)
        x0=np.zeros((n,3,3),np.float32);x1=np.zeros_like(x0);context=np.zeros_like(x0);geo=np.zeros_like(x0)
        x0[k:]=mixed;x1[k:]=target
        context[:k]=local[known_ids][np.arange(k)[:,None],corners[:k]]
        geo[:k]=world[known_ids][np.arange(k)[:,None],corners[:k]]
        ids=np.r_[known_ids,ordered_ids[np.argsort(face)]]
        mapped=np.concatenate((corners[:k],coupling.to_noise_slots(corners[k:],face,corner)))
        known=(np.arange(n)<k)[order]
        x0,x1,context,geo=[a[order].reshape(n,9) for a in (x0,x1,context,geo)]
        ids,mapped=ids[order],mapped[order];u=x1-x0;xt=(np.float32(1)-t)*x0+t*x1;xt[known]=context[known]
        expected=local[ids][np.arange(n)[:,None],mapped].reshape(n,9)
        expected_world=world[ids][np.arange(n)[:,None],mapped].reshape(n,9)
        assert bits(expected[~known],x1[~known]) and bits(expected[known],context[known])
        assert bits(expected_world[known],geo[known]) and not np.any(geo[~known])
        assert bits((geo[known].reshape(-1,3,3)-anchor).reshape(-1,9),context[known])
        s=dict(x0=x0,x1=x1,u=u,xt=xt,context=context,geo_world=geo,C_anchor=anchor,t=t,y=np.asarray(n,np.int64),
            known_mask=known,valid_mask=np.ones(n,bool),source_face_ids=ids,corner_permutations=mapped,face_permutation=order,
            object_id=row['uid'],task_id=row['task_id'],K=k,alpha=row['alpha'],ratio=row['ratio'],
            free_gaussian_before_OT=noise,free_epsilon=epsilon,free_target_before_OT=pre,free_model_target_before_OT=target_before,
            ot_face_map=face,ot_corner_map=corner,ot_cache_key=key,ot_cache_status=cache_status,
            sample_index=row['sample_index'],step=row['step'],global_step=row['global_step'],free_coordinate_count=9*(n-k),
            arm=C.TRAIN_LABEL,lambda_value=.25,condition_type=row['condition_type'],region_type=row['region_type'])
        fields=('x0','x1','u','xt','context','geo_world','C_anchor','t','y','known_mask','valid_mask','source_face_ids','corner_permutations','face_permutation','free_gaussian_before_OT','free_epsilon','free_target_before_OT','free_model_target_before_OT','ot_face_map','ot_corner_map')
        hashes={f:array_hash(s[f]) for f in fields};s['shared_input_sha256']=json_hash(dict(hashes,task_id=row['task_id'],alpha=row['alpha']))
        s['coupled_input_sha256']=s['shared_input_sha256'];s['input_bytehash']=json_hash({f:hashes[f] for f in ('xt','geo_world','t','y','known_mask','valid_mask')});s['label_bytehash']=array_hash(u)
        return s

def batch(d,records,arm,index):
    if arm!=C.TRAIN_LABEL:raise ValueError('Only registered CREF_C200 allowed')
    folder=C.OUT/'ot'/arm;path=folder/f'{index:03d}.npz';rp=path.with_suffix('.json');attempt=path.with_suffix('.attempt.json')
    identity=dict(contract=CONTRACT,records_sha256=json_hash(records),manifest_sha256=d.manifest_sha256,
        registration_sha256=C.sha(C.OUT/'registration.json'),batch_index=index)
    if rp.exists():
        receipt=C.read(rp)
        if receipt['identity']!=identity or receipt['sha256']!=C.sha(path):raise ValueError('Local OT cache identity differs')
        maps=C.arrays(path)
    else:
        if attempt.exists() or path.exists():raise RuntimeError('Incomplete local OT reservation; no silent remapping')
        C.write(attempt,dict(identity=identity,time_unix=time.time(),pid=os.getpid()));maps=None
    begin=time.perf_counter();builder=Builder(d,maps);samples=[builder.sample(r) for r in records]
    for s,r in zip(samples,records):
        olddata.validate_sample(s,r);s['arm']=arm
    if maps is None:
        C.frozen_npz(path,builder.new_maps)
        C.write(rp,dict(identity=identity,sha256=C.sha(path),actual_OT_attempts=len(records),actual_OT_returns=len(records),seconds=time.perf_counter()-begin,status='PASS'))
    return samples

def precompute(workers=2):
    reg=registration();assert workers in (1,2)
    d=olddata.Dataset(C.DATA);p=C.read(C.OUT/'plans/CREF_C200.json');tick=time.perf_counter()
    original_calls=[json.loads(line) for line in (C.QUICK/'training/SMALL_C200/calls.jsonl').read_text().splitlines()]
    actual_inputs={r['context']['sample_index']:r['context'] for r in original_calls if r['event']=='CALL_BEGIN' and r['kind']=='train_forward'}
    assert sorted(actual_inputs)==list(range(1600))
    C.resources(C.OUT/'preparation'/('cpu_resources_'+str(time.time_ns())+'.json'))
    # Each immutable batch is owned once; cached maps reconstruct arrays but do
    # not run matching again. All seeds/record order remain fixed.
    def work(i):
        records=p['records'][8*i:8*i+8];ss=batch(d,records,C.TRAIN_LABEL,i)
        original=olddata.Builder(d,C.arrays(C.QUICK/'ot/SMALL_C200'/f'{i:03d}.npz'))
        baseline=[original.sample(r) for r in records]
        for s,b in zip(ss,baseline):
            actual=actual_inputs[s['sample_index']]
            assert b['input_bytehash']==actual['input_bytehash'] and b['label_bytehash']==actual['label_bytehash']
            assert b['ot_cache_status']=='HIT' and s['ot_cache_key']!=b['ot_cache_key']
            for key in ('free_gaussian_before_OT','free_epsilon','t','y','known_mask','valid_mask','face_permutation'):
                assert bits(s[key],b[key]),key
            assert bits(s['geo_world'],b['context'])
            # Identical x0 is intentional: local noise is not shifted world noise.
            assert bits(s['x0'],b['x0'])
        return [dict(sample_index=s['sample_index'],task_id=s['task_id'],anchor=s['C_anchor'].tolist(),
            input_bytehash=s['input_bytehash'],label_bytehash=s['label_bytehash'],ot_cache_key=s['ot_cache_key'],ot_status=s['ot_cache_status'],
            geo_world_sha256=array_hash(s['geo_world']),local_context_sha256=array_hash(s['context']),
            gaussian_sha256=array_hash(s['free_gaussian_before_OT']),epsilon_sha256=array_hash(s['free_epsilon']),
            baseline_actual_call_input_verified=True,baseline_noise_epsilon_time_masks_order_and_x0_identical=True,
            geo_world_equals_actual_baseline_context=True,baseline_reconstruction_new_OT=0) for s in ss]
    with ThreadPoolExecutor(max_workers=workers) as pool:rows=[r for part in pool.map(work,range(200)) for r in part]
    assert [r['sample_index'] for r in rows]==list(range(1600))
    stable=[{k:v for k,v in r.items() if k!='ot_status'} for r in rows]
    C.write(C.OUT/'preparation/local_stream_manifest.json',dict(status='PASS',records=stable,contract=CONTRACT,registration_sha256=C.sha(C.OUT/'registration.json')))
    receipt=dict(status='PASS',workers=workers,records=1600,new_OT=sum(r['ot_status']=='MISS' for r in rows),cache_hits=sum(r['ot_status']=='HIT' for r in rows),wall_seconds=time.perf_counter()-tick,model_forwards=0,backwards=0,optimizer_updates=0)
    C.write(C.OUT/'preparation'/('precompute_'+str(time.time_ns())+'.json'),receipt)
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--precompute',action='store_true');p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    reg=prepare();print(json.dumps(dict(status=reg['status'],registration=str(C.OUT/'registration.json'),training_updates=200,model_calls=0)))
    if a.precompute:print(json.dumps(precompute(a.workers)))
