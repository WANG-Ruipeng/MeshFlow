"""Fixed teacher bank and H9-only diagnostics for analytic geometry alignment.

Nothing runs at import. The caller owns model loading and the global forward
ledger. This module counts its own phase calls, persists them serially, and
never performs an optimizer step or scans whole-model parameter gradients.
"""
from .runtime import configure_stable_runtime
from pathlib import Path
import copy
import hashlib
import json
import time
import numpy as np
import torch
from .artifacts import atomic_json, file_sha256
from .accounting import count_ot_calls
from .data import load_archive, NativeTrainingStream, native_collate, model_inputs, array_hash, shared_sample_hash
from .model import native_execution
from .training import masked_fm_loss

BANK_SCHEMA='native_t1_representation_teacher_bank_v1'
TEACHER_SEED=20261002
SHUFFLE_SEED=20261003
TIMES=(.1,.3,.5,.7,.9)
TEACHER_LABELS=('Geo_initial','common_aux_initial','G0','G_Angular','G_AngularScale')
GROUPS={'centroid':slice(0,3),'log_edges':slice(3,6),'log_area':slice(6,7),'normal_outer':slice(7,13)}
STATE_KEYS=('xt','x0','x1','u','t','y','known_mask','valid_mask','context','source_face_ids','corner_permutations','face_permutation','known_target_ids')


def _save(r,out):
    atomic_json(Path(out)/'results.json',r)


def _seal(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _record(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=file_sha256(path),bytes=path.stat().st_size)


def _counts(r):
    value=r.setdefault('diagnostic_counts',{})
    for key in ('teacher_bank_OT_attempts','teacher_bank_OT_returns','calibration_forward_attempts','calibration_forward_returns',
                'calibration_H9_grad_attempts','calibration_H9_grad_returns','teacher_forward_attempts','teacher_forward_returns'):
        value.setdefault(key,0)
    return value


def _state_hash(sample):
    return _seal({key:array_hash(np.asarray(sample[key])) for key in STATE_KEYS})


def _state(base,t,*,parent,noise_index,pair_index,time_index):
    sample={k:v.copy() if isinstance(v,np.ndarray) else copy.deepcopy(v) for k,v in base.items()}
    sample['t']=np.asarray(t,dtype=np.float32)
    sample['xt']=(np.float32(1)-sample['t'])*sample['x0']+sample['t']*sample['x1']
    sample['xt'][sample['known_mask']]=sample['context'][sample['known_mask']]
    sample.update(parent_index=parent,donor_index=parent,K=12,noise_index=noise_index,
                  pair_index=pair_index,time_index=time_index,noise_seed=TEACHER_SEED,
                  state_id=f'P{parent}_noise{noise_index}_t{time_index}')
    sample['shared_input_sha256']=shared_sample_hash(sample)
    sample['context_sha256']=array_hash(sample['context'])
    sample['state_sha256']=_state_hash(sample)
    return sample


def _verify_bank(samples):
    if len(samples)!=20:raise ValueError('Teacher bank requires20 fixed states.')
    expected={(p,n,j) for p in (0,1) for n in (0,1) for j in range(5)}
    if {(s['parent_index'],s['noise_index'],s['time_index']) for s in samples}!=expected:
        raise ValueError('Teacher parent/noise/time identity differs.')
    by_pair={}
    for sample in samples:
        if sample['K']!=12 or sample['xt'].shape!=(112,9) or sample['xt'].dtype!=np.float32:
            raise ValueError('Teacher requires actual N112/K12 FP32 coordinates.')
        if not bool(sample['valid_mask'].all()) or int(sample['known_mask'].sum())!=12:
            raise ValueError('Teacher masks differ.')
        if sample['t'].dtype!=np.float32 or sample['t'].item()!=np.float32(TIMES[sample['time_index']]).item():
            raise ValueError('Teacher fixed time changed.')
        if not np.isfinite(sample['xt']).all() or not np.array_equal(sample['xt'][sample['known_mask']],sample['context'][sample['known_mask']]):
            raise ValueError('Teacher nonfinite state or changed C.')
        actual=_state_hash(sample)
        if actual!=sample['state_sha256']:raise ValueError('Teacher state array hash differs.')
        pair=(sample['parent_index'],sample['noise_index'])
        invariant={k:array_hash(sample[k]) for k in ('x0','x1','u','context','known_mask','source_face_ids','corner_permutations','face_permutation')}
        if pair in by_pair and invariant!=by_pair[pair]:raise ValueError('Z/Y/C/permutations changed across teacher times.')
        by_pair[pair]=invariant
    return dict(status='PASS',pairs=4,states=20,times=list(TIMES),same_pair_Z_Y_C_and_permutations=True,
                bank_state_sha256=_seal([s['state_sha256'] for s in samples]))


def build_teacher_bank(r,out):
    """Create four original OT pairs once using a dedicated private input stream.

    Returns (20 sample dicts, audit). NativeTrainingStream._sample draws one
    original random t per pair; that unused draw is recorded, then replaced by
    the five registered t values. The formal training stream is never touched.
    """
    out=Path(out);path=out/'representation_teacher_bank.npz'
    if path.exists() or r.get('teacher_bank'):raise RuntimeError('Teacher bank creation is one-shot; use load_teacher_bank to resume.')
    counts=_counts(r);audit=dict(status='RUNNING',seed=TEACHER_SEED,OT_attempts=0,OT_returns=0,
        formal_training_stream_advanced=False,model_forwards=0,optimizer_updates=0)
    r['teacher_bank']=audit;_save(r,out);tick=time.perf_counter()
    try:
        asset=r['assets']['inputs'];archive=load_archive(asset['path'],asset['sha256'])
        stream=NativeTrainingStream(archive,seed=TEACHER_SEED)
        initial=stream.state_dict();arrays={};metadata=[];samples=[]
        for pair_index,(parent,noise_index) in enumerate(((0,0),(0,1),(1,0),(1,1))):
            local={}
            try:
                with count_ot_calls(local):base=stream._sample(parent,12)
            finally:
                attempted=local.get('OT_calls',0);returned=local.get('OT_returns',0)
                counts['teacher_bank_OT_attempts']+=attempted;counts['teacher_bank_OT_returns']+=returned
                audit['OT_attempts']+=attempted;audit['OT_returns']+=returned;_save(r,out)
            array_keys=[];small={}
            for key,value in base.items():
                if isinstance(value,np.ndarray) or isinstance(value,np.generic):
                    arrays[f'pair{pair_index}_{key}']=np.asarray(value);array_keys.append(key)
                elif isinstance(value,(str,int,float,bool)) or value is None:small[key]=value
                else:raise TypeError('Unexpected teacher sample field: '+key)
            pair=dict(pair_index=pair_index,parent_index=parent,noise_index=noise_index,array_keys=array_keys,
                metadata=small,unused_original_random_t=float(base['t']),
                base_array_hashes={key:array_hash(arrays[f'pair{pair_index}_{key}']) for key in array_keys})
            metadata.append(pair)
            for j,t in enumerate(TIMES):
                state=_state(base,t,parent=parent,noise_index=noise_index,pair_index=pair_index,time_index=j)
                arrays['state_'+state['state_id']+'_xt']=state['xt']
                samples.append(state)
        if audit['OT_attempts']!=4 or audit['OT_returns']!=4:raise RuntimeError('Teacher bank exceeded4 original OT pairs.')
        verified=_verify_bank(samples)
        meta=dict(schema=BANK_SCHEMA,seed=TEACHER_SEED,input_sha256=archive.sha256,pairs=metadata,times=list(TIMES),
            state_hashes={s['state_id']:s['state_sha256'] for s in samples},
            private_stream_start_sha256=_seal(initial),private_stream_end_sha256=_seal(stream.state_dict()),
            original_random_t='One private time draw per pair is unused; five fixed t values replace it.',
            correspondence='Original nested OT reorders only Z; Y scale, face/corner ordering and source IDs are unchanged across times.')
        arrays['metadata_json']=np.frombuffer(json.dumps(meta,sort_keys=True).encode('utf-8'),dtype=np.uint8)
        temporary=path.with_suffix('.npz.partial')
        if temporary.exists():raise FileExistsError(temporary)
        with temporary.open('xb') as output:np.savez_compressed(output,**arrays)
        temporary.replace(path)
        audit.update(status='PASS',artifact=_record(path),input_sha256=archive.sha256,pairs=metadata,
            state_hashes=meta['state_hashes'],verification=verified,private_stream_start_sha256=meta['private_stream_start_sha256'],
            private_stream_end_sha256=meta['private_stream_end_sha256'],original_random_t=meta['original_random_t'],seconds=time.perf_counter()-tick)
        _save(r,out)
        return samples,audit
    except BaseException as error:
        audit.update(status='STOPPED',error=repr(error),seconds=time.perf_counter()-tick);_save(r,out);raise


def load_teacher_bank(path,expected_sha256):
    """Load exactly the saved20 states, with zero OT/model/RNG calls."""
    path=Path(path)
    if file_sha256(path)!=expected_sha256:raise ValueError('Frozen teacher archive hash changed.')
    samples=[]
    with np.load(path,allow_pickle=False) as saved:
        meta=json.loads(saved['metadata_json'].tobytes().decode('utf-8'))
        if meta.get('schema')!=BANK_SCHEMA or meta.get('seed')!=TEACHER_SEED or meta.get('times')!=list(TIMES):
            raise ValueError('Teacher archive schema/seed/times differ.')
        if len(meta['pairs'])!=4:raise ValueError('Teacher archive requires four OT pairs.')
        for pair in meta['pairs']:
            base=dict(pair['metadata']);index=pair['pair_index']
            for key in pair['array_keys']:
                value=saved[f'pair{index}_{key}'].copy()
                if array_hash(value)!=pair['base_array_hashes'][key]:raise ValueError('Saved teacher base pair changed.')
                base[key]=value
            for j,t in enumerate(TIMES):
                state=_state(base,t,parent=pair['parent_index'],noise_index=pair['noise_index'],pair_index=index,time_index=j)
                if state['state_sha256']!=meta['state_hashes'][state['state_id']]:raise ValueError('Teacher reconstructed state differs.')
                original=saved['state_'+state['state_id']+'_xt']
                if state['xt'].tobytes()!=original.tobytes():raise ValueError('Saved FP32 teacher Xt was not reconstructed exactly.')
                samples.append(state)
    verified=_verify_bank(samples)
    return samples,dict(status='PASS',artifact=_record(path),verification=verified,seed=TEACHER_SEED,
        OT_attempts=0,OT_returns=0,model_forwards=0,formal_training_stream_advanced=False,source_metadata=meta)


def _targets(provider,sample,device):
    target=provider.for_sample(sample,device=device)
    q=target['q']
    if not isinstance(q,torch.Tensor) or q.dtype!=torch.float32 or q.requires_grad or q.shape!=(target['free_count'],13):
        raise ValueError('Frozen free-target q[F,13] must be FP32 and gradient-free.')
    if not bool(torch.isfinite(q).all()):raise ValueError('Nonfinite target descriptor.')
    if target['angular_count']<1:raise ValueError('Sample has no valid Angular direction.')
    return target


def _feature_output(value):
    if not isinstance(value,(tuple,list)) or len(value)!=2:raise ValueError('Expected generator (velocity,H9) return.')
    velocity,h9=value
    if h9.shape!=(1,336,768):raise ValueError('Expected true block9 corner hidden[1,336,768].')
    return velocity.float(),h9


def _lambdas(rows):
    if len(rows)!=8:raise ValueError('Calibration requires eight individual samples.')
    medians={key:float(np.median([row[key] for row in rows])) for key in ('g_FM','g_A','g_S')}
    if any(not np.isfinite(v) or v<0 for v in medians.values()) or medians['g_A']<=0 or medians['g_S']<=0:
        raise RuntimeError('Nonfinite or zero denominator in H9 gradient calibration; no fallback weight.')
    result=dict(lambda_A=.05*medians['g_FM']/medians['g_A'],lambda_S=.05*medians['g_FM']/medians['g_S'])
    if not all(np.isfinite(v) for v in result.values()):raise RuntimeError('Nonfinite calibrated lambda.')
    return dict(**result,medians=medians,target_fraction=.05,
        interpretation='Initial H9-only gradient norm calibration; neither a later-step guarantee nor a parameter-gradient ratio.')


def calibrate(r,out,model,heads,samples,target_provider):
    """Exactly8 model forwards and24 autograd.grad calls restricted to true H9.

    samples must come from the caller's independent clone of the formal stream.
    Each loss uses its own single-sample coordinate/valid-face denominator.
    """
    from .geometry_representation import auxiliary_losses
    if r.get('calibration'):raise RuntimeError('Calibration is one-shot; no silent repeat or lambda search.')
    if len(samples)!=8 or heads is None:raise ValueError('Eight cloned training samples and common full heads required.')
    counts=_counts(r);audit=dict(status='RUNNING',rows=[],forward_attempts=0,forward_returns=0,H9_grad_attempts=0,H9_grad_returns=0,
        optimizer_updates=0,parameter_backward_calls=0,gradient_scope='True corner H9 only; no model-parameter grad scan',
        FM_normalization='Single sample own valid-free coordinate mean',
        auxiliary_normalization='Single sample own valid Angular faces / own free faces x13 for Scale')
    r['calibration']=audit;_save(r,out);tick=time.perf_counter()
    try:
        device=next(model.parameters()).device
        for index,sample in enumerate(samples):
            batch=native_collate([sample],device)
            target=_targets(target_provider,sample,device)
            with native_execution(model,coordinates=batch['xt']):
                if audit['forward_attempts']>=8:raise RuntimeError('Calibration8-forward cap reached.')
                audit['forward_attempts']+=1;counts['calibration_forward_attempts']+=1
                velocity,h9=_feature_output(model(*model_inputs(batch),return_feature_layer=9))
                audit['forward_returns']+=1;counts['calibration_forward_returns']+=1
                if not h9.requires_grad:raise RuntimeError('H9 was detached or captured without autograd.')
                fm=masked_fm_loss(velocity,batch['u'],batch['valid_mask'],batch['known_mask'])
                with torch.autocast(device_type=device.type,enabled=False):
                    predictions=heads(h9,batch['valid_mask'],batch['known_mask'])
                    terms=auxiliary_losses(predictions,target['q'])
                if terms['scale'] is None:raise RuntimeError('Calibration requires the full common P/S initialization.')
                norms={};loss_values={}
                for j,(name,loss) in enumerate((('FM',fm),('A',terms['angular']),('S',terms['scale']))):
                    if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite calibration loss.')
                    if audit['H9_grad_attempts']>=24:raise RuntimeError('Calibration24-H9-grad cap reached.')
                    audit['H9_grad_attempts']+=1;counts['calibration_H9_grad_attempts']+=1
                    grad,=torch.autograd.grad(loss,(h9,),retain_graph=j<2,create_graph=False,allow_unused=False)
                    audit['H9_grad_returns']+=1;counts['calibration_H9_grad_returns']+=1
                    if not bool(torch.isfinite(grad).all()):raise RuntimeError('Nonfinite true-H9 gradient.')
                    norms['g_'+name]=float(torch.linalg.vector_norm(grad.detach().double()))
                    loss_values['L_'+name]=float(loss.detach())
                    del grad
            audit['rows'].append(dict(sample_index=index,parent_index=int(sample['parent_index']),K=int(sample['K']),t=float(sample['t']),
                shared_input_sha256=sample['shared_input_sha256'],context_sha256=sample['context_sha256'],labels_sha256=target['labels_sha256'],
                free_count=target['free_count'],angular_count=target['angular_count'],invalid_angular_count=target['invalid_angular_count'],
                H9_shape=list(h9.shape),H9_dtype=str(h9.dtype),**norms,**loss_values))
            del velocity,h9,fm,predictions,terms,loss,batch,target
            _save(r,out)
        if (audit['forward_returns'],audit['H9_grad_returns'])!=(8,24):raise RuntimeError('Calibration call budget incomplete.')
        audit.update(status='PASS',**_lambdas(audit['rows']),seconds=time.perf_counter()-tick)
        _save(r,out);return audit
    except BaseException as error:
        audit.update(status='STOPPED',error=repr(error),seconds=time.perf_counter()-tick);_save(r,out);raise

def _array_statistics(value):
    """Read-only FP64 reductions of an already computed FP32 head output."""
    value=np.asarray(value,dtype=np.float64)
    if value.ndim!=2 or value.shape[1]!=13 or not np.isfinite(value).all():
        raise ValueError('Descriptor diagnostics require finite [F,13] arrays.')
    variance=value.var(axis=0,ddof=0)
    return dict(face_count=len(value),mean_by_dimension=value.mean(axis=0).tolist(),
        variance_by_dimension=variance.tolist(),mean_variance=float(variance.mean()),
        group_mean_variance={name:float(variance[sl].mean()) for name,sl in GROUPS.items()},
        norm_mean=float(np.linalg.norm(value,axis=1).mean()),
        norm_min=float(np.linalg.norm(value,axis=1).min()),norm_max=float(np.linalg.norm(value,axis=1).max()),
        variance_denominator='F (population variance across this sample free faces)')


def _auxiliary_scores(predictions,q):
    from .geometry_representation import auxiliary_losses
    terms=auxiliary_losses(predictions,q)
    values=dict(Angular_loss=float(terms['angular']),
        Angular_cosine=float(terms['cosine'][terms['angular_valid']].mean()),
        free_count=terms['free_count'],angular_count=terms['angular_count'],invalid_angular_count=terms['invalid_angular_count'],
        Scale_MSE='NOT_APPLIED' if terms['scale'] is None else float(terms['scale']),
        Scale_group_MSE='NOT_APPLIED' if terms['group_scale_MSE'] is None else {key:float(value) for key,value in terms['group_scale_MSE'].items()})
    numbers=[values['Angular_loss'],values['Angular_cosine']]
    if isinstance(values['Scale_MSE'],float):numbers.extend([values['Scale_MSE'],*values['Scale_group_MSE'].values()])
    if not np.isfinite(numbers).all():raise RuntimeError('Nonfinite teacher auxiliary score.')
    return values


def _constant_scores(q,metadata):
    mean=np.asarray(metadata['q_mean_FP64'],dtype=np.float64)
    if mean.shape!=(13,) or not np.isfinite(mean).all():raise ValueError('Frozen training descriptor mean is missing or nonfinite.')
    constant=torch.as_tensor(mean.astype(np.float32),device=q.device)
    error=(q-constant).square()
    return dict(source='Frozen training-only deduplicated q mean; never estimated from teacher states.',
        training_statistics_sha256=metadata['statistics_sha256'],value_FP64=mean.tolist(),
        Scale_MSE=float(error.mean()),Scale_group_MSE={name:float(error[:,sl].mean()) for name,sl in GROUPS.items()},
        Angular_cosine='NOT_APPLIED_NEAR_ZERO_CENTERED_MEAN',prediction_variance=0.0)


def _shuffle_permutation(faces,state_index):
    # Keep an unbiased permutation, including its chance fixed points.
    return np.random.RandomState(SHUFFLE_SEED+state_index).permutation(faces).astype(np.int64)


def teacher_evaluate(r,out,model,heads,bank,target_provider,*,label,shuffle_targets=False):
    """Score a frozen 20-state bank once, with no autograd or optimizer updates.

    Auxiliary endpoints optionally rescore exactly the same head predictions
    against one private, fixed permutation of the sample's targets. This adds
    no model or head forward. It is correspondence sensitivity, not a trained
    negative control. G0 never receives auxiliary heads.
    """
    if label not in TEACHER_LABELS:raise ValueError('Unregistered teacher model label.')
    if label in r.setdefault('teacher',{}):raise RuntimeError('Teacher label was already attempted; no automatic rerun.')
    without_heads=label in ('Geo_initial','G0')
    if without_heads!=(heads is None):raise ValueError('G0/Geo_initial must have no auxiliary heads; other labels require their own heads.')
    if shuffle_targets and label not in ('G_Angular','G_AngularScale'):
        raise ValueError('Target shuffling is reserved for the two trained auxiliary endpoints.')
    if heads is not None:
        if (label=='G_Angular')!=(heads.S is None):raise ValueError('Auxiliary head architecture differs from registered teacher label.')
        if any(module.training for module in heads.modules()):raise ValueError('Teacher auxiliary heads must be eval.')
    verification=_verify_bank(bank);counts=_counts(r);device=next(model.parameters()).device
    metadata=target_provider.metadata()
    audit=dict(status='RUNNING',model=label,phase='initial' if label in ('Geo_initial','common_aux_initial') else 'endpoint',
        cumulative_updates=2000 if label in ('Geo_initial','common_aux_initial') else 2500,
        auxiliary_status='NOT_APPLIED' if without_heads else 'APPLIED_READ_ONLY',
        role='Shared initial P/S diagnostic, not a G0 auxiliary score.' if label=='common_aux_initial' else label,
        bank_state_sha256=verification['bank_state_sha256'],training_statistics_sha256=metadata['statistics_sha256'],
        rows=[],forward_attempts=0,forward_returns=0,head_forward_count=0,autograd_calls=0,optimizer_updates=0,
        loss_normalization='Own-sample free-coordinate FM mean; own valid-face Angular mean; own free-face x13 Scale mean.',
        statistical_reductions='Heads/losses FP32; saved prediction population variances use read-only CPU FP64 reductions.',
        shuffled_targets=bool(shuffle_targets),shuffled_targets_are_a_trained_negative_control=False)
    r['teacher'][label]=audit;_save(r,out);tick=time.perf_counter()
    try:
        for index,sample in enumerate(bank):
            batch=native_collate([sample],device)
            if not np.array_equal(batch['xt'][0].detach().cpu().numpy(),sample['xt']):
                raise RuntimeError('Actual teacher coordinates differ from the frozen bank.')
            target=_targets(target_provider,sample,device)
            with torch.no_grad(),native_execution(model,coordinates=batch['xt']):
                if audit['forward_attempts']>=20 or counts['teacher_forward_attempts']>=100:
                    raise RuntimeError('Registered teacher forward cap reached.')
                audit['forward_attempts']+=1;counts['teacher_forward_attempts']+=1
                if without_heads:
                    velocity=model(*model_inputs(batch)).float();h9=None
                else:
                    velocity,h9=_feature_output(model(*model_inputs(batch),return_feature_layer=9))
                audit['forward_returns']+=1;counts['teacher_forward_returns']+=1
                fm=masked_fm_loss(velocity,batch['u'],batch['valid_mask'],batch['known_mask'])
                if not bool(torch.isfinite(fm)):raise RuntimeError('Nonfinite teacher FM loss.')
                row=dict(model=label,phase=audit['phase'],cumulative_updates=audit['cumulative_updates'],
                    state_id=sample['state_id'],state_sha256=sample['state_sha256'],parent_index=int(sample['parent_index']),
                    noise_index=int(sample['noise_index']),time_index=int(sample['time_index']),t=float(sample['t']),
                    shared_input_sha256=sample['shared_input_sha256'],context_sha256=sample['context_sha256'],labels_sha256=target['labels_sha256'],
                    FM_velocity_MSE=float(fm),free_count=target['free_count'],angular_count=target['angular_count'],
                    invalid_angular_count=target['invalid_angular_count'],auxiliary_status=audit['auxiliary_status'])
                if without_heads:
                    row.update(Angular_loss='NOT_APPLIED',Angular_cosine='NOT_APPLIED',Scale_MSE='NOT_APPLIED',
                        Scale_group_MSE='NOT_APPLIED',prediction_statistics='NOT_APPLIED',target_statistics='NOT_APPLIED',
                        constant_baseline='NOT_APPLIED',shuffled_target='NOT_APPLIED')
                else:
                    with torch.autocast(device_type=device.type,enabled=False):
                        predictions=heads(h9,batch['valid_mask'],batch['known_mask']);audit['head_forward_count']+=1
                        row.update(_auxiliary_scores(predictions,target['q']))
                        row['prediction_statistics']={name:'NOT_APPLIED' if predictions[name] is None else _array_statistics(predictions[name].cpu().numpy())
                            for name in ('z','z_norm','q_hat')}
                        row['target_statistics']=_array_statistics(target['q'].cpu().numpy())
                        row['constant_baseline']=_constant_scores(target['q'],metadata)
                        if shuffle_targets:
                            permutation=_shuffle_permutation(target['free_count'],index)
                            shuffled=target['q'][torch.as_tensor(permutation,device=device)]
                            row['shuffled_target']=dict(status='READ_ONLY_SAME_PREDICTION',seed=SHUFFLE_SEED+index,
                                permutation_sha256=array_hash(permutation),fixed_point_count=int((permutation==np.arange(len(permutation))).sum()),
                                added_model_forwards=0,added_head_forwards=0,**_auxiliary_scores(predictions,shuffled))
                        else:row['shuffled_target']='NOT_RUN'
                    del predictions
            audit['rows'].append(row);del batch,target,velocity,h9,fm;_save(r,out)
        if audit['forward_returns']!=20:raise RuntimeError('Teacher bank evaluation incomplete.')
        audit.update(status='PASS',seconds=time.perf_counter()-tick);_save(r,out);return audit
    except BaseException as error:
        audit.update(status='STOPPED',error=repr(error),seconds=time.perf_counter()-tick);_save(r,out);raise


def run_cpu_tests():
    """Bounded pure-NumPy schema/math fixtures; no real data, model, OT or grads."""
    rows=[dict(g_FM=float(i),g_A=float(i*2),g_S=float(i*4)) for i in range(1,9)]
    calibrated=_lambdas(rows)
    assert calibrated['lambda_A']==.025 and calibrated['lambda_S']==.0125
    rejected=False
    try:_lambdas([dict(g_FM=1.,g_A=0.,g_S=1.) for _ in range(8)])
    except RuntimeError:rejected=True
    assert rejected
    samples=[]
    for pair_index,(parent,noise_index) in enumerate(((0,0),(0,1),(1,0),(1,1))):
        x0=(np.arange(1008,dtype=np.float32).reshape(112,9)/1024+pair_index)
        x1=x0+np.float32(.125);known=np.arange(112)<12
        context=np.zeros_like(x0);context[known]=np.float32(pair_index+.5)
        x0[known]=0;x1[known]=0
        base=dict(x0=x0,x1=x1,u=x1-x0,t=np.asarray(.25,dtype=np.float32),y=np.asarray(112,dtype=np.int64),
            known_mask=known,valid_mask=np.ones(112,dtype=bool),context=context,
            free_gaussian_before_OT=x0[~known].reshape(100,3,3).astype(np.float64),
            free_noise_after_OT=x0[~known].reshape(100,3,3).astype(np.float64),
            free_target_before_OT=(x1[~known].reshape(100,3,3)/2).astype(np.float64),
            source_face_ids=np.arange(112,dtype=np.int64),corner_permutations=np.tile(np.arange(3,dtype=np.int64),(112,1)),
            face_permutation=np.arange(112,dtype=np.int64),known_target_ids=np.where(known,np.arange(112),-1).astype(np.int64))
        for j,t in enumerate(TIMES):samples.append(_state(base,t,parent=parent,noise_index=noise_index,pair_index=pair_index,time_index=j))
    checked=_verify_bank(samples);assert checked['pairs']==4 and checked['states']==20
    tampered=copy.deepcopy(samples);tampered[1]['x0'][20,0]+=np.float32(.125)
    tampered[1]['state_sha256']=_state_hash(tampered[1]);rejected=False
    try:_verify_bank(tampered)
    except ValueError:rejected=True
    assert rejected
    permutation=_shuffle_permutation(100,3)
    assert np.array_equal(np.sort(permutation),np.arange(100)) and np.array_equal(permutation,_shuffle_permutation(100,3))
    values=np.arange(52,dtype=np.float32).reshape(4,13)
    statistics=_array_statistics(values)
    assert np.array_equal(statistics['variance_by_dimension'],np.var(values.astype(np.float64),axis=0))
    return dict(status='PASS',fixtures=6,checks=['lambda_median_ratio','zero_denominator_rejected','twenty_state_pair_invariants',
        'changed_pair_noise_rejected','private_shuffle_reproducible','population_variance'],
        model_forwards=0,OT_calls=0,autograd_calls=0,optimizer_updates=0,real_data_reads=0)