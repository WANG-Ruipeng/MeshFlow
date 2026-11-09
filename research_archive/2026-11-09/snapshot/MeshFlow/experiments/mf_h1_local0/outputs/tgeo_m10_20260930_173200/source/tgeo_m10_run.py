"""Fixed two-arm continuation of the trained T_geo model with optional M10.

This local experiment never changes the published Native package or sampler.
No training or sampling runs at import. Each phase must be invoked explicitly.
"""
from native_t1.runtime import configure_stable_runtime
import argparse, copy, gc, hashlib, json, platform, subprocess, time, traceback
from pathlib import Path
import numpy as np
import torch

from native_t1.artifacts import atomic_json, file_sha256, state_sha256, array_hash
from native_t1.checkpoint import DEFAULT_CONFIG
from native_t1.geometry_checkpoint import load_geometry_checkpoint, validate_geometry_checkpoint
from native_t1.portable_checkpoint import config_sha256
from native_t1.data import load_archive, NativeTrainingStream, native_collate, model_inputs
from native_t1.model import native_execution
from native_t1.sampling import make_noise, clamped_sample, new_counts
from native_t1.accounting import count_ot_calls
from native_t1.training import masked_fm_loss
from .t1_interface_relations import load_source_relations, prepare_relations
from .t1_gated_factorial_loss import switch_loss, run_cpu_tests

PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parents[1]
SOURCE = PROJECT/'outputs/native_patch_20260929_010000'
INITIAL = REPO/'native_t1/runs/geometry_context_revised_20260930_153000/T_geo/additional_step500_cumulative_step2000.pt'
INITIAL_SHA = 'c52e102530fe8d50d9f90e5d9df7a5dadff89655b792a019305d373cf83a0571'
INITIAL_STATE = '1742a8aad06ed082dd7d21f91829f88d3ef118e0cfcf1d3c4cd8a6c853b3bf4e'
INPUTS = REPO/'output/t1_publish_20260930/chair_n112.npz'
INPUT_SHA = '7059671d55f2f3be8777d13e02a62255d8a0c0c7c1c1c5b4f3b24357091ba6fc'
EVAL_INPUTS = REPO/'native_t1/runs/geometry_context_revised_20260930_153000/registration/evaluation_inputs.npz'
EVAL_SHA = '2a27cf5d576272947b6944f71b7f17ab32ca1489d43d267204303c114a82248e'
PARENT_IDS = ('2af09bd8df40506c9e646678ef50aa3d','5b0e833cf2ea465e42bd82e95587d8af')
ARMS = ('Geo_continue','Geo_M10')
MODELS = ('Geo_initial',)+ARMS
SEEDS = tuple(range(9801,9809))
SCHEMA = 'tgeo_m10_followup_v1'
PUBLISHED_HEAD = '5ef35b421ad00ccd3a97867ce0e33e5f77f65b1c'


def seal(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def record(path):
    path=Path(path)
    return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=file_sha256(path))


def write(r,out):
    atomic_json(Path(out)/'results.json',r)


def event(name,**kw):
    print(json.dumps(dict(event=name,**kw),allow_nan=False),flush=True)


def release():
    gc.collect()
    if torch.cuda.is_initialized(): torch.cuda.empty_cache()


def cpu(value):
    if torch.is_tensor(value): return value.detach().cpu()
    if isinstance(value,dict): return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,tuple): return tuple(cpu(v) for v in value)
    if isinstance(value,list): return [cpu(v) for v in value]
    return value


def verify_sources(r):
    bad=[p for p,expected in r['execution_sources'].items() if file_sha256(p)!=expected['sha256']]
    if bad: raise RuntimeError('Sealed execution sources changed: '+str(bad))
    return dict(status='PASS',files=len(r['execution_sources']))


def source_records():
    # The public package is frozen. Local historical helpers are read-only.
    files=list((REPO/'native_t1').glob('*.py'))+list((REPO/'models').glob('*.py'))
    files += [p for p in PROJECT.glob('*.py') if p.name != 'tgeo_m10_report.py'] + [DEFAULT_CONFIG, REPO/'utils/ot_utils.py']
    return {str(p):record(p) for p in files}


def verify_assets(r):
    for name, expected in r["assets"].items():
        if not isinstance(expected, dict) or "path" not in expected or "sha256" not in expected:
            continue
        actual = record(expected["path"])
        if actual["sha256"] != expected["sha256"] or actual["bytes"] != expected["bytes"]:
            raise RuntimeError("Registered asset changed: " + name)
    return dict(status="PASS", assets=len(r["assets"]))


def make_stream(r):
    archive=load_archive(r['assets']['inputs']['path'],r['assets']['inputs']['sha256'])
    state=copy.deepcopy(r['stream_start'])
    stream=NativeTrainingStream(archive,seed=state['seed'])
    stream.load_state_dict(state)
    if seal(stream.state_dict())!=r['protocol']['training']['initial_stream_sha256']:
        raise RuntimeError('Input RNG restore changed')
    return stream


def get_relations(r):
    for name in ("relation_inputs", "relation_manifest"):
        if file_sha256(r["assets"][name]["path"]) != r["assets"][name]["sha256"]:
            raise RuntimeError("Registered relation asset changed: " + name)
    return load_source_relations(r['assets']['relation_inputs']['path'],
        expected_sha256=r['assets']['relation_inputs']['sha256'],
        source_manifest_path=r['assets']['relation_manifest']['path'])


def _attach_ledger(model,r):
    def before(module,args):
        counts=r['counts']; stage=r['active_stage']
        if counts['model_forward_attempts']>=11702:
            raise RuntimeError('Hard model-forward cap reached')
        counts['model_forward_attempts']+=1
        counts['forward_by_stage'][stage]=counts['forward_by_stage'].get(stage,0)+1
    def after(module,args,result):
        r['counts']['model_forward_returns']+=1
    model.register_forward_pre_hook(before)
    model.register_forward_hook(after)


def load_model(r,name,trainable=False,initial=False):
    if name not in MODELS: raise ValueError('Unknown model')
    model,audit=load_geometry_checkpoint(INITIAL,DEFAULT_CONFIG)
    if audit['checkpoint_sha256']!=INITIAL_SHA or audit['state_sha256']!=INITIAL_STATE:
        raise RuntimeError('Wrong common trained Geo initialization')
    if name!='Geo_initial' and not initial:
        info=r['arms'][name]['checkpoints']['500']
        if file_sha256(info['path'])!=info['sha256']: raise RuntimeError('New endpoint changed')
        payload=torch.load(info['path'],map_location='cpu',weights_only=True,mmap=True)
        expected=dict(schema=SCHEMA,arm=name,base_cumulative_updates=2000,additional_updates=500,
            cumulative_updates=2500,initial_state_sha256=INITIAL_STATE,
            loss_switches=dict(FC=name=='Geo_M10',FF=False),
            protocol_sha256=r['protocol_sha256'],optimizer_step_in_progress=False,batch_in_progress=False)
        if any(payload.get(k)!=v for k,v in expected.items()):
            raise RuntimeError('New branch identity differs')
        if payload['context']!=model.context_encoder.metadata():
            raise RuntimeError('Geo architecture or original encoder seed differs')
        keys=model.load_state_dict(payload['model'],strict=True)
        if keys.missing_keys or keys.unexpected_keys or state_sha256(model)!=info['state_sha256']:
            raise RuntimeError('Strict endpoint tensor identity differs')
        del payload
        audit=dict(audit,endpoint=info,loaded_state_sha256=info['state_sha256'])
    model.set_trainable(trainable).eval()
    if sum(p.numel() for p in model.parameters())!=130401155:
        raise RuntimeError('Architecture changed')
    if sum(p.numel() for p in model.context_encoder.parameters())!=149888:
        raise RuntimeError('Geo architecture changed')
    _attach_ledger(model,r)
    return model,audit


def save_checkpoint(r,out,arm,model,optimizer,stream,result,step,interrupted=False):
    folder=Path(out)/arm;folder.mkdir(exist_ok=True)
    if not interrupted and (step not in (250,500) or step!=result['completed_updates']):
        raise RuntimeError('Only registered250/500 checkpoints allowed')
    suffix='interrupted' if interrupted else f'additional_step{step}_cumulative_step{2000+step}'
    path=folder/(suffix+'.pt')
    if path.exists() or path.with_suffix('.pt.partial').exists(): raise FileExistsError(path)
    tick=time.perf_counter()
    rng=torch.get_rng_state().clone()
    cuda_rng=[v.clone() for v in torch.cuda.get_rng_state_all()]
    stream_state=copy.deepcopy(stream.state_dict())
    payload=dict(schema=SCHEMA,arm=arm,model=cpu(model.state_dict()),optimizer=cpu(optimizer.state_dict()),
        context=model.context_encoder.metadata(),initial_checkpoint=r['assets']['initial'],
        initial_state_sha256=INITIAL_STATE,base_cumulative_updates=2000,
        additional_updates=result['completed_updates'],cumulative_updates=2000+result['completed_updates'],
        optimizer_step_in_progress=result.get('optimizer_step_in_progress',False),
        batch_in_progress=result.get('batch_in_progress',False),
        protocol_sha256=r['protocol_sha256'],stream_state=stream_state,
        loss_switches=dict(FC=arm=='Geo_M10',FF=False),gate_threshold=.5,FC_weight=.1 if arm=='Geo_M10' else None,
        FC_denominator='All effective8 valid FC groups, including inactive samples',
        config_sha256=config_sha256(DEFAULT_CONFIG),input_sha256=INPUT_SHA,
        optimizer_reset_at_start=True,model_state_sha256=state_sha256(model),
        interrupted=interrupted)
    torch.save(payload,path.with_suffix('.pt.partial'))
    path.with_suffix('.pt.partial').replace(path)
    digest=payload['model_state_sha256'];del payload
    if not torch.equal(rng,torch.get_rng_state()) or not all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all())):
        raise RuntimeError('Saving advanced model RNG')
    if seal(stream_state)!=seal(stream.state_dict()): raise RuntimeError('Saving advanced input stream')
    info=dict(**record(path),state_sha256=digest,additional_updates=result['completed_updates'],
              cumulative_updates=2000+result['completed_updates'],schema=SCHEMA,arm=arm,
              save_rng_unchanged=True,seconds=time.perf_counter()-tick)
    result['checkpoints']['interrupted' if interrupted else str(step)]=info
    write(r,out)
    return info


def initialize(out):
    if out.exists(): raise FileExistsError(out)
    current=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    if current!=PUBLISHED_HEAD: raise RuntimeError('Published baseline commit changed')
    dirty=subprocess.check_output(['git','-c','core.autocrlf=true','status','--short'],cwd=REPO,text=True)
    if dirty.strip(): raise RuntimeError('Published working tree has unregistered changes: '+dirty)
    for p in (REPO.parent/'AGENTS.md',REPO/'AGENTS.md',PROJECT.parent/'AGENTS.md',PROJECT/'AGENTS.md'):
        if p.exists(): raise RuntimeError('Review applicable new AGENTS.md before proceeding: '+str(p))
    from .tgeo_m10_evaluation import audit_new_seeds, load_bank
    seed_audit=audit_new_seeds(SEEDS,out)
    if seed_audit['status']!='PASS': raise RuntimeError('Evaluation noises already used')
    assets=dict(initial=record(INITIAL),inputs=record(INPUTS),evaluation_inputs=record(EVAL_INPUTS),
                relation_inputs=record(SOURCE/'inputs.npz'),relation_manifest=record(SOURCE/'results.json'))
    if assets['initial']['sha256']!=INITIAL_SHA or assets['inputs']['sha256']!=INPUT_SHA or assets['evaluation_inputs']['sha256']!=EVAL_SHA:
        raise RuntimeError('Fixed asset identity differs')
    payload=torch.load(INITIAL,map_location='cpu',weights_only=True,mmap=True)
    validate_geometry_checkpoint(payload,config_sha256(DEFAULT_CONFIG))
    if payload['cumulative_updates']!=2000 or payload['model_state_sha256']!=INITIAL_STATE or payload['input_sha256']!=INPUT_SHA:
        raise RuntimeError('Wrong trained Geo parent identity')
    state=copy.deepcopy(payload['stream_state']);del payload
    if (state['seed'],state['batch_index'],state['sample_index'],state['arm'])!=(1010,500,4000,'Native_correct'):
        raise RuntimeError('Unexpected archived Geo training stream; do not silently reseed')
    archive=load_archive(INPUTS,INPUT_SHA)
    relations=load_source_relations(SOURCE/'inputs.npz',expected_sha256=assets['relation_inputs']['sha256'])
    old=load_archive(SOURCE/'inputs.npz',assets['relation_inputs']['sha256'])
    equal=[]
    for key,case in archive.cases.items():
        reference=old.cases[key]
        for attr in ('constraints','full_target','free_target','source_face_ids','free_source_ids','pre_ot_free'):
            a,b=getattr(case,attr),getattr(reference,attr)
            if a is None or b is None or a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes()!=b.tobytes():
                raise RuntimeError('Portable and source-labeled training arrays differ: '+str((key,attr)))
        equal.append(dict(parent=key[0],K=key[1],all_training_arrays_bitwise_equal=True))
    bank=load_bank(EVAL_INPUTS,EVAL_SHA)
    out.mkdir(parents=True);(out/'raw').mkdir()
    protocol=dict(schema=SCHEMA,initial=assets['initial'],initial_state_sha256=INITIAL_STATE,
        published_head=current,models=list(MODELS),training=dict(arms=list(ARMS),updates_each=500,total_update_cap=1000,
        initial_cumulative_updates=2000,final_cumulative_updates=2500,restore_input_rng=True,
        initial_stream_sha256=seal(state),seed=1010,effective_batch=8,microbatch=1,K=[4,8,12],
        parents=list(PARENT_IDS),optimizer='Fresh AdamW each arm, no momentum restore',lr=1e-5,betas=[.9,.95],
        weight_decay=0.,grad_clip=1.,full_model_trainable=True,modules_eval=True,
        time_distribution='Original logit-normal',OT='Original free-only nested OT, reorders Z only',
        FC_weight=.1,gate='stored_FP32_t>=.5',FC_normalization='All8 valid groups including inactive samples',
        FF=False,new_modules=False,encoder_reinitialized=False,checkpoints=[250,500],log_steps=[1,100,250,500],
        scheduler=False,EMA=False,condition_dropout=False),
        evaluation=dict(seeds=list(SEEDS),original_patch_seeds=list(SEEDS),other_patch_seeds=list(SEEDS[:2]),
        patches=[0,1,2],patch_selection='Read unchanged prior T_geo bank, no reselection',
        models=list(MODELS),formal_rollouts=72,replay_cap=2,planned_replays=1,steps=50,N=112,K=12,free=100,
        guidance=False,postprocessing=False,new_noise_confirmation_only=True,new_objects=False),
        budget=dict(optimizer_updates=1000,training_forwards=8000,training_backwards=8000,
                    gate_forwards=2,gate_backwards=2,formal_rollouts=72,replay_cap=2,max_model_forwards=11702),
        comparison='Geo_M10 minus Geo_continue isolates added loss; Geo_continue minus Geo_initial measures extra training',
        interpretation='Single training stream and two seen parents. New noises and previously fixed extra patches are not unseen-object generalization.',
        stop='No extension, no weight/gate search, no new modules or Phase3')
    r=dict(status='INITIALIZED',active_stage='assets',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        protocol=protocol,protocol_sha256=seal(protocol),assets=assets,stream_start=state,
        asset_checks=dict(portable_source_arrays=equal,source_relations=relations.audit,seed_audit=seed_audit),
        counts=dict(model_forward_attempts=0,model_forward_returns=0,forward_by_stage={},
        backward_attempts=0,backward_returns=0,optimizer_step_attempts=0,optimizer_updates=0,
        training_OT_calls=0,training_OT_returns=0,gate_OT_calls=0,gate_OT_returns=0,
        sampling_forward_attempts=0,sampling_forward_returns=0,rollout_attempts=0,complete_rollouts=0,
        formal_rollouts=0,replay_rollouts=0,euler_updates=0),
        arms={},generated=[],replays=[],sampling_attempts=[],gates={},timing={},
        environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
            numpy=np.__version__,GPU=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used','--format=csv,noheader'],text=True).strip()),
        execution_sources=source_records(),public_git_status_before=dirty)
    # Fixed evaluation noises are separate CPU generators, not training RNGs.
    noise_arrays={f'noise_seed{s}':make_noise(s).numpy() for s in SEEDS}
    np.savez_compressed(out/'evaluation_noise.npz',**noise_arrays)
    r['assets']['evaluation_noise']=record(out/'evaluation_noise.npz')
    atomic_json(out/'protocol.json',protocol)
    write(r,out)
    return r


def gates(r,out):
    if r['status']!='INITIALIZED': raise RuntimeError('No gate reruns')
    verify_sources(r)
    r['active_stage']='gates';write(r,out)
    tick=time.perf_counter()
    r['gates']['CPU_M10_formula']=run_cpu_tests()
    if r['gates']['CPU_M10_formula']['status']!='PASS': raise RuntimeError('CPU formula gate failed')
    stream=make_stream(r); ot={}
    try:
        with count_ot_calls(ot): samples=stream.next_effective_batch()
    finally:
        r['counts']['gate_OT_calls']+=ot.get('OT_calls',0)
        r['counts']['gate_OT_returns']+=ot.get('OT_returns',0)
        write(r,out)
    relations=get_relations(r)
    labels=[relations.prepare_sample_labels(s) for s in samples]
    index=next(i for i,label in enumerate(labels) if label['FC_count']>0)
    batch=native_collate([samples[index]],'cuda',t_override=.75)
    batch['context'].requires_grad_(True)
    prepared=prepare_relations(labels[index],'cuda')
    prepared['fc_anchors'].requires_grad_(True)
    fc_den=sum(v['FC_count'] for v in labels);ff_den=sum(v['FF_count'] for v in labels)
    fm_den=9*sum(int((s['valid_mask']&~s['known_mask']).sum()) for s in samples)
    model,audit=load_model(r,'Geo_initial',trainable=True)
    before=state_sha256(model);known_before=batch['xt'][batch['known_mask']].clone()
    velocities=[];gradient_sets=[];losses=[];all_gradient_checks=[]
    names={n:p for n,p in model.named_parameters() if n.startswith('context_encoder.')}
    for enabled in (False,True):
        model.zero_grad(set_to_none=True)
        with native_execution(model,coordinates=batch['xt']) as runtime:
            velocity=model(*model_inputs(batch)).float()
            fm=masked_fm_loss(velocity,batch['u'],batch['valid_mask'],batch['known_mask'],denominator=fm_den)
            if enabled:
                total,parts,gate=switch_loss(fm,batch,velocity,prepared,fc_den,ff_den,True,False)
                if parts['FF'] is not None or not gate['active_samples']: raise RuntimeError('Wrong gate')
            else: total=fm;parts=dict(FC=None)
            if not bool(torch.isfinite(total)): raise RuntimeError('Nonfinite gate loss')
            r['counts']['backward_attempts']+=1
            total.backward()
            r['counts']['backward_returns']+=1
        family = {}
        for n, parameter in model.named_parameters():
            group = "encoder" if n.startswith("context_encoder.") else "role" if n=="role_embedding" else "backbone"
            value = family.setdefault(group, dict(parameters_with_gradient=0, gradient_squared=0.))
            if parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError("Nonfinite full-model preflight gradient: " + n)
                value["parameters_with_gradient"] += parameter.numel()
                value["gradient_squared"] += float(parameter.grad.detach().float().square().sum())
        all_gradient_checks.append({k:dict(parameters_with_gradient=v["parameters_with_gradient"],
                                          gradient_l2=v["gradient_squared"]**.5) for k,v in family.items()})
        velocities.append(velocity.detach().cpu())
        gradient_sets.append({n:p.grad.detach().clone() for n,p in names.items() if p.grad is not None})
        losses.append(dict(FM=float(fm.detach()),FC=None if parts['FC'] is None else float(parts['FC'].detach()),total=float(total.detach())))
    if not torch.equal(*velocities): raise RuntimeError('Loss selection changed same-input velocity')
    if set(gradient_sets[0])!=set(names) or set(gradient_sets[1])!=set(names):
        raise RuntimeError('Missing Geo parameter gradients')
    increments={}
    for name in names:
        a,b=gradient_sets[0][name],gradient_sets[1][name]
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()): raise RuntimeError('Nonfinite encoder gradient')
        value=increments.setdefault(name.split('.')[1],dict(FM_squared=0.,M10_squared=0.,increment_squared=0.))
        value['FM_squared']+=float(a.square().sum());value['M10_squared']+=float(b.square().sum())
        value['increment_squared']+=float((b-a).square().sum())
    if not all(v['increment_squared']>0 for v in increments.values()):
        raise RuntimeError('FC did not reach every existing Geo module group on the fixed active gate')
    if batch['context'].grad is not None or prepared['fc_anchors'].grad is not None:
        raise RuntimeError('C supervision anchor accepted a gradient')
    if not torch.equal(known_before,batch['xt'][batch['known_mask']]) or state_sha256(model)!=before:
        raise RuntimeError('Gate changed C or model parameters')
    model.zero_grad(set_to_none=True)
    r['gates']['GPU_M10']=dict(status='PASS',forwards=2,backwards=2,optimizer_updates=0,
        same_velocity_bitwise=True,model_state_unchanged=True,C_unchanged=True,C_anchor_gradient_absent=True,
        t=.75,FM_denominator=fm_den,FC_denominator=fc_den,losses=losses,
        encoder_gradients={k:{n.replace('_squared','_l2'):v**.5 for n,v in values.items()} for k,values in increments.items()},
        runtime=runtime,load=audit,gate_sample_index=index,all_parameter_gradient_checks=all_gradient_checks,
        formal_stream_not_advanced=True)
    del model,gradient_sets,velocities,names,stream
    release()
    r['timing']['preflight_seconds']=time.perf_counter()-tick
    r['status']='PREFLIGHT_PASS';write(r,out);event('preflight_pass',counts=r['counts'])


def sample_all(r,out):
    if any(r['arms'].get(a,{}).get('status')!='PASS' for a in ARMS):
        raise RuntimeError('Both fixed endpoints must finish first')
    if r['generated'] or r['replays']: raise RuntimeError('No automatic sampling reruns')
    verify_sources(r)
    from .tgeo_m10_evaluation import load_bank
    banks=load_bank(r['assets']['evaluation_inputs']['path'],r['assets']['evaluation_inputs']['sha256'])
    if file_sha256(r["assets"]["evaluation_noise"]["path"]) != r["assets"]["evaluation_noise"]["sha256"]:
        raise RuntimeError("Registered evaluation noises changed")
    with np.load(r['assets']['evaluation_noise']['path'],allow_pickle=False) as value:
        noises={s:value[f'noise_seed{s}'].copy() for s in SEEDS}
    if any(n.dtype != np.float32 or n.shape != (1,112,9) or not np.isfinite(n).all() for n in noises.values()):
        raise RuntimeError("Invalid fixed evaluation Gaussian")
    tick=time.perf_counter();model_seconds=0.
    for name in MODELS:
        r['active_stage']='sampling_'+name
        model,loaded=load_model(r,name)
        before=state_sha256(model)
        for patch in (0,1,2):
            for parent in (0,1):
                C=banks[patch]['cases_by_key'][parent,12]['constraints']
                for seed in (SEEDS if patch==0 else SEEDS[:2]):
                    row=_sample_one(r,out,model,name,parent,patch,seed,noises[seed],C)
                    model_seconds+=row['sampling']['seconds']
        if name=='Geo_M10':
            reference=next(row for row in r['generated'] if row['model']==name and row['parent']==0 and row['patch']==0 and row['seed']==SEEDS[0])
            replay=_sample_one(r,out,model,name,0,0,SEEDS[0],noises[SEEDS[0]],banks[0]['cases_by_key'][0,12]['constraints'],reference=reference)
            model_seconds+=replay['sampling']['seconds']
        if state_sha256(model)!=before: raise RuntimeError('Sampling changed trained weights')
        del model;release()
        event('sample_model_complete',model=name,formal=r['counts']['formal_rollouts'])
    if r['counts']['formal_rollouts']!=72 or r['counts']['replay_rollouts']!=1:
        raise RuntimeError('Unexpected completed sampling matrix')
    r['timing']['sampling_total_seconds']=time.perf_counter()-tick
    r['timing']['model_sampling_seconds']=model_seconds
    r['status']='SAMPLING_PASS';write(r,out)


def _sample_one(r,out,model,name,parent,patch,seed,noise,C,reference=None):
    counts=r['counts']
    if counts['rollout_attempts']>=74 or (reference and counts['replay_rollouts']>=2):
        raise RuntimeError('Registered rollout cap exceeded')
    counts['rollout_attempts']+=1;write(r,out)
    sc=new_counts()
    try:
        raw,path,stats=clamped_sample(model,torch.from_numpy(noise.copy()).cuda(),torch.from_numpy(C.copy()).cuda(),sc)
    finally:
        counts['sampling_forward_attempts']+=sc['model_forward_attempts']
        counts['sampling_forward_returns']+=sc['model_forward_returns']
        counts['complete_rollouts']+=sc['complete_rollouts']
        counts['euler_updates']+=sc['euler_updates']
        r['sampling_attempts'].append(dict(model=name,parent=parent,patch=patch,seed=seed,replay=reference is not None,counts=sc))
        write(r,out)
    if not np.array_equal(path[:,:12].view(np.uint32),np.broadcast_to(C.reshape(1,12,9),(51,12,9)).copy().view(np.uint32)):
        raise RuntimeError('C not preserved in all51 states')
    if not np.array_equal(path[0,12:].view(np.uint32),noise[0,12:].view(np.uint32)):
        raise RuntimeError('Free initial Gaussian changed')
    row=dict(model=name,parent=parent,patch=patch,seed=seed,K=12,phase='new_noise_confirmation',
             sampling=stats,C_all51_bitwise=True,noise_sha256=array_hash(noise),C_sha256=array_hash(C),
             output_sha256=array_hash(raw),path_sha256=array_hash(path))
    if reference:
        with np.load(reference['raw_path']) as old:
            if path.tobytes()!=old['path'].tobytes() or raw.tobytes()!=old['output'].tobytes():
                raise RuntimeError('Replay differs from saved trajectory')
        row.update(reference_path=reference['raw_path'],all51_bitwise_equal=True)
        r['replays'].append(row);counts['replay_rollouts']+=1
    else:
        target=Path(out)/'raw'/f'{name}_P{parent}_patch{patch}_seed{seed}.npz'
        if target.exists(): raise FileExistsError(target)
        np.savez_compressed(target,output=raw,path=path,initial_gaussian=noise)
        row.update(raw_path=str(target),raw_sha256=file_sha256(target))
        r['generated'].append(row);counts['formal_rollouts']+=1
    write(r,out)
    return row


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--phase',choices=('preflight','train','sample','evaluate','report'),required=True)
    args=parser.parse_args(argv);out=args.out.resolve();r=None
    try:
        if args.phase=='preflight':
            r=initialize(out);gates(r,out)
        else:
            r=json.loads((out/'results.json').read_text())
            if r['protocol_sha256']!=seal(r['protocol']): raise RuntimeError('Registered protocol changed')
            verify_sources(r)
            verify_assets(r)
            if args.phase=='train':
                if r['status']!='PREFLIGHT_PASS': raise RuntimeError('Training requires passed preregistered gates, no retry')
                from .tgeo_m10_train import train_arm
                relations=get_relations(r)
                for arm in ARMS: train_arm(r,out,relations,arm)
                r['status']='TRAINING_PASS';write(r,out)
            elif args.phase=='sample':
                if r['status']!='TRAINING_PASS': raise RuntimeError('Sampling requires completed training')
                sample_all(r,out)
            elif args.phase=='evaluate':
                if r['status']!='SAMPLING_PASS': raise RuntimeError('Evaluate only the completed frozen matrix')
                from .tgeo_m10_evaluation import evaluate
                r['active_stage']='offline_evaluation';write(r,out)
                tick=time.perf_counter();evaluate(r,out)
                r['timing']['offline_evaluation_seconds']=time.perf_counter()-tick
                r['status']='PASS_EXECUTION';write(r,out)
            else:
                if r['status']!='PASS_EXECUTION': raise RuntimeError('Final report requires complete execution')
                from .tgeo_m10_report import write_report
                r['report_source'] = record(PROJECT/'tgeo_m10_report.py')
                r['report'] = write_report(r,out)
                write(r,out)
    except BaseException as error:
        if r is not None:
            r.update(status='STOPPED',error=repr(error),traceback=traceback.format_exc(),automatic_retry=False)
            write(r,out)
        raise


if __name__=='__main__':
    main()
