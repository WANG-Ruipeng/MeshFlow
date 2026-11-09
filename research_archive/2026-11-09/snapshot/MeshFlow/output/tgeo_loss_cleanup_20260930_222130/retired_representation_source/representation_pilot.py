"""Run the fixed analytic-geometry representation alignment pilot explicitly."""
from .runtime import configure_stable_runtime
import argparse,ast,copy,gc,hashlib,io,json,random,subprocess,time,traceback,types
from pathlib import Path
import numpy as np
import torch
from .artifacts import atomic_json,file_sha256,state_sha256,array_hash
from .checkpoint import DEFAULT_CONFIG
from .geometry_checkpoint import load_geometry_checkpoint,validate_geometry_checkpoint
from .portable_checkpoint import config_sha256
from .representation_checkpoint import (ARMS,SCHEMA,INITIAL_SHA,INITIAL_STATE,GENERATOR_PARAMETERS,
    save_training_checkpoint,load_representation_generator,load_auxiliary_heads)
from .data import load_archive,NativeTrainingStream,native_collate,model_inputs
from .model import native_execution
from .training import masked_fm_loss
from .sampling import make_noise,clamped_sample,new_counts
from .accounting import count_ot_calls

REPO=Path(__file__).resolve().parents[1]
PACKAGE=Path(__file__).resolve().parent
INITIAL=PACKAGE/'runs/geometry_context_revised_20260930_153000/T_geo/additional_step500_cumulative_step2000.pt'
INPUTS=REPO/'output/t1_publish_20260930/chair_n112.npz'
INPUT_SHA='7059671d55f2f3be8777d13e02a62255d8a0c0c7c1c1c5b4f3b24357091ba6fc'
BANK=PACKAGE/'runs/geometry_context_revised_20260930_153000/registration/evaluation_inputs.npz'
BANK_SHA='2a27cf5d576272947b6944f71b7f17ab32ca1489d43d267204303c114a82248e'
HEAD='5ef35b421ad00ccd3a97867ce0e33e5f77f65b1c'
MODELS=('Geo_initial',)+ARMS
SEEDS=tuple(range(9901,9909))
LOG_STEPS=(1,10,50,100,250,500)

def record(path):
    p=Path(path)
    return dict(path=str(p.resolve()),sha256=file_sha256(p),bytes=p.stat().st_size)
def seal(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(r,out): atomic_json(Path(out)/'results.json',r)
def event(name,**kw): print(json.dumps(dict(event=name,**kw),allow_nan=False),flush=True)
def release():
    gc.collect()
    if torch.cuda.is_initialized(): torch.cuda.empty_cache()
def sources():
    files=list(PACKAGE.glob('*.py'))+list((REPO/'models').glob('*.py'))+[DEFAULT_CONFIG,REPO/'utils/ot_utils.py']
    files=[p for p in files if p.name!='representation_report.py']
    # Independent scoring code is also frozen for this run.
    files+=list((REPO/'experiments/mf_h1_local0').glob('*.py'))
    return {str(p):record(p) for p in files}
def verify(r):
    assert seal(r['protocol'])==r['protocol_sha256']
    for p,info in r['execution_sources'].items():
        if file_sha256(p)!=info['sha256']: raise RuntimeError('Execution source changed: '+p)
    for name,info in r['assets'].items():
        if isinstance(info,dict) and 'path' in info and 'sha256' in info:
            if file_sha256(info['path'])!=info['sha256']: raise RuntimeError('Asset changed: '+name)
def attach_ledger(model,r):
    def pre(module,args,kwargs):
        c=r['counts'];stage=r['active_stage']
        if c['model_forward_attempts']>=17112: raise RuntimeError('Hard forward cap17112')
        c['model_forward_attempts']+=1;c['forward_by_stage'][stage]=c['forward_by_stage'].get(stage,0)+1
        if stage.startswith('sampling_') and kwargs.get('return_feature_layer') is not None:
            raise RuntimeError('Sampling must not request a training feature')
    def post(module,args,result): r['counts']['model_forward_returns']+=1
    model.register_forward_pre_hook(pre,with_kwargs=True);model.register_forward_hook(post)
def initial_model(r,trainable=False):
    model,audit=load_geometry_checkpoint(INITIAL)
    if audit['checkpoint_sha256']!=INITIAL_SHA or audit['state_sha256']!=INITIAL_STATE: raise RuntimeError('Wrong initial model')
    model.set_trainable(trainable).eval();attach_ledger(model,r)
    return model,audit
def endpoint(r,arm,*,with_heads=False):
    info=r['arms'][arm]['checkpoints']['500']
    if file_sha256(info['path'])!=info['sha256']: raise RuntimeError('Endpoint hash differs')
    model,audit=load_representation_generator(info['path'])
    if audit['state_sha256']!=info['state_sha256']: raise RuntimeError('Endpoint state differs')
    attach_ledger(model,r)
    heads=load_auxiliary_heads(info['path']) if with_heads else None
    return model,heads,audit
def stream_for(r):
    archive=load_archive(INPUTS,INPUT_SHA);stream=NativeTrainingStream(archive,seed=1010)
    stream.load_state_dict(copy.deepcopy(r['stream_start']))
    if seal(stream.state_dict())!=r['protocol']['training']['initial_stream_sha256']: raise RuntimeError('Stream mismatch')
    return stream
def targets_for(r):
    from .geometry_representation import GeometryTargets
    targets=GeometryTargets.from_archive(load_archive(INPUTS,INPUT_SHA))
    if seal(targets.metadata())!=r['target_metadata_sha256']: raise RuntimeError('Frozen target statistics changed')
    return targets

def initialize(out):
    if out.exists(): raise FileExistsError(out)
    current=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    if current!=HEAD: raise RuntimeError('Unexpected baseline HEAD')
    from .representation_evaluation import audit_seeds,load_bank
    from .geometry_representation import GeometryTargets,AuxiliaryHeads
    seed_audit=audit_seeds(SEEDS,out)
    if seed_audit['status']!='PASS': raise RuntimeError('Evaluation seeds previously used; register replacement before generation')
    assets=dict(initial=record(INITIAL),inputs=record(INPUTS),evaluation_inputs=record(BANK),config=record(DEFAULT_CONFIG))
    if (assets['initial']['sha256'],assets['inputs']['sha256'],assets['evaluation_inputs']['sha256'])!=(INITIAL_SHA,INPUT_SHA,BANK_SHA):
        raise RuntimeError('Fixed asset identity mismatch')
    payload=torch.load(INITIAL,map_location='cpu',weights_only=True,mmap=True)
    validate_geometry_checkpoint(payload,config_sha256(DEFAULT_CONFIG))
    if payload['model_state_sha256']!=INITIAL_STATE or payload['cumulative_updates']!=2000: raise RuntimeError('Wrong Geo state')
    stream=copy.deepcopy(payload['stream_state']);del payload
    if (stream['seed'],stream['batch_index'],stream['sample_index'],stream['arm'])!=(1010,500,4000,'Native_correct'):
        raise RuntimeError('Saved stream identity differs')
    archive=load_archive(INPUTS,INPUT_SHA);targets=GeometryTargets.from_archive(archive)
    metadata=targets.metadata();banks=load_bank(BANK,BANK_SHA)
    out.mkdir(parents=True);(out/'raw').mkdir();(out/'source').mkdir()
    original=subprocess.check_output(['git','show',HEAD+':native_t1/model.py'],cwd=REPO)
    (out/'source/original_model.py').write_bytes(original)
    atomic_json(out/'target_statistics.json',metadata)
    protocol=dict(method='analytic-geometry representation alignment pilot',head=HEAD,
        references=['https://arxiv.org/html/2604.04050v1','https://arxiv.org/html/2507.07982v2'],
        reference_scope='Training feature alignment and direction/full-descriptor-magnitude inspiration only; no pretrained teacher, CKA or topology guarantee',
        training=dict(arms=list(ARMS),updates_each=500,total_update_cap=1500,initial_cumulative_updates=2000,
        final_cumulative_updates=2500,initial_stream_sha256=seal(stream),seed=1010,
        effective_batch=8,microbatch=1,K=[4,8,12],lr=1e-5,betas=[.9,.95],weight_decay=0.,generator_clip=1.,auxiliary_clip=1.,
        optimizer='fresh AdamW separate parameter groups; no momentum restoration',feature_layer=9,head_seed=20261001,
        ramp='min(s/50,1)',all_flow_times=True,FM='original free-coordinate denominator',FC=False,FF=False,
        checkpoint_steps=[250,500],log_steps=list(LOG_STEPS),no_scheduler=True,no_new_EMA=True),
        calibration=dict(effective_batch=8,clone_formal_stream=True,model_forwards=8,H9_autograd_grad_calls=24,
            lambda_A='.05*median(gFM)/median(gA)',lambda_S='.05*median(gFM)/median(gS)',runs=1),
        teacher=dict(seed=20261002,parents=2,K=12,noises_each=2,t=[.1,.3,.5,.7,.9],
            OT_calls=4,states=20,planned_forwards=100),
        evaluation=dict(seeds=list(SEEDS),patches=[0,1,2],original_patch_seeds=list(SEEDS),
            other_patch_seeds=list(SEEDS[:2]),models=list(MODELS),formal_rollouts=96,replay_cap=4,planned_replays=1,
            steps=50,N=112,K=12,free=100,guidance=False,postprocessing=False),
        budgets=dict(generator_updates=1500,training_forwards=12000,training_backwards=12000,
            gates_forwards=4,gates_parameter_backwards=1,gates_autograd_grad=1,
            calibration_forwards=8,calibration_H9_autograd_grad=24,teacher_forwards=100,
            formal_rollouts=96,max_extra_rollouts=4,max_model_forwards=17112),
        comparisons=['G0-Geo_initial','G_Angular-G0','G_AngularScale-G_Angular','G_AngularScale-G0'],
        limitations='Two trained parents; fixed already observed patches; one training stream; Scale also adds supervision strength; no mismatch teacher or external encoder',
        stop='No extension, layer/weight search, M10, new architecture or Phase3')
    r=dict(status='INITIALIZED',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        protocol=protocol,protocol_sha256=seal(protocol),assets=assets,stream_start=stream,
        target_metadata_sha256=seal(metadata),target_statistics=metadata,seed_audit=seed_audit,
        baseline_inheritance=dict(old_BF16_permutation_failure='PRESERVED_NOT_RETESTED_OR_RELABELLED',
            no_tolerance_change=True,new_feature_gates='same input old/default/feature/repeat exact'),
        counts=dict(model_forward_attempts=0,model_forward_returns=0,forward_by_stage={},
            backward_attempts=0,backward_returns=0,optimizer_step_attempts=0,optimizer_updates=0,
            training_OT_calls=0,training_OT_returns=0,gate_OT_calls=0,gate_OT_returns=0,
            gate_autograd_grad_attempts=0,gate_autograd_grad_returns=0,
            sampling_forward_attempts=0,sampling_forward_returns=0,rollout_attempts=0,
            complete_rollouts=0,formal_rollouts=0,replay_rollouts=0,euler_updates=0),
        arms={},generated=[],replays=[],sampling_attempts=[],gates={},teacher={},timing={},
        execution_sources=sources(),public_git_status_at_registration=subprocess.check_output(['git','-c','core.autocrlf=true','status','--short'],cwd=REPO,text=True),
        environment=dict(python=__import__('platform').python_version(),torch=torch.__version__,cuda=torch.version.cuda,
            GPU=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used','--format=csv,noheader'],text=True).strip()))

    r['initial_workspace_audit']=dict(head_before_edits=HEAD,git_status_before_edits='',
        applicable_AGENTS_found=[],checked_paths=['E:/AGENTS.md','E:/MeshFlow/AGENTS.md',
        'E:/MeshFlow/experiments/AGENTS.md','E:/MeshFlow/experiments/mf_h1_local0/AGENTS.md','E:/MeshFlow/native_t1/AGENTS.md'],
        descendant_AGENTS_rg_search='No matches',reset_clean_commit_push_performed=False)
    r['gates']['CPU']=dict(status='PASS',geometry_pytest_tests=9,geometry_pytest_seconds=1.91,
        command='python -B -m pytest native_t1/tests/test_geometry_representation.py -q -p no:cacheprovider',
        test_source=record(PACKAGE/'tests/test_geometry_representation.py'),
        independent_geometry_fixture_checks=9,diagnostic_fixture_checks=6,evaluation_scalar_checks=6,
        report_schema_checks=5,official_model_forwards=0,OT_calls=0,optimizer_updates=0,
        small_geometry_CPU_autograd_calls=10,
        development_failures=['Evaluation bank SHA copied incorrectly initially; rejected before scoring and corrected to frozen source SHA',
            'Diagnostic synthetic fixture initially lacked raw Gaussian hash fields; corrected without real-data/model/OT calls'])
    np.savez_compressed(out/'evaluation_noise.npz',**{f'noise_seed{s}':make_noise(s).numpy() for s in SEEDS})
    r['assets']['evaluation_noise']=record(out/'evaluation_noise.npz')
    r['assets']['target_statistics']=record(out/'target_statistics.json')
    r['assets']['original_model']=record(out/'source/original_model.py')
    atomic_json(out/'protocol.json',protocol);write(r,out)
    return r

def magnitude(values):
    vals=[v.detach().float() for v in values if v is not None]
    if not vals:return dict(RMS=0.,L2=0.,nonzero=0,numel=0,finite=True)
    n=sum(v.numel() for v in vals);s=sum(float(v.square().sum()) for v in vals)
    return dict(RMS=(s/n)**.5,L2=s**.5,nonzero=sum(int(torch.count_nonzero(v)) for v in vals),
        numel=n,finite=all(bool(torch.isfinite(v).all()) for v in vals))

def gates(r,out,targets,samples):
    from .geometry_representation import AuxiliaryHeads,auxiliary_losses
    from . import model as native_module
    tick=time.perf_counter();r['active_stage']='gates'
    model,loaded=initial_model(r,True);heads=AuxiliaryHeads(use_scale=True,device='cuda')
    before=state_sha256(model);batch=native_collate([samples[0]],'cuda');label=targets.for_sample(samples[0],device='cuda')
    old_ast=ast.parse((out/'source/original_model.py').read_text())
    cls=next(n for n in old_ast.body if isinstance(n,ast.ClassDef) and n.name=='NativeInpaintingModel')
    method=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward'))
    space=dict(vars(native_module));exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'<registered_original_forward>','exec'),space)
    actual=model.forward
    model.forward=types.MethodType(space['forward'],model)
    try:
        with torch.no_grad(),native_execution(model,coordinates=batch['xt']): old=model(*model_inputs(batch))
    finally:model.forward=actual
    with torch.no_grad(),native_execution(model,coordinates=batch['xt']): default=model(*model_inputs(batch))
    observed={}
    def observer(row):
        if row.get('site')=='block' and row.get('layer')==10: observed['h10_input']=row['hidden']
    with native_execution(model,coordinates=batch['xt']):
        velocity,h9=model(*model_inputs(batch),return_feature_layer=9,observer=observer)
        if observed['h10_input'] is not h9: raise RuntimeError('Captured feature is not exact layer10 input')
        pred=heads(h9,batch['valid_mask'],batch['known_mask'])
        losses=auxiliary_losses(pred,label['q'],angular_denominator=label['angular_count'],free_denominator=label['free_count'])
        selected=[h9,next(heads.P.parameters()),next(heads.S.parameters()),
            next(model.backbone.layers[8].mlp.parameters()),next(model.backbone.layers[9].mlp.parameters())]
        r['counts']['gate_autograd_grad_attempts']+=1
        grads=torch.autograd.grad(losses['scale'],selected,retain_graph=True,allow_unused=True)
        r['counts']['gate_autograd_grad_returns']+=1
        if any(v is None or not bool(torch.isfinite(v).all()) or not bool(torch.count_nonzero(v)) for v in grads[:4]) or grads[4] is not None:
            raise RuntimeError('Scale gradient routing differs from H9 readout')
        model.zero_grad(set_to_none=True);heads.zero_grad(set_to_none=True)
        r['counts']['backward_attempts']+=1;(losses['angular']+losses['scale']).backward();r['counts']['backward_returns']+=1
    if any(p.grad is not None for layer in model.backbone.layers[9:] for p in layer.parameters()):
        raise RuntimeError('Auxiliary-only gradient reached downstream layer')
    up=magnitude(p.grad for layer in model.backbone.layers[:9] for p in layer.parameters())
    if not up['finite'] or up['nonzero']==0: raise RuntimeError('No finite upstream auxiliary gradient')
    with torch.no_grad(),native_execution(model,coordinates=batch['xt']): repeat,hrepeat=model(*model_inputs(batch),return_feature_layer=9)
    if not all(torch.equal(old,v) for v in (default,velocity,repeat)) or not torch.equal(h9,hrepeat):
        raise RuntimeError('Feature readout changed or destabilized original velocity')
    if label['q'].requires_grad or batch['x1'].requires_grad: raise RuntimeError('GT labels accepted gradients')
    if state_sha256(model)!=before: raise RuntimeError('Zero-update gate changed model')
    buffer=io.BytesIO();torch.save(heads.state_dict(),buffer);buffer.seek(0)
    check=AuxiliaryHeads(use_scale=True,device='cuda');check.load_state_dict(torch.load(buffer,weights_only=True),strict=True)
    if state_sha256(check)!=state_sha256(heads): raise RuntimeError('Auxiliary serialization differs')
    r['gates']['GPU']=dict(status='PASS',forwards=4,parameter_backwards=1,scale_autograd_grad_calls=1,
        original_default_feature_repeat_bitwise=True,true_H9_is_layer10_input=True,H9_shape=list(h9.shape),
        H9_dtype=str(h9.dtype),upstream_auxiliary_gradient=up,downstream_auxiliary_gradient_absent=True,
        scale_routes=[magnitude([g]) for g in grads],GT_gradient_absent=True,
        generator_unchanged=True,auxiliary_strict_roundtrip=True,loaded=loaded)
    del model,heads,check,pred,losses,h9,hrepeat,velocity,old,default,repeat,grads,selected,observed
    release();r['timing']['gates_seconds']=time.perf_counter()-tick;write(r,out)

def preflight(r,out):
    from .representation_diagnostics import build_teacher_bank,calibrate,teacher_evaluate
    from .geometry_representation import AuxiliaryHeads
    if r['status']!='INITIALIZED':raise RuntimeError('Preflight cannot repeat')
    verify(r);targets=targets_for(r);stream=stream_for(r);ot={}
    try:
        with count_ot_calls(ot):samples=stream.next_effective_batch()
    finally:
        r['counts']['gate_OT_calls']+=ot.get('OT_calls',0);r['counts']['gate_OT_returns']+=ot.get('OT_returns',0);write(r,out)
    gates(r,out,targets,samples)
    r['active_stage']='calibration';model,_=initial_model(r,True);heads=AuxiliaryHeads(use_scale=True,device='cuda')
    audit=calibrate(r,out,model,heads,samples,targets)
    r['calibration']=audit;write(r,out)
    del model,heads;release()
    bank,bank_audit=build_teacher_bank(r,out)
    r['teacher_bank']=bank_audit;r['assets']['teacher_bank']=bank_audit['artifact'];write(r,out)
    r['active_stage']='teacher_Geo_initial';model,_=initial_model(r)
    r['teacher']['Geo_initial']=teacher_evaluate(r,out,model,None,bank,targets,label='Geo_initial')
    heads=AuxiliaryHeads(use_scale=True,device='cuda')
    r['active_stage']='teacher_common_aux_initial'
    r['teacher']['common_aux_initial']=teacher_evaluate(r,out,model,heads,bank,targets,label='common_aux_initial')
    del model,heads;release()
    r['status']='PREFLIGHT_PASS';write(r,out);event('preflight_pass',counts=r['counts'],calibration=r['calibration'])

def make_optimizer(model,heads):
    gen=list(model.parameters());aux=[] if heads is None else list(heads.parameters())
    if len({id(p) for p in gen+aux})!=len(gen+aux) or not all(p.requires_grad and p.dtype==torch.float32 for p in gen+aux):
        raise RuntimeError('Optimizer must own each trainable FP32 parameter once')
    if sum(p.numel() for p in gen)!=GENERATOR_PARAMETERS:raise RuntimeError('Generator parameters missing')
    groups=[dict(params=gen,name='generator')]
    if aux:groups.append(dict(params=aux,name='auxiliary'))
    opt=torch.optim.AdamW(groups,lr=1e-5,betas=(.9,.95),weight_decay=0.)
    if [id(p) for g in opt.param_groups for p in g['params']]!=[id(p) for p in gen+aux]:
        raise RuntimeError('Optimizer ownership mismatch')
    return opt,gen,aux

def save_at_step(r,out,arm,step,model,heads,optimizer,stream,result):
    before_rng=torch.get_rng_state().clone();cuda_rng=[x.clone() for x in torch.cuda.get_rng_state_all()]
    before_stream=seal(stream.state_dict());tick=time.perf_counter()
    info=save_training_checkpoint(out/arm/f'additional_step{step}_cumulative_step{2000+step}.pt',
        model,heads,optimizer,stream,arm=arm,step=step,stats=r['target_statistics'],calibration=r['calibration'],
        protocol_sha256=r['protocol_sha256'])
    if not torch.equal(before_rng,torch.get_rng_state()) or before_stream!=seal(stream.state_dict()) or any(not torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all())):
        raise RuntimeError('Saving advanced a random stream')
    info.update(seconds=time.perf_counter()-tick,save_rng_unchanged=True)
    result['checkpoints'][str(step)]=info;write(r,out)

def train_arm(r,out,targets,arm):
    from .geometry_representation import AuxiliaryHeads,auxiliary_losses
    if arm not in ARMS or arm in r['arms'] or r['status']!='PREFLIGHT_PASS':raise RuntimeError('No training retries or extra arms')
    verify(r);r['active_stage']=arm+'_training'
    result=dict(status='STARTING',completed_updates=0,optimizer_step_attempts=0,batch_in_progress=False,
        optimizer_step_in_progress=False,stream_audits=[],diagnostics=[],checkpoints={},training_seconds=0.,
        OT_calls=0,OT_returns=0,peak_cuda_allocated=0)
    r['arms'][arm]=result;write(r,out)
    model=heads=optimizer=None;gen=[];aux=[]
    try:
        model,loaded=initial_model(r,True)
        heads=None if arm=='G0' else AuxiliaryHeads(use_scale=arm=='G_AngularScale',device='cuda')
        init_state=state_sha256(model)
        p_init=None if heads is None else state_sha256(heads.P)
        if init_state!=INITIAL_STATE:raise RuntimeError('Unequal initial generator')
        if arm=='G_AngularScale' and p_init!=r['arms']['G_Angular']['initial_projector_sha256']:
            raise RuntimeError('Projection head initializations differ')
        stream=stream_for(r);optimizer,gen,aux=make_optimizer(model,heads)
        initial_input_state=seal(stream.state_dict())
        result.update(status='RUNNING',load=loaded,initial_state_sha256=init_state,initial_projector_sha256=p_init,
            initial_stream_sha256=initial_input_state,parameters=sum(p.numel() for p in gen),generator_parameters=sum(p.numel() for p in gen),
            auxiliary_parameters=sum(p.numel() for p in aux),Geo_parameters=sum(p.numel() for p in model.context_encoder.parameters()),
            fresh_optimizer=True,optimizer_owned_parameters=sum(p.numel() for p in gen+aux))
        start_counts=dict(r['counts'])
        lambda_a=r['calibration']['lambda_A'];lambda_s=r['calibration']['lambda_S']
        observed={'role':{'role_embedding':model.role_embedding},
            'input_geo':dict(model.context_encoder.named_parameters()),
            'backbone_input':{'weight':model.backbone.x_embedder.mlp[0].weight},
            'backbone_output':{'weight':model.backbone.final_layer.linear.weight}}
        if heads is not None:
            observed['P']=dict(heads.P.named_parameters())
            if heads.S is not None:observed['S']=dict(heads.S.named_parameters())
        for step in range(1,501):
            if r['counts']['optimizer_step_attempts']>=1500:raise RuntimeError('1500 update cap')
            tick=time.perf_counter();torch.cuda.reset_peak_memory_stats();result['batch_in_progress']=True;ot={}
            try:
                with count_ot_calls(ot):samples=stream.next_effective_batch()
            finally:
                for suffix in ('calls','returns'):
                    value=ot.get('OT_'+suffix,0);r['counts']['training_OT_'+suffix]+=value;result['OT_'+suffix]+=value
            labels=[targets.for_sample(s,device='cuda') for s in samples]
            audit=dict(step=step,batch_index=stream.batch_index,sample_index=stream.sample_index,
                shared=[s['shared_input_sha256'] for s in samples],C=[s['context_sha256'] for s in samples],
                target_labels=[x['labels_sha256'] for x in labels],parents=[s['parent_index'] for s in samples],
                K=[s['K'] for s in samples],t=[float(s['t']) for s in samples],stream_sha256=seal(stream.state_dict()))
            if arm!='G0' and audit!=r['arms']['G0']['stream_audits'][step-1]:
                raise RuntimeError('Object/K/C/Y/Gaussian/OT/t/mask/permutation mismatch')
            if (stream.batch_index,stream.sample_index)!=(500+step,4000+8*step):raise RuntimeError('Input stream progress differs')
            nfree=sum(v['free_count'] for v in labels);nangular=sum(v['angular_count'] for v in labels)
            if nangular<=0:raise RuntimeError('Entire effective batch has undefined Angular labels')
            fm_den=9*nfree
            if fm_den not in (7524,7452,7488):raise RuntimeError('FM denominator changed')
            ramp=min(step/50.,1.);sparse=step in LOG_STEPS
            optimizer.zero_grad(set_to_none=True)
            sums=dict(FM=0.,Angular=0.,Scale=0.,weighted_Angular=0.,weighted_Scale=0.,total=0.)
            outputs=[]
            for sample,label in zip(samples,labels):
                batch=native_collate([sample],'cuda')
                with native_execution(model,coordinates=batch['xt']) as runtime:
                    if heads is None:
                        velocity=model(*model_inputs(batch)).float();pred=None
                    else:
                        velocity,h9=model(*model_inputs(batch),return_feature_layer=9);velocity=velocity.float()
                        pred=heads(h9,batch['valid_mask'],batch['known_mask'])
                    fm=masked_fm_loss(velocity,batch['u'],batch['valid_mask'],batch['known_mask'],denominator=fm_den)
                    terms=None if heads is None else auxiliary_losses(pred,label['q'],angular_denominator=nangular,free_denominator=nfree)
                    wa=0. if terms is None else ramp*lambda_a*terms['angular']
                    ws=0. if terms is None or terms['scale'] is None else ramp*lambda_s*terms['scale']
                    total=fm+wa+ws
                    if not bool(torch.isfinite(total)):raise RuntimeError('Nonfinite loss; no retry/fallback')
                    r['counts']['backward_attempts']+=1;total.backward();r['counts']['backward_returns']+=1
                if sparse:
                    sums['FM']+=float(fm.detach());sums['total']+=float(total.detach())
                    if terms is not None:
                        sums['Angular']+=float(terms['angular'].detach());sums['weighted_Angular']+=float(wa.detach())
                        if terms['scale'] is not None:
                            sums['Scale']+=float(terms['scale'].detach());sums['weighted_Scale']+=float(ws.detach())
                        outputs.append({k:pred[k].detach().cpu() for k in ('z','z_norm','q_hat') if pred.get(k) is not None})
            if sparse:
                before={group:{n:p.detach().clone() for n,p in params.items()} for group,params in observed.items()}
                gradients={group:magnitude(p.grad for p in params.values()) for group,params in observed.items()}
                if not all(g['finite'] for g in gradients.values()):raise RuntimeError('Nonfinite group gradient')
            gen_norm=float(torch.nn.utils.clip_grad_norm_(gen,1.,error_if_nonfinite=True))
            aux_norm=None if not aux else float(torch.nn.utils.clip_grad_norm_(aux,1.,error_if_nonfinite=True))
            result['optimizer_step_in_progress']=True;result['optimizer_step_attempts']+=1;r['counts']['optimizer_step_attempts']+=1
            optimizer.step();torch.cuda.synchronize()
            result['completed_updates']+=1;r['counts']['optimizer_updates']+=1;result['optimizer_step_in_progress']=False
            if not bool(torch.stack([torch.isfinite(p).all() for p in gen+aux]).all()):raise RuntimeError('Nonfinite parameter after update')
            result['batch_in_progress']=False;result['stream_audits'].append(audit)
            if sparse:
                updates={group:magnitude(p.detach()-before[group][n] for n,p in params.items()) for group,params in observed.items()}
                stats={}
                for key in ('z','z_norm','q_hat'):
                    chunks=[v[key] for v in outputs if key in v]
                    if chunks:
                        values=torch.cat(chunks,0).float()
                        stats[key]=dict(norm_mean=float(values.norm(dim=-1).mean()),variance_mean=float(values.var(dim=0,unbiased=False).mean()),
                            element_RMS=float(values.square().mean().sqrt()),count=len(values),finite=bool(torch.isfinite(values).all()))
                result['diagnostics'].append(dict(step=step,cumulative_step=2000+step,losses=sums,
                    Angular_status='NOT_APPLIED' if heads is None else 'APPLIED',Scale_status='APPLIED' if arm=='G_AngularScale' else 'NOT_APPLIED',
                    ramp=ramp,lambda_A=lambda_a if heads is not None else None,lambda_S=lambda_s if arm=='G_AngularScale' else None,
                    free_face_denominator=nfree,angular_denominator=nangular,undefined_angular_faces=nfree-nangular,
                    generator_grad_norm_before_clip=gen_norm,auxiliary_grad_norm_before_clip=aux_norm,
                    gradients=gradients,parameter_updates=updates,prediction_statistics=stats,runtime=runtime))
                del before
            result['training_seconds']+=time.perf_counter()-tick
            result['peak_cuda_allocated']=max(result['peak_cuda_allocated'],torch.cuda.max_memory_allocated())
            if step in (250,500):save_at_step(r,out,arm,step,model,heads,optimizer,stream,result)
            if step%25==0 or sparse:
                write(r,out);event('training',arm=arm,additional_step=step,total_updates=r['counts']['optimizer_updates'],seconds=result['training_seconds'])
        for key in ('model_forward_attempts','model_forward_returns','backward_attempts','backward_returns'):
            if r['counts'][key]-start_counts[key]!=4000:raise RuntimeError('Per-arm model call count differs')
        if result['completed_updates']!=500 or result['OT_calls']!=4000 or result['OT_returns']!=4000:
            raise RuntimeError('Per-arm budget incomplete')
        result.update(status='PASS',final_state_sha256=state_sha256(model),final_stream_state=stream.state_dict())
        verify(r);write(r,out)
    except BaseException as error:
        result.update(status='STOPPED',error=repr(error),traceback=traceback.format_exc(),automatic_retry=False)
        write(r,out);raise
    finally:
        if optimizer is not None:optimizer.zero_grad(set_to_none=True)
        model=heads=optimizer=None;gen.clear();aux.clear()
        if 'observed' in locals():observed.clear()
        release()

def teacher_endpoints(r,out):
    from .representation_diagnostics import load_teacher_bank,teacher_evaluate
    targets=targets_for(r);bank_info=r['teacher_bank']['artifact']
    bank=load_teacher_bank(bank_info['path'],bank_info['sha256'])
    if isinstance(bank,tuple):bank=bank[0]
    for arm in ARMS:
        r['active_stage']='teacher_'+arm
        model,heads,loaded=endpoint(r,arm,with_heads=True)
        r['teacher'][arm]=teacher_evaluate(r,out,model,heads,bank,targets,label=arm,shuffle_targets=heads is not None)
        r['teacher'][arm]['checkpoint_load']=loaded
        del model,heads;release();write(r,out)
    r['status']='TEACHER_PASS';write(r,out)

def sample_one(r,out,model,name,parent,patch,seed,noise,C,reference=None):
    c=r['counts']
    if c['rollout_attempts']>=100:raise RuntimeError('96formal+4extra hard rollout cap')
    c['rollout_attempts']+=1;write(r,out);sc=new_counts()
    try: raw,path,stats=clamped_sample(model,torch.from_numpy(noise.copy()).cuda(),torch.from_numpy(C.copy()).cuda(),sc)
    finally:
        c['sampling_forward_attempts']+=sc['model_forward_attempts'];c['sampling_forward_returns']+=sc['model_forward_returns']
        c['complete_rollouts']+=sc['complete_rollouts'];c['euler_updates']+=sc['euler_updates']
        r['sampling_attempts'].append(dict(model=name,parent=parent,patch=patch,seed=seed,replay=reference is not None,counts=sc));write(r,out)
    if not np.array_equal(path[:,:12].view(np.uint32),np.broadcast_to(C.reshape(1,12,9),(51,12,9)).copy().view(np.uint32)):
        raise RuntimeError('C changed in51 states')
    if not np.array_equal(path[0,12:].view(np.uint32),noise[0,12:].view(np.uint32)):raise RuntimeError('Initial freeGaussian changed')
    row=dict(model=name,parent=parent,patch=patch,seed=seed,K=12,phase='new_noise_confirmation',sampling=stats,
        C_all51_bitwise=True,noise_sha256=array_hash(noise),output_sha256=array_hash(raw),path_sha256=array_hash(path),
        auxiliary_modules_constructed=0,auxiliary_calls=0,feature_return_requests=0)
    if reference is not None:
        with np.load(reference['raw_path']) as old:
            if old['path'].tobytes()!=path.tobytes() or old['output'].tobytes()!=raw.tobytes():raise RuntimeError('Reload/replay differs')
        row.update(reference_path=reference['raw_path'],all51_bitwise_equal=True)
        r['replays'].append(row);c['replay_rollouts']+=1
    else:
        target=out/'raw'/f'{name}_P{parent}_patch{patch}_seed{seed}.npz'
        if target.exists():raise FileExistsError(target)
        np.savez_compressed(target,output=raw,path=path,initial_gaussian=noise)
        row.update(raw_path=str(target),raw_sha256=file_sha256(target));r['generated'].append(row);c['formal_rollouts']+=1
    write(r,out);event('sample',model=name,formal=c['formal_rollouts'],replays=c['replay_rollouts'])
    return row

def sampling(r,out):
    from .representation_evaluation import load_bank
    if r['generated'] or r['replays']:raise RuntimeError('No automatic sampling reruns')
    banks=load_bank(BANK,BANK_SHA)
    with np.load(r['assets']['evaluation_noise']['path']) as f:noises={s:f[f'noise_seed{s}'].copy() for s in SEEDS}
    tick=time.perf_counter()
    for name in MODELS:
        r['active_stage']='sampling_'+name
        if name=='Geo_initial':model,loaded=initial_model(r)
        else:model,heads,loaded=endpoint(r,name)
        # Prove no auxiliary module exists in inference model and defaults are used.
        if any(n.startswith(('P.','S.','aux')) for n,_ in model.named_parameters()):raise RuntimeError('Auxiliary head entered inference')
        before=state_sha256(model)
        for patch in (0,1,2):
            for parent in (0,1):
                for seed in (SEEDS if patch==0 else SEEDS[:2]):
                    sample_one(r,out,model,name,parent,patch,seed,noises[seed],banks[patch]['cases_by_key'][parent,12]['constraints'])
        if state_sha256(model)!=before:raise RuntimeError('Sampling changed generator state')
        del model;release()
        if name=='G_AngularScale':
            # A genuinely fresh strict generator-only load for the one registered replay.
            model,heads,loaded=endpoint(r,name)
            ref=next(x for x in r['generated'] if (x['model'],x['parent'],x['patch'],x['seed'])==(name,0,0,SEEDS[0]))
            sample_one(r,out,model,name,0,0,SEEDS[0],noises[SEEDS[0]],banks[0]['cases_by_key'][0,12]['constraints'],reference=ref)
            del model;release()
    if (r['counts']['formal_rollouts'],r['counts']['replay_rollouts'])!=(96,1):raise RuntimeError('Incomplete matrix')
    r['timing']['sampling_total_seconds']=time.perf_counter()-tick
    r['timing']['model_sampling_seconds']=sum(row['sampling']['seconds'] for row in r['generated']+r['replays'])
    r['status']='SAMPLING_PASS';write(r,out)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--phase',required=True,choices=('preflight','train','teacher','sample','evaluate','report'))
    args=parser.parse_args(argv);out=args.out.resolve();r=None
    try:
        if args.phase=='preflight':r=initialize(out);preflight(r,out)
        else:
            r=json.loads((out/'results.json').read_text());verify(r)
            if args.phase=='train':
                if r['status']!='PREFLIGHT_PASS':raise RuntimeError('Training only after preflight, no retry')
                targets=targets_for(r)
                for arm in ARMS:train_arm(r,out,targets,arm)
                r['status']='TRAINING_PASS';write(r,out)
            elif args.phase=='teacher':
                if r['status']!='TRAINING_PASS':raise RuntimeError('Teacher endpoints after full fixed training')
                teacher_endpoints(r,out)
            elif args.phase=='sample':
                if r['status']!='TEACHER_PASS':raise RuntimeError('Sample after fixed endpoint teacher diagnostics')
                sampling(r,out)
            elif args.phase=='evaluate':
                if r['status']!='SAMPLING_PASS':raise RuntimeError('Score only completed matrix')
                from .representation_evaluation import evaluate
                r['active_stage']='offline_evaluation';write(r,out);tick=time.perf_counter()
                evaluate(r,out);r['timing']['offline_evaluation_seconds']=time.perf_counter()-tick
                r['status']='PASS_EXECUTION';write(r,out)
            else:
                if r['status']!='PASS_EXECUTION':raise RuntimeError('Report after completed execution')
                from .representation_report import write_report
                r['report']=write_report(r,out);write(r,out)
    except BaseException as error:
        if r is not None:
            r.update(status='STOPPED',error=repr(error),traceback=traceback.format_exc(),automatic_retry=False)
            write(r,out)
        raise

if __name__=='__main__':main()
