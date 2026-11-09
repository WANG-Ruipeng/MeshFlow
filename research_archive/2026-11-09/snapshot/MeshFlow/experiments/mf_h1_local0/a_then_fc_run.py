"""A-start two-branch FC follow-up; fixed budgets and unchanged Native sampler."""
from native_t1.runtime import configure_stable_runtime
from pathlib import Path
import argparse,copy,hashlib,json,platform,subprocess,time,traceback
import numpy as np
import torch
from native_t1.artifacts import atomic_json,file_sha256,state_sha256,array_hash
from native_t1.checkpoint import load_t1,DEFAULT_CONFIG,DEFAULT_CHECKPOINT
from native_t1.data import load_archive,NativeTrainingStream,native_collate,model_inputs
from native_t1.model import native_execution
from native_t1.sampling import clamped_sample,new_counts
from .guided_native_geometry_run import prepare_assets,record,verify_records
from .t1_gated_factorial_run import seal,write,event,release,count_ot,cpu

PROJECT=Path(__file__).resolve().parent;REPO=PROJECT.parents[1]
OLD_A=PROJECT/'outputs/t1_interface_3arm_20260929_181500'
FACTORIAL=PROJECT/'outputs/t1_gated_loss_factorial_20260929_225800'
MODELS=('A_old','FC_old','A_continue','A_then_FC');NEW=MODELS[2:]
SEEDS=tuple(range(9601,9609));DIAGNOSTIC_STEPS=(1,100,250,500)
PHASE='development_followup'
INITIAL_STATE_SHA256='6a2f003d5997b1bbe996e872041ef06e0443ab4e3af4cc13bd42bd2d157a6520'
EXPECTED={'A_old':'43e56e107b0c6d28766aebd8cf8a0a764d8d52ef62ba2ca6e756d881634f876b','FC_old':'5a1c63bf4c68b2a920779a5b6f483605d4772c8b0a38ad994a979d4d55bef735'}


def metadata(out):
    files=[p for p in (PROJECT/'outputs').rglob('*') if p.is_file() and not p.is_relative_to(out)]
    files += [p for p in PROJECT.glob('*.py') if not p.name.startswith('a_then_fc_')]
    files += list((REPO/'native_t1').glob('*.py'))+list((REPO/'models').glob('*.py'))+[REPO/'utils/ot_utils.py',DEFAULT_CONFIG,Path('/mnt/c/Users/warpwang/Downloads/last.pt')]
    files += list((PROJECT/'assets').rglob('*.npz'))
    return {str(p):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sorted(set(files))}


class Ledger:
    def __init__(self,r):self.r=r
    def attach(self,model):
        def before(module,args):
            c=self.r['counts'];stage=self.r['active_stage']
            if c['model_forward_attempts']>=9702:raise RuntimeError('Hard 9702 forward cap')
            c['model_forward_attempts']+=1;c['forward_by_stage'][stage]=c['forward_by_stage'].get(stage,0)+1
        def after(module,args,output):self.r['counts']['model_forward_returns']+=1
        model.register_forward_pre_hook(before);model.register_forward_hook(after)


def initialize(out):
    if out.exists():raise FileExistsError(out)
    tick=time.perf_counter();a=json.loads((OLD_A/'results.json').read_text());f=json.loads((FACTORIAL/'results.json').read_text())
    if a['status']!='PASS_EXECUTION' or f['status']!='PASS_EXECUTION':raise RuntimeError('Incomplete source runs')
    out.mkdir(parents=True);(out/'raw').mkdir()
    r=dict(status='INITIALIZING',active_stage='assets',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),assets={},arms={},generated=[],scores=[],replays=[],gates={},timing={},evaluation_ledger={},sampling_attempts=[],
        protection=dict(before=metadata(out)),AGENTS='No applicable AGENTS.md found in repository and ancestors',
        environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__),
        git_status_before=subprocess.run(['git','status','--short'],cwd=REPO,capture_output=True,text=True).stdout,
        counts=dict(model_forward_attempts=0,model_forward_returns=0,forward_by_stage={},backward_attempts=0,backward_completed=0,training_backward_attempts=0,training_backward_completed=0,gate_backward_attempts=0,gate_backward_completed=0,optimizer_step_attempts=0,optimizer_updates=0,training_OT_attempts=0,training_OT_calls=0,gate_OT_attempts=0,gate_OT_calls=0,formula_autograd_calls=0,formula_autograd_returns=0,rollout_attempts=0,complete_rollouts=0,euler_updates=0,formal_rollouts=0,replay_rollouts=0,geometry_guidance_calls=0,geometry_repairs=0,teacher_forwards=0))
    write(r,out)
    r['assets']['endpoints']={'A_old':copy.deepcopy(a['arms']['A']['checkpoints']['500']),'FC_old':copy.deepcopy(f['arms']['M10']['checkpoints']['500'])}
    for name,cp in r['assets']['endpoints'].items():
        if record(cp['path'])['sha256']!=cp['sha256'] or cp['sha256']!=EXPECTED[name] or cp['cumulative_updates']!=1000:raise RuntimeError('Wrong historical endpoint '+name)
    initial=r['assets']['endpoints']['A_old']
    if initial['state_sha256']!=INITIAL_STATE_SHA256:raise RuntimeError('Wrong A state identity')
    payload=torch.load(initial['path'],map_location='cpu',weights_only=True,mmap=True)
    if (payload['schema'],payload['arm'],payload['additional_updates'],payload['cumulative_updates'])!=('t1_interface_branch_v1','A',500,1000):raise RuntimeError('Wrong initial payload')
    if payload.get('optimizer_step_in_progress') or payload.get('batch_in_progress') or not payload.get('recoverable',False):raise RuntimeError('Initial checkpoint incomplete')
    r['assets']['inputs']=record(f['assets']['inputs']['path'])
    if r['assets']['inputs']!=f['assets']['inputs']:raise RuntimeError('Original inputs changed')
    r['assets']['parent_ids']=f['assets']['parent_ids'];r['assets']['context_audits']=copy.deepcopy(f['assets']['context_audits']);r['input_topology']=copy.deepcopy(f['input_topology'])
    archive=load_archive(r['assets']['inputs']['path'],r['assets']['inputs']['sha256'])
    if 'stream_state' in payload:
        state=copy.deepcopy(payload['stream_state']);stream=NativeTrainingStream(archive,'Native_correct',seed=state['seed']);stream.load_state_dict(state)
        if stream.state_dict()!=state or state['batch_index']!=a['arms']['A']['stream_audits'][-1]['batch_index'] or state['sample_index']!=8*state['batch_index']:raise RuntimeError('A stream restore differs')
        mode='restore_A_stream'
    else:
        stream=NativeTrainingStream(archive,'Native_correct',seed=1010);state=stream.state_dict();mode='fallback_seed1010'
    r['assets']['initial']=copy.deepcopy(initial)
    torch.save(state,out/'training_stream_state.pt');r['assets']['training_stream_state']=record(out/'training_stream_state.pt')
    stream_choice=dict(mode=mode,seed=state['seed'],arm=state['arm'],initial_batch_index=state['batch_index'],initial_sample_index=state['sample_index'],state_sha256=seal(state),optimizer='Fresh AdamW, no old momentum loaded',global_rng_seed=state['seed'])
    del payload,stream
    r['assets']['noise']=record(f['assets']['confirmation_noise']['path'])
    if r['assets']['noise']!=f['assets']['confirmation_noise']:raise RuntimeError('Archived noise changed')
    r['assets']['source_factorial_results']=record(FACTORIAL/'results.json');r['assets']['source_A_results']=record(OLD_A/'results.json')
    runtime=copy.deepcopy(a['protocol']['runtime'])
    if runtime!=f['protocol']['runtime']:raise RuntimeError('Numerical source contract differs')
    r['protocol']=dict(schema='a_then_fc_v1',initial=dict(**initial,file_sha256=initial['sha256']),models={m:dict(FC=m in ('FC_old','A_then_FC'),FF=False,cumulative_updates=1000 if m in MODELS[:2] else 1500) for m in MODELS},
        training=dict(arms=list(NEW),updates_each=500,total_update_cap=1000,initial_cumulative_updates=1000,final_cumulative_updates=1500,effective_batch=8,microbatch=1,lr=1e-5,betas=[.9,.95],weight_decay=0.,grad_clip=1.,optimizer='Fresh AdamW',full_model_trainable=True,modules_eval=True,time_distribution='Original logit-normal',OT='Original free-only nested OT reorders Z only',K=[4,8,12],checkpoint_additional_steps=[500],diagnostic_steps=list(DIAGNOSTIC_STEPS),scheduler=False,EMA=False,condition_dropout=False,FC_weight=.1,FF_weight=None),
        training_stream=stream_choice,gate_threshold=.5,gate='Per stored FP32 sample t>=.5; full effective8 FC-group denominator including inactive groups',runtime=runtime,
        evaluation_seeds=list(SEEDS),phase=PHASE,noise_status='Previously observed 9601-9608; development follow-up, not independent confirmation',
        budget=dict(total_updates=1000,training_forwards=8000,training_backwards=8000,gate_forwards=2,gate_backwards=0,formal_rollouts=32,replay_cap=2,total_rollout_cap=34,total_forward_cap=9702,teacher_forwards=0),
        evaluation=dict(N=112,K=12,free=100,Euler_steps=50,guidance=False,postprocessing=False,original_scorer=True,fixed_figure_seed=9601),
        interpretation='Primary equal-budget comparison A_then_FC vs A_continue; older FC is unequal-budget context, not an early-vs-late schedule control',stop='Exactly two new500 endpoints,32 new formal rollouts,max2 replays; no teacher or extra matrix/Phase3')
    r['protocol_sha256']=seal(r['protocol'])
    with np.load(r['assets']['noise']['path']) as nz:noises={s:nz[f'noise_seed{s}'].copy() for s in SEEDS}
    mapping={'A_old':'M00','FC_old':'M10'}
    for name,prior in mapping.items():
        rows=[x for x in f['generated'] if x['phase']=='confirmation' and x['model']==prior and x['seed'] in SEEDS]
        if len(rows)!=16 or {(x['parent'],x['seed']) for x in rows}!={(p,s) for p in (0,1) for s in SEEDS}:raise RuntimeError('Missing archived matrix')
        for row in rows:
            if file_sha256(row['raw_path'])!=row['raw_sha256']:raise RuntimeError('Archived raw changed')
            with np.load(row['raw_path']) as raw:arr=raw['output'];path=raw['path'];C=archive.cases[row['parent'],12].constraints
            if arr.shape!=(112,9) or path.shape!=(51,112,9) or arr.dtype!=np.float32 or path.dtype!=np.float32 or not np.isfinite(path).all():raise RuntimeError('Archived raw type/shape')
            if array_hash(arr)!=row['output_sha256'] or array_hash(path)!=row['path_sha256'] or not np.array_equal(arr.view(np.uint32),path[-1].view(np.uint32)):raise RuntimeError('Archived arrays differ')
            if not np.array_equal(path[:,:12].view(np.uint32),np.broadcast_to(C.reshape(1,12,9),(51,12,9)).copy().view(np.uint32)) or not np.array_equal(path[0,12:].view(np.uint32),noises[row['seed']][0,12:].view(np.uint32)):raise RuntimeError('Archived C/Gaussian differs')
            new=copy.deepcopy(row);new.update(model=name,source_model=prior,source_phase='confirmation',phase=PHASE,reused=True);r['generated'].append(new)
    inherited=f['assets']['critical_files']
    required=list((REPO/'native_t1').glob('*.py'))+list((REPO/'models').glob('*.py'))+[REPO/'utils/ot_utils.py',DEFAULT_CONFIG,DEFAULT_CHECKPOINT]
    required += [p for p in PROJECT.glob('*.py') if not p.name.startswith('a_then_fc_')]
    old_metadata=f['protection']['before'];checks=0
    for p in required:
        key=str(p)
        if key in inherited:
            if record(p)['sha256']!=inherited[key]['sha256']:raise RuntimeError('Historical source identity changed:'+key)
            checks+=1
        elif key in old_metadata:
            st=p.stat()
            if dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)!=old_metadata[key]:raise RuntimeError('Historical source metadata changed:'+key)
            checks+=1
    if verify_records(f['new_execution_sources'])['status']!='PASS':raise RuntimeError('Previous factorial execution sources changed')
    r['gates']['historical_loss_test_reuse']=dict(status='PASS',source=str(FACTORIAL/'results.json'),fixture_names=f['gates']['CPU_switch']['fixtures'],original_status=f['gates']['CPU_switch']['status'],reuse_scope='Unchanged oracle/gate/full-denominator/permutation/C-detach helpers; no repeated FF split or gradient matrix',new_autograd_calls=0)
    if f['gates']['CPU_switch']['status']!='PASS':raise RuntimeError('Historical formula gates unavailable')
    r['gates']['archived_evaluation_reuse']=dict(status='PASS',rows=32,raw_arrays_verified=True,C_all51_bits=True,free_noise_bits=True,source_code_checks=checks,source_runtime_equal=True,old_scoring_calls=0)
    critical={str(p):record(p) for p in required+[Path('/mnt/c/Users/warpwang/Downloads/last.pt')]}
    for key in ('inputs','initial','noise','training_stream_state','source_factorial_results','source_A_results'):
        rec=r['assets'][key];critical[rec['path']]=rec
    for cp in r['assets']['endpoints'].values():critical[cp['path']]=cp
    for row in r['generated']:critical[row['raw_path']]=record(row['raw_path'])
    r['assets']['critical_files']=critical
    r['environment']['GPU']=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
    r['timing']['asset_seconds']=time.perf_counter()-tick;r.update(status='ASSETS_PASS',active_stage='assets_complete');write(r,out);event('assets_pass',out=str(out),stream=stream_choice)


def load_model(r,name,trainable=False,initial=False):
    if name not in MODELS or (initial and name not in NEW):raise ValueError('Wrong model selector')
    cp=r['assets']['initial'] if initial else (r['assets']['endpoints'][name] if name in MODELS[:2] else r['arms'][name]['checkpoints']['500'])
    if file_sha256(cp['path'])!=cp['sha256']:raise RuntimeError('Checkpoint bytes changed')
    model,architecture=load_t1()
    payload=torch.load(cp['path'],map_location='cpu',weights_only=True,mmap=True)
    old_name='A_old' if initial else name
    expected_arm={'A_old':'A','FC_old':'M10'}.get(old_name,old_name)
    schema={'A_old':'t1_interface_branch_v1','FC_old':'t1_gated_factorial_branch_v1'}.get(old_name,'a_then_fc_branch_v1')
    cumulative=1000 if initial or name in MODELS[:2] else 1500
    if (payload['schema'],payload['arm'],payload['additional_updates'],payload['cumulative_updates'])!=(schema,expected_arm,500,cumulative):raise RuntimeError('Wrong branch checkpoint metadata')
    keys=model.load_state_dict(payload['model'],strict=True);del payload
    if keys.missing_keys or keys.unexpected_keys or state_sha256(model)!=cp['state_sha256']:raise RuntimeError('Strict branch state differs')
    model.set_trainable(trainable).eval()
    if sum(p.numel() for p in model.parameters())!=130251267:raise RuntimeError('Network changed')
    Ledger(r).attach(model)
    return model,dict(status='PASS',strict=True,common_architecture=architecture,endpoint=cp,initial_from_A=initial,old_optimizer_loaded=False)


def sample_one(r,out,model,name,parent,seed,noise,C,reference=None):
    c=r['counts']
    if c['rollout_attempts']>=34 or (reference is not None and c['replay_rollouts']>=2):raise RuntimeError('Hard34 trajectory/2 replay cap')
    c['rollout_attempts']+=1;write(r,out);tick=time.perf_counter();nf=c['model_forward_attempts'];sc=new_counts()
    try:raw,path,stats=clamped_sample(model,torch.from_numpy(noise.copy()).cuda(),torch.from_numpy(C.copy()).cuda(),sc)
    finally:
        c['euler_updates']+=sc['euler_updates'];c['complete_rollouts']+=sc['complete_rollouts'];r['sampling_attempts'].append(dict(model=name,parent=parent,seed=seed,replay=reference is not None,counts=dict(sc)));write(r,out)
    if c['model_forward_attempts']-nf!=50 or not np.isfinite(path).all():raise RuntimeError('Nonfinite/wrong50 NFE')
    if not np.array_equal(path[:,:12].view(np.uint32),np.broadcast_to(C.reshape(1,12,9),(51,12,9)).copy().view(np.uint32)) or not np.array_equal(path[0,12:].view(np.uint32),noise[0,12:].view(np.uint32)):raise RuntimeError('C or Gaussian changed')
    row=dict(model=name,parent=parent,seed=seed,K=12,phase=PHASE,variant='Raw',sampling=stats,C_all51_FP32_bits_exact=True,initial_free_Gaussian_exact=True,all_states_finite=True,output_sha256=array_hash(raw),path_sha256=array_hash(path),reused=False,seconds=time.perf_counter()-tick)
    if reference is not None:
        with np.load(reference['raw_path']) as f:
            if not np.array_equal(raw.view(np.uint32),f['output'].view(np.uint32)) or not np.array_equal(path.view(np.uint32),f['path'].view(np.uint32)):raise RuntimeError('Replay full path differs')
        row.update(reference_path=reference['raw_path'],raw_and_all51_bitwise_match=True);r['replays'].append(row);c['replay_rollouts']+=1
    else:
        pathfile=out/'raw'/f'{name}_P{parent}_K12_seed{seed}.npz'
        if pathfile.exists():raise FileExistsError(pathfile)
        np.savez_compressed(pathfile,output=raw,path=path);row.update(raw_path=str(pathfile),raw_sha256=file_sha256(pathfile));r['generated'].append(row);c['formal_rollouts']+=1
    write(r,out);event('sample',model=name,parent=parent,seed=seed,replay=reference is not None,formal=c['formal_rollouts'])


def preflight(r,out,relations):
    from .a_then_fc_evaluation import run_cpu_tests
    r['gates']['metric_arithmetic']=run_cpu_tests()
    from .t1_interface_relations import prepare_relations
    from .t1_gated_factorial_loss import switch_loss
    archive=load_archive(r['assets']['inputs']['path'],r['assets']['inputs']['sha256'])
    state=torch.load(r['assets']['training_stream_state']['path'],weights_only=True,map_location='cpu')
    stream=NativeTrainingStream(archive,'Native_correct',seed=state['seed']);stream.load_state_dict(state)
    with count_ot(r,'gate'):samples=stream.next_effective_batch()
    labels=[relations.prepare_sample_labels(s) for s in samples];fc_den=sum(x['FC_count'] for x in labels);ff_den=sum(x['FF_count'] for x in labels)
    oracle=[]
    for s,lab in zip(samples,labels):
        b=native_collate([s],'cpu');pack=prepare_relations(lab,'cpu');loss,parts,gate=switch_loss(b['u'].sum()*0,b,b['u'],pack,fc_den,ff_den,True,False)
        if not float(loss)<1e-11 or not torch.equal(b['xt'][b['known_mask']],b['context'][b['known_mask']]):raise RuntimeError('Real-source oracle/C gate failed')
        oracle.append(dict(t=float(s['t']),FC=float(parts['FC']),full_denominator=fc_den,active=gate['active_samples']))
    r['gates']['restored_first_batch_oracle']=dict(status='PASS',rows=oracle,source_mapping_exact=True,stream_batch=stream.batch_index,training_initial_state_unchanged=seal(state)==r['protocol']['training_stream']['state_sha256'],model_calls=0,autograd_calls=0)
    index=next((i for i,s in enumerate(samples) if float(s['t'])>=.5),0);reference=None;r['active_stage']='initial_forward_gates'
    for name in NEW:
        model,loaded=load_model(r,name,trainable=True,initial=True);before=state_sha256(model);b=native_collate([samples[index]],'cuda')
        try:
            with torch.no_grad(),native_execution(model,coordinates=b['xt']) as runtime:answer=model(*model_inputs(b)).float()
            bits=answer.detach().view(torch.int32).cpu()
            if reference is not None and not torch.equal(reference,bits):raise RuntimeError('Two A initial forwards differ')
            reference=bits.clone()
            if state_sha256(model)!=before or before!=INITIAL_STATE_SHA256:raise RuntimeError('Zero-update initial state changed')
            r['gates'][name+'_initial']=dict(status='PASS',state_sha256=before,output_sha256=array_hash(bits.numpy()),strict_load=loaded,runtime=runtime,model_backwards=0,optimizer_updates=0)
        finally:del model;release()
    r['gates']['new_initial_forward_bitwise']=True
    r['active_stage']='historical_A_replay';model,_=load_model(r,'A_old');before=state_sha256(model)
    try:
        with np.load(r['assets']['noise']['path']) as f:noise=f['noise_seed9601'].copy()
        prior=next(x for x in r['generated'] if x['model']=='A_old' and x['parent']==0 and x['seed']==9601)
        sample_one(r,out,model,'A_old',0,9601,noise,archive.cases[0,12].constraints,reference=prior)
        if state_sha256(model)!=before:raise RuntimeError('Replay changed frozen A')
    finally:del model;release()
    r['relations']=relations.audit
    r['new_execution_sources']={str(p):record(p) for p in sorted(PROJECT.glob('a_then_fc_*.py')) if p.name!='a_then_fc_report.py'}
    r.update(status='PREFLIGHT_PASS',active_stage='preflight_complete');write(r,out);event('preflight_pass',counts=r['counts'])


def sample_all(r,out):
    if any(r['arms'][a]['status']!='PASS' or r['arms'][a]['completed_updates']!=500 for a in NEW):raise RuntimeError('Fixed two endpoints required')
    archive=load_archive(r['assets']['inputs']['path'],r['assets']['inputs']['sha256'])
    r['frozen_evaluation_models']={m:r['assets']['endpoints'][m] if m in MODELS[:2] else r['arms'][m]['checkpoints']['500'] for m in MODELS};write(r,out);tick=time.perf_counter()
    with np.load(r['assets']['noise']['path']) as f:noises={s:f[f'noise_seed{s}'].copy() for s in SEEDS}
    for name in NEW:
        r['active_stage']='sampling_'+name;model,loaded=load_model(r,name);before=state_sha256(model);start=time.perf_counter()
        try:
            for parent in (0,1):
                for seed in SEEDS:sample_one(r,out,model,name,parent,seed,noises[seed],archive.cases[parent,12].constraints)
            if name=='A_then_FC':
                r['active_stage']='new_FC_repeat';ref=next(x for x in r['generated'] if x['model']==name and x['parent']==0 and x['seed']==9601)
                sample_one(r,out,model,name,0,9601,noises[9601],archive.cases[0,12].constraints,reference=ref)
            if state_sha256(model)!=before:raise RuntimeError('Sampling changed model')
            r.setdefault('sampling_models',[]).append(dict(model=name,state_before=before,state_after=state_sha256(model),seconds=time.perf_counter()-start,strict_load=loaded))
        finally:del model;release()
    r['timing']['sampling_seconds']=time.perf_counter()-tick;r.update(status='RAW_COMPLETE',active_stage='sampling_complete');write(r,out)


def score_all(r,out):
    from .a_then_fc_evaluation import score_output,aggregate_results,complete_area_metrics
    tick=time.perf_counter();bank,_=prepare_assets();bank['input_topology']=r['input_topology'];bank['context_audits']={int(k):v for k,v in r['assets']['context_audits'].items()}
    if r['scores']:raise RuntimeError('No automatic score rerun')
    old=json.loads((FACTORIAL/'results.json').read_text())
    for name,prior in [('A_old','M00'),('FC_old','M10')]:
        for row in old['scores']:
            if row['phase']=='confirmation' and row['model']==prior:
                new=copy.deepcopy(row);new.update(model=name,phase=PHASE,reused=True,source_model=prior,source_phase='confirmation');r['scores'].append(new)
    if len(r['scores'])!=32:raise RuntimeError('Archived fixed scores incomplete')
    for row in [x for x in r['generated'] if not x['reused']]:
        with np.load(row['raw_path']) as f:raw=f['output'].copy()
        start=time.perf_counter();metrics=score_output(raw,bank,row['parent'],ledger=r['evaluation_ledger'])
        r['scores'].append(dict(phase=PHASE,model=row['model'],parent=row['parent'],K=12,seed=row['seed'],variant='Raw',raw_path=row['raw_path'],metrics=metrics,reused=False,offline_scoring_seconds=time.perf_counter()-start))
        if (len(r['scores'])-32)%4==0:write(r,out);event('scored',count=len(r['scores'])-32)
    r['scores'],r['gates']['free_area_source']=complete_area_metrics(r['scores'])
    r['summary']=aggregate_results(r['scores'],SEEDS,phase=PHASE);r['timing']['offline_scoring_seconds']=time.perf_counter()-tick
    after=metadata(out);changed=[p for p,v in r['protection']['before'].items() if after.get(p)!=v]
    r['protection']['after']=dict(status='PASS' if not changed else 'FAIL',files=len(r['protection']['before']),changed=changed)
    r['protection']['critical_files_after']=verify_records(r['assets']['critical_files']);r['protection']['execution_after']=verify_records(r['new_execution_sources'])
    if changed or any(r['protection'][k]['status']!='PASS' for k in ('critical_files_after','execution_after')):raise RuntimeError('Protected source or asset changed')
    if r['arms']['A_continue']['stream_audits']!=r['arms']['A_then_FC']['stream_audits']:raise RuntimeError('Final two branch stream differs')
    r['gates']['final_stream_match']=dict(status='PASS',updates=500,sha256=seal(r['arms']['A_continue']['stream_audits']))
    c=r['counts'];expected=dict(optimizer_updates=1000,optimizer_step_attempts=1000,training_backward_attempts=8000,training_backward_completed=8000,gate_backward_attempts=0,gate_backward_completed=0,complete_rollouts=34,rollout_attempts=34,euler_updates=1700,formal_rollouts=32,replay_rollouts=2,training_OT_calls=8000,gate_OT_calls=8,model_forward_attempts=9702,model_forward_returns=9702,teacher_forwards=0)
    if any(c[k]!=v for k,v in expected.items()):raise RuntimeError('Actual fixed budget mismatch')
    r['timing']['training_seconds']=sum(a['training_seconds'] for a in r['arms'].values());r['timing']['checkpoint_seconds']=sum(x['save_and_hash_seconds'] for a in r['arms'].values() for x in a['checkpoints'].values())
    r.update(status='PASS_EXECUTION',active_stage='FINISHED',finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));write(r,out);event('complete',counts=c)


def execute(out,stage):
    if stage=='init':return initialize(out)
    r=json.loads((out/'results.json').read_text())
    try:
        if seal(r['protocol'])!=r['protocol_sha256']:raise RuntimeError('Protocol changed')
        configure_stable_runtime()
        from .t1_interface_relations import load_source_relations
        relations=load_source_relations(r['assets']['inputs']['path'],expected_sha256=r['assets']['inputs']['sha256'])
        if stage=='preflight':
            if r['status']!='ASSETS_PASS':raise RuntimeError('No gate retry')
            preflight(r,out,relations)
        elif stage=='train':
            if r['status']!='PREFLIGHT_PASS':raise RuntimeError('No training retry')
            from .a_then_fc_train import train_arm
            with count_ot(r,'training'):
                for name in NEW:train_arm(r,out,relations,name)
            r.update(status='TRAINING_COMPLETE',active_stage='training_complete');write(r,out)
        elif stage=='sample':
            if r['status']!='TRAINING_COMPLETE' or verify_records(r['new_execution_sources'])['status']!='PASS':raise RuntimeError('Sampling gate fails')
            sample_all(r,out)
        elif stage=='score':
            if r['status']!='RAW_COMPLETE':raise RuntimeError('Sampling incomplete')
            score_all(r,out)
        elif stage=='report':
            if r['status']!='PASS_EXECUTION':raise RuntimeError('Execution incomplete')
            from .a_then_fc_report import deliver_report
            r['delivery']=deliver_report(r,out);write(r,out)
    except BaseException as err:
        r.update(status='STOPPED',failure=dict(stage=stage,error=repr(err),traceback=traceback.format_exc(),automatic_retry=False));write(r,out);event('stopped',error=repr(err),counts=r['counts']);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);p.add_argument('--stage',required=True,choices=['init','preflight','train','sample','score','report']);a=p.parse_args();execute(a.out.resolve(),a.stage)
