"""One fixed time-gated relation-loss experiment, exactly500 updates and16 rollouts."""
from native_t1.runtime import configure_stable_runtime
from pathlib import Path
import argparse,copy,gc,hashlib,json,platform,subprocess,time,traceback
import numpy as np
import torch
from native_t1.artifacts import atomic_json,file_sha256,state_sha256,array_hash
from native_t1.checkpoint import load_t1,T1_FILE_SHA256,T1_STATE_SHA256,DEFAULT_CHECKPOINT,DEFAULT_CONFIG
from native_t1.data import load_archive
from native_t1.sampling import clamped_sample,new_counts
from .guided_native_geometry_run import prepare_assets,record,verify_records
from .c_relation_run import discover_source
from .t1_interface_run import cpu

PROJECT=Path(__file__).resolve().parent
REPO=PROJECT.parents[1]
SEEDS=tuple(range(9501,9509));DIAGNOSTIC_STEPS=(1,100,250,500)

def seal(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(r,out):atomic_json(out/'results.json',r)
def event(name,**kw):print(json.dumps(dict(event=name,**kw),allow_nan=False),flush=True)
def release():gc.collect();torch.cuda.empty_cache()
def metadata(out):
    files=[p for p in (PROJECT/'outputs').rglob('*') if p.is_file() and not p.is_relative_to(out)]
    files += [p for p in PROJECT.glob('*.py') if not p.name.startswith('t1_time_gate_')]
    files += list((REPO/'native_t1').glob('*.py'))+list((REPO/'models').glob('*.py'))+[REPO/'utils/ot_utils.py',DEFAULT_CONFIG,Path('/mnt/c/Users/warpwang/Downloads/last.pt')]
    files += list((PROJECT/'assets').rglob('*.npz'))
    return {str(p):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sorted(set(files))}

class Ledger:
    def __init__(self,r):self.r=r
    def attach(self,model):
        def before(module,args):
            c=self.r['counts']
            if c['model_forward_attempts']>=4800:raise RuntimeError('Hard4800 forward budget')
            c['model_forward_attempts']+=1
            stage=self.r['active_stage'];c['forward_by_stage'][stage]=c['forward_by_stage'].get(stage,0)+1
        def after(module,args,output):self.r['counts']['model_forward_returns']+=1
        model.register_forward_pre_hook(before);model.register_forward_hook(after)

def initialize(out):
    if out.exists():raise FileExistsError(out)
    source,saved=discover_source();out.mkdir(parents=True);(out/'raw').mkdir()
    initial=copy.deepcopy(saved['protocol']['initial'])
    if initial['file_sha256']!=T1_FILE_SHA256 or initial['state_sha256']!=T1_STATE_SHA256 or Path(initial['path']).resolve()!=DEFAULT_CHECKPOINT.resolve():raise RuntimeError('Wrong common step500 identity')
    training=copy.deepcopy(saved['protocol']['training'])
    training.update(total_update_cap=500,additional_updates_each=500,gate='stored FP32 sampled t >=0.5',arms=['C_gate'])
    protocol=dict(schema='t1_time_gate_v1',initial=initial,training=training,gate_threshold=.5,
        full_effective_group_denominators=True,auxiliary_formula='FM + I[t>=.5]*(.1 FC + .1 FF); original full effective8 group denominators; no active-fraction compensation',
        seeds=list(SEEDS),seed_status='previously_used_paired_followup',new_seeds=False,
        runtime=copy.deepcopy(saved['protocol']['runtime']),
        budget=dict(optimizer_updates=500,model_forwards=4800,model_backwards=4000,complete_rollouts=16,Euler_updates=800,training_OT_calls=4000),
        evaluation=dict(models=['A','C','C_gate'],new_model='C_gate',saved_baseline_outputs=32,new_outputs=16,N=112,K=12,free=100,steps=50,
            fixed_checkpoint_additional_step=500,postprocessing=False,guidance=False,noise='Existing seed9501..9508 freeGaussian; no GT matching',same_noise_two_contexts=True),
        interpretation='Exploratory paired followup on known seeds and two training objects; gating changes temporal support and total auxiliary exposure. No equal-dose control or generalization claim.',
        stop='500 updates then16 complete generations; no retries, new arms, new gates, modules, or Phase3')
    r=dict(status='INITIALIZING',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),protocol=protocol,protocol_sha256=seal(protocol),source_run=str(source),
        active_stage='assets',counts=dict(model_forward_attempts=0,model_forward_returns=0,forward_by_stage={},backward_attempts=0,backward_completed=0,
            optimizer_step_attempts=0,optimizer_updates=0,training_OT_calls=0,training_OT_attempts=0,rollout_attempts=0,complete_rollouts=0,euler_updates=0,
            geometry_guidance_calls=0,geometry_repairs=0,formula_autograd_calls=0,formula_autograd_returns=0),
        arms={},generated=[],scores=[],gates={},assets={},timing={},evaluation_ledger={},protection=dict(before=metadata(out)),
        AGENTS='No applicable AGENTS.md found by repository and ancestor searches',
        environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__),
        git_status_before=subprocess.run(['git','status','--short'],cwd=REPO,capture_output=True,text=True).stdout)
    write(r,out);tick=time.perf_counter()
    from .t1_time_gate_evaluation import load_saved_baselines
    baseline=load_saved_baselines(source,SEEDS)
    r['scores']=baseline['scores'];r['generated']=baseline['generated'];r['assets']['saved_baseline_identity']=baseline['source_identity']
    r['assets']['inputs']=record(saved['relations']['inputs']['path'])
    if r['assets']['inputs']['sha256']!=saved['relations']['inputs']['sha256']:raise RuntimeError('Source input differs')
    r['assets']['parent_ids']=saved['assets']['parent_ids'];r['input_topology']=saved['input_topology'];r['assets']['context_audits']=saved['assets']['context_audits']
    r['assets']['source_report']=record(source/'THREE_ARM_INTERFACE_REPORT.md');r['assets']['source_results']=record(source/'results.json')
    r['assets']['noise_file']=record(source/'fixed_noise.npz')
    r['assets']['initial_file']=record(DEFAULT_CHECKPOINT)
    if r['assets']['initial_file']['sha256']!=T1_FILE_SHA256:raise RuntimeError('Initial weight hash differs')
    critical={str(p):record(p) for p in list((REPO/'native_t1').glob('*.py'))+list((REPO/'models').glob('*.py'))+
        [REPO/'utils/ot_utils.py',DEFAULT_CONFIG,PROJECT/'t1_interface_relations.py',PROJECT/'t1_interface_train.py',PROJECT/'t1_native_confirmation_evaluation.py',PROJECT/'t1_native_confirmation_topology.py',Path('/mnt/c/Users/warpwang/Downloads/last.pt')]}
    for item in ('inputs','source_report','source_results','noise_file','initial_file'):
        rec=r['assets'][item];critical[rec['path']]=rec
    for arm in ('A','C'):
        cp=saved['arms'][arm]['checkpoints']['500'];rec=record(cp['path'])
        if rec['sha256']!=cp['sha256']:raise RuntimeError('Baseline checkpoint hash differs')
        critical[rec['path']]=rec
    for row in r['generated']:
        rec=record(row['raw_path'])
        if rec['sha256']!=row['raw_sha256']:raise RuntimeError('Saved baseline raw hash differs')
        critical[rec['path']]=rec
    r['assets']['critical_files']=critical
    r['reference_stream_sha256']=seal(saved['arms']['A']['stream_audits'])
    if any(saved['arms'][a]['stream_audits']!=saved['arms']['A']['stream_audits'] for a in ('B','C')):raise RuntimeError('Reference stream mismatch')
    audits=saved['arms']['A']['stream_audits']
    r['preregistered_gate_exposure']=dict(samples=4000,active_samples=sum(t>=.5 for x in audits for t in x['times']),
        active_FC_groups=sum(n for x in audits for t,n in zip(x['times'],x['FC_counts']) if t>=.5),
        active_FF_groups=sum(n for x in audits for t,n in zip(x['times'],x['FF_counts']) if t>=.5),
        total_FC_groups=sum(sum(x['FC_counts']) for x in audits),total_FF_groups=sum(sum(x['FF_counts']) for x in audits),
        inactive_only_steps=[x['step'] for x in audits if not any(t>=.5 for t in x['times'])])
    r['environment']['GPU']=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
    r['timing']['assets_seconds']=time.perf_counter()-tick;r.update(status='ASSETS_PASS',active_stage='assets_complete');write(r,out);event('assets_pass',out=str(out))

def preflight(r,out):
    from .t1_time_gate_loss import run_cpu_tests
    from .t1_time_gate_evaluation import run_cpu_tests as evaluate_tests
    if not r['gates'].get('math'):
        r['active_stage']='CPU_math_gates';original=torch.autograd.grad
        def counted(*args,**kw):
            r['counts']['formula_autograd_calls']+=1;write(r,out)
            result=original(*args,**kw);r['counts']['formula_autograd_returns']+=1
            return result
        torch.autograd.grad=counted
        try:r['gates']['math']=run_cpu_tests()
        finally:torch.autograd.grad=original;write(r,out)
    r['gates']['evaluation_arithmetic']=evaluate_tests()
    r['counts']['evaluation_synthetic_fixture_attempts']=3 if r.get('implementation_fixture_failures') else 1
    r['counts']['evaluation_synthetic_fixture_passes']=2 if r.get('implementation_fixture_failures') else 1
    if r['gates']['math']['status']!='PASS' or r['counts']['formula_autograd_calls']!=r['gates']['math']['CPU_autograd_grad_calls']:raise RuntimeError('Math gate mismatch')
    r['new_execution_sources']={str(p):record(p) for p in sorted(PROJECT.glob('t1_time_gate_*.py')) if p.name!='t1_time_gate_report.py'}
    r.update(status='PREFLIGHT_PASS',active_stage='preflight_complete');write(r,out);event('preflight_pass',counts=r['counts'])

def load_model(r,name='C_gate',checkpoint=None,trainable=False):
    if name!='C_gate':raise RuntimeError('Only new C_gate model permitted')
    model,audit=load_t1()
    if checkpoint is not None:
        if file_sha256(checkpoint['path'])!=checkpoint['sha256']:raise RuntimeError('New checkpoint bytes differ')
        payload=torch.load(checkpoint['path'],map_location='cpu',weights_only=True,mmap=True)
        if payload['schema']!='t1_time_gate_branch_v1' or payload['arm']!='C_gate' or payload['additional_updates']!=500 or payload['cumulative_updates']!=1000:raise RuntimeError('Wrong branch endpoint')
        keys=model.load_state_dict(payload['model'],strict=True);del payload
        if keys.missing_keys or keys.unexpected_keys or state_sha256(model)!=checkpoint['state_sha256']:raise RuntimeError('Strict load differs')
    model.set_trainable(trainable).eval();Ledger(r).attach(model)
    return model,audit

def sample_all(r,out):
    arm=r['arms']['C_gate']
    if arm['completed_updates']!=500 or arm['status']!='PASS':raise RuntimeError('Fixed endpoint required')
    archive=load_archive(r['assets']['inputs']['path'],r['assets']['inputs']['sha256'])
    with np.load(r['assets']['noise_file']['path']) as f:noises={s:f[f'noise_seed{s}'].copy() for s in SEEDS}
    r['active_stage']='C_gate_sampling';model,loaded=load_model(r,checkpoint=arm['checkpoints']['500'])
    before=state_sha256(model);tick=time.perf_counter()
    try:
        for parent in (0,1):
            C=archive.cases[parent,12].constraints.copy()
            for seed in SEEDS:
                if r['counts']['rollout_attempts']>=16:raise RuntimeError('Hard16rollout budget')
                r['counts']['rollout_attempts']+=1;write(r,out);start=time.perf_counter();nf=r['counts']['model_forward_attempts'];sc=new_counts()
                try:raw,path,stats=clamped_sample(model,torch.from_numpy(noises[seed]).cuda(),torch.from_numpy(C).cuda(),sc)
                finally:
                    r['counts']['euler_updates']+=sc['euler_updates'];r['counts']['complete_rollouts']+=sc['complete_rollouts']
                    r.setdefault('sampling_attempts',[]).append(dict(model='C_gate',parent=parent,seed=seed,local_counts=dict(sc),returned=sc['complete_rollouts']==1));write(r,out)
                if r['counts']['model_forward_attempts']-nf!=50:raise RuntimeError('NFE differs')
                expected=np.broadcast_to(C.reshape(1,12,9),(51,12,9)).copy()
                if not np.array_equal(path[:,:12].view(np.uint32),expected.view(np.uint32)) or not np.isfinite(path).all():raise RuntimeError('C altered or nonfinite path')
                if not np.array_equal(path[0,12:].view(np.uint32),noises[seed][0,12:].view(np.uint32)):raise RuntimeError('Noise changed')
                target=out/'raw'/f'C_gate_P{parent}_K12_seed{seed}.npz'
                if target.exists():raise FileExistsError(target)
                export=time.perf_counter();np.savez_compressed(target,output=raw,path=path);export=time.perf_counter()-export
                r['generated'].append(dict(phase='confirmation',model='C_gate',parent=parent,K=12,seed=seed,variant='Raw',raw_path=str(target),raw_sha256=file_sha256(target),
                    output_sha256=array_hash(raw),sampling=stats,C_all51_FP32_bits_exact=True,all_states_finite=True,initial_free_Gaussian_exact=True,
                    costs=dict(complete_generation_wall_seconds=time.perf_counter()-start,NPZ_export_seconds=export,peak_cuda_allocated=stats['peak_cuda_allocated'])))
                write(r,out);event('sample',parent=parent,seed=seed,completed=r['counts']['complete_rollouts'])
        r['sampling_models']={'C_gate':dict(initial_state=before,final_state=state_sha256(model),seconds=time.perf_counter()-tick,strict_load=loaded)}
        if r['sampling_models']['C_gate']['final_state']!=before:raise RuntimeError('Frozen sampling changed')
        r['timing']['sampling_seconds']=time.perf_counter()-tick
    finally:del model;release()
    r.update(status='RAW_COMPLETE',active_stage='sampling_complete');write(r,out)

def score_all(r,out):
    from .t1_time_gate_evaluation import score_output,aggregate_results
    tick=time.perf_counter();bank,_=prepare_assets();bank['input_topology']=r['input_topology'];bank['context_audits']={int(k):v for k,v in r['assets']['context_audits'].items()}
    for row in [x for x in r['generated'] if x['model']=='C_gate']:
        with np.load(row['raw_path']) as f:raw=f['output'].copy()
        start=time.perf_counter();metric=score_output(raw,bank,row['parent'],ledger=r['evaluation_ledger'])
        r['scores'].append(dict(phase='confirmation',model='C_gate',parent=row['parent'],K=12,seed=row['seed'],variant='Raw',raw_path=row['raw_path'],metrics=metric,offline_scoring_seconds=time.perf_counter()-start))
        write(r,out);event('scored',parent=row['parent'],seed=row['seed'],new_scores=len(r['scores'])-32)
    r['summary']=aggregate_results(r['scores'],SEEDS);r['bindings']=r['summary']['bindings'];r['timing']['offline_audit_seconds']=time.perf_counter()-tick
    after=metadata(out);changed=[p for p,v in r['protection']['before'].items() if after.get(p)!=v]
    r['protection']['after']=dict(status='PASS' if not changed else 'FAIL',files=len(r['protection']['before']),changed=changed)
    r['protection']['critical_files_after']=verify_records(r['assets']['critical_files']);r['protection']['execution_after']=verify_records(r['new_execution_sources'])
    if changed or r['protection']['critical_files_after']['status']!='PASS' or r['protection']['execution_after']['status']!='PASS':raise RuntimeError('Protected assets or execution changed')
    c=r['counts'];expected=dict(optimizer_updates=500,optimizer_step_attempts=500,complete_rollouts=16,rollout_attempts=16,euler_updates=800,
        model_forward_attempts=4800,model_forward_returns=4800,backward_attempts=4000,backward_completed=4000,training_OT_calls=4000)
    if any(c[k]!=v for k,v in expected.items()):raise RuntimeError('Final fixed budget differs')
    g=r['arms']['C_gate']['gate_audits'];actual=dict(samples=4000,active_samples=sum(x['active_samples'] for x in g),active_FC_groups=sum(x['active_FC_groups'] for x in g),
        active_FF_groups=sum(x['active_FF_groups'] for x in g),total_FC_groups=sum(x['FC_group_denominator'] for x in g),total_FF_groups=sum(x['FF_group_denominator'] for x in g),inactive_only_steps=[x['step'] for x in g if not x['active_samples']])
    if actual!=r['preregistered_gate_exposure']:raise RuntimeError('Gate exposure mismatch')
    r['gates']['stream_identity']=dict(status='PASS',steps=500,samples=4000,reference='Full old A/B/C per-step audit dictionaries',sha256=seal(r['arms']['C_gate']['stream_audits']))
    if r['gates']['stream_identity']['sha256']!=r['reference_stream_sha256']:raise RuntimeError('Final stream differs')
    r['gates']['gate_exposure']=dict(status='PASS',actual=actual)
    r['timing']['training_seconds']=r['arms']['C_gate']['training_seconds']
    r.update(status='PASS_EXECUTION',active_stage='FINISHED',finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));write(r,out);event('complete',counts=c)

def execute(out,stage):
    if stage=='init':return initialize(out)
    r=json.loads((out/'results.json').read_text())
    try:
        if seal(r['protocol'])!=r['protocol_sha256']:raise RuntimeError('Protocol changed')
        configure_stable_runtime()
        if stage=='preflight':
            if r['status']!='ASSETS_PASS':
                if r['status']!='STOPPED' or r['counts']['model_forward_attempts']!=0 or r['counts']['optimizer_updates']!=0 or r['gates'].get('math',{}).get('status')!='PASS' or 't1_time_gate_evaluation.py' not in r.get('failure',{}).get('traceback',''):
                    raise RuntimeError('No automatic model/numerical retry')
                r.setdefault('implementation_fixture_failures',[]).append(r.pop('failure'))
                r['implementation_fixture_resolution']='Synthetic scalar expected -2 differed by FP64 roundoff; evaluation arithmetic assertion only. No model, optimizer, noise, hyperparameter, or geometry change; passed12 math-gradient checks reused, not repeated.'
            preflight(r,out)
        elif stage=='train':
            if r['status']!='PREFLIGHT_PASS':raise RuntimeError('No training retry')
            from .t1_interface_relations import load_source_relations
            from .t1_time_gate_train import train_arm
            relations=load_source_relations(r['assets']['inputs']['path'],expected_sha256=r['assets']['inputs']['sha256']);r['relations']=relations.audit
            saved=json.loads((Path(r['source_run'])/'results.json').read_text());reference=saved['arms']['A']['stream_audits']
            if seal(reference)!=r['reference_stream_sha256']:raise RuntimeError('Reference changed')
            import utils.ot_utils as data_module
            original_ot=data_module.optimal_sum_numpy
            def counted_ot(*args,**kwargs):
                r['counts']['training_OT_attempts']+=1
                answer=original_ot(*args,**kwargs)
                r['counts']['training_OT_calls']+=1
                return answer
            data_module.optimal_sum_numpy=counted_ot
            try:train_arm(r,out,relations,reference)
            finally:data_module.optimal_sum_numpy=original_ot;write(r,out)
            r['status']='TRAINING_COMPLETE';write(r,out)
        elif stage=='sample':
            if r['status']!='TRAINING_COMPLETE':raise RuntimeError('Training incomplete')
            if verify_records(r['new_execution_sources'])['status']!='PASS':raise RuntimeError('Execution changed')
            sample_all(r,out)
        elif stage=='score':
            if r['status']!='RAW_COMPLETE':raise RuntimeError('Raw incomplete')
            score_all(r,out)
        elif stage=='report':
            if r['status']!='PASS_EXECUTION':raise RuntimeError('Not complete')
            from .t1_time_gate_report import deliver_report
            r['delivery']=deliver_report(r,out);write(r,out)
    except BaseException as error:
        r.update(status='STOPPED',failure=dict(stage=stage,error=repr(error),traceback=traceback.format_exc(),automatic_retry=False));write(r,out);event('stopped',error=repr(error),counts=r['counts']);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);p.add_argument('--stage',required=True,choices=['init','preflight','train','sample','score','report']);a=p.parse_args();execute(a.out.resolve(),a.stage)
