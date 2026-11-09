"""Fixed gated FC/FF factorial experiment. Two new branches; all four confirm."""
from native_t1.runtime import configure_stable_runtime
from pathlib import Path
import argparse,contextlib,copy,gc,hashlib,json,platform,re,subprocess,time,traceback
import numpy as np
import torch
from native_t1.artifacts import atomic_json,file_sha256,state_sha256,array_hash
from native_t1.checkpoint import load_t1,T1_FILE_SHA256,T1_STATE_SHA256,DEFAULT_CHECKPOINT,DEFAULT_CONFIG
from native_t1.data import load_archive,NativeTrainingStream,native_collate,model_inputs
from native_t1.model import native_execution
from native_t1.sampling import clamped_sample,make_noise,new_counts
from .guided_native_geometry_run import prepare_assets,record,verify_records
from .t1_interface_run import cpu

PROJECT=Path(__file__).resolve().parent;REPO=PROJECT.parents[1]
OLD_A=PROJECT/'outputs/t1_interface_3arm_20260929_181500'
OLD_GATE=PROJECT/'outputs/t1_time_gate_20260929_213000'
TEACHER=PROJECT/'outputs/c_relation_diagnosis_20260929_204330'
DEV_SEEDS=tuple(range(9501,9509));CONF_SEEDS=tuple(range(9601,9609))
MODELS=('M00','M10','M01','M11');NEW=('M10','M01');DIAGNOSTIC_STEPS=(1,100,250,500)
EXPECTED={'M00':'43e56e107b0c6d28766aebd8cf8a0a764d8d52ef62ba2ca6e756d881634f876b','M11':'4e473de1ea35617c1e6a2b938e0a85b4ab006a3a266d6a8ad79360c0c349a3b5'}

def seal(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(r,out):atomic_json(out/'results.json',r)
def event(name,**kw):print(json.dumps(dict(event=name,**kw),allow_nan=False),flush=True)
def release():gc.collect();torch.cuda.empty_cache()
def metadata(out):
    files=[p for p in (PROJECT/'outputs').rglob('*') if p.is_file() and not p.is_relative_to(out)]
    files += [p for p in PROJECT.glob('*.py') if not p.name.startswith('t1_gated_factorial_')]
    files += list((REPO/'native_t1').glob('*.py'))+list((REPO/'models').glob('*.py'))+[REPO/'utils/ot_utils.py',DEFAULT_CONFIG,Path('/mnt/c/Users/warpwang/Downloads/last.pt')]
    files += list((PROJECT/'assets').rglob('*.npz'))
    return {str(p):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sorted(set(files))}

class Ledger:
    def __init__(self,r):self.r=r
    def attach(self,model):
        def before(module,args):
            c=self.r['counts'];stage=self.r['active_stage']
            if c['model_forward_attempts']>=13082:raise RuntimeError('Hard model forward cap')
            if stage.startswith('teacher_'):
                if c['teacher_forward_attempts']>=80:raise RuntimeError('Hard80 teacher forward cap')
                c['teacher_forward_attempts']+=1
            c['model_forward_attempts']+=1;c['forward_by_stage'][stage]=c['forward_by_stage'].get(stage,0)+1
        def after(module,args,output):
            self.r['counts']['model_forward_returns']+=1
            if self.r['active_stage'].startswith('teacher_'):self.r['counts']['teacher_forward_returns']+=1
        model.register_forward_pre_hook(before);model.register_forward_hook(after)

@contextlib.contextmanager
def count_ot(r,prefix):
    import utils.ot_utils as ot
    original=ot.optimal_sum_numpy
    def counted(*args,**kwargs):
        r['counts'][prefix+'_OT_attempts']+=1
        answer=original(*args,**kwargs);r['counts'][prefix+'_OT_calls']+=1;return answer
    ot.optimal_sum_numpy=counted
    try:yield
    finally:ot.optimal_sum_numpy=original

def seed_audit(out):
    # Semantic seed fields / output names only; numerical score substrings are not seed use.
    hits=[];files=0
    key_re=re.compile(r'"(?:seed|noise_seed|base_seed)"\s*:\s*960[1-8](?![0-9])|\bseed[_-]?960[1-8]\b')
    for p in (PROJECT/'outputs').rglob('*'):
        if not p.is_file() or p.is_relative_to(out):continue
        if re.search(r'seed[_-]?960[1-8](?![0-9])',p.name):hits.append(str(p))
        if p.suffix in ('.json','.md'):
            files+=1
            if key_re.search(p.read_text(encoding='utf-8-sig',errors='strict')):hits.append(str(p))
    if hits:raise RuntimeError('Requested confirmation seeds have semantic usage; inspect before registering alternative: '+str(sorted(set(hits))))
    return dict(status='PASS',seeds=list(CONF_SEEDS),semantic_usage_matches=[],text_files_scanned=files,output_names_checked=True,before_any_model_call=True,
        rule='Noise-only generation is not model-result use; no semantic seed/output records found. Decimal metric substrings and19601 teacher seeds excluded.')

def initialize(out):
    if out.exists():raise FileExistsError(out)
    a=json.loads((OLD_A/'results.json').read_text());g=json.loads((OLD_GATE/'results.json').read_text())
    if a['status']!='PASS_EXECUTION' or g['status']!='PASS_EXECUTION':raise RuntimeError('Source runs incomplete')
    if a['protocol']['initial']!=g['protocol']['initial']:raise RuntimeError('Source initial identity differs')
    initial=copy.deepcopy(a['protocol']['initial'])
    if initial['file_sha256']!=T1_FILE_SHA256 or initial['state_sha256']!=T1_STATE_SHA256 or Path(initial['path']).resolve()!=DEFAULT_CHECKPOINT.resolve():raise RuntimeError('Wrong common original T1')
    if a['arms']['A']['stream_audits']!=g['arms']['C_gate']['stream_audits']:raise RuntimeError('A/C_gate source streams differ')
    if a['protocol']['runtime']!=g['protocol']['runtime']:raise RuntimeError('Source runtime differs')
    out.mkdir(parents=True);(out/'raw').mkdir()
    train=copy.deepcopy(a['protocol']['training']);train.update(arms=list(NEW),total_update_cap=1000,checkpoint_additional_steps=[500],checkpoint_cumulative_steps=[1000],FC_weight=.1,FF_weight=.1)
    protocol=dict(schema='t1_gated_factorial_v1',initial=initial,models={m:dict(FC=m[1]=='1',FF=m[2]=='1') for m in MODELS},training=train,
        gate_threshold=.5,gate='Per sample stored FP32 t>=0.5; full effective8 valid-group denominators include inactive samples; no compensation',
        development_seeds=list(DEV_SEEDS),confirmation_seeds=list(CONF_SEEDS),runtime=a['protocol']['runtime'],
        budget=dict(new_updates_each=500,total_updates=1000,training_forwards=8000,training_backwards=8000,model_gate_forwards=2,model_gate_backwards=2,
            development_new_rollouts=32,confirmation_new_rollouts=64,replay_cap=4,total_rollout_cap=100,planned_teacher_forwards=60,teacher_forward_cap=80,total_forward_cap=13082),
        evaluation=dict(N=112,K=12,free=100,steps=50,postprocessing=False,guidance=False,original_scorer=True,
            phase_separation=True,confirmation_all_four_unconditional=True,teacher='Existing K12 bank20states; M00 reuse if all identities match'),
        interpretation='Single training seed; generated noise trials are not independent retrainings. Loss switches change auxiliary total strength. No equal-dose control or generalization claim.',
        stop='Two new fixed500 endpoints and all four confirmation models only. No weight/gate/module changes, no Phase3.')
    r=dict(status='INITIALIZING',active_stage='assets',protocol=protocol,protocol_sha256=seal(protocol),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        source_runs=dict(M00=str(OLD_A),M11=str(OLD_GATE),teacher=str(TEACHER)),assets={},arms={},generated=[],scores=[],replays=[],gates={},teacher_rows=[],timing={},evaluation_ledger={},
        counts=dict(model_forward_attempts=0,model_forward_returns=0,forward_by_stage={},backward_attempts=0,backward_completed=0,training_backward_attempts=0,training_backward_completed=0,
            gate_backward_attempts=0,gate_backward_completed=0,teacher_forward_attempts=0,teacher_forward_returns=0,optimizer_step_attempts=0,optimizer_updates=0,
            training_OT_attempts=0,training_OT_calls=0,gate_OT_attempts=0,gate_OT_calls=0,formula_autograd_calls=0,formula_autograd_returns=0,
            rollout_attempts=0,complete_rollouts=0,euler_updates=0,formal_rollouts=0,replay_rollouts=0,development_rollouts=0,confirmation_rollouts=0,geometry_guidance_calls=0,geometry_repairs=0),
        protection=dict(before=metadata(out)),AGENTS='No applicable AGENTS.md found in repository or ancestor searches',
        git_status_before=subprocess.run(['git','status','--short'],cwd=REPO,capture_output=True,text=True).stdout,
        environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__))
    write(r,out);tick=time.perf_counter();r['seed_audit']=seed_audit(out)
    r['assets']['inputs']=record(a['relations']['inputs']['path'])
    if r['assets']['inputs']['sha256']!=a['relations']['inputs']['sha256'] or r['assets']['inputs']!=g['assets']['inputs']:raise RuntimeError('Source input changed')
    r['assets']['parent_ids']=a['assets']['parent_ids'];r['input_topology']=a['input_topology'];r['assets']['context_audits']=a['assets']['context_audits']
    r['assets']['development_noise']=record(OLD_A/'fixed_noise.npz')
    if r['assets']['development_noise']!=g['assets']['noise_file']:raise RuntimeError('C_gate used other dev noise')
    r['assets']['initial']=record(DEFAULT_CHECKPOINT)
    if r['assets']['initial']['sha256']!=T1_FILE_SHA256:raise RuntimeError('Initial file hash differs')
    r['assets']['endpoints']={};r['assets']['sources']={}
    for name,source,saved,arm in [('M00',OLD_A,a,'A'),('M11',OLD_GATE,g,'C_gate')]:
        cp=copy.deepcopy(saved['arms'][arm]['checkpoints']['500']);actual=record(cp['path'])
        if actual['sha256']!=EXPECTED[name] or actual['sha256']!=cp['sha256']:raise RuntimeError('Endpoint SHA mismatch:'+name)
        r['assets']['endpoints'][name]=cp
        r['assets']['sources'][name]=dict(results=record(source/'results.json'),report=record(source/('THREE_ARM_INTERFACE_REPORT.md' if name=='M00' else 'TIME_GATE_RELATION_REPORT.md')),
            training_protocol=saved['protocol']['training'],runtime=saved['protocol']['runtime'],source_arm=arm,available_raw_count=0,available_score_count=0)
        rows=[x for x in saved['generated'] if x['model']==arm and x['seed'] in DEV_SEEDS and x['K']==12]
        metrics=[x for x in saved['scores'] if x['model']==arm and x['seed'] in DEV_SEEDS and x['K']==12]
        if len(rows)!=16 or len(metrics)!=16 or {(x['parent'],x['seed']) for x in rows}!={(p,s) for p in (0,1) for s in DEV_SEEDS}:raise RuntimeError('Missing actual dev records')
        for row in rows:
            rec=record(row['raw_path'])
            if rec['sha256']!=row['raw_sha256'] or not row['C_all51_FP32_bits_exact'] or not row['initial_free_Gaussian_exact'] or row['sampling']['nfe']!=50:raise RuntimeError('Raw dev contract differs')
            new=copy.deepcopy(row);new.update(model=name,phase='development',reused=True,source_model=arm);r['generated'].append(new)
        r['assets']['sources'][name].update(available_raw_count=16,available_score_count=16)
    r['reference_stream_sha256']=seal(a['arms']['A']['stream_audits'])
    r['preregistered_gate_exposure']=copy.deepcopy(g['gates']['gate_exposure']['actual'])
    if r['preregistered_gate_exposure']!=dict(samples=4000,active_samples=1972,active_FC_groups=17682,active_FF_groups=114376,total_FC_groups=36000,total_FF_groups=232000,inactive_only_steps=[28,330,358]):raise RuntimeError('Expected exposure mismatch')
    arrays={f'noise_seed{s}':make_noise(s).numpy() for s in CONF_SEEDS};np.savez_compressed(out/'confirmation_noise.npz',**arrays)
    r['assets']['confirmation_noise']=record(out/'confirmation_noise.npz');r['confirmation_noise_ledger']={str(s):dict(shape=[1,112,9],dtype='float32',sha256=array_hash(arrays[f'noise_seed{s}']),free_sha256=array_hash(arrays[f'noise_seed{s}'][:,12:])) for s in CONF_SEEDS}
    critical={str(p):record(p) for p in list((REPO/'native_t1').glob('*.py'))+list((REPO/'models').glob('*.py'))+[REPO/'utils/ot_utils.py',DEFAULT_CONFIG,PROJECT/'t1_interface_relations.py',PROJECT/'t1_time_gate_loss.py',PROJECT/'t1_time_gate_train.py',PROJECT/'t1_native_confirmation_evaluation.py',PROJECT/'t1_native_confirmation_topology.py',PROJECT/'c_relation_math.py',PROJECT/'c_relation_teacher.py',PROJECT/'c_relation_run.py',Path('/mnt/c/Users/warpwang/Downloads/last.pt')]}
    for k in ('inputs','initial','development_noise'):v=r['assets'][k];critical[v['path']]=v
    for v in r['assets']['endpoints'].values():critical[v['path']]=v
    for v in r['assets']['sources'].values():
        for k in ('results','report'):critical[v[k]['path']]=v[k]
    for row in r['generated']:critical[row['raw_path']]=dict(path=row['raw_path'],bytes=Path(row['raw_path']).stat().st_size,sha256=row['raw_sha256'])
    r['assets']['critical_files']=critical
    r['environment']['GPU']=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
    r['timing']['asset_seconds']=time.perf_counter()-tick;r.update(status='ASSETS_PASS',active_stage='assets_complete');write(r,out);event('assets_pass',out=str(out),seed_audit=r['seed_audit'])

def load_model(r,name,trainable=False,initial=False):
    if name not in MODELS or (initial and name not in NEW):raise ValueError('Invalid factorial model')
    model,audit=load_t1()
    if not initial:
        cp=r['assets']['endpoints'][name] if name in ('M00','M11') else r['arms'][name]['checkpoints']['500']
        if file_sha256(cp['path'])!=cp['sha256']:raise RuntimeError('Checkpoint hash changed')
        payload=torch.load(cp['path'],map_location='cpu',weights_only=True,mmap=True)
        expected_arm={'M00':'A','M11':'C_gate'}.get(name,name);schema={'M00':'t1_interface_branch_v1','M11':'t1_time_gate_branch_v1'}.get(name,'t1_gated_factorial_branch_v1')
        if payload['schema']!=schema or payload['arm']!=expected_arm or payload['additional_updates']!=500 or payload['cumulative_updates']!=1000:raise RuntimeError('Wrong fixed endpoint')
        keys=model.load_state_dict(payload['model'],strict=True);del payload
        if keys.missing_keys or keys.unexpected_keys or state_sha256(model)!=cp['state_sha256']:raise RuntimeError('Strict endpoint tensor identity differs')
        audit=dict(common_architecture=audit,endpoint=cp,strict=True)
    model.set_trainable(trainable).eval()
    if sum(p.numel() for p in model.parameters())!=130251267:raise RuntimeError('Parameter count differs')
    Ledger(r).attach(model);return model,audit

def sample_one(r,out,model,name,parent,seed,noise,C,phase,reference=None):
    c=r['counts']
    if c['rollout_attempts']>=100 or (reference is not None and c['replay_rollouts']>=4):raise RuntimeError('Hard100 trajectories/4replays')
    c['rollout_attempts']+=1;write(r,out);tick=time.perf_counter();nf=c['model_forward_attempts'];sc=new_counts()
    try:raw,path,stats=clamped_sample(model,torch.from_numpy(noise).cuda(),torch.from_numpy(C).cuda(),sc)
    finally:
        c['euler_updates']+=sc['euler_updates'];c['complete_rollouts']+=sc['complete_rollouts']
        r.setdefault('sampling_attempts',[]).append(dict(model=name,parent=parent,seed=seed,phase=phase,replay=reference is not None,counts=dict(sc)));write(r,out)
    if c['model_forward_attempts']-nf!=50:raise RuntimeError('Wrong sampling NFE')
    expected=np.broadcast_to(C.reshape(1,12,9),(51,12,9)).copy()
    if not np.array_equal(path[:,:12].view(np.uint32),expected.view(np.uint32)) or not np.isfinite(path).all():raise RuntimeError('C changed or nonfinite sampling')
    if not np.array_equal(path[0,12:].view(np.uint32),noise[0,12:].view(np.uint32)):raise RuntimeError('Free Gaussian differs')
    row=dict(model=name,parent=parent,seed=seed,K=12,phase=phase,variant='Raw',sampling=stats,C_all51_FP32_bits_exact=True,initial_free_Gaussian_exact=True,all_states_finite=True,output_sha256=array_hash(raw),path_sha256=array_hash(path),reused=False,seconds=time.perf_counter()-tick)
    if reference is not None:
        with np.load(reference['raw_path']) as f:
            if not np.array_equal(raw.view(np.uint32),f['output'].view(np.uint32)) or not np.array_equal(path.view(np.uint32),f['path'].view(np.uint32)):raise RuntimeError('Deterministic replay differs from registered full trajectory')
        row.update(reference_path=reference['raw_path'],raw_and_all51_bitwise_match=True);r['replays'].append(row);c['replay_rollouts']+=1
    else:
        target=out/'raw'/f'{phase}_{name}_P{parent}_K12_seed{seed}.npz'
        if target.exists():raise FileExistsError(target)
        np.savez_compressed(target,output=raw,path=path);row.update(raw_path=str(target),raw_sha256=file_sha256(target));r['generated'].append(row)
        c['formal_rollouts']+=1;c[phase+'_rollouts']+=1
    write(r,out);event('sample',model=name,parent=parent,seed=seed,phase=phase,replay=reference is not None,formal=c['formal_rollouts']);return row

def preflight(r,out,relations):
    from .t1_gated_factorial_loss import run_cpu_tests,switch_loss
    from .t1_gated_factorial_evaluation import run_cpu_tests as metric_tests
    from .t1_interface_relations import prepare_relations
    from .t1_gated_factorial_train import magnitude,selected
    r['active_stage']='CPU_gates';original=torch.autograd.grad
    def counted(*args,**kw):
        r['counts']['formula_autograd_calls']+=1;write(r,out);answer=original(*args,**kw);r['counts']['formula_autograd_returns']+=1;return answer
    torch.autograd.grad=counted
    try:r['gates']['CPU_switch']=run_cpu_tests()
    finally:torch.autograd.grad=original;write(r,out)
    r['gates']['metric_arithmetic']=metric_tests()
    from .t1_gated_factorial_teacher import load_fixed_teacher
    if (TEACHER/'teacher_bank.npz').exists():
        teacher_samples,reused,teacher_audit=load_fixed_teacher(TEACHER,relations)
        r['gates']['teacher_reuse']=teacher_audit
        for key in ('source_report','source_results','bank_file'):
            rec=teacher_audit[key];r['assets']['critical_files'][rec['path']]=rec
        del teacher_samples,reused
    else:r['gates']['teacher_reuse']=dict(status='NOT_AVAILABLE',new_teacher_forwards=0)
    write(r,out)
    archive=load_archive(r['assets']['inputs']['path'],r['assets']['inputs']['sha256'])
    with count_ot(r,'gate'):samples=NativeTrainingStream(archive,'Native_correct',seed=10).next_effective_batch()
    labels=[relations.prepare_sample_labels(s) for s in samples];index=next(i for i,s in enumerate(samples) if float(s['t'])>=.5)
    fc_den=sum(x['FC_count'] for x in labels);ff_den=sum(x['FF_count'] for x in labels);reference=None
    r['active_stage']='model_gates';r['gates']['model_terms']={}
    for name in NEW:
        model,loaded=load_model(r,name,trainable=True,initial=True);before=state_sha256(model)
        batch=native_collate([samples[index]],'cuda');pack=prepare_relations(labels[index],'cuda');batch['context'].requires_grad_(True);pack['fc_anchors'].requires_grad_(True)
        try:
            with native_execution(model,coordinates=batch['xt']) as runtime:
                velocity=model(*model_inputs(batch)).float()
                if reference is not None and not torch.equal(velocity.detach().view(torch.int32).cpu(),reference):raise RuntimeError('New branch initial forward differs')
                reference=velocity.detach().view(torch.int32).cpu().clone()
                loss,parts,audit=switch_loss(velocity.sum()*0,batch,velocity,pack,fc_den,ff_den,name=='M10',name=='M01')
                r['counts']['backward_attempts']+=1;r['counts']['gate_backward_attempts']+=1;loss.backward();r['counts']['backward_completed']+=1;r['counts']['gate_backward_completed']+=1
            grads={n:magnitude(p.grad) for n,p in selected(model).items()}
            if not all(v.get('finite',False) for v in grads.values()) or not any(v.get('max_abs',0)>0 for v in grads.values()) or batch['context'].grad is not None or pack['fc_anchors'].grad is not None:raise RuntimeError('Auxiliary model/C gradient gate failed')
            model.zero_grad(set_to_none=True)
            if state_sha256(model)!=before or before!=T1_STATE_SHA256:raise RuntimeError('Zero-update gate changed weights')
            r['gates']['model_terms'][name]=dict(status='PASS',initial_state_sha256=before,output_sha256=array_hash(reference.numpy()),gradients=grads,C_gradient=None,anchors_gradient=None,runtime=runtime,gate=audit,strict_load=loaded)
        finally:model.zero_grad(set_to_none=True);del model;release()
    r['gates']['new_initial_forward_bitwise']=True
    with np.load(r['assets']['development_noise']['path']) as f:noise=f['noise_seed9501'].copy()
    for name in ('M00','M11'):
        r['active_stage']='historical_replay_'+name;model,_=load_model(r,name);before=state_sha256(model)
        try:
            prior=next(x for x in r['generated'] if x['model']==name and x['parent']==0 and x['seed']==9501)
            sample_one(r,out,model,name,0,9501,noise,archive.cases[0,12].constraints,'development',reference=prior)
            if state_sha256(model)!=before:raise RuntimeError('Historical replay state changed')
        finally:del model;release()
    r['relations']=relations.audit;r['new_execution_sources']={str(p):record(p) for p in sorted(PROJECT.glob('t1_gated_factorial_*.py')) if p.name!='t1_gated_factorial_report.py'}
    r.update(status='PREFLIGHT_PASS',active_stage='preflight_complete');write(r,out);event('preflight_pass',counts=r['counts'])

def sample_all(r,out):
    if any(r['arms'][a]['status']!='PASS' or r['arms'][a]['completed_updates']!=500 for a in NEW):raise RuntimeError('Both fixed endpoints required')
    archive=load_archive(r['assets']['inputs']['path'],r['assets']['inputs']['sha256'])
    r['frozen_evaluation_models']={m:(r['assets']['endpoints'][m] if m in ('M00','M11') else r['arms'][m]['checkpoints']['500']) for m in MODELS};write(r,out)
    tick=time.perf_counter()
    for phase,names,seeds,noise_key in [('development',NEW,DEV_SEEDS,'development_noise'),('confirmation',MODELS,tuple(r['protocol']['confirmation_seeds']),'confirmation_noise')]:
        with np.load(r['assets'][noise_key]['path']) as f:noises={s:f[f'noise_seed{s}'].copy() for s in seeds}
        for name in names:
            r['active_stage']=phase+'_sampling_'+name;model,load=load_model(r,name);before=state_sha256(model);start=time.perf_counter()
            try:
                for parent in (0,1):
                    C=archive.cases[parent,12].constraints.copy()
                    for seed in seeds:sample_one(r,out,model,name,parent,seed,noises[seed],C,phase)
                if phase=='development':
                    first=next(x for x in r['generated'] if x['model']==name and x['parent']==0 and x['seed']==9501 and x['phase']==phase)
                    r['active_stage']='new_repeat_'+name;sample_one(r,out,model,name,0,9501,noises[9501],archive.cases[0,12].constraints,phase,reference=first)
                after=state_sha256(model)
                if after!=before:raise RuntimeError('Sampling changed frozen model')
                r.setdefault('sampling_models',[]).append(dict(model=name,phase=phase,state_before=before,state_after=after,seconds=time.perf_counter()-start,strict_load=load));write(r,out)
            finally:del model;release()
    r['timing']['sampling_seconds']=time.perf_counter()-tick;r.update(status='RAW_COMPLETE',active_stage='all_sampling_complete');write(r,out)

def teacher_all(r,out,relations):
    from .t1_gated_factorial_teacher import load_fixed_teacher,evaluate_state
    tick=time.perf_counter()
    if not (TEACHER/'teacher_bank.npz').exists():
        r['teacher_status']=dict(status='NOT_AVAILABLE',reason='Existing fixed bank missing; no rebuilding');r['status']='TEACHER_COMPLETE';write(r,out);return
    samples,reused,audit=load_fixed_teacher(TEACHER,relations)
    r['teacher_rows']=reused;r['teacher_status']=audit;write(r,out)
    for name in ('M10','M01','M11'):
        model,load=load_model(r,name);before=state_sha256(model);r['active_stage']='teacher_'+name
        try:
            for sample in samples:
                row,runtime=evaluate_state(model,sample,name)
                r['teacher_rows'].append(row)
                r.setdefault('teacher_runtime',{})[name]=runtime
                if len(r['teacher_rows'])%5==0:write(r,out)
            if state_sha256(model)!=before:raise RuntimeError('Teacher altered model')
        finally:del model;release()
    if len(r['teacher_rows'])!=80 or r['counts']['teacher_forward_returns']!=60:raise RuntimeError('Compact teacher matrix differs')
    r['teacher_status'].update(status='PASS',reused_M00_rows=20,new_model_forwards=60);r['timing']['teacher_seconds']=time.perf_counter()-tick
    r.update(status='TEACHER_COMPLETE',active_stage='teacher_complete');write(r,out);event('teacher_complete',counts=r['counts'])

def score_all(r,out):
    from .t1_gated_factorial_evaluation import score_output,aggregate_results
    tick=time.perf_counter();bank,_=prepare_assets();bank['input_topology']=r['input_topology'];bank['context_audits']={int(k):v for k,v in r['assets']['context_audits'].items()}
    if r['scores']:raise RuntimeError('No automatic score-stage rerun')
    for name in ('M00','M11'):
        source=r['assets']['sources'][name];saved=json.loads(Path(source['results']['path']).read_text())
        for row in saved['scores']:
            if row['model']==source['source_arm'] and row['seed'] in DEV_SEEDS and row['K']==12:
                new=copy.deepcopy(row);new.update(phase='development',model=name,reused=True,source_model=row['model']);r['scores'].append(new)
    if len(r['scores'])!=32:raise RuntimeError('Existing raw scores missing')
    for row in [x for x in r['generated'] if not x['reused']]:
        with np.load(row['raw_path']) as f:raw=f['output'].copy()
        start=time.perf_counter();metric=score_output(raw,bank,row['parent'],ledger=r['evaluation_ledger'])
        r['scores'].append(dict(phase=row['phase'],model=row['model'],parent=row['parent'],K=12,seed=row['seed'],variant='Raw',raw_path=row['raw_path'],metrics=metric,reused=False,offline_scoring_seconds=time.perf_counter()-start))
        if (len(r['scores'])-32)%4==0:write(r,out);event('scored',count=len(r['scores'])-32)
    r['summaries']={phase:aggregate_results(r['scores'],phase,seeds) for phase,seeds in [('development',DEV_SEEDS),('confirmation',r['protocol']['confirmation_seeds'])]}
    r['timing']['offline_scoring_seconds']=time.perf_counter()-tick
    after=metadata(out);changed=[p for p,v in r['protection']['before'].items() if after.get(p)!=v]
    r['protection']['after']=dict(status='PASS' if not changed else 'FAIL',files=len(r['protection']['before']),changed=changed)
    r['protection']['critical_files_after']=verify_records(r['assets']['critical_files']);r['protection']['execution_after']=verify_records(r['new_execution_sources'])
    if changed or any(r['protection'][k]['status']!='PASS' for k in ('critical_files_after','execution_after')):raise RuntimeError('Protected files changed')
    for name in NEW:
        arm=r['arms'][name]
        if seal(arm['stream_audits'])!=r['reference_stream_sha256']:raise RuntimeError('Final stream mismatch')
        gates=arm['gate_audits'];actual=dict(samples=4000,active_samples=sum(x['active_samples'] for x in gates),active_FC_groups=sum(x['active_FC_groups'] for x in gates),active_FF_groups=sum(x['active_FF_groups'] for x in gates),total_FC_groups=sum(x['FC_group_denominator'] for x in gates),total_FF_groups=sum(x['FF_group_denominator'] for x in gates),inactive_only_steps=[x['step'] for x in gates if x['active_samples']==0])
        if actual!=r['preregistered_gate_exposure']:raise RuntimeError('Final gate exposure mismatch')
        r['gates'][name+'_stream_and_exposure']=dict(status='PASS',actual=actual)
    c=r['counts'];expected=dict(optimizer_updates=1000,optimizer_step_attempts=1000,training_backward_attempts=8000,training_backward_completed=8000,gate_backward_attempts=2,gate_backward_completed=2,complete_rollouts=100,rollout_attempts=100,euler_updates=5000,formal_rollouts=96,replay_rollouts=4,development_rollouts=32,confirmation_rollouts=64,training_OT_calls=8000,gate_OT_calls=8)
    if any(c[k]!=v for k,v in expected.items()) or c['model_forward_attempts']!=13002+c['teacher_forward_attempts'] or c['model_forward_attempts']!=c['model_forward_returns']:raise RuntimeError('Final actual budget mismatch')
    r['timing']['training_seconds']=sum(a['training_seconds'] for a in r['arms'].values());r['timing']['checkpoint_seconds']=sum(v['save_and_hash_seconds'] for a in r['arms'].values() for v in a['checkpoints'].values())
    r.update(status='PASS_EXECUTION',active_stage='FINISHED',finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));write(r,out);event('complete',counts=c)

def execute(out,stage):
    if stage=='init':return initialize(out)
    r=json.loads((out/'results.json').read_text())
    try:
        if seal(r['protocol'])!=r['protocol_sha256']:raise RuntimeError('Frozen protocol changed')
        configure_stable_runtime()
        from .t1_interface_relations import load_source_relations
        relations=load_source_relations(r['assets']['inputs']['path'],expected_sha256=r['assets']['inputs']['sha256'])
        if stage=='preflight':
            if r['status']!='ASSETS_PASS':raise RuntimeError('No preflight model retry')
            preflight(r,out,relations)
        elif stage=='train':
            if r['status']!='PREFLIGHT_PASS':raise RuntimeError('No training retry')
            from .t1_gated_factorial_train import train_arm
            a=json.loads((OLD_A/'results.json').read_text());g=json.loads((OLD_GATE/'results.json').read_text());reference=a['arms']['A']['stream_audits']
            if reference!=g['arms']['C_gate']['stream_audits'] or seal(reference)!=r['reference_stream_sha256']:raise RuntimeError('Reference stream changed')
            with count_ot(r,'training'):
                for name in NEW:train_arm(r,out,relations,reference,name)
            r.update(status='TRAINING_COMPLETE',active_stage='training_complete');write(r,out)
        elif stage=='sample':
            if r['status']!='TRAINING_COMPLETE':raise RuntimeError('Training incomplete')
            if verify_records(r['new_execution_sources'])['status']!='PASS':raise RuntimeError('Execution changed')
            sample_all(r,out)
        elif stage=='teacher':
            if r['status']!='RAW_COMPLETE':raise RuntimeError('All generation must finish before teacher')
            teacher_all(r,out,relations)
        elif stage=='score':
            if r['status']!='TEACHER_COMPLETE':raise RuntimeError('Teacher stage not recorded')
            score_all(r,out)
        elif stage=='report':
            if r['status']!='PASS_EXECUTION':raise RuntimeError('Execution incomplete')
            from .t1_gated_factorial_report import deliver_report
            r['delivery']=deliver_report(r,out);write(r,out)
    except BaseException as error:
        r.update(status='STOPPED',failure=dict(stage=stage,error=repr(error),traceback=traceback.format_exc(),automatic_retry=False));write(r,out);event('stopped',error=repr(error),counts=r['counts']);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);p.add_argument('--stage',required=True,choices=['init','preflight','train','sample','teacher','score','report']);a=p.parse_args();execute(a.out.resolve(),a.stage)
