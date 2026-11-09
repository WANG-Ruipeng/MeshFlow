"""Fixed three-arm Native T1 branch training; no guidance or repair."""
from native_t1.runtime import configure_stable_runtime
import argparse,contextlib,copy,gc,hashlib,json,platform,random,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from native_t1.artifacts import atomic_json,file_sha256,state_sha256,array_hash
from native_t1.checkpoint import load_t1,T1_FILE_SHA256,T1_STATE_SHA256,DEFAULT_CHECKPOINT,DEFAULT_CONFIG
from native_t1.data import load_archive,NativeTrainingStream,native_collate,model_inputs
from native_t1.model import native_execution
from native_t1.sampling import clamped_sample,make_noise,new_counts
from .guided_native_geometry_run import prepare_assets,record,verify_records

PROJECT=Path(__file__).resolve().parent
REPO=PROJECT.parents[1]
SOURCE=PROJECT/'outputs/native_patch_20260929_010000'
SEEDS=tuple(range(9501,9509));ARMS=('A','B','C');DIAGNOSTIC_STEPS=(1,100,250,500)
INPUT_SHA='e80d52e0906bc762475f35a2526cb99ff5b52858c9d2e6a38d09018e5d5aea73'


def seal(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(r,out):atomic_json(out/'results.json',r)
def event(name,**kw):print(json.dumps(dict(event=name,**kw),allow_nan=False),flush=True)
def release():gc.collect();torch.cuda.empty_cache()
def cpu(v):
    if torch.is_tensor(v):return v.detach().cpu()
    if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
    if isinstance(v,list):return [cpu(x) for x in v]
    if isinstance(v,tuple):return tuple(cpu(x) for x in v)
    return v

def metadata(out):
    files=[x for x in (PROJECT/'outputs').rglob('*') if x.is_file() and not x.is_relative_to(out)]
    files += [x for x in PROJECT.glob('*.py') if not x.name.startswith('t1_interface_')]
    files += list((REPO/'native_t1').glob('*.py'))+list((REPO/'models').glob('*.py'))+[DEFAULT_CONFIG,Path('/mnt/c/Users/warpwang/Downloads/last.pt')]
    files += [x for x in (PROJECT/'assets').rglob('*.npz')]
    return {str(x):dict(bytes=x.stat().st_size,mtime_ns=x.stat().st_mtime_ns) for x in sorted(set(files))}


def protocol():
    return dict(schema='t1_interface_3arm_v1',initial=dict(path=str(DEFAULT_CHECKPOINT),file_sha256=T1_FILE_SHA256,state_sha256=T1_STATE_SHA256,old_updates=500),
      arms=dict(A='Continue original Native/FM',B='Known-only boundary Linear(2,768,bias=False),zero weight1536; original FM',C='Original Native; FM+0.1FC+0.1FF, endpoint shared-source-vertex losses'),
      training=dict(seed=10,additional_updates_each=500,total_update_cap=1500,effective_batch=8,microbatch=1,K=[4,8,12],parents=2,
          stream='Unchanged NativeTrainingStream Native_correct; independent RandomState streams reset identically',
          OT='Original free-only optimal_sum_numpy reorders noise Z only; target Y ordering and labels follow source_face_ids/corner_permutations',
          time='original logit-normal',optimizer='fresh AdamW each arm',lr=1e-5,betas=[.9,.95],weight_decay=0,clip=1,
          scheduler=False,EMA=False,dropout=False,eval_modules=True,all_parameters_trainable=True,
          FC_weight=.1,FF_weight=.1,aux_scale='original coordinates and per-coordinate MSE; no bbox normalization',
          auxiliary_accumulation='Sum group losses across all8 microbatches / total valid groups across those8; FC/FF separately',
          checkpoint_additional_steps=[250,500],checkpoint_cumulative_steps=[750,1000],diagnostic_steps=list(DIAGNOSTIC_STEPS),
          save_optimizer=True,continue_old_optimizer=False,no_training_cache=True),
      runtime=dict(CUBLAS_WORKSPACE_CONFIG=':4096:8',deterministic=True,warn_only=False,TF32=False,SDPA='math',parameters='FP32',forward='BF16',integration='FP32'),
      new_seeds=list(SEEDS),seed_audit='No semantic seed or seed filename9501..9508 prior use found; numeric metric/byte-count matches excluded',
      evaluation=dict(models=['T1','A','B','C'],N=112,K=12,free=100,y=112,steps=50,formal_rollouts=64,formal_forwards=3200,
        known_time=1,free_time='FP32 k/50',dt=1/50,noise='CPU torch.Generator FP32[1,112,9], shared free arrays, no GT/OT',
        stages='Only additionalstep500 full generation; no step250 generation',evaluator='Unchanged complete independent T1 confirmation evaluator',
        comparisons=['A-T1 continued training','B-A interface increment','C-A relation increment']),
      gates=dict(expected_actual_model_forwards=2,expected_actual_model_backwards=2,optimizer_updates=0,
        A_B_zero_output='bitwise equal',u_endpoint_relation_MSE_atol=1e-10),
      forbidden=dict(guidance=0,extra_CFG=0,CPU_geometry_sampling=0,postprocessing=0,B_plus_C=0,fourth_training_arm=0),
      stop='500 new updates per arm and64 endpoint samples; no extension, no Phase3')


class Ledger:
    def __init__(self,r):self.r=r
    def attach(self,model):
        def pre(module,args):
            c=self.r['counts'];stage=self.r['active_stage']
            if c['model_forward_attempts']>=15202:raise RuntimeError('Hard total model-forward budget')
            c['model_forward_attempts']+=1;c['forward_by_stage'][stage]=c['forward_by_stage'].get(stage,0)+1
        def post(module,args,value):self.r['counts']['model_forward_returns']+=1
        model.register_forward_pre_hook(pre);model.register_forward_hook(post)


def initialize(out):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);(out/'raw').mkdir()
    p=protocol();r=dict(status='INITIALIZING',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),protocol=p,protocol_sha256=seal(p),
      counts=dict(model_forward_attempts=0,model_forward_returns=0,forward_by_stage={},backward_attempts=0,backward_completed=0,
          optimizer_step_attempts=0,optimizer_updates=0,gate_backward_calls=0,training_backward_calls=0,rollout_attempts=0,complete_rollouts=0,
          euler_updates=0,geometry_guidance_calls=0,postprocessing_repairs=0,training_OT_calls=0,gate_OT_calls=0),
      arms={},generated=[],scores=[],bindings=[],gates={},evaluation_ledger={},active_stage='INITIALIZING',
      protection=dict(before=metadata(out)),AGENTS='No applicable AGENTS.md found by repository and ancestor checks',
      git_status=subprocess.run(['git','status','--short'],cwd=REPO,capture_output=True,text=True).stdout,
      environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__))
    r['environment']['GPU']=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
    write(r,out)
    bank,assets=prepare_assets();r['assets']=assets
    if assets['checkpoint']['sha256']!=T1_FILE_SHA256 or file_sha256(SOURCE/'inputs.npz')!=INPUT_SHA:raise RuntimeError('Wrong T1 assets')
    r['assets']['official_EMA_protection']=record('/mnt/c/Users/warpwang/Downloads/last.pt')
    r['assets']['critical_files'][r['assets']['official_EMA_protection']['path']]=r['assets']['official_EMA_protection']
    r['assets']['native_t1_core']={str(x):record(x) for x in (REPO/'native_t1').glob('*.py')}
    archive=load_archive(SOURCE/'inputs.npz',INPUT_SHA)
    r['assets']['training_cases']=[dict(parent=p,K=k,N=c.N,L=c.bbox_diagonal,source_face_ids=c.source_face_ids.tolist(),condition_sha256=array_hash(c.constraints)) for (p,k),c in archive.cases.items()]
    from .t1_native_confirmation_topology import run_input_audit
    from .t1_native_confirmation_evaluation import audit_context
    r['input_topology']=run_input_audit(bank)
    r['assets']['context_audits']={str(p):audit_context(bank['cases_by_key'][p,12]['constraints'],bank['cases_by_key'][p,12]['bbox_diagonal'],r['evaluation_ledger']) for p in (0,1)}
    arrays={};r['noise_ledger']={}
    for s in SEEDS:
        a=make_noise(s).numpy();arrays[f'noise_seed{s}']=a
        r['noise_ledger'][str(s)]=dict(shape=list(a.shape),dtype=str(a.dtype),sha256=array_hash(a),free_sha256=array_hash(a[:,12:]))
    np.savez_compressed(out/'fixed_noise.npz',**arrays);r['noise_file']=record(out/'fixed_noise.npz')
    r['seed_usage_audit']=dict(seeds=list(SEEDS),semantic_record_matches=0,output_filename_matches=0,
        broad_false_hits='Scalar metrics, extraction byte counts, or broadphase counts; not seed use',fixed_before_any_model_call=True)
    r['status']='ASSETS_PASS';write(r,out);event('assets_pass',out=str(out),counts=r['counts'])
    return r


def load_model(r,name,checkpoint=None,trainable=False):
    model,audit=load_t1()
    if name=='B':
        from .t1_interface_model import add_boundary_model
        model=add_boundary_model(model)
    if checkpoint is not None:
        if file_sha256(checkpoint['path'])!=checkpoint['sha256']:raise RuntimeError('New checkpoint bytes differ')
        payload=torch.load(checkpoint['path'],map_location='cpu',weights_only=True,mmap=True)
        if payload['arm']!=name or payload['additional_updates']!=500:raise RuntimeError('Wrong branch endpoint')
        keys=model.load_state_dict(payload['model'],strict=True);del payload
        if keys.missing_keys or keys.unexpected_keys or state_sha256(model)!=checkpoint['state_sha256']:raise RuntimeError('Strict endpoint load differs')
    model.set_trainable(trainable).eval();Ledger(r).attach(model)
    return model,audit


def sample_all(r,out,relations):
    if any(r['arms'][a]['completed_updates']!=500 or r['arms'][a]['status']!='PASS' for a in ARMS):raise RuntimeError('All fixed endpoints required')
    archive=load_archive(SOURCE/'inputs.npz',INPUT_SHA)
    with np.load(out/'fixed_noise.npz') as f:noises={s:f[f'noise_seed{s}'].copy() for s in SEEDS}
    for name in ('T1','A','B','C'):
        r['active_stage']=name+'_sampling';model,loaded=load_model(r,name,None if name=='T1' else r['arms'][name]['checkpoints']['500'])
        before=state_sha256(model);tick=time.perf_counter()
        try:
            for parent in (0,1):
                C=archive.cases[parent,12].constraints.copy();phi=None
                if name=='B':
                    phi=np.zeros((1,112,3,2),np.float32);phi[0,:12]=relations.condition_phi(parent,12);phi=torch.from_numpy(phi).cuda()
                for seed in SEEDS:
                    if r['counts']['rollout_attempts']>=64:raise RuntimeError('Hard64rollout budget')
                    r['counts']['rollout_attempts']+=1;write(r,out);start=time.perf_counter();nf=r['counts']['model_forward_attempts']
                    context=model.boundary_condition(phi) if name=='B' else contextlib.nullcontext()
                    sc=new_counts()
                    try:
                        with context:raw,path,stats=clamped_sample(model,torch.from_numpy(noises[seed]).cuda(),torch.from_numpy(C).cuda(),sc)
                    finally:
                        r['counts']['euler_updates']+=sc['euler_updates'];r['counts']['complete_rollouts']+=sc['complete_rollouts']
                        r.setdefault('sampling_attempts',[]).append(dict(model=name,parent=parent,seed=seed,local_counts=dict(sc),returned=sc['complete_rollouts']==1))
                        write(r,out)
                    if r['counts']['model_forward_attempts']-nf!=50:raise RuntimeError('NFE differs')
                    expected=np.broadcast_to(C.reshape(1,12,9),(51,12,9)).copy()
                    if not np.array_equal(path[:,:12].view(np.uint32),expected.view(np.uint32)) or not np.isfinite(path).all():raise RuntimeError('C altered or nonfinite path')
                    if not np.array_equal(path[0,12:].view(np.uint32),noises[seed][0,12:].view(np.uint32)):raise RuntimeError('Noise changed')
                    target=out/'raw'/f'{name}_P{parent}_K12_seed{seed}.npz'
                    if target.exists():raise FileExistsError(target)
                    export=time.perf_counter();np.savez_compressed(target,output=raw,path=path);export=time.perf_counter()-export
                    r['generated'].append(dict(phase='confirmation',model=name,parent=parent,K=12,seed=seed,variant='Raw',raw_path=str(target),raw_sha256=file_sha256(target),
                        output_sha256=array_hash(raw),sampling=stats,C_all51_FP32_bits_exact=True,all_states_finite=True,initial_free_Gaussian_exact=True,
                        costs=dict(complete_generation_wall_seconds=time.perf_counter()-start,NPZ_export_seconds=export,peak_cuda_allocated=stats['peak_cuda_allocated'])))
                    write(r,out);event('sample',model=name,parent=parent,seed=seed,completed=len(r['generated']))
            r.setdefault('sampling_models',{})[name]=dict(initial_state=before,final_state=state_sha256(model),seconds=time.perf_counter()-tick,strict_load=loaded)
            if r['sampling_models'][name]['final_state']!=before:raise RuntimeError('Frozen sampling model changed')
        finally:del model;release()
    r['status']='RAW_COMPLETE';write(r,out)


def score_all(r,out):
    from .t1_interface_evaluation import score_output,aggregate_results
    bank,_=prepare_assets();bank['input_topology']=r['input_topology'];bank['context_audits']={int(k):v for k,v in r['assets']['context_audits'].items()}
    for row in r['generated']:
        with np.load(row['raw_path']) as f:raw=f['output'].copy()
        tick=time.perf_counter();metric=score_output(raw,bank,row['parent'],ledger=r['evaluation_ledger'])
        r['scores'].append(dict(phase='confirmation',model=row['model'],parent=row['parent'],K=12,seed=row['seed'],variant='Raw',raw_path=row['raw_path'],metrics=metric,offline_scoring_seconds=time.perf_counter()-tick))
        write(r,out);event('scored',model=row['model'],parent=row['parent'],seed=row['seed'],count=len(r['scores']))
    r['summary']=aggregate_results(r['scores'],SEEDS);r['bindings']=r['summary']['bindings']
    after=metadata(out);changed=[p for p,v in r['protection']['before'].items() if after.get(p)!=v]
    r['protection']['after']=dict(status='PASS' if not changed else 'FAIL',files=len(r['protection']['before']),changed=changed)
    r['protection']['critical_files_after']=verify_records(r['assets']['critical_files'])
    r['protection']['native_t1_after']=verify_records(r['assets']['native_t1_core'])
    if changed or any(r['protection'][k]['status']!='PASS' for k in ('critical_files_after','native_t1_after')):raise RuntimeError('Protected assets changed')
    c=r['counts']
    if c['optimizer_updates']!=1500 or c['complete_rollouts']!=64 or c['model_forward_attempts']!=15202:raise RuntimeError('Final fixed budget accounting differs')
    r.update(status='PASS_EXECUTION',active_stage='FINISHED',finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write(r,out);event('complete',counts=r['counts'])


def execute(out,stage):
    if stage=='init':return initialize(out)
    r=json.loads((out/'results.json').read_text())
    try:
        if seal(r['protocol'])!=r['protocol_sha256']:raise RuntimeError('Frozen protocol changed')
        configure_stable_runtime()
        from .t1_interface_relations import load_source_relations
        relations=load_source_relations(SOURCE/'inputs.npz',expected_sha256=INPUT_SHA)
        if stage=='preflight':
            if r['status']!='ASSETS_PASS':raise RuntimeError('Preflight cannot repeat automatically')
            from .t1_interface_train import preflight
            preflight(r,out,relations)
        elif stage=='train':
            if r['status']!='PREFLIGHT_PASS':raise RuntimeError('All zero-update gates must pass')
            from .t1_interface_train import train_arm
            for arm in ARMS:train_arm(r,out,relations,arm)
            r['status']='TRAINING_COMPLETE';write(r,out)
        elif stage=='sample':
            if r['status']!='TRAINING_COMPLETE':raise RuntimeError('Fixed training matrix incomplete')
            sample_all(r,out,relations)
        elif stage=='score':
            if r['status']!='RAW_COMPLETE':raise RuntimeError('All raw endpoints required before scoring')
            score_all(r,out)
        elif stage=='report':
            if r['status']!='PASS_EXECUTION':raise RuntimeError('Complete experiment required')
            from .t1_interface_report import deliver_report
            bank,_=prepare_assets();r['delivery']=deliver_report(r,out,bank);write(r,out)
    except BaseException as error:
        r.update(status='STOPPED',failure=dict(stage=stage,error=repr(error),traceback=traceback.format_exc(),automatic_retry=False));write(r,out);event('stopped',error=repr(error),counts=r['counts']);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['init','preflight','train','sample','score','report'],required=True)
    a=p.parse_args();execute(a.out.resolve(),a.stage)
