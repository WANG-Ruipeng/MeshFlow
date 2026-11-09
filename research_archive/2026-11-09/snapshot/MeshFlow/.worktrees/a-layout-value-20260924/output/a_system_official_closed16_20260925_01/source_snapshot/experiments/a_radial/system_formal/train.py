"""Independent formal harness; frozen original architecture, OT, and update order.

The sole numerical change from system_probe.official.Session.update is replacing
four serial dataset/OT calls with the equivalent synchronous parallel-OT adapter.
Fresh formal initialization never loads correctness/profile checkpoint weights.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time
import traceback
import numpy as np
import torch
import yaml
from ..io_utils import new_dir,write_json,atomic_torch,file_hash,array_hash,rng_state
from ..official_train import ema_update
from ..official_contract import audit_loader
from ..verify_repo import verify
from ..layout_student.train import model_sha256
from ..system_probe.official import Session,STATE_KEYS,PARAMETERS,require,exact_same,validate_config,all_parent_probe
from ..system_parallel_ot_probe.adapter import load_microbatch

FORMAL_SCOPE='CLOSED16_FORMAL_FIXED_BUDGET'
PROBE_SCOPE='CORRECTNESS_PROFILE_ONLY_NOT_FORMAL_INITIALIZATION'
BOUNDARY={'data_order':'sequential_getitem_random_sampling','OT_workers':4,
          'cross_microbatch_prefetch_queue':False,'pending_OT_futures':0,
          'at_complete_effective_update_boundary':True}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def source_paths(repo):
    relative=['experiments/a_radial/system_formal/'+name for name in ['__init__.py','train.py','gate.py']]
    relative += ['experiments/a_radial/system_parallel_ot_probe/adapter.py',
        'experiments/a_radial/system_probe/official.py','experiments/a_radial/official_train.py',
        'experiments/a_radial/official_contract.py','experiments/a_radial/io_utils.py',
        'experiments/a_radial/verify_repo.py','experiments/a_radial/layout_student/train.py',
        'experiments/a_radial/assets/UPSTREAM_LOCK.json']
    return {name:file_hash(repo/name) for name in relative}


def protected_sources(repo,commit):
    lines=subprocess.check_output(['git','-C',str(repo),'ls-tree','-r','--name-only',commit],text=True).splitlines()
    names=[p for p in lines if p.startswith(('models/','datasets/')) or
        p in ['utils/ot_utils.py','flow_matching.py','train.py','inference.py','gradio_demo.py']]
    require(len(names)==11,'Expected all 11 protected original source files')
    return {name:file_hash(repo/name) for name in names}


def dataset_manifest(data):
    description=json.loads((data/'DATASET.json').read_text());uids=sorted(row['uid'] for row in description['rows'])
    require(len(uids)==len(set(uids))==16,'Expected exactly16 parents')
    files=['DATASET.json','meshflow512/split/train.npz','meshflow512/split/test.npz']
    files += ['meshflow512/objaverse_occ_v5_ids/'+uid+'.npz' for uid in uids]
    return {'root':str(data),'parent_uids':uids,'files':{name:file_hash(data/name) for name in files}}


def prepare(repo,data,out,gate_out):
    repo=Path(repo).resolve();data=Path(data).resolve();gate_out=Path(gate_out).resolve()
    require(not torch.cuda.is_initialized(),'Prepare is CPU-only')
    identity=verify(repo);cfg_path=repo/'experiments/a_radial/configs/meshflow_closed16_512.yaml'
    cfg=yaml.safe_load(cfg_path.read_text());validate_config(cfg)
    require(cfg['closed16_runner']['checkpoint_every']==5000 and cfg['closed16_runner']['fit_probe_every']==1000,'Cadence changed')
    sources=source_paths(repo);protected=protected_sources(repo,identity['base_commit']);manifest=dataset_manifest(data)
    out=new_dir(out).resolve();write_json(out/'DATA_MANIFEST.json',manifest)
    write_json(out/'SOURCE_MANIFEST.json',{'source_sha256':sources,'protected_sha256':protected})
    for relative in sorted(set(sources)|set(protected)):
        target=out/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(repo/relative,target)
    shutil.copy2(cfg_path,out/'FORMAL_CONFIG.yaml');write_json(out/'CONFIG.json',cfg)
    write_json(out/'REPO_IDENTITY_PREPARE.json',identity)
    protocol={'scope':FORMAL_SCOPE,'repository':str(repo),'data':str(data),'seeds':[10,11],
        'formal_updates':50000,'microbatch':4,'accumulation_steps':4,'effective_batch':16,
        'full_parameter_count':PARAMETERS,'EMA_decay':.9999,'primary_weight':'ema','checkpoint_selection':'fixed_final_50000_only',
        'fresh_initialization':True,'pretrained_or_probe_weights_used':False,
        'optimizer':{'name':'AdamW','lr':1e-4,'betas':[.9,.95],'weight_decay':0.,'foreach':False,'fused':False,'max_grad_norm':2.},
        'scheduler':'explicit_constant_no_LR_change','precision':'float32_parameters_bfloat16_autocast',
        'TF32':False,'deterministic_algorithms':True,'data_adapter':BOUNDARY,
        'progress_every':100,'fit_probe_every':1000,'checkpoint_every':5000,
        'fresh_step0_recovery_checkpoint':True,'intermediate_checkpoints_for_resume_only':True,
        'fit_probe':{'all_parents':16,'noise_time_seed':883021,'current_and_EMA':True,'preserves_training_RNG_and_sampler':True,
            'qualification':'training_support_only_not_convergence_or_generalization'},
        'sample_config_unchanged':cfg['sample'],'parent_uids':manifest['parent_uids'],
        'gate_output':str(gate_out),'gate_required':['serial_parallel_4_updates_fullstate','parallel_4_vs_2_plus_2_fullstate','fixed_probe_rng','full_mask_CFG','isolated_profile100'],
        'base_commit':identity['base_commit'],'formal_config_sha256':file_hash(cfg_path),
        'dataset_manifest_sha256':file_hash(data/'DATASET.json'),'data_manifest_sha256':file_hash(out/'DATA_MANIFEST.json'),
        'source_manifest_sha256':file_hash(out/'SOURCE_MANIFEST.json'),
        'no_automatic_convergence_claim':True,'no_heldout_class_generalization_claim':True}
    write_json(out/'RUN_PROTOCOL.json',protocol)
    write_json(out/'STATUS.json',{'status':'PREPARED_GATE_PENDING','formal_training_started':False,'formal_updates_completed':0,
        'protocol_sha256':file_hash(out/'RUN_PROTOCOL.json')})
    return protocol


def load_contract(run):
    run=Path(run).resolve();p=json.loads((run/'RUN_PROTOCOL.json').read_text());repo=Path(p['repository']);data=Path(p['data'])
    require(p['scope']==FORMAL_SCOPE and p['seeds']==[10,11] and p['formal_updates']==50000,'Wrong formal scope, seeds, or budget')
    exact_same(p['data_adapter'],BOUNDARY,'protocol_adapter_boundary')
    require(file_hash(run/'DATA_MANIFEST.json')==p['data_manifest_sha256'],'Changed data manifest')
    require(file_hash(run/'SOURCE_MANIFEST.json')==p['source_manifest_sha256'],'Changed source manifest')
    require(file_hash(run/'FORMAL_CONFIG.yaml')==p['formal_config_sha256'],'Changed fixed config snapshot')
    require(file_hash(repo/'experiments/a_radial/configs/meshflow_closed16_512.yaml')==p['formal_config_sha256'],'Changed original fixed config')
    cfg=yaml.safe_load((run/'FORMAL_CONFIG.yaml').read_text());validate_config(cfg)
    exact_same(json.loads((run/'CONFIG.json').read_text()),cfg,'config_snapshot')
    actual=verify(repo);require(actual['base_commit']==p['base_commit'],'Wrong official source commit')
    sources=json.loads((run/'SOURCE_MANIFEST.json').read_text())
    exact_same(sources['source_sha256'],source_paths(repo),'adapter_dependency_source_hashes')
    exact_same(sources['protected_sha256'],protected_sources(repo,p['base_commit']),'protected11_source_hashes')
    manifest=json.loads((run/'DATA_MANIFEST.json').read_text())
    exact_same(manifest,dataset_manifest(data),'source_data_files')
    require(manifest['parent_uids']==p['parent_uids'],'Parent list changed')
    identity={'protocol_sha256':file_hash(run/'RUN_PROTOCOL.json'),'dataset_sha256':p['dataset_manifest_sha256'],
        'data_manifest_sha256':p['data_manifest_sha256'],'source_manifest_sha256':p['source_manifest_sha256'],
        'formal_config_sha256':p['formal_config_sha256'],'base_commit':p['base_commit']}
    return p,cfg,identity


def runtime_identity():
    return {'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,
        'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'cuda_capability':list(torch.cuda.get_device_capability(0))}


def validate_final_budget(step,scope):
    require(0<=step<=50000,'Saved step outside fixed budget')
    return scope==FORMAL_SCOPE and step==50000


def validate_checkpoint(state,identity,cfg,scope):
    exact_same(state['identity'],identity,'resume_identity');exact_same(state['config'],cfg,'resume_config')
    require(state['scope']==scope,'Cannot import a probe checkpoint into formal training')
    exact_same(state['boundary'],BOUNDARY,'resume_boundary')
    require(state['fresh_seed']==identity['seed'] and state['formal_budget']==50000,'Fresh seed or formal budget changed')
    require(state['initialization']=='fresh_original_model_no_probe_or_pretrained_checkpoint','Wrong initialization lineage')
    step=int(state['step']);require(state['ema_updates']==step==state['scheduler']['last_epoch'],'Checkpoint update counters disagree')
    require(torch.equal(state['parent_exposures'],torch.full((16,),step,dtype=torch.int64)),'Parent exposure state mismatch')
    require(len(state['trace'])==step and all(row['step']==i+1 for i,row in enumerate(state['trace'])),'Incomplete chronological training trace')
    require(all(row['ema_updates']==row['step']==row['scheduler_updates'] for row in state['trace']),'EMA cadence changed')
    require(all(int(value['step'])==step for value in state['optimizer']['state'].values()),'Adam per-parameter update mismatch')
    require((not state['optimizer']['state']) if step==0 else bool(state['optimizer']['state']),'Adam state initialization count mismatch')
    require(len(state['timing_trace'])==step,'Incomplete timing provenance')
    require(state['is_predeclared_final']==validate_final_budget(step,scope),'Incorrect final checkpoint qualification')
    require(state['model_sha256']==model_sha256(state['model']) and state['ema_sha256']==model_sha256(state['ema']),'Stored model identity mismatch')
    require(state['boundary']['pending_OT_futures']==0,'Pending OT work cannot be resumed')
    if step==0:require(state['model_sha256']==state['ema_sha256'],'Initial EMA differs from fresh model')


class ParallelSession(Session):
    def __init__(self,cfg,ds,seed,identity,scope=FORMAL_SCOPE):
        super().__init__(cfg,ds,seed,identity)
        require(scope in (FORMAL_SCOPE,PROBE_SCOPE),'Unexpected session scope')
        self.scope=scope;self.pool=ThreadPoolExecutor(max_workers=4,thread_name_prefix='formal_original_ot')
        self.safe_boundary=True;self.timing_trace=[]

    def update(self):
        require(self.safe_boundary,'Cannot continue an incomplete effective update')
        self.safe_boundary=False
        row,timing=self._update_core()
        self.timing_trace.append(timing);self.safe_boundary=True
        return row,timing

    def _update_core(self):
        from datasets.mesh_dataset import collate_fn
        torch.cuda.synchronize();started=time.perf_counter();order=torch.randperm(16,generator=self.gen).tolist()
        self.opt.zero_grad(set_to_none=True);lossval=0.;batches=[];loader_seconds=0.
        for k in range(4):
            ids=order[k*4:(k+1)*4];self.exposures[ids]+=1
            before=time.perf_counter();b,_ot_audit=load_microbatch(self.ds,ids,self.pool)
            loader_seconds+=time.perf_counter()-before
            require(tuple(b['tokens'].shape)==(4,512,9) and bool(b['masks'].all()),'Real loader512 batch/mask changed')
            require(bool((b['num_faces']==512).all()),'Wrong official face condition')
            batches.append({'indices':ids,'uids':[self.uids[i] for i in ids],
                'tokens_sha256':array_hash(b['tokens'].numpy()),'noise_sha256':array_hash(b['noise'].numpy())})
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=self.transport.training_losses(self.model,b['tokens'].cuda(),b['noise'].cuda(),
                    {'y':b['num_faces'].cuda(),'mask':b['masks'].cuda()})['loss']/4
            require(bool(torch.isfinite(loss)),'Official CUDA loss is nonfinite')
            loss.backward();lossval+=float(loss.detach())
        require(all(p.grad is not None for p in self.model.parameters()),'Missing full-model gradient')
        finite=torch.stack([torch.isfinite(p.grad).all() for p in self.model.parameters()]).all()
        require(bool(finite),'Nonfinite full-model gradient')
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),2.)
        require(bool(torch.isfinite(norm)) and float(norm)>0,'Zero or nonfinite gradient norm')
        self.opt.step();self.scheduler.step();ema_update(self.ema,self.model,.9999)
        self.ema_updates+=1;self.step_count+=1
        row={'step':self.step_count,'loss':lossval,'gradient_norm_before_clip':float(norm),
            'all_gradients_present_and_finite':True,'microbatches':batches,'effective_batch':16,
            'ema_updates':self.ema_updates,'scheduler_updates':self.scheduler.last_epoch,'lr':self.opt.param_groups[0]['lr']}
        self.trace.append(row);torch.cuda.synchronize()
        timing={'step':self.step_count,'seconds_full_effective_update':time.perf_counter()-started,
            'seconds_loader_collate_OT':loader_seconds,'cpu_GPU_transfer_and_sync_included':True,
            'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved()}
        return row,timing


    def state(self):
        require(self.safe_boundary,'Refusing a partial-update checkpoint')
        state=super().state();state['scope']=self.scope
        state.update({'boundary':deepcopy(BOUNDARY),'timing_trace':self.timing_trace,
            'formal_budget':50000,'is_predeclared_final':validate_final_budget(self.step_count,self.scope),
            'model_sha256':model_sha256(state['model']),'ema_sha256':model_sha256(state['ema']),
            'fresh_seed':self.seed,'initialization':'fresh_original_model_no_probe_or_pretrained_checkpoint'})
        return state

    def load(self,path):
        ck=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        validate_checkpoint(ck,self.identity,self.cfg,self.scope)
        self.timing_trace=ck['timing_trace'];del ck
        super().load(path);self.safe_boundary=True

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=False)


def fixed_probes(session):
    before=rng_state();sampler=session.gen.get_state().clone();started=time.perf_counter()
    current=all_parent_probe(session.model,session.transport,session.ds)
    ema=all_parent_probe(session.ema,session.transport,session.ds)
    exact_same(before,rng_state(),'fixed_probe_all_training_rng')
    require(torch.equal(sampler,session.gen.get_state()),'Fixed probe changed sampler')
    return {'step':session.step_count,'current':current,'ema':ema,'seconds':time.perf_counter()-started,
        'training_rng_and_sampler_bitwise_preserved':True,'EMA_initialization_coefficient':.9999**session.step_count,
        'not_a_checkpoint_selection_criterion':True}


def gate_is_ready(run,p,identity):
    gate=Path(p['gate_output']);complete=json.loads((gate/'COMPLETE.json').read_text())
    require(complete['status']=='PASS' and complete['formal_initialization_from_probe_forbidden'],'Parallel CUDA gate incomplete')
    require(complete['formal_protocol_sha256']==identity['protocol_sha256'],'GPU gate belongs to different formal protocol')
    for name,digest in complete['required_files_sha256'].items():require(file_hash(gate/name)==digest,'GPU gate evidence changed: '+name)
    protocol=json.loads((gate/'RUN_PROTOCOL.json').read_text())
    exact_same(protocol['formal_identity'],identity,'gate_identity')
    exact_same(protocol['runtime'],runtime_identity(),'gate_runtime')
    require(json.loads((gate/'GATE.json').read_text())['status']=='PASS','CUDA semantic gate failed')
    require(json.loads((gate/'PROFILE100.json').read_text())['status']=='COMPLETE','Isolated profile100 incomplete')
    return {'path':str(gate),'complete_sha256':file_hash(gate/'COMPLETE.json')}


def save_complete_checkpoint(session,out):
    before=time.perf_counter();path=out/f'step{session.step_count:08d}.pt'
    atomic_torch(path,session.state())
    record={'step':session.step_count,'path':str(path),'sha256':file_hash(path),'bytes':path.stat().st_size,
        'write_and_hash_seconds':time.perf_counter()-before}
    write_json(out/'checkpoints'/f'step{session.step_count:08d}.json',record)
    return record


def new_invocation(seed_dir):
    root=seed_dir/'invocations';root.mkdir(parents=True,exist_ok=True)
    for index in range(1,100000):
        path=root/f'{index:04d}'
        try:path.mkdir()
        except FileExistsError:continue
        return path
    raise RuntimeError('Invocation index exhausted')


def run_training(run,seed,resume=None):
    run=Path(run).resolve();p,cfg,base_identity=load_contract(run)
    require(seed in p['seeds'],'Only declared training seeds10/11')
    require(torch.cuda.is_available(),'True CUDA required')
    gate=gate_is_ready(run,p,base_identity)
    seed_dir=run/'training'/f'seed_{seed}';seed_dir.mkdir(parents=True,exist_ok=True)
    require(not (seed_dir/'TRAINING_COMPLETE.json').exists(),'This seed already completed its fixed budget')
    if resume is None:require(not any((seed_dir/'invocations').glob('*')),'Existing attempt requires an explicit complete checkpoint for resume')
    else:
        resume=Path(resume).resolve()
        require(resume.is_relative_to(seed_dir.resolve()),'Formal resume must come from this seed directory, never a probe')
        record_path=resume.parent/'checkpoints'/(resume.stem+'.json')
        record=json.loads(record_path.read_text())
        require(Path(record['path']).resolve()==resume and record['sha256']==file_hash(resume)
            and record['bytes']==resume.stat().st_size,'Resume checkpoint disagrees with its immutable completion record')
    out=new_invocation(seed_dir);identity={**base_identity,'seed':seed};session=None;last_checkpoint=None
    trace_start=0;timing_start=0;started=time.perf_counter();save_events=[]
    invocation={'status':'TRAINING_IN_PROGRESS','seed':seed,'scope':FORMAL_SCOPE,'protocol_sha256':identity['protocol_sha256'],
        'gate':gate,'resume':None if resume is None else str(resume),
        'resume_sha256':None if resume is None else file_hash(resume),'fresh_init':resume is None,'fixed_final_step':50000,'runtime':runtime_identity()}
    write_json(out/'INVOCATION.json',invocation)
    try:
        ds,loader=audit_loader(p['repository'],p['data']);write_json(out/'LOADER_GATE.json',loader)
        session=ParallelSession(cfg,ds,seed,identity)
        if resume is not None:session.load(resume);last_checkpoint=str(resume)
        start_step=session.step_count;trace_start=start_step;timing_start=len(session.timing_trace)
        torch.cuda.reset_peak_memory_stats()
        if resume is None:
            initial_record=save_complete_checkpoint(session,out);save_events.append(initial_record)
            last_checkpoint=initial_record['path']
        write_json(out/'INITIALIZATION.json',{'fresh':resume is None,'step':start_step,'model_sha256':model_sha256(session.model.state_dict()),
            'ema_sha256':model_sha256(session.ema.state_dict()),'no_probe_or_pretrained_weights_loaded':True,
            'sampler_seed':853000+seed,'training_seed':seed,'complete_recovery_checkpoint':last_checkpoint})
        write_json(run/'STATUS.json',{'status':'TRAINING_IN_PROGRESS','active_seed':seed,'active_invocation':str(out),
            'step':start_step,'fixed_budget_per_seed':50000,'current_complete_checkpoint':last_checkpoint,'formal_qualification':'PENDING_FIXED_BUDGET_AND_FIT_SAMPLING_EVALUATION'},replace=True)
        while session.step_count<50000:
            row,timing=session.update();step=session.step_count
            if step%1000==0:
                write_json(out/'probes'/f'step{step:08d}.json',fixed_probes(session))
            if step%5000==0:
                checkpoint=save_complete_checkpoint(session,out)
                save_events.append(checkpoint);last_checkpoint=checkpoint['path']
            if step%100==0:
                chunk=session.trace[trace_start:];times=session.timing_trace[timing_start:]
                progress={'status':'TRAINING_IN_PROGRESS','seed':seed,'step':step,'fixed_final_step':50000,
                    'latest_loss':row['loss'],'latest_gradient_norm':row['gradient_norm_before_clip'],
                    'parent_uids':session.uids,'parent_exposures':session.exposures.tolist(),'EMA_updates':session.ema_updates,
                    'elapsed_invocation_seconds':time.perf_counter()-started,
                    'all_saved_update_seconds':float(sum(x['seconds_full_effective_update'] for x in session.timing_trace)),
                    'recent_100_update_mean_seconds':float(np.mean([x['seconds_full_effective_update'] for x in times])),
                    'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),
                    'current_complete_checkpoint':last_checkpoint,'no_automatic_convergence_claim':True}
                write_json(out/'trace_chunks'/f'step{step:08d}.json',{'trace':chunk,'timings':times})
                write_json(out/'progress'/f'step{step:08d}.json',progress)
                write_json(seed_dir/'PROGRESS.json',progress,replace=(seed_dir/'PROGRESS.json').exists())
                print(json.dumps(progress),flush=True);trace_start=len(session.trace);timing_start=len(session.timing_trace)
        final={'status':'FIXED_BUDGET_COMPLETE_UNQUALIFIED','seed':seed,'scope':FORMAL_SCOPE,'step':50000,
            'primary_weight':'ema','primary_checkpoint':last_checkpoint,'primary_checkpoint_sha256':file_hash(last_checkpoint),
            'protocol_sha256':identity['protocol_sha256'],'parent_exposures':session.exposures.tolist(),
            'elapsed_invocation_seconds':time.perf_counter()-started,'fit_sampling_and_evaluation_pending':True,
            'no_automatic_convergence_or_generalization_claim':True,'invocation':str(out)}
        write_json(out/'TRAIN_REPORT.json',final);write_json(seed_dir/'TRAINING_COMPLETE.json',final)
        completed=[s for s in p['seeds'] if (run/'training'/f'seed_{s}'/'TRAINING_COMPLETE.json').exists()]
        write_json(run/'STATUS.json',{'status':'FIXED_BUDGET_COMPLETE_UNQUALIFIED' if len(completed)==2 else 'TRAINING_IN_PROGRESS',
            'completed_seeds':completed,'pending_seeds':[s for s in p['seeds'] if s not in completed],
            'formal_qualification':'PENDING_FIT_SAMPLING_EVALUATION'},replace=True)
        return final
    except BaseException as exc:
        failure={'status':'TRAINING_INTERRUPTED','type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),
            'last_complete_checkpoint':last_checkpoint,'last_complete_checkpoint_sha256':None if last_checkpoint is None else file_hash(last_checkpoint),
            'last_finished_step':None if session is None else session.step_count,
            'partial_update_checkpoint_written':False,'elapsed_invocation_seconds':time.perf_counter()-started}
        write_json(out/'FAILED.json',failure)
        if session is not None:
            write_json(out/'UNSAVED_COMPLETED_TRACE.json',{'trace':session.trace[trace_start:],'timings':session.timing_trace[timing_start:]})
        write_json(run/'STATUS.json',{'status':'TRAINING_INTERRUPTED','seed':seed,'failure_record':str(out/'FAILED.json'),
            'last_complete_checkpoint':last_checkpoint,'formal_qualification':'PENDING'},replace=True)
        raise
    finally:
        if session is not None:session.close()
        del session;gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--repo',type=Path,default=Path('.'));prep.add_argument('--data',type=Path,required=True)
    prep.add_argument('--out',type=Path,required=True);prep.add_argument('--gate-out',type=Path,required=True)
    training=sub.add_parser('train');training.add_argument('--run',type=Path,required=True)
    training.add_argument('--seed',type=int,choices=[10,11],required=True);training.add_argument('--resume',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.repo,args.data,args.out,args.gate_out)
    else:run_training(args.run,args.seed,args.resume)


if __name__=='__main__':main()
