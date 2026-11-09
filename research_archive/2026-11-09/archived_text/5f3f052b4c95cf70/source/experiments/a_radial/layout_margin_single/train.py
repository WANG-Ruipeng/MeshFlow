"""CPU-only, paired-target single-parent C0 fitting with a fixed final budget."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from ..legacy.networks import DomainVelocity, point_features
from ..layout_student.train import BalancedSampler, file_sha256, model_sha256
from ..layout_endpoint.train import read_json, write_json, require, resolve_under
from ..layout_fitdiag.train import step as frozen_c0_step


def config_digest(protocol):
    value={k:protocol[k] for k in ('training','loss','jobs')}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def configure_cpu(seed):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CUDA must be hidden in this CPU experiment')
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(seed);np.random.seed(seed)
    # torch.manual_seed also touches CUDA's lazy RNG; only seed the CPU generator.
    torch.random.default_generator.manual_seed(seed)


def cpu_rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state().clone()}


def restore_cpu_rng(state):
    require(set(state)=={'python','numpy','torch_cpu'},'CPU RNG schema changed or CUDA state supplied')
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch_cpu'])


def step_sequence_hash(trace):
    """Hash only the ordered k draws (uint8 B4), independent of target and timing."""
    h=hashlib.sha256()
    for update,row in enumerate(trace,1):
        require(row['update']==update,'Trace update sequence differs')
        ks=row['rng_matched_k']
        require(len(ks)==4 and all(isinstance(k,int) and 0<=k<32 for k in ks),'Invalid bridge time draws')
        h.update(bytes(ks))
    return h.hexdigest()


def load_inputs(run,protocol,job_name):
    run=Path(run).resolve();require(config_digest(protocol)==protocol['hashes']['config_sha256'],'Config hash mismatch')
    require(job_name in protocol['jobs'],'Unknown registered job')
    job=protocol['jobs'][job_name];t=protocol['training']
    for key,value in [('batch_size',4),('learning_rate',1e-4),('weight_decay',1e-4),
                      ('device','cpu'),('torch_threads',1),('interop_threads',1)]:
        require(t[key]==value,'Fixed CPU setting changed: '+key)
    require(protocol['loss']['name']=='c0_unscaled_discrete_bridge_velocity_MSE'
            and protocol['loss']['bridge_steps']==32
            and protocol['loss']['endpoint_scale_unused_by_c0'] is True,'C0 objective changed')
    require(job['seed']==10 and job['start_update']==0 and job['target_kind'] in ('old','margin'),
            'Job seed/start/target differs')
    require(job['target_key']=='directions1_'+job['target_kind'],'Target key/kind mismatch')
    require(job['initialization_mode']=='shared_model_fresh_cpu_AdamW','Initialization changed')
    total=job['total_updates']
    require(total==t['total_updates'] and total>0,'Job budget mismatch')
    grid=t['checkpoint_updates']
    require(grid==sorted(set(grid)) and grid[0]==0 and grid[-1]==total,'Checkpoint grid invalid')
    if protocol.get('scope')!='SMOKE_ONLY':
        require(total==20000 and grid==[0,5000,20000],'Formal budget or observation points changed')
    manifest_path=resolve_under(run,protocol['paths']['data_manifest'])
    require(file_sha256(manifest_path)==protocol['hashes']['data_manifest_sha256'],'Data manifest changed')
    manifest=read_json(manifest_path)['files']
    archive=resolve_under(run,protocol['paths']['training_labels']);archive_sha=file_sha256(archive)
    require(archive_sha==protocol['hashes']['training_labels_sha256']
            ==manifest.get(archive.relative_to(run).as_posix()),'Training archive not bound')
    with np.load(archive,allow_pickle=False) as z:
        required={'train_uids','directions0','directions1_old','directions1_margin',
                  'proxy_points','point_features','faces'}
        require(set(z.files)==required,'Unexpected training input; no GT/mainfield allowed')
        data={k:np.array(z[k]) for k in required}
    uids=data.pop('train_uids').tolist();n=len(uids)
    require(n==4 and uids==protocol['train_uids'] and len(set(uids))==4,'Only the four registered train parents allowed')
    require(not set(uids)&set(protocol.get('internal_check_uids',[])),'Internal-check label leakage')
    shapes={'directions0':(n,258,3),'directions1_old':(n,258,3),'directions1_margin':(n,258,3),
            'proxy_points':(n,256,3),'point_features':(n,256,4),'faces':(512,3)}
    for key,shape in shapes.items():
        require(data[key].shape==shape and np.isfinite(data[key]).all(),'Invalid array '+key)
    for key in ('directions0','directions1_old','directions1_margin'):
        require(data[key].dtype==np.float64 and np.allclose(np.linalg.norm(data[key],axis=-1),1,rtol=0,atol=1e-12),
                'Nonunit or non-float64 directions '+key)
    require(data['faces'].dtype==np.int32 and data['point_features'].dtype==np.float32,'Connectivity/feature dtype changed')
    require(np.array_equal(point_features(data['proxy_points']),data['point_features']),'Original proxy256 input changed')
    index=int(job['index']);require(0<=index<n and uids[index]==job['uid']==job['pair_id'],'Job UID/index/pair mismatch')
    require(job['active_training_uids']==[job['uid']],'Active parent differs')
    require(job['diagnostic_uids']==[job['uid']],'Diagnostic parent differs')
    initial=resolve_under(run,job['initial_checkpoint']);initial_sha=file_sha256(initial)
    require(initial_sha==job['initial_checkpoint_sha256']==protocol['hashes']['initial_checkpoint_sha256']
            ==manifest.get(initial.relative_to(run).as_posix()),'Initial checkpoint binding differs')
    source=torch.load(initial,map_location='cpu',weights_only=False)
    require(not set(source)&{'optimizer','scheduler','rng','sampler'},'Initial model-only file contains inherited training state')
    require(model_sha256(source['model'])==source['model_sha256']==job['initial_model_sha256']
            ==protocol['hashes']['initial_model_sha256'],'Initial model hash differs')
    root=Path(__file__).resolve().parents[1]
    for relative,digest in protocol.get('source_sha256',{}).items():
        require(file_sha256(root/relative)==digest,'Frozen dependency changed: '+relative)
    return job,uids,data,source,initial_sha


def make_session(run,job_name,resume=None):
    run=Path(run).resolve();p=read_json(run/'RUN_PROTOCOL.json')
    job,uids,data,source,initial_sha=load_inputs(run,p,job_name)
    configure_cpu(job['seed'])
    model=DomainVelocity().cpu().float();model.load_state_dict(source['model'],strict=True);model.train()
    require(sum(x.numel() for x in model.parameters())==29907,'DomainVelocity architecture changed')
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4,foreach=False,fused=False)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    require(not optimizer.state and scheduler.last_epoch==0,'Fresh optimizer/scheduler not at zero')
    sampler=BalancedSampler(1,job['seed'])
    tensors=[torch.as_tensor(data[k],dtype=d) for k,d in
             [('directions0',torch.float64),(job['target_key'],torch.float64),('point_features',torch.float32)]]
    code_root=Path(__file__).resolve().parents[1]
    deps=['layout_margin_single/train.py','layout_fitdiag/train.py','layout_endpoint/train.py',
          'layout_endpoint/rollout.py','layout_student/train.py','layout_value/optimizer.py','legacy/networks.py','legacy/chart.py']
    energy=float(np.mean((data[job['target_key']][job['index']]-data['directions0'][job['index']])**2))
    config={'job':job_name,'job_id':job_name,'kind':'single','uid':job['uid'],'seed':job['seed'],
        'arm':'c0','target_kind':job['target_kind'],'target_key':job['target_key'],'pair_id':job['pair_id'],
        'total_updates':job['total_updates'],'start_update':0,'batch_size':4,'learning_rate':1e-4,'weight_decay':1e-4,
        'optimizer':'AdamW','scheduler':'constant','initialization_mode':job['initialization_mode'],
        'parameter_count':29907,'proxy_points':256,'network_extra_inputs':[],
        'device':'cpu','torch_threads':1,'interop_threads':1,'cuda_visible_devices':'','cuda_rng_recorded':False,
        'network_dtype':'float32','geometry_dtype':'float64','deterministic_algorithms':True,'rollout_steps':32,
        'objective':'frozen_endpoint_compute_loss_c0','train_uids':uids,'active_training_uids':[job['uid']],
        'target_displacement_energy':energy,'zero_target':energy==0,'relative_error_denominator_defined':energy>0,
        'endpoint_scale_squared':p['endpoint_scale_squared'],'endpoint_scale_unused_by_c0':True,
        'checkpoint_updates':p['training']['checkpoint_updates'],'checkpoint_selection':'predeclared_final_update_only',
        'protocol_sha256':file_sha256(run/'RUN_PROTOCOL.json'),'config_sha256':config_digest(p),
        'training_labels_sha256':p['hashes']['training_labels_sha256'],'data_manifest_sha256':p['hashes']['data_manifest_sha256'],
        'initial_checkpoint_sha256':initial_sha,'initial_model_sha256':source['model_sha256'],
        'source_sha256':{name:file_sha256(code_root/name) for name in deps}}
    s={'run':run,'protocol':p,'job':job,'config':config,'uids':uids,'data':data,'tensors':tensors,
       'model':model,'optimizer':optimizer,'scheduler':scheduler,'sampler':sampler,'trace':[],
       'timing_trace':[],'update':0,'elapsed_training_seconds':0.}
    if resume is not None:
        saved=torch.load(Path(resume),map_location='cpu',weights_only=False)
        validate_state(saved,config)
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']);sampler.load_state_dict(saved['sampler'])
        s.update(trace=list(saved['trace']),timing_trace=list(saved['timing_trace']),update=saved['update'],
                 elapsed_training_seconds=float(saved['elapsed_training_seconds']))
        restore_cpu_rng(saved['rng'])
    return s


def validate_state(state,config):
    require(state['config']==config,'Resume config/protocol/data/source mismatch')
    require(state['model_sha256']==model_sha256(state['model']),'Model hash mismatch')
    n=state['update'];require(len(state['trace'])==n,'Incomplete trace')
    require(state['step_sequence_sha256']==step_sequence_hash(state['trace']),'Bridge RNG sequence hash mismatch')
    require(set(state['rng'])=={'python','numpy','torch_cpu'},'CUDA RNG state forbidden')
    require(state['sampler']['count']==1 and state['sampler']['exposures']==[4*n],'Sampler exposure mismatch')
    require(all(x['uids']==config['active_training_uids']*4 for x in state['trace']),'Other parent in trace')
    require(state['scheduler']['last_epoch']==n and state['scheduler']['_step_count']==n+1,'Scheduler counter mismatch')
    require(state['scheduler']['base_lrs']==[1e-4] and state['scheduler']['_last_lr']==[1e-4],'Scheduler LR changed')
    for group in state['optimizer']['param_groups']:
        require(group['lr']==group['initial_lr']==1e-4 and group['weight_decay']==1e-4,'Adam settings changed')
        require(group['foreach'] is False and group['fused'] is False,'Adam execution mode changed')
    require(bool(state['optimizer']['state'])==(n>0),'Optimizer state/update mismatch')
    for item in state['optimizer']['state'].values():require(int(item['step'])==n,'Adam step mismatch')
    require(state['protocol_sha256']==config['protocol_sha256'] and state['job_id']==config['job_id'],'State identity changed')
    require(state['training_labels_sha256']==config['training_labels_sha256'] and
            state['initial_checkpoint_sha256']==config['initial_checkpoint_sha256'] and
            state['diagnostic_only'] is True,'State data/init/scope identity changed')
    require(state['is_predeclared_final']==(n==config['total_updates']),'Fixed final marker changed')


def update_once(s):
    update=s['update']+1;j=s['job']
    row=frozen_c0_step(s['model'],s['optimizer'],s['scheduler'],s['sampler'],[j['index']],
        s['uids'],s['tensors'],s['data']['faces'],s['protocol']['endpoint_scale_squared'],update,torch.device('cpu'))
    s['trace'].append(row);s['update']=update
    return row


def semantic_state(s,elapsed=None):
    c=s['config'];state={'model':{k:v.detach().cpu().clone() for k,v in s['model'].state_dict().items()},
        'optimizer':s['optimizer'].state_dict(),'scheduler':s['scheduler'].state_dict(),'rng':cpu_rng_state(),
        'sampler':s['sampler'].state_dict(),'trace':list(s['trace']),'timing_trace':list(s['timing_trace']),
        'config':c,'job_id':c['job_id'],'update':s['update'],'is_predeclared_final':s['update']==c['total_updates'],
        'diagnostic_only':True,'elapsed_training_seconds':s['elapsed_training_seconds'] if elapsed is None else elapsed,
        'protocol_sha256':c['protocol_sha256'],'training_labels_sha256':c['training_labels_sha256'],
        'initial_checkpoint_sha256':c['initial_checkpoint_sha256'],'step_sequence_sha256':step_sequence_hash(s['trace'])}
    state['model_sha256']=model_sha256(state['model']);validate_state(state,c)
    return state


def train_run(run,job_name,resume=None,*,stop_after=None,output_dir=None):
    s=make_session(run,job_name,resume);run=s['run'];c=s['config'];j=s['job']
    end=j['total_updates'] if stop_after is None else int(stop_after);first=s['update']
    require(first<end<=j['total_updates'],'Invalid continuation budget')
    out=resolve_under(run,output_dir if output_dir is not None else run/'training'/job_name)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'TRAIN_CONFIG.json',c)
    write_json(out/'SOURCE_IMPORT.json',{'model_only_source':j['initial_checkpoint'],
        'initial_checkpoint_sha256':c['initial_checkpoint_sha256'],'initial_model_sha256':c['initial_model_sha256'],
        'optimizer_inherited':False,'scheduler_inherited':False,'source_rng_inherited':False,
        'initialization_mode':j['initialization_mode'],'initial_checkpoint_means_fresh_cpu_update_zero':True})
    write_json(out/'START.json',{'start_update':first,'end_update':end,'final_budget':j['total_updates'],
        'resume':str(resume) if resume else None,'resume_sha256':file_sha256(resume) if resume else None})
    started=time.perf_counter();block_start=started;block_first=first+1;last=None
    def save(initial=False):
        value=semantic_state(s,s['elapsed_training_seconds']+time.perf_counter()-started)
        name='initial.pt' if initial else 'final.pt' if value['is_predeclared_final'] else f"step_{s['update']:05d}.pt"
        path=out/name
        with path.open('xb') as f:torch.save(value,f)
        return path
    if first==0:save(initial=True)
    try:
        for update in range(first+1,end+1):
            row=update_once(s)
            if update%1000==0 or update==end:
                timing={'start_update':block_first,'end_update':update,'updates':update-block_first+1,
                        'wall_seconds':time.perf_counter()-block_start,'device':'cpu','torch_threads':1}
                s['timing_trace'].append(timing)
                print(json.dumps({'job':job_name,'update':update,'loss':row['loss'],**timing}),flush=True)
                block_start=time.perf_counter();block_first=update+1
            if update in c['checkpoint_updates'] or update==end:last=save()
    except Exception as exc:
        write_json(out/'FAILED.json',{'type':type(exc).__name__,'error':str(exc),'completed_update':s['update']});raise
    counts=Counter(u for row in s['trace'] for u in row['uids'])
    require(counts==Counter({j['uid']:end*4}),'Unexpected parent exposure')
    report={'status':'COMPLETE' if end==j['total_updates'] else 'INTERRUPTED_FOR_RESUME_TEST',
        'job':job_name,'job_id':job_name,'uid':j['uid'],'target_kind':j['target_kind'],'seed':10,'update':end,
        'final_budget':j['total_updates'],'checkpoint':str(last),'checkpoint_sha256':file_sha256(last),
        'model_sha256':model_sha256(s['model'].state_dict()),'protocol_sha256':c['protocol_sha256'],
        'elapsed_training_seconds':s['elapsed_training_seconds']+time.perf_counter()-started,
        'elapsed_this_invocation_seconds':time.perf_counter()-started,'updates_this_invocation':end-first,
        'parent_exposures':dict(counts),'parameter_count':29907,'device':'cpu','torch_threads':1,
        'step_sequence_sha256':step_sequence_hash(s['trace']),'step_sequence_hash_encoding':'ordered uint8 bridge k, four bytes/update',
        'target_displacement_energy':c['target_displacement_energy'],'zero_target':c['zero_target'],
        'relative_error_denominator_defined':c['relative_error_denominator_defined'],'diagnostic_only':True,
        'selection_uses_internal_check':False,'internal_check_teacher_files_opened':0,'cuda_rng_recorded':False,
        'checkpoint_selection':'predeclared_final_update_only','last_loss':s['trace'][-1]['loss']}
    write_json(out/'TRAIN_TRACE.json',s['trace']);write_json(out/'TRAIN_TIMING.json',s['timing_trace'])
    write_json(out/'TRAIN_REPORT.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--job',required=True);p.add_argument('--resume',type=Path)
    p.add_argument('--stop-after',type=int);p.add_argument('--output-dir',type=Path)
    a=p.parse_args();print(json.dumps(train_run(a.run,a.job,a.resume,stop_after=a.stop_after,output_dir=a.output_dir)),flush=True)


if __name__=='__main__':main()
