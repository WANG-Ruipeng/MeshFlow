"""One-shot fixed queue: balanced240k completion -> official2x50000 -> finish.

Waiting and heartbeats use only CPU filesystem reads. No retry, checkpoint
selection, smaller budget, or training operation occurs in --prepare-only.
"""
from __future__ import annotations
import argparse
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

REPOSITORY=Path(__file__).resolve().parents[3]
BALANCED_REL='output/a_balanced_20260925_01'
FORMAL_REL='output/a_system_official_closed16_20260925_01'
BALANCED_SHA='406c20c6b1c48518ae3c719fdf30c09d5a0651ea221b08d454dd8a04e52f8f75'
FORMAL_SHA='c63936c5db6ed510130138621cd8dc1b6034195b26f585e7b70c032bf016c545'
SAMPLING_SHA='8d2c7bcafd0bc1263226eb6d16bb826aeb68f06bd70a3494a0b255e213546cb3'
JOBS=['joint_10','joint_11']
HEARTBEAT_SECONDS=30


def require(ok,message):
    if not ok:raise ValueError(message)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):digest.update(chunk)
    return digest.hexdigest()

def put(path,value,replace=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not replace:raise FileExistsError(path)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def cuda_initialized():
    module=sys.modules.get('torch');return bool(module is not None and module.cuda.is_initialized())

def code_files(repo):
    names=['experiments/a_radial/system_formal/'+name for name in
        ['overnight_queue.py','test_overnight_queue.py','run_all.py','train.py','gate.py','sample.py','finish.py']]
    names+=['experiments/a_radial/layout_balanced/run_all.py','experiments/a_radial/layout_balanced/train.py']
    return {name:sha(repo/name) for name in names}

def bound_files(repo):
    balanced=repo/BALANCED_REL;formal=repo/FORMAL_REL
    paths=[balanced/'RUN_PROTOCOL.json',balanced/'TRAINING_GATE.json',formal/'RUN_PROTOCOL.json',
        formal/'SOURCE_MANIFEST.json',formal/'CONFIG.json',formal/'FORMAL_CONFIG.yaml',formal/'SAMPLING_PROTOCOL.json',
        formal/'SAMPLING_CPU_CHECKS.json']
    for relative in ('DATA_MANIFEST.json','SOURCE_MANIFEST.json'):
        path=balanced/relative
        if path.exists():paths.append(path)
    gate=Path(read(formal/'RUN_PROTOCOL.json')['gate_output'])
    require(gate.is_relative_to(repo/'output'),'Unregistered official gate location')
    paths.append(gate/'COMPLETE.json')
    return {str(path.relative_to(repo)):sha(path) for path in paths}

def protocol_gate(repo):
    balanced=repo/BALANCED_REL;formal=repo/FORMAL_REL
    require(sha(balanced/'RUN_PROTOCOL.json')==BALANCED_SHA,'Different balanced protocol')
    require(sha(formal/'RUN_PROTOCOL.json')==FORMAL_SHA,'Different official protocol')
    require(sha(formal/'SAMPLING_PROTOCOL.json')==SAMPLING_SHA,'Different fixed sampling/finish protocol')
    bp=read(balanced/'RUN_PROTOCOL.json');fp=read(formal/'RUN_PROTOCOL.json')
    require(set(bp['jobs'])==set(JOBS) and bp['training']['updates']==240000,'Wrong balanced job set or budget')
    for job,seed in zip(JOBS,(10,11)):
        item=bp['jobs'][job]
        require(item['seed']==seed and item['start_update']==20000 and item['total_updates']==240000
            and item['new_optimizer_updates']==220000,'Changed balanced continuation budget')
    require(fp['seeds']==[10,11] and fp['formal_updates']==50000 and fp['primary_weight']=='ema','Wrong official queue budget/selection')
    gate=read(balanced/'TRAINING_GATE.json')
    require(gate['status']=='APPROVED_BALANCED_OBJECTIVE_FIXED_BUDGET' and gate['protocol_sha256']==BALANCED_SHA,'Balanced CUDA gate absent')
    for relative,digest in gate['training_source_hashes'].items():
        require(sha(repo/'experiments/a_radial'/relative)==digest,'Frozen balanced training source changed')
    formal_gate=read(Path(fp['gate_output'])/'COMPLETE.json')
    require(formal_gate['status']=='PASS' and formal_gate['formal_protocol_sha256']==FORMAL_SHA,'Official CUDA gate absent')
    return bp,fp

def ensure_official_fresh(formal):
    for marker in ('QUEUE_PROTOCOL.json','QUEUE_COMPLETE.json','QUEUE_FAILED.json','posttraining/FINISH_STARTED.json'):
        require(not (formal/marker).exists(),'Official queue already attempted: '+marker)
    for seed in (10,11):
        directory=formal/'training'/f'seed_{seed}'
        require(not (directory/'TRAINING_COMPLETE.json').exists() and not list((directory/'invocations').glob('*')),
            'Official training already exists; this queue never silently resumes or retries')

def prepare(queue,timeout_seconds=10800):
    repo=REPOSITORY;queue=Path(queue).resolve()
    require(queue.is_relative_to(repo/'output'),'Queue evidence must remain inside this repository output directory')
    require(isinstance(timeout_seconds,int) and 1<=timeout_seconds<=86400,'Invalid fixed wait timeout')
    protocol_gate(repo);ensure_official_fresh(repo/FORMAL_REL);require(not cuda_initialized(),'Queue preparation must be CPU-only')
    queue.mkdir(parents=True,exist_ok=False)
    plan={'schema':1,'scope':'FIXED_ONE_SHOT_BALANCED_THEN_OFFICIAL_THEN_FINISH','repository':str(repo),
        'balanced_run':str(repo/BALANCED_REL),'official_run':str(repo/FORMAL_REL),
        'balanced_protocol_sha256':BALANCED_SHA,'official_protocol_sha256':FORMAL_SHA,'sampling_protocol_sha256':SAMPLING_SHA,
        'balanced_jobs':JOBS,'balanced_global_update_each':240000,'balanced_new_optimizer_updates_total':440000,
        'official_seeds':[10,11],'official_updates_each':50000,'official_weight':'ema',
        'heartbeat_seconds':HEARTBEAT_SECONDS,'balanced_wait_timeout_seconds':timeout_seconds,
        'timeout_applies_only_to_waiting_for_balanced_not_to_official_training':True,
        'python_executable':str(Path(sys.executable).resolve()),'source_sha256':code_files(repo),'bound_files_sha256':bound_files(repo),
        'automatic_retries':False,'checkpoint_selection':'registered fixed finals only','quality_loss_does_not_stop_or_select':True,
        'stages':['WAIT_BALANCED240K','OFFICIAL_RUN_ALL_2X50000','FROZEN_FINISH'],
        'prepare_only_does_not_launch_any_process':True,'created_utc':utc()}
    put(queue/'PLAN.json',plan)
    for name in plan['source_sha256']:
        target=queue/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((repo/name).read_bytes())
    put(queue/'STATUS.json',{'status':'PREPARED_NOT_STARTED','plan_sha256':sha(queue/'PLAN.json'),'GPU_initialized':False})
    return plan

def load_plan(queue):
    queue=Path(queue).resolve();plan=read(queue/'PLAN.json');repo=REPOSITORY
    require(plan['schema']==1 and plan['scope']=='FIXED_ONE_SHOT_BALANCED_THEN_OFFICIAL_THEN_FINISH','Wrong queue scope')
    require(Path(plan['repository']).resolve()==repo and Path(plan['balanced_run']).resolve()==repo/BALANCED_REL
        and Path(plan['official_run']).resolve()==repo/FORMAL_REL,'Queue permits only the two explicitly registered runs')
    require(plan['balanced_protocol_sha256']==BALANCED_SHA and plan['official_protocol_sha256']==FORMAL_SHA
        and plan['sampling_protocol_sha256']==SAMPLING_SHA,'Queue protocol identity changed')
    require(plan['heartbeat_seconds']==30 and plan['automatic_retries'] is False,'Polling/retry policy changed')
    require(isinstance(plan['balanced_wait_timeout_seconds'],int) and 1<=plan['balanced_wait_timeout_seconds']<=86400,'Invalid prepared wait timeout')
    require(plan['balanced_new_optimizer_updates_total']==440000 and plan['official_updates_each']==50000
        and plan['official_weight']=='ema' and plan['official_seeds']==[10,11] and plan['balanced_jobs']==JOBS,'Changed queue budgets')
    require(plan['python_executable']==str(Path(sys.executable).resolve()),'Different Python environment')
    require(plan['source_sha256']==code_files(repo),'Queue/driver/finish source changed after registration')
    for name,digest in plan['source_sha256'].items():require(sha(queue/'source_snapshot'/name)==digest,'Queue source snapshot changed')
    for name,digest in plan['bound_files_sha256'].items():require(sha(repo/name)==digest,'Bound protocol/gate/data/source file changed: '+name)
    protocol_gate(repo);return plan

def last_logged_update(path):
    if not path.exists():return None
    with path.open('rb') as stream:
        stream.seek(0,2);length=stream.tell();stream.seek(max(0,length-32768));lines=stream.read().decode('utf-8',errors='replace').splitlines()
    for line in reversed(lines):
        try:row=json.loads(line)
        except (ValueError,TypeError):continue
        if isinstance(row,dict) and isinstance(row.get('update'),int):return {k:row[k] for k in ['job','update','loss'] if k in row}
    return None

def failure_files(balanced):
    paths=[balanced/'TRAINING_FAILURE.json',balanced/'FAILED.json']
    paths += [balanced/'training'/job/'FAILED.json' for job in JOBS]
    return [{'path':str(path),'sha256':sha(path),'record':read(path)} for path in paths if path.exists()]

def balanced_status(balanced):
    balanced=Path(balanced);failures=failure_files(balanced)
    if failures:return {'status':'FAILED','failures':failures}
    if (balanced/'TRAINING_COMPLETE.json').exists():return {'status':'COMPLETION_PRESENT_REQUIRES_VALIDATION'}
    jobs={}
    for job in JOBS:
        root=balanced/'training'/job
        saved=[update for update in (20000,50000,100000) if (root/f'step_{update:05d}.pt').exists()]
        jobs[job]={'last_logged':last_logged_update(balanced/'training_logs'/(job+'.log')),
            'last_saved_intermediate_update':max(saved) if saved else None,'final_checkpoint_exists':(root/'final.pt').exists(),
            'completion_report_exists':(root/'TRAIN_REPORT.json').exists()}
    progress=read(balanced/'TRAINING_PROGRESS.json') if (balanced/'TRAINING_PROGRESS.json').exists() else None
    return {'status':'WAITING','driver_progress':progress,'jobs':jobs,'completion_sentinel_exists':False}

def validate_completion_record(complete,protocol_sha,driver_sha):
    require(complete['status']=='COMPLETE' and complete['protocol_sha256']==protocol_sha and complete['driver_sha256']==driver_sha,'Balanced completion provenance differs')
    require(complete['new_optimizer_updates_total']==440000 and complete['global_updates_sum_including_imported_parent_traces']==480000,'Balanced effective budget differs')
    require([row['job'] for row in complete['jobs']]==JOBS and all(row['returncode']==0 for row in complete['jobs']),'Both balanced jobs must exit0 exactly once')

def tensor_model_hash(state):
    digest=hashlib.sha256()
    for name in sorted(state):
        value=state[name].detach().cpu().contiguous()
        digest.update(json.dumps([name,str(value.dtype),list(value.shape)],separators=(',',':')).encode());digest.update(value.numpy().tobytes())
    return digest.hexdigest()

def validate_balanced_state(state,report,config,protocol,job,protocol_sha):
    item=protocol['jobs'][job];seed=item['seed']
    require(report['status']=='COMPLETE' and report['job']==job and report['seed']==seed
        and report['update']==report['final_budget']==240000,'Balanced job report is incomplete or wrong seed')
    require(report['imported_parent_updates']==20000 and report['new_optimizer_updates_since_import']==220000,'Balanced imported/new update accounting changed')
    require(report['protocol_sha256']==state['protocol_sha256']==config['protocol_sha256']==protocol_sha,'Balanced checkpoint protocol changed')
    require(state['update']==240000 and state['is_predeclared_final'] is True and state['job_id']==job,'Only registered balanced final240k allowed')
    require(state['config']==config and config['job_id']==job and config['job']==job and config['seed']==seed
        and config['arm']=='c0_parent_weighted' and config['total_updates']==240000 and config['start_update']==20000,'Balanced config identity differs')
    require(config['active_training_uids']==item['active_training_uids']==protocol['train_uids'],'Balanced active parents changed')
    require(state['initial_checkpoint_sha256']==config['initial_checkpoint_sha256']==item['initial_checkpoint_sha256'],'Balanced initialization differs')
    require(state['training_labels_sha256']==config['training_labels_sha256']==protocol['hashes']['training_labels_sha256'],'Balanced training data differs')
    require(state['model_sha256']==report['model_sha256']==tensor_model_hash(state['model']),'Balanced final model bytes differ')
    require(len(state['trace'])==240000 and all(row['update']==i+1 for i,row in enumerate(state['trace'])),'Balanced full trace is incomplete')
    require(state['scheduler']['last_epoch']==240000 and state['optimizer']['state']
        and all(int(v['step'])==240000 for v in state['optimizer']['state'].values()),'Balanced optimizer/scheduler counters differ')
    n=state['rng']['numpy']
    require(isinstance(n,tuple) and len(n)==5 and n[0]=='MT19937' and str(n[1].dtype)=='uint32' and n[1].shape==(624,)
        and 'python' in state['rng'] and 'torch_cpu' in state['rng'],'Original complete NumPy/Python/CPU RNG schema differs')
    require(state['rng']['torch_cuda'] and len(state['sampler']['exposures'])==12
        and state['sampler']['exposures']==[80000]*12,'Balanced RNG/sampler/exposure state differs')
    require(report['parent_exposures']=={uid:80000 for uid in protocol['train_uids']}
        and report['global_parent_exposures_total']==960000,'Balanced report exposure totals differ')

def load_balanced_checkpoint_cpu(path):
    """Original NumPy MT19937 RNG needs four narrowly scoped safe constructors.

    weights_only stays True; unregistered functions/classes remain rejected.
    The context restores the process allowlist on both success and failure.
    """
    import numpy as np
    import torch
    allowed=[(np.core.multiarray._reconstruct,'numpy.core.multiarray._reconstruct'),
        np.ndarray,np.dtype,type(np.dtype('uint32'))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path,map_location='cpu',weights_only=True,mmap=True)


def completed_balanced_evidence(plan):
    balanced=Path(plan['balanced_run']);require(not failure_files(balanced),'Balanced failure sentinel exists')
    require(sha(balanced/'RUN_PROTOCOL.json')==plan['balanced_protocol_sha256'],'Balanced protocol changed')
    complete=read(balanced/'TRAINING_COMPLETE.json');protocol=read(balanced/'RUN_PROTOCOL.json')
    driver_sha=plan['source_sha256']['experiments/a_radial/layout_balanced/run_all.py']
    validate_completion_record(complete,plan['balanced_protocol_sha256'],driver_sha);jobs=[]
    for job in JOBS:
        folder=balanced/'training'/job;path=folder/'final.pt';report=read(folder/'TRAIN_REPORT.json');config=read(folder/'TRAIN_CONFIG.json')
        require(Path(report['checkpoint']).resolve()==path.resolve(),'Balanced checkpoint path differs')
        digest=sha(path);require(digest==report['checkpoint_sha256'],'Balanced final checkpoint file SHA differs')
        state=load_balanced_checkpoint_cpu(path)
        validate_balanced_state(state,report,config,protocol,job,plan['balanced_protocol_sha256'])
        require(sha(path)==digest,'Balanced checkpoint changed during validation')
        jobs.append({'job':job,'checkpoint':str(path),'checkpoint_sha256':digest,'model_sha256':state['model_sha256'],
            'report_sha256':sha(folder/'TRAIN_REPORT.json'),'config_sha256':sha(folder/'TRAIN_CONFIG.json'),'final_update':240000,'new_updates':220000})
        del state;gc.collect()
    require(not cuda_initialized(),'CPU gate unexpectedly initialized GPU')
    require(not failure_files(balanced),'Balanced failure appeared during validation')
    return {'status':'PASS','protocol_sha256':plan['balanced_protocol_sha256'],'completion_sha256':sha(balanced/'TRAINING_COMPLETE.json'),
        'new_optimizer_updates_total':440000,'global_updates_sum':480000,'jobs':jobs,'GPU_initialized':False,
        'scientific_quality_used_as_gate':False}

def wait_for_balanced(plan,emit,status_reader=balanced_status,evidence_reader=completed_balanced_evidence,clock=time.monotonic,sleep=time.sleep):
    started=clock();timeout=plan['balanced_wait_timeout_seconds']
    while True:
        status=status_reader(Path(plan['balanced_run']));elapsed=clock()-started
        emit('WAIT_BALANCED240K',{'elapsed_wait_seconds':elapsed,'timeout_seconds':timeout,'balanced':status})
        require(status['status']!='FAILED','Balanced training failure sentinel: '+json.dumps(status))
        if status['status']=='COMPLETION_PRESENT_REQUIRES_VALIDATION':return evidence_reader(plan)
        if elapsed>=timeout:raise TimeoutError('Balanced fixed completion was not available within registered wait timeout')
        sleep(min(plan['heartbeat_seconds'],timeout-elapsed))

def formal_snapshot(formal):
    result={}
    for name in ('STATUS.json','QUEUE_PROGRESS.json'):
        path=formal/name
        if path.exists():result[name]=read(path)
    for seed in (10,11):
        path=formal/'training'/f'seed_{seed}'/'PROGRESS.json'
        if path.exists():result[f'seed_{seed}']=read(path)
    return result

def terminate_group(process):
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait()

def run_process(queue,plan,stage,module,emit):
    log=queue/'logs'/(stage+'.log');log.parent.mkdir(exist_ok=True);formal=Path(plan['official_run'])
    command=[plan['python_executable'],'-B','-m',module,'--run',str(formal)]
    put(queue/(stage+'_INVOCATION.json'),{'command':command,'stage':stage,'created_utc':utc(),
        'source_sha256':plan['source_sha256'],'plan_sha256':sha(queue/'PLAN.json'),'automatic_retry':False})
    process=None;started=time.monotonic()
    try:
        with log.open('x',encoding='utf-8') as stream:
            process=subprocess.Popen(command,cwd=REPOSITORY,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            while process.poll() is None:
                emit(stage,{'child_pid':process.pid,'elapsed_stage_seconds':time.monotonic()-started,'log':str(log),
                    'formal':formal_snapshot(formal),'GPU_initialized_in_controller':cuda_initialized()})
                time.sleep(HEARTBEAT_SECONDS)
            code=process.returncode
        record={'stage':stage,'returncode':code,'command':command,'log':str(log),'log_sha256':sha(log),
            'elapsed_seconds':time.monotonic()-started,'child_pid':process.pid}
        put(queue/(stage+'_EXIT.json'),record)
        if code:
            terminate_group(process);raise RuntimeError('Registered child stage failed without retry: '+stage+' returncode='+str(code))
        return record
    except BaseException:
        if process is not None:terminate_group(process)
        raise

def formal_queue_evidence(plan):
    formal=Path(plan['official_run']);done=read(formal/'QUEUE_COMPLETE.json')
    require(not (formal/'QUEUE_FAILED.json').exists(),'Official queue failure sentinel exists')
    require(done['status']=='TWO_SEED_FIXED_BUDGET_DONE_FIT_SAMPLING_PENDING'
        and done['protocol_sha256']==plan['official_protocol_sha256']
        and done['source_sha256']==plan['source_sha256']['experiments/a_radial/system_formal/run_all.py'],'Official queue completion identity differs')
    require([r['seed'] for r in done['records']]==[10,11] and all(r['returncode']==0 for r in done['records']),'Official two-seed processes did not both exit0')
    seeds=[]
    for seed in (10,11):
        path=formal/'training'/f'seed_{seed}'/'TRAINING_COMPLETE.json';record=read(path)
        require(record['status']=='FIXED_BUDGET_COMPLETE_UNQUALIFIED' and record['seed']==seed and record['step']==50000
            and record['primary_weight']=='ema' and record['protocol_sha256']==plan['official_protocol_sha256'],
            'Official fixed final EMA completion differs')
        seeds.append({'seed':seed,'completion_sha256':sha(path),'checkpoint':record['primary_checkpoint'],
            'checkpoint_sha256':record['primary_checkpoint_sha256'],'full_checkpoint_revalidated_by_frozen_finish':True})
    return {'status':'PASS','queue_complete_sha256':sha(formal/'QUEUE_COMPLETE.json'),'seeds':seeds,'scientific_qualification':False}

def run(queue):
    queue=Path(queue).resolve()
    require((queue/'PLAN.json').exists(),'Prepared queue PLAN is required')
    require(not (queue/'STARTED.json').exists() and not (queue/'FAILED.json').exists(),'Queue was already attempted; automatic retry is forbidden')
    counter=0;stage='PRELAUNCH_VALIDATION';records=[]
    def emit(active,details):
        nonlocal counter,stage
        stage=active;counter+=1;record={'status':'RUNNING','stage':active,'heartbeat':counter,'utc':utc(),**details}
        put(queue/'heartbeats'/f'{counter:06d}.json',record);put(queue/'STATUS.json',record,replace=True)
        print(json.dumps(record,ensure_ascii=False),flush=True)
    try:
        plan=load_plan(queue);formal=Path(plan['official_run']);ensure_official_fresh(formal)
        require(not cuda_initialized(),'Controller must remain CPU-only')
        # Exclusive markers prevent another controller from launching the same official run.
        put(queue/'STARTED.json',{'plan_sha256':sha(queue/'PLAN.json'),'pid':os.getpid(),'started_utc':utc()})
        claim=formal/'OVERNIGHT_QUEUE_CLAIM.json'
        put(claim,{'queue':str(queue),'plan_sha256':sha(queue/'PLAN.json'),'pid':os.getpid(),'utc':utc(),'automatic_retries':False})
        evidence=wait_for_balanced(plan,emit);put(queue/'BALANCED_GATE.json',evidence)
        load_plan(queue);ensure_official_fresh(formal)
        records.append(run_process(queue,plan,'OFFICIAL_RUN_ALL_2X50000','experiments.a_radial.system_formal.run_all',emit))
        put(queue/'OFFICIAL_COMPLETION_GATE.json',formal_queue_evidence(plan));load_plan(queue)
        records.append(run_process(queue,plan,'FROZEN_FINISH','experiments.a_radial.system_formal.finish',emit))
        finished=read(formal/'posttraining/FINISH_COMPLETE.json')
        require(finished['status']=='FIXED_BUDGET_SAMPLING_AND_MAIN_TABLE_DONE_UNQUALIFIED'
            and finished['training_protocol_sha256']==FORMAL_SHA and finished['sampling_protocol_sha256']==SAMPLING_SHA,'Finish completion differs')
        result={'status':'COMPLETE_UNQUALIFIED','plan_sha256':sha(queue/'PLAN.json'),'stages':records,
            'finish_complete_sha256':sha(formal/'posttraining/FINISH_COMPLETE.json'),'finished_utc':utc(),
            'scientific_convergence_or_generalization_claim':False,'automatic_retries':False,'GPU_initialized_in_controller':cuda_initialized()}
        put(queue/'COMPLETE.json',result);put(queue/'STATUS.json',result,replace=True);return result
    except BaseException as exc:
        failed={'status':'FAILED','stage':stage,'error':repr(exc),'traceback':traceback.format_exc(),
            'completed_stages':records,'utc':utc(),'automatic_retries':False,'all_logs_and_checkpoints_retained':True,
            'GPU_initialized_in_controller':cuda_initialized()}
        put(queue/'FAILED.json',failed);put(queue/'STATUS.json',failed,replace=True);print(json.dumps(failed),flush=True);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--queue',required=True);parser.add_argument('--timeout-seconds',type=int,default=10800);args=parser.parse_args()
    if args.prepare_only:print(json.dumps(prepare(args.queue,args.timeout_seconds),indent=2),flush=True)
    else:
        require(args.timeout_seconds==10800,'Runtime timeout overrides forbidden; use the prepared plan')
        run(args.queue)


if __name__=='__main__':main()
