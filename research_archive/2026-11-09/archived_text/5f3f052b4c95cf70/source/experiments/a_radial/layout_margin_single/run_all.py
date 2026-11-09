"""Run the eight registered CPU-only jobs to their fixed final update, once."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,subprocess,sys,time,datetime
from pathlib import Path
from ..io_utils import file_hash,write_json

def require(ok,message):
    if not ok:raise ValueError(message)

def validate_report(report,job,p):
    j=p['jobs'][job]
    require(report['status']=='COMPLETE' and report['update']==20000,'Not a fixed final completion')
    require(report['job_id']==job and report['uid']==j['uid'] and report['target_kind']==j['target_kind'],'Job identity changed')
    require(report['updates_this_invocation']==20000 and report['parent_exposures']=={j['uid']:80000},'Budget/exposure differs')
    require(report['device']=='cpu' and report['torch_threads']==1 and report['parameter_count']==29907,'Execution/architecture differs')
    require(report['checkpoint_selection']=='predeclared_final_update_only' and not report['cuda_rng_recorded'],'Selection/RNG differs')
    return report['step_sequence_sha256']

def run_all(run):
    run=Path(run).resolve();root=Path(__file__).resolve().parents[1]
    p=json.loads((run/'RUN_PROTOCOL.json').read_text());gate=json.loads((run/'TRAINING_GATE.json').read_text())
    require(gate['status']=='APPROVED_FIXED_CPU_8X20000' and gate['protocol_sha256']==file_hash(run/'RUN_PROTOCOL.json'),'CPU training gate missing/mismatched')
    smoke_path=Path(gate['smoke_path']);require(file_hash(smoke_path)==gate['smoke_sha256'],'CPU smoke changed')
    smoke=json.loads(smoke_path.read_text())
    require(smoke['status']=='PASS' and smoke['actual_optimizer_updates']==144 and smoke['formal_protocol_sha256']==gate['protocol_sha256'],'Actual CPU gate differs')
    for rel,digest in gate['training_source_hashes'].items():require(file_hash(root/rel)==digest,'Gated source changed: '+rel)
    require(file_hash(__file__)==gate['driver_sha256'],'Driver changed')
    jobs=[prefix+'_'+uid[:8] for uid in p['train_uids'] for prefix in ('old','margin')]
    require(len(jobs)==8 and set(jobs)==set(p['jobs']),'Registered job set differs')
    logs=run/'training_logs';logs.mkdir(exist_ok=False);done=[];pairs={};started=time.perf_counter();job=None
    try:
        for job in jobs:
            command=[sys.executable,'-B','-m','experiments.a_radial.layout_margin_single.train','--run',str(run),'--job',job]
            begin=time.perf_counter();print(json.dumps({'event':'start','job':job,'final_update':20000}),flush=True)
            with (logs/(job+'.log')).open('x') as stream:
                proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
                for line in proc.stdout:stream.write(line);stream.flush();print(line,end='',flush=True)
                code=proc.wait()
            require(code==0,'CPU training failed: '+job)
            folder=run/'training'/job;report=json.loads((folder/'TRAIN_REPORT.json').read_text())
            sequence=validate_report(report,job,p)
            path=folder/'final.pt';require(Path(report['checkpoint']).resolve()==path.resolve() and file_hash(path)==report['checkpoint_sha256'],'Final checkpoint differs')
            require(report['protocol_sha256']==gate['protocol_sha256'],'Report protocol differs')
            import torch
            from .train import validate_state
            require(not torch.cuda.is_initialized(),'CPU driver initialized CUDA')
            state=torch.load(path,map_location='cpu',weights_only=False)
            config=json.loads((folder/'TRAIN_CONFIG.json').read_text());validate_state(state,config)
            require(state['update']==20000 and state['model_sha256']==report['model_sha256'],'Full final state differs')
            del state
            uid=p['jobs'][job]['uid'];pairs.setdefault(uid,[]).append(sequence)
            require(len(set(pairs[uid]))==1,'Paired target bridge random sequences differ')
            done.append({'job':job,'returncode':code,'wall_seconds_including_setup':time.perf_counter()-begin,
                'checkpoint_sha256':report['checkpoint_sha256'],'model_sha256':report['model_sha256'],
                'report_sha256':file_hash(folder/'TRAIN_REPORT.json'),'step_sequence_sha256':sequence})
            write_json(run/'TRAINING_PROGRESS.json',{'completed_jobs':done,'planned_jobs':jobs,'complete':len(done)==8},replace=(run/'TRAINING_PROGRESS.json').exists())
        require(len(pairs)==4 and all(len(v)==2 and len(set(v))==1 for v in pairs.values()),'Incomplete matched pairs')
        result={'status':'COMPLETE','jobs':done,'actual_optimizer_updates':160000,'paired_k_sequence_hashes_equal':True,
            'per_parent_per_target_exposures':80000,'protocol_sha256':gate['protocol_sha256'],'driver_sha256':file_hash(__file__),
            'training_gate_sha256':file_hash(run/'TRAINING_GATE.json'),'CPU_only':True,'no_loss_early_stopping':True,
            'elapsed_wall_seconds':time.perf_counter()-started,'timing_is_not_isolated_benchmark':True,
            'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        write_json(run/'TRAINING_COMPLETE.json',result);print(json.dumps(result),flush=True)
    except BaseException as exc:
        write_json(run/'TRAINING_FAILURE.json',{'job':job,'error':repr(exc),'completed_jobs':done,'automatic_retries':False})
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args();run_all(args.run)