"""Serial overnight queue for the already-frozen two-seed full MeshFlow budget."""
from pathlib import Path
import argparse,datetime,json,subprocess,sys,time
from ..io_utils import file_hash,write_json
from .train import load_contract,gate_is_ready

def run(run):
    run=Path(run).resolve();protocol,cfg,identity=load_contract(run)
    gate=gate_is_ready(run,protocol,identity)
    if protocol['seeds']!=[10,11] or protocol['formal_updates']!=50000:
        raise ValueError('Only the frozen two-seed50000 queue is authorized by this driver')
    for seed in protocol['seeds']:
        directory=run/'training'/f'seed_{seed}'
        if list((directory/'invocations').glob('*')) or (directory/'TRAINING_COMPLETE.json').exists():
            raise ValueError('This fresh queue refuses existing training; resume must be explicit')
    logs=run/'queue_logs';logs.mkdir(exist_ok=False)
    plan={'scope':'FROZEN_FORMAL_BUDGET_SERIAL_QUEUE','protocol_sha256':identity['protocol_sha256'],
        'seeds':protocol['seeds'],'updates_each':50000,'loss_based_early_stop':False,'automatic_retries':False,
        'source_sha256':file_hash(__file__),'CUDA_gate':gate,
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'fit_and_sampling_qualification_pending_after_training':True}
    write_json(run/'QUEUE_PROTOCOL.json',plan);started=time.perf_counter();records=[]
    for seed in protocol['seeds']:
        command=[sys.executable,'-B','-m','experiments.a_radial.system_formal.train','train','--run',str(run),'--seed',str(seed)]
        before=time.perf_counter();print(json.dumps({'event':'FORMAL_SEED_START','seed':seed,'fixed_updates':50000}),flush=True)
        with (logs/f'seed_{seed}.log').open('x',encoding='utf-8') as stream:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:stream.write(line);stream.flush();print(line,end='',flush=True)
            code=process.wait()
        record={'seed':seed,'returncode':code,'elapsed_seconds':time.perf_counter()-before,'log':str((logs/f'seed_{seed}.log').relative_to(run))}
        records.append(record)
        write_json(run/'QUEUE_PROGRESS.json',{'completed_processes':records,'planned_seeds':protocol['seeds'],
            'all_processes_finished':len(records)==2,'all_success':all(r['returncode']==0 for r in records)},replace=(run/'QUEUE_PROGRESS.json').exists())
        if code:
            write_json(run/'QUEUE_FAILED.json',record)
            raise RuntimeError('Training process failed; original invocation and full checkpoints retained')
        result=json.loads((run/'training'/f'seed_{seed}'/'TRAINING_COMPLETE.json').read_text())
        if result['step']!=50000 or result['protocol_sha256']!=identity['protocol_sha256']:
            raise ValueError('Child did not complete frozen budget')
    write_json(run/'QUEUE_COMPLETE.json',{'status':'TWO_SEED_FIXED_BUDGET_DONE_FIT_SAMPLING_PENDING',
        'records':records,'elapsed_seconds':time.perf_counter()-started,'source_sha256':file_hash(__file__),
        'protocol_sha256':identity['protocol_sha256'],'automatic_scientific_success_claim':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run(a.run)