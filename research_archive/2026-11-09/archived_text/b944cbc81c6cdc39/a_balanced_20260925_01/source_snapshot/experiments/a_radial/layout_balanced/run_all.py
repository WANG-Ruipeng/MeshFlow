"""Serial fixed-budget driver; entry requires the separate recorded CUDA gate."""
import argparse, datetime, json, subprocess, sys, time
from pathlib import Path
from ..io_utils import file_hash, write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    run=Path(args.run).resolve();root=Path(__file__).resolve().parents[1]
    protocol=json.loads((run/'RUN_PROTOCOL.json').read_text());gate=json.loads((run/'TRAINING_GATE.json').read_text())
    if gate['status']!='APPROVED_BALANCED_OBJECTIVE_FIXED_BUDGET' or gate['protocol_sha256']!=file_hash(run/'RUN_PROTOCOL.json'):
        raise ValueError('Missing or different diagnostic training gate')
    for relative,digest in gate['training_source_hashes'].items():
        if file_hash(root/relative)!=digest:raise ValueError('Tested training source changed: '+relative)
    jobs=['joint_10','joint_11']
    if set(protocol['jobs'])!=set(jobs):raise ValueError('Unregistered diagnostic jobs')
    logs=run/'training_logs';logs.mkdir(exist_ok=False)
    started=time.perf_counter();done=[]
    for job in jobs:
        print(json.dumps({'event':'balanced_start','job':job,'fixed_final_update':240000}),flush=True)
        command=[sys.executable,'-B','-m','experiments.a_radial.layout_balanced.train','--run',str(run),'--job',job]
        begin=time.perf_counter()
        with (logs/(job+'.log')).open('x') as log:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            code=process.wait()
        record={'job':job,'returncode':code,'wall_seconds_including_process_setup':time.perf_counter()-begin,'log':str((logs/(job+'.log')).relative_to(run))}
        done.append(record)
        write_json(run/'TRAINING_PROGRESS.json',{'jobs':done,'planned_jobs':jobs,'complete':len(done)==2 and code==0},replace=(run/'TRAINING_PROGRESS.json').exists())
        if code:
            write_json(run/'TRAINING_FAILURE.json',record);raise RuntimeError('Diagnostic training failed; log preserved: '+job)
    result={'status':'COMPLETE','jobs':done,'new_optimizer_updates_total':440000,'global_updates_sum_including_imported_parent_traces':480000,
        'serial_GPU_training_jobs':True,'concurrent_CPU_evaluation_possible':True,'training_wall_time_is_not_isolated_benchmark':True,'elapsed_wall_seconds':time.perf_counter()-started,
        'protocol_sha256':file_hash(run/'RUN_PROTOCOL.json'),'driver_sha256':file_hash(__file__),
        'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json(run/'TRAINING_COMPLETE.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()