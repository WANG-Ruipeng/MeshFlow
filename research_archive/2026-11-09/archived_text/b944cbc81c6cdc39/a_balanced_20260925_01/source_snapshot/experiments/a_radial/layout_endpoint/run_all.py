"""Run the six authorized frozen training jobs serially, preserving full logs."""
import argparse,datetime,json,os,subprocess,sys,time
from pathlib import Path
from ..io_utils import file_hash,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run).resolve()
    protocol=json.loads((run/'RUN_PROTOCOL.json').read_text());gate=json.loads((run/'TRAINING_GATE.json').read_text())
    assert gate['status']=='APPROVED_FOR_PREDECLARED_V1_INITIALIZED_STUDY'
    assert gate['protocol_sha256']==file_hash(run/'RUN_PROTOCOL.json')
    for relative,expected in gate['training_source_hashes'].items():
        assert file_hash(Path(__file__).parents[1]/relative)==expected
    logs=run/'training_logs';logs.mkdir(exist_ok=False)
    start=time.perf_counter();records=[]
    env=dict(os.environ);env['PYTHONUNBUFFERED']='1';env['PYTHONDONTWRITEBYTECODE']='1'
    for arm in protocol['training']['arms']:
        for seed in protocol['training']['seeds']:
            command=[sys.executable,'-B','-m','experiments.a_radial.layout_endpoint.train','--run',str(run),
                     '--arm',arm,'--seed',str(seed)]
            before=time.perf_counter()
            print(json.dumps({'event':'formal_training_start','arm':arm,'seed':seed,'updates':1000}),flush=True)
            with (logs/f'{arm}_{seed}.log').open('x') as log:
                job=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env)
                for line in job.stdout:
                    log.write(line);log.flush();print(line,end='',flush=True)
                code=job.wait()
            record={'arm':arm,'seed':seed,'returncode':code,'wall_seconds_including_process_setup':time.perf_counter()-before,
                    'log':str((logs/f'{arm}_{seed}.log').relative_to(run))}
            records.append(record)
            write_json(run/'TRAINING_PROGRESS.json',{'jobs':records,'next_jobs_not_started':6-len(records),
                'elapsed_wall_seconds':time.perf_counter()-start},replace=(run/'TRAINING_PROGRESS.json').exists())
            if code:
                write_json(run/'TRAINING_FAILURE.json',{'failed_job':record,'previous_jobs':records,
                    'no_automatic_parameter_change':True})
                raise SystemExit(code)
    write_json(run/'TRAINING_COMPLETE.json',{'status':'COMPLETE','jobs':records,'optimizer_updates_total':6000,
        'elapsed_wall_seconds':time.perf_counter()-start,'serial_training_no_competing_evaluation':True,
        'protocol_sha256':file_hash(run/'RUN_PROTOCOL.json'),'driver_sha256':file_hash(__file__),
        'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})

if __name__=='__main__':main()