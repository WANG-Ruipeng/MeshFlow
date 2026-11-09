"""Run only predeclared observations after completed CPU training reports appear."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--run',required=True);args=a.parse_args()
    run=Path(args.run).resolve();p=json.loads((run/'RUN_PROTOCOL.json').read_text());env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    receipt=run/'EVALUATION_FOLLOWER.json'
    if receipt.exists():raise FileExistsError(receipt)
    completed=[];started=time.monotonic();failure=None
    def call(module,*extra):
        subprocess.run([sys.executable,'-B','-m','experiments.a_radial.layout_margin_single.'+module,'--run',str(run),*extra],env=env,check=True)
    try:
        for job in p['jobs']:
            path=run/'training'/job/'TRAIN_REPORT.json'
            while not path.exists():
                if (path.parent/'FAILED.json').exists():raise RuntimeError('Training failure: '+job)
                if time.monotonic()-started>14400:raise TimeoutError('No completed training report within four hours')
                time.sleep(10)
            report=json.loads(path.read_text())
            if report['status']!='COMPLETE' or report['update']!=20000:raise ValueError('Not the predeclared completed training report')
            call('evaluate','--mode','diagnostics','--job',job);completed.append(job)
            print(json.dumps({'event':'fixed_job_diagnostics_complete','job':job}),flush=True)
        call('evaluate','--mode','final')
        call('report','--render')
        call('review')
    except Exception as exc:
        failure={'type':type(exc).__name__,'error':str(exc)}
        raise
    finally:
        with receipt.open('x') as stream:
            json.dump({'status':'COMPLETED' if failure is None else 'FAILED','completed_diagnostic_jobs':completed,
                'failure':failure,'elapsed_seconds':time.monotonic()-started,'checkpoint_selection':False,
                'CPU_only':True,'scientific_success_not_inferred_from_completion':True},stream,indent=2)


if __name__=='__main__':main()
