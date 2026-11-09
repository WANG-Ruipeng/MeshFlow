"""Resumable fixed task; stage 2 is conditional, never a mandatory GPU job."""
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name,'2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import traceback
from .common import *


def run(out=OUT,through='all'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    from .ledger import Budget
    budget=Budget(out)
    atomic_json(out/'controller_status.json',dict(status='RUNNING',through=through,entrypoint=__package__+'.execute'))
    try:
        from . import registration,protocol,endpoint
        registration.prepare(out)
        protocol.freeze(out);protocol.verify(out)
        if through=='prepare':return dict(status='PREPARED')
        if (out/'endpoint_summary.json').exists():
            prior=read_json(out/'endpoint_summary.json')
            assert prior['status']=='COMPLETE' and prior['outputs']==16
            for row in prior['requests']:
                actual=read_json(result_path(out,row))
                assert actual==row and actual['status']=='PROJECTED'
                assert digest(resolve(actual['raw_path']))==actual['raw_sha256']
                assert digest(resolve(actual['intervention_log_path']))==actual['intervention_log_sha256']
        else:
            endpoint.run(out,budget,workers=2)
        if through=='endpoint':
            result=dict(status='ENDPOINT_COMPLETE',model_calls=0)
            atomic_json(out/'controller_status.json',result);return result
        from . import evaluation,aggregation
        evaluation.run(out,phase=1)
        aggregation.finalize(out,phase=1)
        gate=read_json(out/'phase1_gate.json')
        if gate['status']=='PASS':
            from . import trajectory
            trajectory.dispatch(out)
            evaluation.run(out,phase=2)
            aggregation.finalize(out,phase=2)
        else:
            for row in read_json(out/'requests.json'):
                if row['label']=='TRAJECTORY':
                    freeze_json(result_path(out,row),dict(row,status='NOT_RUN_NO_PHASE1_SIGNAL',reason='Phase 1 gate '+gate['status'],model_calls=0))
            freeze_json(out/'trajectory_summary.json',dict(status='NOT_RUN_NO_PHASE1_SIGNAL',phase1_gate_status=gate['status'],outputs=0,model_calls=0))
        from . import gallery,reporting,delivery
        gallery.run(out);reporting.run(out)
        final=delivery.run(out)
        atomic_json(out/'final_completion.json',final)
        atomic_json(out/'controller_status.json',dict(status='DONE',result=final))
        return final
    except BaseException as exc:
        detail=dict(status='BLOCKED',error=repr(exc),traceback=traceback.format_exc(),entrypoint=__package__+'.execute')
        atomic_json(out/'controller_status.json',detail)
        raise
    finally:
        # Child stages can append calls; never overwrite their counts with an old in-memory snapshot.
        try:Budget(out).snapshot()
        except RuntimeError:budget.snapshot()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(OUT))
    p.add_argument('--through',choices=['prepare','endpoint','all'],default='all')
    a=p.parse_args();print(run(a.out,a.through),flush=True)
