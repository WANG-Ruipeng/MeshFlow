"""CPU checks for real frozen stream identity and no-replay recovery boundaries."""
import ast
from pathlib import Path
import torch
from .common import atomic_json, REPO
from .stream import BudgetStream
from .training import check_recovery

def run(source,out):
    report=dict(status='PASS',model_forwards=0,optimizer_updates=0,actual_OT_calls=0,tests=[])
    for name in ('training','stream','checkpoint'):
        ast.parse((Path(__file__).parent/(name+'.py')).read_text())
    for arm in ('T1','T1_fit2_probe'):
        stream=BudgetStream(source,out,arm);state=stream.state_dict();stream.load_state_dict(state)
        assert stream.state_dict()==state
        report['tests'].append(dict(arm=arm,state=state,record_count=len(stream.records),restore='PASS'))
    check_recovery([],1000,1000)
    check_recovery([dict(event='UPDATE_BEGIN',logical_step=1001),dict(event='UPDATE_VALIDATED',logical_step=1001)],1001,1000)
    for events,step in [([dict(event='UPDATE_BEGIN',logical_step=1001)],1000),
                       ([dict(event='UPDATE_VALIDATED',logical_step=1002)],1002),
                       ([dict(event='UPDATE_BEGIN',logical_step=1001)]*2,1001)]:
        try:check_recovery(events,step,1000)
        except RuntimeError:pass
        else:raise AssertionError('Invalid replay accepted')
    report['recovery_tests']=['fresh','reliable completed checkpoint','reject uncovered attempt','reject gap','reject duplicate']
    assert not torch.cuda.is_initialized();report['CUDA_initialized']=False
    atomic_json(Path(out)/'registration/training_cpu_tests.json',report)
    return report

if __name__=='__main__':
    print(run(REPO/'native_t1/runs/control_mvp48_v1',REPO/'native_t1/runs/control_mvp48_budget_single_v1'))
